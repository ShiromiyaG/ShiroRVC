"""ChouwaGAN's decoder: a compact anti-aliased NSF generator for 44.1 kHz.

Ported back from ``ShiroRVC``.  What it is, against the two decoders it sits
beside: RefineGAN drives a *deterministic pulse template* through dense
ResBlocks, Wavehax works in the complex STFT domain, and this one drives a
*band-limited harmonic bank plus noise* through depthwise-separable blocks whose
activations are 2x-oversampled SnakeBeta.  The excitation is the part that
distinguishes it -- every harmonic is masked as it approaches Nyquist, so the
source never aliases and the decoder is not spending capacity cancelling tones
it was handed.

Two things the original had are **not** here, both because what fed them was
deleted with the SVAE:

* the hierarchical latent path (separate ``content_latent``/``detail_latent``
  inputs with their own gates).  ``RefineVitsLatent`` hands the decoder one
  concatenated ``z``, exactly as it does for the other two decoders, so a second
  entry point would be dead configuration.
* late detail fusion, which consumed the SVAE's slow/fast latents.

``template_amplitude`` is accepted and ignored -- see :meth:`ChouwaGANGenerator.forward`.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint


def _safe_pad(x: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return x
    mode = "reflect" if x.shape[-1] > padding else "replicate"
    return F.pad(x, (padding, padding), mode=mode)


def _lowpass_kernel(factor: int, width: int, rolloff: float) -> Tensor:
    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=14.0, dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


class FixedLowPass1d(nn.Module):
    def __init__(
        self,
        factor: int,
        width: int = 4,
        rolloff: float = 0.95,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = int(stride)
        self.register_buffer(
            "kernel", _lowpass_kernel(factor, width, rolloff), persistent=False
        )

    def _grouped_kernel(self, x: Tensor) -> Tensor:
        """The per-channel kernel, cached across calls.

        ``.to()`` plus ``.expand()`` ran on every forward of every instance.
        Each is cheap on its own and neither shows up in the GPU trace, but at
        26 instances called several times per step they are pure dispatch on a
        step that is already CPU-bound.  The cache is keyed by the only three
        things that can change it.
        """
        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_kernel_key", None) != key:
            self._kernel_cache = (
                self.kernel.to(device=x.device, dtype=x.dtype)
                .expand(channels, -1, -1)
                .contiguous()
            )
            self._kernel_key = key
        return self._kernel_cache

    def forward(self, x: Tensor) -> Tensor:
        kernel = self._grouped_kernel(x)
        padding = (kernel.shape[-1] - 1) // 2
        return F.conv1d(
            _safe_pad(x, padding),
            kernel,
            stride=self.stride,
            groups=x.shape[1],
        )


class AntiAliasedUpsample1d(nn.Module):
    def __init__(self, factor: int, filter_width: int, rolloff: float):
        super().__init__()
        self.factor = int(factor)
        kernel = _lowpass_kernel(self.factor, filter_width, rolloff)
        self.register_buffer("kernel", kernel, persistent=False)

        kernel_size = int(kernel.shape[-1])
        self.pad = kernel_size // self.factor - 1
        self.pad_left = self.pad * self.factor + (kernel_size - self.factor) // 2
        self.pad_right = self.pad * self.factor + (kernel_size - self.factor + 1) // 2

    def _grouped_kernel(self, x: Tensor) -> Tensor:
        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_kernel_key", None) != key:
            self._kernel_cache = (
                self.kernel.to(device=x.device, dtype=x.dtype)
                .expand(channels, -1, -1)
                .contiguous()
            )
            self._kernel_key = key
        return self._kernel_cache

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        channels = x.shape[1]
        kernel = self._grouped_kernel(x)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        # ``padding`` crops the transposed convolution's own output instead of
        # computing 43 samples per call and then throwing them away.  It is
        # symmetric, and ``pad_left``/``pad_right`` differ by one whenever
        # ``kernel_size - factor`` is odd -- which it always is here, since the
        # kernel length is odd and the factor even.  So the symmetric part goes
        # to the convolution and only the remaining single sample is trimmed;
        # using ``output_padding`` to absorb it instead would shift the whole
        # signal by one sample at 2x rate, which is a phase change in the
        # anti-aliasing path, not an optimisation.
        x = self.factor * F.conv_transpose1d(
            x,
            kernel,
            stride=self.factor,
            padding=self.pad_left,
            groups=channels,
        )
        trim = self.pad_right - self.pad_left
        return x if trim == 0 else x[..., :-trim]


def soft_clip(
    x: Tensor,
    threshold: float = 0.85,
    ceiling: float = 1.0,
) -> Tensor:
    """Soft limiter: identity below ``threshold``, smooth squeeze above it.

    ``tanh`` forces the generator to drive large pre-activations to reach loud
    output, which parks the operating point in the saturated tail where the
    activation gradient vanishes.  Staying linear below the threshold means a
    given loudness is reached with small pre-activations and healthy gradients,
    and the ``tanh``-style compression above it still bounds the waveform
    without introducing a discontinuity.
    """

    span = float(ceiling) - float(threshold)
    magnitude = x.abs()
    excess = (magnitude - float(threshold)).clamp_min(0.0)
    compressed = span * torch.tanh(excess / span)
    return x.sign() * (magnitude.clamp(max=float(threshold)) + compressed)


class DCBlocker(nn.Module):
    """Remove the per-example DC component without changing healthy peaks.

    A soft-clipped head bounds the waveform but does not prevent a positive or
    negative operating-point drift, and mean removal makes a constant output
    impossible.  Centering can push an already extreme sample just outside
    ``[-1, 1]``, so only that exceptional case is scaled back; normal waveforms
    keep their original loudness.
    """

    def forward(self, waveform: Tensor) -> Tensor:
        centered = waveform - waveform.mean(dim=-1, keepdim=True)
        peak = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return centered / peak


class NoiseInjection(nn.Module):
    """Learned per-channel Gaussian noise injection (RefineGAN's "AdaIN").

    Zero-initialised so the deterministic path establishes itself first.  This
    is largely redundant with ChouwaGAN's stochastic NSF source, which already
    injects noise at every stage, so it is opt-in: each instance costs one
    full-resolution tensor that has to survive until backward.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        return x + torch.randn_like(x) * self.weight.view(1, -1, 1)


class AntiAliasedSnakeBeta(nn.Module):
    """SnakeBeta, optionally surrounded by a 2x low-pass resampling pair.

    The pair is the single most expensive thing in this decoder.  Measured on
    CPU at batch 8 / 0.4 s, removing it from all 21 activations takes the
    decoder from 782 ms to 469 ms -- **40%** -- which is why a 3.9 M ChouwaGAN
    runs slower than a 15.7 M NSF-HiFiGAN that has no such pair at all.

    ``filter_width`` used to be hardcoded to 4 here while
    ``chouwagan_filter_width`` was threaded everywhere else, so the config key
    reached the excitation encoder and the upsamplers but *not* the 21
    activations that carry the cost.  Setting it to 2 measured as a 2% change,
    i.e. noise, because it was not connected to anything that mattered.  It is
    connected now: 4 gives a 17-tap kernel, 2 gives 9.

    ``antialias=False`` drops the pair entirely.  It changes no parameter --
    ``alpha`` and ``beta`` are the only weights -- so the choice is free to make
    per stage without touching the state dict.
    """

    def __init__(
        self,
        channels: int,
        rolloff: float = 0.95,
        filter_width: int = 4,
        antialias: bool = True,
    ):
        super().__init__()
        initial = math.log(math.expm1(1.0))
        self.alpha = nn.Parameter(torch.full((channels,), initial))
        self.beta = nn.Parameter(torch.full((channels,), initial))
        self.antialias = bool(antialias)
        if self.antialias:
            self.upsample = AntiAliasedUpsample1d(
                2, filter_width=int(filter_width), rolloff=rolloff
            )
            self.downsample = FixedLowPass1d(
                2, width=int(filter_width), rolloff=rolloff, stride=2
            )
        else:
            self.upsample = self.downsample = None

    def _snake(self, x: Tensor) -> Tensor:
        alpha = F.softplus(self.alpha).view(1, -1, 1) + 1e-4
        beta = F.softplus(self.beta).view(1, -1, 1) + 1e-4
        return x + torch.sin(alpha * x).square() / beta

    def forward(self, x: Tensor) -> Tensor:
        if not self.antialias:
            return self._snake(x)
        original_length = x.shape[-1]
        x = self.upsample(x)
        x = self._snake(x)
        x = self.downsample(x)
        if x.shape[-1] > original_length:
            x = x[..., :original_length]
        elif x.shape[-1] < original_length:
            x = F.pad(x, (0, original_length - x.shape[-1]), mode="replicate")
        return x


# Seed for the reproducible excitation noise used when ``deterministic`` is set.
DETERMINISTIC_NOISE_SEED = 0

# Same idea for the harmonic phase offsets.  Training draws a fresh offset per
# harmonic on every batch, so the decoder only ever sees excitations from that
# random-phase distribution and must be phase-agnostic.  ``deterministic``
# inference used to leave the offsets at zero instead, which is not a draw from
# that distribution -- it is its extreme: aligning every harmonic maximises the
# crest factor.  Measured at f0=200 Hz against 200 random draws, the aligned
# excitation is 1.31x peakier than a training one at 16 harmonics and 1.57x at
# 32, so the gap widens with every harmonic added.  Drawing a fixed-seed offset
# keeps inference reproducible while keeping it in distribution, exactly as
# DETERMINISTIC_NOISE_SEED does for the noise term.
DETERMINISTIC_PHASE_SEED = 1

# Half-width of the voiced/unvoiced amplitude crossfade.  Measured on a 180 Hz
# burst, this drops the waveform discontinuity at the boundary from 0.140 to
# 0.006; anything longer buys nothing and only smears the transition.
VOICED_CROSSFADE_SECONDS = 0.002


class BandLimitedNSFSource(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_count: int = 7,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.harmonic_count = int(harmonic_count)
        self.noise_std = float(noise_std)
        self.deterministic = False
        self.crossfade_samples = max(
            1, int(round(self.sample_rate * VOICED_CROSSFADE_SECONDS))
        )

    def _smooth_mask(self, mask: Tensor) -> Tensor:
        """Crossfade a 0/1 voicing mask over ``crossfade_samples`` either side."""
        radius = self.crossfade_samples
        if radius < 1:
            return mask
        window = torch.hann_window(
            2 * radius + 1, periodic=False, device=mask.device, dtype=mask.dtype
        )
        window = window / window.sum()
        return F.conv1d(
            mask.unsqueeze(1), window.view(1, 1, -1), padding=radius
        ).squeeze(1)

    # Inductor miscompiles this body (the cumsum-driven harmonic bank fed by the
    # two interpolations) and takes the whole decoder down with it, so
    # `enable_decoder_compile` silently fell back to eager for every step.
    # Both methods run under `no_grad` and are a pure function of f0, so keeping
    # them out of the graph costs no fusion and restores compilation of the
    # convolutional trunk, which is where the time actually goes.
    @torch.compiler.disable
    def prepare(self, f0: Tensor, length: int, periodicity: Optional[Tensor] = None):
        if f0 is None:
            raise ValueError("ChouwaGAN requires an F0 sequence.")
        if f0.ndim == 3:
            f0 = f0.squeeze(-1)
        if f0.ndim != 2:
            raise ValueError("F0 must have shape [batch, frames].")

        f0 = f0.float().clamp_min(0.0)
        voiced_frames = (f0 > 0).float()
        f0 = F.interpolate(
            f0.unsqueeze(1), size=length, mode="linear", align_corners=False
        ).squeeze(1)
        voiced = F.interpolate(
            voiced_frames.unsqueeze(1), size=length, mode="nearest"
        ).squeeze(1)
        # Pitch gating stays hard: a fractional mask here would scale f0 itself
        # and bend the phase around every boundary.
        f0 = f0 * voiced
        # The amplitude mask, by contrast, must not be a step.  It multiplies a
        # harmonic wave whose instantaneous value at the boundary is arbitrary,
        # so a 0/1 transition splices a discontinuity into the excitation - a
        # broadband click on every voiced/unvoiced edge.  A couple of
        # milliseconds of crossfade removes it without smearing the boundary
        # (one frame is 10 ms at 44.1 kHz).
        voiced_envelope = self._smooth_mask(voiced)

        # Measured harmonicity, if the caller supplied it.  Without it the mix
        # falls back to the binary voicing flag, which is what this model did
        # before -- the fallback is the old behaviour, never a zeroed signal.
        if periodicity is None:
            harmonicity = voiced_envelope
        else:
            harmonicity = periodicity.float()
            if harmonicity.ndim == 3:
                harmonicity = harmonicity.squeeze(1)
            harmonicity = F.interpolate(
                harmonicity.unsqueeze(1),
                size=length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            # Still gated by voicing: a frame with no f0 has no comb to sit on,
            # whatever the measurement says.
            harmonicity = harmonicity.clamp(0.0, 1.0) * voiced_envelope

        # Accumulated in cycles, in float64, and wrapped to ``[0, 1)`` before
        # anything multiplies it by a harmonic index.
        #
        # This used to be a float32 ``cumsum`` of radians, unbounded, feeding
        # ``sin(phase * h)`` with ``h`` up to ``harmonic_count``.  The phase of
        # the top harmonic therefore reached 4.8e6 rad over a 30 s span, where
        # a float32 ULP is 0.5 rad, and the error *grows along the span*:
        # measured against float64 at 200 Hz, the h=128 term is off by -62 dB
        # over 0.4 s, -44 dB over 3 s and **-24 dB over 30 s**.  Training reads
        # ``segment_size`` samples -- 0.4 s, where this is inaudible -- while
        # validation previews and the inference pipeline render a whole span,
        # so the decoder is never taught to cancel a defect it only ever meets
        # at generation time.  Broadband phase noise on the upper harmonics is
        # what "metallic" sounds like.
        #
        # Wrapping is exact rather than an approximation: ``frac(c)`` differs
        # from ``c`` by an integer, ``h`` is an integer, so their product
        # differs by an integer number of cycles and ``sin`` does not see it.
        # The float64 accumulation is what keeps the wrap itself honest -- the
        # running total is unbounded before ``frac`` takes it.
        cycles = torch.cumsum(f0.double() / self.sample_rate, dim=-1)
        phase = ((2.0 * math.pi) * (cycles - cycles.floor())).to(f0.dtype)
        harmonics = torch.arange(
            1,
            self.harmonic_count + 1,
            device=f0.device,
            dtype=f0.dtype,
        )
        phase_offset = torch.zeros(
            f0.shape[0],
            1,
            self.harmonic_count,
            device=f0.device,
            dtype=f0.dtype,
        )
        if self.harmonic_count > 1:
            if self.training or not self.deterministic:
                phase_offset[..., 1:] = torch.rand_like(phase_offset[..., 1:]) * (
                    2.0 * math.pi
                )
            else:
                # See DETERMINISTIC_PHASE_SEED: leaving these at zero hands the
                # decoder the one excitation shape it never trained on.
                generator = torch.Generator(device=f0.device)
                generator.manual_seed(DETERMINISTIC_PHASE_SEED)
                phase_offset[..., 1:] = (
                    torch.rand(
                        phase_offset[..., 1:].shape,
                        generator=generator,
                        device=f0.device,
                        dtype=f0.dtype,
                    )
                    * (2.0 * math.pi)
                )
        harmonic_wave = torch.sin(
            phase.unsqueeze(-1) * harmonics.view(1, 1, -1) + phase_offset
        )
        if self.training or not self.deterministic:
            noise = torch.randn_like(f0)
        else:
            # `deterministic` asks for a reproducible excitation, not a silent
            # one.  Unvoiced frames carry no harmonic component, so the noise
            # term is the *entire* excitation there; zeroing it hands the
            # decoder an input it never saw in training and those regions come
            # out as digital silence -- no room tone, no breath, a hard cut at
            # the end of every utterance.  A fixed generator keeps the output
            # reproducible while staying in distribution.
            generator = torch.Generator(device=f0.device)
            generator.manual_seed(DETERMINISTIC_NOISE_SEED)
            noise = torch.randn(
                f0.shape, generator=generator, device=f0.device, dtype=f0.dtype
            )
        return f0, voiced, voiced_envelope, harmonic_wave, noise, harmonicity

    @torch.compiler.disable
    def render(
        self,
        state,
        length: int,
        sample_rate: float,
    ) -> Tensor:
        f0, voiced, voiced_envelope, harmonic_wave, noise, harmonicity = state
        full_length = int(f0.shape[-1])
        if full_length % int(length):
            raise ValueError(
                f"Excitation length {full_length} is not a multiple of {length}."
            )

        factor = full_length // int(length)
        # The excitation U-Net renders once at full rate, where the decimation
        # below is the identity.  Skipping it there avoids four full-resolution
        # copies of the harmonic bank per forward.
        if factor != 1:
            offset = (factor - 1) // 2
            indices = (
                torch.arange(int(length), device=f0.device, dtype=torch.long) * factor
                + offset
            )
            f0 = f0.index_select(-1, indices)
            voiced = voiced.index_select(-1, indices)
            voiced_envelope = voiced_envelope.index_select(-1, indices)
            harmonic_wave = harmonic_wave.index_select(1, indices)
            noise = noise.index_select(-1, indices)
            harmonicity = harmonicity.index_select(-1, indices)

        harmonics = torch.arange(
            1,
            self.harmonic_count + 1,
            device=f0.device,
            dtype=f0.dtype,
        )
        harmonic_frequency = f0.unsqueeze(-1) * harmonics.view(1, 1, -1)
        # The centre sits at 0.45 rather than 0.48 of the sample rate.  This is
        # a soft mask on purpose -- a hard cutoff makes a harmonic blink on and
        # off as vibrato walks it across the threshold -- but soft means it
        # leaks, and at 0.48 the leak was 27% *at Nyquist itself*.  Anything
        # past Nyquist aliases back into the top of the band, so with 48
        # harmonics an f0 above 459 Hz (routine in singing) put inharmonic tones
        # at -29 to -31 dB into 20-21.7 kHz.  Inaudible, but the MS-STFT term
        # and the discriminator both see it and the decoder has to learn to
        # cancel it.  At 0.45 the leak at Nyquist is 7.6%, the transition stays
        # 882 Hz wide, and all that is given up is 19.8-22 kHz.
        transition = max(1.0, float(sample_rate) * 0.02)
        nyquist_mask = torch.sigmoid(
            (float(sample_rate) * 0.45 - harmonic_frequency) / transition
        )
        amplitude = harmonics.rsqrt().view(1, 1, -1)
        harmonic_wave = (harmonic_wave * amplitude * nyquist_mask).sum(dim=-1)
        normalizer = (amplitude.square() * nyquist_mask.square()).sum(dim=-1).sqrt()
        harmonic_wave = harmonic_wave / normalizer.clamp_min(1e-4)

        # Both terms ride the crossfaded envelope so neither steps at the edge.
        #
        # ``harmonicity`` is the measured harmonic share where one was supplied
        # and the voicing envelope otherwise, so with the feature off this is
        # bit-identical to the two-constant mix it replaces.  With it on, a
        # breathy frame moves continuously toward the noise term instead of
        # having to be either fully voiced or fully unvoiced.
        noise_level = self.noise_std * (1.0 - 0.75 * harmonicity)
        source = harmonic_wave * harmonicity + noise * noise_level
        return source.unsqueeze(1)

    def forward(
        self,
        f0: Tensor,
        length: int,
        sample_rate: float,
        periodicity: Optional[Tensor] = None,
    ) -> Tensor:
        state = self.prepare(f0, length, periodicity)
        return self.render(state, length, sample_rate)


class DepthwiseSeparableUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int,
        rolloff: float,
        filter_width: int = 4,
        antialias: bool = True,
    ):
        super().__init__()
        hidden = int(channels * expansion)
        self.activation = AntiAliasedSnakeBeta(
            channels, rolloff, filter_width, antialias
        )
        self.expand = weight_norm(nn.Conv1d(channels, hidden, 1))
        self.depthwise = weight_norm(
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size,
                padding=((kernel_size - 1) * dilation) // 2,
                dilation=dilation,
                groups=hidden,
            )
        )
        self.project = weight_norm(nn.Conv1d(hidden, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.activation(x)
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return residual + x


class DenseUnit(nn.Module):
    """One dense convolution where :class:`DepthwiseSeparableUnit` runs three.

    Same residual shape, same activation, same receptive field -- the inverted
    bottleneck (1x1 expand, depthwise k, 1x1 project) collapses into a single
    ``Conv1d(channels, channels, k)``.

    This trades FLOPs for dispatches, which is the right trade here and not a
    general one.  Measured on this box (RTX 5060, eager, batch 8, 300 frames),
    the whole training step spends 100% of its wall clock enqueueing work:
    10 439 aten dispatches per generator forward against 2 579 CUDA kernels, so
    75% of the dispatches launch nothing at all and the GPU idles for over half
    the step.  The ``conv1d -> convolution -> _convolution -> cudnn_convolution``
    chain alone costs ~63 us of CPU per call, and the separable unit pays it
    three times to do less arithmetic than one dense conv does paying it once.

    ``expansion`` is accepted and ignored: it sizes the bottleneck this unit
    does not have.  It stays in the signature so the two units are
    interchangeable at every call site, and so a config carrying
    ``chouwagan_expansion`` still builds either one.

    The parameter count moves -- ``C^2 k`` against ``2 C^2 e + C e k`` -- so
    this is a different model, not a faster spelling of the same one.  At the
    shipped schedule that is 5.15 M against 3.93 M.

    **Measured, it does not pay, which is why the shipped config does not use
    it.**  On the decoder alone (RTX 5060, eager, batch 8, 40-frame segment) it
    drops the dispatch count 27% -- 2 631 aten ops to 1 931 -- and converts
    almost none of that into wall clock: against the separable unit with the
    same anti-alias schedule, 31.45 ms against 28.79 ms forward (9% *slower*)
    and 75.68 ms against 77.68 ms forward-and-backward (3% faster).  The reason
    is that the decoder is not the dispatch-bound half of this model: at a
    40-frame segment its kernels already carry real work, and once the stage-3
    anti-alias pair is gone it is GPU-bound, where the extra FLOPs stop being
    free.  The frontend is the dispatch-bound half, and it runs 300 frames of
    much smaller kernels.
    It is kept because the trade is real where dispatch dominates, and because
    the measurement above is worth being able to repeat.

    Switching to it moves every block's checkpoint keys, and the architecture id
    is what keeps a checkpoint of one style from loading non-strictly into the
    other.  ``train.py`` pins that id from ``vocoders.json`` rather than from
    the model config, so enabling this **requires bumping it there** as well.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int,
        rolloff: float,
        filter_width: int = 4,
        antialias: bool = True,
    ):
        super().__init__()
        del expansion
        self.activation = AntiAliasedSnakeBeta(
            channels, rolloff, filter_width, antialias
        )
        self.conv = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=((kernel_size - 1) * dilation) // 2,
                dilation=dilation,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(self.activation(x))


#: The residual units a stage can be built from.  ``separable`` is the original
#: inverted bottleneck; ``dense`` is the one-convolution form above.
UNIT_STYLES = {
    "separable": DepthwiseSeparableUnit,
    "dense": DenseUnit,
}


def _unit_class(style: str):
    try:
        return UNIT_STYLES[str(style)]
    except KeyError:
        raise ValueError(
            "Unknown ChouwaGAN unit style %r; expected one of %s."
            % (style, ", ".join(sorted(UNIT_STYLES)))
        ) from None


class LiteAMPBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int,
        dilations: Sequence[int],
        expansion: int,
        rolloff: float,
        filter_width: int = 4,
        antialias: bool = True,
        unit_style: str = "separable",
    ):
        super().__init__()
        unit = _unit_class(unit_style)
        self.units = nn.ModuleList(
            [
                unit(
                    channels,
                    int(kernel),
                    int(dilation),
                    expansion,
                    rolloff,
                    filter_width,
                    antialias,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.units:
            x = unit(x)
        return x


class LiteAMPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernels: Iterable[int],
        dilations: Sequence[int],
        expansion: int,
        rolloff: float,
        filter_width: int = 4,
        antialias: bool = True,
        unit_style: str = "separable",
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                LiteAMPBranch(
                    channels,
                    int(kernel),
                    dilations,
                    expansion,
                    rolloff,
                    filter_width,
                    antialias,
                    unit_style,
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.branches[0](x)
        for branch in self.branches[1:]:
            output = output + branch(x)
        return output / len(self.branches)


class ExcitationUnit(nn.Module):
    """Depthwise-separable residual unit for the excitation encoder.

    RefineGAN runs a dense ResBlock at every excitation resolution, full
    waveform rate included.  The excitation path only has to preserve pitch
    phase, not synthesise texture, so a depthwise-separable residual with a
    plain LeakyReLU buys the same receptive field far more cheaply.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        slope: float = 0.2,
    ):
        super().__init__()
        self.slope = float(slope)
        self.depthwise = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                int(kernel_size),
                padding=((int(kernel_size) - 1) * int(dilation)) // 2,
                dilation=int(dilation),
                groups=channels,
            )
        )
        self.project = weight_norm(nn.Conv1d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.leaky_relu(x, self.slope)
        x = self.depthwise(x)
        x = self.project(x)
        return residual + x


class ExcitationEncoder(nn.Module):
    """RefineGAN-style multi-resolution excitation encoder.

    The NSF source is rendered once at full waveform rate and decimated with
    the same fixed windowed-sinc filters the decoder uses for upsampling, so
    every skip is a properly band-limited view of one single excitation rather
    than an independently re-rendered harmonic bank.  That is the part of
    RefineGAN that buys F0 fidelity: the decoder sees the excitation at the
    resolution of each upsampling stage instead of a 1-channel signal folded in
    additively.

    Skips are returned highest-rate first; the decoder consumes them reversed.
    """

    def __init__(
        self,
        upsample_rates: Sequence[int],
        skip_channels: Sequence[int],
        bottleneck_channels: int,
        kernel_size: int = 7,
        filter_width: int = 4,
        rolloff: float = 0.95,
        slope: float = 0.2,
    ):
        super().__init__()
        self.slope = float(slope)
        skip_channels = tuple(int(value) for value in skip_channels)
        self.skip_channels = skip_channels
        self.bottleneck_channels = int(bottleneck_channels)
        self.pre = weight_norm(
            nn.Conv1d(
                1, skip_channels[0], int(kernel_size), padding=int(kernel_size) // 2
            )
        )
        self.downs = nn.ModuleList()
        self.units = nn.ModuleList()
        current = skip_channels[0]
        targets = list(skip_channels[1:]) + [self.bottleneck_channels]
        for factor, target in zip(reversed(tuple(upsample_rates)), targets, strict=True):
            self.downs.append(
                nn.Sequential(
                    FixedLowPass1d(
                        int(factor),
                        width=int(filter_width),
                        rolloff=rolloff,
                        stride=int(factor),
                    ),
                    weight_norm(nn.Conv1d(current, int(target), 1)),
                )
            )
            self.units.append(ExcitationUnit(int(target), kernel_size, 1, slope))
            current = int(target)

    def forward(self, source: Tensor, checkpointing: bool = False):
        x = F.leaky_relu(self.pre(source), self.slope)
        skips = []
        for down, unit in zip(self.downs, self.units, strict=True):
            skips.append(x)
            x = down(x)
            if checkpointing and self.training:
                x = checkpoint(unit, x, use_reentrant=False)
            else:
                x = unit(x)
        return skips, x


class ChouwaGANGenerator(nn.Module):
    """Compact full-band anti-aliased NSF generator for 44.1 kHz audio."""

    def __init__(
        self,
        initial_channel: int,
        gin_channels: int,
        sr: int,
        upsample_rates: Sequence[int],
        upsample_initial_channel: int = 320,
        checkpointing: bool = False,
        chouwagan_channels: Sequence[int] = (256, 160, 80, 40),
        chouwagan_block_kernels: Sequence[Sequence[int]] = (
            (3, 7),
            (3, 7),
            (7,),
            (7,),
        ),
        chouwagan_dilations: Sequence[int] = (1, 3, 5),
        chouwagan_expansion: int | Sequence[int] = (2, 2, 1, 1),
        chouwagan_harmonics: int = 7,
        chouwagan_noise_std: float = 0.01,
        chouwagan_filter_width: int = 4,
        chouwagan_antialias_stages: Optional[Sequence[int]] = None,
        chouwagan_rolloff: float = 0.95,
        chouwagan_latent_mixer_kernel: int = 7,
        chouwagan_latent_mixer_dilations: Sequence[int] = (1, 3),
        chouwagan_latent_mixer_expansion: int = 2,
        chouwagan_excitation_unet: bool = False,
        chouwagan_excitation_kernel: int = 7,
        chouwagan_noise_injection: bool = False,
        chouwagan_output_head_threshold: float = 0.85,
        chouwagan_output_head_ceiling: float = 1.0,
        chouwagan_remove_output_dc: bool = True,
        chouwagan_unit_style: str = "separable",
        **_: object,
    ):
        super().__init__()
        if int(sr) != 44100:
            raise ValueError("ChouwaGAN only supports 44.1 kHz configurations.")

        self.sr = int(sr)
        self.checkpointing = bool(checkpointing)
        self.excitation_unet = bool(chouwagan_excitation_unet)
        self.initial_channel = int(initial_channel)
        self.upsample_rates = tuple(int(value) for value in upsample_rates)
        self.total_upsample = math.prod(self.upsample_rates)
        self.channels = tuple(int(value) for value in chouwagan_channels)
        if len(self.channels) != len(self.upsample_rates):
            raise ValueError("ChouwaGAN channel and upsample schedules must match.")
        if isinstance(chouwagan_expansion, int):
            stage_expansions = (int(chouwagan_expansion),) * len(self.channels)
        else:
            stage_expansions = tuple(int(value) for value in chouwagan_expansion)
        if len(stage_expansions) != len(self.channels):
            raise ValueError("ChouwaGAN expansion and upsample schedules must match.")
        if any(value < 1 for value in stage_expansions):
            raise ValueError("ChouwaGAN expansion values must be positive.")

        self.source = BandLimitedNSFSource(
            sample_rate=self.sr,
            harmonic_count=chouwagan_harmonics,
            noise_std=chouwagan_noise_std,
        )
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, int(upsample_initial_channel), 5, padding=2)
        )
        self.cond = (
            weight_norm(nn.Conv1d(gin_channels, int(upsample_initial_channel), 1))
            if gin_channels
            else None
        )
        # Validated once, here, so an unknown style fails at construction rather
        # than at the first stage that happens to build a unit.
        self.unit_style = str(chouwagan_unit_style)
        unit = _unit_class(self.unit_style)
        self.latent_mixer = nn.Sequential(
            *(
                unit(
                    int(upsample_initial_channel),
                    int(chouwagan_latent_mixer_kernel),
                    int(dilation),
                    int(chouwagan_latent_mixer_expansion),
                    chouwagan_rolloff,
                )
                for dilation in chouwagan_latent_mixer_dilations
            )
        )

        # ---- Excitation path ------------------------------------------------
        # Two mutually exclusive designs.  The legacy path re-renders the NSF
        # source once per stage and folds it in additively through a 1x1 gate.
        # The RefineGAN-style path renders it once at full rate and runs a
        # downsampling encoder whose skips are concatenated at every stage, so
        # the decoder sees a band-limited excitation *feature map* rather than a
        # single channel.
        self.fusion_proj = nn.ModuleList()
        self.source_projections = nn.ModuleList()
        self.exc_encoder = None
        self.exc_bottleneck = None
        if self.excitation_unet:
            skip_channels = tuple(
                max(4, int(value) // 2) for value in reversed(self.channels)
            )
            self.exc_skip_channels = skip_channels
            self.exc_encoder = ExcitationEncoder(
                self.upsample_rates,
                skip_channels,
                skip_channels[-1],
                kernel_size=int(chouwagan_excitation_kernel),
                filter_width=chouwagan_filter_width,
                rolloff=chouwagan_rolloff,
            )
            self.exc_bottleneck = weight_norm(
                nn.Conv1d(skip_channels[-1], int(upsample_initial_channel), 1)
            )
            self.exc_bottleneck_gate = nn.Parameter(torch.tensor(math.log(0.25 / 0.75)))
            self.register_parameter("source_gates", None)
        else:
            self.exc_skip_channels = ()
            source_gate = math.log(0.25 / 0.75)
            self.source_gates = nn.Parameter(
                torch.full((len(self.channels),), source_gate)
            )

        # Which upsampling stages get the anti-aliased activation pair.  ``None``
        # means all of them, which is what BigVGAN does and what this decoder
        # did until 2026-08-28.
        #
        # Cost is *not* spread evenly, because it scales with channels x length
        # and the stages trade one for the other unevenly.  Measured per stage
        # as a share of the whole decoder (CPU, batch 8, 0.4 s):
        #
        #   stage 0  256 ch @   300 Hz internal   5.6%
        #   stage 1  160 ch @   900 Hz internal   7.1%
        #   stage 2   80 ch @  6300 Hz internal  10.4%
        #   stage 3   40 ch @ 44100 Hz internal  27.0%
        #   post_activation                      11.6%
        #
        # The last stage plus the post activation are 38.6% on their own,
        # against 23.1% for the first three together.  And the quality argument
        # runs the *other* way: folding at stage 0 lands inside the audible
        # core, because its internal Nyquist is 150 Hz, while folding at stage 3
        # lands in the top octave under a 22.05 kHz Nyquist.  So the cheap
        # stages are the ones worth protecting, which is why dropping the last
        # is the first thing to try and not the last.
        #
        # Re-measured on GPU (RTX 5060, eager, batch 8, 0.4 s), per activation,
        # wall with the pair against wall without, and how much of that is GPU
        # rather than dispatch:
        #
        #   stage 0  256 ch @   120   0.53 -> 0.26 ms   (GPU 0.10 -> 0.02)
        #   stage 1  160 ch @   360   0.55 -> 0.56 ms   (GPU 0.20 -> 0.02)
        #   stage 2   80 ch @  2520   0.79 -> 0.40 ms   (GPU 0.72 -> 0.08)
        #   stage 3   40 ch @ 17640   3.43 -> 0.68 ms   (GPU 3.33 -> 0.65)
        #
        # Weighted by the activations per stage (6, 6, 3, 3 plus the post one),
        # dropping stage 3 is worth ~11 ms of a 117.8 ms forward while dropping
        # stages 0 and 1 is worth ~1.6 ms.  The two arguments agree, so the
        # shipped config drops the last stage and keeps the cheap ones.
        stage_count = len(self.upsample_rates)
        if chouwagan_antialias_stages is None:
            self.antialias_stages = tuple(range(stage_count))
        else:
            self.antialias_stages = tuple(
                sorted({int(value) % stage_count for value in chouwagan_antialias_stages})
            )

        self.ups = nn.ModuleList()
        self.noise_injection = nn.ModuleList() if chouwagan_noise_injection else None
        self.blocks = nn.ModuleList()
        current_channels = int(upsample_initial_channel)
        for index, (factor, channels, expansion) in enumerate(
            zip(
                self.upsample_rates,
                self.channels,
                stage_expansions,
                strict=True,
            )
        ):
            self.ups.append(
                nn.Sequential(
                    AntiAliasedUpsample1d(
                        factor,
                        filter_width=chouwagan_filter_width,
                        rolloff=chouwagan_rolloff,
                    ),
                    weight_norm(nn.Conv1d(current_channels, channels, 1)),
                )
            )
            if self.excitation_unet:
                # Skips arrive highest-rate first, stages consume them reversed.
                skip = self.exc_skip_channels[len(self.channels) - 1 - index]
                self.fusion_proj.append(
                    weight_norm(nn.Conv1d(channels + skip, channels, 1))
                )
            else:
                self.source_projections.append(
                    weight_norm(nn.Conv1d(1, channels, 1, bias=False))
                )
            if self.noise_injection is not None:
                self.noise_injection.append(NoiseInjection(channels))
            self.blocks.append(
                LiteAMPBlock(
                    channels,
                    chouwagan_block_kernels[index],
                    chouwagan_dilations,
                    expansion,
                    chouwagan_rolloff,
                    chouwagan_filter_width,
                    index in self.antialias_stages,
                    self.unit_style,
                )
            )
            current_channels = channels

        # ``post_activation`` runs at the output rate, exactly like the last
        # stage, so it follows the same decision rather than carrying its own key.
        self.post_activation = AntiAliasedSnakeBeta(
            current_channels,
            chouwagan_rolloff,
            chouwagan_filter_width,
            (stage_count - 1) in self.antialias_stages,
        )
        self.conv_post = weight_norm(nn.Conv1d(current_channels, 1, 7, padding=3))
        self.output_upsample = AntiAliasedUpsample1d(
            2, filter_width=chouwagan_filter_width, rolloff=chouwagan_rolloff
        )
        self.output_downsample = FixedLowPass1d(
            2, width=chouwagan_filter_width, rolloff=chouwagan_rolloff, stride=2
        )
        # The bounded head runs inside the 2x oversampled pair so the limiter's
        # own harmonics are filtered instead of aliasing back into the band.
        self.output_head_threshold = float(chouwagan_output_head_threshold)
        self.output_head_ceiling = float(chouwagan_output_head_ceiling)
        if not 0.0 < self.output_head_threshold < self.output_head_ceiling:
            raise ValueError("ChouwaGAN output head requires 0 < threshold < ceiling.")
        self.dc_blocker = DCBlocker() if chouwagan_remove_output_dc else None

    def _source_for_stage(self, state, stage: int, length: int) -> Tensor:
        prefix = math.prod(self.upsample_rates[: stage + 1])
        remaining = self.total_upsample // prefix
        stage_rate = self.sr / remaining
        return self.source.render(state, length, stage_rate)

    def forward(
        self,
        x: Tensor,
        f0: Tensor,
        g: Optional[Tensor] = None,
        template_amplitude: Optional[Tensor] = None,
        periodicity: Optional[Tensor] = None,
    ) -> Tensor:
        """``template_amplitude`` is accepted for call-site compatibility only.

        The synthesizer hands every VITS-latent decoder the measured frame
        energy, because RefineGAN's pulse template scales its impulses by it.
        ChouwaGAN's excitation has no such scale to set: the harmonic bank is
        divided by its own Nyquist-masked normaliser, so it leaves ``render``
        at unit RMS by construction, and multiplying a measured envelope back in
        would undo exactly that.  The frame energy still reaches the model, one
        layer earlier, as frontend conditioning on the latent.
        """
        del template_amplitude

        x = self.conv_pre(x)
        if self.cond is not None and g is not None:
            x = x + self.cond(g)
        x = self.latent_mixer(x)

        output_length = x.shape[-1] * self.total_upsample
        with torch.no_grad():
            source_state = self.source.prepare(f0, output_length, periodicity)

        excitation_skips = None
        if self.excitation_unet:
            # One render at full rate: the encoder derives every lower-rate
            # view from it, which is both cheaper than the four per-stage
            # renders and phase-consistent across resolutions.
            with torch.no_grad():
                full_source = self.source.render(
                    source_state, output_length, float(self.sr)
                )
            excitation_skips, excitation_bottleneck = self.exc_encoder(
                full_source.to(dtype=x.dtype),
                checkpointing=self.checkpointing,
            )
            x = x + torch.sigmoid(self.exc_bottleneck_gate).to(x.dtype) * (
                self.exc_bottleneck(excitation_bottleneck)
            )
            source_gates = None
        else:
            source_gates = torch.sigmoid(self.source_gates)

        stage_count = len(self.ups)
        for index in range(stage_count):
            upsample = self.ups[index]
            block = self.blocks[index]
            x = upsample(x)
            if self.excitation_unet:
                skip = excitation_skips[stage_count - 1 - index]
                x = self.fusion_proj[index](
                    torch.cat((x, skip.to(dtype=x.dtype)), dim=1)
                )
            else:
                with torch.no_grad():
                    source = self._source_for_stage(source_state, index, x.shape[-1])
                source = self.source_projections[index](source.to(dtype=x.dtype))
                x = x + source_gates[index].to(dtype=x.dtype) * source
            if self.noise_injection is not None:
                x = self.noise_injection[index](x)
            if self.training and self.checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.post_activation(x)
        x = self.conv_post(x)

        x = self.output_upsample(x)
        x = soft_clip(x, self.output_head_threshold, self.output_head_ceiling)
        x = self.output_downsample(x)
        if self.dc_blocker is not None:
            x = self.dc_blocker(x)
        return x

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self
