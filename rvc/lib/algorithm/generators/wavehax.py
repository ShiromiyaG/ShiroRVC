"""Wavehax generator: harmonic-prior complex spectrogram estimation.

Port of the standard (non multi-scale, non complex-valued) generator from
Yoneyama et al., *Wavehax: Aliasing-Free Neural Waveform Synthesis Based on
2D Convolution and Harmonic Prior for Reliable Complex Spectrogram Estimation*
(2024), https://github.com/chomeyama/wavehax -- MIT licensed, Copyright 2024
Reo Yoneyama (Nagoya University).

The idea is orthogonal to every other decoder in this repository.  HiFi-GAN and
RefineGAN both build a waveform by upsampling in the *time* domain, so every
transposed convolution and every nonlinearity is an opportunity to fold energy
back across Nyquist.  Wavehax never leaves the frame rate: it stacks the STFT of
a harmonic prior with the projected conditioning into a ``(channels, bins,
frames)`` image, runs a 2D ConvNeXt over it, emits a two-channel complex
spectrogram and reaches the waveform in one iSTFT.  There is no upsampling
layer at all, so there is nothing to alias, and the pitch response is inherited
from the prior rather than learned.

What this port keeps
--------------------
The network: the five-channel input stack, the ConvNeXt block with its layer
scale and stochastic depth, the log-magnitude/phase output option and the
pseudo-constant-power harmonic (PCPH) prior.  The defaults follow
``wavehax.v2.yaml`` -- ``mult_channels=3``, a ``(13, 7)`` kernel that is wider
across frequency than across time, and the closed-form PCPH prior.

Two upstream options are absent because nothing selects them: the batch-norm
alternative to ``LayerNorm2d`` inside the ConvNeXt block, and the ``sawtooth``
prior, which needs an anti-aliased resample of an 8x oversampled signal.  The
complex-valued and multi-scale generators are separate models and are not
ported at all.

Where it differs from ``RefineGANGenerator``
--------------------------------------------
The speaker embedding ``g`` is added to the projected conditioning, which is the
one addition upstream does not have.  It is not a RefineGAN import: the original
Wavehax is a single-speaker vocoder conditioned on mel, and the RefineGAN paper
has no speaker embedding either.  The ``self.cond`` layer is RVC's, inherited
from the VITS decoder -- ``hifigan_nsf.py`` does ``x = x + self.cond(g)`` at its
bottleneck and ``refinegan.py`` copied it -- so a decoder in this repository
that lacks it is the odd one out, not the faithful one.  It sits at the same
place in the graph: the frame-rate conditioning path, before anything reads it.

RefineGAN's measured intensity envelope is still not used.  That one *is*
RefineGAN's, it belongs to a pitch template this decoder does not have, and the
harmonic prior sets its own amplitude from the F0 contour.  The argument is
accepted so every decoder can be called the same way, and ignored.

Sizing at 44.1 kHz
------------------
``hop_length`` is ``prod(upsample_rates) = 441``, which is what the rest of the
pipeline means by a frame.  ``n_fft`` defaults to twice that: 50% overlap is the
ratio upstream uses (480/240) and is what makes the Hann window envelope
constant across the interior of the signal.

Mixed precision.  The prior and its STFT run in FP32 under ``no_grad``: the
phase accumulator is a cumulative sum over ``frames * hop_length`` samples and
in FP16 its increments stop registering against the accumulated value within a
few hundred samples.  The ConvNeXt stack runs in whatever dtype autocast picks.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Prior waveforms
# ---------------------------------------------------------------------------


def generate_noise(f0: Tensor, hop_length: int, *_, **__) -> Tensor:
    """Gaussian noise of the length the F0 sequence implies."""
    batch, _, frames = f0.size()
    return torch.randn((batch, 1, frames * hop_length), device=f0.device)


def generate_sine(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: float = 0.03,
    random_init_phase: bool = True,
    *_,
    **__,
) -> Tensor:
    """A single sine at F0 plus noise; the ablation baseline, not the default."""
    device = f0.device
    f0 = F.interpolate(f0, f0.size(2) * hop_length)
    voiced = f0 > 0

    radius = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radius = radius + F.pad(
            torch.rand((f0.size(0), 1, 1), device=device, dtype=torch.float64),
            (0, radius.size(-1) - 1),
        )

    phase = 2.0 * math.pi * torch.cumsum(radius, dim=-1)
    sine = torch.sin(phase).to(torch.float32)
    return voiced * sine + noise_amplitude * torch.randn(sine.size(), device=device)


def generate_pcph(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: float = 0.01,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
    max_frequency: Optional[float] = None,
    *_,
    **__,
) -> Tensor:
    """Pseudo-constant-power harmonics, summed explicitly over the comb.

    Every harmonic below Nyquist gets the same amplitude, so the prior has a
    flat spectral envelope and the network is left to shape it.  This is the
    literal form from the paper; it materialises a
    ``(batch, max_harmonics, samples)`` tensor, which at 44.1 kHz and a low F0
    is the most expensive thing in the decoder.  Prefer
    ``pcph_closed_form``, which is the same signal in O(1) memory per sample.
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 == 0.0):
        return noise

    voiced = f0 > 0
    min_f0_value = torch.min(f0[voiced]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[voiced] = sample_rate / 2.0 / f0[voiced]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    harmonic_amplitude = voiced * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmonic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radius = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radius = radius + F.pad(
            torch.rand((batch, 1, 1), device=device, dtype=torch.float64),
            (0, radius.size(-1) - 1),
        )
    radius = torch.cumsum(radius, dim=2)
    harmonics = torch.sin(2.0 * math.pi * radius * indices).to(torch.float32)

    harmonics = harmonic_mask * harmonics
    harmonics = harmonic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise


def generate_pcph_closed_form(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: float = 0.01,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
    max_frequency: Optional[float] = None,
    epsilon: float = 1e-6,
    *_,
    **__,
) -> Tensor:
    """The same PCPH signal via the Dirichlet kernel, in O(1) per sample.

    ``sum_{k=1..N} sin(k * theta)`` has the closed form
    ``(cos(theta/2) - cos((N + 1/2) * theta)) / (2 * sin(theta/2))``, so the
    harmonic sum never has to be materialised.  The expression is singular
    wherever ``sin(theta/2)`` vanishes, and the limit of the sine sum there is
    zero, which is what the guarded division returns.

    Two departures from the upstream implementation, both behaviour-preserving:
    the singular points are handled with ``torch.where`` rather than boolean
    masked assignment, so the function traces cleanly under ``torch.compile``;
    and unvoiced frames get ``N = 1`` instead of ``N = max_frequency / 1e-5``.
    The unvoiced result is multiplied by a zero voicing mask either way, but at
    ``N ~ 2e9`` the product ``(N + 1/2) * theta`` exceeds FP32's precision by
    four orders of magnitude, and computing garbage that is about to be
    discarded is a poor way to spend a transcendental.
    """
    batch, _, frames = f0.size()
    device = f0.device

    f0_upsampled = F.interpolate(
        f0, scale_factor=float(hop_length), mode="linear", align_corners=False
    )
    total_length = f0_upsampled.shape[-1]
    noise = noise_amplitude * torch.randn((batch, 1, total_length), device=device)
    if torch.all(f0 == 0.0):
        return noise

    # phase = 2 * pi * integral(f0 / sr)
    phase_increment = f0_upsampled / sample_rate
    if random_init_phase:
        phase_increment = phase_increment + F.pad(
            torch.rand((batch, 1, 1), device=device), (0, total_length - 1)
        )

    # The closed form is 2*pi-periodic in theta, so folding the accumulator back
    # into one period is exact -- and it is what keeps the FP32 cast below from
    # quantising the phase of a long utterance.
    phase = torch.cumsum(phase_increment.double(), dim=2) * 2.0 * math.pi
    phase = torch.fmod(phase, 2.0 * math.pi).float()

    limit_frequency = (
        float(max_frequency) if max_frequency is not None else sample_rate / 2.0
    )
    voiced = f0_upsampled > 0
    safe_f0 = torch.where(
        voiced, f0_upsampled, torch.full_like(f0_upsampled, limit_frequency)
    )
    n_harmonics = torch.floor(limit_frequency / safe_f0)

    half_phase = phase / 2.0
    numerator = torch.cos(half_phase) - torch.cos((n_harmonics + 0.5) * phase)
    denominator = 2.0 * torch.sin(half_phase)
    regular = denominator.abs() > epsilon
    harmonics = torch.where(
        regular,
        numerator / torch.where(regular, denominator, torch.ones_like(denominator)),
        torch.zeros_like(numerator),
    )

    amplitude = power_factor * torch.sqrt(2.0 / torch.clamp(n_harmonics, min=1.0))
    return harmonics * amplitude * voiced.to(harmonics.dtype) + noise


#: Magnitude past which a predicted spectrogram bin is treated as an overflow
#: rather than as a prediction; see ``WavehaxGenerator.forward_core``.
SPECTROGRAM_BOUND = 1e4


PRIOR_GENERATORS = {
    "noise": generate_noise,
    "sine": generate_sine,
    "pcph": generate_pcph,
    "pcph_closed_form": generate_pcph_closed_form,
}


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


def to_log_magnitude_and_phase(
    real: Tensor, imag: Tensor, clip_value: float = 1e-10
) -> Tuple[Tensor, Tensor]:
    magnitude = torch.sqrt(torch.clamp(real**2 + imag**2, min=clip_value))
    return torch.log(magnitude), torch.atan2(imag, real)


def to_real_imaginary(
    log_magnitude: Tensor, phase: Tensor, clip_value: float = 1e2
) -> Tuple[Tensor, Tensor]:
    """Vocos-style implicit phase wrapping; the ceiling keeps ``exp`` bounded."""
    magnitude = torch.clip(torch.exp(log_magnitude), max=clip_value)
    return magnitude * torch.cos(phase), magnitude * torch.sin(phase)


class STFT(nn.Module):
    """Framed STFT and its overlap-add inverse, both differentiable.

    Upstream frames with an identity ``conv1d`` so the module exports to ONNX
    without an FFT-adjacent op; ``unfold``/``fold`` compute the same thing
    without registering an ``(n_fft, 1, n_fft)`` buffer, which at
    ``n_fft = 882`` would put 3 MB of constant identity matrix in every
    checkpoint.

    **Both directions run in FP32 regardless of autocast**, and the guard is
    inside the methods rather than at the call site so the module is correct
    whoever calls it.  This is not a precision preference, it is a hard
    constraint: cuFFT's half-precision path only accepts signal sizes that are
    powers of two, and this fork's ``n_fft`` is ``2 * hop_length = 882``, which
    is ``2 x 3^2 x 7^2``.  Under autocast the decoder's output projection emits
    FP16, ``torch.complex`` of two FP16 tensors is ComplexHalf, and the inverse
    transform then raises rather than degrading.  There is no ``n_fft`` that
    satisfies both cuFFT and the hop: a power of two is never a multiple of 441.

    The waveform therefore leaves this module in FP32.  That is the dtype the
    losses and the discriminator's own STFT want anyway, and it matches the real
    audio coming off the loader, so nothing downstream has to reconcile a pair.
    """

    def __init__(self, n_fft: int, hop_length: int, window: str = "hann_window"):
        super().__init__()
        self.n_fft = int(n_fft)
        self.n_bins = self.n_fft // 2 + 1
        self.hop_length = int(hop_length)

        window_tensor = getattr(torch, window)(self.n_fft)
        self.register_buffer("window", window_tensor.reshape(1, self.n_fft, 1))
        self.register_buffer(
            "window_envelope", window_tensor.square().reshape(1, self.n_fft, 1)
        )

    def forward(self, x: Tensor, norm: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """``x``: ``(batch, samples)`` or ``(batch, 1, samples)``.  FP32 out."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            pad = self.n_fft - self.hop_length
            pad_left = pad // 2
            x = F.pad(x, (pad_left, pad - pad_left))

            x = x.unsqueeze(1) if x.dim() == 2 else x
            # (batch, 1, frames, n_fft) -> (batch, n_fft, frames)
            x = x.unfold(-1, self.n_fft, self.hop_length).squeeze(1).transpose(1, 2)

            spectrum = torch.fft.rfft(x * self.window, dim=1, norm=norm)
            return spectrum.real, spectrum.imag

    def inverse(self, real: Tensor, imag: Tensor, norm: Optional[str] = None) -> Tensor:
        """``real``/``imag``: ``(batch, n_bins, frames)``.  Returns ``(batch, 1, samples)``."""
        assert real.shape == imag.shape and real.ndim == 3
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length
        padded = (frames - 1) * self.hop_length + self.n_fft

        with torch.autocast(device_type=real.device.type, enabled=False):
            # ``.float()`` before ``torch.complex``: a ComplexHalf input is what
            # cuFFT rejects at a signal size of 882.  See the class docstring.
            spectrum = torch.complex(real.float(), imag.float())
            x = torch.fft.irfft(spectrum, n=self.n_fft, dim=1, norm=norm)
            x = self._overlap_add(x * self.window, padded)
            window_envelope = self._overlap_add(
                self.window_envelope.expand(1, -1, frames).to(x.dtype), padded
            )

            pad = (self.n_fft - self.hop_length) // 2
            x = x[..., pad : samples + pad]
            window_envelope = window_envelope[..., pad : samples + pad]

            # Upstream asserts the envelope never vanishes.  At 50% overlap it
            # does not, but an assertion on a tensor value is a graph break
            # under ``torch.compile`` for a guarantee a clamp already provides.
            return x / window_envelope.clamp_min(1e-11)

    def _overlap_add(self, frames: Tensor, length: int) -> Tensor:
        return F.fold(
            frames,
            output_size=(1, length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).reshape(frames.shape[0], 1, length)


class LayerNorm2d(nn.Module):
    """Layer norm over ``(channels, bins, frames)``, or per frame if asked.

    ``framewise`` drops the time axis from the reduction, which makes the
    statistics causal in the sense that matters for streaming: a frame's
    normalisation no longer depends on the rest of the utterance.
    """

    def __init__(
        self,
        channels: int,
        framewise: bool = False,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.reduced_dim = [1, 2] if framewise else [1, 2, 3]
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.channels))
            self.beta = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(self.reduced_dim, keepdim=True)
        x = x - mean
        variance = (x**2).mean(dim=self.reduced_dim, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)
        return x


class DropPath(nn.Module):
    """Stochastic depth, per sample, on the residual branch."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:0.3f}"


class ConvNeXtBlock2d(nn.Module):
    """Depthwise 2D conv, norm, and an inverted bottleneck, closed residually.

    The kernel may be asymmetric: ``(13, 7)`` in the shipped config reaches
    further across frequency than across time, because the structure the block
    has to model -- a harmonic comb -- runs along the frequency axis.

    Reference: https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        channels: int,
        mult_channels: int,
        kernel_size: Union[int, Sequence[int]],
        drop_prob: float = 0.0,
        framewise_norm: bool = False,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Wavehax kernel sizes must be odd.")

        channels = int(channels)
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=channels,
            bias=False,
            padding_mode="reflect",
        )
        self.norm = LayerNorm2d(channels, framewise=framewise_norm)
        self.pwconv1 = nn.Conv2d(channels, channels * int(mult_channels), 1)
        self.nonlinear = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * int(mult_channels), channels, 1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(1, channels, 1, 1))
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.nonlinear(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return residual + self.drop_path(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class WavehaxGenerator(nn.Module):
    """Complex spectrogram estimation over a harmonic prior.

    ``initial_channel`` takes the place of the paper's ``in_channels``: this
    fork conditions on the frame-rate latent from the VITS frontend rather than
    on a mel spectrogram.  ``upsample_rates`` is not used to upsample anything
    -- there is no upsampling layer in this decoder -- it only carries the hop
    length the rest of the pipeline agreed on, as the product of its entries.
    """

    def __init__(
        self,
        initial_channel: int,
        sr: int,
        upsample_rates: Sequence[int],
        gin_channels: int = 0,
        checkpointing: bool = False,
        channels: int = 16,
        mult_channels: int = 3,
        convnext_kernel_size: Union[int, Sequence[int]] = (13, 7),
        num_blocks: int = 8,
        n_fft: Optional[int] = None,
        prior_type: str = "pcph_closed_form",
        prior_power_factor: float = 0.1,
        prior_noise_amplitude: float = 0.01,
        drop_prob: float = 0.0,
        framewise_norm: bool = False,
        use_logmag_phase: bool = False,
        **_: object,
    ):
        super().__init__()

        self.sr = int(sr)
        self.checkpointing = bool(checkpointing)
        self.upsample_rates = tuple(int(value) for value in upsample_rates)
        self.hop_length = math.prod(self.upsample_rates)
        self.initial_channel = int(initial_channel)
        # 50% overlap, the ratio the published configs use.  An n_fft that is
        # not a multiple of the hop would leave the Hann envelope non-constant
        # and the iSTFT would colour the output on a frame period.
        self.n_fft = int(n_fft or 2 * self.hop_length)
        if self.n_fft % self.hop_length != 0:
            raise ValueError(
                f"Wavehax n_fft ({self.n_fft}) must be a multiple of the hop "
                f"length ({self.hop_length})."
            )
        self.n_bins = self.n_fft // 2 + 1
        self.use_logmag_phase = bool(use_logmag_phase)

        prior_type = str(prior_type)
        if prior_type not in PRIOR_GENERATORS:
            raise ValueError(
                f"Unknown Wavehax prior '{prior_type}'; expected one of "
                f"{sorted(PRIOR_GENERATORS)}."
            )
        self.prior_type = prior_type
        self.prior_power_factor = float(prior_power_factor)
        self.prior_noise_amplitude = float(prior_noise_amplitude)

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        # The prior's two spectrogram planes share one projection: they are the
        # real and imaginary parts of the same signal, and upstream deliberately
        # ties the weights that read them.
        self.prior_proj = nn.Conv1d(
            self.n_bins, self.n_bins, 7, padding=3, padding_mode="reflect"
        )
        self.cond_proj = nn.Conv1d(
            self.initial_channel, self.n_bins, 7, padding=3, padding_mode="reflect"
        )
        # RVC's speaker conditioning, at the same point in the graph as
        # ``hifigan_nsf``'s and ``refinegan``'s: added to the frame-rate
        # conditioning before any block reads it.  Upstream Wavehax has no
        # equivalent because it is single-speaker.
        self.cond = (
            nn.Conv1d(int(gin_channels), self.n_bins, 1) if gin_channels else None
        )

        channels = int(channels)
        # Five planes: the prior's two, their projections, and the conditioning.
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels, framewise=framewise_norm)

        num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock2d(
                channels,
                mult_channels,
                convnext_kernel_size,
                drop_prob=float(drop_prob),
                framewise_norm=bool(framewise_norm),
                layer_scale_init_value=1.0 / num_blocks,
            )
            for _ in range(num_blocks)
        )

        self.output_norm = LayerNorm2d(channels, framewise=framewise_norm)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def generate_prior(self, f0: Tensor) -> Tensor:
        """``f0``: ``(batch, 1, frames)``.  Returns ``(batch, 1, frames * hop)``."""
        return PRIOR_GENERATORS[self.prior_type](
            f0,
            hop_length=self.hop_length,
            sample_rate=self.sr,
            power_factor=self.prior_power_factor,
            noise_amplitude=self.prior_noise_amplitude,
        )

    def forward_core(
        self,
        conditioning: Tensor,
        prior1: Tensor,
        prior2: Tensor,
        g: Optional[Tensor] = None,
    ) -> Tensor:
        """The 2D stack, kept free of the prior so it compiles as one graph."""
        prior1_proj = self.prior_proj(prior1)
        prior2_proj = self.prior_proj(prior2)
        conditioning = self.cond_proj(conditioning)
        if self.cond is not None and g is not None:
            conditioning = conditioning + self.cond(g)

        x = torch.stack(
            [prior1, prior2, prior1_proj, prior2_proj, conditioning], dim=1
        )
        x = self.input_proj(x)
        x = self.input_norm(x)

        for block in self.blocks:
            if self.training and self.checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.output_norm(x)
        x = self.output_proj(x)

        if self.use_logmag_phase:
            real, imag = to_real_imaginary(x[:, 0], x[:, 1])
        else:
            real, imag = x[:, 0], x[:, 1]

        # A single non-finite bin does not stay local: the iSTFT sums every bin
        # into every sample of its frame, so one overflow reaches every loss as
        # a NaN gradient over the whole frame.  ``nan_to_num`` alone is not
        # enough here -- its default maps an infinity to the largest finite
        # float, which the inverse FFT promptly overflows back to infinity.
        # A real spectrogram bin for audio in [-1, 1] cannot exceed ``n_fft`` in
        # magnitude, so this bound is an order of magnitude clear of anything
        # the model can legitimately emit and only ever binds on an overflow.
        bound = float(SPECTROGRAM_BOUND)
        real = torch.nan_to_num(real, posinf=bound, neginf=-bound).clamp(-bound, bound)
        imag = torch.nan_to_num(imag, posinf=bound, neginf=-bound).clamp(-bound, bound)
        return self.stft.inverse(real, imag)

    def forward(
        self,
        x: Tensor,
        f0: Tensor,
        g: Optional[Tensor] = None,
        template_amplitude: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        """``x``: ``(batch, initial_channel, frames)``, ``f0``: ``(batch, frames)``.

        ``template_amplitude`` is part of this fork's decoder contract so the
        synthesizer and the ablation probe can call every decoder the same way.
        It belongs to RefineGAN's pitch template, which this decoder does not
        have -- the harmonic prior sets its own amplitude from F0 -- so it is
        accepted and ignored.
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        # FP32, no grad: the phase accumulator underflows in FP16, and the
        # prior is a fixed function of F0 with nothing to learn.
        with torch.autocast(device_type=f0.device.type, enabled=False):
            with torch.no_grad():
                prior = self.generate_prior(f0.float().clamp_min(0.0))
                real, imag = self.stft(prior)
                if self.use_logmag_phase:
                    prior1, prior2 = to_log_magnitude_and_phase(real, imag)
                else:
                    prior1, prior2 = real, imag

        return self.forward_core(x, prior1.to(x.dtype), prior2.to(x.dtype), g)

    def remove_weight_norm(self) -> None:
        """No-op: Wavehax carries no weight normalization anywhere."""

    def __prepare_scriptable__(self):
        return self
