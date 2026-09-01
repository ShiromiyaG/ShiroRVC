import math

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights, get_padding
from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d


class ResBlock(nn.Module):
    """Residual block of dilated convolutions at multiple dilation rates."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        # The noise is a training-time regulariser, and at inference it is pure
        # cost: this module runs six times per decoder stage, so at the output
        # rate it draws 13.5 M samples per forward.  Measured on CPU at batch 4
        # over 0.4 s, dropping it in eval takes the whole generator from 418 ms
        # to 313 ms -- 25% of the forward.  ChouwaGAN's ``NoiseInjection``
        # already does exactly this.  Inference stays stochastic either way:
        # the NSF source draws its own noise, and that one is not optional --
        # it *is* the unvoiced content.
        if not self.training:
            return self.activation(x)

        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """Runs several ResBlocks (different kernel sizes) in parallel and averages them."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.input_conv.apply(init_weights)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class SineGenerator(nn.Module):
    """Sine + additive-noise harmonic excitation source."""
    #: f0 arrives already interpolated to the output rate.
    takes_frame_f0 = False

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim), dim = fundamental + overtones."""
        # rad_values is F0 in rad mod 1 (the integer cycle count doesn't affect phase)
        rad_values = (f0_values / self.sampling_rate) % 1

        # random initial phase per harmonic, none for the fundamental
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    # Inductor cannot compile this body.  ``_f02sine`` is a cumsum over the
    # sample axis, and Inductor lowers it to a ``SplitScan`` whose codegen
    # raises ``TypeError: list indices must be integers or slices, not
    # NoneType`` -- reproduced on torch 2.10 + cu130, RTX 5060.  A failure
    # inside the compiled region takes the *whole* decoder down with it, so
    # ``enable_decoder_compile`` fell back to eager for every step.
    #
    # Everything up to ``merge`` runs under ``no_grad`` and is a pure function
    # of f0, so keeping it out of the graph costs no fusion; ``merge`` is a
    # 1x1 linear plus a tanh, which is nothing to give up.  This is exactly
    # what ChouwaGAN's ``BandLimitedNSFSource`` does, and for the same reason.
    @torch.compiler.disable
    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # merge with grad
        return self.merge(sine_waves)


class CombToothGenerator(nn.Module):
    """Band-limited comb-tooth excitation, as an alternative to ``SineGenerator``.

    Ported from fish-diffusion's ``CombToothGen``
    (``fish_diffusion/modules/vocoders/refinegan/generator.py``), with the phase
    accumulated in float64 -- see ``_phase`` for why that is not optional here.

    Why it exists: ``SineGenerator`` is instantiated with ``harmonic_num=0``, so
    the excitation this decoder is handed is a *single* sine at the fundamental
    plus a 1x1 linear and a tanh.  Every partial above f0 therefore has to be
    manufactured by the trunk out of feature maps that live on the frame grid,
    and content synthesised from the frame grid is exactly what carries
    modulation at ``sr / hop``.  A comb tooth arrives with every harmonic
    already present at the output rate, so the trunk shapes an envelope instead
    of inventing partials, and the anti-aliased ``_decimate`` path feeds those
    harmonics into every upsampling stage rather than only the last.

    The waveform is alias-free by construction, which is the whole point of the
    ``sinc`` and is worth spelling out because it is not obvious: the phase
    ``x`` advances by exactly ``f0 / sr`` per sample, so ``sr * x / f0``
    advances by exactly 1 per sample.  The sampled values are therefore
    ``sinc(n - phi)`` for integer ``n`` -- an ideal band-limited unit impulse at
    a fractional offset, flat to Nyquist with nothing above it, at any f0.  The
    only approximation is the truncation to one period, and ``sinc`` is already
    down to ``2 f0 / (pi sr)`` at the period edge.
    """
    #: f0 arrives already interpolated to the output rate.
    takes_frame_f0 = False

    def __init__(
        self,
        samp_rate,
        wave_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sampling_rate = samp_rate
        self.wave_amp = wave_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    # Same reason as ``SineGenerator.forward``: the body is a cumsum over the
    # sample axis, Inductor lowers it to a ``SplitScan``, and its codegen
    # raises.  A failure inside the compiled region takes the whole decoder
    # down, so this stays out of the graph.  It is a pure function of f0 under
    # ``no_grad`` and carries no parameters, so nothing is given up.
    @torch.compiler.disable
    @torch.no_grad()
    def forward(self, f0: torch.Tensor):
        """``f0``: (batch, length, 1) at the output rate.  Returns the same shape."""
        # Accumulated in float64 and wrapped before anything else touches it.
        # The reference accumulates in float32, which is the same defect the
        # ChouwaGAN source had: the running cycle count is unbounded and grows
        # with the render, so the phase error grows *along the span*.  At 44.1
        # kHz a 30 s render reaches ~6e3 cycles, where a float32 ULP is ~5e-4
        # cycles; training only ever reads ``segment_size`` (0.4 s), so the
        # decoder would never be taught to cancel a defect it only meets at
        # inference.  ``x - round(x)`` is exact under the wrap because the two
        # differ by an integer number of cycles.
        cycles = torch.cumsum(f0.double() / self.sampling_rate, dim=1)
        x = (cycles - torch.round(cycles)).to(f0.dtype)

        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3)) * self.wave_amp

        # Gating is kept identical to ``SineGenerator`` -- a hard 0/1 mask and
        # the same two noise levels -- so switching sources changes the
        # excitation spectrum and nothing else.
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amp = uv * self.noise_std + (1 - uv) * self.wave_amp / 3
        noise = noise_amp * torch.randn_like(combtooth)

        return combtooth * uv + noise


class HarmonicBankGenerator(nn.Module):
    """A band-limited harmonic bank, as a third excitation for this decoder.

    ``SineGenerator`` gives the trunk one partial and ``CombToothGenerator``
    gives it a flat spectrum with a crest factor of 20; this sits between them.
    Measured against both at f0 = 80/220/880 Hz: RMS constant to four decimals
    where the comb drifts 8.7 dB, crest 2.6-3.0 where the comb reaches 20.7,
    20th harmonic at -13 dB where the sine is at -67 and the comb at 0, and
    0-17% of the energy above 10 kHz where the comb puts 55%.

    Derived from ``chouwagan.BandLimitedNSFSource``, deliberately as a *copy*
    rather than an import.  The two decoders train separately and that module's
    resamplers are already non-persistent buffers, so editing it in place
    silently changes an in-flight ChouwaGAN run -- the exact failure the
    decoder notes warn about.  The three corrections below therefore live here
    until a ChouwaGAN run is between checkpoints and they can be ported back.

    What is fixed relative to it is stated per method.  The frame-rate ghosting
    that motivated this class is **not** among them; ``forward`` says why.
    """

    #: ``RefineGANGenerator.forward`` hands this source f0 on the *frame* grid
    #: rather than the sample grid.  The gap fill and the voicing envelope both
    #: need to know where a frame boundary is, and neither can recover that
    #: from an already-interpolated f0: a sample sitting on the ramp between a
    #: voiced frame and a silent one reads as voiced at any threshold.
    takes_frame_f0 = True

    def __init__(
        self,
        samp_rate,
        harmonic_num: int = 64,
        wave_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        crossfade_seconds: float = 0.002,
        nyquist_centre: float = 0.45,
        nyquist_transition: float = 0.02,
    ):
        super().__init__()
        self.sampling_rate = float(samp_rate)
        self.harmonic_num = int(harmonic_num)
        self.wave_amp = float(wave_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)
        self.nyquist_centre = float(nyquist_centre)
        self.nyquist_transition = float(nyquist_transition)
        self.crossfade_samples = max(
            1, int(round(self.sampling_rate * float(crossfade_seconds)))
        )

        # Fix 1 of 3.  The reference draws these with ``torch.rand_like`` inside
        # its forward, so the *shape* of the excitation pulse is redrawn on
        # every step: the harmonics keep their amplitudes and change their
        # relative phases.  ChouwaGAN can carry that; this decoder cannot as
        # cheaply, because its excitation is decimated and concatenated into
        # all four upsampling stages rather than entering once, so a re-rolled
        # pulse shape reaches every stage of the trunk.  Registered persistent
        # rather than merely stored: the pulse shape is part of what the
        # decoder learns to invert, so it has to survive a checkpoint
        # round-trip.  Random once, not zero -- aligned phases are exactly the
        # comb, and the offsets are what buy the low crest factor.
        offsets = torch.rand(self.harmonic_num) * (2.0 * math.pi)
        offsets[0] = 0.0  # the fundamental keeps a defined phase origin
        self.register_buffer("phase_offset", offsets, persistent=True)

    @staticmethod
    def _fill_unvoiced(f0: torch.Tensor, voiced: torch.Tensor):
        """Hold the nearest voiced f0 across an unvoiced gap.

        Fix 2 of 3.  The reference interpolates the raw f0 sequence, in which
        unvoiced frames are literally ``0``, so every voiced/unvoiced boundary
        drags f0 down a linear ramp toward zero across one whole frame -- 10 ms
        at 44.1 kHz -- *inside the region the hard mask still calls voiced*.
        The phase integrates that ramp, so the pitch sags into the edge of
        every voiced run and snaps back out of it.  Holding the last voiced
        value keeps f0 flat through the gap; the hard gate then zeroes it
        anyway and the crossfaded envelope takes the amplitude down smoothly,
        so nothing is heard from the gap itself.
        """
        frames = f0.shape[-1]
        index = torch.arange(frames, device=f0.device).expand_as(f0)
        forward_src = torch.where(voiced, index, torch.full_like(index, -1))
        forward_src = forward_src.cummax(dim=-1).values
        backward_src = torch.where(voiced, index, torch.full_like(index, frames))
        backward_src = backward_src.flip(-1).cummin(dim=-1).values.flip(-1)
        source = torch.where(forward_src >= 0, forward_src, backward_src)
        filled = f0.gather(-1, source.clamp(0, frames - 1))
        # A sequence with no voiced frame at all has nothing to hold, and must
        # stay silent rather than inherit frame 0.
        return torch.where(voiced.any(-1, keepdim=True), filled, f0)

    def _voiced_envelope(self, mask: torch.Tensor):
        """Crossfade the 0/1 voicing mask without fading the segment's ends.

        Fix 3 of 3.  The reference smooths with ``F.conv1d(..., padding=radius)``,
        which zero-pads: fed a mask that is 1 everywhere it returns 0.506 at the
        first sample and reaches 1.0 only after ``radius``, at both ends.  Every
        training segment is therefore amplitude-faded at both boundaries --
        0.15% of a 0.4 s segment's energy, and at inference the first and last
        2 ms of a whole render.  Replicating the edge makes an all-voiced mask
        a fixed point, which is what a *crossfade between voiced and unvoiced*
        was supposed to be.
        """
        radius = self.crossfade_samples
        window = torch.hann_window(
            2 * radius + 1, periodic=False, device=mask.device, dtype=mask.dtype
        )
        window = window / window.sum()
        padded = F.pad(mask.unsqueeze(1), (radius, radius), mode="replicate")
        return F.conv1d(padded, window.view(1, 1, -1)).squeeze(1)

    # Same reason as the other two sources: a cumsum over the sample axis that
    # Inductor lowers to a ``SplitScan`` whose codegen raises, taking the whole
    # compiled decoder down with it.  A pure function of f0 under ``no_grad``.
    @torch.compiler.disable
    @torch.no_grad()
    def forward(self, f0: torch.Tensor, length: int):
        """``f0``: (batch, frames) on the frame grid.  Returns (batch, length, 1).

        NOT fixed here: the reported frame-rate ghosting.  It was not
        reproduced -- see ``tests/test_refinegan_harmonic_bank.py`` -- and with
        a constant f0 every term below is constant, so this source emits
        nothing at ``sr / hop`` by construction.  What it does emit, and the
        sine does not, is real energy at 5-20 kHz, which is the band the
        trunk's own upsampling images live in.  Until those are measured apart
        the suspect is the decoder's up path, not this.
        """
        if f0.ndim == 3:
            f0 = f0.squeeze(-1)
        if f0.ndim != 2:
            raise ValueError("F0 must have shape [batch, frames].")
        f0 = f0.float().clamp_min(0.0)

        voiced_frames = f0 > self.voiced_threshold
        f0 = self._fill_unvoiced(f0, voiced_frames)

        f0 = F.interpolate(
            f0.unsqueeze(1), size=length, mode="linear", align_corners=False
        ).squeeze(1)
        voiced = F.interpolate(
            voiced_frames.float().unsqueeze(1), size=length, mode="nearest"
        ).squeeze(1)
        # The pitch gate stays hard -- a fractional mask would scale f0 itself
        # and bend the phase around every boundary -- while the amplitude gate
        # is the crossfade, which is what keeps the edge from clicking.
        f0 = f0 * voiced
        envelope = self._voiced_envelope(voiced)

        # float64, wrapped before any harmonic index multiplies it.  A float32
        # cumsum of an unbounded phase loses precision *along the render*: the
        # error is inaudible over the 0.4 s a training segment reads and 20 dB
        # worse over a 30 s inference span, so the decoder would never be
        # taught to cancel a defect it only meets at generation time.  ``frac``
        # is exact under the wrap -- it shifts the phase by an integer number
        # of cycles, which an integer harmonic index preserves.
        cycles = torch.cumsum(f0.double() / self.sampling_rate, dim=-1)
        phase = ((2.0 * math.pi) * (cycles - cycles.floor())).to(f0.dtype)

        harmonics = torch.arange(
            1, self.harmonic_num + 1, device=f0.device, dtype=f0.dtype
        )
        wave = torch.sin(
            phase.unsqueeze(-1) * harmonics.view(1, 1, -1)
            + self.phase_offset.to(f0.dtype).view(1, 1, -1)
        )

        # Soft, and centred at 0.45 rather than 0.48 of the sample rate.  A hard
        # cutoff makes a harmonic blink on and off as vibrato walks it across
        # the threshold; soft leaks instead, and at 0.48 the leak is 27% *at
        # Nyquist itself*, where anything past it folds back inharmonically.
        # At 0.45 the leak is 7.6%, the transition stays 882 Hz wide, and all
        # that is given up is 19.8-22 kHz.
        transition = max(1.0, self.sampling_rate * self.nyquist_transition)
        mask = torch.sigmoid(
            (self.sampling_rate * self.nyquist_centre - f0.unsqueeze(-1) * harmonics)
            / transition
        )
        amplitude = harmonics.rsqrt().view(1, 1, -1)  # -3 dB per octave
        wave = (wave * amplitude * mask).sum(dim=-1)
        # Normalising by the surviving amplitudes is what holds the level
        # constant as f0 climbs and the mask retires the top harmonics; the
        # comb's level moves 8.7 dB across the pitch range for want of this.
        normalizer = (amplitude.square() * mask.square()).sum(dim=-1).sqrt()
        wave = wave / normalizer.clamp_min(1e-4)

        # ``wave_amp`` and not ``wave_amp / sqrt(2)``: the normaliser divides by
        # the *amplitude* norm, so the summed bank already lands at RMS
        # ``1/sqrt(2)`` -- the same place a unit sine lands.  Scaling by
        # ``wave_amp`` therefore puts this source at ``SineGenerator``'s RMS of
        # 0.0707 exactly, so an A/B changes the spectrum and nothing else.
        wave = wave * self.wave_amp * envelope

        noise_amp = voiced * self.noise_std + (1 - voiced) * self.wave_amp / 3
        return (wave + noise_amp * torch.randn_like(wave)).unsqueeze(-1)


class RefineGANGenerator(nn.Module):
    """
    RefineGAN generator: downsamples/upchannels the excitation source, fuses it
    with the mel input, and upsamples through parallel residual blocks.

    Args:
        source_type (str, optional): Excitation generator -- ``"sine"`` (a single
            sine at the fundamental), ``"comb"`` (a band-limited comb tooth,
            every harmonic present to Nyquist at a crest factor of up to 20) or
            ``"bank"`` (a harmonic bank at -3 dB/octave, level-normalised and
            Nyquist-masked). Defaults to ``"sine"``.
        source_harmonics (int, optional): Harmonic count for ``"bank"``, ignored
            by the other two. Defaults to 64.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        downsample_rates: tuple[int] = (2, 2, 8, 8),  # unused
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,  # unused
        gin_channels: int = 256,
        checkpointing: bool = False,
        upsample_initial_channel=512,
        filter_width: int = 12,
        rolloff: float = 0.90,
        filter_beta: float = 6.0,
        source_type: str = "sine",
        source_harmonics: int = 64,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        # ``int``, not the ``np.int64`` ``np.prod`` returns.  Dynamo wraps a
        # numpy scalar used inside a traced function as a *CPU* tensor, and one
        # CPU node is enough to make Inductor emit a C++ kernel -- which on
        # Windows needs ``cl.exe`` and fails the whole compile with
        # ``InvalidCxxCompiler``.  ChouwaGAN uses ``math.prod`` here.  This is
        # one of three independent CPU/codegen sources in this decoder -- the
        # other two are marked ``torch.compiler.disable`` below -- and removing
        # any one of them alone still fails the compile.
        self.upp = int(np.prod(upsample_rates))

        # ``sine`` is the default because it is what every existing checkpoint
        # was trained with, and because it is the one that owns a parameter:
        # ``SineGenerator.merge`` is a ``Linear(1, 1)`` and lands in the state
        # dict as ``dec.m_source.merge.0.weight``.  ``comb`` has no parameters
        # at all, so the two state dicts differ by that key -- and ``net_g``
        # loads non-strictly, which would leave a comb-trained checkpoint's
        # sine ``merge`` at its random init instead of failing.  The choice is
        # therefore spelled into ``architecture_id`` (``rvc/configs/vocoders.py``)
        # so the guard catches the swap by name rather than by luck.
        self.source_type = str(source_type).lower()
        if self.source_type == "sine":
            self.m_source = SineGenerator(sample_rate)
        elif self.source_type == "comb":
            self.m_source = CombToothGenerator(sample_rate)
        elif self.source_type == "bank":
            self.m_source = HarmonicBankGenerator(
                sample_rate, harmonic_num=int(source_harmonics)
            )
        else:
            raise ValueError(
                "refinegan_source must be 'sine', 'comb' or 'bank', "
                f"not {source_type!r}."
            )

        self.pre_conv = weight_norm(
            nn.Conv1d(
                1,
                16,
                7,
                1,
                padding=3,
            )
        )

        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []
        for i, u in enumerate(upsample_rates):

            new_size = int(size / upsample_rates[-i - 1])
            # T dimension factors for torchaudio.functional.resample
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        channels = upsample_initial_channel

        self.mel_conv = weight_norm(
            nn.Conv1d(
                num_mels,
                channels // 2,
                7,
                1,
                padding=3,
            )
        )

        self.mel_conv.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels // 2, 1)

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for rate in upsample_rates:
            new_channels = channels // 2

            # Was ``nn.Upsample(mode="linear")``.  Linear interpolation is a
            # triangular kernel, and a triangular kernel is a bad anti-image
            # filter: measured, it rejects the first image of a component at
            # 95% of the stage's Nyquist by 1.7 dB at 2x and 9.6 dB at 8x.  The
            # feature maps entering the first stage sit on a 100 Hz frame grid
            # and carry content right up to that Nyquist, so the images came
            # through as a mirrored partial on either side of every harmonic,
            # spaced by the frame rate -- the same defect ChouwaGAN had from a
            # rolloff with no transition band left.  The down path already
            # resamples with a windowed sinc (below); this makes the up path
            # agree with it instead of undoing it.
            self.upsample_blocks.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=filter_width,
                    rolloff=rolloff,
                    filter_beta=filter_beta,
                )
            )

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

    # The other half of the compile story.  ``torchaudio.functional.resample``
    # builds its sinc kernel from Python ints on every call, and Inductor
    # compiles that construction to a *CPU* kernel, which on Windows needs
    # ``cl.exe``; without MSVC the whole decoder fails with
    # ``InvalidCxxCompiler: Compiler: cl is not found`` and drops to eager.
    # Kept out of the graph rather than replaced: torchaudio's filter is
    # 385/953 taps against the 73/169 a ``FixedLowPass1d`` of the same shape as
    # the upsamplers would build, and its stopband is 135-156 dB against 68-78,
    # so swapping it would be a numerical change smuggled in under a build fix.
    @torch.compiler.disable
    def _decimate(self, x: torch.Tensor, orig_freq: int, new_freq: int):
        return torchaudio.functional.resample(
            x.contiguous(),
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        # The harmonic bank needs f0 on the *frame* grid -- it fills unvoiced
        # gaps and builds the voicing crossfade, and neither survives being
        # handed an already-interpolated f0, where a sample on the ramp into a
        # silent frame reads as voiced at any threshold.  It does its own
        # interpolation; the other two sources keep the call they always had.
        if self.m_source.takes_frame_f0:
            har_source = self.m_source(f0, f0_size * self.upp).transpose(1, 2)
        else:
            f0 = F.interpolate(
                f0.unsqueeze(1), size=f0_size * self.upp, mode="linear"
            )
            har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        x = self.pre_conv(har_source)
        downs = []
        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """Fold every weight norm back into its weight, by walking the modules.

        The hand-written version this replaces could not work: it called the
        *old* ``torch.nn.utils.remove_weight_norm`` on layers built with the
        parametrization API, which raises ``weight_norm of 'weight' not
        found`` on the very first one; it then called ``.remove_weight_norm()``
        on ``downsample_blocks``, which are bare ``Conv1d`` objects with no such
        method; and it reached ``ParallelResBlock.input_conv``, which carries no
        weight norm at all.  Nothing caught it because ``Synthesizer`` walks the
        decoder itself and never calls this.  Walking the modules is what
        ChouwaGAN does, and it cannot go stale when a layer is added.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
