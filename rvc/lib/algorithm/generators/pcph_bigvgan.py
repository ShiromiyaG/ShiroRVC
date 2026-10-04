"""PCPH-BigVGAN: BigVGAN v2's anti-aliased SnakeBeta trunk, driven by a
harmonic-plus-noise excitation.

    mel -> conv_pre (+ speaker) -> prenet
        -> per stage: projection -> sinc upsampler -> + excitation -> AMP blocks
        -> SnakeBeta -> conv_post (+ filtered noise) -> DC removal -> tanh

The excitation is made at the output rate from f0 and decimated to each
stage's rate.
"""

from contextlib import nullcontext
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import (
    register_parametrization,
    remove_parametrizations,
)
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import expand_f0, get_padding, init_weights
from rvc.lib.algorithm.generators.refinegan2 import (
    DC_WINDOW_SECONDS,
    DEFAULT_UPSAMPLE_BETA,
    DEFAULT_UPSAMPLE_WIDTH,
    SOURCE_GAIN_KERNEL,
    SineGenerator,
    UnitNorm,
    _match_output_gain,
    _refuse_learned_output_gain,
    remove_dc,
)
from rvc.lib.algorithm.resampling import (
    AntiAliasedActivation,
    AntiAliasedUpsample1d,
    FixedLowPass1d,
    filter_schedule,
)


try:
    from rvc.lib.algorithm.generators.snake_triton import (
        FusedResidualSnakeBeta,
        FusedSnakeBeta,
    )
except ImportError:
    FusedSnakeBeta = FusedResidualSnakeBeta = None


#: Decimation filter that brings the excitation down to each stage's rate.
#: With hundreds of partials in the source, a strided conv would fold them.
#: Long enough that the stopband (~80 dB) starts just below the new Nyquist,
#: since the PCPH source has full power up to the old one.
SOURCE_DECIMATION = dict(width=24, rolloff=0.88, filter_beta=8.0)

#: Channels of the rectified source branch at the output rate; doubled per
#: stage on the way down, as RefineGAN2's ``start_channels``.
SOURCE_BRANCH_CHANNELS = 16
SOURCE_BRANCH_SLOPE = 0.1

#: Round trip of the deep source's rectifiers. Three in series at the default
#: width 16 put 21 kHz at -12.5 dB; width 32 keeps it at -2.9. Rolloff low
#: enough that the stopband starts at Nyquist: at 1.0 the transition band
#: straddles it, and with the PCPH source's energy there training diverged.
DEEP_SOURCE_ACTIVATION = dict(filter_width=64, rolloff=0.97)

#: The ``fundamental`` source joins the last stage at or below this rate, Hz:
#: its sine then reaches a 3 kHz f0, and the stages after it make the harmonics.
FUNDAMENTAL_SOURCE_RATE = 6000.0

#: Bias that starts a ``softplus`` gain at 1: ``log(e - 1)``.
UNIT_SOFTPLUS_BIAS = 0.5413248546129181

#: Hidden width of a pre-net block, in multiples of its channels.
PRENET_EXPANSION = 3

#: RefineGAN2's upsampler widths and betas with each rolloff lowered until the
#: stopband starts at the input's Nyquist, so no image lands just above it.
UPSAMPLE_ROLLOFF = (0.84, 0.92, 0.94, 0.94)

#: SnakeBeta's round trip, 65 taps as the fused kernel takes: the stopband
#: starts at the stage's Nyquist, so nothing it makes folds into the band.
SNAKE_ACTIVATION = dict(filter_width=16, rolloff=0.89, filter_beta=5.0)
#: The output stage may fold into its top band while that stays above this.
AUDIBLE_LIMIT = 20000.0
#: Below 1: at 1.0 the PCPH source's energy at Nyquist made training diverge.
MAX_OUTPUT_ROLLOFF = 0.99


def output_design(sample_rate: int, filter_width: int = 16, filter_beta: float = 6.0) -> dict:
    """A 2x round trip at the output rate whose folds stay above
    ``AUDIBLE_LIMIT``, or fold nowhere when the rate leaves no room."""
    # Kaiser's design formulas: the stopband attenuation ``filter_beta`` gives,
    # then the transition width, as a fraction of Nyquist, halved.
    attenuation = filter_beta / 0.1102 + 8.7
    half_band = (attenuation - 8.0) / (28.72 * filter_width)
    # The stopband starts at Nyquist: nothing folds.
    no_fold = 1.0 - half_band
    # The stopband starts where its mirror image around Nyquist is the limit.
    inaudible_fold = 2.0 - AUDIBLE_LIMIT / (sample_rate / 2.0) - half_band
    rolloff = min(MAX_OUTPUT_ROLLOFF, max(no_fold, inaudible_fold))
    return dict(filter_width=filter_width, rolloff=round(rolloff, 3), filter_beta=filter_beta)


class _SnakeBetaFunction(torch.autograd.Function):
    """SnakeBeta that saves only its input for backward.

    Autograd over the plain expression keeps ``alpha * x``, its sine and the
    square alive -- three more tensors at the oversampled rate per activation.
    """

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta):
        ctx.save_for_backward(x, log_alpha, log_beta)
        # Per-channel parameters, broadcast over (batch, channels, time).
        alpha = log_alpha.exp()[:, None]
        inv_beta = torch.exp(-log_beta)[:, None]
        return torch.addcmul(x, torch.sin(x * alpha).square(), inv_beta)

    @staticmethod
    def backward(ctx, grad):
        x, log_alpha, log_beta = ctx.saved_tensors
        alpha = log_alpha.exp()[:, None]
        inv_beta = torch.exp(-log_beta)[:, None]
        sine = torch.sin(x * alpha)
        # d/du sin^2(u) = sin(2u) = 2 sin(u) cos(u)
        slope = grad * (2.0 * sine * torch.cos(x * alpha))
        grad_x = torch.addcmul(grad, slope, alpha * inv_beta).to(x.dtype)
        # The parameters are logs, so each gradient carries the parameter
        # itself as a factor; summed over batch and time, one per channel.
        grad_log_alpha = (slope * x).sum((0, 2)) * (alpha * inv_beta)[:, 0]
        grad_log_beta = -(grad * sine.square()).sum((0, 2)) * inv_beta[:, 0]
        return grad_x, grad_log_alpha.to(log_alpha.dtype), grad_log_beta.to(log_beta.dtype)


class SnakeBeta(nn.Module):
    """BigVGAN v2's ``x + sin^2(alpha * x) / beta``, per-channel, log-scale."""

    def __init__(self, channels: int):
        super().__init__()
        # Logs of alpha and beta: zeros start both at 1.
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SnakeBetaFunction.apply(x, self.alpha, self.beta)


class StageRateActivation(nn.Module):
    """SnakeBeta at the stage's own rate, with ``AntiAliasedActivation``'s keys."""

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class AntiAliasedSnakeBeta(AntiAliasedActivation):
    """``AntiAliasedActivation(SnakeBeta)``, fused into one Triton op on CUDA.

    Same keys and same output; the unfused path runs on the CPU or without
    Triton.
    """

    def __init__(self, channels: int, design: dict = SNAKE_ACTIVATION):
        super().__init__(SnakeBeta(channels), **design)
        # The fused kernel's lowpass is fixed at 65 taps: factor 2, width 16.
        self.fusable = self.design[:2] == (2, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FusedSnakeBeta is None or not x.is_cuda or not self.fusable:
            return super().forward(x)
        return FusedSnakeBeta.apply(x, *self._fused_args(x))

    def forward_residual(self, x: torch.Tensor, r: torch.Tensor):
        """``(self(x + r), x + r)`` with the add inside the kernel; CUDA only."""
        return FusedResidualSnakeBeta.apply(x, r, *self._fused_args(x))

    def _fused_args(self, x: torch.Tensor):
        """The fused op's arguments after its inputs: SnakeBeta's parameters,
        both filters and the output dtype."""
        # Written in the dtype the next conv runs in rather than cast to it.
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        else:
            dtype = x.dtype
        return (
            self.activation.alpha,
            self.activation.beta,
            self.up.phase_weight,
            self.down.kernel[0, 0],
            self.up.phase_pad,
            dtype,
        )


def snake_activation(channels: int, antialias: bool = True,
                     design: dict = SNAKE_ACTIVATION) -> nn.Module:
    """SnakeBeta, oversampled 2x with ``antialias``."""
    if antialias:
        return AntiAliasedSnakeBeta(channels, design)
    return StageRateActivation(SnakeBeta(channels))


def _conv(channels: int, kernel_size: int, dilation: int = 1) -> nn.Module:
    """A weight-normed conv that keeps both the channels and the length."""
    conv = weight_norm(
        nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )
    conv.apply(init_weights)
    return conv


class AMPBlock(nn.Module):
    """BigVGAN's AMPBlock: dilated convs behind SnakeBeta activations.

    ``pairs`` is AMPBlock1 (two convs and two activations per dilation);
    without it, AMPBlock2, with half of each.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: Sequence[int],
        antialias: bool = True,
        pairs: bool = True,
        design: dict = SNAKE_ACTIVATION,
    ):
        super().__init__()
        # One residual unit per dilation: act -> dilated conv, and with
        # ``pairs`` a second act -> undilated conv.
        self.convs1 = nn.ModuleList([_conv(channels, kernel_size, d) for d in dilation])
        self.acts1 = nn.ModuleList(
            [snake_activation(channels, antialias, design) for _ in dilation]
        )
        if pairs:
            self.convs2 = nn.ModuleList([_conv(channels, kernel_size) for _ in dilation])
            self.acts2 = nn.ModuleList(
                [snake_activation(channels, antialias, design) for _ in dilation]
            )

        # Set by ``apply_precision_policy`` under AMP.
        self.fp32_residuals = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_residuals:
            x = x.float()
        if self._fusable(x):
            return self._forward_fused(x)
        if not hasattr(self, "convs2"):
            for conv, act in zip(self.convs1, self.acts1):
                x = conv(act(x)) + x
            return x
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = conv2(act2(conv1(act1(x)))) + x
        return x

    def _fusable(self, x: torch.Tensor) -> bool:
        """Whether ``_forward_fused`` can take ``x``."""
        return (
            FusedResidualSnakeBeta is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and all(isinstance(act, AntiAliasedSnakeBeta) and act.fusable for act in self.acts1)
        )

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` with each residual add done inside the next activation."""
        # The previous unit's output, still to be added to ``x``.
        branch = None
        for index, (conv1, act1) in enumerate(zip(self.convs1, self.acts1)):
            if branch is None:
                hidden = act1(x)
            else:
                hidden, x = act1.forward_residual(x, branch)
            branch = conv1(hidden)
            if hasattr(self, "convs2"):
                branch = self.convs2[index](self.acts2[index](branch))
        return branch + x


class PrenetBlock(nn.Module):
    """ConvNeXt block at the mel frame rate, where width is cheap."""

    def __init__(self, channels: int, layer_scale: float):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * PRENET_EXPANSION)
        self.down = nn.Linear(channels * PRENET_EXPANSION, channels)
        self.gamma = nn.Parameter(torch.full((channels,), float(layer_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channels last for the LayerNorm and the two Linears.
        y = self.depthwise(x).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y)))) * self.gamma
        return x + y.transpose(1, 2)


def source_conv(in_channels: int, channels: int, deep: bool) -> nn.Module:
    """The conv that adds a stage's excitation; ``deep`` makes it three convs
    with oversampled rectifiers between, to shape the excitation at that
    stage's rate. At the stage rate the rectifiers would fold inharmonically."""
    if not deep:
        return nn.Conv1d(in_channels, channels, 7, 1, padding=3)
    return nn.Sequential(
        nn.Conv1d(in_channels, channels, 7, 1, padding=3),
        AntiAliasedActivation(leaky_relu_slope=SOURCE_BRANCH_SLOPE, **DEEP_SOURCE_ACTIVATION),
        nn.Conv1d(channels, channels, 7, 1, padding=3),
        AntiAliasedActivation(leaky_relu_slope=SOURCE_BRANCH_SLOPE, **DEEP_SOURCE_ACTIVATION),
        nn.Conv1d(channels, channels, 7, 1, padding=3),
    )


class PCPHSource(nn.Module):
    """Pseudo-constant-power harmonic excitation, Wavehax's PCPH prior.

    Every harmonic up to ``max_frequency`` (Nyquist when ``None``), each at
    ``k`` times the fundamental's phase, so they add up to one band-limited
    pulse per period; the sine source's per-partial random phases leave the
    high band unpulsed. The amplitude ``sine_amp * sqrt(2 / n)`` keeps the
    sum's power at ``sine_amp**2`` for any f0. The harmonics below
    ``1 - NYQUIST_TAPER`` of the limit are summed in closed form; the ones
    above fade out one by one, as ``SineGenerator``'s do, so f0 crossing a
    harmonic boundary does not click. ``pulsed_noise`` fills the band above
    the limit with noise at the harmonics' power per Hz, in one burst per
    period at the pulse: the pulse's timing without the harmonics' lines.
    Noise as the sine source's: ``noise_std`` in voiced samples,
    ``sine_amp / 3`` in unvoiced. ``noise_eq``, ``(hz, db)`` points
    interpolated linearly, shapes the voiced noise's spectrum; the stages'
    filters pass it unevenly. Owns no state-dict key.
    """

    NYQUIST_TAPER = SineGenerator.NYQUIST_TAPER
    #: The bursts' envelope is ``cos(pi * phase) ** 4``, whose mean power is
    #: 35 / 128; this brings it to 1.
    BURST_GAIN = (128.0 / 35.0) ** 0.5

    def __init__(self, samp_rate, sine_amp=0.1, noise_std=0.003, voiced_threshold=0,
                 random_start_phase=False, noise_eq=None, max_frequency=None,
                 pulsed_noise=False):
        super().__init__()
        self.sampling_rate = samp_rate
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.random_start_phase = bool(random_start_phase)
        nyquist = samp_rate / 2.0
        self.max_frequency = nyquist if max_frequency is None else float(max_frequency)
        if not 0.0 < self.max_frequency <= nyquist:
            raise ValueError(
                f"max_frequency must be above 0 and at most {nyquist} Hz, "
                f"received {max_frequency}."
            )
        self.pulsed_noise = bool(pulsed_noise)
        if self.pulsed_noise and self.max_frequency >= nyquist:
            raise ValueError("pulsed_noise fills the band above max_frequency; set one below Nyquist.")
        # (hz, db) arrays sorted by frequency, as ``np.interp`` takes them.
        self.noise_eq = None
        if noise_eq:
            points = sorted((float(hz), float(db)) for hz, db in noise_eq)
            self.noise_eq = (np.array([p[0] for p in points]), np.array([p[1] for p in points]))

    def _equalize(self, noise: torch.Tensor) -> torch.Tensor:
        """``noise`` with the ``noise_eq`` curve applied to its spectrum."""
        hz, db = self.noise_eq
        freqs = np.fft.rfftfreq(noise.shape[-1], 1.0 / self.sampling_rate)
        gain = torch.from_numpy(10.0 ** (np.interp(freqs, hz, db) / 20.0)).to(noise.device, torch.float32)
        return torch.fft.irfft(torch.fft.rfft(noise.float(), dim=-1) * gain, n=noise.shape[-1])

    @torch.compiler.disable  # the sample-axis cumsum, as ``SineGenerator._phase``
    def _phase(self, f0):
        """The fundamental's phase in cycles, float64, (batch, length)."""
        phase = torch.cumsum(f0.double() / self.sampling_rate, dim=-1)
        if self.random_start_phase and self.training:
            phase = phase + torch.rand(phase.shape[0], 1, device=phase.device, dtype=phase.dtype)
        return phase % 1.0

    @staticmethod
    def _sine_sum(phase, count):
        """``sum_{k=1..count} sin(2 pi k phase)`` by the Dirichlet identity.
        The angles are reduced in float64: ``count`` reaches hundreds."""
        half = phase * 0.5
        numerator = torch.sin(2 * np.pi * ((count * half) % 1.0)).float() * torch.sin(
            2 * np.pi * (((count + 1) * half) % 1.0)
        ).float()
        denominator = torch.sin(np.pi * phase).float()
        # At the start of a period the sum is 0 and the identity 0 / 0.
        safe = denominator.abs() > 1e-6
        return torch.where(safe, numerator / torch.where(safe, denominator, 1.0), 0.0)

    def _band(self, safe_f0, phase, limit, lowest_f0):
        """(sum, power) of the harmonics up to ``limit``: their sines summed,
        and their squared amplitudes. Without ``phase``, the power alone."""
        taper = limit * self.NYQUIST_TAPER
        # Harmonics below the taper band, all at full amplitude.
        full_count = torch.floor((limit - taper) / safe_f0)
        harmonics = None if phase is None else self._sine_sum(phase, full_count)
        power = full_count.float()
        # Harmonics in the top band, faded by their own frequency. The lowest
        # f0 has the most of them.
        fading_count = int(np.ceil(taper / lowest_f0)) + 1 if lowest_f0 else 0
        for offset in range(1, fading_count + 1):
            order = full_count + offset
            fade = ((limit - order * safe_f0) / taper).clamp(0.0, 1.0).float()
            if phase is not None:
                harmonics = harmonics + fade * torch.sin(2 * np.pi * ((order * phase) % 1.0)).float()
            power = power + fade.square()
        return harmonics, power

    def _harmonics(self, f0, voiced, phase, limit=None):
        """The pulse train, (batch, length); silent in unvoiced samples. A
        ``limit`` below ``max_frequency`` ends the harmonics there at the level
        they have in the whole train: the same pulses, band-limited."""
        top = self.max_frequency
        limit = top if limit is None else min(float(limit), top)
        # Unvoiced samples take an f0 that leaves them no harmonic.
        safe_f0 = torch.where(voiced > 0, f0, top).double()
        voiced_f0 = f0[voiced > 0]
        lowest_f0 = voiced_f0.min().item() if voiced_f0.numel() else None

        harmonics, power = self._band(safe_f0, phase, limit, lowest_f0)
        if limit < top:
            # The sum of the whole train's squared amplitudes sets the level.
            power = self._band(safe_f0, None, top, lowest_f0)[1]
        return harmonics * self.sine_amp * torch.sqrt(2.0 / power.clamp(min=1.0)) * voiced

    def _pulsed_noise(self, voiced, phase):
        """Noise above ``max_frequency``, (batch, length), in one burst per
        period; silent in unvoiced samples."""
        limit = self.max_frequency
        taper = limit * self.NYQUIST_TAPER
        envelope = self.BURST_GAIN * torch.cos(np.pi * phase).float() ** 4
        noise = torch.randn_like(voiced) * envelope * voiced
        # The harmonics' fade, power-complemented: the two cross over flat.
        freqs = torch.fft.rfftfreq(noise.shape[-1], 1.0 / self.sampling_rate, device=noise.device)
        gain = torch.sqrt(1.0 - ((limit - freqs) / taper).clamp(0.0, 1.0).square())
        # The harmonics' power sits in this width; the fade squared holds a
        # third of the taper's.
        band = limit - 2.0 * taper / 3.0
        level = self.sine_amp * (self.sampling_rate / 2.0 / band) ** 0.5
        return level * torch.fft.irfft(torch.fft.rfft(noise, dim=-1) * gain, n=noise.shape[-1])

    def _noise(self, voiced):
        """White noise in unvoiced samples; quieter, and shaped by
        ``noise_eq``, in voiced ones."""
        noise = torch.randn_like(voiced)
        voiced_noise = self._equalize(noise) if self.noise_eq is not None else noise
        return voiced * self.noise_std * voiced_noise + (1 - voiced) * self.sine_amp / 3 * noise

    def components(self, f0):
        """f0: (batch, length) in Hz. Returns (pulses, noise), each (batch,
        length): the harmonics, with the pulsed noise, and the steady noise."""
        with torch.no_grad():
            voiced = (f0 > self.voiced_threshold).float()
            phase = self._phase(torch.where(voiced > 0, f0, 0.0))
            pulses = self._harmonics(f0, voiced, phase)
            if self.pulsed_noise:
                pulses = pulses + self._pulsed_noise(voiced, phase)
            return pulses, self._noise(voiced)

    def forward(self, f0, gain=None):
        """f0: (batch, length, 1) in Hz. Returns (batch, length, 1)."""
        if gain is not None:
            raise ValueError("PCPHSource takes no source gain.")
        pulses, noise = self.components(f0[..., 0])
        return (pulses + noise).unsqueeze(-1)


class FundamentalSource(nn.Module):
    """A sine at the fundamental alone, at the power ``PCPHSource``'s pulses
    have, made at the rate of the one stage it is added to: an anchor for the
    phase, with the harmonics and the noise left to the trunk and the noise
    branch. Nothing to band-limit, decimate or equalise. Silent in unvoiced
    samples and where f0 passes that rate's Nyquist. Owns no state-dict key.
    """

    def __init__(self, samp_rate, sine_amp=0.1, voiced_threshold=0, random_start_phase=False):
        super().__init__()
        self.sampling_rate = samp_rate
        self.sine_amp = sine_amp
        self.voiced_threshold = voiced_threshold
        self.random_start_phase = bool(random_start_phase)

    @torch.compiler.disable  # the sample-axis cumsum, as ``PCPHSource._phase``
    def forward(self, f0):
        """f0: (batch, length) in Hz at this source's rate. Returns (batch, 1, length)."""
        with torch.no_grad():
            voiced = ((f0 > self.voiced_threshold) & (f0 < self.sampling_rate / 2.0)).float()
            phase = torch.cumsum((f0 * voiced).double() / self.sampling_rate, dim=-1)
            if self.random_start_phase and self.training:
                phase = phase + torch.rand(phase.shape[0], 1, device=phase.device, dtype=phase.dtype)
            sine = torch.sin(2 * np.pi * (phase % 1.0)).float()
            return (sine * voiced * self.sine_amp * 2.0**0.5).unsqueeze(1)


def exp_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """DDSP's positive, bounded gain: 2 at most, 1e-7 at least."""
    return 2.0 * torch.sigmoid(x) ** np.log(10.0) + 1e-7


def _widen_noise_head(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    """Load hook: a head saved without ``f0_input`` takes zero weights on the
    two new inputs, which leaves its gains as they were."""
    key = prefix + "head.0.weight"
    saved = state_dict.get(key)
    extra = module.head[0].in_channels - saved.shape[1] if saved is not None else 0
    if extra > 0:
        state_dict[key] = F.pad(saved, (0, 0, 0, extra))


class NoiseBranch(nn.Module):
    """Filtered noise added at the output, and gains on the excitation, both
    per frame and per band, read from the mel by one small head.

    The excitation's noise reaches the output through every stage's filters,
    which pass little of it above ~8 kHz and leave a dip at the next-to-last
    stage's Nyquist; this noise bypasses them. The excitation gains let the
    mel scale the harmonics too, which the PCPH source puts at full power up
    to Nyquist. ``bands`` are spaced on the mel scale up to Nyquist, and the
    gains between their centres interpolated linearly. The noise starts at
    about -80 dB and the excitation gains at 1. With ``split_source`` the
    excitation's pulses and its noise take separate gains, so the mel sets
    their ratio in each band. With ``f0_input`` the head also reads each
    frame's pitch and whether it is voiced: the mel alone does not say whether
    a high band holds harmonics or noise.
    """

    HIDDEN = 128
    #: exp_sigmoid(-4.3) ~ 1e-4.
    NOISE_START = -4.3
    #: The head's pitch input is ``log2(f0 / F0_REFERENCE)``, 0 when unvoiced.
    F0_REFERENCE = 220.0

    def __init__(self, num_mels: int, bands: int, sample_rate: int, hop: int,
                 split_source: bool = False, f0_input: bool = False):
        super().__init__()
        self.bands, self.hop, self.n_fft = int(bands), int(hop), 4 * int(hop)
        self.sample_rate = int(sample_rate)
        self.split_source = bool(split_source)
        self.f0_input = bool(f0_input)
        # Outputs ``bands`` noise gains, then ``bands`` excitation gains, or
        # with ``split_source`` the pulses' and then the excitation noise's.
        groups = 3 if self.split_source else 2
        self.head = nn.Sequential(
            # The pitch and the voiced flag come after the mel's channels.
            nn.Conv1d(num_mels + (2 if self.f0_input else 0), self.HIDDEN, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.HIDDEN, self.HIDDEN, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.HIDDEN, groups * self.bands, 3, padding=1),
        )
        # Zero weights: the gains start at their biases, whatever the mel.
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias[: self.bands], self.NOISE_START)
        nn.init.zeros_(last.bias[self.bands :])

        # Band centres evenly spaced on the mel scale, from 0 Hz to Nyquist.
        top_mel = 2595.0 * np.log10(1.0 + sample_rate / 2.0 / 700.0)
        self.centres = 700.0 * (10.0 ** (np.linspace(0.0, top_mel, self.bands) / 2595.0) - 1.0)
        self.register_buffer("interp", self._interp(self.n_fft, 1), persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        # ``_tables`` of the rates below the output's, by (decimation, device).
        self._decimated = {}
        self.register_load_state_dict_pre_hook(_widen_noise_head)

    def _interp(self, n_fft: int, decimation: int) -> torch.Tensor:
        """(bins, bands): the weight of each band's gain in each STFT bin of a
        signal at ``1 / decimation`` of the output rate."""
        freqs = np.fft.rfftfreq(n_fft, decimation / self.sample_rate)
        interp = np.stack([np.interp(freqs, self.centres, row) for row in np.eye(self.bands)], axis=1)
        return torch.from_numpy(interp).float()

    def _tables(self, decimation: int, device):
        """(hop, n_fft, window, interp) for a signal at ``1 / decimation`` of
        the output rate, with the same frames."""
        if decimation == 1:
            return self.hop, self.n_fft, self.window, self.interp
        key = (decimation, device)
        if key not in self._decimated:
            hop = self.hop // decimation
            self._decimated[key] = (
                hop, 4 * hop, torch.hann_window(4 * hop, device=device),
                self._interp(4 * hop, decimation).to(device),
            )
        return self._decimated[key]

    def gains(self, mel: torch.Tensor, f0: torch.Tensor = None):
        """(noise, excitation) gains, each (batch, bands, frames); with
        ``split_source`` the excitation's are a (pulses, noise) pair. ``f0``
        is Hz per frame, (batch, 1, frames), read with ``f0_input``."""
        if self.f0_input:
            if f0.shape[-1] != mel.shape[-1]:
                f0 = F.interpolate(f0, size=mel.shape[-1], mode="nearest")
            voiced = (f0 > 0).to(mel.dtype)
            pitch = torch.log2(f0.clamp_min(1.0) / self.F0_REFERENCE).to(mel.dtype) * voiced
            mel = torch.cat((mel, pitch, voiced), dim=1)
        noise, *excitation = self.head(mel).float().split(self.bands, dim=1)
        excitation = [2.0 * torch.sigmoid(gain) for gain in excitation]
        return exp_sigmoid(noise), excitation if self.split_source else excitation[0]

    def shape(self, signal: torch.Tensor, gains: torch.Tensor, decimation: int = 1) -> torch.Tensor:
        """``signal`` (batch, samples) through ``gains`` (batch, bands,
        frames), frame t centred on sample t * hop. ``decimation`` is how far
        below the output rate the signal is."""
        hop, n_fft, window, interp = self._tables(decimation, signal.device)
        with torch.autocast(signal.device.type, enabled=False):
            spectrum = torch.stft(signal.float(), n_fft, hop, window=window, return_complex=True)
            # The centred STFT has one frame more than the mel.
            frames = spectrum.shape[-1]
            if gains.shape[-1] < frames:
                gains = F.pad(gains, (0, frames - gains.shape[-1]), mode="replicate")
            per_bin = torch.einsum("fk,bkt->bft", interp, gains[..., :frames].float())
            return torch.istft(spectrum * per_bin, n_fft, hop, window=window, length=signal.shape[-1])


class PCPHBigVGANGenerator(nn.Module):
    """BigVGAN v2 at voice scale, driven by a harmonic-plus-noise excitation.

    Against BigVGAN v2: RefineGAN2's width (``upsample_initial_channel`` 512
    instead of 1536) and stage layout, so the same ``v4`` discriminator and
    32 kHz config apply; the learned transposed convolutions replaced by a
    channel projection at the input rate followed by RefineGAN2's fixed
    windowed-sinc upsampler; and the excitation, band-limited down to each
    stage's rate, added after every upsampler as in NSF-HiFi-GAN. The
    activations stay BigVGAN's -- SnakeBeta at 2x oversampling.  The output is
    ``tanh`` of a unit-norm ``conv_post``, as in RefineGAN2, with its DC
    removed first (``remove_dc``).

    Args:
        source_gain (bool, optional): Scale the excitation by per-band
            envelopes projected from ``z`` and the speaker, as RefineGAN2 does.
        source_harmonics (int, optional): Partials above the fundamental in
            the excitation. Sizes ``m_source.merge.0.weight``.
        source_tilt (float, optional): Partial ``j`` gets amplitude
            ``j ** -tilt``.
        source_phase (str, optional): ``random`` or ``coherent`` initial
            phases for the partials; see ``SineGenerator``.
        source_phase_jitter (float, optional): Per-call spread of the
            coherent phases, in cycles.
        source_type (str, optional): ``sine``, the NSF sine source the
            ``source_harmonics``/``source_tilt``/``source_phase`` options
            shape, or ``pcph``, ``PCPHSource``: every harmonic to Nyquist,
            phase-locked, at constant power. PCPH takes none of those options
            nor ``source_gain``. ``pcph_staged`` is the same pulses made at
            each stage's own rate from one phase track, with the harmonics
            that fit under that rate's Nyquist, instead of made at the output
            rate and filtered down; its noise is still filtered down.
            ``fundamental`` is ``FundamentalSource``: one sine, added at one
            stage, and no source noise. The last two need ``source_branch``
            ``linear``.
        source_stage (int, optional): The stage ``fundamental`` is added
            after; ``None`` takes the last one below ``FUNDAMENTAL_SOURCE_RATE``.
        source_max_frequency (float, optional): Hz where the PCPH source's
            harmonics end, and over which its power is normalised; Nyquist
            when ``None``.
        source_pulsed_noise (bool, optional): Fill the band above
            ``source_max_frequency`` with pitch-synchronous noise; see
            ``PCPHSource``.
        source_branch (str, optional): ``linear`` adds the one-channel
            excitation to every stage through a conv. ``rectified`` runs it
            through a conv and an oversampled ``leaky_relu`` at the output
            rate first, then a learned multi-channel pyramid, as RefineGAN2
            does: SnakeBeta is smooth and makes few harmonics of its own, and
            the rectifier's kink gives every harmonic the fundamental's phase,
            one pulse per period.
        source_random_start_phase (bool, optional): Start the excitation's
            fundamental at a random phase on every training call; see
            ``SineGenerator``. Inference is unaffected.
        source_noise_eq (sequence of (hz, db), optional): Shape of the PCPH
            source's voiced noise; see ``PCPHSource``.
        noise_branch_bands (int, optional): With more than 0, a
            ``NoiseBranch`` of that many bands: mel-driven noise added at the
            output and band gains on the excitation.
        noise_branch_split_source (bool, optional): Separate band gains for
            the PCPH source's pulses and its noise. Sizes
            ``noise_branch.head.4.weight``.
        noise_branch_f0 (bool, optional): The ``NoiseBranch`` head reads the
            pitch and the voiced flag beside the mel. Sizes
            ``noise_branch.head.0.weight``; a checkpoint saved without it
            loads, with zeros on the new inputs.
        output_gain (bool, optional): A learned output level ``exp(s)`` on the
            unit-norm ``conv_post``, as in RefineGAN2.
        resblock (str, optional): ``"1"`` for AMPBlock1, ``"2"`` for AMPBlock2,
            which has half the convs and half the activations.
        antialias (bool or sequence of bool, optional): Oversample the
            activations, per stage. The work is proportional to channels x
            rate, which doubles every stage, so the last stage is about half
            of it and the first two about a fifth.
        stage_channels (sequence of int, optional): Width of each stage;
            ``None`` halves ``upsample_initial_channel`` at every stage.
        prenet_blocks (int, optional): ConvNeXt blocks after ``conv_pre``, at
            the mel frame rate.
        deep_source_stages (int, optional): How many of the last stages add
            the excitation through three convs instead of one.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        upsample_rates: Sequence[int] = (5, 4, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5),) * 3,
        resblock: str = "1",
        antialias: "bool | Sequence[bool]" = True,
        num_mels: int = 192,
        gin_channels: int = 256,
        checkpointing: bool = False,
        filter_width: "int | Sequence[int]" = DEFAULT_UPSAMPLE_WIDTH,
        rolloff: "float | Sequence[float]" = UPSAMPLE_ROLLOFF,
        filter_beta: "float | Sequence[float]" = DEFAULT_UPSAMPLE_BETA,
        source_gain: bool = False,
        source_noise_std: float = 0.003,
        source_harmonics: int = 0,
        source_tilt: float = 1.0,
        source_phase: str = "random",
        source_phase_jitter: float = 0.0,
        source_branch: str = "linear",
        source_type: str = "sine",
        source_random_start_phase: bool = False,
        source_noise_eq: "Sequence[Sequence[float]] | None" = None,
        source_max_frequency: "float | None" = None,
        source_pulsed_noise: bool = False,
        source_stage: "int | None" = None,
        noise_branch_bands: int = 0,
        noise_branch_split_source: bool = False,
        noise_branch_f0: bool = False,
        output_gain: bool = False,
        stage_channels: "Sequence[int] | None" = None,
        prenet_blocks: int = 0,
        deep_source_stages: int = 0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        # Odd, so the moving average is centred.
        self.dc_window = int(DC_WINDOW_SECONDS * self.sample_rate) | 1
        self.upsample_rates = tuple(int(rate) for rate in upsample_rates)
        # Output samples per input frame.
        self.upp = int(np.prod(self.upsample_rates))
        self.checkpointing = checkpointing
        self.num_kernels = len(resblock_kernel_sizes)

        # Per-stage options, each brought to one entry per stage.
        count = len(self.upsample_rates)
        if stage_channels is None:
            stage_channels = [
                int(upsample_initial_channel) // 2 ** (stage + 1) for stage in range(count)
            ]
        stage_channels = [int(value) for value in stage_channels]
        if len(stage_channels) != count:
            raise ValueError(
                f"stage_channels has {len(stage_channels)} entries for {count} stages."
            )
        deep_source_stages = int(deep_source_stages)
        if not 0 <= deep_source_stages <= count:
            raise ValueError(
                f"deep_source_stages must be between 0 and {count}, received {deep_source_stages}."
            )
        # First stage with a deep source.
        self.deep_source_from = count - deep_source_stages
        if str(resblock) not in ("1", "2"):
            raise ValueError(f"resblock must be '1' or '2', received {resblock!r}.")
        if isinstance(antialias, bool):
            antialias = (antialias,) * count
        antialias = tuple(bool(value) for value in antialias)
        if len(antialias) != count:
            raise ValueError(
                f"antialias has {len(antialias)} entries for {count} stages."
            )
        # Keys are the same either way, so ``decoder_layout`` records this.
        self.snake_layout = {"resblock": str(resblock), "antialias": list(antialias)}
        self.filter_width = filter_schedule(filter_width, count, "filter_width", 1)
        self.rolloff = filter_schedule(rolloff, count, "rolloff", 0.0)
        self.filter_beta = filter_schedule(filter_beta, count, "filter_beta", 0.0)

        # Read by the checkpoint guards in ``rvc/train/checkpoints.py``.
        self.source_type = str(source_type)
        self.source_harmonics = int(source_harmonics)
        self.source_tilt = float(source_tilt)
        self.source_phase = str(source_phase)
        self.source_phase_jitter = float(source_phase_jitter)
        pcph_options = dict(
            noise_eq=source_noise_eq,
            max_frequency=source_max_frequency,
            pulsed_noise=bool(source_pulsed_noise),
        )
        # Output samples per sample of each stage's output, first stage first.
        self.stage_steps = [
            int(np.prod(self.upsample_rates[stage + 1 :])) for stage in range(count)
        ]
        # The stages that take the excitation: all, or ``fundamental``'s one.
        self.source_stages = tuple(range(count))
        if self.source_type == "fundamental":
            if source_stage is None:
                below = [
                    stage for stage, step in enumerate(self.stage_steps)
                    if self.sample_rate / step <= FUNDAMENTAL_SOURCE_RATE
                ]
                source_stage = below[-1] if below else 0
            if not 0 <= int(source_stage) < count:
                raise ValueError(f"source_stage must be between 0 and {count - 1}, received {source_stage}.")
            self.source_stages = (int(source_stage),)
        elif source_stage is not None:
            raise ValueError("source_stage is the 'fundamental' source's.")
        if self.source_type in ("pcph_staged", "fundamental") and source_branch != "linear":
            raise ValueError(f"source_type {self.source_type!r} needs source_branch 'linear'.")
        self.m_source = self._build_source(
            source_gain, float(source_noise_std), source_random_start_phase, pcph_options
        )

        # Output-rate excitation -> each earlier stage's rate, last stage
        # first. ``fundamental`` is made at its stage's rate.
        self.source_downs = nn.ModuleList(
            [
                FixedLowPass1d(rate, stride=rate, **SOURCE_DECIMATION)
                for rate in reversed(self.upsample_rates[1:])
            ]
            if self.source_type != "fundamental"
            else []
        )

        if source_branch not in ("linear", "rectified"):
            raise ValueError(
                f"source_branch must be 'linear' or 'rectified', not {source_branch!r}."
            )
        self.source_branch = source_branch
        source_channels = self._build_source_branch(count)

        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
            self._build_source_gain(num_mels, gin_channels)

        self.split_source_gains = bool(noise_branch_split_source)
        if self.split_source_gains and (
            self.source_type not in ("pcph", "pcph_staged") or int(noise_branch_bands) <= 0
        ):
            raise ValueError(
                "noise_branch_split_source needs source_type 'pcph' or 'pcph_staged' and "
                "noise_branch_bands."
            )
        if noise_branch_f0 and int(noise_branch_bands) <= 0:
            raise ValueError("noise_branch_f0 needs noise_branch_bands.")
        self.noise_branch = (
            NoiseBranch(
                num_mels, noise_branch_bands, self.sample_rate, self.upp,
                self.split_source_gains, noise_branch_f0,
            )
            if int(noise_branch_bands) > 0
            else None
        )

        # The trunk at the mel frame rate.
        channels = int(upsample_initial_channel)
        self.conv_pre = weight_norm(nn.Conv1d(num_mels, channels, 7, 1, padding=3))
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, channels, 1)
        self.prenet = (
            nn.Sequential(
                *[PrenetBlock(channels, 1.0 / prenet_blocks) for _ in range(prenet_blocks)]
            )
            if prenet_blocks > 0
            else None
        )

        # The stages. ``resblocks`` is flat: ``num_kernels`` blocks per stage.
        output_snake = output_design(self.sample_rate)
        self.projections = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.source_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for stage, rate in enumerate(self.upsample_rates):
            new_channels = stage_channels[stage]
            # Narrowing the channels before the upsampler keeps the conv at the
            # input rate, where it costs 1/rate of the same conv after it.
            self.projections.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            self.ups.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=self.filter_width[stage],
                    rolloff=self.rolloff[stage],
                    filter_beta=self.filter_beta[stage],
                )
            )
            if stage in self.source_stages:
                self.source_convs.append(
                    source_conv(
                        source_channels[stage],
                        new_channels,
                        deep=stage >= self.deep_source_from,
                    )
                )
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    AMPBlock(
                        new_channels,
                        kernel,
                        dilation,
                        antialias=antialias[stage],
                        pairs=str(resblock) == "1",
                        design=output_snake if stage == count - 1 else SNAKE_ACTIVATION,
                    )
                )
            channels = new_channels
        self.projections.apply(init_weights)

        # No skip around these two, so their lowpass is the output's band: long
        # filters, unfused, to keep it flat close to Nyquist.
        final = output_design(self.sample_rate, filter_width=64, filter_beta=8.0)
        self.activation_post = snake_activation(channels, antialias[-1], final)
        self.output_act = AntiAliasedActivation(nn.Tanh(), **final)
        # Unit norm, not weight norm: a learned output gain took 99% of the
        # generator's gradient in the first pretrain, as it did in RefineGAN2.
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        register_parametrization(self.conv_post, "weight", UnitNorm())
        self.register_load_state_dict_pre_hook(_refuse_learned_output_gain)
        self.has_output_gain = bool(output_gain)
        if self.has_output_gain:
            self.output_log_gain = nn.Parameter(torch.zeros(()))
        self.register_load_state_dict_pre_hook(_match_output_gain)

        # Set by ``apply_precision_policy`` under AMP: the excitation, the
        # upsampling filters and the output layer then run in FP32.
        self.fp32_residuals = False

    def _build_source(self, source_gain, noise_std, random_start_phase, pcph_options) -> nn.Module:
        """The excitation generator ``source_type`` names; ``pcph_options``
        are ``PCPHSource``'s own."""
        if self.source_type in ("pcph", "pcph_staged", "fundamental") and (
            source_gain or self.source_harmonics or self.source_phase != "random"
            or self.source_phase_jitter
        ):
            raise ValueError(
                f"source_type {self.source_type!r} sets its own harmonics, phases and "
                "level; source_gain, source_harmonics, source_phase and "
                "source_phase_jitter are the sine source's."
            )
        if self.source_type == "fundamental":
            if any(pcph_options.values()):
                raise ValueError(
                    "source_noise_eq, source_max_frequency and source_pulsed_noise are the "
                    "PCPH sources'; 'fundamental' has neither harmonics nor noise."
                )
            stage = self.source_stages[0]
            return FundamentalSource(
                self.sample_rate / self.stage_steps[stage], random_start_phase=random_start_phase
            )
        if self.source_type in ("pcph", "pcph_staged"):
            if self.source_type == "pcph_staged" and pcph_options["pulsed_noise"]:
                raise ValueError("source_pulsed_noise is not made per stage; use source_type 'pcph'.")
            return PCPHSource(
                self.sample_rate,
                noise_std=noise_std,
                random_start_phase=random_start_phase,
                **pcph_options,
            )
        if any(pcph_options.values()):
            raise ValueError(
                "source_noise_eq, source_max_frequency and source_pulsed_noise are the "
                "PCPH source's; source_type is 'sine'."
            )
        if self.source_type == "sine":
            return SineGenerator(
                self.sample_rate,
                harmonic_num=self.source_harmonics,
                noise_std=noise_std,
                harmonic_tilt=self.source_tilt,
                harmonic_phase=self.source_phase,
                phase_jitter=self.source_phase_jitter,
                random_start_phase=random_start_phase,
            )
        raise ValueError(
            "source_type must be 'sine', 'pcph', 'pcph_staged' or 'fundamental', "
            f"not {self.source_type!r}."
        )

    def _build_source_branch(self, count: int) -> list:
        """Adds the ``rectified`` branch's layers. Returns the excitation's
        channels at each stage, first stage first."""
        if self.source_branch != "rectified":
            return [1] * count
        source_channels = [
            SOURCE_BRANCH_CHANNELS * 2**index for index in range(count)
        ][::-1]
        self.source_pre = weight_norm(
            nn.Conv1d(1, SOURCE_BRANCH_CHANNELS, 7, 1, padding=3)
        )
        self.source_act = AntiAliasedActivation(
            leaky_relu_slope=SOURCE_BRANCH_SLOPE,
            **output_design(self.sample_rate),
        )
        # One per decimation, last stage first: each doubles the channels.
        self.source_blocks = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels * 2, 7, 1, padding=3))
                for channels in reversed(source_channels[1:])
            ]
        )
        return source_channels

    def _build_source_gain(self, num_mels: int, gin_channels: int) -> None:
        """Adds the layers that project the sine source's gains from the
        input frames and the speaker, and bring them to the output rate."""
        gain_channels = self.m_source.gain_channels
        self.source_gain = nn.Conv1d(
            num_mels,
            gain_channels,
            SOURCE_GAIN_KERNEL,
            padding=SOURCE_GAIN_KERNEL // 2,
        )
        # Same start as RefineGAN2: unit gain on every partial band, the
        # two filtered noise channels off.
        nn.init.zeros_(self.source_gain.weight)
        nn.init.constant_(self.source_gain.bias, UNIT_SOFTPLUS_BIAS)
        with torch.no_grad():
            self.source_gain.bias[-3] = -6.0
            self.source_gain.bias[-1] = -6.0
        if gin_channels != 0:
            self.source_gain_cond = nn.Conv1d(gin_channels, gain_channels, 1)
            nn.init.zeros_(self.source_gain_cond.weight)
            nn.init.zeros_(self.source_gain_cond.bias)
        self.source_gain_ups = nn.ModuleList(
            [
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=self.filter_width[stage],
                    rolloff=self.rolloff[stage],
                    filter_beta=self.filter_beta[stage],
                )
                for stage, rate in enumerate(self.upsample_rates)
            ]
        )

    def _fp32_region(self, x: torch.Tensor):
        """A context with autocast off, when ``fp32_residuals`` is set."""
        if not self.fp32_residuals:
            return nullcontext()
        return torch.autocast(x.device.type, enabled=False)

    def _fp32(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() if self.fp32_residuals else x

    def _source_gain(self, z: torch.Tensor, g: torch.Tensor = None):
        """Excitation gains, (batch, frames * upp, gain_channels), or None."""
        if not self.has_source_gain:
            return None
        gain = self.source_gain(z)
        if g is not None and hasattr(self, "source_gain_cond"):
            gain = gain + self.source_gain_cond(g)
        for ups in self.source_gain_ups:
            gain = ups(gain)
        return F.softplus(gain).transpose(1, 2)

    def _excitation(self, z, f0, g, band_gains=None):
        """The excitation at every stage's output rate, first stage first;
        ``band_gains`` are the noise branch's on the excitation."""
        if self.source_type == "fundamental":
            stage = self.source_stages[0]
            f0 = expand_f0(f0, z.shape[-1] * self.upp // self.stage_steps[stage])
            sources = [None] * len(self.upsample_rates)
            with self._fp32_region(z):
                sources[stage] = self.m_source(f0[:, 0])
            return sources
        f0 = expand_f0(f0, z.shape[-1] * self.upp)
        if self.source_type == "pcph_staged":
            with self._fp32_region(z):
                return self._staged_excitation(f0[:, 0], band_gains)
        with self._fp32_region(z):
            gain = self._source_gain(self._fp32(z), None if g is None else self._fp32(g))
            if self.split_source_gains:
                pulses, noise = self.m_source.components(f0[:, 0])
                pulse_gains, noise_gains = band_gains
                shape = self.noise_branch.shape
                source = (shape(pulses, pulse_gains) + shape(noise, noise_gains)).unsqueeze(1)
            else:
                # The source takes and returns (batch, length, 1).
                source = self.m_source(f0.transpose(1, 2), gain).transpose(1, 2)
                if band_gains is not None:
                    source = self.noise_branch.shape(source[:, 0], band_gains).unsqueeze(1)
            # Built from the output rate down, each step one decimation.
            if self.source_branch == "rectified":
                sources = [self.source_act(self.source_pre(source))]
                for down, block in zip(self.source_downs, self.source_blocks):
                    sources.append(block(down(sources[-1])))
            else:
                sources = [source]
                for down in self.source_downs:
                    sources.append(down(sources[-1]))
        return sources[::-1]

    def _staged_excitation(self, f0, band_gains):
        """``pcph_staged``: the pulses made at each stage's rate from the
        output-rate phase track, so the stages' pulses line up, and the
        output-rate noise filtered down. f0: (batch, samples) in Hz."""
        source = self.m_source
        with torch.no_grad():
            voiced = (f0 > source.voiced_threshold).float()
            phase = source._phase(torch.where(voiced > 0, f0, 0.0))
            noise = source._noise(voiced)
        pulse_gains = noise_gains = band_gains
        if self.split_source_gains:
            pulse_gains, noise_gains = band_gains
        shape = None if self.noise_branch is None else self.noise_branch.shape
        if noise_gains is not None:
            noise = shape(noise, noise_gains)
        # From the output rate down, as ``source_downs``.
        noises = [noise.unsqueeze(1)]
        for down in self.source_downs:
            noises.append(down(noises[-1]))
        sources = []
        for step, noise in zip(self.stage_steps, noises[::-1]):
            with torch.no_grad():
                pulses = source._harmonics(
                    f0[:, ::step], voiced[:, ::step], phase[:, ::step],
                    limit=self.sample_rate / step / 2.0,
                )
            if pulse_gains is not None:
                pulses = shape(pulses, pulse_gains, step)
            sources.append(pulses.unsqueeze(1) + noise)
        return sources

    def _upsample(self, stage: int, x: torch.Tensor) -> torch.Tensor:
        x = self.projections[stage](x)
        with self._fp32_region(x):
            return self.ups[stage](self._fp32(x))

    def _add_source(self, stage: int, source: torch.Tensor) -> torch.Tensor:
        """The stage's excitation, in the trunk's channels."""
        # A deep source filters in its rectifiers, and the filters stay in FP32.
        conv = self.source_convs[self.source_stages.index(stage)]
        if stage < self.deep_source_from:
            return conv(source)
        with self._fp32_region(source):
            return conv(self._fp32(source))

    def _resblocks(self, stage: int, x: torch.Tensor, checkpointed: bool) -> torch.Tensor:
        """The mean of the stage's AMP blocks, one per kernel size, each on ``x``."""
        blocks = self.resblocks[
            stage * self.num_kernels : (stage + 1) * self.num_kernels
        ]
        total = None
        for block in blocks:
            y = checkpoint(block, x, use_reentrant=False) if checkpointed else block(x)
            total = y if total is None else total + y
        return total / self.num_kernels

    def _output(self, x: torch.Tensor, noise_gains) -> torch.Tensor:
        """The last stage's features -> the waveform, (batch, 1, samples)."""
        with self._fp32_region(x):
            x = self.activation_post(self._fp32(x))
            x = self.conv_post(x)
            if self.has_output_gain:
                x = x * self.output_log_gain.exp()
            if noise_gains is not None:
                noise = torch.randn(x.shape[0], x.shape[-1], device=x.device)
                x = x + self.noise_branch.shape(noise, noise_gains).unsqueeze(1)
            # SnakeBeta's features all have a positive mean, and the waveform
            # discriminators push the output's DC around: summed over every
            # sample, that coherent term was ~99% of conv_post's gradient.
            x = remove_dc(x, self.dc_window)
            # Not v2's clamp: with no gain on ``conv_post`` the output starts
            # past 1, and a saturated clamp passes no gradient to the trunk.
            # Oversampled, since its odd harmonics would fold at this rate.
            return self.output_act(x)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        """x: (batch, num_mels, frames); f0: Hz per frame, (batch, frames) or
        (batch, 1, frames); g: speaker embedding, (batch, gin_channels, 1).
        Returns (batch, 1, frames * upp)."""
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        noise_gains = excitation_gains = None
        if self.noise_branch is not None:
            noise_gains, excitation_gains = self.noise_branch.gains(x, f0)
        sources = self._excitation(x, f0, g, excitation_gains)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        if self.prenet is not None:
            x = self.prenet(x)

        checkpointed = self.training and self.checkpointing
        for stage in range(len(self.upsample_rates)):
            if checkpointed:
                x = checkpoint(self._upsample, stage, x, use_reentrant=False)
            else:
                x = self._upsample(stage, x)
            if sources[stage] is not None:
                x = x + self._add_source(stage, sources[stage])
            x = self._resblocks(stage, x, checkpointed)

        return self._output(x, noise_gains)

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
