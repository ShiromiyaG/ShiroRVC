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
    attenuation = filter_beta / 0.1102 + 8.7
    # Kaiser's transition width, as a fraction of Nyquist, halved.
    half_band = (attenuation - 8.0) / (28.72 * filter_width)
    audible = 2.0 - AUDIBLE_LIMIT / (sample_rate / 2.0) - half_band
    rolloff = min(MAX_OUTPUT_ROLLOFF, max(1.0 - half_band, audible))
    return dict(filter_width=filter_width, rolloff=round(rolloff, 3), filter_beta=filter_beta)


class _SnakeBetaFunction(torch.autograd.Function):
    """SnakeBeta that saves only its input for backward.

    Autograd over the plain expression keeps ``alpha * x``, its sine and the
    square alive -- three more tensors at the oversampled rate per activation.
    """

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta):
        ctx.save_for_backward(x, log_alpha, log_beta)
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
        grad_log_alpha = (slope * x).sum((0, 2)) * (alpha * inv_beta)[:, 0]
        grad_log_beta = -(grad * sine.square()).sum((0, 2)) * inv_beta[:, 0]
        return grad_x, grad_log_alpha.to(log_alpha.dtype), grad_log_beta.to(log_beta.dtype)


class SnakeBeta(nn.Module):
    """BigVGAN v2's ``x + sin^2(alpha * x) / beta``, per-channel, log-scale."""

    def __init__(self, channels: int):
        super().__init__()
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
        # The fused kernel's lowpass is fixed at 65 taps.
        self.fusable = self.design[:2] == (2, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FusedSnakeBeta is None or not x.is_cuda or not self.fusable:
            return super().forward(x)
        return FusedSnakeBeta.apply(x, *self._fused_args(x))

    def forward_residual(self, x: torch.Tensor, r: torch.Tensor):
        """``(self(x + r), x + r)`` with the add inside the kernel; CUDA only."""
        return FusedResidualSnakeBeta.apply(x, r, *self._fused_args(x))

    def _fused_args(self, x: torch.Tensor):
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
    if antialias:
        return AntiAliasedSnakeBeta(channels, design)
    return StageRateActivation(SnakeBeta(channels))


def _conv(channels: int, kernel_size: int, dilation: int = 1) -> nn.Module:
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
            for c1, a1 in zip(self.convs1, self.acts1):
                x = c1(a1(x)) + x
            return x
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x

    def _fusable(self, x: torch.Tensor) -> bool:
        return (
            FusedResidualSnakeBeta is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and all(isinstance(act, AntiAliasedSnakeBeta) and act.fusable for act in self.acts1)
        )

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` with each residual add done inside the next activation."""
        y = None
        for index, (c1, a1) in enumerate(zip(self.convs1, self.acts1)):
            if y is None:
                h = a1(x)
            else:
                h, x = a1.forward_residual(x, y)
            y = c1(h)
            if hasattr(self, "convs2"):
                y = self.convs2[index](self.acts2[index](y))
        return y + x


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

    Every harmonic up to Nyquist, each at ``k`` times the fundamental's phase,
    so they add up to one band-limited pulse per period; the sine source's
    per-partial random phases leave the high band unpulsed. The amplitude
    ``sine_amp * sqrt(2 / n)`` keeps the sum's power at ``sine_amp**2`` for
    any f0. The harmonics below ``1 - NYQUIST_TAPER`` of Nyquist are summed in
    closed form; the ones above fade out one by one, as ``SineGenerator``'s do,
    so f0 crossing a harmonic boundary does not click. Noise as the sine
    source's: ``noise_std`` in voiced samples, ``sine_amp / 3`` in unvoiced.
    ``noise_eq``, ``(hz, db)`` points interpolated linearly, shapes the voiced
    noise's spectrum; the stages' filters pass it unevenly. Owns no state-dict
    key.
    """

    NYQUIST_TAPER = SineGenerator.NYQUIST_TAPER

    def __init__(self, samp_rate, sine_amp=0.1, noise_std=0.003, voiced_threshold=0,
                 random_start_phase=False, noise_eq=None):
        super().__init__()
        self.sampling_rate = samp_rate
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.random_start_phase = bool(random_start_phase)
        self.noise_eq = None
        if noise_eq:
            points = sorted((float(hz), float(db)) for hz, db in noise_eq)
            self.noise_eq = (np.array([p[0] for p in points]), np.array([p[1] for p in points]))

    def _equalize(self, noise: torch.Tensor) -> torch.Tensor:
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
        safe = denominator.abs() > 1e-6
        return torch.where(safe, numerator / torch.where(safe, denominator, 1.0), 0.0)

    def forward(self, f0, gain=None):
        """f0: (batch, length, 1) in Hz. Returns (batch, length, 1)."""
        if gain is not None:
            raise ValueError("PCPHSource takes no source gain.")
        with torch.no_grad():
            f0 = f0[..., 0]
            uv = (f0 > self.voiced_threshold).float()
            nyquist = self.sampling_rate / 2.0
            taper = nyquist * self.NYQUIST_TAPER
            safe_f0 = torch.where(uv > 0, f0, nyquist).double()
            phase = self._phase(torch.where(uv > 0, f0, 0.0))

            full = torch.floor((nyquist - taper) / safe_f0)
            harmonics = self._sine_sum(phase, full)
            power = full.float()
            # Harmonics in the top band, faded by their own frequency.
            voiced_f0 = f0[uv > 0]
            fading = int(np.ceil(taper / voiced_f0.min().item())) + 1 if voiced_f0.numel() else 0
            for offset in range(1, fading + 1):
                order = full + offset
                fade = ((nyquist - order * safe_f0) / taper).clamp(0.0, 1.0).float()
                harmonics = harmonics + fade * torch.sin(2 * np.pi * ((order * phase) % 1.0)).float()
                power = power + fade.square()

            harmonics = harmonics * self.sine_amp * torch.sqrt(2.0 / power.clamp(min=1.0)) * uv
            noise = torch.randn_like(uv)
            voiced = self._equalize(noise) if self.noise_eq is not None else noise
            noise = uv * self.noise_std * voiced + (1 - uv) * self.sine_amp / 3 * noise
            return (harmonics + noise).unsqueeze(-1)


def exp_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """DDSP's positive, bounded gain: 2 at most, 1e-7 at least."""
    return 2.0 * torch.sigmoid(x) ** np.log(10.0) + 1e-7


class NoiseBranch(nn.Module):
    """Filtered noise added at the output, and gains on the excitation, both
    per frame and per band, read from the mel by one small head.

    The excitation's noise reaches the output through every stage's filters,
    which pass little of it above ~8 kHz and leave a dip at the next-to-last
    stage's Nyquist; this noise bypasses them. The excitation gains let the
    mel scale the harmonics too, which the PCPH source puts at full power up
    to Nyquist. ``bands`` are spaced on the mel scale up to Nyquist, and the
    gains between their centres interpolated linearly. The noise starts at
    about -80 dB and the excitation gains at 1.
    """

    HIDDEN = 128
    #: exp_sigmoid(-4.3) ~ 1e-4.
    NOISE_START = -4.3

    def __init__(self, num_mels: int, bands: int, sample_rate: int, hop: int):
        super().__init__()
        self.bands, self.hop, self.n_fft = int(bands), int(hop), 4 * int(hop)
        self.head = nn.Sequential(
            nn.Conv1d(num_mels, self.HIDDEN, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.HIDDEN, self.HIDDEN, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.HIDDEN, 2 * self.bands, 3, padding=1),
        )
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias[: self.bands], self.NOISE_START)
        nn.init.zeros_(last.bias[self.bands :])

        top = 2595.0 * np.log10(1.0 + sample_rate / 2.0 / 700.0)
        centres = 700.0 * (10.0 ** (np.linspace(0.0, top, self.bands) / 2595.0) - 1.0)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        interp = np.stack([np.interp(freqs, centres, row) for row in np.eye(self.bands)], axis=1)
        self.register_buffer("interp", torch.from_numpy(interp).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def gains(self, mel: torch.Tensor):
        """(noise, excitation) gains, each (batch, bands, frames)."""
        noise, excitation = self.head(mel).float().chunk(2, dim=1)
        return exp_sigmoid(noise), 2.0 * torch.sigmoid(excitation)

    def shape(self, signal: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """``signal`` (batch, samples) through ``gains`` (batch, bands,
        frames), frame t centred on sample t * hop."""
        with torch.autocast(signal.device.type, enabled=False):
            spectrum = torch.stft(
                signal.float(), self.n_fft, self.hop, window=self.window, return_complex=True
            )
            frames = spectrum.shape[-1]
            if gains.shape[-1] < frames:
                gains = F.pad(gains, (0, frames - gains.shape[-1]), mode="replicate")
            per_bin = torch.einsum("fk,bkt->bft", self.interp, gains[..., :frames].float())
            return torch.istft(
                spectrum * per_bin, self.n_fft, self.hop, window=self.window, length=signal.shape[-1]
            )


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
            nor ``source_gain``.
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
        noise_branch_bands: int = 0,
        output_gain: bool = False,
        stage_channels: "Sequence[int] | None" = None,
        prenet_blocks: int = 0,
        deep_source_stages: int = 0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.dc_window = int(DC_WINDOW_SECONDS * self.sample_rate) | 1
        self.upsample_rates = tuple(int(rate) for rate in upsample_rates)
        self.upp = int(np.prod(self.upsample_rates))
        self.checkpointing = checkpointing
        self.num_kernels = len(resblock_kernel_sizes)

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
        if self.source_type == "pcph":
            if source_gain or self.source_harmonics or self.source_phase != "random" \
                    or self.source_phase_jitter:
                raise ValueError(
                    "source_type 'pcph' has every harmonic, phase-locked, at a fixed "
                    "level; source_gain, source_harmonics, source_phase and "
                    "source_phase_jitter are the sine source's."
                )
            self.m_source = PCPHSource(
                self.sample_rate,
                noise_std=float(source_noise_std),
                random_start_phase=source_random_start_phase,
                noise_eq=source_noise_eq,
            )
        elif source_noise_eq:
            raise ValueError("source_noise_eq shapes the PCPH source's noise; source_type is 'sine'.")
        elif self.source_type == "sine":
            self.m_source = SineGenerator(
                self.sample_rate,
                harmonic_num=self.source_harmonics,
                noise_std=float(source_noise_std),
                harmonic_tilt=self.source_tilt,
                harmonic_phase=self.source_phase,
                phase_jitter=self.source_phase_jitter,
                random_start_phase=source_random_start_phase,
            )
        else:
            raise ValueError(f"source_type must be 'sine' or 'pcph', not {source_type!r}.")

        # Output-rate excitation -> each earlier stage's rate, last stage first.
        self.source_downs = nn.ModuleList(
            [
                FixedLowPass1d(rate, stride=rate, **SOURCE_DECIMATION)
                for rate in reversed(self.upsample_rates[1:])
            ]
        )

        if source_branch not in ("linear", "rectified"):
            raise ValueError(
                f"source_branch must be 'linear' or 'rectified', not {source_branch!r}."
            )
        self.source_branch = source_branch
        # Source channels at each stage, first stage first.
        if source_branch == "rectified":
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
            self.source_blocks = nn.ModuleList(
                [
                    weight_norm(nn.Conv1d(channels, channels * 2, 7, 1, padding=3))
                    for channels in reversed(source_channels[1:])
                ]
            )
        else:
            source_channels = [1] * count

        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
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
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)
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

        self.noise_branch = (
            NoiseBranch(num_mels, noise_branch_bands, self.sample_rate, self.upp)
            if int(noise_branch_bands) > 0
            else None
        )

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
            self.source_convs.append(
                source_conv(
                    source_channels[stage],
                    new_channels,
                    deep=stage >= count - deep_source_stages,
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

    def _fp32_region(self, x: torch.Tensor):
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
        f0 = expand_f0(f0, z.shape[-1] * self.upp)
        with self._fp32_region(z):
            gain = self._source_gain(self._fp32(z), None if g is None else self._fp32(g))
            source = self.m_source(f0.transpose(1, 2), gain).transpose(1, 2)
            if band_gains is not None:
                source = self.noise_branch.shape(source[:, 0], band_gains).unsqueeze(1)
            if self.source_branch == "rectified":
                sources = [self.source_act(self.source_pre(source))]
                for down, block in zip(self.source_downs, self.source_blocks):
                    sources.append(block(down(sources[-1])))
            else:
                sources = [source]
                for down in self.source_downs:
                    sources.append(down(sources[-1]))
        return sources[::-1]

    def _upsample(self, stage: int, x: torch.Tensor) -> torch.Tensor:
        x = self.projections[stage](x)
        with self._fp32_region(x):
            return self.ups[stage](self._fp32(x))

    def _add_source(self, stage: int, source: torch.Tensor) -> torch.Tensor:
        # A deep source filters in its rectifiers, and the filters stay in FP32.
        if stage < self.deep_source_from:
            return self.source_convs[stage](source)
        with self._fp32_region(source):
            return self.source_convs[stage](self._fp32(source))

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        noise_gains = excitation_gains = None
        if self.noise_branch is not None:
            noise_gains, excitation_gains = self.noise_branch.gains(x)
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
            x = x + self._add_source(stage, sources[stage])

            blocks = self.resblocks[
                stage * self.num_kernels : (stage + 1) * self.num_kernels
            ]
            xs = None
            for block in blocks:
                y = checkpoint(block, x, use_reentrant=False) if checkpointed else block(x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels

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

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
