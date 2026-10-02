import math
from contextlib import nullcontext
from typing import Callable, Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

from rvc.lib.algorithm.content_bottleneck import ContentBottleneck

#: Centre and spread of log f0, so the normalised pitch sits roughly in [-2, 2].
LOG_F0_CENTER = math.log(200.0)
LOG_F0_SCALE = 0.7
#: "mean" takes each step with the mean velocity over it, which only a model
#: trained with mean flow has.
SAMPLERS = ("euler", "heun", "mean")
#: Share of the frames that get the second time under ``dual_timestep``.
DUAL_TIMESTEP_SHARE = 0.25


def adaptive_weight(error: torch.Tensor) -> torch.Tensor:
    """MeanFlow's per-item loss weight, 1 / (error + c): each item pulls alike
    whatever its error, which a bootstrapped target far off would otherwise
    dominate."""
    return (error.detach() + 1e-3).reciprocal()
#: Step spacing over the sampled time range: even; sway (F5-TTS), denser near
#: the start; logit-normal, the density training draws its times from.
SCHEDULES = ("uniform", "sway", "logit-normal")
#: Where guidance rescale measures the output's spread.
RESCALE_MODES = ("global", "frame")


def time_grid(schedule: str, steps: int, start: float, device) -> torch.Tensor:
    """``steps + 1`` times from ``start`` to 1 under ``schedule``."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}.")
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "sway":
        # F5-TTS's sway with coefficient -1.
        g = 1.0 - torch.cos(0.5 * math.pi * u)
    elif schedule == "logit-normal":
        g = torch.sigmoid(math.sqrt(2.0) * torch.erfinv((2.0 * u - 1.0).clamp(-1.0, 1.0)))
    else:
        g = u
    g[0], g[-1] = 0.0, 1.0
    return start + (1.0 - start) * g


def pitch_features(f0: torch.Tensor, fourier: int = 0) -> torch.Tensor:
    """``f0`` [batch, frames] in Hz -> [batch, 2 + 2 * fourier, frames]:
    normalised log f0 and the voiced flag, then sines and cosines of the log
    f0 at ``fourier`` octave-spaced frequencies, all 0 where unvoiced. The
    finest of 6 resolves under a semitone, which a linear map of one scalar
    does not."""
    voiced = (f0 > 0).float()
    log_f0 = (torch.log(f0.clamp_min(1.0)) - LOG_F0_CENTER) / LOG_F0_SCALE
    features = [log_f0 * voiced, voiced]
    for index in range(fourier):
        angle = (2.0**index * math.pi) * log_f0
        features += [torch.sin(angle) * voiced, torch.cos(angle) * voiced]
    return torch.stack(features, dim=1)


class HarmonicPrior(nn.Module):
    """Where f0's harmonics fall in the mel, [batch, n_mels, frames].

    Each bin is the share of its filter covered by a flat harmonic comb, so a
    resolved harmonic reads near 1 and the gaps near 0, while bins wider than
    the harmonic spacing read the comb's mean density. 0 where unvoiced.
    """

    def __init__(self, sample_rate, n_fft, n_mels, fmin, fmax):
        super().__init__()
        basis = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        ).float()
        basis = basis / basis.sum(1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "freqs", torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).float(), persistent=False
        )
        # One STFT bin: half the Hann window's main lobe.
        self.sigma = sample_rate / n_fft

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        f0 = f0.float()
        ratio = self.freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
        nearest = ratio.round()
        distance = (ratio - nearest).abs() * f0[:, None, :]
        comb = torch.exp(-0.5 * (distance / self.sigma).square()) * (nearest >= 1)
        with torch.autocast(f0.device.type, enabled=False):
            prior = torch.matmul(self.basis, comb)
        return prior * (f0 > 0).float()[:, None, :]


class ConvNeXtBlock(nn.Module):
    """``layer_scale`` > 0 starts the branch that small, as DiffSinger's aux
    decoder does."""

    def __init__(self, channels: int, layer_scale: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale)) if layer_scale > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x * mask).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y))))
        if self.gamma is not None:
            y = y * self.gamma
        return (x + self.dropout(y.transpose(1, 2))) * mask


class ConditionEncoder(nn.Module):
    """Content, pitch, loudness, breathiness, tension, key shift and speaker
    -> per-frame conditioning.

    Speaker row ``speaker_count`` is the null speaker used for classifier-free
    guidance. ``key_shift`` is the formant shift in semitones: training shifts
    the mel's whole spectrum with the pitch, so 0 keeps the voice's own
    formants at any pitch. ``speed`` is the time stretch training applied, 1
    at inference.
    """

    def __init__(
        self,
        content_channels: int,
        hidden_channels: int,
        speaker_count: int,
        speaker_channels: int,
        layers: int,
        content_bottleneck: int = 0,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[HarmonicPrior] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        content_bottleneck_noise: float = 0.0,
        speed: bool = False,
        tension: bool = False,
    ):
        super().__init__()
        self.speaker_count = int(speaker_count)
        self.bottleneck = (
            ContentBottleneck(content_channels, content_bottleneck, content_bottleneck_noise)
            if content_bottleneck > 0
            else None
        )
        self.content = nn.Linear(content_channels, hidden_channels)
        self.pitch_fourier = int(pitch_fourier)
        self.pitch = nn.Conv1d(2 + 2 * self.pitch_fourier, hidden_channels, 3, padding=1)
        self.harmonic_prior = harmonic_prior
        if harmonic_prior is not None:
            self.harmonics = nn.Conv1d(harmonic_prior.basis.shape[0], hidden_channels, 1)
        self.energy = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.breathiness = nn.Conv1d(1, hidden_channels, 3, padding=1) if breathiness else None
        self.tension = None
        if tension:
            # From zero, so a checkpoint without the input starts unchanged.
            self.tension = nn.Conv1d(1, hidden_channels, 3, padding=1)
            nn.init.zeros_(self.tension.weight)
            nn.init.zeros_(self.tension.bias)
        self.key_shift = nn.Linear(1, hidden_channels) if key_shift else None
        self.speed = nn.Linear(1, hidden_channels) if speed else None
        self.speaker = nn.Embedding(self.speaker_count + 1, speaker_channels)
        self.speaker_proj = nn.Linear(speaker_channels, hidden_channels)
        self.blocks = nn.ModuleList([ConvNeXtBlock(hidden_channels) for _ in range(layers)])

    def voice(self, speaker: torch.Tensor) -> torch.Tensor:
        """The speaker's embedding, [B, hidden]; the null row for guidance."""
        return self.speaker_proj(self.speaker(speaker))

    def forward(self, content, f0, energy, speaker, mask, breathiness=None, key_shift=None, speed=None,
                tension=None):
        """``content`` [B, T, C], ``f0``, ``energy``, ``breathiness`` and
        ``tension`` [B, T], ``speaker``, ``key_shift`` and ``speed`` [B],
        ``mask`` [B, 1, T] -> [B, hidden, T]. Missing ``breathiness``,
        ``tension``, ``key_shift`` and ``speed`` read as fully aperiodic, 0, 0
        and 1."""
        if self.bottleneck is not None:
            content = self.bottleneck(content)
        x = self.content(content).transpose(1, 2)
        x = x + self.pitch(pitch_features(f0, self.pitch_fourier))
        if self.harmonic_prior is not None:
            x = x + self.harmonics(self.harmonic_prior(f0))
        x = x + self.energy(energy.unsqueeze(1))
        if self.breathiness is not None:
            if breathiness is None:
                breathiness = torch.ones_like(energy)
            x = x + self.breathiness(breathiness.unsqueeze(1))
        if self.tension is not None:
            if tension is None:
                tension = torch.zeros_like(energy)
            x = x + self.tension(tension.unsqueeze(1))
        if self.key_shift is not None:
            if key_shift is None:
                key_shift = torch.zeros(content.shape[0], device=content.device)
            x = x + self.key_shift(key_shift.float().view(-1, 1) / 12.0).unsqueeze(-1)
        if self.speed is not None:
            if speed is None:
                speed = torch.ones(content.shape[0], device=content.device)
            x = x + self.speed(speed.float().view(-1, 1)).unsqueeze(-1)
        x = x + self.voice(speaker).unsqueeze(-1)
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        return x


def timestep_embedding(t: torch.Tensor, channels: int, scale: float = 1000.0) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = (t.float() * scale)[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _ATanGLU(torch.autograd.Function):
    """``out * atan(gate)``, keeping two tensors for backward instead of three."""

    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        ctx.save_for_backward(out / gate.square().add(1.0), atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad):
        decay_out, atan_gate = ctx.saved_tensors
        return grad * atan_gate, grad * decay_out


def atan_glu(x: torch.Tensor, fused: bool = True) -> torch.Tensor:
    """``fused`` off takes the plain product, which forward-mode
    differentiation needs."""
    out, gate = x.chunk(2, dim=-1)
    if fused and torch.is_grad_enabled():
        return _ATanGLU.apply(out, gate)
    return out * torch.atan(gate)


class LYNXNet2Block(nn.Module):
    """Depthwise conv, then two ATanGLU projections, pre-norm and residual.

    With ``adaln``, the global embedding (time and speaker) shifts and scales
    the norm and gates the residual (DiT's adaLN-Zero); zero-initialised, so
    the block starts as the plain one.
    """

    def __init__(self, channels, expansion, kernel_size, adaln=False):
        super().__init__()
        inner = int(channels * expansion)
        self.norm = nn.LayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, mask, embedding=None, fused=True):
        """``x`` [B, T, C], ``mask`` [B, T, 1], ``embedding`` [B, 1 or T, C];
        ``fused`` is ``atan_glu``'s."""
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            # Not ``1 + scale``: in BF16 that rounds modulations under ~0.004 to nothing.
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y), fused)), fused))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    """DiffSinger's current LYNXNet2: condition and time added once at the
    input, then depthwise-separable gated blocks. ``adaln`` also modulates
    every block by time and speaker, as RIFT-SVC's DiT does. ``span`` adds
    MeanFlow's second time input, the length of the step whose mean velocity
    is predicted.

    ``time_scale`` multiplies the flow time before its sinusoids: 1000 is
    DiffSinger's; 1, as SiT has it, keeps the network smooth in time, which
    mean flow needs because its target is the network's own time derivative."""

    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 adaln=False, span=False, time_scale=1000.0):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.span_mlp = None
        if span:
            # No biases, on features that vanish at 0: a zero span adds exactly
            # nothing, so the instantaneous velocity is the network without it.
            self.span_mlp = nn.Sequential(
                nn.Linear(channels, channels * 4, bias=False), nn.GELU(),
                nn.Linear(channels * 4, channels, bias=False),
            )
            nn.init.zeros_(self.span_mlp[-1].weight)
        self.layers = nn.ModuleList(
            [LYNXNet2Block(channels, expansion, kernel_size, adaln) for _ in range(layers)]
        )
        self.voice = nn.Linear(cond_channels, channels) if adaln else None
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, cond, mask, voice=None, span=None):
        """``t`` is [B], or [B, T] for a time per frame. ``span`` [B] is the
        length of the step from ``t`` whose mean velocity is wanted; None is
        the velocity at ``t``."""
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels, self.time_scale))
        time = time.view(t.shape[0], -1, self.channels)
        if span is not None:
            features = timestep_embedding(span.reshape(-1), self.channels, self.time_scale)
            half = self.channels // 2
            features = torch.cat((features[:, :half], 1.0 - features[:, half:]), dim=-1)
            time = time + self.span_mlp(features).view(span.shape[0], -1, self.channels)
        frame_mask = mask.transpose(1, 2)
        # Full precision in, which also keeps the residual stream in FP32: at
        # late t the leftover noise is smaller than BF16's step on x_t.
        with torch.autocast(x.device.type, enabled=False):
            h = self.input(x.transpose(1, 2).to(self.input.weight.dtype))
        h = h + self.input_cond(cond).transpose(1, 2) + time
        h = h * frame_mask
        embedding = None
        if self.voice is not None:
            embedding = time + self.voice(voice)[:, None, :]
        for layer in self.layers:
            h = layer(h, frame_mask, embedding, fused=span is None)
        if span is None:
            h = self.norm(h)
        else:
            # The affine applied apart: Dynamo cannot trace the forward-mode
            # derivative of a layer norm with weights.
            h = F.layer_norm(h, (self.channels,), eps=self.norm.eps) * self.norm.weight + self.norm.bias
        return (self.output(h) * frame_mask).transpose(1, 2)


class AuxDecoder(nn.Module):
    """A deterministic mel from the conditioning, where shallow sampling
    starts. DiffSinger's ConvNeXt aux decoder."""

    def __init__(self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(channels, layer_scale=1e-6, dropout=dropout) for _ in range(layers)]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x) * mask


class RectifiedFlow(nn.Module):
    """Velocity field from Gaussian noise (t = 0) to the normalised log mel
    (t = 1), conditioned per frame.

    With an ``aux_decoder`` the flow is shallow, as in DiffSinger: it is
    trained on ``t >= t_start`` only, and sampling starts at ``t_start`` from
    the aux decoder's mel mixed with noise, so fewer steps cover the rest.

    With ``mean_flow`` the backbone also learns the mean velocity over a step
    (MeanFlow), which the "mean" sampler takes in one or two steps; the
    velocity at a time is still the same network at a zero span.
    """

    def __init__(
        self,
        n_mels: int,
        speaker_count: int,
        content_channels: int = 768,
        hidden_channels: int = 384,
        encoder_layers: int = 4,
        content_bottleneck: int = 0,
        content_bottleneck_noise: float = 0.0,
        speaker_channels: int = 256,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[dict] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        speed: bool = False,
        backbone: str = "lynxnet2",
        backbone_args: Optional[dict] = None,
        aux_decoder: Optional[dict] = None,
        t_start: float = 0.0,
        aux_grad: float = 0.1,
        tension: bool = False,
        dual_timestep: bool = False,
        mean_flow: bool = False,
    ):
        """``harmonic_prior`` is the mel's ``sample_rate``, ``n_fft``, ``fmin``
        and ``fmax``, or None for no prior. ``aux_decoder`` is its
        ``channels`` and ``layers``, or None for a flow from pure noise;
        ``aux_grad`` scales its gradient into the encoder. ``backbone`` is kept
        because configs and exports name it; LYNXNet2 is the only one.
        ``dual_timestep`` trains a share of each item's frames at a second
        time, as DiffSinger does."""
        super().__init__()
        if backbone != "lynxnet2":
            raise ValueError(f"Only the lynxnet2 backbone is supported, not {backbone!r}.")
        self.n_mels = int(n_mels)
        self.hidden_channels = int(hidden_channels)
        self.encoder = ConditionEncoder(
            content_channels,
            hidden_channels,
            speaker_count,
            speaker_channels,
            encoder_layers,
            content_bottleneck,
            pitch_fourier,
            HarmonicPrior(n_mels=n_mels, **harmonic_prior) if harmonic_prior else None,
            breathiness,
            key_shift,
            content_bottleneck_noise,
            speed,
            tension,
        )
        self.backbone = LYNXNet2Backbone(
            n_mels, hidden_channels, span=bool(mean_flow), **(backbone_args or {})
        )
        # Time and speaker reach the network through these; on AdamW, where
        # gradient clipping bounds the step, since Muon's step ignores the
        # gradient's size.
        conditioning = [self.backbone.time_mlp, self.backbone.span_mlp, self.encoder.speaker_proj,
                        self.backbone.voice]
        conditioning += [layer.modulation for layer in self.backbone.layers]
        for module in filter(None, conditioning):
            for child in module.modules():
                child.use_adamw = True
        self.aux = AuxDecoder(hidden_channels, n_mels, **aux_decoder) if aux_decoder else None
        self.t_start = float(t_start) if self.aux is not None else 0.0
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)

    @property
    def speaker_count(self) -> int:
        return self.encoder.speaker_count

    def _drop_speakers(self, speaker, speaker_dropout):
        if speaker_dropout <= 0:
            return speaker
        dropped = torch.rand(speaker.shape, device=speaker.device) < speaker_dropout
        return torch.where(dropped, torch.full_like(speaker, self.speaker_count), speaker)

    def _flow_error(self, mel, cond, voice, mask, t, noise, backbone):
        """Flow loss per item [B]; ``t`` is [B], or [B, T] for a time per frame."""
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        prediction = backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    def _aux_loss(self, mel, cond, mask):
        """The aux decoder's L1, or None without one."""
        if self.aux is None:
            return None
        aux_cond = cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad)
        error = (self.aux(aux_cond, mask).float() - mel.float()).abs() * mask
        return error.sum() / (mask.sum() * self.n_mels).clamp_min(1.0)

    def mean_velocity(self, x, t, span, velocity, cond, mask, voice):
        """The mean velocity over ``span`` from ``t``, and its derivative along
        the path ``velocity`` with the step's end held, detached."""

        def field(x, t, span):
            return self.backbone(x, t, cond, mask, voice, span)

        tangents = (velocity, torch.ones_like(t), -torch.ones_like(span))
        mean, derivative = torch.func.jvp(field, (x, t, span), tangents)
        # Detached here: a compiled graph cannot take the empty gradient the
        # derivative would otherwise get.
        return mean, derivative.detach()

    def _mean_error(self, mel, cond, voice, mask, noise, mean_field, bootstrap=1.0):
        """MeanFlow loss per item [B], over steps between two drawn times, and
        how large the bootstrapped part of the target is against the velocity,
        [B]. ``bootstrap`` scales that part: 0 is plain flow matching."""
        first, second = (self._times(mel.shape[0], mel.device) for _ in range(2))
        t, span = torch.minimum(first, second), (first - second).abs()
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * mel
        velocity = mel - noise
        # The time derivative passes FP16's range, so it is taken in FP32 there.
        kind = mel.device.type
        fp16 = torch.is_autocast_enabled(kind) and torch.get_autocast_dtype(kind) == torch.float16
        if fp16:
            cond, voice = cond.float(), voice.float()
        with torch.autocast(kind, enabled=False) if fp16 else nullcontext():
            mean, derivative = mean_field(x_t, t, span, velocity, cond, mask, voice)
        # The mean over [t, t + span] is the velocity at t plus span times its
        # own derivative in t; the network's derivative stands in as a target.
        correction = span[:, None, None] * derivative.float() * mask
        count = (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)
        size = (correction.square().sum((1, 2)) / count).sqrt()
        ratio = size / ((velocity.square() * mask).sum((1, 2)) / count).sqrt().clamp_min(1e-8)
        # A path's curvature keeps it well under the velocity; past that the
        # target is feeding on itself, and is held there.
        correction = correction * (1.0 / ratio.clamp_min(1.0))[:, None, None]
        target = velocity + bootstrap * correction
        error = (mean.float() - target).square() * mask
        return error.sum((1, 2)) / count, ratio

    def _times(self, batch, device):
        """Logit-normal times, stratified across the batch so every batch spans
        the distribution, over the trained range. Logit-normal puts more steps
        mid-trajectory, where the velocity is hardest to predict."""
        u = (torch.arange(batch, device=device) + torch.rand(batch, device=device)) / batch
        u = u[torch.randperm(batch, device=device)].clamp(1e-6, 1.0 - 1e-6)
        t = torch.sigmoid(math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0))
        return self.t_start + (1.0 - self.t_start) * t

    def forward(self, mel, content, f0, energy, speaker, mask, speaker_dropout=0.0,
                breathiness=None, key_shift=None, speed=None, backbone=None, tension=None,
                mean_ratio=0.0, mean_field=None, tension_dropout=0.0, mean_bootstrap=1.0):
        """Flow-matching loss for normalised ``mel`` [B, n_mels, T], the aux
        decoder's L1 (None without one) and the MeanFlow losses (None when
        off). ``backbone`` stands in for ``self.backbone``, as its compiled
        wrapper does, and ``mean_field`` for ``self.mean_velocity``.

        The first ``mean_ratio`` of the batch trains the mean velocity instead
        of the flow. Every item's loss then takes ``adaptive_weight``, as in
        MeanFlow, and the third value is [mean loss to optimise, plain flow
        loss, plain mean loss, bootstrap ratio], the last three to read.
        ``mean_bootstrap`` is ``_mean_error``'s, ramped up from 0 early in
        training.

        ``tension_dropout`` is the share of items trained with a flat tension,
        so the input can be turned down at inference."""
        speaker = self._drop_speakers(speaker, speaker_dropout)
        if tension is not None and tension_dropout > 0:
            kept = torch.rand(tension.shape[0], 1, device=tension.device) >= tension_dropout
            tension = tension * kept
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, tension)
        voice = self.encoder.voice(speaker)
        noise = torch.randn_like(mel)
        batch = mel.shape[0]
        mean_items = 0
        if self.backbone.span_mlp is not None:
            mean_items = min(batch - 1, int(round(mean_ratio * batch)))
        rest = slice(mean_items, None)
        t = self._times(batch - mean_items, mel.device)
        if self.dual_timestep:
            other = self._times(batch - mean_items, mel.device)
            swap = torch.rand(batch - mean_items, mel.shape[-1], device=mel.device) < DUAL_TIMESTEP_SHARE
            t = torch.where(swap, other[:, None], t[:, None])
        # Not ``backbone or ...``: a compiled module's truth test calls len().
        backbone = self.backbone if backbone is None else backbone
        error = self._flow_error(mel[rest], cond[rest], voice[rest], mask[rest], t, noise[rest], backbone)

        def pooled(error, frames, weighted=False):
            weight = adaptive_weight(error) if weighted else 1.0
            return (weight * error * frames).sum() / frames.sum().clamp_min(1.0)

        frames = mask[rest].sum((1, 2))
        flow = pooled(error, frames)
        mean = None
        if mean_items:
            head = slice(0, mean_items)
            mean_field = self.mean_velocity if mean_field is None else mean_field
            mean_error, bootstrap_ratio = self._mean_error(
                mel[head], cond[head], voice[head], mask[head], noise[head], mean_field, mean_bootstrap
            )
            mean_frames = mask[head].sum((1, 2))
            mean = torch.stack((
                pooled(mean_error, mean_frames, True), flow.detach(),
                pooled(mean_error, mean_frames).detach(), bootstrap_ratio.mean(),
            ))
            flow = pooled(error, frames, True)
        return flow, self._aux_loss(mel, cond, mask), mean

    @torch.no_grad()
    def validation_losses(self, mel, content, f0, energy, speaker, mask, breathiness, key_shift,
                          speed, noise, fractions, tension=None):
        """Flow loss at each of ``fractions`` of the trained time range, [len],
        from fixed ``noise``, and the aux decoder's L1 (None without one)."""
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, tension)
        voice = self.encoder.voice(speaker)
        frames = mask.sum((1, 2))
        losses = []
        for fraction in fractions:
            t = torch.full((mel.shape[0],), self.t_start + (1.0 - self.t_start) * fraction, device=mel.device)
            error = self._flow_error(mel, cond, voice, mask, t, noise, self.backbone)
            losses.append((error * frames).sum() / frames.sum().clamp_min(1.0))
        return torch.stack(losses), self._aux_loss(mel, cond, mask)

    @torch.no_grad()
    def sample(
        self,
        content,
        f0,
        energy,
        speaker,
        mask,
        steps: int = 16,
        method: str = "euler",
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[], None]] = None,
        breathiness: Optional[torch.Tensor] = None,
        key_shift: Optional[torch.Tensor] = None,
        content_guidance: float = 0.0,
        guidance_rescale: float = 0.0,
        temperature: float = 1.0,
        start: Optional[float] = None,
        guidance_interval: tuple = (0.0, 1.0),
        rescale_mode: str = "global",
        schedule: str = "uniform",
        churn: float = 0.0,
        churn_noise: Optional[Callable[[int], torch.Tensor]] = None,
        tension: Optional[torch.Tensor] = None,
    ):
        """Integrate the ODE from ``noise`` (drawn when None) to a normalised
        mel, [B, n_mels, T]; from ``t_start`` on the aux decoder's mel when the
        flow is shallow. ``callback`` is called after every step.

        Guidance, as in RIFT-SVC: ``cfg_scale`` pushes away from the null
        speaker (1 is off), ``content_guidance`` away from the same content
        blurred to a quarter of its frame rate (0 is off), which sharpens the
        articulation. ``guidance_rescale`` pulls the guided velocity's spread
        back toward the unguided one's, against oversaturation, over the whole
        input (``rescale_mode`` "global") or per frame ("frame").
        ``guidance_interval`` is the ``[from, until)`` flow time both guidances
        apply in (closed at 1); outside it only the plain pass runs.

        ``temperature`` scales the starting noise. ``start`` is the flow time
        sampling begins at, from the aux decoder's mel; None, or anything
        before ``t_start``, is ``t_start``, and it is ignored without an aux
        decoder. ``schedule`` is one of ``SCHEDULES``. The "mean" ``method``
        is meant for one or two steps; the guidances apply to it as they are,
        though it was not trained under them.

        ``churn`` makes the sampler stochastic, as EDM's: before each step the
        state is re-noised back ``churn`` times that step's length (never
        before the trained ``t_start``), and the step then covers the longer
        span; 0 is the plain ODE. ``churn_noise(step)`` gives that step's fresh
        noise, drawn when None.
        """
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if method == "mean" and self.backbone.span_mlp is None:
            raise ValueError("The mean sampler needs a model trained with mean flow.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}.")
        batch = content.shape[0]
        null = torch.full_like(speaker, self.speaker_count)
        # Each guided variant rides along in the batch: (content, speaker) pairs.
        variants = [(content, speaker)]
        if cfg_scale != 1.0:
            variants.append((content, null))
        if content_guidance > 0:
            frames = content.shape[1]
            blurred = F.interpolate(content.transpose(1, 2), size=max(1, frames // 4), mode="linear")
            blurred = F.interpolate(blurred, size=frames, mode="linear").transpose(1, 2)
            variants.append((blurred, speaker))
        count = len(variants)

        def repeat(value):
            return None if value is None else value.repeat(count, *([1] * (value.dim() - 1)))

        cond = self.encoder(
            torch.cat([c for c, _ in variants]), repeat(f0), repeat(energy),
            torch.cat([s for _, s in variants]), repeat(mask), repeat(breathiness), repeat(key_shift),
            tension=repeat(tension),
        )
        voice = self.encoder.voice(torch.cat([s for _, s in variants]))
        masks = repeat(mask)
        guide_from, guide_until = (float(value) for value in guidance_interval)

        def spread(y):
            if rescale_mode == "frame":
                return y.square().mean(1, keepdim=True).sqrt()
            frames = mask.sum((1, 2)).clamp_min(1.0) * self.n_mels
            return ((y.square() * mask).sum((1, 2)) / frames).sqrt()[:, None, None]

        def field(x, t, span=None):
            now = float(t[0])
            # An interval reaching 1 includes it, where Heun's last evaluation lands.
            if count == 1 or not (guide_from <= now and (now < guide_until or guide_until >= 1.0)):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], span)
            v = self.backbone(repeat(x), repeat(t), cond, masks, voice, repeat(span)).chunk(count)
            guided, index = v[0], 1
            if cfg_scale != 1.0:
                guided = guided + (cfg_scale - 1.0) * (v[0] - v[index])
                index += 1
            if content_guidance > 0:
                guided = guided + content_guidance * (v[0] - v[index])
            if guidance_rescale > 0:
                rescaled = guided * spread(v[0]) / spread(guided).clamp_min(1e-6)
                guided = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
            return guided

        shape = (batch, self.n_mels, content.shape[1])
        if noise is None:
            noise = torch.randn(shape, device=content.device)
        noise = noise * float(temperature)
        t0 = 0.0
        if self.t_start > 0:
            t0 = self.t_start if start is None else min(max(self.t_start, float(start)), 0.99)
            x = ((1.0 - t0) * noise + t0 * self.aux(cond[:batch], mask)) * mask
        else:
            x = noise * mask
        times = time_grid(schedule, max(1, int(steps)), t0, x.device)
        for index in range(times.shape[0] - 1):
            now = float(times[index])
            back = max(self.t_start, now - float(churn) * float(times[index + 1] - times[index]))
            if churn > 0 and 0 < back < now:
                # Keeps x on the path (1 - t) noise + t mel: the kept noise shrinks
                # with the data and fresh noise tops it up to (1 - back).
                fresh = torch.randn_like(x) if churn_noise is None else churn_noise(index)
                scale = back / now
                top_up = math.sqrt(max((1.0 - back) ** 2 - (scale * (1.0 - now)) ** 2, 0.0))
                x = (scale * x + float(temperature) * top_up * fresh) * mask
                now = back
            t = torch.full((batch,), now, device=x.device, dtype=times.dtype)
            dt = times[index + 1] - now
            v = field(x, t, dt.expand(batch) if method == "mean" else None)
            if method == "heun":
                v_next = field(x + dt * v, times[index + 1].expand(batch))
                v = 0.5 * (v + v_next)
            x = x + dt * v
            if callback is not None:
                callback()
        return x * mask


def resize_speakers(state_dict: dict, speaker_count: int) -> dict:
    """Fit a checkpoint's speaker table to ``speaker_count`` speakers.

    A fine-tune's speakers are not the pretrain's, so every row starts at the
    mean of the trained speakers; the null row is kept.
    """
    key = "encoder.speaker.weight"
    table = state_dict[key]
    if table.shape[0] == speaker_count + 1:
        return state_dict
    trained, null = table[:-1], table[-1:]
    rows = trained.mean(0, keepdim=True).expand(speaker_count, -1).clone()
    state_dict = dict(state_dict)
    state_dict[key] = torch.cat((rows, null), dim=0)
    return state_dict


#: Inputs that start at zero, so a checkpoint from before them loads unchanged.
ZERO_INPUTS = ("encoder.tension.", "backbone.span_mlp.")


def match_inputs(state_dict: dict, model: RectifiedFlow) -> dict:
    """Fit a pretrain's weights to ``model``'s optional inputs: the ones it
    lacks keep ``model``'s zero start, and a span input ``model`` does not
    have is dropped, which leaves the plain flow."""
    own = model.state_dict()
    state_dict = {
        key: value for key, value in state_dict.items()
        if key in own or not key.startswith("backbone.span_mlp.")
    }
    for key, value in own.items():
        if key not in state_dict and key.startswith(ZERO_INPUTS):
            state_dict[key] = value
    return state_dict


def build_flow(config: dict, speaker_count: int) -> RectifiedFlow:
    model = dict(config["flow"]["model"])
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"], n_fft=data["n_fft"],
            fmin=data["mel_fmin"], fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
