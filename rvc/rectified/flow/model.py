"""The rectified flow: its losses, its sampler and how its weights load."""

import math
from typing import Callable, Optional

import torch
from torch import nn

from rvc.rectified.flow.backbone import AuxDecoder, LYNXNet2Backbone
from rvc.rectified.flow.conditioning import ConditionEncoder, Conditioning, HarmonicPrior
from rvc.rectified.flow.sampling import RESCALE_MODES, SAMPLERS, Guidance, time_grid

#: Share of the frames that get the second time under ``dual_timestep``.
DUAL_TIMESTEP_SHARE = 0.25
#: Inputs that start at zero, so a checkpoint from before them loads unchanged.
ZERO_INPUTS = ("encoder.tension.", "backbone.step.")
#: What a model without the input leaves out of the weights it loads.
DROPPED_INPUTS = ("backbone.span_mlp.", "backbone.step.", "encoder.phonation.")
#: The per-bin mel statistics; a checkpoint from before them keeps the identity.
MEL_STATS = ("mel_shift", "mel_scale")
#: Floor of a bin's spread, in the data's normalisation, so a bin that hardly
#: moves is not blown up into noise.
MIN_MEL_SCALE = 0.1


class RectifiedFlow(nn.Module):
    """Velocity field from Gaussian noise (t = 0) to the normalised log mel
    (t = 1), conditioned per frame.

    With an ``aux_decoder`` and ``t_start`` > 0 it is DiffSinger's shallow
    flow: trained on ``t >= t_start`` only, and sampling starts there from the
    aux decoder's mel mixed with noise.

    The mel goes in and out in the data's normalisation; inside, each bin is
    normalised again by ``set_mel_stats``' statistics.
    """

    def __init__(
        self,
        n_mels: int,
        speaker_count: int,
        content_channels: int = 768,
        hidden_channels: int = 384,
        encoder_layers: int = 4,
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
        shortcut: bool = False,
        shortcut_steps: int = 128,
    ):
        """``harmonic_prior`` is the mel's ``sample_rate``, ``n_fft``, ``fmin``
        and ``fmax``, or None for no prior. ``aux_decoder`` is ``AuxDecoder``'s
        arguments, or None for a flow from pure noise; ``aux_grad`` scales its
        gradient into the encoder and the speaker table. ``backbone`` is kept
        because configs and exports name it; LYNXNet2 is the only one.
        ``dual_timestep`` trains a share of each item's frames at a second
        time, as DiffSinger does. ``shortcut`` gives the backbone a jump length to
        read (Frans et al., "One Step Diffusion via Shortcut Models"), down to
        one of ``shortcut_steps`` of the trained range, a power of two."""
        super().__init__()
        if shortcut and (shortcut_steps < 2 or shortcut_steps & (shortcut_steps - 1)):
            raise ValueError(f"shortcut_steps must be a power of two of 2 or more, not {shortcut_steps}.")
        #: Jump lengths the backbone knows, 0 for a plain flow.
        self.shortcut_levels = int(math.log2(shortcut_steps)) if shortcut else 0
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
            pitch_fourier,
            HarmonicPrior(n_mels=n_mels, **harmonic_prior) if harmonic_prior else None,
            breathiness,
            key_shift,
            speed,
            tension,
        )
        self.backbone = LYNXNet2Backbone(
            n_mels, hidden_channels, step_levels=self.shortcut_levels, **(backbone_args or {})
        )
        self.aux = AuxDecoder(hidden_channels, n_mels, **aux_decoder) if aux_decoder else None
        # Time and speaker reach the network through these; on AdamW, where
        # gradient clipping bounds the step, since Muon's step ignores the
        # gradient's size.
        conditioning = [self.backbone.time_mlp, *self.speaker_layers()]
        for module in filter(None, conditioning):
            for child in module.modules():
                child.use_adamw = True
        self.t_start = float(t_start) if self.aux is not None else 0.0
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)
        #: Draws training's noise, times and dropouts when set, so they do
        #: not depend on what else draws from the global generator.
        self.generator = None
        self.register_buffer("mel_shift", torch.zeros(self.n_mels, 1))
        self.register_buffer("mel_scale", torch.ones(self.n_mels, 1))

    @property
    def speaker_count(self) -> int:
        return self.encoder.speaker_count

    @property
    def starts_from_aux(self) -> bool:
        """Whether sampling starts from the aux decoder's mel."""
        return self.t_start > 0

    @torch.no_grad()
    def set_mel_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Normalise each mel bin inside the model by its ``mean`` and ``std``
        [n_mels], taken over the training set in the data's normalisation."""
        self.mel_shift.copy_(mean.view(-1, 1))
        self.mel_scale.copy_(std.view(-1, 1).clamp_min(MIN_MEL_SCALE))

    def _encode(self, mel):
        return (mel - self.mel_shift) / self.mel_scale

    def _decode(self, mel):
        return mel * self.mel_scale + self.mel_shift

    def speaker_layers(self) -> list:
        """The layers the speaker's embedding reaches the network through,
        None for one the model lacks. The adaLN ones take the time as well."""
        layers = [self.encoder.speaker_proj, self.backbone.voice]
        layers += [layer.modulation for layer in self.backbone.layers]
        if self.aux is not None:
            layers += [block.speaker for block in self.aux.blocks]
        return layers

    def _drop_speakers(self, speaker, speaker_dropout):
        if speaker_dropout <= 0:
            return speaker
        dropped = torch.rand(speaker.shape, device=speaker.device, generator=self.generator) < speaker_dropout
        return torch.where(dropped, torch.full_like(speaker, self.speaker_count), speaker)

    def _flow_error(self, mel, noise, cond, voice, mask, t, backbone, jump=None):
        """Flow loss per item [B] on the path from ``noise`` to ``mel``;
        ``t`` is [B], or [B, T] for a time per frame. ``jump`` is
        ``_jump_targets``'s, whose items stand in for the batch's first ones."""
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        target = mel - noise
        if jump is None:
            prediction = backbone(x_t, t, cond, mask, voice)
        else:
            count, x_jump, t_jump, level, target_jump = jump
            x_t = torch.cat((x_jump.to(x_t.dtype), x_t[count:]))
            target = torch.cat((target_jump.to(target.dtype), target[count:]))
            t = torch.cat((t_jump if t.dim() == 1 else t_jump[:, None].expand(-1, t.shape[1]), t[count:]))
            # The rest of the batch trains the plain flow, the finest level.
            level = torch.cat((level, level.new_full((mel.shape[0] - count,), self.shortcut_levels)))
            prediction = backbone(x_t, t, cond, mask, voice, level)
        error = (prediction.float() - target.float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    @torch.no_grad()
    def _jump_targets(self, teacher, mel, noise, inputs: Conditioning, share: float):
        """Self-consistency targets for the first items of the batch, a
        ``share`` of it: a jump of one of the lengths, from a time on that
        length's grid, has to land where two of ``teacher``'s half as long do.
        Returns (count, state, time, level, target), or None when the draw
        gives no item. ``teacher`` is a copy of the model that lags it."""
        device = mel.device
        count = min(mel.shape[0], int(share * mel.shape[0] + torch.rand((), device=device, generator=self.generator).item()))
        if count < 1:
            return None
        mel, noise, inputs = mel[:count], noise[:count], inputs.map(lambda value: value[:count])
        # Per item: spread over the batch instead, a small one trains few lengths.
        level = torch.randint(0, self.shortcut_levels, (count,), device=device, generator=self.generator)
        sections = 2.0**level
        span = 1.0 - self.t_start
        slot = torch.floor(torch.rand(count, device=device, generator=self.generator) * sections)
        t = self.t_start + span * slot / sections
        half = 0.5 * span / sections
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * mel
        cond, voice = teacher.encoder(inputs), teacher.encoder.voice(inputs.speaker)
        finer = (level + 1).clamp_max(self.shortcut_levels)
        first = teacher.backbone(x_t, t, cond, inputs.mask, voice, finer).float()
        second = teacher.backbone(x_t + half[:, None, None] * first, t + half, cond, inputs.mask, voice, finer).float()
        return count, x_t, t, level, 0.5 * (first + second)

    def _aux_loss(self, mel, cond, voice, mask):
        """The aux decoder's L1, or None without one."""
        if self.aux is None:
            return None
        cond, voice = (
            value * self.aux_grad + value.detach() * (1.0 - self.aux_grad) for value in (cond, voice)
        )
        error = (self.aux(cond, mask, voice).float() - mel.float()).abs() * mask
        return error.sum() / (mask.sum() * self.n_mels).clamp_min(1.0)

    def _times(self, batch, device):
        """Times over the trained range, stratified across the batch so every
        batch spans the range."""
        u = (torch.arange(batch, device=device) + torch.rand(batch, device=device, generator=self.generator)) / batch
        u = u[torch.randperm(batch, device=device, generator=self.generator)]
        return self.t_start + (1.0 - self.t_start) * u

    def _train_times(self, batch, frames, device):
        """A time per item, [B]; under ``dual_timestep`` a share of each
        item's frames gets a second one, [B, T]."""
        t = self._times(batch, device)
        if not self.dual_timestep:
            return t
        other = self._times(batch, device)
        swap = torch.rand(batch, frames, device=device, generator=self.generator) < DUAL_TIMESTEP_SHARE
        return torch.where(swap, other[:, None], t[:, None])

    @staticmethod
    def _pooled(error, frames):
        """Per-item ``error`` averaged over the batch's frames."""
        return (error * frames).sum() / frames.sum().clamp_min(1.0)

    def forward(self, mel, inputs: Conditioning, speaker_dropout=0.0, backbone=None, tension_dropout=0.0,
                teacher=None, shortcut_share=0.0):
        """Flow-matching loss for normalised ``mel`` [B, n_mels, T] and the aux
        decoder's L1 (None without one), both on the per-bin normalised mel.
        ``backbone`` stands in for ``self.backbone``, as its compiled wrapper
        does.

        ``speaker_dropout`` is the share of items trained on the null speaker,
        and ``tension_dropout`` the share trained with a flat tension, so the
        input can be turned down at inference. With ``teacher``, ``shortcut_share`` of a shortcut model's
        batch trains the jumps against it, as ``_jump_targets`` has it."""
        speaker = self._drop_speakers(inputs.speaker, speaker_dropout)
        tension = inputs.tension
        if tension is not None and tension_dropout > 0:
            kept = torch.rand(tension.shape[0], 1, device=tension.device, generator=self.generator) >= tension_dropout
            tension = tension * kept
        inputs = inputs._replace(speaker=speaker, tension=tension)
        mask = inputs.mask
        mel = self._encode(mel) * mask
        cond = self.encoder(inputs)
        voice = self.encoder.voice(speaker)
        noise = torch.randn_like(mel) if self.generator is None else torch.randn(
            mel.shape, device=mel.device, dtype=mel.dtype, generator=self.generator
        )
        t = self._train_times(mel.shape[0], mel.shape[-1], mel.device)
        # Not ``backbone or ...``: a compiled module's truth test calls len().
        backbone = self.backbone if backbone is None else backbone
        jump = None
        if teacher is not None and self.shortcut_levels and shortcut_share > 0:
            jump = self._jump_targets(teacher, mel, noise, inputs, shortcut_share)
        error = self._flow_error(mel, noise, cond, voice, mask, t, backbone, jump)
        return self._pooled(error, mask.sum((1, 2))), self._aux_loss(mel, cond, voice, mask)

    @torch.no_grad()
    def validation_losses(self, mel, inputs: Conditioning, noise, fractions):
        """Flow loss at each of ``fractions`` of the trained time range, [len],
        from fixed ``noise``, and the aux decoder's L1 (None without one)."""
        mask = inputs.mask
        mel = self._encode(mel) * mask
        cond = self.encoder(inputs)
        voice = self.encoder.voice(inputs.speaker)
        frames = mask.sum((1, 2))
        losses = []
        for fraction in fractions:
            t = torch.full((mel.shape[0],), self.t_start + (1.0 - self.t_start) * fraction, device=mel.device)
            error = self._flow_error(mel, noise, cond, voice, mask, t, self.backbone)
            losses.append(self._pooled(error, frames))
        return torch.stack(losses), self._aux_loss(mel, cond, voice, mask)

    @torch.no_grad()
    def aux_mel(self, inputs: Conditioning) -> torch.Tensor:
        """The aux decoder's normalised mel, [B, n_mels, T], where sampling
        starts under ``t_start``."""
        mel = self.aux(self.encoder(inputs), inputs.mask, self.encoder.voice(inputs.speaker))
        return self._decode(mel) * inputs.mask

    def _start(self, noise, mel, mask, start):
        """Where sampling begins, as (state, flow time): noise at 0, or from
        the aux decoder's ``mel`` at ``start``."""
        if self.t_start <= 0:
            return noise * mask, 0.0
        t0 = self.t_start if start is None else min(max(self.t_start, float(start)), 0.99)
        return ((1.0 - t0) * noise + t0 * mel) * mask, t0

    @staticmethod
    def _renoise(x, now, back, fresh):
        """``x`` taken from the time ``now`` back to ``back`` with ``fresh``
        noise."""
        # Keeps x on the path (1 - t) noise + t mel: the kept noise shrinks
        # with the data and fresh noise tops it up to (1 - back).
        scale = back / now
        top_up = math.sqrt(max((1.0 - back) ** 2 - (scale * (1.0 - now)) ** 2, 0.0))
        return scale * x + top_up * fresh

    @torch.no_grad()
    def sample(
        self,
        inputs: Conditioning,
        steps: int = 16,
        method: str = "euler",
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[], None]] = None,
        content_guidance: float = 0.0,
        guidance_rescale: float = 0.0,
        temperature: float = 1.0,
        start: Optional[float] = None,
        guidance_interval: tuple = (0.0, 1.0),
        rescale_mode: str = "global",
        schedule: str = "uniform",
        churn: float = 0.0,
        churn_noise: Optional[Callable[[int], torch.Tensor]] = None,
        start_mel: Optional[torch.Tensor] = None,
    ):
        """Integrate the ODE from ``noise`` (drawn when None) to a normalised
        mel, [B, n_mels, T]; from the aux decoder's mel under ``t_start``.
        ``callback`` is called after every step.

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
        before the model's own ``t_start``, is that start, and it is ignored
        without an aux decoder. ``start_mel``
        [B, n_mels, T] stands in for the aux decoder's mel there, as the real
        mel does to tell the flow's error from the aux decoder's. ``schedule``
        is one of ``SCHEDULES``.

        ``churn`` makes the sampler stochastic, as EDM's: before each step the
        state is re-noised back ``churn`` times that step's length (never
        before the trained ``t_start``), and the step then covers the longer
        span; 0 is the plain ODE. ``churn_noise(step)`` gives that step's fresh
        noise, drawn when None.
        """
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}.")
        guidance = Guidance(
            cfg_scale, content_guidance, guidance_rescale, rescale_mode,
            tuple(float(value) for value in guidance_interval),
        )
        batch, frames = inputs.content.shape[:2]
        mask = inputs.mask

        # Each guided variant rides along in the batch, after the plain pass.
        variants = guidance.variants(inputs, self.speaker_count)
        count = len(variants)

        def repeat(value):
            return None if value is None else value.repeat(count, *([1] * (value.dim() - 1)))

        stacked = inputs.map(repeat)._replace(
            content=torch.cat([content for content, _ in variants]),
            speaker=torch.cat([speaker for _, speaker in variants]),
        )
        cond = self.encoder(stacked)
        voice = self.encoder.voice(stacked.speaker)

        level = None

        def velocity(x, t):
            if count == 1 or not guidance.active(float(t[0])):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], level)
            passes = self.backbone(repeat(x), repeat(t), cond, stacked.mask, voice, repeat(level))
            return guidance.combine(passes.chunk(count), mask)

        if noise is None:
            noise = torch.randn((batch, self.n_mels, frames), device=inputs.content.device)
        noise = noise * float(temperature)
        if start_mel is not None:
            start_mel = self._encode(start_mel)
        elif self.starts_from_aux:
            start_mel = self.aux(cond[:batch], mask, voice[:batch])
        x, t0 = self._start(noise, start_mel, mask, start)
        steps = max(1, int(steps))
        times = time_grid(schedule, steps, t0, x.device)
        # A shortcut model takes its trained jumps on the grid it learned them
        # on; any other sampling integrates the plain flow.
        on_grid = method == "euler" and schedule == "uniform" and churn <= 0 and t0 == self.t_start
        if on_grid and steps & (steps - 1) == 0 and steps < 2**self.shortcut_levels:
            level = torch.full((batch,), int(math.log2(steps)), device=x.device)
        for index in range(times.shape[0] - 1):
            now = float(times[index])
            back = max(self.t_start, now - float(churn) * float(times[index + 1] - times[index]))
            if churn > 0 and 0 < back < now:
                fresh = torch.randn_like(x) if churn_noise is None else churn_noise(index)
                x = self._renoise(x, now, back, float(temperature) * fresh) * mask
                now = back
            t = torch.full((batch,), now, device=x.device, dtype=times.dtype)
            dt = times[index + 1] - now
            v = velocity(x, t)
            if method == "heun":
                v_next = velocity(x + dt * v, times[index + 1].expand(batch))
                v = 0.5 * (v + v_next)
            x = x + dt * v
            if callback is not None:
                callback()
        return self._decode(x) * mask


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


def match_inputs(state_dict: dict, model: RectifiedFlow) -> dict:
    """Fit other weights to ``model``: the optional inputs and mel statistics
    they lack keep ``model``'s own, and what ``model`` has no input for (a
    mean-flow run's span, a shortcut one's jump lengths) is dropped, which
    leaves the plain flow."""
    own = model.state_dict()
    state_dict = {
        key: value for key, value in state_dict.items()
        if key in own or not key.startswith(DROPPED_INPUTS)
    }
    for key, value in own.items():
        if key not in state_dict and key.startswith(ZERO_INPUTS + MEL_STATS):
            state_dict[key] = value
    return state_dict


def build_flow(config: dict, speaker_count: int) -> RectifiedFlow:
    model = dict(config["flow"]["model"])
    # Exports and configs from when these were options name them.
    for name in ("mean_flow", "phonation", "prior_noise", "prior_degrade", "time_sampling"):
        model.pop(name, None)
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"], n_fft=data["n_fft"],
            fmin=data["mel_fmin"], fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
