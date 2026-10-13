"""The rectified flow: its losses, its sampler and how its weights load."""

import math
from contextlib import nullcontext
from typing import Callable, NamedTuple, Optional

import torch
from torch import nn

from rvc.rectified.flow.backbone import AuxDecoder, LYNXNet2Backbone
from rvc.rectified.flow.conditioning import ConditionEncoder, Conditioning, HarmonicPrior
from rvc.rectified.flow.sampling import RESCALE_MODES, SAMPLERS, Guidance, time_grid

#: Share of the frames that get the second time under ``dual_timestep``.
DUAL_TIMESTEP_SHARE = 0.25
#: Share of the mean-flow items trained at a zero span: the velocity itself
#: under a guidance scale, which the other items' targets read.
BOUNDARY_SHARE = 0.25
#: Inputs that start at zero, so a checkpoint from before them loads unchanged.
ZERO_INPUTS = ("encoder.tension.", "backbone.span_mlp.", "backbone.guide_mlp.")
#: What a model without the input leaves out of the weights it loads.
DROPPED_INPUTS = ("backbone.span_mlp.", "backbone.guide_mlp.", "backbone.step.", "encoder.phonation.")
#: The per-bin mel statistics; a checkpoint from before them keeps the identity.
MEL_STATS = ("mel_shift", "mel_scale")
#: Floor of a bin's spread, in the data's normalisation, so a bin that hardly
#: moves is not blown up into noise.
MIN_MEL_SCALE = 0.1


class MeanFlowLosses(NamedTuple):
    """The losses of the share of a batch that trains the mean velocity.
    Only ``objective`` is optimised; the rest are detached, to read."""

    #: Mean-flow loss under ``adaptive_weight``.
    objective: torch.Tensor
    #: The rest of the batch's flow loss, unweighted.
    flow: torch.Tensor
    #: Mean-flow loss, unweighted.
    mean: torch.Tensor
    #: Size of the bootstrapped part of the target against the velocity.
    bootstrap_ratio: torch.Tensor


def adaptive_weight(error: torch.Tensor) -> torch.Tensor:
    """MeanFlow's per-item loss weight, 1 / (error + c): each item pulls alike
    whatever its error, which a bootstrapped target far off would otherwise
    dominate."""
    return (error.detach() + 1e-3).reciprocal()


class RectifiedFlow(nn.Module):
    """Velocity field from Gaussian noise (t = 0) to the normalised log mel
    (t = 1), conditioned per frame.

    With an ``aux_decoder`` and ``t_start`` > 0 it is DiffSinger's shallow
    flow: trained on ``t >= t_start`` only, and sampling starts there from the
    aux decoder's mel mixed with noise.

    With ``mean_flow`` the backbone also learns the mean velocity over a step
    (MeanFlow), which the "mean" sampler takes in one or two steps, under a
    speaker guidance scale it reads; the velocity at a time is still the same
    network at a zero span and a scale of 1.

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
        mean_flow: bool = False,
    ):
        """``harmonic_prior`` is the mel's ``sample_rate``, ``n_fft``, ``fmin``
        and ``fmax``, or None for no prior. ``aux_decoder`` is ``AuxDecoder``'s
        arguments, or None for a flow from pure noise; ``aux_grad`` scales its
        gradient into the encoder and the speaker table. ``backbone`` is kept
        because configs and exports name it; LYNXNet2 is the only one.
        ``dual_timestep`` trains a share of each item's frames at a second
        time, as DiffSinger does. ``mean_flow`` gives the backbone the length
        of a step and a guidance scale to read (Geng et al., "Mean Flows for
        One-step Generative Modeling" and "Improved Mean Flows")."""
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
            pitch_fourier,
            HarmonicPrior(n_mels=n_mels, **harmonic_prior) if harmonic_prior else None,
            breathiness,
            key_shift,
            speed,
            tension,
        )
        self.backbone = LYNXNet2Backbone(
            n_mels, hidden_channels, span=bool(mean_flow), **(backbone_args or {})
        )
        self.aux = AuxDecoder(hidden_channels, n_mels, **aux_decoder) if aux_decoder else None
        # Time and speaker reach the network through these; on AdamW, where
        # gradient clipping bounds the step, since Muon's step ignores the
        # gradient's size.
        conditioning = [
            self.backbone.time_mlp, self.backbone.span_mlp, self.backbone.guide_mlp, *self.speaker_layers(),
        ]
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

    def _flow_error(self, mel, noise, cond, voice, mask, t, backbone):
        """Flow loss per item [B] on the path from ``noise`` to ``mel``;
        ``t`` is [B], or [B, T] for a time per frame."""
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        prediction = backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    def _aux_loss(self, mel, cond, voice, mask):
        """The aux decoder's L1, or None without one."""
        if self.aux is None:
            return None
        cond, voice = (
            value * self.aux_grad + value.detach() * (1.0 - self.aux_grad) for value in (cond, voice)
        )
        error = (self.aux(cond, mask, voice).float() - mel.float()).abs() * mask
        return error.sum() / (mask.sum() * self.n_mels).clamp_min(1.0)

    def mean_velocity(self, x, t, span, velocity, cond, mask, voice, guidance=None):
        """The mean velocity over ``span`` from ``t`` under ``guidance``, and
        its derivative along the path ``velocity`` with the step's end held,
        detached."""

        def field(x, t, span):
            return self.backbone(x, t, cond, mask, voice, span, guidance)

        tangents = (velocity, torch.ones_like(t), -torch.ones_like(span))
        mean, derivative = torch.func.jvp(field, (x, t, span), tangents)
        # Detached here: a compiled graph cannot take the empty gradient the
        # derivative would otherwise get.
        return mean, derivative.detach()

    def _mean_error(self, mel, inputs: Conditioning, cond, voice, noise, mean_field, guidance_max=1.0):
        """MeanFlow loss per item [B], over steps between two drawn times, and
        how large the bootstrapped part of the target is against the velocity,
        [B]. With ``guidance_max`` over 1 each item draws a speaker guidance
        scale up to it and trains the mean of the velocity guided by it."""
        mask, batch, device = inputs.mask, mel.shape[0], mel.device
        first, second = (self._times(batch, device) for _ in range(2))
        t, span = torch.minimum(first, second), (first - second).abs()
        span = span * (torch.rand(batch, device=device, generator=self.generator) >= BOUNDARY_SHARE)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * mel
        velocity = mel - noise
        guidance = None
        with torch.no_grad():
            if guidance_max > 1.0:
                # Log-uniform: the smaller scales are the ones used most.
                draw = torch.rand(batch, device=device, generator=self.generator)
                guidance = torch.exp(draw * math.log(guidance_max))
                null = inputs._replace(speaker=torch.full_like(inputs.speaker, self.speaker_count))
                free = self.backbone(
                    x_t, t, self.encoder(null), mask, self.encoder.voice(null.speaker), None, guidance
                )
            # The derivative is taken along the network's own velocity: the
            # item's is that plus its noise, which the target would carry span
            # times over.
            path = self.backbone(x_t, t, cond, mask, voice, None, guidance).to(x_t.dtype)
            if guidance is not None:
                # The guided velocity, from the item's own and the network's
                # guided and speaker-free ones.
                push = (1.0 - 1.0 / guidance)[:, None, None]
                velocity = velocity + push * (path - free.to(x_t.dtype)) * mask
        # The time derivative passes FP16's range, so it is taken in FP32 there.
        kind = device.type
        fp16 = torch.is_autocast_enabled(kind) and torch.get_autocast_dtype(kind) == torch.float16
        if fp16:
            cond, voice = cond.float(), voice.float()
        with torch.autocast(kind, enabled=False) if fp16 else nullcontext():
            mean, derivative = mean_field(x_t, t, span, path, cond, mask, voice, guidance)
        # The mean over [t, t + span] is the velocity at t plus span times its
        # own derivative in t; the network's derivative stands in as a target.
        correction = span[:, None, None] * derivative.float() * mask
        count = (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)
        size = (correction.square().sum((1, 2)) / count).sqrt()
        # A path's curvature keeps this well under 1; past it the target is
        # feeding on itself.
        ratio = size / ((velocity.square() * mask).sum((1, 2)) / count).sqrt().clamp_min(1e-8)
        error = (mean.float() - (velocity + correction)).square() * mask
        return error.sum((1, 2)) / count, ratio

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
    def _pooled(error, frames, weighted=False):
        """Per-item ``error`` averaged over the batch's frames; ``weighted``
        applies ``adaptive_weight``."""
        weight = adaptive_weight(error) if weighted else 1.0
        return (weight * error * frames).sum() / frames.sum().clamp_min(1.0)

    def forward(self, mel, inputs: Conditioning, speaker_dropout=0.0, backbone=None, tension_dropout=0.0,
                mean_ratio=0.0, mean_field=None, mean_guidance=1.0):
        """Flow-matching loss for normalised ``mel`` [B, n_mels, T], the aux
        decoder's L1 (None without one) and the ``MeanFlowLosses`` (None when
        off), all on the per-bin normalised mel. ``backbone`` stands in for
        ``self.backbone``, as its compiled wrapper does, and ``mean_field`` for
        ``self.mean_velocity``.

        The first ``mean_ratio`` of the batch trains the mean velocity instead
        of the flow. Every item's loss then takes ``adaptive_weight``, as in
        MeanFlow. ``mean_guidance`` is ``_mean_error``'s largest guidance
        scale; 1 trains the mean velocity unguided.

        ``speaker_dropout`` is the share of items trained on the null speaker,
        and ``tension_dropout`` the share trained with a flat tension, so the
        input can be turned down at inference."""
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
        # The batch's head trains the mean velocity, the rest the flow.
        batch = mel.shape[0]
        mean_items = 0
        if self.backbone.span_mlp is not None:
            mean_items = min(batch - 1, int(round(mean_ratio * batch)))
        head, rest = slice(0, mean_items), slice(mean_items, None)

        t = self._train_times(batch - mean_items, mel.shape[-1], mel.device)
        # Not ``backbone or ...``: a compiled module's truth test calls len().
        backbone = self.backbone if backbone is None else backbone
        error = self._flow_error(mel[rest], noise[rest], cond[rest], voice[rest], mask[rest], t, backbone)
        frames = mask[rest].sum((1, 2))
        flow = self._pooled(error, frames)
        mean = None
        if mean_items:
            mean_field = self.mean_velocity if mean_field is None else mean_field
            mean_error, bootstrap_ratio = self._mean_error(
                mel[head], inputs.map(lambda value: value[head]), cond[head], voice[head], noise[head],
                mean_field, mean_guidance,
            )
            mean_frames = mask[head].sum((1, 2))
            mean = MeanFlowLosses(
                objective=self._pooled(mean_error, mean_frames, True),
                flow=flow.detach(),
                mean=self._pooled(mean_error, mean_frames).detach(),
                bootstrap_ratio=bootstrap_ratio.mean(),
            )
            flow = self._pooled(error, frames, True)
        if self.backbone.span_mlp is not None:
            # DDP wants every parameter in the graph, also on a step that left
            # these inputs out.
            idle = [] if mean_items else [self.backbone.span_mlp]
            if not mean_items or mean_guidance <= 1.0:
                idle.append(self.backbone.guide_mlp)
            flow = flow + 0.0 * sum(weight.sum() for module in idle for weight in module.parameters())
        return flow, self._aux_loss(mel, cond, voice, mask), mean

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
        is one of ``SCHEDULES``. The "mean" ``method`` is meant for one or two
        steps. It takes ``cfg_scale`` as the scale the network was trained to
        read, in the one pass; the content guidance applies to it as it is,
        though it was not trained under it.

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
        batch, frames = inputs.content.shape[:2]
        mask = inputs.mask
        # The mean velocity under a speaker guidance scale is one pass of a
        # network that reads the scale.
        scale = None
        if method == "mean" and self.backbone.guide_mlp is not None and cfg_scale != 1.0:
            scale = torch.full((batch,), float(cfg_scale), device=mask.device)
            cfg_scale = 1.0
        guidance = Guidance(
            cfg_scale, content_guidance, guidance_rescale, rescale_mode,
            tuple(float(value) for value in guidance_interval),
        )

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

        def velocity(x, t, span=None):
            active = guidance.active(float(t[0]))
            read = scale if active else None
            if count == 1 or not active:
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], span, read)
            passes = self.backbone(repeat(x), repeat(t), cond, stacked.mask, voice, repeat(span), repeat(read))
            return guidance.combine(passes.chunk(count), mask)

        if noise is None:
            noise = torch.randn((batch, self.n_mels, frames), device=inputs.content.device)
        noise = noise * float(temperature)
        if start_mel is not None:
            start_mel = self._encode(start_mel)
        elif self.starts_from_aux:
            start_mel = self.aux(cond[:batch], mask, voice[:batch])
        x, t0 = self._start(noise, start_mel, mask, start)
        times = time_grid(schedule, max(1, int(steps)), t0, x.device)
        for index in range(times.shape[0] - 1):
            now = float(times[index])
            back = max(self.t_start, now - float(churn) * float(times[index + 1] - times[index]))
            if churn > 0 and 0 < back < now:
                fresh = torch.randn_like(x) if churn_noise is None else churn_noise(index)
                x = self._renoise(x, now, back, float(temperature) * fresh) * mask
                now = back
            t = torch.full((batch,), now, device=x.device, dtype=times.dtype)
            dt = times[index + 1] - now
            v = velocity(x, t, dt.expand(batch) if method == "mean" else None)
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
    leaves the plain flow. A span or guidance input the weights lack starts
    at zero."""
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
    for name in ("phonation", "prior_noise", "prior_degrade", "time_sampling", "shortcut", "shortcut_steps"):
        model.pop(name, None)
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"], n_fft=data["n_fft"],
            fmin=data["mel_fmin"], fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
