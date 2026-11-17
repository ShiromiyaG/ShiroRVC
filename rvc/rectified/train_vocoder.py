"""Train the mel vocoder of the rectified pipeline, ChouwaGAN or SiFi-GAN, by
its recipe (``rvc/configs/rectified/<architecture>/<sample rate>.json``,
copied into the experiment on first use).

The GAN and multi-resolution STFT losses and the volume augmentation are
ported from OpenVPI SingingVocoders (MIT, see THIRD_PARTY_NOTICES).

Usage: python rvc/rectified/train_vocoder.py <spec.json>
"""

import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.lib.terminal import (
    info,
    print_model_summary,
    print_settings_panel,
    progress_task,
    success,
    warning,
)
from rvc.rectified import chouwagan, nagi, sifigan, wavehax
from rvc.rectified.common import (
    RectifiedDataset,
    amp_setup,
    collate_vocoder,
    latest_checkpoint,
    precision_label,
    read_filelist,
    remove_older,
    split_holdout,
    vocoder_run_dir,
)
from rvc.rectified.distributed import Ranks, launch, parse_gpus
from rvc.rectified.mel import LogMel, degrade_mel, normalize_mel
from rvc.rectified.previews import RectifiedPreviews
from rvc.rectified.vocoder import (
    GENERATORS,
    MEL_KEYS,
    generator_architecture,
    load_recipe_config,
    recipe_architecture,
    resolve_discriminator,
)
from rvc.train.diagnostics import clip_or_sample_grad_norm
from rvc.train.ema import WeightEMA
from rvc.train.losses import MultiScaleSTFTLoss, discriminator_loss as repo_discriminator_loss, generator_loss
from rvc.train.progress import EpochRecorder, emit_machine_progress
from rvc.train.schedules import STEP_SCHEDULERS, prepare_schedulers
from rvc.train.setup import enable_discriminator_compile, loader_workers, normalize_san_weights
from rvc.train.stop import finish_stop, install_stop_handlers, uninterruptible_save

TAG = "[VOCODER]"
METRICS_INTERVAL = 8
SAN_DIRECTION_WEIGHT = 0.25
#: Ceiling of the FP16 GradScaler's scale: left to grow, it doubles until a
#: gradient overflows and a step is skipped.
MAX_GRAD_SCALE = 2.0**16
#: Most of a run the spectral-only stage may take, so a short run still meets the discriminator.
AUX_MAX_SHARE = 0.25
#: Under a compressed magnitude in ``LockedSpectralLoss``: bounds its slope at silence.
COMPRESSION_FLOOR = 1e-4
VOCODER_NAMES = {chouwagan.ARCHITECTURE: "ChouwaGAN", sifigan.ARCHITECTURE: "SiFi-GAN", wavehax.ARCHITECTURE: "Wavehax",
                 nagi.ARCHITECTURE: "Nagi"}


def generator_gradient_metrics(net_g):
    """``rvc.train.diagnostics.generator_gradient_metrics`` for a bare vocoder,
    whose parameters carry no ``dec.`` prefix: the trunk, and the output
    ``head``, the ``post_filter``, the Wavehax ``synthesis`` and ``ddsp``
    kept apart; SiFi-GAN's ``source`` and ``filter`` networks; Nagi's ``free`` branch."""
    groups = {"decoder": [], "output": [], "post_filter": [], "synthesis": [], "source": [], "filter": [],
              "allpass": [], "exciter": [], "ddsp": [], "free": []}
    for name, parameter in net_g.named_parameters():
        if parameter.requires_grad:
            group = name.partition(".")[0]
            group = {"sn": "source", "fn": "filter", "free": "free"}.get(group.partition("_")[0], group)
            groups[{"head": "output", "post_filter": "post_filter", "synthesis": "synthesis", "source": "source",
                    "filter": "filter", "allpass_head": "allpass", "exciter": "exciter",
                    "ddsp": "ddsp", "free": "free"}.get(group, "decoder")].append(
                parameter
            )
    metrics = {}
    for group, parameters in groups.items():
        if not parameters:
            continue
        parameter_norm = torch.sqrt(sum(p.detach().float().square().sum() for p in parameters))
        terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
        gradient_norm = torch.sqrt(sum(terms)) if terms else parameter_norm.new_zeros(())
        metrics[f"grad_norm_{group}"] = gradient_norm
        metrics[f"grad_to_param_{group}"] = gradient_norm / parameter_norm.clamp_min(1e-8)
    if isinstance(net_g, sifigan.SiFiGAN):
        metrics.update(sifigan_gradient_metrics(net_g))
    return metrics


def sifigan_clip_groups(net_g):
    """SiFi-GAN's parameters in the groups ``grad_clip_by_group`` clips
    apart: the source network, the filter network, the filter's output conv
    (whose gradient the discriminator drives) and the rest."""
    groups = defaultdict(list)
    for name, parameter in net_g.named_parameters():
        if parameter.requires_grad:
            group = {"sn": "source", "fn": "filter"}.get(name.partition("_")[0], "decoder")
            groups["output" if name.startswith("fn_output.") else group].append(parameter)
    return list(groups.values())


def clip_groups_grad_norm(groups, max_norm: float):
    """Each group clipped to ``max_norm`` on its own; the norm of them all,
    before clipping."""
    norms = torch.stack([torch.nn.utils.clip_grad_norm_(group, max_norm) for group in groups])
    return norms.square().sum().sqrt()


def sifigan_gradient_metrics(net_g):
    """SiFi-GAN's gradient norms per part of each network: every stage's
    upsampling, blocks and downsampling into it, and the output conv's weight
    and bias apart; and, read from its last forward, ``stream_rms``."""
    stages = len(net_g.upsample_rates)
    parts = defaultdict(list)
    for name, parameter in net_g.named_parameters():
        if parameter.grad is None:
            continue
        module, _, rest = name.partition(".")
        prefix, _, kind = module.partition("_")
        network = {"sn": "source", "fn": "filter"}.get(prefix)
        if network is None:
            continue
        index = int(rest.partition(".")[0]) if rest[:1].isdigit() else None
        if kind == "output":
            part = "output_bias" if name.endswith("bias") else "output_weight"
        elif kind == "upsamples":
            part = f"up{index}"
        elif kind == "blocks":
            part = f"blocks{index // net_g.filter_kernels if network == 'filter' else index}"
        elif kind == "downsamples":
            # Layer j lands at the output rate of stage stages - 2 - j.
            part = f"down{stages - 2 - index}"
        else:
            part = kind
        parts[f"grad_norm_{network}_{part}"].append(parameter.grad.detach().float().square().sum())
    metrics = {key: torch.sqrt(sum(terms)) for key, terms in parts.items()}
    metrics.update({f"stream_rms_{key}": value for key, value in net_g.stream_rms.items()})
    return metrics


class PlateauScale:
    """A factor on the scheduled LR, cut by ``factor`` whenever the monitored
    mean fails to beat its best by ``threshold`` (relative) for ``patience``
    evaluations in a row; never below ``min_scale``."""

    def __init__(self, factor: float, patience: int, threshold: float, min_scale: float):
        self.factor = factor
        self.patience = max(1, patience)
        self.threshold = threshold
        self.min_scale = min_scale
        self.scale = 1.0
        self.best = math.inf
        self.bad = 0

    def update(self, value: float) -> bool:
        """Record one evaluation; returns whether the scale was cut."""
        if value < self.best * (1.0 - self.threshold):
            self.best, self.bad = value, 0
            return False
        self.bad += 1
        if self.bad < self.patience or self.scale <= self.min_scale:
            return False
        self.scale = max(self.min_scale, self.scale * self.factor)
        self.bad = 0
        return True

    def state_dict(self) -> dict:
        return {"scale": self.scale, "best": self.best, "bad": self.bad}

    def load_state_dict(self, state: dict) -> None:
        self.scale, self.best, self.bad = state["scale"], state["best"], state["bad"]


def build_plateau(settings: dict, eval_interval: int):
    """Reduce on plateau of a spectral loss, the only non-adversarial one: the
    training mean over ``plateau_interval`` steps, or with ``plateau_metric``
    "val", the held-out one at each evaluation. Returns the scale, the metric
    and the steps between two readings of it."""
    metric = str(settings.get("plateau_metric", "train"))
    if metric not in ("train", "val"):
        raise ValueError(f"plateau_metric must be 'train' or 'val', not {metric!r}.")
    if metric == "val" and not eval_interval:
        raise ValueError("plateau_metric 'val' needs holdout_clips and eval_interval.")
    interval = eval_interval if metric == "val" else int(settings.get("plateau_interval", 0))
    plateau = PlateauScale(
        float(settings.get("plateau_factor", 0.5)),
        int(settings.get("plateau_patience", 3)),
        float(settings.get("plateau_threshold", 0.005)),
        float(settings.get("plateau_min_scale", 0.1)),
    )
    return plateau, metric, interval


def plateau_update(plateau: PlateauScale, metric: str, value: float, writer, step: int) -> None:
    """Give the plateau one reading; ``writer`` is None off the main rank."""
    cut = plateau.update(value)
    if writer is None:
        return
    if cut:
        info(f"{metric.capitalize()} spectral loss plateaued at {plateau.best:.3f}; "
             f"LR scale now x{plateau.scale:g}.", tag=TAG)
    writer.add_scalar("learning_rate/plateau_metric", value, step)


def ramp(step: int, length: int) -> float:
    """Linear warmup factor of ``step`` over ``length`` steps."""
    return 1.0 if length <= 0 else min(1.0, max(0.0, (step + 1) / length))


def build_loaders(name: str, config: dict, batch_size: int, ranks: Ranks):
    """The training dataset, its loader and sampler, the held-out batches and
    how many clips they hold. ``batch_size`` is per GPU, as in the RVC trainer."""
    settings = config["vocoder"]
    hop = config["data"]["hop_length"]
    entries, holdout_entries = split_holdout(read_filelist(name), int(settings.get("holdout_clips", 0)))
    dataset = RectifiedDataset(entries, config, "vocoder", settings["segment_size"] // hop)
    # Loaded once: fixed middle crops, the same on every rank and every call.
    holdout = list(
        DataLoader(
            RectifiedDataset(
                holdout_entries, config, "vocoder",
                int(settings.get("eval_segment_size", 131072)) // hop, augment=False,
            ),
            batch_size=min(batch_size, len(holdout_entries)),
            collate_fn=collate_vocoder,
        )
    ) if holdout_entries else []
    if len(dataset) // ranks.world < batch_size:
        raise ValueError(
            f"{len(dataset)} clips is fewer than one batch of {batch_size} on each of {ranks.world} GPU(s)."
        )
    workers = loader_workers(4)
    sampler = ranks.sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_vocoder,
        drop_last=True,
        pin_memory=ranks.device.type == "cuda",
        persistent_workers=workers > 0,
    )
    return dataset, loader, sampler, holdout, len(holdout_entries)


class Discriminators(nn.ModuleDict):
    """ChouwaGAN's discriminator families; ``forward`` runs every one and
    returns ``{family: (outputs, feature maps)}``."""

    @property
    def supports_san(self) -> bool:
        """Whether a head ends in a SAN projection, which ``san_training`` splits."""
        return any(getattr(head, "use_san", False) for family in self.values() for head in family.discriminators)

    def forward(self, y, san_training=False):
        if san_training:
            return {name: family(y, san_training=True) for name, family in self.items()}
        return {name: family(y) for name, family in self.items()}

    def enable_compile(self, mode: str = "default") -> bool:
        """torch.compile every family's forward in place, which keeps the keys,
        so the branchwise update runs compiled too: a graph per family and pass
        (D's update, G's, and G's no-grad real one). Not head by head: weight
        norm gives every conv a class of its own, so the period heads alone
        would pass Dynamo's recompile limit. Training only; a family whose
        compiled call fails runs eager from then on."""
        if getattr(self, "_compile_enabled", False):
            return True
        for family in self.values():
            _compile_forward(family, mode)
        self._compile_enabled = True
        return True


def _compile_forward(module: nn.Module, mode: str) -> None:
    eager = module.forward
    compiled = torch.compile(eager, dynamic=False, mode=mode)
    failed = False

    def forward(*args, **kwargs):
        nonlocal failed
        if failed or not module.training:
            return eager(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as error:
            failed = True
            warning(f"{type(module).__name__} runs eager from here on; its compiled forward failed: {error}",
                    tag="[TRAIN]")
            return eager(*args, **kwargs)

    module.forward = forward


def build_networks(config: dict, device):
    """The generator, the discriminator and the generator's constructor arguments."""
    data, settings = config["data"], config["vocoder"]
    hparams = dict(
        sample_rate=int(data["sample_rate"]), num_mels=int(data["n_mels"]),
        hop_length=int(data["hop_length"]), **settings["model"],
    )
    if recipe_architecture(settings) == wavehax.ARCHITECTURE and hparams.get("mel_prior"):
        # Kept in the export, which then builds the same prior from its own settings.
        hparams.update(
            mel_n_fft=int(data["n_fft"]), mel_win_length=int(data["win_length"]), mel_fmin=float(data["mel_fmin"]),
            mel_fmax=float(data["mel_fmax"]), mel_mean=float(data["mel_mean"]), mel_std=float(data["mel_std"]),
        )
    if recipe_architecture(settings) == nagi.ARCHITECTURE:
        # The mel its gains start from, kept in the export as the prior's above.
        hparams.update(mel_fmin=float(data["mel_fmin"]), mel_fmax=float(data["mel_fmax"]),
                       mel_mean=float(data["mel_mean"]), mel_std=float(data["mel_std"]))
    net_g = GENERATORS[recipe_architecture(settings)](**hparams).to(device)
    d_settings = settings["discriminator"]
    # ``architecture`` "sifigan": SiFi-GAN's own; ChouwaGAN's otherwise.
    families = sifigan if d_settings.get("architecture") == sifigan.ARCHITECTURE else chouwagan
    net_d = Discriminators(families.build_families(d_settings, data["sample_rate"])).to(device)
    return net_g, net_d, hparams


def pretrained_state(path: str, sample_rate: int) -> dict:
    """The weights of a G or D checkpoint or an export to start from. An
    export says its rate, and one at another rate is refused."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    rate = checkpoint.get("config", {}).get("data", {}).get("sample_rate")
    if rate is not None and int(rate) != int(sample_rate):
        raise ValueError(f"{path} renders {rate} Hz audio; this experiment trains at {sample_rate} Hz.")
    return checkpoint["model"]


def load_families(net_d, weights: dict) -> bool:
    """Load each discriminator family ``weights`` holds in the network's
    shapes; one it lacks, or holds in other shapes, keeps its fresh weights.
    True when every family loaded and the file holds no other."""
    loaded = set()
    for name, family in net_d.items():
        prefix = name + "."
        part = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        own = family.state_dict()
        if part.keys() == own.keys() and all(part[key].shape == own[key].shape for key in own):
            family.load_state_dict(part)
            loaded.add(name)
    return loaded == set(net_d) == {key.partition(".")[0] for key in weights}


def load_optimizer_state(optim, saved, net_g=None):
    """Load the optimiser's state from a checkpoint whose generator had other
    parameters: the DDSP branch's, when it was turned off. The state is mapped
    by the parameters' order -- the branch's parameters sit at the end -- so
    the ones the other run had and this one does not are dropped, and the ones
    it did not have start fresh."""
    groups, saved_groups = optim.param_groups, saved.get("param_groups", [])
    if len(groups) != 1 or len(saved_groups) != 1:
        optim.load_state_dict(saved)
        return
    current, previous = groups[0]["params"], saved_groups[0]["params"]
    if len(current) == len(previous):
        optim.load_state_dict(saved)
        return
    keep = previous[:len(current)]
    padding = list(range(-1, -1 - (len(current) - len(keep))))
    state = saved.get("state", {})
    trimmed = dict(saved_groups[0], params=keep + padding)
    optim.load_state_dict({"state": {k: v for k, v in state.items() if k in keep}, "param_groups": [trimmed]})


def adapt_optimizer_state(optim, net_g):
    """Reshape the moments Adam keeps for the parameters whose shape the run
    changed: the Wavehax input projection, whose width follows the mel prior.
    The planes the other run had and this one does not are dropped, and the
    ones it did not have start at zero."""
    adapter = getattr(net_g, "adapt_plane_weight", None)
    if adapter is None:
        return
    names = {id(parameter): name for name, parameter in net_g.named_parameters()}
    for parameter, state in optim.state.items():
        if not names.get(id(parameter), "").endswith("input_proj.weight"):
            continue
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or value.ndim != parameter.ndim:
                continue
            if value.shape == parameter.shape:
                continue
            if value.shape[0] != parameter.shape[0] or value.shape[2:] != parameter.shape[2:]:
                continue
            adapted = adapter(value, torch.zeros_like(parameter))
            if adapted.shape == parameter.shape:
                state[key] = adapted


def restore(spec, out_dir, sample_rate, net_g, net_d, optim_g, optim_d, ema, scaler, plateau, plateau_metric):
    """Resume the run's latest checkpoints, or start from the pretrained
    weights of ``spec``. Returns where training starts: ``epoch``, ``step``,
    ``skipped``, ``resumed``, the resumed run's ``aux_steps`` and a
    ``starting_point`` label."""
    start = SimpleNamespace(epoch=1, step=0, skipped=0, resumed=False, aux_steps=None, starting_point="scratch")
    resume_g = None if spec.get("fresh") else latest_checkpoint(out_dir, "G")
    resume_d = None if spec.get("fresh") else latest_checkpoint(out_dir, "D")
    if resume_g and resume_d:
        state_g = torch.load(resume_g, map_location="cpu", weights_only=True)
        state_d = torch.load(resume_d, map_location="cpu", weights_only=True)
        architecture = generator_architecture(net_g)
        saved = (state_g.get("architecture", chouwagan.ARCHITECTURE), state_d.get("discriminator", "original"))
        if saved != (architecture, "original"):
            raise ValueError(
                f"{resume_g} is a {saved[0]} checkpoint with the {saved[1]} discriminator, not {architecture} with "
                f"the rectified one. Turn on Fresh Training or use another model name."
            )
        net_g.load_state_dict(state_g["model"])
        load_optimizer_state(optim_g, state_g["optimizer"], net_g)
        # A parameter the run resized keeps its moments in the old shape.
        adapt_optimizer_state(optim_g, net_g)
        # A family added or changed in the config since starts fresh, and so do
        # D's Adam moments; its LR stays where the schedule had it.
        same_d = load_families(net_d, state_d["model"])
        if same_d:
            optim_d.load_state_dict(state_d["optimizer"])
        else:
            for group, saved in zip(optim_d.param_groups, state_d["optimizer"]["param_groups"]):
                group.update({key: saved[key] for key in ("lr", "initial_lr") if key in saved})
        if "ema" in state_g:
            ema.load_state_dict(state_g["ema"], net_g)
        if scaler is not None and state_g.get("scaler"):
            scaler.load_state_dict(state_g["scaler"])
        start.epoch, start.step = state_g["epoch"] + 1, state_g["step"]
        start.skipped = int(state_g.get("amp_skipped_steps", 0))
        start.aux_steps = state_g.get("aux_steps")
        # A best from the other metric is on another scale; that one restarts.
        if state_g.get("plateau") and state_g["plateau"].get("metric", "train") == plateau_metric:
            plateau.load_state_dict(state_g["plateau"])
        start.resumed = True
        start.starting_point = f"resumed from {os.path.basename(resume_g)}"
        if not same_d:
            start.starting_point += ", new D families and D's Adam moments fresh"
        return start
    if spec.get("pretrained_g"):
        net_g.load_state_dict(pretrained_state(spec["pretrained_g"], sample_rate))
        ema.reseed(net_g)
        start.starting_point = f"G from {spec['pretrained_g']}"
    if spec.get("pretrained_d"):
        # A recipe may train against some of the families the file holds.
        load_families(net_d, pretrained_state(spec["pretrained_d"], sample_rate))
        start.starting_point += f", D from {spec['pretrained_d']}"
    return start


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude losses, averaged over the resolutions."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, x, index):
        fft_size, hop_size, win_length = self.resolutions[index]
        window = getattr(self, f"window_{index}")
        spec = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        return torch.clamp(spec.abs(), min=10**-3.5)

    def forward(self, x, y):
        """Predicted ``x`` and target ``y``, [B, T] each."""
        x, y = x.float(), y.float()
        sc_loss = mag_loss = 0.0
        for index in range(len(self.resolutions)):
            x_mag, y_mag = self._magnitude(x, index), self._magnitude(y, index)
            sc_loss = sc_loss + torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            mag_loss = mag_loss + F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss / len(self.resolutions), mag_loss / len(self.resolutions)


class RandomScaleSpectralLoss(nn.Module):
    """pc-ddsp's random-scale spectral loss (https://github.com/yxlllc/pc-ddsp,
    MIT): spectral convergence and log-magnitude L1 at ``scales`` FFT sizes
    drawn each call from [``fft_min``, ``fft_max``), with no overlap, from a
    random start. The sizes and frame edges move every step, so the magnitude
    depends on how the harmonics add up inside each frame: what lets it train
    phase. Only sizes without a prime factor above 7: cuFFT plans every new
    size, slowly for the others, and would plan anew for most steps."""

    def __init__(self, fft_min=256, fft_max=2048, scales=4, eps=1e-7):
        super().__init__()
        self.scales, self.eps = int(scales), float(eps)

        def smooth(n):
            for prime in (2, 3, 5, 7):
                while n % prime == 0:
                    n //= prime
            return n == 1

        self.sizes = [n for n in range(int(fft_min), int(fft_max)) if smooth(n)]
        self.windows = {}

    def _magnitude(self, x, n_fft):
        window = self.windows.get(n_fft)
        if window is None or window.device != x.device:
            window = self.windows[n_fft] = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(x, n_fft, n_fft, window=window, center=False, return_complex=True)
        # As torchaudio's ``normalized=True``: over the window's energy.
        return spec.abs() / window.square().sum().sqrt() + self.eps

    def forward(self, x, y):
        """Predicted ``x`` and target ``y``, [B, T] each."""
        with torch.autocast(x.device.type, enabled=False):
            x, y = x.float(), y.float()
            loss = 0.0
            for index in torch.randint(len(self.sizes), (self.scales,)).tolist():
                n_fft = self.sizes[index]
                start = int(torch.randint(n_fft, ()))
                x_mag, y_mag = self._magnitude(x[:, start:], n_fft), self._magnitude(y[:, start:], n_fft)
                convergence = torch.linalg.norm(y_mag - x_mag, dim=(1, 2)) / torch.linalg.norm(y_mag + x_mag, dim=(1, 2))
                loss = loss + convergence.mean() + F.l1_loss(y_mag.log(), x_mag.log())
            return loss / self.scales


class LockedSpectralLoss(nn.Module):
    """The complex spectrum of Nagi's filtered pulses against the target's,
    both compressed in magnitude, over the voiced frames under ``max_hz``:
    with the pulses locked to the target it trains their phase directly. On
    the pulses alone: a noise the target's is not correlated with would only
    be pushed down. Each frame of the pulses is first turned by the delay,
    within half a period, and the one angle that bring it closest to the
    target: a recording's own (its polarity among them), not heard and not in
    the mel, and without them pulses of the right shape cost more than none."""

    #: Steps per sample of the delay search.
    oversample = 4

    def __init__(self, sample_rate, hop_length, fft_sizes, hop_sizes, win_lengths, max_hz, compression=0.3):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.sample_rate = int(sample_rate)
        self.hop_length, self.compression = int(hop_length), float(compression)
        for index, (fft_size, _, win_length) in enumerate(self.resolutions):
            window = torch.hann_window(win_length)
            # A sine's bin reads its amplitude.
            self.register_buffer(f"window_{index}", window * 2 / window.sum(), persistent=False)
            setattr(self, f"bins_{index}", int(min(max_hz, sample_rate / 2) * fft_size / sample_rate) + 1)

    def _spectrum(self, x, index):
        fft_size, hop_size, win_length = self.resolutions[index]
        spec = torch.stft(x, fft_size, hop_size, win_length, getattr(self, f"window_{index}"), return_complex=True)
        spec = spec[:, :getattr(self, f"bins_{index}")]
        return spec * (spec.abs() + COMPRESSION_FLOOR) ** (self.compression - 1)

    @torch.no_grad()
    def _alignment(self, pulses, target, f0, fft_size):
        """The factor [B, bins, frames] that turns ``pulses``' spectrum by each
        frame's best delay and angle against ``target``'s; ``f0`` [B, frames]."""
        size = self.oversample * fft_size
        # Lag m is the pulses turned by exp(2j pi bin m / size).
        correlation = torch.fft.ifft(pulses * target.conj(), n=size, dim=1)
        lag = torch.fft.fftfreq(size, 1 / size, device=pulses.device) / self.oversample
        period = self.sample_rate / f0.clamp(min=chouwagan.SOURCE_F0_FLOOR)
        outside = lag.abs().view(1, -1, 1) > 0.5 * period.unsqueeze(1)
        best = correlation.abs().masked_fill(outside, 0.0).argmax(1, keepdim=True)
        bins = torch.arange(pulses.shape[1], device=pulses.device).view(1, -1, 1)
        angle = 2 * math.pi * bins * lag[best] / fft_size - correlation.gather(1, best).angle()
        return torch.polar(torch.ones_like(angle), angle)

    def forward(self, pulses, y, f0):
        """``pulses`` and ``y`` [B, 1, T], ``f0`` [B, frames] (0 unvoiced)."""
        with torch.autocast(y.device.type, enabled=False):
            pulses, y = pulses.squeeze(1).float(), y.squeeze(1).float()
            loss = 0.0
            for index, (fft_size, hop_size, _) in enumerate(self.resolutions):
                predicted, target = self._spectrum(pulses, index), self._spectrum(y, index)
                frame = (torch.arange(target.shape[-1], device=y.device) * hop_size // self.hop_length)
                frame_f0 = f0[:, frame.clamp(max=f0.shape[-1] - 1)].float()
                predicted = predicted * self._alignment(predicted, target, frame_f0, fft_size)
                error = (predicted - target).abs().square().mean(1)
                voiced = (frame_f0 > 0).to(error.dtype)
                loss = loss + (error * voiced).sum() / voiced.sum().clamp(min=1)
            return loss / len(self.resolutions)


class SyncEnvelopeLoss(nn.Module):
    """How each band's power moves inside the glottal cycle, generated against
    target: the first ``harmonics`` Fourier coefficients of its envelope along
    the locked cycle, over its mean, read under the lock's window in the
    voiced frames. The generated ones are first turned by each frame's best
    delay, one for all bands. It gives the burst of each cycle its depth,
    shape and place across bands, in noise as in harmonics, which a magnitude
    averages away and the locked spectrum cannot see in noise."""

    def __init__(self, sample_rate, hop_length, band_edges, harmonics=6, hop=256, delays=256):
        super().__init__()
        self.hop_length, self.hop = int(hop_length), int(hop)
        self.bands = chouwagan.EnvelopeLoss(sample_rate, band_edges)
        window = torch.hann_window(int(nagi.LOCK_SECONDS * sample_rate) | 1, periodic=False)
        self.register_buffer("window", (window / window.sum()).view(1, 1, -1), persistent=False)
        order = torch.arange(1, harmonics + 1)
        self.register_buffer("order", order.view(1, -1, 1), persistent=False)
        angle = -2 * math.pi * order.view(-1, 1) * torch.arange(delays) / delays
        self.register_buffer("delays", torch.polar(torch.ones_like(angle), angle), persistent=False)

    def _coefficients(self, x, turns):
        """[N, T] -> the coefficients [N, bands, harmonics, frames] of each
        band's power along the cycle ``turns``, and the bands that sound
        [N, bands, 1]."""
        envelopes, means = self.bands._envelopes(x)
        power = envelopes.square()
        angle = -2 * math.pi * torch.remainder(self.order * turns.unsqueeze(1), 1.0).float()
        moving = power.unsqueeze(2) * torch.polar(torch.ones_like(angle), angle).unsqueeze(1)
        batch, bands, harmonics, length = moving.shape
        moving = torch.view_as_real(moving).permute(0, 1, 2, 4, 3).reshape(-1, 1, length)
        moving = F.conv1d(moving, self.window, stride=self.hop).view(batch, bands, harmonics, 2, -1)
        mean = F.conv1d(power.reshape(-1, 1, length), self.window, stride=self.hop).view(batch, bands, 1, -1)
        return torch.complex(moving[:, :, :, 0], moving[:, :, :, 1]) / mean.clamp(min=1e-4), means > 1e-4

    def forward(self, y_hat, y, turns, f0):
        """``y_hat`` and ``y`` [B, 1, T], ``turns`` [B, T] the locked cycle
        (cycles, in double), ``f0`` [B, frames] (0 unvoiced)."""
        with torch.autocast(y.device.type, enabled=False):
            generated, _ = self._coefficients(y_hat.squeeze(1).float(), turns)
            with torch.no_grad():
                target, audible = self._coefficients(y.squeeze(1).float(), turns)
                cross = (generated * target.conj() * audible.unsqueeze(-1)).sum(1)
                best = torch.einsum("bhf,hd->bdf", cross, self.delays).real.argmax(1)
            generated = generated * self.delays[:, best].permute(1, 0, 2).unsqueeze(1)
            error = (generated - target).abs().square().mean(2)
            centre = torch.arange(error.shape[-1], device=y.device) * self.hop + self.window.shape[-1] // 2
            voiced = f0[:, (centre // self.hop_length).clamp(max=f0.shape[-1] - 1)] > 0
            weight = (audible & voiced.unsqueeze(1)).to(error.dtype)
            return (error * weight).sum() / weight.sum().clamp(min=1)


class SpectralLoss:
    """The recipe's reconstruction loss: the L1 of the input's mel read up to
    Nyquist (``loss_n_mels`` and ``loss_fmin`` resize it) at ``c_mel``, and
    with ``c_stft``, the multi-resolution STFT loss: the mel's wide high bands
    cannot tell a harmonic from noise, the linear bins can."""

    def __init__(self, config: dict, device):
        data, settings = config["data"], config["vocoder"]
        sample_rate = data["sample_rate"]
        # The last call's terms, for the log, when there are two.
        self.terms = {}
        self.mel = LogMel(
            sample_rate, data["n_fft"], data["win_length"], data["hop_length"],
            settings.get("loss_n_mels", data["n_mels"]), settings.get("loss_fmin", data["mel_fmin"]),
            sample_rate / 2,
        ).to(device)
        self.weight = settings["c_mel"]
        self.label = f"mel x{self.weight:g}"
        self.stft = None
        if settings.get("c_stft"):
            self.stft = MultiResolutionSTFTLoss(
                settings["loss_fft_sizes"], settings["loss_hop_sizes"], settings["loss_win_lengths"]
            ).to(device)
            self.stft_weight = settings["c_stft"]
            self.label += f" + MR-STFT x{self.stft_weight:g}"

    def mel_error(self, y, y_hat):
        """The per-band, per-frame absolute log-mel error of ``y_hat`` against
        ``y``, [B, 1, T] each -> [B, n_mels, T]."""
        return (self.mel(y_hat.squeeze(1)) - self.mel(y.squeeze(1))).abs()

    def mel_l1(self, y, y_hat):
        """The log-mel L1 of ``y_hat`` against ``y``, [B, 1, T] each, at the
        loss mel's scale."""
        return self.mel_error(y, y_hat).mean()

    def __call__(self, y, y_hat):
        loss_mel = self.mel_l1(y, y_hat) * self.weight
        if self.stft is None:
            return loss_mel
        sc_loss, mag_loss = self.stft(y_hat.squeeze(1), y.squeeze(1))
        loss_stft = (sc_loss + mag_loss) * self.stft_weight
        self.terms = {"loss_mel": loss_mel.detach(), "loss_stft": loss_stft.detach()}
        return loss_stft + loss_mel


def bigvsan_discriminator_loss(real: dict, fake: dict):
    """BigVSAN's loss over every SAN head, and its real and fake halves:
    softplus² on the function output, and on the direction output the real
    term against minus ``softplus(1 - d)²`` on the fake one."""
    heads_real = [d for outputs, _ in real.values() for d in outputs]
    heads_fake = [d for outputs, _ in fake.values() for d in outputs]
    loss_real = loss_fake = 0.0
    for (real_fun, real_dir), (fake_fun, fake_dir) in zip(heads_real, heads_fake):
        loss_real = loss_real + torch.mean(F.softplus(1 - real_fun.float()) ** 2) + torch.mean(
            F.softplus(1 - real_dir.float()) ** 2
        )
        loss_fake = loss_fake + torch.mean(F.softplus(fake_fun.float()) ** 2) - torch.mean(
            F.softplus(1 - fake_dir.float()) ** 2
        )
    return loss_real + loss_fake, loss_real, loss_fake


def feature_loss(fmaps_real, fmaps_fake, relative=False) -> torch.Tensor:
    """Feature matching over every layer of every head, x2. ``relative``
    divides each layer's error by the mean magnitude of its real features:
    a discriminator whose features grow then cannot grow the loss with them."""
    loss = 0.0
    for head_real, head_fake in zip(fmaps_real, fmaps_fake):
        for real, fake in zip(head_real, head_fake):
            real = real.float()
            error = torch.mean(torch.abs(real - fake.float()))
            loss = loss + (error / (real.abs().mean() + 1e-6) if relative else error)
    return loss * 2


def volume_augment(mel, audio, probability: float):
    """SingingVocoders' volume augmentation: with ``probability`` per clip, a
    gain of e^-3 up to e^3 that keeps the peak under 1, on the audio and, as a
    shift, on its log mel."""
    peak = audio.abs().amax(dim=(1, 2)) + 1e-5
    high = torch.clamp(torch.log(1 / peak), max=3.0)
    shift = -3.0 + (high + 3.0) * torch.rand_like(high)
    shift = torch.where(torch.rand_like(high) < probability, shift, torch.zeros_like(shift))
    mel = torch.clamp(mel + shift.view(-1, 1, 1), min=math.log(1e-5))
    return mel, audio * shift.exp().view(-1, 1, 1)


def _halves(value, size):
    """``value``, a tensor or a nest of dicts, lists and tuples of tensors
    batched real then fake, as its (real, fake) pair."""
    if torch.is_tensor(value):
        return value[:size], value[size:]
    if isinstance(value, dict):
        pairs = {key: _halves(item, size) for key, item in value.items()}
        return {key: pair[0] for key, pair in pairs.items()}, {key: pair[1] for key, pair in pairs.items()}
    pairs = [_halves(item, size) for item in value]
    return type(value)(pair[0] for pair in pairs), type(value)(pair[1] for pair in pairs)


class Adversary:
    """The discriminator and the GAN losses it trains by. ``train_d`` is the
    network as the discriminator update calls it (DDP-wrapped)."""

    def __init__(self, net_d, train_d, settings: dict):
        self.net = net_d
        self.train_net = train_d
        d_settings = settings["discriminator"]
        self.hinge = settings.get("gan_loss", "lsgan") == "hinge"
        # ``average_heads``: the GAN losses as a mean over the heads instead of
        # their sum. The discriminator's settings may override the recipe's.
        self.heads = 1
        if d_settings.get("average_heads", settings.get("average_heads", False)):
            self.heads = sum(len(family.discriminators) for family in net_d.values())
        # ``branchwise_discriminator``: D's update one family at a time, each
        # one's graph freed before the next, for the memory of the largest
        # instead of their sum; the gradients are the same. On one GPU only:
        # the gradient sync wraps the whole network. G's update always runs
        # that way, reading D frozen and outside the sync.
        self.branchwise = bool(settings.get("branchwise_discriminator", False)) and train_d is net_d
        # ``fm_relative``: ``feature_loss`` over the real features' magnitude.
        # ``adv_balance``: the norm of G's gradient from D (adversarial and
        # feature matching) at the output is capped at that many times the
        # spectral losses', so a discriminator that sharpens cannot drown them.
        self.fm_relative = bool(settings.get("fm_relative", False))
        self.adv_balance = float(settings.get("adv_balance", 0.0))
        self.san = net_d.supports_san
        # ``san_loss`` "bigvsan": BigVSAN's discriminator loss on the SAN
        # heads, in place of the repo's bounded mirror of it.
        self.bigvsan = self.san and settings.get("san_loss", "repo") == "bigvsan"
        # The generator update reads D's frozen weights in two passes, real
        # then fake: cached, each weight norm is built once. Not compiled,
        # where the graph already fuses it.
        self.cache_weights = not getattr(net_d, "_compile_enabled", False)
        # ``remove_dc``: D sees each segment without its mean. Real audio has
        # next to no DC, so a compressed magnitude's DC bin otherwise sends
        # the generator a gradient far larger than anything audible.
        self.remove_dc = bool(d_settings.get("remove_dc", settings.get("remove_dc", False)))

    def _input(self, x):
        return x - x.mean(-1, keepdim=True) if self.remove_dc else x

    def describe(self, d_settings: dict) -> str:
        net = self.net
        names = [name.upper() for name in net if name != "mpd"]
        if "mpd" in net:
            names.append(f"MPD {d_settings['periods']}")
        cap = int(d_settings.get("mpd_max_channels", 1024))
        return " + ".join(names) + (f", MPD width capped at {cap}" if "mpd" in net and cap < 1024 else "") + (
            (", SAN, BigVSAN loss" if self.bigvsan else ", SAN") if self.san else ""
        ) + (", hinge" if self.hinge else "") + (f", mean over {self.heads} heads" if self.heads > 1 else "") + (
            ", one family at a time" if self.branchwise else ""
        ) + (", compiled" if getattr(net, "_compile_enabled", False) else "") + (
            ", DC removed" if self.remove_dc else ""
        ) + (", relative FM" if self.fm_relative else "") + (
            f", G's gradient from D capped at {self.adv_balance:g}x the spectral one's" if self.adv_balance > 0
            else ""
        )

    def _loss(self, real, fake):
        """D's loss on ``real`` and ``fake`` outputs, and its real and fake parts."""
        if self.bigvsan:
            return bigvsan_discriminator_loss(real, fake)
        heads_real = [d for outputs, _ in real.values() for d in outputs]
        heads_fake = [d for outputs, _ in fake.values() for d in outputs]
        return repo_discriminator_loss(
            heads_real, heads_fake, san_direction_weight=SAN_DIRECTION_WEIGHT, hinge=self.hinge
        )

    def _adversarial(self, outputs) -> torch.Tensor:
        """The generator's loss on a family's heads; against SAN heads, the
        softplus one their training pairs with."""
        return generator_loss(outputs, use_softplus=self.san, hinge=self.hinge)

    def discriminator_losses(self, y, fake):
        """D's loss and its real and fake parts; real and fake go through D
        as one batch, one pass instead of two."""
        joint = torch.cat((self._input(y), self._input(fake.detach())))
        losses = self._loss(*_halves(self.train_net(joint, san_training=self.san), y.shape[0]))
        return tuple(loss / self.heads for loss in losses)

    def discriminator_step(self, y, fake, scaler, autocast):
        """``discriminator_losses`` under ``branchwise``: each family's
        backward is run here, into gradients the caller steps on. Returns D's
        loss and its real and fake parts, detached."""
        joint = torch.cat((self._input(y), self._input(fake.detach())))
        totals = [y.new_zeros(()) for _ in range(3)]
        for name, family in self.net.items():
            with autocast():
                real, generated = _halves(family(joint, san_training=self.san), y.shape[0])
                losses = self._loss({name: real}, {name: generated})
            loss = losses[0] / self.heads
            (loss if scaler is None else scaler.scale(loss)).backward()
            totals = [total + part.detach() / self.heads for total, part in zip(totals, losses)]
        return totals

    def generator_gradient(self, y, leaf, adv_scale: float, grad_scale: float):
        """The generator's adversarial (times ``adv_scale``) and
        feature-matching losses through the frozen discriminator, detached,
        and their gradient at ``leaf``, the output as an FP32 leaf. Each
        family's gradient is taken there and its graph freed, for the memory
        of the largest instead of their sum. ``grad_scale`` is the
        GradScaler's, which keeps an FP16 gradient from underflowing."""
        y = self._input(y)
        gradient = torch.zeros_like(leaf)
        loss_adv = loss_fm = y.new_zeros(())
        for family in self.net.values():
            outputs, fmaps = family(self._input(leaf))
            with torch.no_grad():
                _, fmaps_real = family(y)
            adv = self._adversarial(outputs) / self.heads
            fm = feature_loss(fmaps_real, fmaps, self.fm_relative) / self.heads
            (part,) = torch.autograd.grad((adv * adv_scale + fm) * grad_scale, leaf)
            gradient += part.float() / grad_scale
            loss_adv, loss_fm = loss_adv + adv.detach(), loss_fm + fm.detach()
        return loss_adv, loss_fm, gradient


class RollingStats:
    """Means over the last ``size`` steps, under the RVC trainer's names."""

    def __init__(self, size: int):
        self.size = size
        self.values = defaultdict(lambda: deque(maxlen=size))
        self.skips = deque(maxlen=size)

    def scalars(self) -> dict:
        size, scalars = self.size, {}
        if self.skips:
            scalars[f"AMP/skip_rate_{size}"] = sum(self.skips) / len(self.skips)
        for key, queue in self.values.items():
            if queue:
                category = "loss" if key.startswith("loss_") else "grad" if key.startswith("grad_") else "diag"
                scalars[f"{category}_avg_{size}/{key}_{size}"] = torch.stack(list(queue)).mean().item()
        return scalars


def optimizer_step(optimizer, scaler, lr_scale: float) -> None:
    """Step at the warmed-up, plateau-scaled LR, then put the scheduler's
    back: the exponential schedulers scale whatever LR the group holds."""
    scheduled = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] *= lr_scale
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
    for group, lr in zip(optimizer.param_groups, scheduled):
        group["lr"] = lr


def backward(loss, optimizer, parameters, scaler, grad_clip: float, step: int, lr_scale: float):
    """Backward and step; returns the gradient norm on sampled steps. Without
    a ``loss`` the gradients are already there."""
    if loss is not None:
        (loss if scaler is None else scaler.scale(loss)).backward()
    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = clip_or_sample_grad_norm(parameters, grad_clip, step, METRICS_INTERVAL)
    optimizer_step(optimizer, scaler, lr_scale)
    return norm


def measured_conditions(net_g, y, f0) -> tuple:
    """The generator's inputs past the mel and f0 that are measured from the
    target audio: the band contrast, when it reads one."""
    if net_g.contrast is None:
        return ()
    return (net_g.measure_contrast(y, f0),)


def render_preview(net_g, ema, previews, reference, data: dict, device, epoch: int, step: int):
    """Log the reference clip through the EMA weights; ``reference`` is None
    off the main rank. A generator with measured inputs is also rendered
    without them, as inference has none yet."""
    if reference is None:
        return
    mel = normalize_mel(reference.mel.to(device), data)
    f0 = reference.inputs.f0.to(device)
    audio = reference.audio.to(device)
    extra = {}
    with ema.applied(net_g), torch.no_grad():
        conditions = measured_conditions(net_g, audio.reshape(1, -1), f0)
        if getattr(net_g, "ddsp", None) is not None:
            generated, aux = net_g(mel, f0, *conditions, return_aux=True)
            extra["ddsp"] = aux["ddsp"][0]
        else:
            generated = net_g(mel, f0, *conditions)
        if conditions:
            extra["unmeasured"] = net_g(mel, f0)[0]
    previews.log(epoch, step, reference.path, generated, audio, extra_audio=extra)


@torch.no_grad()
def evaluate(net_g, ema, holdout, data: dict, spectral_loss, stft_distance, ranks: Ranks):
    """Held-out mel loss (at the ``c_mel`` scale of ``loss_spectral``) and
    MR-STFT distance through the EMA weights, with the noise drawn from the
    same seeds on every call. Every rank must call it."""
    device = ranks.device
    totals = torch.zeros(2, device=device)
    items = 0
    with ema.applied(net_g):
        for index, (mel, f0, y) in enumerate(holdout):
            mel = normalize_mel(mel.to(device), data)
            y = y.to(device)
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(index)
                f0 = f0.to(device)
                y_hat = net_g(mel, f0, *measured_conditions(net_g, y, f0)).float()
            totals[0] += spectral_loss(y, y_hat) * y.shape[0]
            totals[1] += stft_distance(y_hat, y) * y.shape[0]
            items += y.shape[0]
    return ranks.mean(totals / max(1, items)).tolist()


def export_payload(config: dict, hparams: dict, ema) -> dict:
    """What the inference export holds besides its position in the run: the
    EMA weights, the generator's settings and the mel it renders, tied to its
    normalisation."""
    keys = MEL_KEYS + ("mel_mean", "mel_std")
    return {
        "architecture": recipe_architecture(config["vocoder"]), "recipe": recipe_architecture(config["vocoder"]),
        "config": {"data": {key: config["data"][key] for key in keys}, "vocoder": {"model": hparams}},
        "model": {key: value.detach().float().cpu().contiguous() for key, value in ema.cpu_state_dict().items()},
    }


def save(out_dir: str, name: str, keep: str, epoch: int, step: int, state_g: dict, state_d: dict, exported: dict):
    """Write the G and D checkpoints by the ``keep`` policy ("none", "latest"
    or all of them) and the inference export."""
    saved = []
    with uninterruptible_save("Vocoder checkpoint"):
        if keep == "none":
            # An older run's checkpoints would otherwise be resumed from,
            # silently undoing everything trained since.
            remove_older(out_dir, "G")
            remove_older(out_dir, "D")
        else:
            g_path = os.path.join(out_dir, f"G_{step}.pth")
            d_path = os.path.join(out_dir, f"D_{step}.pth")
            torch.save(state_g, g_path)
            torch.save(state_d, d_path)
            if keep == "latest":
                remove_older(out_dir, "G", g_path)
                remove_older(out_dir, "D", d_path)
            saved.append(os.path.basename(g_path))
        export = os.path.join(out_dir, f"{name}_vocoder_{epoch}e_{step}s.pth")
        torch.save({"kind": "rectified_vocoder", **exported, "epoch": epoch, "step": step}, export)
    saved.append(os.path.basename(export))
    success(f"Saved {' and '.join(saved)}.", tag=TAG)


def main(spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    launch(train, spec_path, parse_gpus(spec.get("gpu", "0")))


def train(ranks: Ranks, spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    ranks.setup()
    main_rank = ranks.main

    name = spec["model_name"]
    config = load_recipe_config(name)
    config["vocoder"] = settings = resolve_discriminator(config["vocoder"])
    d_settings = settings["discriminator"]
    data = config["data"]
    sample_rate = data["sample_rate"]
    out_dir = vocoder_run_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    device = ranks.device
    seed = settings.get("seed")
    if seed is not None:
        # Offset per rank, as ``Ranks.setup`` does; rank 0 keeps the seed.
        random.seed(int(seed) + ranks.rank)
        torch.manual_seed(int(seed) + ranks.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    batch_size = int(spec["batch_size"])
    dataset, loader, sampler, holdout, held_out = build_loaders(name, config, batch_size, ranks)
    eval_interval = int(settings.get("eval_interval", 0)) if holdout else 0

    net_g, net_d, hparams = build_networks(config, device)
    # ``train_post_filter_only``: the rest of G frozen as the pretrained one renders.
    post_filter_only = bool(settings.get("train_post_filter_only", False))
    if post_filter_only:
        if getattr(net_g, "post_filter", None) is None:
            raise ValueError("train_post_filter_only needs a ChouwaGAN with post_filter_channels above 0.")
        net_g.requires_grad_(False)
        net_g.post_filter.requires_grad_(True)
    if settings.get("compile_generator") and device.type == "cuda":
        net_g.compile_blocks()
    learning_rate = float(settings["learning_rate"])
    if spec.get("pretrained_g"):
        learning_rate = float(settings.get("finetune_learning_rate", learning_rate))
    adam = {"betas": tuple(settings["betas"]), "eps": float(settings.get("eps", 1e-9)), "fused": device.type == "cuda"}
    if "weight_decay" in settings:
        adam["weight_decay"] = float(settings["weight_decay"])
    optim_g = torch.optim.AdamW(net_g.parameters(), learning_rate, **adam)
    optim_d = torch.optim.AdamW(net_d.parameters(), learning_rate, **adam)
    # ``filter_weight_decay``: SiFi-GAN's filter network alone decays, as AdamW
    # does, applied after each step so the optimiser keeps one group. Only its
    # gains and biases: a weight-norm direction has no scale to pull down.
    filter_decay = float(settings.get("filter_weight_decay", 0.0)) if isinstance(net_g, sifigan.SiFiGAN) else 0.0
    decayed = [
        parameter for name, parameter in net_g.named_parameters()
        if name.startswith("fn_") and parameter.requires_grad and not name.endswith("original1")
    ] if filter_decay > 0 else []
    amp_dtype, scaler = amp_setup(spec.get("precision", "fp32"), device, TAG)
    plateau, plateau_metric, plateau_interval = build_plateau(settings, eval_interval)
    ema = WeightEMA(net_g, settings["ema_decay"])

    start = restore(
        spec, out_dir, sample_rate, net_g, net_d, optim_g, optim_d, ema, scaler, plateau, plateau_metric,
    )
    epoch, step, skipped = start.epoch, start.step, start.skipped

    # Steps on the spectral losses alone before the discriminator joins, kept
    # in the checkpoint so that a resume does not move that point. Given in
    # epochs, they last as much audio whatever the batch size.
    aux_steps = int(settings.get("aux_steps", 0))
    if settings.get("aux_epochs") is not None:
        aux_steps = round(float(settings["aux_epochs"]) * len(loader))
    if start.resumed:
        aux_steps = int(start.aux_steps if start.aux_steps is not None else aux_steps)
    elif spec.get("pretrained_g"):
        # A pretrained G already renders, and its D would drift out of step with it.
        aux_steps = 0
    else:
        aux_steps = min(aux_steps, int(AUX_MAX_SHARE * int(spec["total_epochs"]) * len(loader)))

    # As in the RVC trainer, plus "cosine annealing step"; "exp decay step"
    # spreads ``lr_decay`` over the epoch's steps. D's schedule starts when
    # it joins: with ``lr_final_ratio`` it decays over the steps it trains,
    # from the base rate.
    lr_scheduler = str(settings.get("lr_scheduler", "exp decay epoch"))
    lr_final_ratio = settings.get("lr_final_ratio")
    schedule = dict(
        exp_decay_gamma=float(settings["lr_decay"]), total_epoch_count=int(spec["total_epochs"]),
        epoch_str=epoch, global_step=step, train_loader=loader, fresh_start=not start.resumed,
        lr_final_ratio=None if lr_final_ratio is None else float(lr_final_ratio),
    )
    step_schedulers = lr_scheduler in STEP_SCHEDULERS
    scheduler_g, _ = prepare_schedulers(optim_g, optim_d, lr_scheduler, "none", **schedule)
    # Counted in steps, the horizon's span takes the partial epoch, so D too
    # ends at the final ratio.
    joined = aux_steps / len(loader) if step_schedulers else aux_steps // len(loader)
    _, scheduler_d = prepare_schedulers(
        optim_g, optim_d, "none", lr_scheduler, **schedule, horizon_start_epoch=joined, horizon_start_step=aux_steps,
    )
    # Linear ramps: the LR of both optimizers and the generator's adversarial
    # weight, the discriminator's from the step it joins. The early phase
    # explodes without them.
    lr_warmup = int(settings.get("warmup_steps", 0))
    adv_warmup = int(settings.get("adv_warmup_steps", 0))

    compile_d = bool(settings.get("compile_discriminator", False))
    enable_discriminator_compile(
        net_d, SimpleNamespace(train=SimpleNamespace(compile_discriminator=compile_d)), device, 0
    )
    # Only the two training forwards go through DDP. The generator update's
    # D pass uses ``net_d`` itself: with D frozen no gradient hook fires, and
    # an armed allreduce would never complete (see rvc/train/train.py).
    train_g = ranks.wrap(net_g)
    train_d = ranks.wrap(net_d)
    adversary = Adversary(net_d, train_d, settings)
    autocast = partial(
        torch.autocast, device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
    )
    spectral_loss = SpectralLoss(config, device)
    stft_distance = MultiScaleSTFTLoss().to(device)
    volume_prob = float(settings.get("volume_aug_prob", 0.0))

    writer = SummaryWriter(os.path.join(out_dir, "eval")) if main_rank else None
    previews = RectifiedPreviews(out_dir, config, step, device) if main_rank else None
    reference = dataset.reference() if main_rank else None
    total_epochs = int(spec["total_epochs"])
    save_every = max(1, int(spec["save_every"]))
    # Only the generator's input: the target, the losses, D's input and the
    # held-out evaluation stay on the real mel.
    degradation = min(1.0, max(0.0, float(spec.get("mel_degradation", 0.0))))
    c_harmonic = float(settings.get("c_harmonic", 0.0))
    # ``c_harmonic_peak``: the same loss's per-harmonic peak term.
    c_harmonic_peak = float(settings.get("c_harmonic_peak", 0.0))
    harmonic_loss = (
        chouwagan.HarmonicLoss(sample_rate, data["hop_length"]).to(device)
        if c_harmonic > 0 or c_harmonic_peak > 0 else None
    )
    c_ddsp = float(settings.get("c_ddsp", 0.0)) if getattr(net_g, "ddsp", None) is not None else 0.0
    c_ddsp_guard = float(settings.get("c_ddsp_guard", 0.0)) if getattr(net_g, "ddsp", None) is not None else 0.0
    c_envelope = float(settings.get("c_envelope", 0.0))
    envelope_loss = None
    if c_envelope > 0:
        envelope_bands = settings.get("envelope_loss_bands_hz") or (3000, 6000, 10000, 16000)
        envelope_loss = chouwagan.EnvelopeLoss(sample_rate, envelope_bands).to(device)
    c_rss = float(settings.get("c_rss", 0.0))
    rss_loss = RandomScaleSpectralLoss(
        settings.get("rss_fft_min", 256), settings.get("rss_fft_max", 2048), settings.get("rss_scales", 4),
    ) if c_rss > 0 else None
    # SiFi-GAN's: the excitation's spectrum against the target's with its envelope flattened.
    c_source = float(settings.get("c_source", 0.0)) if isinstance(net_g, sifigan.SiFiGAN) else 0.0
    source_loss = sifigan.SourceLoss(sample_rate, data["hop_length"]).to(device) if c_source > 0 else None
    grad_clip = float(settings.get("grad_clip") or "inf")
    # ``grad_clip_by_group``: SiFi-GAN's groups each clipped to ``grad_clip``,
    # so a spike at the output conv does not shrink the source's step.
    clip_groups = (
        sifigan_clip_groups(net_g)
        if settings.get("grad_clip_by_group") and isinstance(net_g, sifigan.SiFiGAN) and math.isfinite(grad_clip)
        else None
    )
    # Nagi's pulses locked to the target's fundamental, and their complex spectrum against its.
    phase_lock = isinstance(net_g, nagi.Nagi) and bool(settings.get("phase_lock", False))
    c_locked = float(settings.get("c_locked", 0.0)) if phase_lock else 0.0
    locked_loss = LockedSpectralLoss(
        sample_rate, data["hop_length"], settings["loss_fft_sizes"], settings["loss_hop_sizes"],
        settings["loss_win_lengths"], float(settings.get("locked_max_hz", 6000.0)),
    ).to(device) if c_locked > 0 else None
    # Under the same lock, each band's envelope along the glottal cycle against the target's.
    c_sync = float(settings.get("c_sync", 0.0)) if phase_lock else 0.0
    sync_loss = SyncEnvelopeLoss(
        sample_rate, data["hop_length"],
        settings.get("sync_loss_bands_hz") or (2000, 4000, 6000, 9000, 12000, 16000),
    ).to(device) if c_sync > 0 else None
    # Clips trained without the measured contrast, as inference runs.
    contrast_dropout = float(settings.get("contrast_dropout", 0.0))
    # Clips rendered without Nagi's free branch, so the filtered sources stay whole without it.
    free_dropout = float(settings.get("free_dropout", 0.0)) if getattr(net_g, "free_head", None) is not None else 0.0

    if main_rank:
        print_model_summary(
            [("Generator", net_g), ("Discriminator", net_d)],
            title=f"Rectified vocoder {sample_rate} Hz",
        )
        print_settings_panel(
            [
                ("Model", name),
                ("Vocoder", f"{VOCODER_NAMES[generator_architecture(net_g)]}, "
                            f"{sample_rate} Hz, hop {data['hop_length']}"
                 + (f", {hparams.get('upsampler', 'sinc')} upsampling" if isinstance(net_g, sifigan.SiFiGAN) else "")
                 + (f", n_fft {net_g.n_fft}, {hparams.get('channels', 16)} channels"
                    if isinstance(net_g, wavehax.Wavehax) else "")
                 + (f", control hop {net_g.control_hop}, {net_g.pulse_taps}-tap filters"
                    if isinstance(net_g, nagi.Nagi) else "")),
                ("Clips", f"{len(dataset)} ({len(loader)} steps per epoch)"
                 + (f", {held_out} held out" if held_out else "")),
                ("Batch size", batch_size),
                ("Mel degradation", f"{degradation:g}" if degradation > 0 else "off"),
                ("Volume augmentation", f"{volume_prob:g}" if volume_prob > 0 else "off"),
                ("Epochs", f"{epoch} -> {total_epochs}, saving every {save_every}"),
                ("Starting point", start.starting_point),
                ("PRECISION", precision_label(amp_dtype)),
                ("Seed", "random" if seed is None else int(seed)),
                ("Device", (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
                           + (f" x {ranks.world} GPUs" if ranks.world > 1 else "")),
                ("Optimizer", f"AdamW, lr {learning_rate:g}, {lr_scheduler} {settings['lr_decay']}"
                 + (f", final ratio {lr_final_ratio}" if lr_final_ratio is not None else "")
                 + (f", warmup {lr_warmup} steps" if lr_warmup > 0 else "")
                 + (f", filter weight decay {filter_decay:g}" if decayed else "")
                 + (f", G clipped to {grad_clip:g} per group" if clip_groups else "")
                 + (f", {plateau_metric} plateau x{plateau.factor:g} after {plateau.patience} x"
                    f" {plateau_interval} steps (at x{plateau.scale:g})" if plateau_interval > 0 else "")),
                ("Losses", spectral_loss.label + ", FM x2, adversarial"
                 + (f" ramped over {adv_warmup} steps" if adv_warmup > 0 else "")
                 + (f", spectral only until step {aux_steps}" if step < aux_steps else "")
                 + (f", harmonic contrast and level x{c_harmonic:g}" if c_harmonic > 0 else "")
                 + (f", harmonic peaks x{c_harmonic_peak:g}" if c_harmonic_peak > 0 else "")
                 + (f", band envelope x{c_envelope:g}" if envelope_loss is not None else "")
                 + (f", DDSP mel x{c_ddsp:g}" if c_ddsp > 0 else "")
                 + (f", DDSP guard x{c_ddsp_guard:g}" if c_ddsp_guard > 0 else "")
                 + (f", random-scale spectral x{c_rss:g}" if rss_loss is not None else "")
                 + (f", source x{c_source:g}" if source_loss is not None else "")
                 + (f", locked pulses x{c_locked:g}" if locked_loss is not None else "")
                 + (f", cycle envelope x{c_sync:g}" if sync_loss is not None else "")),
                *([("Band contrast input", f"measured, dropped on {contrast_dropout:g} of clips")]
                  if net_g.contrast is not None else []),
                *([("Phase lock", "pulses on the target's fundamental")] if phase_lock else []),
                *([("Free branch", f"{net_g.free_bands} bands, muted on {free_dropout:g} of clips")]
                  if getattr(net_g, "free_head", None) is not None else []),
                *([("DDSP input", f"{net_g.ddsp.bands} bands for harmonics and noise, "
                                  f"{hparams.get('ddsp_channels', 32)} channels"
                    + (f", mel loss x{c_ddsp:g}" if c_ddsp > 0 else ", no own loss"))]
                  if getattr(net_g, "ddsp", None) is not None else []),
                *([("Post-filter", f"{hparams['post_filter_channels']} channels"
                    + (", anti-aliased" if hparams.get("post_filter_antialias") else "")
                    + (", the only part trained" if post_filter_only else ""))]
                  if getattr(net_g, "post_filter", None) is not None else []),
                *([("Allpass", f"per bin, phase {hparams.get('phase', 'mixed')} under it, mean over "
                               f"{net_g.allpass_smooth_frames} frames")]
                  if getattr(net_g, "allpass_head", None) is not None else []),
                *([("Source extras", ", ".join(name for name, on in (
                    ("cycle alternation", getattr(net_g, "cycle_alternation", False)),
                    ("roughness", getattr(net_g, "roughness", False)),
                    (f"transients ({getattr(net_g, 'transient_fft', 0)}-sample window)",
                     getattr(net_g, "transient_knots", 0)),
                ) if on))] if any((getattr(net_g, "cycle_alternation", False), getattr(net_g, "roughness", False),
                                   getattr(net_g, "transient_knots", 0))) else []),
                *([("Harmonic envelope", "zero-phase FIR, 1.5 pitch periods each side")]
                  if getattr(net_g, "pitch_sync_envelope", False) else []),
                *([("Exciter input", "pulse and noise samples per frame")]
                  if getattr(net_g, "exciter", None) is not None else []),
                *([("Synthesis", f"Wavehax, {hparams.get('synthesis_channels', 32)} channels, "
                    f"{hparams.get('synthesis_blocks', 8)} blocks, bins pooled x{hparams.get('synthesis_pool', 4)}"
                    + (" but the last" if hparams.get("synthesis_pool", 4) > 1 else "")
                    + (", band contrast planes" if net_g.contrast is not None else ""))]
                  if getattr(net_g, "synthesis", None) is not None else []),
                ("Discriminator", adversary.describe(d_settings)),
            ],
            title="Rectified vocoder",
        )

    rolling = int(settings.get("rolling_loss_steps", 50))
    preview_interval = int(settings.get("preview_interval", 500))
    stats = RollingStats(rolling)
    # Held off until the discriminator has joined and both warmups end: the
    # adversarial ramp raises the spectral loss by itself.
    plateau_start = aux_steps + max(lr_warmup, adv_warmup)
    plateau_sum = torch.zeros((), device=device)
    plateau_count = 0

    def checkpoint(finished: int) -> None:
        """Save the run after ``finished`` whole epochs; a resume starts the next."""
        save(
            out_dir, name, spec.get("checkpoints", "latest"), epoch, step,
            {"model": net_g.state_dict(), "optimizer": optim_g.state_dict(),
             "ema": ema.state_dict(), "epoch": finished, "step": step,
             "architecture": generator_architecture(net_g),
             "scaler": scaler.state_dict() if scaler is not None else None,
             "amp_skipped_steps": skipped, "plateau": {**plateau.state_dict(), "metric": plateau_metric},
             "aux_steps": aux_steps},
            {"model": net_d.state_dict(), "optimizer": optim_d.state_dict(),
             "epoch": finished, "step": step, "discriminator": "original"},
            export_payload(config, hparams, ema),
        )

    # ``save_steps``: also inside an epoch, for one too long to lose.
    save_steps = int(spec.get("save_steps") or 0)
    recorder = EpochRecorder()
    net_g.train()
    net_d.train()
    while epoch <= total_epochs:
        metrics = ""
        epoch_sums = defaultdict(lambda: torch.zeros((), device=device))
        epoch_steps = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        with progress_task(
            len(loader), f"Epoch {epoch}/{total_epochs}", training=True, disable=not main_rank
        ) as (progress, task):
            for batch_index, (mel, f0, y) in enumerate(loader):
                mel = mel.to(device, non_blocking=True)
                f0 = f0.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if volume_prob > 0:
                    mel, y = volume_augment(mel, y, volume_prob)
                mel = normalize_mel(mel, data)
                aux_only = step < aux_steps
                lr_scale = ramp(step, lr_warmup) * plateau.scale
                d_lr_scale = ramp(step - aux_steps, lr_warmup) * plateau.scale
                adv_weight = ramp(step - aux_steps, adv_warmup)

                g_mel = degrade_mel(mel, degradation) if degradation > 0 else mel
                # The per-module metrics are measured on the step that writes them.
                measure_modules = (step + 1) % rolling == 0
                if isinstance(net_g, sifigan.SiFiGAN):
                    net_g.track_stream = measure_modules
                conditions = measured_conditions(net_g, y, f0)
                if conditions and contrast_dropout > 0:
                    keep = torch.rand(y.shape[0], 1, 1, device=device) >= contrast_dropout
                    conditions = (conditions[0] * keep,)
                if free_dropout > 0:
                    net_g.free_keep = torch.rand(y.shape[0], 1, device=device) >= free_dropout
                with autocast():
                    if c_ddsp > 0 or c_ddsp_guard > 0:
                        y_hat, aux = train_g(g_mel, f0, *conditions, return_aux=True)
                        ddsp_audio = aux["ddsp"]
                    elif phase_lock:
                        ddsp_audio = None
                        y_hat, pulses = train_g(g_mel, f0, *conditions, target=y, return_harmonic=True)
                    else:
                        ddsp_audio = None
                        y_hat = train_g(g_mel, f0, *conditions)
                if free_dropout > 0:
                    net_g.free_keep = None
                grad_norm_d = None
                if aux_only:
                    loss_d = loss_d_real = loss_d_fake = y.new_zeros(())
                elif adversary.branchwise:
                    optim_d.zero_grad(set_to_none=True)
                    loss_d, loss_d_real, loss_d_fake = adversary.discriminator_step(y, y_hat, scaler, autocast)
                    grad_norm_d = backward(None, optim_d, net_d.parameters(), scaler, grad_clip, step, d_lr_scale)
                    normalize_san_weights(net_d, optim_d)
                else:
                    with autocast():
                        loss_d, loss_d_real, loss_d_fake = adversary.discriminator_losses(y, y_hat)
                    optim_d.zero_grad(set_to_none=True)
                    grad_norm_d = backward(loss_d, optim_d, net_d.parameters(), scaler, grad_clip, step, d_lr_scale)
                    # A no-op without SAN heads.
                    normalize_san_weights(net_d, optim_d)

                # The generator update reads D and never trains it: frozen, its
                # backward skips D's weight gradients and still reaches y_hat.
                # The losses on the output are taken on a leaf of it, their
                # gradients summed there (D's capped by ``adv_balance``) and
                # sent through G once; each loss's graph is freed as it goes.
                net_d.requires_grad_(False)
                leaf = y_hat.detach().float().requires_grad_(True)
                with autocast():
                    loss_mel = spectral_loss(y, leaf)
                    loss_rec = loss_mel
                    if ddsp_audio is not None and c_ddsp_guard > 0:
                        # One-sided per band: never worse than the DDSP render,
                        # but do not pull the output down where DDSP is worse.
                        error_wave = spectral_loss.mel_error(y, leaf)
                        error_ddsp = spectral_loss.mel_error(y, ddsp_audio.detach())
                        loss_ddsp_guard = F.relu(error_wave - error_ddsp).mean() * c_ddsp_guard
                        loss_rec = loss_rec + loss_ddsp_guard
                    if harmonic_loss is not None:
                        bands, peaks = harmonic_loss(leaf, y, f0, peaks=True)
                        loss_harmonic, loss_harmonic_peak = bands * c_harmonic, peaks * c_harmonic_peak
                        loss_rec = loss_rec + loss_harmonic + loss_harmonic_peak
                    if envelope_loss is not None:
                        loss_envelope = envelope_loss(leaf, y) * c_envelope
                        loss_rec = loss_rec + loss_envelope
                    if sync_loss is not None:
                        loss_sync = sync_loss(leaf, y, net_g.lock_turns, f0) * c_sync
                        loss_rec = loss_rec + loss_sync
                    if rss_loss is not None:
                        loss_rss = rss_loss(leaf.squeeze(1), y.squeeze(1)) * c_rss
                        loss_rec = loss_rec + loss_rss
                (gradient,) = torch.autograd.grad(loss_rec, leaf)
                balance = y.new_ones(())
                if aux_only:
                    loss_fm = loss_adv = y.new_zeros(())
                else:
                    with autocast(), (
                        torch.nn.utils.parametrize.cached() if adversary.cache_weights else nullcontext()
                    ):
                        loss_adv, loss_fm, gradient_d = adversary.generator_gradient(
                            y, leaf, adv_weight, 1.0 if scaler is None else scaler.get_scale()
                        )
                    if adversary.adv_balance > 0:
                        balance = adversary.adv_balance * gradient.norm() / gradient_d.norm().clamp_min(1e-12)
                        balance = balance.clamp(max=1.0)
                    gradient = gradient + gradient_d * balance
                loss_g = (y_hat.float() * gradient).sum()
                loss_total = loss_rec.detach() + loss_fm + loss_adv * adv_weight
                # The losses on G's other outputs stay in the graph.
                if ddsp_audio is not None:
                    loss_ddsp = spectral_loss.mel_l1(y, ddsp_audio) * c_ddsp
                    loss_g, loss_total = loss_g + loss_ddsp, loss_total + loss_ddsp.detach()
                if source_loss is not None:
                    loss_source = source_loss(net_g.excitation, y, f0) * c_source
                    loss_g, loss_total = loss_g + loss_source, loss_total + loss_source.detach()
                if locked_loss is not None:
                    loss_locked = locked_loss(pulses, y, f0) * c_locked
                    loss_g, loss_total = loss_g + loss_locked, loss_total + loss_locked.detach()
                optim_g.zero_grad(set_to_none=True)
                if scaler is None:
                    loss_g.backward()
                else:
                    scaler.scale(loss_g).backward()
                    scaler.unscale_(optim_g)
                module_metrics = generator_gradient_metrics(net_g) if measure_modules else {}
                if clip_groups:
                    grad_norm_g = clip_groups_grad_norm(clip_groups, grad_clip)
                else:
                    grad_norm_g = clip_or_sample_grad_norm(net_g.parameters(), grad_clip, step, METRICS_INTERVAL)
                optimizer_step(optim_g, scaler, lr_scale)
                if decayed:
                    with torch.no_grad():
                        torch._foreach_mul_(decayed, 1 - optim_g.param_groups[0]["lr"] * lr_scale * filter_decay)
                net_d.requires_grad_(True)
                if scaler is not None:
                    scale = scaler.get_scale()
                    scaler.update()
                    overflowed = scaler.get_scale() < scale
                    skipped += int(overflowed)
                    stats.skips.append(float(overflowed))
                    if not overflowed and scaler.get_scale() > MAX_GRAD_SCALE:
                        scaler.update(MAX_GRAD_SCALE)
                ema.update(net_g)
                if step_schedulers:
                    scheduler_g.step()
                    if not aux_only:
                        scheduler_d.step()
                step += 1

                losses = {
                    "loss_disc": loss_d, "loss_disc_real": loss_d_real, "loss_disc_fake": loss_d_fake,
                    "loss_adv": loss_adv, "loss_gen_total": loss_total, "loss_fm": loss_fm,
                    "loss_spectral": loss_mel, **spectral_loss.terms,
                }
                if harmonic_loss is not None:
                    losses["loss_harmonic"] = loss_harmonic
                    if c_harmonic_peak > 0:
                        losses["loss_harmonic_peak"] = loss_harmonic_peak
                if adversary.adv_balance > 0:
                    losses["adv_scale"] = balance
                if ddsp_audio is not None and c_ddsp > 0:
                    losses["loss_ddsp"] = loss_ddsp
                if ddsp_audio is not None and c_ddsp_guard > 0:
                    losses["loss_ddsp_guard"] = loss_ddsp_guard
                if envelope_loss is not None:
                    losses["loss_envelope"] = loss_envelope
                if sync_loss is not None:
                    losses["loss_sync"] = loss_sync
                if rss_loss is not None:
                    losses["loss_rss"] = loss_rss
                if source_loss is not None:
                    losses["loss_source"] = loss_source
                if locked_loss is not None:
                    losses["loss_locked"] = loss_locked
                for key, value in losses.items():
                    stats.values[key].append(value.detach().float())
                    epoch_sums[key] += value.detach().float()
                epoch_steps += 1
                if plateau_metric == "train" and plateau_interval > 0 and step > plateau_start:
                    plateau_sum += loss_mel.detach().float()
                    plateau_count += 1
                    # Counted rather than ``step % interval``, so every rank
                    # reaches the reduction together and a resume starts a
                    # whole interval.
                    if plateau_count == plateau_interval:
                        value = ranks.mean(plateau_sum / plateau_count).item()
                        plateau_sum.zero_()
                        plateau_count = 0
                        plateau_update(plateau, plateau_metric, value, writer, step)
                if eval_interval and step % eval_interval == 0:
                    val_mel, val_stft = evaluate(net_g, ema, holdout, data, spectral_loss, stft_distance, ranks)
                    if main_rank:
                        writer.add_scalar("val/mel", val_mel, step)
                        writer.add_scalar("val/mrstft", val_stft, step)
                    if plateau_metric == "val" and step > plateau_start:
                        plateau_update(plateau, plateau_metric, val_mel, writer, step)
                for key, value in (("grad_norm_d", grad_norm_d), ("grad_norm_g", grad_norm_g)):
                    if value is None:
                        continue
                    if torch.isfinite(value):
                        stats.values[key].append(value.detach().float())
                    elif main_rank:
                        writer.add_scalar(f"Grad_Norm_Diag/{key[-1].upper()}_Skipped", 1, step)

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        metrics = f"G={loss_total.item():.4f}  " + (
                            f"D=off until step {aux_steps}" if aux_only else f"D={loss_d.item():.4f}"
                        )
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)
                    if step % rolling == 0:
                        scalars = {
                            "learning_rate/lr_d": optim_d.param_groups[0]["lr"] * plateau.scale,
                            "learning_rate/lr_g": optim_g.param_groups[0]["lr"] * plateau.scale,
                            **stats.scalars(),
                            **{
                                f"{'stream' if key.startswith('stream_') else 'grad'}/{key}": value.item()
                                for key, value in module_metrics.items() if torch.isfinite(value)
                            },
                        }
                        if plateau_interval > 0:
                            scalars["learning_rate/plateau_scale"] = plateau.scale
                        if lr_warmup > 0 or adv_warmup > 0:
                            scalars["learning_rate/warmup"] = ramp(step, lr_warmup)
                            scalars["learning_rate/warmup_d"] = ramp(step - aux_steps, lr_warmup)
                            scalars["diag/adv_weight"] = ramp(step - aux_steps, adv_warmup)
                        if scaler is not None:
                            scalars["AMP/grad_scaler_scale"] = scaler.get_scale()
                            scalars["AMP/skipped_steps_total"] = skipped
                        for key, value in scalars.items():
                            writer.add_scalar(key, value, step)
                    if step % preview_interval == 0:
                        render_preview(net_g, ema, previews, reference, data, device, epoch, step)
                if main_rank and save_steps and step % save_steps == 0:
                    checkpoint(epoch - 1)
                if ranks.stop_requested():
                    # Nothing is being written at a batch boundary.
                    finish_stop(writer)

        # Averaged over the GPUs; every rank takes part in the reduction.
        epoch_means = {key: ranks.mean(total / max(1, epoch_steps)) for key, total in sorted(epoch_sums.items())}
        if main_rank:
            print(f"{name} | epoch={epoch} | step={step} | {recorder.record()}")
            if epoch_steps:
                for key, mean in epoch_means.items():
                    writer.add_scalar(f"{'loss_avg' if key.startswith('loss_') else 'diag'}/{key}", mean.item(), step)
                writer.add_scalar("learning_rate/lr_d", optim_d.param_groups[0]["lr"] * plateau.scale, step)
                writer.add_scalar("learning_rate/lr_g", optim_g.param_groups[0]["lr"] * plateau.scale, step)
        if not step_schedulers:
            for scheduler in (scheduler_g, scheduler_d if step > aux_steps else None):
                if scheduler is not None:
                    scheduler.step()
        if main_rank:
            if skipped and scaler is not None:
                info(f"GradScaler at {scaler.get_scale():.0f}; {skipped} step(s) skipped so far.", tag=TAG)
            if epoch % save_every == 0 or epoch == total_epochs:
                checkpoint(epoch)
                render_preview(net_g, ema, previews, reference, data, device, epoch, step)
            writer.flush()
        epoch += 1

    if main_rank:
        writer.close()
        success("Vocoder training finished.", tag=TAG)


if __name__ == "__main__":
    main(sys.argv[1])
