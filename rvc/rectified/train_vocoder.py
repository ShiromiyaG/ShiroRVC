"""Train ChouwaGAN, the mel vocoder of the rectified pipeline, by its recipe
(``rvc/configs/rectified/chouwagan/<sample rate>.json``, copied into the
experiment on first use).

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
from rvc.rectified import chouwagan
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
from rvc.rectified.vocoder import MEL_KEYS, load_recipe_config
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


def generator_gradient_metrics(net_g):
    """``rvc.train.diagnostics.generator_gradient_metrics`` for a bare vocoder,
    whose parameters carry no ``dec.`` prefix: the trunk, and the output
    ``head`` kept apart as ``output``."""
    groups = {"decoder": [], "output": []}
    for name, parameter in net_g.named_parameters():
        if parameter.requires_grad:
            groups["output" if name.startswith("head.") else "decoder"].append(parameter)
    metrics = {}
    for group, parameters in groups.items():
        if not parameters:
            continue
        parameter_norm = torch.sqrt(sum(p.detach().float().square().sum() for p in parameters))
        terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
        gradient_norm = torch.sqrt(sum(terms)) if terms else parameter_norm.new_zeros(())
        metrics[f"grad_norm_{group}"] = gradient_norm
        metrics[f"grad_to_param_{group}"] = gradient_norm / parameter_norm.clamp_min(1e-8)
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
    net_g = chouwagan.ChouwaGAN(**hparams).to(device)
    net_d = Discriminators(chouwagan.build_families(settings["discriminator"], data["sample_rate"])).to(device)
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
        saved = (state_g.get("architecture", chouwagan.ARCHITECTURE), state_d.get("discriminator", "original"))
        if saved != (chouwagan.ARCHITECTURE, "original"):
            raise ValueError(
                f"{resume_g} is a {saved[0]} checkpoint with the {saved[1]} discriminator, not ChouwaGAN with "
                f"its own. Turn on Fresh Training or use another model name."
            )
        net_g.load_state_dict(state_g["model"])
        optim_g.load_state_dict(state_g["optimizer"])
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

    def __call__(self, y, y_hat):
        loss_mel = F.l1_loss(self.mel(y_hat.squeeze(1)), self.mel(y.squeeze(1))) * self.weight
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


def feature_loss(fmaps_real, fmaps_fake) -> torch.Tensor:
    """Feature matching over every layer of every head, x2."""
    loss = 0.0
    for head_real, head_fake in zip(fmaps_real, fmaps_fake):
        for real, fake in zip(head_real, head_fake):
            loss = loss + torch.mean(torch.abs(real.float() - fake.float()))
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
        # ``branchwise_discriminator``: the families one at a time, each one's
        # graph freed before the next, for the memory of the largest instead
        # of their sum; the gradients are the same. On one GPU only: the
        # gradient sync wraps the whole network.
        self.branchwise = bool(settings.get("branchwise_discriminator", False)) and train_d is net_d
        self.san = net_d.supports_san
        # ``san_loss`` "bigvsan": BigVSAN's discriminator loss on the SAN
        # heads, in place of the repo's bounded mirror of it.
        self.bigvsan = self.san and settings.get("san_loss", "repo") == "bigvsan"
        # The generator update reads D's frozen weights in two passes, real
        # then fake: cached, each weight norm is built once. Not compiled,
        # where the graph already fuses it.
        self.cache_weights = not getattr(net_d, "_compile_enabled", False)

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
        ) + (", compiled" if getattr(net, "_compile_enabled", False) else "")

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
        """D's loss and its real and fake parts."""
        losses = self._loss(
            self.train_net(y, san_training=self.san), self.train_net(fake.detach(), san_training=self.san)
        )
        return tuple(loss / self.heads for loss in losses)

    def discriminator_step(self, y, fake, scaler, autocast):
        """``discriminator_losses`` under ``branchwise``: each family's
        backward is run here, into gradients the caller steps on. Returns D's
        loss and its real and fake parts, detached."""
        fake = fake.detach()
        totals = [y.new_zeros(()) for _ in range(3)]
        for name, family in self.net.items():
            with autocast():
                if self.san:
                    real, generated = family(y, san_training=True), family(fake, san_training=True)
                else:
                    real, generated = family(y), family(fake)
                losses = self._loss({name: real}, {name: generated})
            loss = losses[0] / self.heads
            (loss if scaler is None else scaler.scale(loss)).backward()
            totals = [total + part.detach() / self.heads for total, part in zip(totals, losses)]
        return totals

    def branchwise_generator_losses(self, y, fake, adv_scale: float, grad_scale: float):
        """``generator_losses`` under ``branchwise``. Each family's gradient
        is taken at ``fake`` and its graph freed; their sum reaches the
        generator through one product with ``fake``. The losses carry their
        values, and the feature-matching one that gradient, the adversarial
        loss's times ``adv_scale`` included. ``grad_scale`` is the
        GradScaler's, which keeps an FP16 gradient from underflowing."""
        leaf = fake.detach().requires_grad_(True)
        gradient = torch.zeros_like(leaf, dtype=torch.float32)
        loss_adv = loss_fm = y.new_zeros(())
        for family in self.net.values():
            outputs, fmaps = family(leaf)
            with torch.no_grad():
                _, fmaps_real = family(y)
            adv = self._adversarial(outputs) / self.heads
            fm = feature_loss(fmaps_real, fmaps) / self.heads
            (part,) = torch.autograd.grad((adv * adv_scale + fm) * grad_scale, leaf)
            gradient += part.float() / grad_scale
            loss_adv, loss_fm = loss_adv + adv.detach(), loss_fm + fm.detach()
        surrogate = (fake.float() * gradient).sum()
        return loss_adv, loss_fm + surrogate - surrogate.detach()

    def generator_losses(self, y, fake, adv_scale: float = 1.0, grad_scale: float = 1.0):
        """The generator's adversarial and feature-matching losses through
        the frozen discriminator. ``adv_scale`` and ``grad_scale`` are
        ``branchwise_generator_losses``'."""
        if self.branchwise:
            return self.branchwise_generator_losses(y, fake, adv_scale, grad_scale)
        d_fake = self.net(fake)
        with torch.no_grad():
            d_real = self.net(y)
        loss_adv = sum(self._adversarial(outputs) for outputs, _ in d_fake.values())
        loss_fm = sum(feature_loss(d_real[family][1], d_fake[family][1]) for family in d_fake)
        return loss_adv / self.heads, loss_fm / self.heads


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
                category = "loss" if key.startswith("loss_") else "grad"
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
    extra = None
    with ema.applied(net_g), torch.no_grad():
        conditions = measured_conditions(net_g, audio.reshape(1, -1), f0)
        generated = net_g(mel, f0, *conditions)
        if conditions:
            extra = {"unmeasured": net_g(mel, f0)[0]}
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
        "architecture": chouwagan.ARCHITECTURE, "recipe": chouwagan.ARCHITECTURE,
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
    settings = config["vocoder"]
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
    if settings.get("compile_generator") and device.type == "cuda":
        net_g.compile_blocks()
    learning_rate = float(settings["learning_rate"])
    if spec.get("pretrained_g"):
        learning_rate = float(settings.get("finetune_learning_rate", learning_rate))
    adam = {"betas": tuple(settings["betas"]), "eps": float(settings.get("eps", 1e-9))}
    if "weight_decay" in settings:
        adam["weight_decay"] = float(settings["weight_decay"])
    optim_g = torch.optim.AdamW(net_g.parameters(), learning_rate, **adam)
    optim_d = torch.optim.AdamW(net_d.parameters(), learning_rate, **adam)
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
    harmonic_loss = chouwagan.HarmonicLoss(sample_rate, data["hop_length"]).to(device) if c_harmonic > 0 else None
    c_envelope = float(settings.get("c_envelope", 0.0))
    envelope_loss = None
    if c_envelope > 0:
        envelope_bands = settings.get("envelope_loss_bands_hz") or (3000, 6000, 10000, 16000)
        envelope_loss = chouwagan.EnvelopeLoss(sample_rate, envelope_bands).to(device)
    grad_clip = float(settings.get("grad_clip") or "inf")
    # Clips trained without the measured contrast, as inference runs.
    contrast_dropout = float(settings.get("contrast_dropout", 0.0))

    if main_rank:
        print_model_summary(
            [("Generator", net_g), ("Discriminator", net_d)],
            title=f"Rectified vocoder {sample_rate} Hz",
        )
        print_settings_panel(
            [
                ("Model", name),
                ("Vocoder", f"ChouwaGAN, {sample_rate} Hz, hop {data['hop_length']}"),
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
                 + (f", {plateau_metric} plateau x{plateau.factor:g} after {plateau.patience} x"
                    f" {plateau_interval} steps (at x{plateau.scale:g})" if plateau_interval > 0 else "")),
                ("Losses", spectral_loss.label + ", FM x2, adversarial"
                 + (f" ramped over {adv_warmup} steps" if adv_warmup > 0 else "")
                 + (f", spectral only until step {aux_steps}" if step < aux_steps else "")
                 + (f", harmonic contrast and level x{c_harmonic:g}" if harmonic_loss is not None else "")
                 + (f", band envelope x{c_envelope:g}" if envelope_loss is not None else "")),
                *([("Band contrast input", f"measured, dropped on {contrast_dropout:g} of clips")]
                  if net_g.contrast is not None else []),
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
             "architecture": chouwagan.ARCHITECTURE,
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
                conditions = measured_conditions(net_g, y, f0)
                if conditions and contrast_dropout > 0:
                    keep = torch.rand(y.shape[0], 1, 1, device=device) >= contrast_dropout
                    conditions = (conditions[0] * keep,)
                with autocast():
                    y_hat = train_g(g_mel, f0, *conditions)
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
                net_d.requires_grad_(False)
                with autocast(), (
                    torch.nn.utils.parametrize.cached() if adversary.cache_weights else nullcontext()
                ):
                    if aux_only:
                        loss_fm = loss_adv = y.new_zeros(())
                    else:
                        loss_adv, loss_fm = adversary.generator_losses(
                            y, y_hat, adv_scale=adv_weight,
                            grad_scale=1.0 if scaler is None else scaler.get_scale(),
                        )
                    loss_mel = spectral_loss(y, y_hat)
                    loss_g = loss_mel + loss_fm + loss_adv * adv_weight
                if harmonic_loss is not None:
                    loss_harmonic = harmonic_loss(y_hat, y, f0) * c_harmonic
                    loss_g = loss_g + loss_harmonic
                if envelope_loss is not None:
                    loss_envelope = envelope_loss(y_hat, y) * c_envelope
                    loss_g = loss_g + loss_envelope
                optim_g.zero_grad(set_to_none=True)
                if scaler is None:
                    loss_g.backward()
                else:
                    scaler.scale(loss_g).backward()
                    scaler.unscale_(optim_g)
                module_metrics = generator_gradient_metrics(net_g) if step % METRICS_INTERVAL == 0 else {}
                grad_norm_g = clip_or_sample_grad_norm(net_g.parameters(), grad_clip, step, METRICS_INTERVAL)
                optimizer_step(optim_g, scaler, lr_scale)
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
                    "loss_adv": loss_adv, "loss_gen_total": loss_g, "loss_fm": loss_fm,
                    "loss_spectral": loss_mel, **spectral_loss.terms,
                }
                if harmonic_loss is not None:
                    losses["loss_harmonic"] = loss_harmonic
                if envelope_loss is not None:
                    losses["loss_envelope"] = loss_envelope
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
                for key, value in module_metrics.items():
                    if torch.isfinite(value):
                        stats.values[key].append(value.detach().float())

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        metrics = f"G={loss_g.item():.4f}  " + (
                            f"D=off until step {aux_steps}" if aux_only else f"D={loss_d.item():.4f}"
                        )
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)
                    if step % rolling == 0:
                        scalars = {
                            "learning_rate/lr_d": optim_d.param_groups[0]["lr"] * plateau.scale,
                            "learning_rate/lr_g": optim_g.param_groups[0]["lr"] * plateau.scale,
                            **stats.scalars(),
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
                    writer.add_scalar(f"loss_avg/{key}", mean.item(), step)
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
