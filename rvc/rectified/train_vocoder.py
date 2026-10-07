"""Train the mel vocoder of the rectified pipeline: PCPH-BigVGAN, or one of
the SingingVocoders architectures by its own recipe (``singing_vocoders.py``).

Usage: python rvc/rectified/train_vocoder.py <spec.json>
"""

import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

sys.path.append(os.getcwd())

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.lib.terminal import (
    info,
    print_model_summary,
    print_settings_panel,
    progress_task,
    success,
)
from rvc.rectified.common import (
    RectifiedDataset,
    amp_setup,
    collate_vocoder,
    latest_checkpoint,
    load_run_config,
    precision_label,
    pretrained_weights,
    read_filelist,
    run_dir,
    remove_older,
    split_holdout,
)
from rvc.rectified.distributed import Ranks, launch, parse_gpus
from rvc.rectified import singing_vocoders as sv
from rvc.rectified.mel import LogMel, degrade_mel, denormalize_mel, normalize_mel
from rvc.rectified.previews import RectifiedPreviews
from rvc.rectified.vocoder import MEL_KEYS, build_discriminator, build_vocoder
from rvc.train.balance import FamilyReducer, head_accuracies
from rvc.train.diagnostics import branch_separation, clip_or_sample_grad_norm
from rvc.train.ema import WeightEMA
from rvc.train.losses import (
    MultiScaleSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    loud_crop,
    r1_penalty,
)
from rvc.train.mel_processing import build_ms_mel_loss
from rvc.train.progress import EpochRecorder, emit_machine_progress
from rvc.train.schedules import prepare_schedulers
from rvc.train.setup import (
    apply_precision_policy,
    enable_discriminator_compile,
    loader_workers,
    normalize_san_weights,
)
from rvc.train.stop import finish_stop, install_stop_handlers, uninterruptible_save

TAG = "[VOCODER]"
METRICS_INTERVAL = 8
ACCURACY_SAMPLE_EVERY = 4
SAN_DIRECTION_WEIGHT = 0.25
#: Ceiling of the FP16 GradScaler's scale: left to grow, it doubles until a
#: gradient overflows and a step is skipped.
MAX_GRAD_SCALE = 2.0**16


def generator_gradient_metrics(net_g):
    """``rvc.train.diagnostics.generator_gradient_metrics`` for a bare vocoder,
    whose parameters carry no ``dec.`` prefix: the decoder, and ``conv_post``
    kept apart as ``output``."""
    groups = {"decoder": [], "output": []}
    for name, parameter in net_g.named_parameters():
        if parameter.requires_grad:
            groups["output" if name.startswith("conv_post.") else "decoder"].append(parameter)
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
    return 1.0 if length <= 0 else min(1.0, (step + 1) / length)


def generator_input(mel, data: dict, raw_mel: bool):
    """The normalised mel as the generator takes it: SingingVocoders'
    are trained on the raw log mel."""
    return denormalize_mel(mel, data) if raw_mel else mel


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


def build_networks(architecture: str, config: dict, d_settings: dict, own_d: bool, device):
    """The generator, the discriminator and, for a recipe architecture, the
    generator's hyperparameters (else None)."""
    hparams = None
    if architecture != sv.PCPH:
        hparams = sv.generator_hparams(architecture, config)
        net_g = sv.build_generator(architecture, hparams).to(device)
    else:
        net_g = build_vocoder(config).to(device)
    if own_d:
        net_d = sv.build_discriminators(architecture, d_settings).to(device)
    else:
        net_d = build_discriminator({"data": config["data"], "vocoder": {"discriminator": d_settings}}).to(device)
    return net_g, net_d, hparams


def restore(spec, out_dir, architecture, discriminator, net_g, net_d, optim_g, optim_d, ema, scaler,
            plateau, plateau_metric):
    """Resume the run's latest checkpoints, or start from the pretrained
    weights of ``spec``. Returns where training starts: ``epoch``, ``step``,
    ``skipped``, ``mel_cond_since`` (the step at which the discriminator
    started taking the mel), ``resumed`` and a ``starting_point`` label."""
    singing = architecture != sv.PCPH
    own_d = discriminator == "original"
    start = SimpleNamespace(epoch=1, step=0, skipped=0, mel_cond_since=0, resumed=False, starting_point="scratch")
    resume_g = None if spec.get("fresh") else latest_checkpoint(out_dir, "G")
    resume_d = None if spec.get("fresh") else latest_checkpoint(out_dir, "D")
    if resume_g and resume_d:
        state_g = torch.load(resume_g, map_location="cpu", weights_only=True)
        state_d = torch.load(resume_d, map_location="cpu", weights_only=True)
        saved = (state_g.get("architecture", sv.PCPH), state_d.get("discriminator", discriminator))
        if saved != (architecture, discriminator):
            raise ValueError(
                f"{resume_g} is a {saved[0]} checkpoint with the {saved[1]} discriminator, not "
                f"{architecture} with {discriminator}. Turn on Fresh Training or use another model name."
            )
        if singing:
            # Checkpoints written under the older weight norm's names.
            state_g["model"] = sv.training_state(state_g["model"], net_g.state_dict())
            if state_g.get("ema"):
                state_g["ema"]["shadow"] = sv.training_state(state_g["ema"]["shadow"], net_g.state_dict())
            if own_d:
                state_d["model"] = sv.training_state(state_d["model"], net_d.state_dict())
        # A D saved before mel conditioning loads with a fresh mel embedding;
        # its optimizer state has no slots for those parameters.
        upgraded_d = not set(net_d.state_dict()) <= set(state_d["model"])
        net_g.load_state_dict(state_g["model"])
        net_d.load_state_dict(state_d["model"])
        optim_g.load_state_dict(state_g["optimizer"])
        if upgraded_d:
            info("D gains the mel projection; its optimizer state starts fresh.", tag=TAG)
            # At the base LR otherwise; it follows G's schedule.
            for group in optim_d.param_groups:
                group["lr"] = optim_g.param_groups[0]["lr"]
        else:
            optim_d.load_state_dict(state_d["optimizer"])
        if "ema" in state_g:
            ema.load_state_dict(state_g["ema"], net_g)
        if scaler is not None and state_g.get("scaler"):
            scaler.load_state_dict(state_g["scaler"])
        start.epoch, start.step = state_g["epoch"] + 1, state_g["step"]
        saved_since = None if upgraded_d else state_d.get("mel_cond_since")
        start.mel_cond_since = start.step if saved_since is None else saved_since
        start.skipped = int(state_g.get("amp_skipped_steps", 0))
        # A best from the other metric is on another scale; that one restarts.
        if state_g.get("plateau") and state_g["plateau"].get("metric", "train") == plateau_metric:
            plateau.load_state_dict(state_g["plateau"])
        start.resumed = True
        start.starting_point = f"resumed from {os.path.basename(resume_g)}"
        return start
    # A SingingVocoders checkpoint holds both networks; its discriminator
    # only fits the recipe's own.
    d_weights, d_source = None, spec.get("pretrained_d")
    if spec.get("pretrained_g"):
        if singing:
            g_weights, d_weights = sv.pretrained_states(spec["pretrained_g"])
            sv.load_pretrained_generator(net_g, g_weights)
        else:
            net_g.load_state_dict(pretrained_weights(spec["pretrained_g"]))
        ema.reseed(net_g)
        start.starting_point = f"G from {spec['pretrained_g']}"
    if not own_d:
        d_weights = pretrained_weights(d_source) if d_source else None
    elif d_source:
        weights, d_weights = sv.pretrained_states(d_source)
        d_weights = d_weights or weights
    if d_weights:
        net_d.load_state_dict(sv.training_state(d_weights, net_d.state_dict()) if own_d else d_weights)
        start.starting_point += f", D from {d_source or spec['pretrained_g']}"
    return start


class SpectralLoss:
    """The architecture's reconstruction loss, at its weight."""

    def __init__(self, architecture: str, config: dict, device):
        data, settings = config["data"], config["vocoder"]
        sample_rate = data["sample_rate"]
        self.stft = self.mel = self.ms_mel = None
        if architecture == sv.UNIVNET:
            self.stft = sv.MultiResolutionSTFTLoss(
                settings["loss_fft_sizes"], settings["loss_hop_sizes"], settings["loss_win_lengths"]
            ).to(device)
            self.weight = settings["c_stft"]
            self.label = f"MR-STFT x{self.weight:g}"
        elif architecture != sv.PCPH:
            # The recipe's loss mel: the input's, read up to Nyquist, unless
            # the recipe sizes it.
            self.mel = LogMel(
                sample_rate, data["n_fft"], data["win_length"], data["hop_length"],
                settings.get("loss_n_mels", data["n_mels"]), settings.get("loss_fmin", data["mel_fmin"]),
                sample_rate / 2,
            ).to(device)
            self.weight = settings["c_mel"]
            self.label = f"mel x{self.weight:g}"
        else:
            self.ms_mel = build_ms_mel_loss(sample_rate).to(device)
            self.weight = settings["c_mel"]
            self.label = f"multi-scale mel x{self.weight:g}"

    def __call__(self, y, y_hat):
        if self.stft is not None:
            sc_loss, mag_loss = self.stft(y_hat.squeeze(1), y.squeeze(1))
            return (sc_loss + mag_loss) * self.weight
        if self.mel is not None:
            return F.l1_loss(self.mel(y_hat.squeeze(1)), self.mel(y.squeeze(1))) * self.weight
        return self.ms_mel(y.float(), y_hat.float()) * self.weight


class PitchControl:
    """SingingVocoders' pitch-controllable augmentation, over the first
    ``count`` clips of a batch (``pc_aug_rate`` of it; 0 turns it off)."""

    def __init__(self, net_g, config: dict, batch_size: int, device):
        settings = config["vocoder"]
        rate = float(settings.get("pc_aug_rate", 0.0))
        self.key = float(settings.get("pc_aug_key", 5))
        self.count = math.ceil(batch_size * rate) if rate > 0 else 0
        if self.count and not getattr(net_g, "mini_nsf", False):
            raise ValueError("Pitch-controllable augmentation (pc_aug_rate) needs a mini_nsf generator.")
        # The sine is made at the mini-NSF's rate; above its Nyquist it aliases.
        self.max_f0 = getattr(net_g, "source_sr", config["data"]["sample_rate"]) / 2
        self.input_mel = LogMel.from_config(config["data"]).to(device) if self.count else None
        self.device = device

    def __call__(self, generator, mel, f0):
        """One pass of ``generator`` over the raw ``mel``. The first ``count``
        clips are rendered at a shifted pitch c and brought back from the mel
        of that render; they are also rendered at a pitch a, and from that mel
        at c. Returns the batch (shifted clips brought back) and the renders
        at c, at a, and at c through a."""
        count, key = self.count, self.key
        low = f0[:count]
        key_c = (2 * torch.rand(count, 1, device=self.device) - 1) * key
        f0_c = torch.clamp(low * 2 ** (key_c / 12), max=self.max_f0)
        mixed = generator(mel, torch.cat((f0_c, f0[count:])))
        shift_c, plain = mixed[:count], mixed[count:]
        back = generator(self.input_mel(shift_c.squeeze(1)), low)
        key_a_max = key + key_c.clamp(max=0)
        key_a_min = -key + key_c.clamp(min=0)
        key_a = (key_a_max - key_a_min) * torch.rand(count, 1, device=self.device) + key_a_min
        f0_a = torch.clamp(low * 2 ** (key_a / 12), max=self.max_f0)
        shift_a = generator(mel[:count], f0_a)
        shift_ab = generator(self.input_mel(shift_a.squeeze(1)), f0_c)
        return torch.cat((back, plain)), shift_c, shift_a, shift_ab


class Adversary:
    """The discriminator and the GAN losses it trains by: the recipe's own
    (``own``), or the repo's, with SAN, branch weights, R1 and mel
    conditioning as the network declares them. ``train_d`` is the network as
    the discriminator update calls it (DDP-wrapped)."""

    def __init__(self, net_d, train_d, settings: dict, d_settings: dict, own: bool, univnet: bool, amp_dtype):
        self.net = net_d
        self.train_net = train_d
        self.own = own
        self.univnet = univnet
        self.hinge = settings.get("gan_loss", "lsgan") == "hinge"
        self.c_fm = None if own else settings["c_fm"]
        self.labels = tuple(getattr(net_d, "branch_labels", ()))
        self.san = bool(getattr(net_d, "supports_san", False))
        self.mel_cond = bool(getattr(net_d, "cond_branches", None))
        self.mel_cond_warmup = int(d_settings.get("d_mrd_mel_cond_warmup", 0))
        self.branch_weights = tuple(net_d.branch_weights) if getattr(net_d, "uses_branch_weights", False) else None
        self.r1_gamma = float(getattr(net_d, "r1_gamma", 0.0))
        self.r1_branches = len(getattr(net_d, "discriminators", ())) if self.r1_gamma > 0 else 0
        interval = int(getattr(net_d, "r1_interval", 16))
        self.r1_every = max(1, math.ceil(interval / self.r1_branches)) if self.r1_branches else 1
        self.r1_period = self.r1_every * max(1, self.r1_branches)
        # R1's squared input gradient can underflow in FP16, so it runs in FP32 there.
        self.r1_dtype = amp_dtype if amp_dtype == torch.bfloat16 else None
        # The generator update reads D's frozen weights in two passes, real
        # then fake: cached, each weight norm is built once. Not under spectral
        # norm, whose power iteration advances per build, nor compiled, where
        # the graph already fuses it.
        self.cache_weights = not getattr(net_d, "use_spectral_norm", False) and not getattr(
            net_d, "_compile_enabled", False
        )

    def describe(self, d_settings: dict) -> str:
        net = self.net
        if self.own:
            return ("MSD" if "msd" in net else "MRD") + f" + MPD {d_settings['periods']}" + (
                ", hinge" if self.hinge else ""
            )
        return (
            f"{d_settings.get('d_version', 'v4')}, R1 gamma {self.r1_gamma:g}"
            + (f", UnivHD x{net.univhd_weight:g}" if getattr(net, "use_univhd", False) else "")
            + (", mel-conditioned MRD" if self.mel_cond else "")
            + (", compiled" if getattr(net, "_compile_enabled", False) else "")
        )

    def mel_cond_scale(self, step: int, since: int) -> float:
        """How much of the mel the discriminator takes, ``since`` being the
        step at which it started."""
        if self.mel_cond_warmup <= 0:
            return 1.0
        return min(1.0, (step - since) / self.mel_cond_warmup)

    def discriminator_losses(self, y, real, fake, cond):
        """D's loss with its real and fake parts, then the repo
        discriminator's outputs on ``real`` and ``fake`` (None for the
        recipe's own, which scores ``y`` against ``fake``)."""
        if self.own:
            losses = sv.discriminator_loss(self.train_net(y), self.train_net(fake.detach()), self.hinge)
            return (*losses, None, None)
        y_d_r, y_d_g, _, _ = self.train_net(
            real, fake.detach(), san_training=self.san, combine_inputs=True, cond=cond
        )
        losses = discriminator_loss(
            y_d_r, y_d_g, san_direction_weight=SAN_DIRECTION_WEIGHT, branch_weights=self.branch_weights
        )
        return (*losses, y_d_r, y_d_g)

    def r1(self, y, cond, step: int, hop: int):
        """The branch whose turn it is at ``step`` and its R1 penalty."""
        branch = (step // self.r1_every) % self.r1_branches
        count = max(1, int(round(y.shape[0] * float(getattr(self.net, "r1_batch_fraction", 0.5)))))
        segment = int(getattr(self.net, "r1_segment", 0))
        if cond is None:
            real, real_cond = loud_crop(y, count, segment), None
        else:
            real, real_cond = loud_crop(y, count, segment, cond=cond, hop=hop)
        return branch, r1_penalty(self.net, real, branch, dtype=self.r1_dtype, cond=real_cond)

    def generator_losses(self, y, y_hat, real, fake, cond):
        """The generator's adversarial and feature-matching losses through
        the frozen discriminator, and the adversarial one per branch (the
        repo's only, else None). ``real`` and ``fake`` continue ``y`` and
        ``y_hat`` with the pitch-shifted renders, if any."""
        net = self.net
        if self.own:
            d_fake = net(fake)
            with torch.no_grad():
                d_real = net(y)
            loss_adv = sum(sv.adversarial_loss(outputs, self.hinge) for outputs, _ in d_fake.values())
            # NSF-UnivNet matches features on the MPD alone.
            loss_fm = sum(
                sv.feature_loss(d_real[family][1], d_fake[family][1])
                for family in d_fake if not self.univnet or family == "mpd"
            )
            return loss_adv, loss_fm, None
        # Features are matched on the batch itself; shifted renders have no
        # target and take the adversarial term alone.
        _, y_d_g, fmap_r, fmap_g = net(y, y_hat, no_grad_real=True, cond=cond)
        loss_fm = feature_loss(fmap_r, fmap_g, branch_weights=self.branch_weights) * self.c_fm
        del fmap_r, fmap_g
        loss_adv, branch_adv = generator_loss(
            y_d_g, san_direction_weight=SAN_DIRECTION_WEIGHT, use_softplus=self.san,
            branch_weights=self.branch_weights, per_branch=True,
        )
        if fake.shape[0] > y.shape[0]:
            extra = fake[y.shape[0]:]
            _, y_d_g, _, _ = net(real[y.shape[0]:], extra, no_grad_real=True)
            extra_adv = generator_loss(
                y_d_g, san_direction_weight=SAN_DIRECTION_WEIGHT, use_softplus=self.san,
                branch_weights=self.branch_weights,
            )
            loss_adv = (loss_adv * y.shape[0] + extra_adv * extra.shape[0]) / fake.shape[0]
        return loss_adv, loss_fm, branch_adv

    def scalars(self, step: int, since: int) -> dict:
        """The mel conditioning's diagnostics."""
        if not self.mel_cond:
            return {}
        scalars = {"diag/mrd_mel_cond_scale": self.mel_cond_scale(step, since)}
        for index in sorted(self.net.cond_branches):
            branch = self.net.discriminators[index]
            scalars[f"diag/mrd_mel_proj_scale/{self.labels[index]}"] = (
                branch.MEL_PROJ_SCALE_MAX * torch.sigmoid(branch.mel_proj_logit.detach()).item()
            )
        return scalars


class RollingStats:
    """Windows over the last ``size`` steps: the same series, names and
    windows as the RVC trainer's."""

    def __init__(self, size: int, adversary: Adversary, device):
        self.size = size
        self.labels = adversary.labels
        self.families = FamilyReducer(self.labels, adversary.branch_weights, device) if self.labels else None
        self.values = defaultdict(lambda: deque(maxlen=size))
        self.disc, self.adv, self.accuracy, self.r1, self.skips = (deque(maxlen=size) for _ in range(5))

    def scalars(self, r1_gamma: float) -> dict:
        size, labels, scalars = self.size, self.labels, {}
        if self.accuracy:
            for family, value in zip(self.families.names, torch.stack(list(self.accuracy)).mean(0).tolist()):
                scalars[f"balance_accuracy_{size}/{family}"] = value
        if self.r1:
            per_branch = defaultdict(list)
            for branch, value in zip((b for b, _ in self.r1), torch.stack([v for _, v in self.r1]).tolist()):
                per_branch[branch].append(value)
            total = 0.0
            for branch, values in sorted(per_branch.items()):
                mean = sum(values) / len(values)
                total += mean
                scalars[f"r1_grad_sq/{labels[branch] if branch < len(labels) else branch}"] = mean
            scalars["r1/penalty"] = 0.5 * r1_gamma * total
        if self.skips:
            scalars[f"AMP/skip_rate_{size}"] = sum(self.skips) / len(self.skips)
        for key, queue in self.values.items():
            if queue:
                category = "loss" if key.startswith("loss_") else "grad"
                scalars[f"{category}_avg_{size}/{key}_{size}"] = torch.stack(list(queue)).mean().item()
        for prefix, cache in ((f"disc_sep_{size}", self.disc), (f"adv_sep_{size}", self.adv)):
            if cache:
                means = torch.stack(list(cache)).mean(0).tolist()
                for label, value in zip(labels, means):
                    scalars[f"{prefix}/{label}"] = value
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
    """Backward and step; returns the gradient norm on sampled steps."""
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    norm = clip_or_sample_grad_norm(parameters, grad_clip, step, METRICS_INTERVAL)
    optimizer_step(optimizer, scaler, lr_scale)
    return norm


def render_preview(net_g, ema, previews, reference, data: dict, raw_mel: bool, device, epoch: int, step: int):
    """Log the reference clip through the EMA weights; ``reference`` is None
    off the main rank."""
    if reference is None:
        return
    mel = normalize_mel(reference.mel.to(device), data)
    with ema.applied(net_g), torch.no_grad():
        generated = net_g(generator_input(mel, data, raw_mel), reference.inputs.f0.to(device))
    previews.log(epoch, step, reference.path, generated, reference.audio.to(device))


@torch.no_grad()
def evaluate(net_g, ema, holdout, data: dict, raw_mel: bool, spectral_loss, stft_distance, ranks: Ranks):
    """Held-out mel loss (at the ``c_mel`` scale of ``loss_spectral``) and
    MR-STFT distance through the EMA weights, with the source drawn from
    the same seeds on every call. Every rank must call it."""
    device = ranks.device
    totals = torch.zeros(2, device=device)
    items = 0
    with ema.applied(net_g):
        for index, (mel, f0, y) in enumerate(holdout):
            mel = normalize_mel(mel.to(device), data)
            y = y.to(device)
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(index)
                y_hat = net_g(generator_input(mel, data, raw_mel), f0.to(device)).float()
            totals[0] += spectral_loss(y, y_hat) * y.shape[0]
            totals[1] += stft_distance(y_hat, y) * y.shape[0]
            items += y.shape[0]
    return ranks.mean(totals / max(1, items)).tolist()


def export_payload(architecture: str, config: dict, hparams, ema) -> dict:
    """What the inference export holds besides its position in the run."""
    if architecture == sv.PCPH:
        return {"config": config, "model": ema.cpu_state_dict()}
    # Weight norm folded and only the generator's own settings, as
    # ``load_vocoder`` reads the OpenVPI exports. A generator fed the
    # normalised mel is tied to its statistics.
    keys = MEL_KEYS + (() if architecture in sv.RAW_MEL else ("mel_mean", "mel_std"))
    return {
        "architecture": sv.EXPORT_ARCHITECTURE[architecture], "recipe": architecture,
        "config": {"data": {key: config["data"][key] for key in keys}, "vocoder": {"model": hparams}},
        "model": sv.export_state(ema.cpu_state_dict()),
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
    architecture = spec.get("architecture", sv.PCPH)
    # An architecture with its own recipe (SingingVocoders', Wavehax): the
    # recipe is the ``vocoder`` section, read by the same keys as
    # PCPH-BigVGAN's where they mean the same.
    singing = architecture != sv.PCPH
    univnet = architecture == sv.UNIVNET
    raw_mel = architecture in sv.RAW_MEL
    config = sv.load_recipe_config(name, architecture) if singing else load_run_config(name)
    settings = config["vocoder"]
    # PCPH-BigVGAN trains against the repo's discriminator, as its config sets
    # it; the others by the recipe's choice, unless the run names one.
    discriminator = "config"
    if singing:
        discriminator = spec.get("discriminator") or settings.get("discriminator_choice", "original")
        if discriminator not in sv.DISCRIMINATORS:
            raise ValueError(f"discriminator must be one of {sv.DISCRIMINATORS}, not {discriminator!r}.")
    # The recipe's own discriminators and GAN losses, instead of the repo's.
    own_d = discriminator == "original"
    d_settings = settings["repo_discriminator"] if singing and not own_d else settings["discriminator"]
    data = config["data"]
    sample_rate = data["sample_rate"]
    hop = data["hop_length"]
    out_dir = run_dir(name, "vocoder")
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

    net_g, net_d, hparams = build_networks(architecture, config, d_settings, own_d, device)
    if settings.get("compile_generator") and device.type == "cuda" and hasattr(net_g, "compile_blocks"):
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
        spec, out_dir, architecture, discriminator, net_g, net_d, optim_g, optim_d, ema, scaler,
        plateau, plateau_metric,
    )
    epoch, step, skipped, mel_cond_since = start.epoch, start.step, start.skipped, start.mel_cond_since
    # "exp decay epoch" or "exp decay step", as in the RVC trainer; the step
    # variant spreads ``lr_decay`` over the epoch's steps.
    lr_scheduler = str(settings.get("lr_scheduler", "exp decay epoch"))
    lr_final_ratio = settings.get("lr_final_ratio")
    scheduler_g, scheduler_d = prepare_schedulers(
        optim_g, optim_d, lr_scheduler, lr_scheduler, float(settings["lr_decay"]),
        int(spec["total_epochs"]), epoch, step, loader,
        fresh_start=not start.resumed,
        lr_final_ratio=None if lr_final_ratio is None else float(lr_final_ratio),
    )
    step_schedulers = lr_scheduler == "exp decay step"
    # Linear ramps from the first step: the LR of both optimizers, and the
    # generator's adversarial weight. The early phase explodes without them.
    lr_warmup = int(settings.get("warmup_steps", 0))
    adv_warmup = int(settings.get("adv_warmup_steps", 0))

    apply_precision_policy(net_g, amp_dtype)
    compile_d = bool(settings.get("compile_discriminator", False))
    enable_discriminator_compile(
        net_d, SimpleNamespace(train=SimpleNamespace(compile_discriminator=compile_d)), device, 0
    )
    # Only the two training forwards go through DDP. The generator update's
    # D pass and R1 use ``net_d`` itself: with D frozen no gradient hook
    # fires, and an armed allreduce would never complete (see rvc/train/train.py).
    train_g = ranks.wrap(net_g)
    train_d = ranks.wrap(
        net_d, find_unused_parameters=bool(getattr(net_d, "uses_branchwise_r1", False))
    )
    adversary = Adversary(net_d, train_d, settings, d_settings, own_d, univnet, amp_dtype)
    autocast = partial(
        torch.autocast, device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
    )
    spectral_loss = SpectralLoss(architecture, config, device)
    stft_distance = MultiScaleSTFTLoss().to(device)

    # Steps on the spectral loss alone before the discriminator joins (NSF-UnivNet).
    aux_steps = int(settings.get("aux_steps", 0))
    if aux_steps > 0:
        # D's schedule runs through those steps too, so that it joins at G's
        # learning rate; PyTorch warns of a scheduler stepped before its optimizer.
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
    volume_prob = float(settings.get("volume_aug_prob", 0.0))
    pitch = PitchControl(net_g, config, batch_size, device)
    pc_count = pitch.count
    mel_cond_d = adversary.mel_cond
    if mel_cond_d and pc_count:
        raise ValueError("The mel-conditioned discriminator cannot score pitch-shifted renders; turn off d_mrd_mel_cond.")

    writer = SummaryWriter(os.path.join(out_dir, "eval")) if main_rank else None
    previews = RectifiedPreviews(out_dir, config, step, device) if main_rank else None
    reference = dataset.reference() if main_rank else None
    total_epochs = int(spec["total_epochs"])
    save_every = max(1, int(spec["save_every"]))
    # Only the generator's input: the target, the losses, D's mel and the
    # held-out evaluation stay on the real mel.
    degradation = min(1.0, max(0.0, float(spec.get("mel_degradation", 0.0))))
    c_pc_wav = float(settings.get("c_pc_wav", 30.0))
    grad_clip = float(settings.get("grad_clip") or "inf")

    if main_rank:
        print_model_summary(
            [("Generator", net_g), ("Discriminator", net_d)],
            title=f"Rectified vocoder {sample_rate} Hz",
        )
        print_settings_panel(
            [
                ("Model", name),
                ("Architecture", architecture),
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
                ("Losses", spectral_loss.label
                 + (", FM x2 on MPD" if own_d and univnet else ", FM x2" if own_d else f", FM x{settings['c_fm']:g}")
                 + ", adversarial"
                 + (f" ramped over {adv_warmup} steps" if adv_warmup > 0 else "")
                 + (f", after {aux_steps} spectral-only steps" if aux_steps > 0 else "")
                 + (f", PC waveform x{c_pc_wav:g} on {pc_count} clips (+-{pitch.key:g} keys)" if pc_count else "")),
                ("Discriminator", adversary.describe(d_settings)),
            ],
            title="Rectified vocoder",
        )

    rolling = int(settings.get("rolling_loss_steps", 50))
    preview_interval = int(settings.get("preview_interval", 500))
    stats = RollingStats(rolling, adversary, device)
    # Held off until both warmups end: the adversarial ramp raises the spectral
    # loss by itself.
    plateau_start = max(lr_warmup, adv_warmup)
    plateau_sum = torch.zeros((), device=device)
    plateau_count = 0

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
                    mel, y = sv.volume_augment(mel, y, volume_prob)
                mel = normalize_mel(mel, data)
                d_cond = mel if mel_cond_d else None
                aux_only = step < aux_steps
                lr_scale = ramp(step, lr_warmup) * plateau.scale
                if mel_cond_d:
                    net_d.set_mel_cond_scale(adversary.mel_cond_scale(step, mel_cond_since))

                g_mel = generator_input(degrade_mel(mel, degradation) if degradation > 0 else mel, data, raw_mel)
                with autocast():
                    if pc_count:
                        y_hat, shift_c, shift_a, shift_ab = pitch(train_g, g_mel, f0)
                        fake = torch.cat((y_hat, shift_c, shift_a, shift_ab))
                    else:
                        y_hat = fake = train_g(g_mel, f0)
                # The repo's D scores real and fake in pairs: the shifted
                # renders go against the clips they were made from.
                real = torch.cat((y, *[y[:pc_count]] * 3)) if pc_count and not own_d else y
                grad_norm_d = None
                y_d_r = y_d_g = None
                if aux_only:
                    loss_d = loss_d_real = loss_d_fake = y.new_zeros(())
                else:
                    with autocast():
                        loss_d, loss_d_real, loss_d_fake, y_d_r, y_d_g = adversary.discriminator_losses(
                            y, real, fake, d_cond
                        )
                    loss_d_step = loss_d
                    if y_d_r is not None and adversary.r1_branches and step % adversary.r1_every == 0:
                        branch, r1 = adversary.r1(y, d_cond, step, hop)
                        stats.r1.append((branch, r1.detach()))
                        loss_d_step = loss_d + 0.5 * adversary.r1_gamma * adversary.r1_period * r1
                    families = stats.families
                    if (
                        y_d_r is not None and families is not None and families.names
                        and step % ACCURACY_SAMPLE_EVERY == 0
                    ):
                        stats.accuracy.append(families(head_accuracies(y_d_r, y_d_g)))
                    optim_d.zero_grad(set_to_none=True)
                    grad_norm_d = backward(
                        loss_d_step, optim_d, net_d.parameters(), scaler, grad_clip, step, lr_scale
                    )
                    if y_d_r is not None:
                        normalize_san_weights(net_d, optim_d)
                        stats.disc.append(branch_separation(y_d_r, y_d_g))
                del y_d_r, y_d_g

                # The generator update reads D and never trains it: frozen, its
                # backward skips D's weight gradients and still reaches y_hat.
                net_d.requires_grad_(False)
                with autocast(), (
                    torch.nn.utils.parametrize.cached() if adversary.cache_weights else nullcontext()
                ):
                    if aux_only:
                        loss_fm = loss_adv = y.new_zeros(())
                        branch_adv = None
                    else:
                        loss_adv, loss_fm, branch_adv = adversary.generator_losses(y, y_hat, real, fake, d_cond)
                    if pc_count:
                        # The render at c through a against the render at c,
                        # as mel beside the batch's and as waveform.
                        loss_mel = spectral_loss(torch.cat((y, shift_c)), torch.cat((y_hat, shift_ab)))
                        loss_pc = F.l1_loss(shift_ab.float(), shift_c.float()) * c_pc_wav
                    else:
                        loss_mel = spectral_loss(y, y_hat)
                        loss_pc = 0.0
                    loss_g = loss_mel + loss_fm + loss_adv * ramp(step, adv_warmup) + loss_pc
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
                    for scheduler in (scheduler_g, scheduler_d):
                        scheduler.step()
                step += 1

                losses = {
                    "loss_disc": loss_d, "loss_disc_real": loss_d_real, "loss_disc_fake": loss_d_fake,
                    "loss_adv": loss_adv, "loss_gen_total": loss_g, "loss_fm": loss_fm,
                    "loss_spectral": loss_mel,
                }
                if pc_count:
                    losses["loss_pc_wav"] = loss_pc
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
                    val_mel, val_stft = evaluate(
                        net_g, ema, holdout, data, raw_mel, spectral_loss, stft_distance, ranks
                    )
                    if main_rank:
                        writer.add_scalar("val/mel", val_mel, step)
                        writer.add_scalar("val/mrstft", val_stft, step)
                    if plateau_metric == "val" and step > plateau_start:
                        plateau_update(plateau, plateau_metric, val_mel, writer, step)
                if branch_adv is not None:
                    stats.adv.append(branch_adv)
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
                            **stats.scalars(adversary.r1_gamma),
                            **adversary.scalars(step, mel_cond_since),
                        }
                        if plateau_interval > 0:
                            scalars["learning_rate/plateau_scale"] = plateau.scale
                        if lr_warmup > 0 or adv_warmup > 0:
                            scalars["learning_rate/warmup"] = ramp(step, lr_warmup)
                            scalars["diag/adv_weight"] = ramp(step, adv_warmup)
                        if scaler is not None:
                            scalars["AMP/grad_scaler_scale"] = scaler.get_scale()
                            scalars["AMP/skipped_steps_total"] = skipped
                        if getattr(net_g, "has_output_gain", False):
                            scalars["diag/output_gain"] = net_g.output_log_gain.exp().item()
                        for key, value in scalars.items():
                            writer.add_scalar(key, value, step)
                    if step % preview_interval == 0:
                        render_preview(net_g, ema, previews, reference, data, raw_mel, device, epoch, step)
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
            for scheduler in (scheduler_g, scheduler_d):
                if scheduler is not None:
                    scheduler.step()
        if main_rank:
            if skipped and scaler is not None:
                info(f"GradScaler at {scaler.get_scale():.0f}; {skipped} step(s) skipped so far.", tag=TAG)
            if epoch % save_every == 0 or epoch == total_epochs:
                save(
                    out_dir, name, spec.get("checkpoints", "latest"), epoch, step,
                    {"model": net_g.state_dict(), "optimizer": optim_g.state_dict(),
                     "ema": ema.state_dict(), "epoch": epoch, "step": step,
                     # ``pretrained_weights`` takes a file naming one for a foreign vocoder.
                     **({"architecture": architecture} if singing else {}),
                     "scaler": scaler.state_dict() if scaler is not None else None,
                     "amp_skipped_steps": skipped, "plateau": {**plateau.state_dict(), "metric": plateau_metric}},
                    {"model": net_d.state_dict(), "optimizer": optim_d.state_dict(),
                     "epoch": epoch, "step": step, "discriminator": discriminator,
                     "mel_cond_since": mel_cond_since if mel_cond_d else None},
                    export_payload(architecture, config, hparams, ema),
                )
                render_preview(net_g, ema, previews, reference, data, raw_mel, device, epoch, step)
            writer.flush()
        epoch += 1

    if main_rank:
        writer.close()
        success("Vocoder training finished.", tag=TAG)


if __name__ == "__main__":
    main(sys.argv[1])
