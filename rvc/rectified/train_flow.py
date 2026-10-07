"""Train the rectified-flow model: content, pitch, loudness and speaker ->
log mel.

Usage: python rvc/rectified/train_flow.py <spec.json>
"""

import copy
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from functools import partial

sys.path.append(os.getcwd())

import torch
from torch._functorch import config as aot_config
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
from rvc.rectified.common import (
    RectifiedDataset,
    amp_setup,
    check_pretrain_embedder,
    collate_flow,
    embedder_of,
    latest_checkpoint,
    load_run_config,
    precision_label,
    pretrained_weights,
    read_filelist,
    run_dir,
    remove_older,
    speaker_count,
    split_holdout,
)
from rvc.rectified.distributed import Ranks, launch, parse_gpus
from rvc.rectified.feature_cache import BucketBatchSampler, CachedFlowDataset, build_cache, load_cache
from rvc.rectified.flow import build_flow, match_inputs, resize_speakers
from rvc.rectified.flow_validation import evaluate, preview
from rvc.rectified.mel import normalize_mel
from rvc.rectified.muon import MuonAdamW
from rvc.rectified.previews import RectifiedPreviews
from rvc.rectified.vocoder import load_vocoder
from rvc.train.ema import WeightEMA
from rvc.train.progress import EpochRecorder, emit_machine_progress
from rvc.train.schedules import cosine_progress
from rvc.train.setup import loader_workers
from rvc.train.stop import finish_stop, install_stop_handlers, uninterruptible_save

TAG = "[FLOW]"
LOG_INTERVAL = 50
METRICS_INTERVAL = 8
#: Ceiling of the FP16 GradScaler's scale: left to grow, it reaches ~2^22,
#: where the weight gradients overflow and a step is skipped.
MAX_GRAD_SCALE = 2.0**16
#: Steps skipped in a row over a non-finite gradient before the run stops: by
#: then it is the weights or the data, not one batch.
MAX_SKIPPED_IN_A_ROW = 10
#: Batches a new run reads for the mel's per-bin statistics.
MEL_STATS_BATCHES = 200
#: How much of the data is shifted and stretched: drawn per item, or made
#: ahead for the feature cache. A fine-tune has its own, under ``finetune_``.
FINETUNE_AUGMENTATION = ("key_shift_prob", "time_stretch_prob", "key_shift_scale", "time_stretch_scale")


def learning_rate(base, step, warmup, total, final_ratio, anchor=(0, 0.0)):
    """Linear warmup, then cosine decay to ``final_ratio`` of ``base`` at ``total``.

    ``anchor`` is the (step, progress along the cosine) a resumed run continues
    from, so a changed ``total`` stretches or compresses what is left of it.
    """
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    start, done = max(anchor[0], warmup), anchor[1]
    progress = done + (1.0 - done) * min(1.0, max(0.0, (step - start) / max(1, total - start)))
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def freeze_voice(model) -> int:
    """Freeze what maps time and speaker into the network: the time MLP and
    ``speaker_layers``. The speaker's own row still trains; the null row gets
    no gradient without speaker dropout. Keeps the pretrain's speaker space,
    so speaker guidance still separates the voice from the null speaker after
    a one-speaker fine-tune (RIFT-SVC's recipe). Returns the number of tensors
    frozen."""
    frozen = 0
    for module in filter(None, [model.backbone.time_mlp, *model.speaker_layers()]):
        for param in module.parameters():
            param.requires_grad_(False)
            frozen += 1
    return frozen


def pretrain_time_scale(path: str):
    """The time scale a flow export was trained with; None for a checkpoint,
    which records no config."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = checkpoint.get("config", {}).get("flow", {}).get("model")
    if model is None:
        return None
    return float((model.get("backbone_args") or {}).get("time_scale", 1000.0))


def conditioning_norms(model) -> dict:
    """Weight norms of the time and speaker paths, for TensorBoard: a norm
    that keeps climbing is the conditioning running away."""
    backbone = model.backbone
    norms = {"diag/time_mlp_norm": sum(p.norm().item() ** 2 for p in backbone.time_mlp.parameters()) ** 0.5}
    modulation = [layer.modulation.weight.norm().item()
                  for layer in backbone.layers if layer.modulation is not None]
    if modulation:
        norms["diag/adaln_norm_max"] = max(modulation)
        norms["diag/adaln_norm_mean"] = sum(modulation) / len(modulation)
    if backbone.voice is not None:
        norms["diag/voice_proj_norm"] = backbone.voice.weight.norm().item()
    return norms


def compiled_backbone(model, enabled: bool, mode: str, device, dynamic: bool = False):
    """``torch.compile`` of the backbone for training; None when left eager.
    ``dynamic`` is for batches of more than one shape. Evaluation and sampling
    keep the eager module, whose shapes vary."""
    if not enabled:
        return None
    if device.type != "cuda" or importlib.util.find_spec("triton") is None:
        warning("torch.compile needs CUDA and Triton; training uncompiled.", tag=TAG)
        return None
    return torch.compile(model.backbone, mode=mode, dynamic=True if dynamic else None)


def load_preview_vocoder(path: str, config: dict, device):
    """The vocoder that renders previews, or None when it cannot."""
    if not path:
        warning("No vocoder: previews show the mel only, and exports are not paired "
                "with a vocoder, so inference will not pick one.", tag=TAG)
        return None
    if not os.path.exists(path):
        warning(f"Vocoder {path} not found; no audio previews.", tag=TAG)
        return None
    try:
        vocoder, _ = load_vocoder(path, config["data"])
    except ValueError as error:
        warning(f"{error} No audio previews.", tag=TAG)
        return None
    return vocoder.to(device)


def apply_spec(spec: dict, settings: dict) -> None:
    """The run's own switches over the config's: the feature cache can be
    turned off for a run, which then augments as it goes, in fixed segments,
    and a fine-tune takes the ``finetune_`` augmentation shares the config has."""
    settings["feature_cache"] = bool(settings.get("feature_cache", False) and spec.get("feature_cache", True))
    if spec.get("pretrained_flow"):
        for key in FINETUNE_AUGMENTATION:
            if "finetune_" + key in settings:
                settings[key] = settings["finetune_" + key]


def bucketed(settings: dict) -> bool:
    """Whether batches are whole clips of similar length, which takes the
    lengths the feature cache knows."""
    return bool(settings.get("feature_cache", False) and settings.get("bucket_batches", False))


def build_loaders(ranks: Ranks, name: str, config: dict, entries, holdout_entries, batch_size: int):
    """The training set with its loader and sampler, and the held-out clips'
    loader: None without any, or off the main rank.

    With ``feature_cache`` the loader reads the cached items; with
    ``bucket_batches`` too, in batches of whole clips up to ``batch_size``
    segments' worth of padded frames."""
    settings = config["flow"]
    segment = int(settings["segment_frames"])
    dataset = RectifiedDataset(entries, config, "flow", segment)
    # Per GPU, as in the RVC trainer.
    if len(dataset) // ranks.world < batch_size:
        raise ValueError(
            f"{len(dataset)} clips is fewer than one batch of {batch_size} on each of {ranks.world} GPU(s)."
        )
    workers = loader_workers(settings.get("num_workers", 4))
    # Every batch padded to the crop length, one shape for the compiled backbone.
    collate = partial(collate_flow, frames=segment)
    stream_seed = settings.get("stream_seed")
    common = dict(
        num_workers=workers,
        pin_memory=ranks.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        # With ``stream_seed``, the order and the workers' own seeds come from it alone.
        generator=None if stream_seed is None else torch.Generator().manual_seed(int(stream_seed) + ranks.rank),
    )
    items = dataset
    if settings.get("feature_cache", False):
        limit = int(settings.get("bucket_max_frames", segment)) if bucketed(settings) else segment
        items = CachedFlowDataset(dataset, *load_cache(name, config, entries), limit)
    if bucketed(settings):
        sampler = BucketBatchSampler(
            items.lengths(), batch_size * segment, int(settings.get("bucket_max_items", 64)),
            int(settings.get("seed", 1234)), ranks.rank, ranks.world,
        )
        loader = DataLoader(items, batch_sampler=sampler, collate_fn=collate_flow, **common)
    else:
        sampler = ranks.sampler(items)
        loader = DataLoader(
            items,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate,
            drop_last=True,
            **common,
        )
    holdout = None
    if holdout_entries and ranks.main:
        holdout = DataLoader(
            RectifiedDataset(holdout_entries, config, "flow", segment, augment=False),
            batch_size=min(batch_size, len(holdout_entries)),
            num_workers=min(2, workers),
            collate_fn=collate,
        )
    return dataset, loader, sampler, holdout


def augment_share(settings: dict, kind: str) -> str:
    """How much of the data is augmented ``kind``, in words: copies made ahead
    under ``feature_cache``, else a share of the items drawn."""
    if settings.get("feature_cache", False):
        return f"x{settings.get(kind + '_scale', 0):g} offline"
    return f"{100 * settings.get(kind + '_prob', 0):g}%"


def configure_model(spec: dict, settings: dict) -> None:
    """Write what the run decides, not the config, into ``settings["model"]``:
    whether it is a shortcut model and, for a fine-tune, the pretrain's time
    scale."""
    settings["model"]["shortcut"] = bool(spec.get("shortcut", False))
    if spec.get("pretrained_flow"):
        # Not in the weights, and a network reads another scale's times as noise.
        time_scale = pretrain_time_scale(spec["pretrained_flow"])
        if time_scale is not None:
            settings["model"].setdefault("backbone_args", {})["time_scale"] = time_scale


@torch.no_grad()
def mel_statistics(ranks: Ranks, loader, data: dict):
    """Mean and spread of each mel bin, [n_mels] each, in the data's
    normalisation, over the first ``MEL_STATS_BATCHES`` batches of every rank.
    Every rank must call it."""
    total = squares = frames = 0.0
    for index, (mel, inputs) in enumerate(loader):
        if index >= MEL_STATS_BATCHES:
            break
        mask = inputs.mask.to(ranks.device).double()
        mel = normalize_mel(mel.to(ranks.device), data).double() * mask
        total = total + mel.sum((0, 2))
        squares = squares + mel.square().sum((0, 2))
        frames = frames + mask.sum()
    frames = ranks.mean(frames)
    mean = ranks.mean(total) / frames
    std = (ranks.mean(squares) / frames - mean.square()).clamp_min(0.0).sqrt()
    return mean.float(), std.float()


def build_optimizer(model, settings: dict, base_lr: float):
    if settings.get("optimizer", "adamw") == "muon":
        return MuonAdamW(
            model, base_lr, muon_weight_decay=settings["weight_decay"],
            betas=tuple(settings["betas"]),
        )
    return torch.optim.AdamW(
        model.parameters(), base_lr, betas=tuple(settings["betas"]),
        weight_decay=settings["weight_decay"],
    )


def restore(spec: dict, name: str, resume, state, model, optimizer, ema, scaler):
    """Load the checkpoint ``state`` of ``resume``, or the pretrain of a
    fine-tune. Returns the epoch and step to continue from, the FP16 steps
    skipped so far and where the run starts from, in words."""
    if resume:
        model.load_state_dict(match_inputs(state["model"], model))
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            shadow = match_inputs(state["ema"]["shadow"], model)
            ema.load_state_dict({**state["ema"], "shadow": shadow}, model)
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        skipped = int(state.get("skipped_steps", 0))
        return state["epoch"] + 1, state["step"], skipped, f"resumed from {os.path.basename(resume)}"
    pretrain = spec.get("pretrained_flow")
    if pretrain:
        check_pretrain_embedder(pretrain, embedder_of(name))
        weights = resize_speakers(pretrained_weights(pretrain), model.speaker_count)
        model.load_state_dict(match_inputs(weights, model))
        ema.reseed(model)
        return 1, 0, 0, f"fine-tune from {os.path.basename(pretrain)}"
    return 1, 0, 0, "scratch"


def optimizer_step(loss, model, optimizer, scaler, grad_clip: float):
    """Backward, clip and step. Returns the gradient norm and whether the step
    was skipped over a non-finite gradient, which would otherwise reach every
    weight."""
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            return grad_norm, True
        optimizer.step()
        return grad_norm, False
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scale = scaler.get_scale()
    scaler.update()
    skipped = scaler.get_scale() < scale
    if not skipped and scaler.get_scale() > MAX_GRAD_SCALE:
        scaler.update(MAX_GRAD_SCALE)
    return grad_norm, skipped


def nonfinite_names(mel, inputs, model) -> str:
    """What holds a non-finite value among a batch and the weights, in words."""
    found = [
        name for name, value in (("mel", mel), *zip(inputs._fields, inputs))
        if value is not None and not torch.isfinite(value).all()
    ]
    if any(not torch.isfinite(param).all() for param in model.parameters()):
        found.append("model weights")
    return ", ".join(found) or "neither the batch nor the weights, so the loss or a gradient overflowed"


def loss_scalars(ranks: Ranks, flow_loss, aux_loss) -> dict:
    """The losses averaged over the ranks, by TensorBoard tag, under the
    groups the RVC trainer logs to. Every rank must call it."""
    scalars = {f"loss_avg_{LOG_INTERVAL}/loss_flow_{LOG_INTERVAL}": ranks.mean(flow_loss).item()}
    if aux_loss is not None:
        scalars[f"loss_avg_{LOG_INTERVAL}/loss_aux_mel_l1_{LOG_INTERVAL}"] = ranks.mean(aux_loss).item()
    return scalars


def save(out_dir: str, name: str, keep: str, model, optimizer, ema, scaler, epoch: int, step: int,
         skipped: int, export: dict) -> None:
    """Write the checkpoint to resume from and the export of the averaged
    weights. ``keep`` is the run's ``checkpoints``: every one, the latest or
    none. ``export`` is what the export records beside its weights."""
    if any(not torch.isfinite(param).all() for param in model.parameters()):
        raise FloatingPointError(
            "The flow has non-finite weights. Nothing was saved, so the last checkpoint stands."
        )
    saved = []
    with uninterruptible_save("Flow checkpoint"):
        if keep == "none":
            # An older run's checkpoint would otherwise be resumed from,
            # silently undoing everything trained since.
            remove_older(out_dir, "F")
        else:
            path = os.path.join(out_dir, f"F_{step}.pth")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "ema": ema.state_dict(), "epoch": epoch, "step": step,
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "skipped_steps": skipped},
                path,
            )
            if keep == "latest":
                remove_older(out_dir, "F", path)
            saved.append(os.path.basename(path))
        export = os.path.join(out_dir, f"{name}_flow_{epoch}e_{step}s.pth")
        torch.save(
            {"kind": "rectified_flow", **export, "model": ema.cpu_state_dict(), "epoch": epoch, "step": step},
            export,
        )
    saved.append(os.path.basename(export))
    success(f"Saved {' and '.join(saved)}.", tag=TAG)


def main(spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    config = load_run_config(spec["model_name"])
    settings = config["flow"]
    apply_spec(spec, settings)
    speakers = speaker_count(read_filelist(spec["model_name"]))
    if spec.get("pretrained_flow") and speakers > 1:
        raise ValueError(
            f"A pretrained flow does not fine-tune to {speakers} speakers. Use a dataset of one speaker, "
            "or train without a pretrained flow."
        )
    if settings.get("feature_cache", False):
        # Before the ranks start, which then only read it.
        # Sorted: every extraction shuffles the filelist anew, and the cache is keyed by its order.
        entries = sorted(read_filelist(spec["model_name"]))
        entries, _ = split_holdout(entries, int(settings.get("holdout_clips", 0)))
        build_cache(spec["model_name"], config, entries, loader_workers(settings.get("num_workers", 4)))
    launch(train, spec_path, parse_gpus(spec.get("gpu", "0")))


def train(ranks: Ranks, spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    ranks.setup()
    main_rank = ranks.main

    name = spec["model_name"]
    config = load_run_config(name)
    settings = config["flow"]
    apply_spec(spec, settings)
    out_dir = run_dir(name, "flow")
    os.makedirs(out_dir, exist_ok=True)

    device = ranks.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Training batches are one shape; evaluation and previews are not.
    # Nor are bucketed batches, and the benchmark would run on every new one.
    torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False)) and not bucketed(settings)

    entries = sorted(read_filelist(name))
    speakers = speaker_count(entries)
    entries, holdout_entries = split_holdout(entries, int(settings.get("holdout_clips", 0)))
    batch_size = int(spec["batch_size"])
    dataset, loader, sampler, holdout = build_loaders(ranks, name, config, entries, holdout_entries, batch_size)

    resume = None if spec.get("fresh") else latest_checkpoint(out_dir, "F")
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    finetune = bool(spec.get("pretrained_flow"))
    configure_model(spec, settings)
    model = build_flow(config, speakers).to(device)
    speaker_dropout = float(settings["speaker_dropout"])
    tension_dropout = float(settings.get("tension_dropout", 0.0)) if model.encoder.tension is not None else 0.0
    voice_frozen = finetune and speakers == 1 and settings.get("finetune_freeze_voice", True)
    if voice_frozen:
        freeze_voice(model)
        speaker_dropout = 0.0
    base_lr = settings["finetune_learning_rate"] if finetune else settings["learning_rate"]
    if spec.get("learning_rate"):
        base_lr = float(spec["learning_rate"])
    optimizer = build_optimizer(model, settings, base_lr)
    amp_dtype, scaler = amp_setup(spec.get("precision", "fp32"), device, TAG)

    # A fine-tune is too short for the pretrain's 10k-step horizon.
    ema_decay = settings.get("finetune_ema_decay", settings["ema_decay"]) if finetune else settings["ema_decay"]
    ema = WeightEMA(model, ema_decay)
    epoch, step, skipped, starting_point = restore(spec, name, resume, state, model, optimizer, ema, scaler)
    # Its CPU copy would otherwise stay in RAM for the whole run.
    state = None
    # A resumed run and a fine-tune keep the statistics their weights came with.
    if settings.get("mel_bin_norm", False) and not resume and not finetune:
        model.set_mel_stats(*mel_statistics(ranks, loader, config["data"]))
        ema.reseed(model)

    writer = SummaryWriter(os.path.join(out_dir, "eval")) if main_rank else None
    previews = RectifiedPreviews(out_dir, config, step, device) if main_rank else None
    # The preview clip, with whether it is held out: the reference clip, which
    # may be another voice's, or else one held-out clip, a sure reconstruction.
    clips = []
    if main_rank:
        reference = dataset.reference()
        if reference is not None:
            clips = [(reference, False)]
        elif holdout is not None:
            clips = [(clip, True) for clip in holdout.dataset.speaker_clips(1)]
    # A fine-tune is short, so it previews more often; experiments whose config
    # predates the key get the same.
    preview_interval = int(
        settings.get("finetune_preview_interval", 500) if finetune
        else settings.get("preview_interval", 100)
    )
    total_epochs = int(spec["total_epochs"])
    save_every = max(1, int(spec["save_every"]))
    total_steps = total_epochs * len(loader)
    # A fine-tune's optimizer starts cold on trained weights, and Muon's step
    # does not shrink with the gradient; a short one is not all warmup.
    warmup = int(settings["warmup_steps"])
    if finetune:
        warmup = min(int(settings.get("finetune_warmup_steps", 0)), total_steps // 4)
    final_ratio = float(settings.get("lr_final_ratio", 1.0))
    lr_anchor = (0, 0.0)
    if resume and step >= warmup:
        lr_anchor = (step, cosine_progress(optimizer.param_groups[0]["lr"] / base_lr, final_ratio))
    aux_weight = float(settings.get("aux_mel_weight", 0.0))
    eval_interval = int(settings.get("eval_interval", 0))
    backbone = compiled_backbone(
        model, bool(spec.get("compile", False)), spec.get("torch_compile_mode", "default"), device,
        dynamic=bucketed(settings),
    )
    # The backward runs outside autocast here, and a compiled graph's would
    # otherwise run under the forward's.
    backward_context = (
        partial(aot_config.patch, backward_pass_autocast="off") if backbone is not None else nullcontext
    )
    # A shortcut model learns its jumps from a copy of itself that lags it,
    # which starts at the averaged weights.
    teacher, teacher_weights, shortcut_share = None, None, 0.0
    if model.shortcut_levels:
        shortcut_share = float(settings.get("shortcut_share", 0.125))
        teacher_decay = float(settings.get("shortcut_ema_decay", 0.999))
        with ema.applied(model):
            teacher = copy.deepcopy(model).requires_grad_(False).eval()
        teacher_weights = (list(teacher.parameters()), list(model.parameters()))
    # Frozen parameters are left out of the gradient sync. Sampling, previews
    # and evaluation go through ``model`` itself.
    train_model = ranks.wrap(model)
    if settings.get("stream_seed") is not None:
        # Two models with different layers then train on the same noise and times.
        model.generator = torch.Generator(device).manual_seed(int(settings["stream_seed"]) + 1 + ranks.rank)
    embedder = embedder_of(name)
    vocoder_path = spec.get("vocoder", "")
    vocoder = load_preview_vocoder(vocoder_path, config, device) if main_rank else None

    if main_rank:
        print_model_summary([("Flow", model)], title="Rectified flow")
        print_settings_panel(
            [
                ("Model", f"{name}, {speakers} speakers" + (", voice frozen" if voice_frozen else "")),
                ("Data", f"{len(dataset)} clips ({len(holdout_entries)} held out), "
                         + (f"{len(loader.dataset)} cached items, " if settings.get("feature_cache", False) else "")
                         + (f"whole clips in batches of up to {batch_size * settings['segment_frames']} frames, "
                            if bucketed(settings) else f"batch {batch_size} x {settings['segment_frames']} frames, ")
                         + f"{len(loader)} steps per epoch"),
                ("Epochs", f"{epoch} -> {total_epochs}, saving every {save_every}, {starting_point}"),
                ("Backbone", "LYNXNet2"
                             + (" with adaLN" if model.backbone.voice is not None else "")
                             + (f", shallow from t={model.t_start:g}" if model.t_start > 0 else "")
                             + (", dual timestep" if model.dual_timestep else "")
                             + (f", shortcut on {100 * shortcut_share:g}% of each batch" if teacher is not None else "")
                             + (", per-bin mel" if settings.get("mel_bin_norm", False) else "")
                             + (", compiled" if backbone is not None else "")),
                ("Training", f"{'Muon + AdamW' if isinstance(optimizer, MuonAdamW) else 'AdamW'}, lr {base_lr:g}, "
                             + (f"{warmup}-step warmup, " if warmup else "")
                             + f"cosine to {final_ratio:g}x at step {total_steps}, {precision_label(amp_dtype)} on "
                             + (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
                             + (f" x {ranks.world} GPUs" if ranks.world > 1 else "")),
                ("Augment", f"key ±{settings.get('key_shift_range', 0):g} st "
                            f"({augment_share(settings, 'key_shift')}), "
                            "stretch x{:g}-{:g} ({}), ".format(
                                *settings.get("time_stretch_range", (1, 1)), augment_share(settings, "time_stretch"))
                            + f"speaker dropout {speaker_dropout:g}"
                            + (f", tension dropout {tension_dropout:g}" if tension_dropout > 0 else "")),
                ("Vocoder", os.path.basename(vocoder_path) if vocoder is not None else "none (previews without audio)"),
            ],
            title="Rectified flow",
        )

    export = {
        "config": config, "speaker_count": speakers, "embedder_model": embedder,
        "vocoder": vocoder_path if vocoder is not None else "",
    }
    recorder = EpochRecorder()
    model.train()
    data_wait = step_time = 0.0
    skipped_in_a_row = 0
    # Flow loss, aux loss and gradient norm summed over the applied steps since
    # the last log.
    window = torch.zeros(3, device=device)
    window_steps = 0
    while epoch <= total_epochs:
        metrics = ""
        if sampler is not None:
            sampler.set_epoch(epoch)
        with progress_task(
            len(loader), f"Epoch {epoch}/{total_epochs}", training=True, disable=not main_rank
        ) as (progress, task):
            fetch_started = time.perf_counter()
            for batch_index, (mel, inputs) in enumerate(loader):
                step_started = time.perf_counter()
                data_wait += step_started - fetch_started
                mel = normalize_mel(mel.to(device, non_blocking=True), config["data"])
                inputs = inputs.to(device, non_blocking=True)

                lr = learning_rate(base_lr, step, warmup, total_steps, final_ratio, lr_anchor)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with backward_context(), torch.autocast(
                    device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
                ):
                    flow_loss, aux_loss = train_model(
                        mel * inputs.mask, inputs,
                        speaker_dropout=speaker_dropout, tension_dropout=tension_dropout,
                        backbone=backbone,
                        teacher=teacher, shortcut_share=shortcut_share,
                    )
                    loss = flow_loss
                    if aux_loss is not None:
                        loss = loss + aux_weight * aux_loss
                grad_norm, step_skipped = optimizer_step(loss, model, optimizer, scaler, settings["grad_clip"])
                if step_skipped:
                    skipped += 1
                    skipped_in_a_row += 1
                    culprit = nonfinite_names(mel, inputs, model)
                    if skipped_in_a_row >= MAX_SKIPPED_IN_A_ROW:
                        raise FloatingPointError(
                            f"{skipped_in_a_row} steps in a row had a non-finite gradient. Non-finite: {culprit}."
                        )
                    if main_rank:
                        warning(f"Step {step + 1} skipped over a non-finite gradient. Non-finite: {culprit}.", tag=TAG)
                else:
                    skipped_in_a_row = 0
                    window += torch.stack([
                        flow_loss.detach(),
                        aux_loss.detach() if aux_loss is not None else flow_loss.new_zeros(()),
                        grad_norm,
                    ]).float()
                    window_steps += 1
                    ema.update(model)
                    if teacher is not None:
                        with torch.no_grad():
                            torch._foreach_lerp_(*teacher_weights, 1.0 - teacher_decay)
                step += 1
                step_time += time.perf_counter() - step_started

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        metrics = f"loss={flow_loss.item():.4f}"
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)

                if step % LOG_INTERVAL == 0:
                    # Collective, so outside the rank check.
                    flow_mean, aux_mean, grad_mean = window / max(1, window_steps)
                    scalars = loss_scalars(ranks, flow_mean, aux_mean if aux_loss is not None else None)
                    window.zero_()
                    window_steps = 0
                if step % LOG_INTERVAL == 0 and main_rank:
                    # Time the loop spent waiting on the loader; near zero when
                    # the workers keep up.
                    scalars["perf/data_wait_ms"] = 1000 * data_wait / LOG_INTERVAL
                    scalars["perf/step_ms"] = 1000 * step_time / LOG_INTERVAL
                    data_wait = step_time = 0.0
                    scalars[f"grad_avg_{LOG_INTERVAL}/grad_norm_{LOG_INTERVAL}"] = grad_mean.item()
                    scalars.update(conditioning_norms(model))
                    scalars["learning_rate/lr"] = lr
                    scalars["diag/skipped_steps"] = skipped
                    if scaler is not None:
                        scalars["AMP/grad_scaler_scale"] = scaler.get_scale()
                    for tag, value in scalars.items():
                        writer.add_scalar(tag, value, step)
                if clips and step % preview_interval == 0:
                    preview(model, ema, previews, clips, vocoder, config["data"], device, epoch, step)
                if holdout is not None and eval_interval and step % eval_interval == 0:
                    evaluate(model, ema, holdout, config["data"], device, amp_dtype, writer, step)
                if ranks.stop_requested():
                    # Nothing is being written at a batch boundary.
                    finish_stop(writer)
                fetch_started = time.perf_counter()

        if main_rank:
            print(f"{name} | epoch={epoch} | step={step} | {recorder.record()}")
            if skipped:
                scale = "" if scaler is None else f" GradScaler at {scaler.get_scale():.0f}."
                info(f"{skipped} step(s) skipped so far.{scale}", tag=TAG)
            if epoch % save_every == 0 or epoch == total_epochs:
                save(out_dir, name, spec.get("checkpoints", "latest"), model, optimizer, ema, scaler, epoch, step,
                     skipped, export)
                if clips:
                    preview(model, ema, previews, clips, vocoder, config["data"], device, epoch, step)
            writer.flush()
        epoch += 1

    if main_rank:
        writer.close()
        success("Flow training finished.", tag=TAG)


if __name__ == "__main__":
    main(sys.argv[1])
