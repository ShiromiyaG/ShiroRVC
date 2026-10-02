"""Train the rectified-flow model: content, pitch, loudness and speaker ->
log mel.

Usage: python rvc/rectified/train_flow.py <spec.json>
"""

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
from rvc.rectified.flow_model import build_flow, match_inputs, resize_speakers
from rvc.rectified.mel import denormalize_mel, normalize_mel
from rvc.rectified.muon import MuonAdamW
from rvc.rectified.previews import RectifiedPreviews
from rvc.rectified.vocoder import load_vocoder
from rvc.train.ema import WeightEMA
from rvc.train.progress import EpochRecorder, emit_machine_progress
from rvc.train.setup import loader_workers
from rvc.train.stop import finish_stop, install_stop_handlers, uninterruptible_save

TAG = "[FLOW]"
LOG_INTERVAL = 50
METRICS_INTERVAL = 8
PREVIEW_STEPS = 16
#: Where in the trained time range the validation loss is taken.
EVAL_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
#: Ceiling of the FP16 GradScaler's scale: left to grow, it reaches ~2^22,
#: where the weight gradients overflow and a step is skipped.
MAX_GRAD_SCALE = 2.0**16


def learning_rate(base, step, warmup, total, final_ratio):
    """Linear warmup, then cosine decay to ``final_ratio`` of ``base`` at ``total``."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def freeze_voice(model) -> int:
    """Freeze what maps time and speaker into the network: the time MLP, the
    adaLN modulation and both speaker projections. The speaker's own row still
    trains; the null row gets no gradient without speaker dropout. Keeps the
    pretrain's speaker space, so speaker guidance still separates the voice
    from the null speaker after a one-speaker fine-tune (RIFT-SVC's recipe).
    Returns the number of tensors frozen."""
    modules = [model.backbone.time_mlp, model.encoder.speaker_proj, model.backbone.voice]
    modules += [layer.modulation for layer in model.backbone.layers]
    frozen = 0
    for module in filter(None, modules):
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


def compiled_backbone(model, enabled: bool, mode: str, device, mean_flow: bool = False):
    """``torch.compile`` of the backbone for training, and of the mean
    velocity with ``mean_flow``; None for each left eager. Evaluation and
    sampling keep the eager module, whose shapes vary."""
    if not enabled:
        return None, None
    if device.type != "cuda" or importlib.util.find_spec("triton") is None:
        warning("torch.compile needs CUDA and Triton; training uncompiled.", tag=TAG)
        return None, None
    mean_field = torch.compile(model.mean_velocity, mode=mode) if mean_flow else None
    return torch.compile(model.backbone, mode=mode), mean_field


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


def build_loaders(ranks: Ranks, config: dict, entries, holdout_entries, batch_size: int):
    """The training set with its loader and sampler, and the held-out clips'
    loader: None without any, or off the main rank."""
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
    sampler = ranks.sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=ranks.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
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


def configure_model(spec: dict, settings: dict, state, main_rank: bool) -> bool:
    """Write what the run decides, not the config, into ``settings["model"]``:
    mean flow and, for a fine-tune, the pretrain's time scale. ``state`` is the
    checkpoint being resumed, or None. Returns whether the run trains mean
    flow."""
    # A resumed run keeps what it started with.
    mean_flow = bool(spec.get("mean_flow", False))
    if state is not None:
        started_with = any(key.startswith("backbone.span_mlp.") for key in state["model"])
        if started_with != mean_flow and main_rank:
            warning(f"This run was started {'with' if started_with else 'without'} mean flow "
                    f"and resumes that way; start fresh to change it.", tag=TAG)
        mean_flow = started_with
    settings["model"]["mean_flow"] = mean_flow
    backbone_args = settings["model"].setdefault("backbone_args", {})
    if spec.get("pretrained_flow"):
        # Not in the weights, and a network reads another scale's times as noise.
        time_scale = pretrain_time_scale(spec["pretrained_flow"])
        if time_scale is not None:
            backbone_args["time_scale"] = time_scale
    if mean_flow and float(backbone_args.get("time_scale", 1000.0)) > 10 and main_rank:
        warning("Mean flow with a time scale over 10 has diverged: its target is the network's own "
                "derivative in time. Set flow.model.backbone_args.time_scale to 1 for a new pretrain.", tag=TAG)
    return mean_flow


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
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"], model)
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        skipped = int(state.get("amp_skipped_steps", 0))
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
    """Backward, clip and step. Returns the gradient norm and whether the FP16
    scaler skipped the step over a non-finite gradient."""
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def loss_scalars(ranks: Ranks, flow_loss, aux_loss, mean_losses) -> dict:
    """The step's losses averaged over the ranks, by TensorBoard tag. Every
    rank must call it."""
    plain_flow = flow_loss if mean_losses is None else mean_losses.flow
    scalars = {"loss/flow": ranks.mean(plain_flow).item()}
    if aux_loss is not None:
        scalars["loss/aux_mel_l1"] = ranks.mean(aux_loss).item()
    if mean_losses is not None:
        scalars["loss/mean_flow"] = ranks.mean(mean_losses.mean).item()
        # Over 1 the target is feeding on itself and is being held.
        scalars["diag/mean_flow_bootstrap"] = ranks.mean(mean_losses.bootstrap_ratio).item()
    return scalars


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
    config = load_run_config(name)
    settings = config["flow"]
    out_dir = run_dir(name, "flow")
    os.makedirs(out_dir, exist_ok=True)

    device = ranks.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Training batches are one shape; evaluation and previews are not.
    torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False))

    entries = read_filelist(name)
    speakers = speaker_count(entries)
    entries, holdout_entries = split_holdout(entries, int(settings.get("holdout_clips", 0)))
    batch_size = int(spec["batch_size"])
    dataset, loader, sampler, holdout = build_loaders(ranks, config, entries, holdout_entries, batch_size)

    resume = None if spec.get("fresh") else latest_checkpoint(out_dir, "F")
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    finetune = bool(spec.get("pretrained_flow"))
    mean_flow = configure_model(spec, settings, state, main_rank)
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

    writer = SummaryWriter(os.path.join(out_dir, "eval")) if main_rank else None
    previews = RectifiedPreviews(out_dir, config, step, device) if main_rank else None
    reference = dataset.reference() if main_rank else None
    # A fine-tune is short, so it previews more often; experiments whose config
    # predates the key get the same.
    preview_interval = int(
        settings.get("finetune_preview_interval", 500) if finetune
        else settings.get("preview_interval", 100)
    )
    total_epochs = int(spec["total_epochs"])
    save_every = max(1, int(spec["save_every"]))
    warmup = 0 if finetune else int(settings["warmup_steps"])
    total_steps = total_epochs * len(loader)
    final_ratio = float(settings.get("lr_final_ratio", 1.0))
    aux_weight = float(settings.get("aux_mel_weight", 0.0))
    eval_interval = int(settings.get("eval_interval", 0))
    mean_ratio = float(settings.get("mean_flow_ratio", 0.25)) if mean_flow else 0.0
    # The bootstrapped target comes in once the network has a field to differentiate.
    mean_warmup = 0 if finetune else int(settings.get("mean_flow_warmup_steps", 0))
    backbone, mean_field = compiled_backbone(
        model, bool(spec.get("compile", False)), spec.get("torch_compile_mode", "default"), device,
        mean_flow=mean_ratio > 0,
    )
    # The backward runs outside autocast here, and a compiled graph's would
    # otherwise run under the forward's.
    backward_context = (
        partial(aot_config.patch, backward_pass_autocast="off") if backbone is not None else nullcontext
    )
    # Frozen parameters are left out of the gradient sync. Sampling, previews
    # and evaluation go through ``model`` itself.
    train_model = ranks.wrap(model)
    embedder = embedder_of(name)
    vocoder_path = spec.get("vocoder", "")
    vocoder = load_preview_vocoder(vocoder_path, config, device) if main_rank else None

    if main_rank:
        print_model_summary([("Flow", model)], title="Rectified flow")
        print_settings_panel(
            [
                ("Model", f"{name}, {speakers} speakers" + (", voice frozen" if voice_frozen else "")),
                ("Data", f"{len(dataset)} clips ({len(holdout_entries)} held out), batch {batch_size} x "
                         f"{settings['segment_frames']} frames, {len(loader)} steps per epoch"),
                ("Epochs", f"{epoch} -> {total_epochs}, saving every {save_every}, {starting_point}"),
                ("Backbone", "LYNXNet2"
                             + (" with adaLN" if model.backbone.voice is not None else "")
                             + (f", shallow from t={model.t_start:g}" if model.t_start > 0 else "")
                             + (", dual timestep" if model.dual_timestep else "")
                             + (f", mean flow on {100 * mean_ratio:g}%" if mean_ratio > 0 else "")
                             + (", compiled" if backbone is not None else "")),
                ("Training", f"{'Muon + AdamW' if isinstance(optimizer, MuonAdamW) else 'AdamW'}, lr {base_lr:g}, "
                             f"cosine to {final_ratio:g}x at step {total_steps}, {precision_label(amp_dtype)} on "
                             + (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
                             + (f" x {ranks.world} GPUs" if ranks.world > 1 else "")),
                ("Augment", f"key ±{settings.get('key_shift_range', 0):g} st "
                            f"({100 * settings.get('key_shift_prob', 0):g}%), "
                            "stretch x{:g}-{:g} ({:g}%), ".format(
                                *settings.get("time_stretch_range", (1, 1)), 100 * settings.get("time_stretch_prob", 0))
                            + f"speaker dropout {speaker_dropout:g}"
                            + (f", tension dropout {tension_dropout:g}" if tension_dropout > 0 else "")),
                ("Vocoder", os.path.basename(vocoder_path) if vocoder is not None else "none (previews without audio)"),
            ],
            title="Rectified flow",
        )

    def save(current_epoch: int):
        keep = spec.get("checkpoints", "latest")
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
                     "ema": ema.state_dict(), "epoch": current_epoch, "step": step,
                     "scaler": scaler.state_dict() if scaler is not None else None,
                     "amp_skipped_steps": skipped},
                    path,
                )
                if keep == "latest":
                    remove_older(out_dir, "F", path)
                saved.append(os.path.basename(path))
            export = os.path.join(out_dir, f"{name}_flow_{current_epoch}e_{step}s.pth")
            torch.save(
                {"kind": "rectified_flow", "config": config, "model": ema.cpu_state_dict(),
                 "speaker_count": speakers, "embedder_model": embedder,
                 "vocoder": vocoder_path if vocoder is not None else "",
                 "epoch": current_epoch, "step": step},
                export,
            )
        saved.append(os.path.basename(export))
        success(f"Saved {' and '.join(saved)}.", tag=TAG)

    @torch.no_grad()
    def preview():
        """The reference clip through the flow's EMA weights. With a vocoder,
        also its real mel through the vocoder alone, which separates the two
        models' errors; without one, the mel figure only."""
        inputs = reference.inputs.to(device)
        f0 = inputs.f0
        one_step = None
        with ema.applied(model):
            model.eval()
            generated = model.sample(inputs, steps=PREVIEW_STEPS)
            if mean_ratio > 0:
                one_step = model.sample(inputs, steps=1, method="mean")
            model.train()
        if vocoder is None:
            previews.log(
                epoch, step, reference.path,
                generated_mel=denormalize_mel(generated, config["data"]), reference_mel=reference.mel,
            )
            return
        previews.log(
            epoch, step, reference.path,
            generated_audio=vocoder(generated, f0),
            reference_audio=reference.audio.to(device),
            extra_audio={
                "vocoder_on_real_mel": vocoder(normalize_mel(reference.mel.to(device), config["data"]), f0),
                **({} if one_step is None else {"mean_flow_1_step": vocoder(one_step, f0)}),
            },
        )

    @torch.no_grad()
    def evaluate():
        """Held-out flow loss through the EMA weights, from the same noise and
        at the same times on every call, so the curve compares across steps."""
        totals = torch.zeros(len(EVAL_FRACTIONS), device=device)
        aux_total, items = 0.0, 0
        with ema.applied(model):
            model.eval()
            for index, (mel, inputs) in enumerate(holdout):
                inputs = inputs.to(device)
                mel = normalize_mel(mel.to(device), config["data"]) * inputs.mask
                generator = torch.Generator(device=device).manual_seed(index)
                noise = torch.randn(mel.shape, device=device, generator=generator)
                with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    losses, aux = model.validation_losses(mel, inputs, noise, EVAL_FRACTIONS)
                totals += losses.float() * mel.shape[0]
                aux_total += (aux.item() if aux is not None else 0.0) * mel.shape[0]
                items += mel.shape[0]
            model.train()
        totals /= max(1, items)
        writer.add_scalar("val/flow", totals.mean().item(), step)
        for fraction, value in zip(EVAL_FRACTIONS, totals.tolist()):
            writer.add_scalar(f"val/flow_t{fraction:g}", value, step)
        if model.aux is not None:
            writer.add_scalar("val/aux_mel_l1", aux_total / max(1, items), step)

    recorder = EpochRecorder()
    model.train()
    data_wait = step_time = 0.0
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

                lr = learning_rate(base_lr, step, warmup, total_steps, final_ratio)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with backward_context(), torch.autocast(
                    device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
                ):
                    flow_loss, aux_loss, mean_losses = train_model(
                        mel * inputs.mask, inputs,
                        speaker_dropout=speaker_dropout, tension_dropout=tension_dropout,
                        backbone=backbone, mean_field=mean_field, mean_ratio=mean_ratio,
                        mean_bootstrap=min(1.0, step / mean_warmup) if mean_warmup else 1.0,
                    )
                    loss = flow_loss
                    if mean_losses is not None:
                        loss = (1.0 - mean_ratio) * flow_loss + mean_ratio * mean_losses.objective
                    if aux_loss is not None:
                        loss = loss + aux_weight * aux_loss
                grad_norm, step_skipped = optimizer_step(loss, model, optimizer, scaler, settings["grad_clip"])
                skipped += int(step_skipped)
                ema.update(model)
                step += 1
                step_time += time.perf_counter() - step_started

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        shown = flow_loss if mean_losses is None else mean_losses.flow
                        metrics = f"loss={shown.item():.4f}"
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)

                if step % LOG_INTERVAL == 0:
                    # Collective, so outside the rank check.
                    scalars = loss_scalars(ranks, flow_loss, aux_loss, mean_losses)
                if step % LOG_INTERVAL == 0 and main_rank:
                    # Time the loop spent waiting on the loader; near zero when
                    # the workers keep up.
                    scalars["perf/data_wait_ms"] = 1000 * data_wait / LOG_INTERVAL
                    scalars["perf/step_ms"] = 1000 * step_time / LOG_INTERVAL
                    data_wait = step_time = 0.0
                    scalars["grad_norm"] = grad_norm.item()
                    scalars.update(conditioning_norms(model))
                    scalars["lr"] = lr
                    if scaler is not None:
                        scalars["amp/scale"] = scaler.get_scale()
                        scalars["amp/skipped_steps"] = skipped
                    for tag, value in scalars.items():
                        writer.add_scalar(tag, value, step)
                if reference is not None and step % preview_interval == 0:
                    preview()
                if holdout is not None and eval_interval and step % eval_interval == 0:
                    evaluate()
                if ranks.stop_requested():
                    # Nothing is being written at a batch boundary.
                    finish_stop(writer)
                fetch_started = time.perf_counter()

        if main_rank:
            print(f"{name} | epoch={epoch} | step={step} | {recorder.record()}")
            if skipped and scaler is not None:
                info(f"GradScaler at {scaler.get_scale():.0f}; {skipped} step(s) skipped so far.", tag=TAG)
            if epoch % save_every == 0 or epoch == total_epochs:
                save(epoch)
                if reference is not None:
                    preview()
            writer.flush()
        epoch += 1

    if main_rank:
        writer.close()
        success("Flow training finished.", tag=TAG)


if __name__ == "__main__":
    main(sys.argv[1])
