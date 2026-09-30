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
from functools import partial

sys.path.append(os.getcwd())

import torch
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
    ClipCurves,
    RectifiedDataset,
    amp_setup,
    check_pretrain_embedder,
    collate_flow,
    embedder_of,
    latest_checkpoint,
    load_run_config,
    measure_clip_curves,
    precision_label,
    pretrained_weights,
    read_filelist,
    run_dir,
    remove_older,
    speaker_count,
    split_holdout,
)
from rvc.rectified.distributed import Ranks, launch, parse_gpus
from rvc.rectified.flow_model import build_flow, resize_speakers
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


def compiled_backbone(model, enabled: bool, mode: str, device):
    """``torch.compile`` of the backbone for training, or None. Evaluation and
    sampling keep the eager module, whose shapes vary."""
    if not enabled:
        return None
    if device.type != "cuda" or importlib.util.find_spec("triton") is None:
        warning("torch.compile needs CUDA and Triton; training uncompiled.", tag=TAG)
        return None
    return torch.compile(model.backbone, mode=mode)


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


def main(spec_path: str) -> None:
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    install_stop_handlers()
    # Once, before the ranks start, which would otherwise wait on it.
    name = spec["model_name"]
    config = load_run_config(name)
    measure_clip_curves(
        name, read_filelist(name), config["data"]["sample_rate"],
        loader_workers(config["flow"].get("num_workers", 4)),
    )
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

    entries = read_filelist(name)
    speakers = speaker_count(entries)
    segment = int(settings["segment_frames"])
    workers = loader_workers(settings.get("num_workers", 4))
    curves = ClipCurves(name)
    entries, holdout_entries = split_holdout(entries, int(settings.get("holdout_clips", 0)))
    dataset = RectifiedDataset(entries, config, "flow", segment, curves=curves)
    # Per GPU, as in the RVC trainer.
    batch_size = int(spec["batch_size"])
    if len(dataset) // ranks.world < batch_size:
        raise ValueError(
            f"{len(dataset)} clips is fewer than one batch of {batch_size} on each of {ranks.world} GPU(s)."
        )
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
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    holdout = None
    if holdout_entries and main_rank:
        holdout = DataLoader(
            RectifiedDataset(holdout_entries, config, "flow", segment, augment=False, curves=curves),
            batch_size=min(batch_size, len(holdout_entries)),
            num_workers=min(2, workers),
            collate_fn=collate,
        )

    model = build_flow(config, speakers).to(device)
    finetune = bool(spec.get("pretrained_flow"))
    speaker_dropout = float(settings["speaker_dropout"])
    voice_frozen = finetune and speakers == 1 and settings.get("finetune_freeze_voice", True)
    if voice_frozen:
        freeze_voice(model)
        speaker_dropout = 0.0
    base_lr = settings["finetune_learning_rate"] if finetune else settings["learning_rate"]
    if spec.get("learning_rate"):
        base_lr = float(spec["learning_rate"])
    optimizer_name = settings.get("optimizer", "adamw")
    if optimizer_name == "muon":
        optimizer = MuonAdamW(
            model, base_lr, muon_weight_decay=settings["weight_decay"],
            betas=tuple(settings["betas"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), base_lr, betas=tuple(settings["betas"]),
            weight_decay=settings["weight_decay"],
        )
    amp_dtype, scaler = amp_setup(spec.get("precision", "fp32"), device, TAG)

    epoch, step, skipped = 1, 0, 0
    # A fine-tune is too short for the pretrain's 10k-step horizon.
    ema_decay = settings.get("finetune_ema_decay", settings["ema_decay"]) if finetune else settings["ema_decay"]
    ema = WeightEMA(model, ema_decay)
    starting_point = "scratch"
    resume = None if spec.get("fresh") else latest_checkpoint(out_dir, "F")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"], model)
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        epoch, step = state["epoch"] + 1, state["step"]
        skipped = int(state.get("amp_skipped_steps", 0))
        starting_point = f"resumed from {os.path.basename(resume)}"
    elif finetune:
        check_pretrain_embedder(spec["pretrained_flow"], embedder_of(name))
        weights = resize_speakers(pretrained_weights(spec["pretrained_flow"]), speakers)
        model.load_state_dict(weights)
        ema.reseed(model)
        starting_point = f"fine-tune from {os.path.basename(spec['pretrained_flow'])}"

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
    backbone = compiled_backbone(
        model, bool(spec.get("compile", False)), spec.get("torch_compile_mode", "default"), device
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
                             + (", compiled" if backbone is not None else "")),
                ("Training", f"{'Muon + AdamW' if optimizer_name == 'muon' else 'AdamW'}, lr {base_lr:g}, "
                             f"cosine to {final_ratio:g}x at step {total_steps}, {precision_label(amp_dtype)} on "
                             + (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
                             + (f" x {ranks.world} GPUs" if ranks.world > 1 else "")),
                ("Augment", f"key ±{settings.get('key_shift_range', 0):g} st "
                            f"({100 * settings.get('key_shift_prob', 0):g}%), "
                            "stretch x{:g}-{:g} ({:g}%), ".format(
                                *settings.get("time_stretch_range", (1, 1)), 100 * settings.get("time_stretch_prob", 0))
                            + f"speaker dropout {speaker_dropout:g}"),
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
        ref_mel, content, f0, energy, breathiness, audio, sid, path = reference
        content, f0, energy = content.to(device), f0.to(device), energy.to(device)
        mask = torch.ones(1, 1, f0.shape[1], device=device)
        speaker = torch.tensor([sid], device=device)
        with ema.applied(model):
            model.eval()
            generated = model.sample(
                content, f0, energy, speaker, mask, steps=PREVIEW_STEPS,
                breathiness=breathiness.to(device),
            )
            model.train()
        if vocoder is None:
            previews.log(
                epoch, step, path,
                generated_mel=denormalize_mel(generated, config["data"]), reference_mel=ref_mel,
            )
            return
        previews.log(
            epoch, step, path,
            generated_audio=vocoder(generated, f0),
            reference_audio=audio.to(device),
            extra_audio={
                "vocoder_on_real_mel": vocoder(normalize_mel(ref_mel.to(device), config["data"]), f0)
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
            for index, batch in enumerate(holdout):
                mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask = (
                    item.to(device) for item in batch
                )
                mel = normalize_mel(mel, config["data"]) * mask
                generator = torch.Generator(device=device).manual_seed(index)
                noise = torch.randn(mel.shape, device=device, generator=generator)
                with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    losses, aux = model.validation_losses(
                        mel, content, f0, energy, speaker, mask, breathiness, key_shift, speed,
                        noise, EVAL_FRACTIONS,
                    )
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
            for batch_index, batch in enumerate(loader):
                mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask = batch
                step_started = time.perf_counter()
                data_wait += step_started - fetch_started
                mel = normalize_mel(mel.to(device, non_blocking=True), config["data"])
                content = content.to(device, non_blocking=True)
                f0 = f0.to(device, non_blocking=True)
                energy = energy.to(device, non_blocking=True)
                breathiness = breathiness.to(device, non_blocking=True)
                key_shift = key_shift.to(device, non_blocking=True)
                speed = speed.to(device, non_blocking=True)
                speaker = speaker.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                lr = learning_rate(base_lr, step, warmup, total_steps, final_ratio)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    flow_loss, aux_loss = train_model(
                        mel * mask, content, f0, energy, speaker, mask,
                        speaker_dropout=speaker_dropout,
                        breathiness=breathiness, key_shift=key_shift, speed=speed, backbone=backbone,
                    )
                    loss = flow_loss if aux_loss is None else flow_loss + aux_weight * aux_loss
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    if scaler.get_scale() < scale:
                        skipped += 1
                ema.update(model)
                step += 1
                step_time += time.perf_counter() - step_started

                if main_rank:
                    if not metrics or (batch_index + 1) % METRICS_INTERVAL == 0:
                        metrics = f"loss={flow_loss.item():.4f}"
                    progress.update(task, advance=1, metrics=metrics)
                    emit_machine_progress(epoch, total_epochs, batch_index + 1, len(loader), step, metrics, 0)

                if step % LOG_INTERVAL == 0:
                    # Collective, so outside the rank check.
                    logged_flow = ranks.mean(flow_loss)
                    logged_aux = ranks.mean(aux_loss) if aux_loss is not None else None
                if step % LOG_INTERVAL == 0 and main_rank:
                    writer.add_scalar("loss/flow", logged_flow.item(), step)
                    if logged_aux is not None:
                        writer.add_scalar("loss/aux_mel_l1", logged_aux.item(), step)
                    # Time the loop spent waiting on the loader; near zero when
                    # the workers keep up.
                    writer.add_scalar("perf/data_wait_ms", 1000 * data_wait / LOG_INTERVAL, step)
                    writer.add_scalar("perf/step_ms", 1000 * step_time / LOG_INTERVAL, step)
                    data_wait = step_time = 0.0
                    writer.add_scalar("grad_norm", grad_norm.item(), step)
                    for tag, value in conditioning_norms(model).items():
                        writer.add_scalar(tag, value, step)
                    writer.add_scalar("lr", lr, step)
                    if scaler is not None:
                        writer.add_scalar("amp/scale", scaler.get_scale(), step)
                        writer.add_scalar("amp/skipped_steps", skipped, step)
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
