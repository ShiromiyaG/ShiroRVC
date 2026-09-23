"""Training loop shared by the pretrain and the per-voice fine-tune."""

from __future__ import annotations

import copy
import functools
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch

from . import checkpoint
from .augment import AugmentConfig
from .data import CODEBOOK, clip_paths, pooled_descriptors, read_manifest
from .dataset import MixedBatchSampler, StyleDataset, collate, parse_speakers, scan_clips
from .descriptors import DESCRIPTOR_NAMES, DescriptorConfig, analyze, descriptor_vector, pool
from .discriminator import VibratoDiscriminator, discriminator_loss, generator_loss
from .evaluation import descriptor_distance, melody_error_cents
from .f0_repr import Normalizer, ReprConfig, compose_f0
from .flow import flow_loss, sample
from .frontend import VOICING_THRESHOLD
from .lora import apply_lora, merge_lora
from .model import ModelConfig, StyleDiT
from .units import UnitCodebook

CHECKPOINT_DIR = "checkpoints"
VAL_LIST = "val_clips.txt"
METRICS_FILE = "val_metrics.jsonl"


class Setup:
    """What a run is trained on, read from a style dataset."""

    def __init__(self, data_dir: str):
        manifest = read_manifest(data_dir)
        self.data_dir = data_dir
        self.rcfg = ReprConfig.from_dict(manifest["representation"])
        self.dcfg = DescriptorConfig.from_dict(manifest["descriptor_config"])
        self.normalizer = Normalizer.from_dict(manifest["normalizer"])
        self.embedder = manifest.get("embedder", "contentvec")
        self.voicing_threshold = float(manifest.get("voicing_threshold", VOICING_THRESHOLD))
        self.codebook = UnitCodebook.load(os.path.join(data_dir, manifest.get("codebook", CODEBOOK)))
        self.speaker_descriptors = manifest.get("speaker_descriptors", {})


def _split(paths, speakers, speech, n_val, out_dir, seed):
    """Validation clips are singing only and kept across resumes."""
    val_file = os.path.join(out_dir, VAL_LIST)
    names = [os.path.basename(p) for p in paths]
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        val = [i for i, n in enumerate(names) if n in wanted]
    else:
        singing = [i for i, s in enumerate(speakers) if s not in speech]
        rng = np.random.default_rng(seed)
        val = sorted(rng.choice(singing, min(n_val, len(singing) // 10), replace=False).tolist()) if singing else []
        with open(val_file, "w", encoding="utf-8") as f:
            f.write("".join(names[i] + "\n" for i in val))
    val_set = set(val)
    return [i for i in range(len(paths)) if i not in val_set], val


def _lr_lambda(warmup, total, min_ratio):
    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _autocast(dtype):
    return torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


def _precision(tcfg, device):
    """Autocast dtype for ``train.precision`` (bf16, fp16 or fp32); None is fp32."""
    from rvc.lib.terminal import warning

    name = str(tcfg.get("precision", "bf16")).lower()
    if not str(device).startswith("cuda"):
        return "fp32", None
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        warning("This GPU has no BF16; training in FP16 instead.", tag="[STYLE]")
        name = "fp16"
    tf32 = bool(tcfg.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return name, {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name)


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


#: Batch lengths are rounded up to this when compiling: one static graph per
#: rounded length, nine between the 5 s and 15 s clips (512 to 1536).
COMPILE_PAD = 128


def _compiled(model):
    """``model``'s forward behind ``torch.compile``, back to eager for good if
    compiling fails.  Static graphs: marking the length dynamic still ended in
    one graph per length, which ran past Dynamo's limit of 8."""
    from rvc.lib.terminal import warning

    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
    compiled = torch.compile(model, dynamic=False)
    failed = False

    def forward(*args, **kwargs):
        nonlocal failed
        if not failed:
            try:
                return compiled(*args, **kwargs)
            except Exception as error:
                failed = True
                warning(f"Compiling the style model failed; training eager from here. {error}", tag="[STYLE]")
        return model(*args, **kwargs)

    return forward


def _latest_checkpoint(out_dir):
    found = sorted(glob.glob(os.path.join(out_dir, CHECKPOINT_DIR, "step_*.pt")))
    return found[-1] if found else None


class Validator:
    """Fixed-noise validation loss, sampled contours and the descriptor
    response test, on the same clips every time."""

    def __init__(self, setup: Setup, paths, vcfg: dict, max_frames: int, device, style_descriptors=None, source_acfg=None,
                 loudness=False):
        """With ``style_descriptors`` (raw values, ``None`` for unknown) every
        clip is generated with those, as inference does with a singer's.  A
        converter rewrites each clip re-sung by ``source_acfg``, the same draw
        every time."""
        self.setup, self.vcfg, self.device = setup, vcfg, device
        self.style = None
        if style_descriptors is not None:
            raw = np.array([np.nan if v is None else v for v in style_descriptors], dtype=np.float32)
            present = np.isfinite(raw)
            self.style = (
                torch.from_numpy(np.where(present, setup.normalizer.encode_descriptors(raw), 0.0).astype(np.float32)),
                torch.from_numpy(present.astype(np.float32)),
            )
        ds = StyleDataset(
            paths, setup.normalizer, setup.rcfg, setup.dcfg, max_frames=max_frames, random_crop=False,
            source_acfg=source_acfg, seed=1234, loudness=loudness,
        )
        self.items = [ds[i] for i in range(len(ds))]
        self.raw = []
        for path in paths[: vcfg["sample_clips"]]:
            with np.load(path, allow_pickle=False) as data:
                n = min(len(data["vuv"]), max_frames)
                self.raw.append({k: data[k][:n] for k in ("f0", "coarse", "residual", "vuv")})
        self.names = [os.path.basename(p) for p in paths[: vcfg["sample_clips"]]]

    @torch.no_grad()
    def loss(self, model, batch_size, dtype):
        gen = torch.Generator(device=self.device).manual_seed(1234)
        total = n = 0.0
        for i in range(0, len(self.items), batch_size):
            batch = _to(collate(self.items[i : i + batch_size]), self.device)
            B = batch["target"].shape[0]
            t = torch.rand(B, device=self.device, generator=gen)
            noise = torch.randn(batch["target"].shape, device=self.device, generator=gen)
            with _autocast(dtype):
                _, per_sample, _, _, _ = flow_loss(model, batch, t=t, noise=noise)
            total += per_sample.sum().item()
            n += B
        return total / max(n, 1)

    @torch.no_grad()
    def _generate(self, model, batch, dtype, desc=None, desc_mask=None):
        gen = torch.Generator(device=self.device).manual_seed(4321)
        with _autocast(dtype):
            x = sample(
                model, batch["units"], batch["coarse"],
                batch["desc"] if desc is None else desc,
                batch["desc_mask"] if desc_mask is None else desc_mask,
                batch["mask"], steps=self.vcfg["steps"], cfg_scale=self.vcfg["cfg_scale"],
                cfg_drop=self.vcfg.get("cfg_drop", "descriptors"), generator=gen, convert=batch.get("source"),
                loudness=batch.get("loudness"),
            )
        out = []
        for i, raw in enumerate(self.raw):
            res, vuv = self.setup.normalizer.decode_target(x[i, :, : len(raw["vuv"])].cpu().numpy())
            f0 = compose_f0(raw["coarse"], res, raw["vuv"])
            out.append({"f0": f0, "residual": res, "vuv": vuv})
        return out

    def _pooled(self, contours):
        rcfg, dcfg = self.setup.rcfg, self.setup.dcfg
        return descriptor_vector(pool(analyze(c["f0"], rcfg, dcfg, c["residual"]) for c in contours))

    def run(self, model, batch_size, dtype):
        rcfg, norm = self.setup.rcfg, self.setup.normalizer
        metrics = {"val/loss": self.loss(model, batch_size, dtype)}
        batch = _to(collate(self.items[: len(self.raw)]), self.device)
        if self.style is not None:
            B = batch["desc"].shape[0]
            batch["desc"] = self.style[0].to(self.device).expand(B, -1).clone()
            batch["desc_mask"] = self.style[1].to(self.device).expand(B, -1).clone()
        generated = self._generate(model, batch, dtype)

        errs = [melody_error_cents(g["f0"], r["f0"], rcfg) for g, r in zip(generated, self.raw)]
        frames = sum(e["frames"] for e in errs)
        metrics["val/melody_error_cents"] = sum(e["mean"] * e["frames"] for e in errs if e["frames"]) / max(frames, 1)
        metrics["val/vuv_accuracy"] = float(np.mean(np.concatenate([g["vuv"] == r["vuv"] for g, r in zip(generated, self.raw)])))
        gt_desc = self._pooled([{"f0": r["f0"], "residual": r["residual"]} for r in self.raw])
        gen_desc = self._pooled(generated)
        metrics["val/descriptor_distance"] = descriptor_distance(gen_desc, gt_desc, norm)
        for name, g, r in zip(DESCRIPTOR_NAMES, gen_desc, gt_desc):
            metrics[f"val_desc/{name}/generated"] = float(g)
            metrics[f"val_desc/{name}/ground_truth"] = float(r)

        # Each group moves its descriptors together, ``sign`` times the offset:
        # moving one alone leaves its partners (e.g. the drop fraction) contradicting it.
        offset = float(self.vcfg.get("response_offset", 1.5))
        for group, members in self.vcfg.get("response_groups", {}).items():
            cols = {name: (DESCRIPTOR_NAMES.index(name), float(sign)) for name, sign in members.items()}
            for label, value in (("low", -offset), ("high", offset)):
                desc, mask = batch["desc"].clone(), batch["desc_mask"].clone()
                for j, sign in cols.values():
                    desc[:, j], mask[:, j] = sign * value, 1.0
                pooled = self._pooled(self._generate(model, batch, dtype, desc, mask))
                for name, (j, sign) in cols.items():
                    requested = norm.decode_descriptors(np.full(len(DESCRIPTOR_NAMES), sign * value))[j]
                    metrics[f"response/{group}/{name}/{label}"] = float(pooled[j])
                    metrics[f"response/{group}/{name}/{label}_requested"] = float(requested)
        return metrics, generated

    def plot(self, generated, path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # Clip names can be CJK; fall back to an installed CJK font, and don't
        # warn per glyph when there is none (TensorBoard redraws the figure too).
        installed = {f.name for f in font_manager.fontManager.ttflist}
        cjk = [f for f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "Microsoft YaHei", "MS Gothic") if f in installed]
        plt.rcParams["font.family"] = ["DejaVu Sans", *cjk]
        warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

        n = min(self.vcfg.get("plot_clips", 3), len(generated))
        fr = self.setup.rcfg.frame_rate
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), squeeze=False)
        for ax, raw, gen, name in zip(axes[:, 0], self.raw, generated, self.names):
            t = np.arange(len(raw["f0"])) / fr
            ax.plot(t, np.where(raw["f0"] > 0, raw["f0"], np.nan), color="0.5", lw=1.2, label="ground truth")
            ax.plot(t, np.where(gen["f0"] > 0, gen["f0"], np.nan), color="C0", lw=1.1, label="generated")
            ax.plot(t, np.where(raw["vuv"], raw["coarse"], np.nan), color="0.15", lw=0.9, ls="--", label="coarse")
            ax.set_yscale("log")
            ax.set_title(name)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=100)
        return fig


def train(cfg: dict, data_dir: str, out_dir: str, *, device="cuda:0", init=None, kind="base",
          export_name="style_base.pt", overfit=0, progress=None):
    """Train on the style dataset in ``data_dir``; checkpoints, TensorBoard
    logs, plots and the exported model go to ``out_dir``.  ``init`` is a
    style checkpoint to start from; ``overfit`` > 0 trains and validates on
    that many clips with no augmentation or dropout.  Resumes from the last
    checkpoint in ``out_dir``."""
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from rvc.lib.terminal import info, progress_task, success, warning

    tcfg, vcfg, dcfg_data = cfg["train"], cfg["validation"], cfg.get("data", {})
    os.makedirs(os.path.join(out_dir, CHECKPOINT_DIR), exist_ok=True)
    torch.manual_seed(int(tcfg.get("seed", 0)))
    setup = Setup(data_dir)

    base = checkpoint.load(init) if init else None
    if base is not None:
        if base.embedder != setup.embedder or not np.array_equal(base.codebook.centroids, setup.codebook.centroids):
            raise ValueError("The dataset was not built with this base's embedder and codebook (extract with --reuse).")
        mcfg = base.model.cfg
    else:
        mcfg = ModelConfig.from_dict({
            **cfg.get("model", {}), "n_units": setup.codebook.size, "n_descriptors": len(DESCRIPTOR_NAMES),
        })

    paths = clip_paths(data_dir)
    loudness = bool(mcfg.loudness_channels)
    if loudness:
        with np.load(paths[0], allow_pickle=False) as data:
            if "loudness" not in data.files:
                raise ValueError(
                    f"{data_dir} has no loudness, which this model takes; extract it again into a new folder."
                )
    lengths, speakers = scan_clips(paths)
    speech = parse_speakers(dcfg_data.get("speech_speakers"))
    max_frames = int(dcfg_data.get("max_frames", 1500))
    if overfit:
        pick = [i for i, s in enumerate(speakers) if s not in speech][:overfit]
        train_idx, val_idx = pick, pick
    else:
        train_idx, val_idx = _split(paths, speakers, speech, int(dcfg_data.get("val_clips", 64)), out_dir, int(tcfg.get("seed", 0)))
    if not val_idx:
        warning("Too few clips for a validation set; validating on training clips.", tag="[STYLE]")
        val_idx = train_idx[: int(dcfg_data.get("val_clips", 64))]

    acfg = None if overfit else AugmentConfig.from_dict(cfg.get("augment"))
    ccfg = cfg.get("converter") or {}
    source_acfg = AugmentConfig.from_dict(ccfg.get("degrade")) if mcfg.source_channels else None
    dataset = StyleDataset(
        paths, setup.normalizer, setup.rcfg, setup.dcfg, acfg, max_frames,
        0.0 if overfit else float(tcfg.get("descriptor_dim_dropout", 0.0)),
        source_acfg=source_acfg, source_dropout=0.0 if overfit else float(ccfg.get("source_dropout", 0.1)),
        loudness=loudness,
    )
    singing = [i for i in train_idx if speakers[i] not in speech]
    speech_idx = [i for i in train_idx if speakers[i] in speech]
    batch_size = min(int(tcfg["batch_size"]), len(train_idx)) if overfit else int(tcfg["batch_size"])
    sampler = MixedBatchSampler(
        singing, speech_idx, lengths, batch_size, float(dcfg_data.get("max_speech_fraction", 0.0)),
        pool=1 if overfit else 50, seed=int(tcfg.get("seed", 0)),
    )
    workers = int(tcfg.get("num_workers", 4))
    use_compile = bool(tcfg.get("compile", False)) and str(device).startswith("cuda")
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=functools.partial(collate, pad_multiple=COMPILE_PAD if use_compile else 1),
        num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
    )

    model = StyleDiT(mcfg).to(device)
    model.grad_checkpointing = bool(tcfg.get("checkpointing", False))
    if base is not None:
        model.load_state_dict(base.model.state_dict())
    lcfg = cfg.get("lora") or {}
    use_lora = bool(lcfg.get("enabled", False))
    if use_lora:
        apply_lora(model, int(lcfg["rank"]), float(lcfg["alpha"]), lcfg["targets"], lcfg.get("train_extra", ()))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    # Training forward only: validation samples the EMA, eagerly.
    train_model = model
    if use_compile:
        if model.grad_checkpointing:
            warning("Compiling is off with gradient checkpointing.", tag="[STYLE]")
        else:
            info("Compiling the style model; the first steps take longer.", tag="[STYLE]")
            train_model = _compiled(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    ema_trainable = [pe for pe, p in zip(ema.parameters(), model.parameters()) if p.requires_grad]
    groups = [{"params": trainable, "lr": float(tcfg["lr"])}]
    if use_lora:
        # Adapters at the LoRA rate; train_extra weights are trained in full,
        # so they keep the full fine-tune's.
        adapter = [p for n, p in model.named_parameters() if p.requires_grad and (".down." in n or ".up." in n)]
        extra = [p for n, p in model.named_parameters() if p.requires_grad and not (".down." in n or ".up." in n)]
        groups = [{"params": adapter, "lr": float(lcfg.get("lr", tcfg["lr"]))}]
        if extra:
            groups.append({"params": extra, "lr": float(tcfg["lr"])})
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.99), weight_decay=float(tcfg.get("weight_decay", 0.0)),
        fused=str(device).startswith("cuda"),
    )
    precision, amp_dtype = _precision(tcfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    total_steps = int(tcfg["steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(int(tcfg.get("warmup_steps", 0)), total_steps, float(tcfg.get("min_lr_ratio", 0.1)))
    )

    gcfg = cfg.get("discriminator") or {}
    disc = opt_d = None
    if gcfg.get("enabled", False) and not overfit:
        disc = VibratoDiscriminator(
            setup.dcfg.vibrato_band_hz, setup.rcfg.frame_rate, tuple(gcfg.get("periods", (2, 3, 5, 7, 11))),
            cond_dim=2 * mcfg.n_descriptors,
        ).to(device)
        opt_d = torch.optim.AdamW(
            disc.parameters(), lr=float(gcfg.get("lr", 2e-4)), betas=(0.8, 0.99), fused=str(device).startswith("cuda")
        )
    adv_start, t_min = int(gcfg.get("start_step", 0)), float(gcfg.get("t_min", 0.5))
    adv_weight, fm_weight = float(gcfg.get("adv_weight", 0.05)), float(gcfg.get("fm_weight", 0.1))

    step = 0
    resume = _latest_checkpoint(out_dir)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        if disc is not None:
            loaded = disc.load_state_dict(state["disc"], strict=False) if "disc" in state else None
            if loaded is not None and not loaded.missing_keys and not loaded.unexpected_keys:
                opt_d.load_state_dict(state["opt_d"])
            else:
                # Added or changed since this checkpoint: what matches is kept,
                # the rest starts at its init, and the discriminator gets its
                # head start again before the model feels it.
                adv_start = step + adv_start
                warning(f"The discriminator is new to this checkpoint; the model feels it from step {adv_start}.", tag="[STYLE]")
        info(f"Resumed from {resume} at step {step}.", tag="[STYLE]")
        sampler.seed += step

    params = sum(p.numel() for p in model.parameters())
    if speech_idx:
        clips = f"{len(singing)} singing + {len(speech_idx)} speech clips for training"
        batch_note = f"batch {batch_size} ({sampler.n_speech} speech)"
    else:
        clips, batch_note = f"{len(singing)} clips for training", f"batch {batch_size}"
    info(
        f"{'Converter' if mcfg.source_channels else 'Generator'}{' with loudness' if loudness else ''}, "
        f"{params / 1e6:.1f}M parameters ({sum(p.numel() for p in trainable) / 1e6:.2f}M trained), {precision}"
        f"{' + TF32' if torch.backends.cuda.matmul.allow_tf32 else ''}; {clips}, "
        f"{len(val_idx)} for validation; {batch_note}.",
        tag="[STYLE]",
    )

    style_descriptors = pooled_descriptors([paths[i] for i in singing + list(val_idx)]) if kind != "base" else None
    validator = Validator(
        setup, [paths[i] for i in val_idx], vcfg, max_frames, device,
        style_descriptors if vcfg.get("condition", "clip") == "style" else None,
        source_acfg=source_acfg, loudness=loudness,
    )
    writer = SummaryWriter(out_dir)
    ema_decay = float(tcfg.get("ema_decay", 0.999))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every, val_every = int(tcfg.get("log_every", 50)), int(tcfg.get("val_every", 5000))
    keep = int(tcfg.get("keep_checkpoints", 3))
    unit_dropout = 0.0 if overfit else float(tcfg.get("unit_dropout", 0.15))
    desc_dropout = 0.0 if overfit else float(tcfg.get("descriptor_dropout", 0.15))

    def save():
        path = os.path.join(out_dir, CHECKPOINT_DIR, f"step_{step:08d}.pt")
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "config": cfg}
        if disc is not None:
            state.update(disc=disc.state_dict(), opt_d=opt_d.state_dict())
        torch.save(state, path)
        for old in sorted(glob.glob(os.path.join(out_dir, CHECKPOINT_DIR, "step_*.pt")))[:-keep]:
            os.remove(old)
        checkpoint.export(
            os.path.join(out_dir, export_name), merge_lora(copy.deepcopy(ema)) if use_lora else ema,
            representation=setup.rcfg, descriptor_config=setup.dcfg, normalizer=setup.normalizer,
            embedder=setup.embedder, codebook=setup.codebook, kind=kind, step=step,
            style_descriptors=style_descriptors, voicing_threshold=setup.voicing_threshold,
        )

    def validate():
        metrics, generated = validator.run(ema, batch_size, amp_dtype)
        for key, value in metrics.items():
            if np.isfinite(value):
                writer.add_scalar(key, value, step)
        fig = validator.plot(generated, os.path.join(out_dir, "plots", f"step_{step:08d}.png"))
        writer.add_figure("val/contours", fig, step)
        with open(os.path.join(out_dir, METRICS_FILE), "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **{k: (v if np.isfinite(v) else None) for k, v in metrics.items()}}) + "\n")
        info(
            f"step {step}: val loss {metrics['val/loss']:.4f}, melody error {metrics['val/melody_error_cents']:.1f} c, "
            f"descriptor distance {metrics['val/descriptor_distance']:.3f}, vuv acc {metrics['val/vuv_accuracy']:.3f}",
            tag="[STYLE]",
        )
        if progress is not None:
            progress(step, total_steps, metrics)
        # Validation's CFG batches are a different size from training's; let the
        # training steps have that memory back.
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if base is not None and step == 0:
        # The base before any fine-tuning, as the baseline at step 0.
        validate()

    model.train()
    t_mean, t_std = float(tcfg.get("t_logit_mean", 0.0)), float(tcfg.get("t_logit_std", 1.0))
    # Accumulated on the device and read once per log_every: a per-step
    # .item() stalls the CPU until the GPU catches up.
    loss_sum = torch.zeros((), device=device)
    # Discriminator, adversarial and feature-matching losses, and their count.
    gan_sum, gan_n = torch.zeros(3, device=device), torch.zeros((), device=device)
    bin_sum, bin_n = torch.zeros(4, device=device), torch.zeros(4, device=device)
    window, tic = 0, time.time()
    # Rich draws nothing when stdout is piped, so the step lines stay there.
    live, metrics = sys.stdout.isatty(), ""
    try:
        with progress_task(total_steps, "Style", initial=step, leave=True, training=True) as (bar, task_id):
            for batch in loader:
                if step >= total_steps:
                    break
                batch = _to(batch, device)
                with _autocast(amp_dtype):
                    loss, per_sample, t, estimate, drop_desc = flow_loss(
                        train_model, batch, unit_dropout=unit_dropout, descriptor_dropout=desc_dropout, t_mean=t_mean, t_std=t_std
                    )
                # train/loss stays the flow loss alone.
                total = loss
                if disc is not None:
                    # Estimates from early t are rightly an average; only later ones should
                    # sound real.  Nor is a sample drawn without descriptors held to them.
                    weight = ((t >= t_min) & ~drop_desc).float()
                    voiced = (batch["target"][:, 1] > 0) & batch["mask"]
                    cond = torch.cat([batch["desc"] * batch["desc_mask"], batch["desc_mask"]], dim=-1)
                    real, fake = batch["target"][:, 0], estimate[:, 0]
                    with _autocast(amp_dtype):
                        loss_d = discriminator_loss(disc(real, voiced, cond), disc(fake.detach(), voiced, cond), weight)
                    opt_d.zero_grad(set_to_none=True)
                    scaler.scale(loss_d).backward()
                    scaler.step(opt_d)
                    adv = fm = torch.zeros((), device=device)
                    if step >= adv_start:
                        with _autocast(amp_dtype):
                            with torch.no_grad():
                                real_out = disc(real, voiced, cond)
                            adv, fm = generator_loss(real_out, disc(fake, voiced, cond), weight)
                        total = loss + adv_weight * adv + fm_weight * fm
                    with torch.no_grad():
                        gan_sum += torch.stack([loss_d.detach(), adv.detach(), fm.detach()]).float()
                        gan_n += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                with torch.no_grad():
                    torch._foreach_lerp_(ema_trainable, trainable, 1.0 - ema_decay)
                    loss_sum += loss.detach()
                    bins = (t * 4).long().clamp_(max=3)
                    bin_sum.index_add_(0, bins, per_sample.detach().float())
                    bin_n.index_add_(0, bins, torch.ones_like(bin_sum[bins]))
                step += 1
                window += 1

                if step % log_every == 0:
                    elapsed = time.time() - tic
                    mean_loss = loss_sum.item() / window
                    writer.add_scalar("train/loss", mean_loss, step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    writer.add_scalar("train/steps_per_second", window / elapsed, step)
                    for i, (total, count) in enumerate(zip(bin_sum.tolist(), bin_n.tolist())):
                        if count:
                            writer.add_scalar(f"train/loss_t{i / 4:.2f}-{(i + 1) / 4:.2f}", total / count, step)
                    metrics = f"loss={mean_loss:.4f}"
                    if disc is not None and gan_n.item():
                        d_loss, adv_loss, fm_loss = (gan_sum / gan_n).tolist()
                        writer.add_scalar("train/disc_loss", d_loss, step)
                        if step > adv_start:
                            writer.add_scalar("train/adv_loss", adv_loss, step)
                            writer.add_scalar("train/fm_loss", fm_loss, step)
                        metrics += f", disc={d_loss:.3f}"
                    if not live:
                        info(f"step {step}/{total_steps}: {metrics}, {window / elapsed:.1f} steps/s", tag="[STYLE]")
                    for acc in (loss_sum, bin_sum, bin_n, gan_sum, gan_n):
                        acc.zero_()
                    window, tic = 0, time.time()
                bar.update(task_id, advance=1, metrics=metrics)
                if step % val_every == 0 or step == total_steps:
                    ema.eval()
                    validate()
                    save()
                    model.train()
    except KeyboardInterrupt:
        warning("Interrupted; saving a checkpoint.", tag="[STYLE]")
        save()
        raise
    writer.close()
    success(f"Done at step {step}: {os.path.join(out_dir, export_name)}", tag="[STYLE]")
    return os.path.join(out_dir, export_name)

