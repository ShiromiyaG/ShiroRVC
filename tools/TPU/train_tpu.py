"""Vocoder training on TPU (torch_xla), one process per TPU chip.

Covers every vocoder in ``rvc/configs/vocoders.json`` -- RefineGAN v2, HiFi-GAN
and HiFi-GAN++ -- including those the registry disables for the UI.  Runs the
same step as ``rvc/train/train.py`` (the config's spectral loss, KL, feature
matching and adversarial terms against the vocoder's discriminator) with the
pieces XLA cannot compile replaced by ``xla_patches``.  Every batch has one
fixed shape, so the step compiles once.

Sized for a pretrain: a large global batch, the config's learning rates scaled
from ``--reference-batch`` to it, and a linear warmup in front of the decay.

Not ported: holdout/overtrain detection, previews, freeze stages and the
non-AdamW optimizers.

Checkpoints use the CUDA trainer's format (``G_<step>.pth`` / ``D_<step>.pth``
in ``logs/<model>``), so a run can move between TPU and CUDA; see
``convert_to_cuda.py``.

    python tools/TPU/train_tpu.py --model-name my-pretrain --epochs 100 --batch-size 16
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "rvc", "train"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

CHECKPOINT_RE = re.compile(r"^G_(\d+)\.pth$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-name", required=True, help="Folder under logs/ with config.json and filelist.txt.")
    parser.add_argument("--vocoder", default="", help="hifi, hifi++ or refinegan2; defaults to model_info.json's vocoder_architecture.")
    parser.add_argument("--pretrain-g", default="", help="Pretrained G; ignored when the folder already has G_*.pth.")
    parser.add_argument("--pretrain-d", default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16, help="Per TPU chip; the global batch is this times the chip count.")
    parser.add_argument("--reference-batch", type=int, default=8, help="Global batch the config's learning rates were tuned for.")
    parser.add_argument(
        "--lr-scaling",
        choices=("sqrt", "linear", "none"),
        default="sqrt",
        help="How the learning rates follow the global batch. sqrt is the safe rule for Adam on a GAN.",
    )
    parser.add_argument("--warmup-steps", type=int, default=-1, help="Linear LR warmup; -1 picks min(1000, 5%% of the run).")
    parser.add_argument("--save-every", type=int, default=10, help="Epochs between checkpoints.")
    parser.add_argument("--save-only-latest", action="store_true", help="Delete older G_/D_ checkpoints on each save.")
    parser.add_argument("--no-weight-export", action="store_true", help="Do not export the inference .pth on each save.")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-frames", type=int, default=0, help="Fixed frame window per item; 0 sizes it from the dataset.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Extra multiplier on top of --lr-scaling.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per chip; 0 splits the host's CPUs across chips.")
    parser.add_argument("--log-every", type=int, default=50, help="Steps between log lines.")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="Print the XLA metrics report after the first epoch (to spot recompiles).")
    parser.add_argument("--single-process", action="store_true", help="One chip only, for debugging.")
    return parser.parse_args(argv)


def experiment_paths(model_name):
    directory = os.path.join(REPO_ROOT, "logs", model_name)
    return {
        "dir": directory,
        "config": os.path.join(directory, "config.json"),
        "filelist": os.path.join(directory, "filelist.txt"),
        "model_info": os.path.join(directory, "model_info.json"),
        "frames_cache": os.path.join(directory, "clip_frames.json"),
    }


def resolve_vocoder(model_info_path, override="", sample_rate=None):
    """The registry id to build: ``override``, else what extraction recorded."""
    import json

    from rvc.configs.vocoders import get_vocoder_sample_rates, normalize_vocoder

    recorded = ""
    if os.path.isfile(model_info_path):
        with open(model_info_path, encoding="utf-8") as handle:
            recorded = json.load(handle).get("vocoder_architecture", "")
    if not (override or recorded):
        raise SystemExit("No vocoder recorded in model_info.json; pass --vocoder.")
    vocoder = normalize_vocoder(override or recorded)
    # config.json was copied from the recorded vocoder's folder.
    if recorded and normalize_vocoder(recorded) != vocoder:
        raise SystemExit(
            f"The data was prepared for {recorded!r}, not {vocoder!r}; extract it again for {vocoder!r}."
        )
    if sample_rate is not None and int(sample_rate) not in get_vocoder_sample_rates(vocoder):
        raise SystemExit(f"{vocoder} has no configuration for {sample_rate} Hz.")
    return vocoder


def latest_checkpoints(directory):
    steps = []
    for name in os.listdir(directory):
        match = CHECKPOINT_RE.match(name)
        if match and os.path.isfile(os.path.join(directory, f"D_{match.group(1)}.pth")):
            steps.append(int(match.group(1)))
    if not steps:
        return None
    step = max(steps)
    return step, os.path.join(directory, f"G_{step}.pth"), os.path.join(directory, f"D_{step}.pth")


def wav_samples(path):
    """Sample count from the RIFF header, falling back to soundfile for anything else."""
    with open(path, "rb") as f:
        head = f.read(4096)
        file_size = os.fstat(f.fileno()).st_size
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        pos, align = 12, 0
        while pos + 8 <= len(head):
            chunk, size = head[pos : pos + 4], int.from_bytes(head[pos + 4 : pos + 8], "little")
            if chunk == b"fmt ":
                align = int.from_bytes(head[pos + 20 : pos + 22], "little")
            elif chunk == b"data" and align:
                if size in (0, 0xFFFFFFFF):
                    break
                # A truncated file claims more data than it holds; soundfile clamps the same way.
                return min(size, file_size - pos - 8) // align
            pos += 8 + size + (size & 1)
    return sf.info(path).frames


def _frames_chunk(chunk, hop):
    return [wav_samples(path) // hop for path in chunk]


def build_frames_cache(paths):
    """Writes every filelist clip's frame count to ``frames_cache`` once, before the chips spawn."""
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    stat = os.stat(paths["filelist"])
    with open(paths["config"]) as f:
        hop = int(json.load(f)["data"]["hop_length"])
    stamp = [stat.st_size, stat.st_mtime_ns, hop]
    if os.path.isfile(paths["frames_cache"]):
        with open(paths["frames_cache"]) as f:
            cached = json.load(f)
        if cached.get("stamp") == stamp:
            return
    from rvc.train.utils import load_filepaths_and_text

    # Same path resolution as the dataset, so the keys match its audiopaths.
    audio = sorted({row[0] for row in load_filepaths_and_text(paths["filelist"])})
    print(f"[TPU] Reading the length of {len(audio)} clips (cached for later runs)...", flush=True)
    frames = {}
    started = time.time()
    chunks = [audio[i : i + 2000] for i in range(0, len(audio), 2000)]
    # Processes, not threads: the per-file Python work holds the GIL. Fork is safe, XLA is not up yet.
    workers = min(64, os.cpu_count() or 8)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for chunk, counts in zip(chunks, pool.map(_frames_chunk, chunks, [hop] * len(chunks))):
            frames.update(zip(chunk, counts))
            done = len(frames)
            if done % 20000 < len(chunk) or done == len(audio):
                rate = done / max(time.time() - started, 1e-6)
                print(f"[TPU]   {done}/{len(audio)} ({rate:.0f} clips/s)", flush=True)
    tmp = paths["frames_cache"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"stamp": stamp, "frames": frames}, f)
    os.replace(tmp, paths["frames_cache"])


def clip_frames(dataset, indices, cache_path):
    import json

    with open(cache_path) as f:
        frames = json.load(f)["frames"]
    return {index: frames[dataset.audiopaths_and_text[index][0]] for index in indices}


def auto_max_frames(frames):
    # A multiple of 16 so a slightly longer dataset reuses the same shape.
    return min(900, int(math.ceil(max(frames) / 16.0) * 16))


class FixedCollate:
    """Pads or randomly crops every item to ``frames``, so each batch has one shape."""

    def __init__(self, frames, hop_length):
        self.frames = int(frames)
        self.hop = int(hop_length)

    def __call__(self, batch):
        count, frames, hop = len(batch), self.frames, self.hop
        spec_channels = batch[0][0].shape[0]
        phone_dim = batch[0][2].shape[1]
        spec = torch.zeros(count, spec_channels, frames)
        wave = torch.zeros(count, 1, frames * hop)
        phone = torch.zeros(count, frames, phone_dim)
        pitch = torch.zeros(count, frames, dtype=torch.long)
        pitchf = torch.zeros(count, frames)
        lengths = torch.zeros(count, dtype=torch.long)
        sid = torch.zeros(count, dtype=torch.long)
        for i, (s, w, ph, p, pf, speaker) in enumerate(batch):
            n = min(s.shape[1], ph.shape[0])
            start = random.randint(0, n - frames) if n > frames else 0
            n = min(n, frames)
            spec[i, :, :n] = s[:, start : start + n]
            phone[i, :n] = ph[start : start + n]
            pitch[i, :n] = p[start : start + n]
            pitchf[i, :n] = pf[start : start + n]
            audio = w[:, start * hop : (start + n) * hop]
            wave[i, :, : audio.shape[-1]] = audio
            lengths[i] = n
            sid[i] = speaker
        return phone, lengths, pitch, pitchf, spec, lengths.clone(), wave, sid


class ShardedShuffleSampler(torch.utils.data.Sampler):
    """Per-epoch shuffle, split across chips, trimmed to whole global batches."""

    def __init__(self, indices, batch_size, world, rank, seed):
        self.indices = list(indices)
        self.world, self.rank, self.seed = world, rank, seed
        self.per_rank = (len(self.indices) // (batch_size * world)) * batch_size
        if self.per_rank == 0:
            raise ValueError(
                f"{len(self.indices)} usable clips cannot fill one global batch of "
                f"{batch_size} x {world}; lower --batch-size."
            )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        shard = order[self.rank :: self.world][: self.per_rank]
        return iter(self.indices[i] for i in shard)

    def __len__(self):
        return self.per_rank


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def xla_adamw_flags(optimizer):
    # Without ``capturable`` AdamW reads ``step`` with ``.item()``, a host sync
    # per parameter per step.  A CUDA checkpoint brings ``fused=True`` back in.
    for group in optimizer.param_groups:
        group.update(capturable=True, foreach=False, fused=None)


def make_adamw(params, lr):
    from rvc.train.optimizers import BASE_BETAS, BASE_WEIGHT_DECAY

    optimizer = torch.optim.AdamW(
        params, lr=lr, betas=BASE_BETAS, eps=1e-9, weight_decay=BASE_WEIGHT_DECAY,
        capturable=True, foreach=False,
    )
    for group in optimizer.param_groups:
        group["lazy_reg_scale"] = 1.0
    return optimizer


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                value = value.to(device)
                state[key] = value.float() if key == "step" else value


def grad_norm(parameters):
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    if not norms:
        return torch.zeros(())
    return torch.stack(norms).norm()


def checkpoint_payload(model, state, optimizer, learning_rate, epoch):
    from rvc.train.utils import (
        decoder_layout,
        discriminator_has_msd,
        discriminator_periods,
        excitation_source,
        optimizer_param_names,
    )

    data = {
        "model": state,
        "iteration": epoch,
        "optimizer": optimizer.state_dict(),
        "optimizer_param_names": optimizer_param_names(optimizer, model),
        "learning_rate": learning_rate,
    }
    extras = {
        "architecture_id": getattr(model, "architecture_id", None),
        "excitation_source": excitation_source(model),
        "discriminator_periods": discriminator_periods(model),
        "discriminator_msd": discriminator_has_msd(model),
        "decoder_layout": decoder_layout(model),
    }
    data.update({key: value for key, value in extras.items() if value is not None})
    return data


def _mp_fn(index, args):
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr

    import xla_patches

    xla_patches.apply()

    from rvc.lib.algorithm import commons
    from rvc.lib.terminal import info, success, warning
    from rvc.train.ema import WeightEMA
    from rvc.train.losses import (
        BandWeightedSpectralLoss,
        discriminator_loss,
        feature_loss,
        generator_loss,
        kl_loss,
        mel_frequency_tilt_weights,
        mel_low_frequency_weights,
    )
    from rvc.train.mel_processing import build_ms_mel_loss
    from rvc.train.process.extract_model import extract_model
    from rvc.train.setup import (
        apply_precision_policy,
        assert_resumable_architecture,
        get_d_model,
        get_g_model,
        normalize_san_weights,
    )
    from rvc.train.utils import (
        assert_decoder_layout_matches,
        assert_excitation_matches,
        assert_msd_matches,
        assert_periods_match,
        latest_checkpoint_path,
        load_config_from_json,
        small_model_naming,
        substitute_speaker_embeddings,
        verify_spk_dim,
        wave_to_mel,
    )
    from data_utils import TextAudioLoaderMultiNSFsid

    device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    sync = getattr(torch_xla, "sync", xm.mark_step)
    rank, world = xr.global_ordinal(), xr.world_size()
    master = rank == 0

    paths = experiment_paths(args.model_name)
    config = load_config_from_json(paths["config"])
    config.data.training_files = paths["filelist"]
    sample_rate = int(config.data.sample_rate)
    hop = int(config.data.hop_length)
    segment_size = int(config.train.segment_size)

    vocoder = resolve_vocoder(paths["model_info"], args.vocoder, sample_rate)

    spectral_loss_name = str(getattr(config.train, "spectral_loss", "L1 Mel Loss"))
    if spectral_loss_name not in ("L1 Mel Loss", "Multi-Scale Mel Loss"):
        raise ValueError(f"The TPU trainer runs 'L1 Mel Loss' or 'Multi-Scale Mel Loss', not {spectral_loss_name!r}.")
    multi_scale = spectral_loss_name == "Multi-Scale Mel Loss"
    if float(getattr(config.train, "hf_floor_negative_weight", 0.0)) > 0.0:
        raise ValueError("hf_floor_negative_weight > 0 needs an ISTFT and is not ported; set it to 0.")
    if str(getattr(config.train, "optimizer_g", "AdamW")) != "AdamW" or str(
        getattr(config.train, "optimizer_d", "AdamW")
    ) != "AdamW":
        if master:
            warning("The TPU trainer only runs AdamW; the config's optimizer is ignored.", tag="[TPU]")

    torch.manual_seed(int(config.train.seed))
    random.seed(int(config.train.seed) + rank)

    # Data
    dataset = TextAudioLoaderMultiNSFsid(config.data, n_mel_bins=config.model.inter_channels)
    # The CUDA sampler's bucket bounds, so both trainers see the same clips.
    bucketed = [i for i, length in enumerate(dataset.lengths) if 50 < length <= 900]
    lengths = clip_frames(dataset, bucketed, paths["frames_cache"])
    # The random slice needs a whole segment inside the clip; the margin
    # covers features a frame or two shorter than the audio.
    usable = [i for i in bucketed if lengths[i] >= segment_size // hop + 2]
    if not usable:
        raise ValueError("No clip is long enough for one training segment.")
    frames = args.max_frames or auto_max_frames([lengths[i] for i in usable])
    if frames * hop < segment_size:
        raise ValueError(f"--max-frames must be at least {segment_size // hop}.")
    sampler = ShardedShuffleSampler(usable, args.batch_size, world, rank, int(config.train.seed))
    # One process per chip shares the host, so the CPUs are split between them.
    workers = args.num_workers or max(2, min(12, (os.cpu_count() or 8) // world - 1))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=FixedCollate(frames, hop),
        num_workers=workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    steps_per_epoch = len(sampler) // args.batch_size
    device_loader = pl.MpDeviceLoader(loader, device)

    # Models
    speakers = verify_spk_dim(
        config, paths["model_info"], paths["dir"], latest_checkpoint_path, rank, args.pretrain_g
    )
    config.model.spk_embed_dim = speakers.embed_dim
    net_g = get_g_model(config, sample_rate, vocoder, False)
    net_d = get_d_model(config, vocoder, False)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None
    apply_precision_policy(net_g, amp_dtype)

    resume = latest_checkpoints(paths["dir"])
    epoch_start, global_step = 1, 0
    g_blob = d_blob = None
    if resume is not None:
        global_step, g_path, d_path = resume
        assert_resumable_architecture(net_g, g_path)
        g_blob = torch.load(g_path, map_location="cpu", weights_only=True)
        d_blob = torch.load(d_path, map_location="cpu", weights_only=True)
        assert_periods_match(net_d, d_blob)
        assert_msd_matches(net_d, d_blob)
        net_g.load_state_dict(g_blob["model"], strict=True)
        net_d.load_state_dict(d_blob["model"], strict=True)
        epoch_start = int(g_blob["iteration"]) + 1
        if master:
            success(f"Resuming from step {global_step}, epoch {epoch_start - 1}.", tag="[TPU]")
    else:
        if args.pretrain_g:
            blob = torch.load(args.pretrain_g, map_location="cpu", weights_only=True)
            assert_excitation_matches(net_g, blob, origin="pretrain")
            assert_decoder_layout_matches(net_g, blob, origin="pretrain")
            state = blob.get("model", blob)
            if speakers.reset_pretrained:
                state = substitute_speaker_embeddings(state, net_g)
            net_g.load_state_dict(state, strict=True)
        if args.pretrain_d:
            blob = torch.load(args.pretrain_d, map_location="cpu", weights_only=True)
            assert_periods_match(net_d, blob, origin="pretrained discriminator")
            assert_msd_matches(net_d, blob, origin="pretrained discriminator")
            net_d.load_state_dict(blob.get("model", blob), strict=True)
        if master and not (args.pretrain_g and args.pretrain_d):
            info("No pretrain pair given: the missing side starts from scratch.", tag="[TPU]")

    net_g.to(device)
    net_d.to(device)
    if world > 1 and resume is None:
        xm.broadcast_master_param(net_g)
        xm.broadcast_master_param(net_d)

    global_batch = args.batch_size * world
    batch_ratio = global_batch / max(1, args.reference_batch)
    lr_multiplier = args.lr_scale * {
        "sqrt": math.sqrt(batch_ratio),
        "linear": batch_ratio,
        "none": 1.0,
    }[args.lr_scaling]
    lr_g = float(config.train.learning_rate_g) * lr_multiplier
    lr_d = float(config.train.learning_rate_d) * lr_multiplier
    optim_g = make_adamw([p for p in net_g.parameters() if p.requires_grad], lr_g)
    optim_d = make_adamw([p for p in net_d.parameters() if p.requires_grad], lr_d)
    if resume is not None:
        for optimizer, blob, base in ((optim_g, g_blob, lr_g), (optim_d, d_blob, lr_d)):
            if blob.get("optimizer"):
                optimizer.load_state_dict(blob["optimizer"])
                xla_adamw_flags(optimizer)
                optimizer_to(optimizer, device)
            for group in optimizer.param_groups:
                group["initial_lr"] = base
    for optimizer in (optim_g, optim_d):
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    del g_blob, d_blob

    ema = None
    if not args.no_ema and bool(getattr(config.train, "ema", True)):
        ema = WeightEMA(net_g, decay=float(getattr(config.train, "ema_decay", WeightEMA.DEFAULT_DECAY)))

    total_steps = max(1, args.epochs * steps_per_epoch)
    lr_final_ratio = getattr(config.train, "lr_final_ratio", None)
    gamma = float(getattr(config.train, "lr_decay", 0.999875)) ** (1.0 / steps_per_epoch)
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else min(1000, total_steps // 20)

    def lr_factor(step):
        if lr_final_ratio is not None:
            ratio = min(1.0, max(1e-6, float(lr_final_ratio)))
            factor = ratio ** min(1.0, max(0, step) / total_steps)
        else:
            factor = gamma ** max(0, step)
        if step < warmup_steps:
            factor *= (step + 1) / warmup_steps
        return factor

    # Losses
    tilt = float(getattr(config.train, "mel_frequency_tilt", 0.0))
    tilt_max_ratio = float(getattr(config.train, "mel_frequency_tilt_max_ratio", 8.0))
    low_emphasis = float(getattr(config.train, "mel_low_emphasis", 1.0))
    low_emphasis_hz = float(getattr(config.train, "mel_low_emphasis_hz", 1000.0))

    def band_weights(num_mels):
        weights = mel_frequency_tilt_weights(
            num_mels=num_mels,
            sample_rate=sample_rate,
            mel_fmin=config.data.mel_fmin,
            mel_fmax=config.data.mel_fmax,
            tilt=tilt,
            max_ratio=tilt_max_ratio,
        )
        if low_emphasis != 1.0:
            weights = weights * mel_low_frequency_weights(
                num_mels=num_mels,
                sample_rate=sample_rate,
                mel_fmin=config.data.mel_fmin,
                mel_fmax=config.data.mel_fmax,
                emphasis=low_emphasis,
                cutoff_hz=low_emphasis_hz,
            )
            weights = weights / weights.mean()
        return weights

    distance = torch.nn.L1Loss()
    if tilt != 0.0 or low_emphasis != 1.0:
        distance = BandWeightedSpectralLoss(
            torch.nn.L1Loss(reduction="none"),
            band_weights(config.data.n_mel_channels),
            weight_factory=band_weights,
        ).to(device)
    fn_spectral = build_ms_mel_loss(sample_rate, loss_fn=distance) if multi_scale else distance

    c_mel = float(config.train.c_mel)
    c_kl = float(config.train.c_kl)
    san_direction_weight = max(0.0, min(1.0, float(getattr(config.train, "san_direction_weight", 0.25))))
    san_active = bool(getattr(net_d, "supports_san", False))
    branch_weights = tuple(net_d.branch_weights) if getattr(net_d, "uses_branch_weights", False) else None

    def autocast():
        return torch.autocast("xla", dtype=torch.bfloat16, enabled=amp_dtype is not None)

    def train_step(batch):
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, sid = batch
        with autocast():
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                spec, spec_lengths, sid, phone, phone_lengths, pitchf, pitch
            )
            y = commons.slice_segments(y, ids_slice * hop, segment_size, dim=3)

            y_d_r, y_d_g, _, _ = net_d(y, y_hat.detach(), san_training=san_active, combine_inputs=True)
            loss_disc, _, _ = discriminator_loss(
                y_d_r,
                y_d_g,
                san_direction_weight=san_direction_weight,
                normalize=False,
                branch_weights=branch_weights,
            )
        optim_d.zero_grad(set_to_none=True)
        loss_disc.backward()
        xm.reduce_gradients(optim_d)
        norm_d = grad_norm(net_d.parameters())
        optim_d.step()
        if san_active:
            normalize_san_weights(net_d, optim_d)
        optim_d.zero_grad(set_to_none=True)

        # D is only read here; freezing it skips its weight gradients.
        net_d.requires_grad_(False)
        with autocast():
            _, y_d_g, fmap_r, fmap_g = net_d(y, y_hat, no_grad_real=True)
            if multi_scale:
                loss_mel = fn_spectral(y, y_hat) * c_mel
            else:
                loss_mel = fn_spectral(wave_to_mel(config, y), wave_to_mel(config, y_hat)) * c_mel
            loss_fm = feature_loss(fmap_r, fmap_g, normalize=False, branch_weights=branch_weights) * 2.0
            loss_adv = generator_loss(
                y_d_g,
                normalize=False,
                san_direction_weight=san_direction_weight,
                use_softplus=san_active,
                branch_weights=branch_weights,
            )
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl
            loss_gen = loss_mel + loss_kl + loss_adv + loss_fm
        optim_g.zero_grad(set_to_none=True)
        loss_gen.backward()
        xm.reduce_gradients(optim_g)
        norm_g = grad_norm(net_g.parameters())
        optim_g.step()
        net_d.requires_grad_(True)
        if ema is not None:
            ema.update(net_g)
        return {
            "loss_disc": loss_disc,
            "loss_gen": loss_gen,
            "loss_mel": loss_mel,
            "loss_kl": loss_kl,
            "loss_fm": loss_fm,
            "loss_adv": loss_adv,
            "grad_norm_d": norm_d,
            "grad_norm_g": norm_g,
        }

    # Updated every step with the same ops, so logging adds no second graph.
    smoothing = 1.0 / max(1, int(getattr(config.train, "rolling_loss_steps", 50)))
    running = None
    writer = None
    if master:
        import types

        # tensorboard imports a full tensorflow when present, whose own TPU runtime aborts
        # this process (runtime_metric_aggregator); the notf marker makes it use its stub.
        sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(paths["dir"], "eval"))
        info(
            f"{vocoder} at {sample_rate} Hz ({spectral_loss_name}), "
            f"{world} chip(s), batch {args.batch_size} per chip ({global_batch} global), "
            f"{frames} frames per item, {steps_per_epoch} steps/epoch, {workers} loader workers "
            f"per chip, precision {args.precision}.",
            tag="[TPU]",
        )
        info(
            f"LR G {lr_g:.2e} / D {lr_d:.2e} ({args.lr_scaling} scaling x{lr_multiplier:.2f} from "
            f"batch {args.reference_batch}), warmup {warmup_steps} of {total_steps} steps.",
            tag="[TPU]",
        )

    def log_closure(step, epoch, values, started):
        if not master:
            return
        scalars = {key: float(value) for key, value in values.items()}
        for key, value in scalars.items():
            writer.add_scalar(f"tpu/{key}", value, step)
        writer.add_scalar("tpu/learning_rate_g", optim_g.param_groups[0]["lr"], step)
        rate = args.log_every / max(1e-6, time.time() - started[0])
        started[0] = time.time()
        summary = " ".join(f"{key}={value:.3f}" for key, value in scalars.items())
        info(f"epoch {epoch} step {step} ({rate:.2f} it/s) {summary}", tag="[TPU]")

    def save(epoch):
        sync()
        g_state = ema.shadow if ema is not None else net_g.state_dict()
        g_payload = checkpoint_payload(net_g, g_state, optim_g, lr_g, epoch)
        d_payload = checkpoint_payload(net_d, net_d.state_dict(), optim_d, lr_d, epoch)
        if master:
            if args.save_only_latest:
                for old in glob.glob(os.path.join(paths["dir"], "G_*.pth")) + glob.glob(
                    os.path.join(paths["dir"], "D_*.pth")
                ):
                    os.remove(old)
            g_payload = to_cpu(g_payload)
            torch.save(g_payload, os.path.join(paths["dir"], f"G_{global_step}.pth"))
            torch.save(to_cpu(d_payload), os.path.join(paths["dir"], f"D_{global_step}.pth"))
            success(f"Saved G_{global_step}.pth / D_{global_step}.pth.", tag="[TPU]")
            if not args.no_weight_export:
                extract_model(
                    ckpt=g_payload["model"],
                    sr=sample_rate,
                    name=args.model_name,
                    model_path=os.path.join(
                        paths["dir"], small_model_naming(args.model_name, epoch, global_step)
                    ),
                    epoch=epoch,
                    step=global_step,
                    hps=config,
                    vocoder=vocoder,
                    architecture="RVC",
                    weights_source="EMA weights" if ema is not None else "live weights",
                )
        xm.rendezvous("tpu_checkpoint_saved")

    started = [time.time()]
    epoch = epoch_start - 1
    for epoch in range(epoch_start, args.epochs + 1):
        sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        for batch in device_loader:
            factor = lr_factor(global_step)
            for optimizer in (optim_g, optim_d):
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
            global_step += 1
            values = train_step(batch)
            with torch.no_grad():
                if running is None:
                    running = {key: value.detach().float().clone() for key, value in values.items()}
                else:
                    for key, value in values.items():
                        running[key].lerp_(value.detach().float(), smoothing)
            if global_step % args.log_every == 0:
                xm.add_step_closure(log_closure, args=(global_step, epoch, dict(running), started))
        if master and args.metrics and epoch == epoch_start:
            import torch_xla.debug.metrics as met

            print(met.short_metrics_report(), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save(epoch)

    if master:
        writer.flush()
        writer.close()
        success(f"Training finished at epoch {epoch}, step {global_step}.", tag="[TPU]")


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.chdir(REPO_ROOT)
    paths = experiment_paths(args.model_name)
    for key in ("config", "filelist"):
        if not os.path.isfile(paths[key]):
            sys.exit(f"{paths[key]} is missing: run preprocess + extract first.")
    build_frames_cache(paths)

    if not args.single_process:
        # Kaggle presets a one-process topology (TPU_PROCESS_ADDRESSES=local); torch_xla only
        # setdefault()s these, so each spawned chip would keep it and fail to find its peers.
        for key in ("TPU_PROCESS_ADDRESSES", "TPU_PROCESS_BOUNDS", "TPU_VISIBLE_CHIPS", "TPU_PROCESS_PORT"):
            os.environ.pop(key, None)

    import torch_xla

    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_mp_fn, args=(args,), debug_single_process=args.single_process)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(_mp_fn, args=(args,), nprocs=1 if args.single_process else None)


if __name__ == "__main__":
    main()
