import os
import copy
import datetime
import glob
import math
import re
import socket
import sys

from itertools import islice
from collections import deque
from distutils.util import strtobool
from random import randint, Random
import signal
import threading
from contextlib import contextmanager
from time import time as ttime

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cuda, cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from rvc.lib.terminal import (
    configure_logging,
    error as print_error,
    info,
    install_rich_print,
    print_model_summary,
    progress_task,
    success,
    warning,
)
from rvc.train.ema import WeightEMA
from rvc.train.optimizers import (
    _make_optimizer,
    averaged_weights,
    is_schedule_free,
)
from rvc.train.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_FALLBACK_NAMESPACE,
    TENSORBOARD_MEDIA_SOURCE_NAME,
    DISCRIMINATOR_COMPILE_ENABLED,
    DISCRIMINATOR_COMPILE_NO_CUDA,
    DISCRIMINATOR_COMPILE_NOT_SUPPORTED,
    VOCODER_COMPILE_ENABLED,
    VOCODER_COMPILE_NO_CUDA,
    VOCODER_COMPILE_NOT_SUPPORTED,
)

install_rich_print()

from utils import (
    summarize,
    assert_excitation_matches,
    assert_periods_match,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    load_config_from_json,
    flush_writer,
    block_tensorboard_flush_on_exit,
    log_tensorboard_media,
    log_validation_preview,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    print_init_setup,
    train_loader_safety,
    substitute_speaker_embeddings,
    verify_spk_dim
)

from losses import (
    discriminator_loss,
    generator_loss,
    feature_loss,
    kl_loss,
    mel_low_frequency_weights,
    BandWeightedSpectralLoss,
    MultiScaleSTFTLoss,
)

from mel_processing import MultiScaleMelSpectrogramLoss

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.configs.vocoders import (
    get_discriminator_id,
    normalize_vocoder,
)
from rvc.train.run_spec import TrainRunSpec

# argv[1] is the run spec written by the launcher (not the same indexing as
# ``core._find_trainer_processes``, which reads the OS command line and so
# sees the interpreter at cmdline[0]).  DDP's ``spawn`` re-executes this
# module in every child, so each rank re-reads the same file.
spec = TrainRunSpec.load(sys.argv[1])

model_name = spec.model_name
epoch_save_frequency = spec.epoch_save_frequency
total_epoch_count = spec.total_epoch_count
pretrainG = spec.pretrain_g
pretrainD = spec.pretrain_d
gpus = spec.gpus
batch_size = spec.batch_size
sample_rate = spec.sample_rate
save_only_latest_net_models = spec.save_only_latest_net_models
save_weight_models = spec.save_weight_models
use_warmup = spec.use_warmup
warmup_duration = spec.warmup_duration
cleanup = spec.cleanup
vocoder = normalize_vocoder(spec.vocoder)
architecture = "RVC"
# G and D always share one optimizer choice; kept as two names because the
# rest of the file distinguishes them.
optimizer_choice_g = optimizer_choice_d = spec.optimizer_choice
use_checkpointing = spec.use_checkpointing
use_tf32 = spec.use_tf32
use_fp16 = spec.use_fp16
use_benchmark = spec.use_benchmark
lr_scheduler = spec.lr_scheduler

use_custom_lr = spec.use_custom_lr
custom_lr_g, custom_lr_d = (spec.custom_lr_g, spec.custom_lr_d) if use_custom_lr else (None, None)
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

compile_vocoder = spec.compile_vocoder
torch_compile_mode = spec.torch_compile_mode
overtrain_detector = spec.overtrain_detector
stop_on_overtrain = spec.stop_on_overtrain
use_ema = spec.use_ema
# No manual phase/step controls: a pretrained source selects fine-tuning,
# its absence selects pretraining.
training_phase = spec.training_phase
max_steps = 0

cuda.matmul.allow_tf32 = use_tf32
cudnn.allow_tf32 = use_tf32
cudnn.benchmark = use_benchmark

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

exp_decay_gamma = float(getattr(config.train, "lr_decay", 0.999875))
# Belongs to the vocoder, not the run: a 128-band mel can't resolve a harmonic
# comb above ~2 kHz, so RefineGAN needs the MS-STFT term ("Hybrid L1") while
# HiFi-GAN uses the plain mel it was designed around.
spectral_loss = str(getattr(config.train, "spectral_loss", "L1 Mel Loss"))
# "End at this fraction of the starting LR"; the per-epoch gamma is derived
# from the run's real length so changing the epoch count restretches the
# schedule instead of silently changing the decay.  ``None`` keeps the old
# ``lr_decay`` behaviour.
lr_final_ratio = getattr(config.train, "lr_final_ratio", None)
lr_final_ratio = None if lr_final_ratio is None else float(lr_final_ratio)
# Sticky: baked into the TensorBoard tag names, so changing it starts a fresh
# set of series.
rolling_loss_steps = int(getattr(config.train, "rolling_loss_steps", 50))

# dpi scales the whole figure; width/height are inches.  Every preview is
# embedded in the event file at full size, so raising these grows the log too.
validation_preview_dpi = max(
    10.0, float(getattr(config.train, "validation_preview_dpi", 67))
)
validation_preview_figsize = (
    max(1.0, float(getattr(config.train, "validation_preview_width", 24.0))),
    max(1.0, float(getattr(config.train, "validation_preview_height", 5.8))),
)

# Default: FP32 + TF32, no autocast/scaler. ``use_fp16`` enables autocast at
# FP16 with GradScaler; the autocast-disable wrappers are narrowed to protect
# only distribution math and the NSF source, so the compiled decoder graph
# stays in one dtype end-to-end.
use_amp = bool(use_fp16)
amp_dtype = torch.float16 if use_amp else None

# Globals ( Do not alter these )
global_step = 0
warmup_completed = False
from_scratch = False
finetune_phase = training_phase == "finetune"
phase_start_step = 0
phase_step = 0
phase_limit_reached = False
overtrain_flagged = False
overtrain_exported = False
reset_optimizer_for_run = finetune_phase
use_lr_scheduler = lr_scheduler != "none"
# Steps the GradScaler discarded for non-finite gradients. A steady run of
# skips looks like a stalled loss otherwise. Cumulative across resumes.
amp_skipped_steps = 0



# ========  Advanced / Manual and exp tweaks  ========================
enable_persistent_workers = True

pretrain_preview = True
pretrain_preview_interval = 500  # Measured in steps.
finetune_preview_interval = 100  # Measured in steps.

force_from_scratch = False
strict_load = True

clip_grad_norm_override = False
clip_grad_norm_override_value_g = 100
clip_grad_norm_override_value_d = 100

# linear-warmup in steps. 0 = disabled
warmup_steps = 0

# L1 mel -> multi-scale mel transition, over swap_duration_steps (kicks in at resume).
swap_l1_to_ms = False
swap_duration_steps = 500
swap_start_step = 0  # Filled at resume: global_step when the swap was enabled.
swap_completed = False

# Freezes the whole frontend: everything outside `dec.` and `emb_g.`
# ( enc_p + enc_q + flow )
freeze_vae = False # If true, lets only vocoder ( dec ), spk embedding and discriminator learn

# ----  Global LR scales  ----
# Multipliers of the base LR ( 0.1 = 10%, 1.0 = 100% ).
dec_lr_scale = None  # everything under `dec.` ( the decoder/vocoder )
vae_lr_scale = None  # everything else ( frontend + emb_g ), see `freeze_vae` above

# ----  Resume LR override  ----
resume_lr = None  # e.g. 5e-5 ( None = Override disabled. )
resume_lr_target = "full"  # Pick what you want it applied to: "g", "d" or "full" where full refers to both G/D

# True = Gamma is applied as-is each step.
# False = Per-epoch budget, re-binned to per-step so the total decay per epoch equals "exp decay epoch" ( VITS-style ) ~ Default
exp_decay_step_raw = False

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


# ----  Interruptible-only-at-safe-points shutdown  ----
# The launcher asks for a stop with SIGTERM / CTRL_BREAK and only kills after a
# grace period.  These handlers never terminate the process themselves: they
# record the request and return, so a `torch.save` already in flight always runs
# to completion and no checkpoint is ever left truncated.  The training loop
# then acts on the flag at its next safe point.
_stop_requested = threading.Event()
_saving_depth = 0
_saving_lock = threading.Lock()


def stop_was_requested():
    return _stop_requested.is_set()


@contextmanager
def uninterruptible_save(description):
    """Mark a region that must reach the disk before the process may exit."""
    global _saving_depth
    with _saving_lock:
        _saving_depth += 1
    try:
        yield
    finally:
        with _saving_lock:
            _saving_depth -= 1
        if _stop_requested.is_set():
            info(f"{description} finished; the stop can proceed now.", tag="[TRAIN]")


def _handle_stop_signal(signum, frame):
    """Record a stop request. Deliberately does not exit."""
    if _stop_requested.is_set():
        return
    _stop_requested.set()
    with _saving_lock:
        mid_write = _saving_depth > 0
    if mid_write:
        warning(
            "Stop requested while writing a checkpoint - "
            "finishing the write first, then exiting.",
            tag="[TRAIN]",
        )
    else:
        warning("Stop requested - exiting at the next safe point.", tag="[TRAIN]")


def install_stop_handlers():
    """Route the launcher's stop signals into the flag above."""
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_stop_signal)
        except (ValueError, OSError):
            # Not the main thread, or unsupported on this platform.
            pass


def finish_stop(writer=None):
    """Leave once nothing is being written."""
    info("Stopping cleanly.", tag="[TRAIN]")
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
    os._exit(0)


def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        if hasattr(net_g, "module"):
            o, *_ = net_g.module.infer(*reference)
        else:
            o, *_ = net_g.infer(*reference)
    net_g.train()
    return o

class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"

def setup_env_and_distr(rank, n_gpus, device, device_id, config):
    if n_gpus > 1 and device.type == "cuda":
        dist.init_process_group(
            backend="gloo" if sys.platform == "win32" else "nccl",
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )

    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

class _HoldoutSet:
    """Fixed-length excerpts, kept in RAM and batched on demand.

    Cropped to equal length rather than padded: padding silence into the mel
    would put a batch-dependent constant into the metric. Held as excerpts
    rather than pre-formed batches so :meth:`shrink` can re-batch them.
    """

    def __init__(self, items, batch_size, label="holdout"):
        self.items = list(items)
        self.batch_size = max(1, int(batch_size))
        self.label = label
        # Ground truth does not depend on the weights, so its mel is computed
        # once for the life of the run rather than once per evaluation.  Keyed
        # by length as well as by batch, so a decoder whose output length moves
        # cannot silently be compared against the wrong excerpt.
        self._target_mels = {}

    def __len__(self):
        return len(self.items)

    @property
    def frames(self):
        return self.items[0][0].shape[0] if self.items else 0

    def seconds(self, config):
        return self.frames * config.data.hop_length / config.data.sample_rate

    def batches(self):
        for start in range(0, len(self.items), self.batch_size):
            chunk = self.items[start : start + self.batch_size]
            yield start // self.batch_size, self._collate(chunk)

    @staticmethod
    def _collate(chunk):
        phone = torch.stack([item[0] for item in chunk])
        pitch = torch.stack([item[1] for item in chunk])
        pitchf = torch.stack([item[2] for item in chunk])
        spectrogram = torch.stack([item[3] for item in chunk])
        wave = torch.stack([item[4] for item in chunk])
        sid = torch.cat([item[5] for item in chunk])
        # Every excerpt is the same length by construction, which is the whole
        # point: no padding, so no per-item lengths to carry.
        frames = torch.full((len(chunk),), phone.shape[1], dtype=torch.long)
        samples = torch.full((len(chunk),), wave.shape[-1], dtype=torch.long)
        return (
            phone,
            frames,
            pitch,
            pitchf,
            spectrogram,
            frames,
            wave,
            samples,
            sid,
        )

    def target_mel(self, index, length, factory):
        cached = self._target_mels.get(index)
        if cached is None or cached[0] != length:
            cached = (length, factory())
            self._target_mels[index] = cached
        return cached[1]

    def shrink(self):
        """Halve the batch.  False once there is nothing left to halve."""
        if self.batch_size <= 1:
            return False
        self.batch_size = max(1, self.batch_size // 2)
        self._target_mels.clear()
        return True


def _uniform_excerpts(
    dataset,
    indices,
    crop_frames,
    config,
    batch_size,
    label="holdout",
    fixed=False,
    limit=None,
):
    """Load rows and crop them all to one shared length.

    ``crop_frames`` is a ceiling rather than a demand: unless ``fixed``, the
    crop lands at the lower quartile of what the rows actually carry, so three
    excerpts in four survive it.  Taking the ceiling literally would leave the
    set to whichever recording happened to be longest.

    ``fixed`` is for the training probe, which has to be cropped exactly like
    the holdout or the two numbers are not comparable.
    """
    hop = config.data.hop_length
    rows = []
    for index in indices:
        spectrogram, wave, phone, pitch, pitchf, sid = dataset[index]
        frames = min(spectrogram.shape[-1], phone.shape[0], wave.shape[-1] // hop)
        if frames > 0:
            rows.append((frames, spectrogram, wave, phone, pitch, pitchf, sid))
    if not rows:
        return None

    if fixed:
        crop = max(1, int(crop_frames))
    else:
        ordered = sorted(row[0] for row in rows)
        crop = max(1, min(int(crop_frames), ordered[len(ordered) // 4]))

    items = []
    for frames, spectrogram, wave, phone, pitch, pitchf, sid in rows:
        if frames < crop:
            continue
        # Cloned: the crops are views onto whole files, and keeping views would
        # keep every one of those files resident for the life of the run.
        items.append(
            (
                phone[:crop, :].clone(),
                pitch[:crop].clone(),
                pitchf[:crop].clone(),
                spectrogram[:, :crop].clone(),
                wave[:, : crop * hop].clone(),
                sid.clone(),
            )
        )
        if limit is not None and len(items) >= int(limit):
            break
    if not items:
        return None
    return _HoldoutSet(items, batch_size, label=label)


def prepare_dataloaders(config, n_gpus, rank, batch_size):
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
        holdout_split_indices,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data, n_mel_bins=config.model.inter_channels)

    # Carve a held-out set out of the dataset before anything else sees it.
    # Training loss cannot detect overtraining by construction -- it keeps
    # falling while generalisation rots -- so this is the only signal that can.
    holdout_dataset = None
    if overtrain_detector:
        # The evaluation cost is seconds of audio to synthesise, so the budget
        # is set in seconds.  ``lengths`` is in frames (see ``_filter``).
        max_seconds = float(getattr(config.train, "holdout_max_seconds", 120.0))
        train_indices, holdout_indices = holdout_split_indices(
            train_dataset.audiopaths_and_text,
            fraction=float(getattr(config.train, "holdout_fraction", 0.02)),
            minimum=int(getattr(config.train, "holdout_min_slices", 16)),
            maximum=int(getattr(config.train, "holdout_max_slices", 96)),
            seed=int(getattr(config.train, "seed", 1234)),
            lengths=train_dataset.lengths,
            max_frames=(
                max_seconds * config.data.sample_rate / config.data.hop_length
                if max_seconds > 0
                else None
            ),
        )
        if holdout_indices:
            rows = train_dataset.audiopaths_and_text
            lengths = train_dataset.lengths
            holdout_dataset = copy.copy(train_dataset)
            holdout_dataset.audiopaths_and_text = [rows[i] for i in holdout_indices]
            holdout_dataset.lengths = [lengths[i] for i in holdout_indices]
            train_dataset.audiopaths_and_text = [rows[i] for i in train_indices]
            train_dataset.lengths = [lengths[i] for i in train_indices]
            if rank == 0:
                sources = len({r[0].rsplit("_", 1)[0] for r in holdout_dataset.audiopaths_and_text})
                info(
                    f"{len(holdout_indices)} slices from {sources} source "
                    f"recordings held out of {len(rows)} - never trained on.",
                    tag="[HOLDOUT]",
                )
        elif rank == 0:
            warning(
                "Dataset too small to hold one out; overtrain detection is off.",
                tag="[HOLDOUT]",
            )

    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=enable_persistent_workers,
        prefetch_factor=2
    )
    train_loader_safety(train_loader)

    # Materialised once and kept: the point of a holdout is that every
    # evaluation scores the exact same audio, and re-reading it through a
    # loader each time would also cost more than the forward pass.
    holdout_set = None
    probe_set = None
    if holdout_dataset is not None and rank == 0:
        crop_frames = max(
            1,
            int(
                float(getattr(config.train, "holdout_crop_seconds", 3.0))
                * config.data.sample_rate
                / config.data.hop_length
            ),
        )
        eval_batch = int(getattr(config.train, "holdout_batch_size", 4))
        holdout_set = _uniform_excerpts(
            holdout_dataset,
            range(len(holdout_dataset.audiopaths_and_text)),
            crop_frames,
            config,
            eval_batch,
            label="holdout",
        )
        if holdout_set is None:
            warning(
                "Held-out rows were all too short to score; detection is off.",
                tag="[HOLDOUT]",
            )
        else:
            info(
                f"Scoring {len(holdout_set)} excerpts of "
                f"{holdout_set.seconds(config):.1f}s each, batched "
                f"{holdout_set.batch_size} at a time.",
                tag="[HOLDOUT]",
            )

        # The same measurement on slices the model *has* been trained on.  The
        # held-out curve alone mixes two movements -- the model is still
        # learning, and it is starting to memorise -- and only their difference
        # is overtraining.  Subtracting the probe removes the shared trend, so
        # the turn shows up earlier and more cleanly than in the absolute
        # number.  Same crop and same count as the holdout, or the two are not
        # comparable; sampled deterministically, so a resume scores the same
        # slices rather than leaking a fresh draw into the comparison.
        if holdout_set is not None and bool(
            getattr(config.train, "holdout_train_probe", True)
        ):
            pool = list(range(len(train_dataset.audiopaths_and_text)))
            sampled = Random(int(getattr(config.train, "seed", 1234))).sample(
                pool, min(len(pool), 2 * len(holdout_set))
            )
            probe_set = _uniform_excerpts(
                train_dataset,
                sampled,
                holdout_set.frames,
                config,
                eval_batch,
                label="train probe",
                fixed=True,
                limit=len(holdout_set),
            )

    return train_loader, holdout_set, probe_set

def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    model_config = config.model.__dict__.copy()
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **model_config,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
    )

def get_d_model(config, vocoder, use_checkpointing):
    vocoder = normalize_vocoder(vocoder)
    discriminator_id = get_discriminator_id(vocoder)
    # JSON has no tuples, so a configured schedule arrives as nested lists.  The
    # discriminators normalise them themselves; passing them through untouched
    # keeps this function free of the branch layout.
    def setting(name, default=None):
        """``None``/absent means "use the default"; ``[]`` means "none of these".

        This was ``value or default``, which collapsed the two: a config asking
        for *no* period branches got the full set back, and there was no way to
        turn a whole family off from JSON.
        """

        value = getattr(config.model, name, None)
        return default if value is None else value

    from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined

    # ``mpd_msd`` is Applio's v2 (8 periods); ``mpd_msd_v3`` is what it picks
    # for RefineGAN (5 periods + 3 multi-resolution spectrogram branches).
    # ``d_version`` overrides the default -- v3 does not fit an 8 GB card at
    # batch 8 (6.42 GiB / 5912 ms/step vs v2's 4.64 GiB / 498 ms/step); v3l
    # (v3 with the last two layers frequency-downsampled) was tried as a
    # cheaper default but measured no improvement over v3 on newer hardware,
    # so it stays available only as an explicit override.
    version = "v3" if discriminator_id == "mpd_msd_v3" else "v2"
    version = str(getattr(config.model, "d_version", None) or version)
    # ``d_use_*`` switches a whole family off; ``d_periods``/``d_resolutions``
    # replace its content when it's on (``None`` keeps the preset's).
    return MPD_MSD_Combined(
        config.model.use_spectral_norm,
        use_checkpointing=use_checkpointing,
        version=version,
        periods=[] if not setting("d_use_periods", True) else setting("d_periods"),
        resolutions=(
            [] if not setting("d_use_resolutions", True) else setting("d_resolutions")
        ),
        frequency_strides=setting("d_frequency_strides"),
        use_msd=bool(setting("d_use_msd", True)),
        # UnivHD (arXiv 2512.03486) is opt-in and *additive*: it appends a
        # harmonic-order branch and removes nothing, which is how the paper
        # runs it.  Off by default because it is unmeasured on this fork -- the
        # gains it reports are on harmonic structure and F0RMSE for singing,
        # not on the frame-rate mirroring, which its ERB bandwidths are too
        # wide to separate above ~700 Hz.
        sample_rate=int(config.data.sample_rate),
        use_univhd=bool(setting("d_use_univhd", False)),
        univhd_n_fft=int(setting("d_univhd_n_fft", 2048)),
        univhd_hop_length=int(setting("d_univhd_hop_length", 256)),
        univhd_harmonics=int(setting("d_univhd_harmonics", 10)),
        univhd_bins_per_octave=int(setting("d_univhd_bins_per_octave", 24)),
        univhd_f_min=float(setting("d_univhd_f_min", 80.0)),
        univhd_channels=int(setting("d_univhd_channels", 32)),
        univhd_half_harmonic=bool(setting("d_univhd_half_harmonic", True)),
    )


def _prior_gap(
    model,
    m_p,
    x_mask,
    ids_slice,
    segment_size,
    pitchf,
    sid,
    target,
    full_output,
    config,
):
    """Is the Gaussian posterior carrying anything the prior does not have?

    Decodes twice -- once from the posterior sample ``z`` that produced
    ``full_output``, once from the prior mean pushed through the flow -- and
    returns the mel L1 gap between them.  A rate near zero in ``diag/kl_*`` is
    ambiguous between "prior predicts posterior" (fine) and "posterior
    collapsed" (not); a large gap here means the former, a small one the
    latter.  Diagnostic only, under ``no_grad``: optimising this gap directly
    would reward a destructive latent rather than an informative one.
    """
    with torch.no_grad():
        speaker = model.emb_g(sid).unsqueeze(-1)
        z_prior = model.flow(m_p * x_mask, x_mask, g=speaker, reverse=True) * x_mask
        z_slice = commons.slice_segments(z_prior, ids_slice, segment_size, dim=3)
        pitchf_slice = commons.slice_segments(pitchf, ids_slice, segment_size, dim=2)
        prior_output = model.dec(z_slice, pitchf_slice, g=speaker)

        target_mel = wave_to_mel(config, target)
        full_error = F.l1_loss(
            wave_to_mel(config, full_output).float(), target_mel.float()
        )
        prior_error = F.l1_loss(
            wave_to_mel(config, prior_output).float(), target_mel.float()
        )
    return (prior_error - full_error).detach(), prior_error.detach()


def _cache_mean(cache) -> float:
    """Mean of a rolling cache of device scalars, in one transfer.

    The caches these read are filled every step and read once per logging
    interval, so they hold tensors rather than floats: a per-step ``.item()``
    is a synchronisation in the middle of the training step, and this is the
    one place the window actually has to reach the host.
    """
    return torch.stack(list(cache)).mean().item()


def _clip_or_sample_grad_norm(
    parameters,
    max_norm,
    step,
    sample_interval,
):
    max_norm = float(max_norm)
    should_measure = math.isfinite(max_norm) or step % max(1, sample_interval) == 0
    if not should_measure:
        return None
    return clip_grad_norm_(parameters, max_norm=max_norm)


#: Wall-clock seconds between machine-readable progress lines.  Rich's bar is
#: for a terminal; this is for whatever is reading the pipe.
_MACHINE_PROGRESS_INTERVAL = 1.0
_last_machine_progress = 0.0


def _emit_machine_progress(
    epoch, total_epochs, batch, total_batches, step, metrics, rank
):
    """Print one parseable progress line for a GUI front-end, when stdout isn't
    a terminal (Rich's bar renders nothing there). Throttled by wall clock
    rather than batch count, since batch rate varies widely by configuration.
    """
    global _last_machine_progress
    if rank != 0:
        return
    try:
        if sys.stdout.isatty():
            return
    except (AttributeError, ValueError):
        return

    now = ttime()
    finished = batch >= total_batches
    if not finished and now - _last_machine_progress < _MACHINE_PROGRESS_INTERVAL:
        return
    _last_machine_progress = now

    print(
        f"[PROGRESS] epoch={epoch}/{total_epochs} "
        f"batch={batch}/{total_batches} step={step} "
        f"{metrics or ''}".rstrip(),
        flush=True,
    )


def planned_step_count(total_epoch_count: int, train_loader, max_steps: int = 0) -> int:
    """How many optimizer steps this run will take.

    Every schedule below this line is step-denominated but stated as an
    absolute literal, so it means something different on an 8k fine-tune than
    on a hundred-thousand-step pretrain; this is what :func:`fit_schedule` and
    :func:`fit_eval_interval` rescale against.  Returns 0 when the budget is
    unknown, which every caller treats as "leave the configured value alone".
    """
    per_epoch = max(1, len(train_loader))
    planned = max(0, int(total_epoch_count)) * per_epoch
    limit = max(0, int(max_steps))
    if limit:
        planned = min(planned, limit) if planned else limit
    return planned


def fit_schedule(configured: int, planned_steps: int, fraction: float, minimum: int = 1) -> int:
    """Shrink a step-denominated schedule to fit inside the run.

    Only ever shrinks: a run long enough for the configured value gets it back
    untouched.  ``fraction`` is the share of the run the schedule may occupy.
    """
    configured = max(0, int(configured))
    if planned_steps <= 0 or configured <= 0:
        return configured
    return max(minimum, min(configured, int(planned_steps * fraction)))


def fit_eval_interval(configured: int, planned_steps: int, patience: int) -> int:
    """An evaluation interval that lets ``patience`` actually be reached.

    At one evaluation per 2000 steps, an 8k fine-tune only gets 4 evaluations
    against a patience of 8, so the detector can never fire. Sizing for ~3x
    patience keeps it a detector; only ever shrinks, like :func:`fit_schedule`.
    """
    configured = max(1, int(configured))
    if planned_steps <= 0:
        return configured
    wanted = planned_steps // max(1, 3 * max(1, int(patience)))
    return max(1, min(configured, wanted)) if wanted else configured


def _cpu_state_dict(source):
    """Detached CPU copy of a model's or a ``WeightEMA``'s weights."""
    if hasattr(source, "cpu_state_dict"):
        return source.cpu_state_dict()
    module = source.module if hasattr(source, "module") else source
    return {
        key: tensor.detach().to("cpu", copy=True)
        for key, tensor in module.state_dict().items()
    }


def _holdout_metrics(
    net_g,
    excerpts,
    config,
    device,
    want_latent=True,
    noise_scale=0.0,
):
    """Score held-out audio down the *inference* path, in a single pass.

    ``mel_l1`` scores the prior path, not the training forward: the training
    path samples a posterior that has seen the target spectrogram, so a
    memorising model would keep scoring well there after it stops
    generalising.  Prior, posterior and their gap come from one pass over one
    set of weights, so the numbers stay comparable.

    ``noise_scale=0`` decodes the prior mean instead of ``infer``'s default
    0.66666 draw, making the metric a pure function of the weights.  Plain L1
    on the log-mel: the adversarial/feature-matching/KL terms are scored
    against a discriminator and schedule that keep moving independently of
    the generator.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    # ``flow`` is what turns a prior draw into something the decoder can use;
    # without it there is no inference path to rebuild by hand and ``infer`` is
    # the only way in.  ``enc_q`` is dropped for export, which is the one state
    # in which the posterior half cannot be measured at all.
    manual_prior = getattr(model, "flow", None) is not None
    want_latent = (
        want_latent and manual_prior and getattr(model, "enc_q", None) is not None
    )
    was_training = model.training
    model.eval()
    totals = {"mel_l1": 0.0}
    if want_latent:
        totals["latent_gap"] = 0.0
        totals["latent_posterior"] = 0.0
    count = 0

    # The metric has to be a pure function of the weights, and the prior draw
    # is noise: at ``noise_scale`` 0 there is none, and above it the seed is
    # pinned so every evaluation sees the same draw.  Restoring the state
    # afterwards keeps a variable number of draws from shifting the training
    # stream underneath the run.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(0x5EED)
    if cuda_rng_state is not None:
        torch.cuda.manual_seed_all(0x5EED)

    try:
        # The same autocast the training step runs under, so the metric
        # measures the model as it is actually being trained, and
        # ``inference_mode`` rather than ``no_grad`` because nothing here is
        # ever differentiated.
        with torch.inference_mode(), autocast(
            device_type="cuda", enabled=use_amp, dtype=amp_dtype
        ):
            for index, batch in excerpts.batches():
                # Not named ``spec``: ``test_run_spec`` audits every
                # ``spec.<field>`` in this file against the run spec, and a
                # local of that name collides with the check.
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    holdout_spec,
                    holdout_spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = batch
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
                pitch = pitch.to(device, non_blocking=True)
                pitchf = pitchf.to(device, non_blocking=True)
                sid = sid.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)

                g = model.emb_g(sid).unsqueeze(-1)
                if manual_prior:
                    m_p, logs_p, x_mask = model.enc_p(
                        phone=phone, pitch=pitch, lengths=phone_lengths
                    )
                    z_p = m_p
                    if noise_scale:
                        z_p = (
                            m_p
                            + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale
                        )
                    z_prior = model.flow(z_p * x_mask, x_mask, g=g, reverse=True)
                    z_prior = z_prior * x_mask
                    frames = min(z_prior.shape[-1], pitchf.shape[-1])
                    prior_wave = model.dec(
                        z_prior[..., :frames], pitchf[..., :frames], g=g
                    )
                else:
                    prior_wave, *_ = model.infer(
                        phone, phone_lengths, pitch, pitchf, sid, 0
                    )
                    frames = pitchf.shape[-1]

                posterior_wave = None
                if want_latent:
                    holdout_spec = holdout_spec.to(device, non_blocking=True)
                    holdout_spec_lengths = holdout_spec_lengths.to(
                        device, non_blocking=True
                    )
                    z_q, _, _, spec_mask = model.enc_q(
                        holdout_spec, holdout_spec_lengths, g=g
                    )
                    posterior_wave = model.dec(
                        (z_q * spec_mask)[..., :frames], pitchf[..., :frames], g=g
                    )

                # The decoder rebuilds the waveform from frame-rate features,
                # so its length lands within a hop of the excerpt rather than
                # on it.
                length = min(prior_wave.shape[-1], wave.shape[-1])
                if posterior_wave is not None:
                    length = min(length, posterior_wave.shape[-1])
                if length <= config.data.filter_length:
                    continue
                target_mel = excerpts.target_mel(
                    index,
                    length,
                    lambda: wave_to_mel(config, wave[..., :length], num_mels=None),
                )
                prior_mel = wave_to_mel(config, prior_wave[..., :length], num_mels=None)
                totals["mel_l1"] += float(F.l1_loss(prior_mel, target_mel))
                if posterior_wave is not None:
                    posterior_mel = wave_to_mel(
                        config, posterior_wave[..., :length], num_mels=None
                    )
                    totals["latent_gap"] += float(F.l1_loss(prior_mel, posterior_mel))
                    totals["latent_posterior"] += float(
                        F.l1_loss(posterior_mel, target_mel)
                    )
                count += 1
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            model.train()

    if not count:
        return {"mel_l1": float("nan")}
    return {key: value / count for key, value in totals.items()}


def _holdout_metrics_resilient(net_g, excerpts, config, device, **kwargs):
    """:func:`_holdout_metrics`, but an oversized batch is halved, not fatal.

    The evaluation decodes seconds of audio per item where a training step
    decodes a fraction of one, so its batch can be the largest allocation in
    the run despite storing no gradients -- and it would be a poor trade to
    lose a training run to a diagnostic.
    """
    while True:
        try:
            return _holdout_metrics(net_g, excerpts, config, device, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not excerpts.shrink():
                raise
            warning(
                f"{excerpts.label} evaluation ran out of memory; retrying at "
                f"batch {excerpts.batch_size}.",
                tag="[HOLDOUT]",
            )


class _OvertrainMonitor:
    """Find the last point where held-out quality was still improving.

    Overtraining isn't visible in the training loss, so this scores audio the
    model has never trained on and watches for the minimum. Two things are
    judged against a *noise band* rather than a fixed threshold (which means
    something different in every run): the score is median-filtered over
    ``smoothing`` evaluations before anything is decided on it, and
    ``min_delta`` is a floor under a band computed as the residual spread of
    recent scores about a trend line, so a still-improving run gets a narrow
    band and a flat noisy one gets a wide one.

    Weights to keep: lowest smoothed score; ties inside the band favour the
    earlier step. Has the run stopped improving: the patience counter ignores
    anything inside the band, so noise can't reset it forever.
    """

    def __init__(self, patience=8, min_delta=0.001, smoothing=3, noise_window=6):
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.smoothing = max(1, int(smoothing))
        self.noise_window = max(4, int(noise_window))
        #: Best *smoothed* score, and the step whose weights produced it.
        self.best = float("inf")
        self.best_step: int | None = None
        self.state_dict: dict | None = None
        #: Best single score ever seen.  Not a selection criterion; it is what
        #: decides whether an evaluation is close enough to the running best to
        #: be worth cloning weights for.
        self.best_raw = float("inf")
        # The last score good enough to count as progress.  Separate from
        # ``best`` because it moves on a coarser ratchet.
        self.patience_reference = float("inf")
        self.since_progress = 0
        self.history: list[tuple[int, float]] = []
        self.smoothed = float("nan")
        self.sigma = 0.0
        # Once the run has been called, every further evaluation is paying for
        # a number nothing acts on; see :meth:`backoff`.
        self.interval_scale = 1
        self._window = deque(maxlen=self.smoothing)

    def _band(self, reference):
        """How large a change has to be before it is evidence rather than noise."""
        if not math.isfinite(reference):
            return 0.0
        return max(self.sigma, self.min_delta * abs(reference))

    def _noise_sigma(self) -> float:
        """Residual spread of recent scores about a trend line (not a mean: a
        genuinely improving run has a large spread about its mean but a small
        one about its slope). 0 until there are enough points to fit.
        """
        recent = [value for _, value in self.history[-self.noise_window :]]
        count = len(recent)
        if count < 4:
            return 0.0
        mean_x = (count - 1) / 2.0
        mean_y = sum(recent) / count
        variance_x = sum((index - mean_x) ** 2 for index in range(count))
        covariance = sum(
            (index - mean_x) * (value - mean_y) for index, value in enumerate(recent)
        )
        slope = covariance / variance_x if variance_x else 0.0
        intercept = mean_y - slope * mean_x
        residuals = [
            value - (slope * index + intercept) for index, value in enumerate(recent)
        ]
        return math.sqrt(sum(r * r for r in residuals) / max(1, count - 2))

    def update(self, source, value: float, step: int) -> bool:
        """``source``: a model or a ``WeightEMA``. Returns whether this moved
        the best. With a centred filter, the weights kept are the middle of
        the window, not the newest evaluation -- otherwise a snapshot of the
        live model would be credited with a score it never earned.
        """
        if not math.isfinite(value):
            return False
        self.history.append((int(step), float(value)))
        self.sigma = self._noise_sigma()

        # Only clone weights for evaluations close enough to the running best
        # to still win one; a window entry without weights can never be
        # selected.
        competitive = (
            not math.isfinite(self.best_raw)
            or value <= self.best_raw + 2.0 * self._band(self.best_raw)
        )
        self._window.append(
            (int(step), float(value), _cpu_state_dict(source) if competitive else None)
        )
        self.best_raw = min(self.best_raw, float(value))

        if len(self._window) < self.smoothing:
            # Until the window fills, behave unsmoothed rather than report a
            # half-formed median.
            self.smoothed = float(value)
            center_step, _center_value, center_state = self._window[-1]
        else:
            scores = sorted(entry[1] for entry in self._window)
            self.smoothed = scores[len(scores) // 2]
            center_step, _center_value, center_state = self._window[
                len(self._window) // 2
            ]

        improved = False
        if center_state is not None and self.smoothed < self.best - self._band(
            self.best
        ):
            improved = True
            self.best = float(self.smoothed)
            self.best_step = int(center_step)
            self.state_dict = center_state

        if self.smoothed < self.patience_reference - self._band(
            self.patience_reference
        ):
            self.patience_reference = float(self.smoothed)
            self.since_progress = 0
        else:
            self.since_progress += 1
        return improved

    def backoff(self, factor: int = 4) -> int:
        """Evaluate less often once overtraining has already been flagged
        (with ``stop_on_overtrain`` off, the run keeps going and a run can
        still come back, so evaluation continues, just less often).
        """
        self.interval_scale = max(1, int(self.interval_scale * max(1, int(factor))))
        return self.interval_scale

    @property
    def overtrained(self) -> bool:
        return self.state_dict is not None and self.since_progress >= self.patience


def _deliverable_weights(overtrain_monitor, ema, model_g):
    """The weights this run would hand you if it stopped right now: holdout
    best, then EMA, then live weights, in order of how much each source
    knows. Returns ``(state_dict, label)``.
    """
    if overtrain_monitor is not None and overtrain_monitor.state_dict is not None:
        return overtrain_monitor.state_dict, f"holdout best @ {overtrain_monitor.best_step}"
    if ema is not None:
        return ema.cpu_state_dict(), f"EMA ({ema.updates} updates)"
    # A copy, not ``model_g.state_dict()`` itself: that hands back live
    # parameter tensors, so anything mutating the weights before the export is
    # written -- a schedule-free optimizer returning to its training iterate,
    # for one -- would change what has already been chosen.  The other two
    # branches already return CPU copies.
    return _cpu_state_dict(model_g), "live weights"


def _checkpoint_extra(grad_scaler):
    """Training-loop state that a resume cannot re-derive, for the D checkpoint.

    Plain scalars/dicts only: ``extra`` is unpickled under
    ``weights_only=True``. Returns ``None`` when empty, which
    ``save_checkpoint`` treats as "omit the key".
    """
    extra = {}
    if grad_scaler is not None:
        extra["grad_scaler"] = grad_scaler.state_dict()
        extra["amp_skipped_steps"] = int(amp_skipped_steps)
    return extra or None


def _generator_gradient_metrics(net_g):
    """Return gradient and gradient-to-parameter norms for generator subsystems."""
    model = net_g.module if hasattr(net_g, "module") else net_g
    groups = {
        "content_encoder": [],
        "posterior": [],
        "prior": [],
        "decoder": [],
        "speaker": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("enc_p."):
            group = "content_encoder"
        elif name.startswith("enc_q."):
            group = "posterior"
        elif name.startswith("flow."):
            group = "prior"
        elif name.startswith("dec."):
            group = "decoder"
        elif name.startswith("emb_g."):
            group = "speaker"
        else:
            group = "other"
        groups[group].append(parameter)

    metrics = {}
    for group, parameters in groups.items():
        if not parameters:
            continue
        parameter_norm = torch.sqrt(
            sum(parameter.detach().float().square().sum() for parameter in parameters)
        )
        gradient_terms = [
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        gradient_norm = (
            torch.sqrt(sum(gradient_terms))
            if gradient_terms
            else parameter_norm.new_zeros(())
        )
        metrics[f"grad_norm_{group}"] = gradient_norm
        metrics[f"grad_to_param_{group}"] = gradient_norm / parameter_norm.clamp_min(1e-8)
    return metrics


def build_decoder_param_groups(net_g, base_lr):
    """
    Build optimizer param groups with differential LR.

    `dec_lr_scale` covers the decoder/vocoder, `vae_lr_scale` everything else
    ( the frontend and the speaker embedding ).  Returns None when neither is
    configured, so the optimizer falls back to a single group.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return None

    if dec_lr_scale is None and vae_lr_scale is None:
        return None

    # scale -> [params]  ( same scale == same effective LR )
    groups = {}

    def add(param, scale):
        groups.setdefault(scale, []).append(param)

    decoder_scale = dec_lr_scale if dec_lr_scale is not None else 1.0
    rest_scale = vae_lr_scale if vae_lr_scale is not None else 1.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            add(param, decoder_scale if name.startswith("dec.") else rest_scale)

    return [
        {"params": params, "lr": base_lr * scale, "lr_scale": scale}
        for scale, params in groups.items()
    ]


def get_optimizers(
    net_g,
    net_d,
    config,
    optimizer_choice_g,
    optimizer_choice_d,
    custom_lr_g,
    custom_lr_d,
    use_custom_lr,
    total_epoch_count,
    train_loader
):
    lr_g = custom_lr_g if use_custom_lr else config.train.learning_rate_g
    lr_d = custom_lr_d if use_custom_lr else config.train.learning_rate_d
    num_batches = len(train_loader)

    g_param_groups = build_decoder_param_groups(net_g, lr_g)

    optim_g = _make_optimizer(net_g, optimizer_choice_g, lr_g, num_epochs=total_epoch_count, num_batches=num_batches, param_groups=g_param_groups)
    optim_d = _make_optimizer(
        net_d,
        optimizer_choice_d,
        lr_d,
        num_epochs=total_epoch_count,
        num_batches=num_batches,
        lazy_reg_interval=None,
    )

    return optim_g, optim_d


def apply_frontend_freeze(net_g, rank):
    """
    Freeze the frontend for fine-tuning, driven by the `freeze_vae` flag.

    Everything outside `dec.` and `emb_g.` is frozen, which means `enc_p`, the
    posterior encoder and the flow.
    """
    if not freeze_vae:
        if rank == 0:
            info("Frontend: nothing frozen.", tag="[INIT]")
        return

    model = net_g.module if hasattr(net_g, "module") else net_g
    frozen_params = 0
    for name, param in model.named_parameters():
        if not name.startswith("dec.") and not name.startswith("emb_g.") and param.requires_grad:
            param.requires_grad = False
            frozen_params += param.numel()

    if rank == 0:
        info(f"Frontend frozen: everything but dec/emb_g ({frozen_params:,} params).", tag="[INIT]")


def apply_training_freezes(net_g, rank):
    apply_frontend_freeze(net_g, rank)


def apply_resume_lr_override(optim_g, optim_d=None):
    """Re-anchor G and/or D param groups to `resume_lr` after loading optimizer
    state. Must run after load_checkpoint (needs the saved lr/initial_lr) and
    before prepare_schedulers (which snapshots the new initial_lr). Each
    group's saved decay ratio is preserved so the schedule keeps its position
    while the value restarts at resume_lr x its per-group lr_scale.
    """
    if resume_lr is None:
        return

    targets = {"full": ("g", "d"), "g": ("g",), "d": ("d",)}.get(resume_lr_target, ("g", "d"))
    optims = [(optim_g, "G")] if "g" in targets else []
    if "d" in targets and optim_d is not None:
        optims.append((optim_d, "D"))

    for optim, label in optims:
        for param_group in optim.param_groups:
            saved_lr = param_group.get("lr", 0.0)
            saved_initial = param_group.get("initial_lr", 0.0)
            decay = (saved_lr / saved_initial) if saved_initial else 1.0
            scale = param_group.get("lr_scale", 1.0)
            new_lr = resume_lr * scale * param_group.get("lazy_reg_scale", 1.0)
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr / decay if decay else new_lr

    parts = []
    for optim, label in optims:
        lrs = ", ".join("{:.2e}".format(g["lr"]) for g in optim.param_groups)
        parts.append(f"{label}: {lrs}")
    info(f"Resume LR override: base {resume_lr:.2e} -> " + " | ".join(parts), tag="[OVERRIDE]")


def setup_models_for_training(net_g, net_d, device, device_id, n_gpus):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)
    net_d = net_d.to(device_id) if device.type == "cuda" else net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(
            net_d,
            device_ids=[device_id],
            find_unused_parameters=bool(
                getattr(net_d, "uses_branchwise_r1", False)
            ),
        )

    return net_g, net_d


def enable_vocoder_compile(net_g, device, rank):
    if not compile_vocoder:
        return False
    if device.type != "cuda":
        if rank == 0:
            info(VOCODER_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    cache_dir = os.path.join(current_dir, "logs", ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    mode = torch_compile_mode
    model = net_g.module if hasattr(net_g, "module") else net_g
    enabled = model.enable_decoder_compile(mode=mode)
    if not enabled and rank == 0:
        info(VOCODER_COMPILE_NOT_SUPPORTED, tag="[INIT]")
    if enabled and rank == 0:
        info(VOCODER_COMPILE_ENABLED.format(mode=mode), tag="[INIT]")
    return enabled


def enable_discriminator_compile(net_d, config, device, rank):
    """Compile the discriminator, driven by ``compile_discriminator`` in the
    config (not a run-spec flag, since it travels with the architecture).
    ``torch_compile_mode`` is not consulted: ``reduce-overhead`` records CUDA
    graphs, which this loop can't support, so the mode is fixed to plain
    fusion.
    """
    if not bool(getattr(config.train, "compile_discriminator", False)):
        return False
    if device.type != "cuda":
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    cache_dir = os.path.join(current_dir, "logs", ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    model = net_d.module if hasattr(net_d, "module") else net_d
    enable = getattr(model, "enable_compile", None)
    if enable is None:
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NOT_SUPPORTED, tag="[INIT]")
        return False
    enabled = enable(mode="default")
    if enabled and rank == 0:
        info(DISCRIMINATOR_COMPILE_ENABLED.format(mode="default"), tag="[INIT]")
    return enabled


def _assert_resumable_architecture(net_g, checkpoint_path):
    """Refuse to resume from a checkpoint built for a different architecture.

    Unlike the pretrained-path guard, a missing id here counts as a mismatch:
    a resumed run predating the id is one of this fork's own old runs, whose
    layout has since changed.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The excitation is checked whatever the id says, because the id is pinned
    # to ``vits_gaussian_v1`` for every RefineGAN source on purpose -- see
    # ``get_architecture_id`` -- and this stack resumes non-strictly.
    assert_excitation_matches(model, checkpoint, origin="checkpoint")
    expected = getattr(model, "architecture_id", None)
    if not expected or expected == "vits_gaussian_v1":
        return
    found = checkpoint.get("architecture_id")
    if found == expected:
        return
    raise ValueError(
        f"Cannot resume from '{os.path.basename(checkpoint_path)}': it was "
        f"written for architecture '{found or 'unknown'}', but this run builds "
        f"'{expected}'. Move the old checkpoints out of the experiment folder "
        f"to start this architecture fresh."
    )


def checkpoint_step_from_path(path):
    if not path or path in ("", "None"):
        return 0
    match = re.search(r"(?:^|[\\/])G_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else 0


def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, sample_rate, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank, reset_pretrained_embeddings=False):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    resumed_g_path = None
    # Training-loop controller state travelling with the D checkpoint.  Empty on
    # a fresh run, on a pretrained start, and on every checkpoint written before
    # the key existed -- all three of which mean "start the controller cold".
    resumed_extra_d = {}
    try:
        info("Starting the training ...", tag="[INIT]")

        # Get latest G and D based on the highest steps count in the filename
        def get_highest_checkpoint(prefix):
            pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
            files = []
            for f in os.listdir(experiment_dir):
                match = pattern.match(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(experiment_dir, f)))
            return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

        # Confirm presence of checkpoints
        # If they exist, attempt to resume the training
        g_checkpoint_path = get_highest_checkpoint("G_")
        d_checkpoint_path = get_highest_checkpoint("D_")
        resumed_g_path = g_checkpoint_path
        if g_checkpoint_path and d_checkpoint_path:

            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

            # Apply decoder / frontend layer freezes for the selected phase.
            apply_training_freezes(net_g, rank)

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

            # Resuming loads the generator non-strictly for the VITS-latent
            # vocoders, so nothing downstream would complain if the checkpoint
            # in this folder belonged to a different decoder: the frontend
            # weights match either way, the decoder is silently left at its
            # init, and the run restarts at the old step count with a fully
            # trained discriminator against a random generator.  The vocoders
            # share a config shape and a folder layout, which makes that one
            # edited ``vocoder`` field away.
            _assert_resumable_architecture(net_g, g_checkpoint_path)

            # Load the model and optim states
            generator_strict_load = strict_load
            _, _, _, epoch_str, _ = load_checkpoint(
                g_checkpoint_path,
                net_g,
                None if reset_optimizer_for_run else optim_g,
                generator_strict_load,
            )
            _, _, _, epoch_str, extra_d = load_checkpoint(
                d_checkpoint_path,
                net_d,
                None if reset_optimizer_for_run else optim_d,
                strict_load,
            )
            resumed_extra_d = extra_d or {}

            # resume_lr re-anchors G and/or D to the given base LR.
            if not reset_optimizer_for_run:
                apply_resume_lr_override(optim_g, optim_d)
            elif rank == 0:
                info("Fine-tune optimizer state reset.", tag="[INIT]")

            #epoch_str += 1
            #global_step = (epoch_str - 1) * len(train_loader)

            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            success(f"(G) & (D) resumed at global_step {global_step}, epoch {epoch_str - 1}.", tag="[RESUME]")

        else:
            raise FileNotFoundError("No checkpoints found.")

    except FileNotFoundError:
    # If no checkpoints are available, using the Pretrains directly
        epoch_str = 1
        global_step = (
            checkpoint_step_from_path(pretrainG)
            if finetune_phase
            else 0
        )

        # Loading the pretrained Generator model
        if pretrainG not in ["", "None"]:
            if rank == 0:
                info(f"Loading pretrained (G) '{pretrainG}'", tag="[INIT]")
            checkpoint = torch.load(pretrainG, map_location="cpu", weights_only=True)
            expected_architecture = getattr(
                net_g.module if hasattr(net_g, "module") else net_g,
                "architecture_id",
                None,
            )
            checkpoint_architecture = checkpoint.get("architecture_id")
            # An *absent* id is not a mismatch here, unlike the resume guard: a
            # pretrain comes from upstream, where the key never existed (every
            # stock RVC v2 checkpoint has no id, the bundled HiFi-GAN pretrains
            # included), while a resumed `G_*.pth` missing the key is one of
            # this fork's own old runs whose layout has since changed.  An
            # id-less pretrain is instead verified by the strict load below.
            if (
                expected_architecture
                and expected_architecture != "vits_gaussian_v1"
                and checkpoint_architecture is not None
                and checkpoint_architecture != expected_architecture
            ):
                raise ValueError(
                    f"Pretrained generator architecture mismatch: this run builds "
                    f"'{expected_architecture}' and "
                    f"'{os.path.basename(pretrainG)}' was written for "
                    f"'{checkpoint_architecture}'."
                )
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            # ``verify_spk_dim`` decided this pretrain's speaker table belongs
            # to other speakers, so ``emb_g`` doesn't even fit; substitute a
            # fresh tensor rather than dropping the key, since dropping it
            # would fail the strict load below.
            if reset_pretrained_embeddings and "emb_g.weight" in state_dict:
                state_dict = substitute_speaker_embeddings(state_dict, net_g)
                if rank == 0:
                    info(
                        "Pretrained speaker embeddings discarded: "
                        f"starting {state_dict['emb_g.weight'].shape[0]} fresh ones.",
                        tag="[INIT]",
                    )

            # A pretrain predating this key is a stock upstream sine, so an
            # absent key is read as "sine" rather than waved through --
            # loading an RVC v2 pretrain into a comb or bank run is exactly
            # the mismatch worth naming.
            assert_excitation_matches(
                net_g.module if hasattr(net_g, "module") else net_g,
                checkpoint,
                origin="pretrain",
            )
            net_g.load_state_dict(
                state_dict,
                strict=True,
            )

        # Loading the pretrained Discriminator model
        if pretrainD not in ["", "None"]:
            if rank == 0:
                info(f"Loading pretrained (D) '{pretrainD}'", tag="[INIT]")
            checkpoint = torch.load(pretrainD, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            # The periods are the one thing a strict load cannot check -- they
            # are in the ``view``, not in any weight -- and every pretrained D
            # in circulation was trained on the un-scaled set.  Loading one
            # into a rate-scaled discriminator succeeds and starts every branch
            # folding at a frequency its weights never saw.
            assert_periods_match(
                net_d.module if hasattr(net_d, "module") else net_d,
                checkpoint,
                origin="pretrained discriminator",
            )
            net_d.load_state_dict(state_dict, strict=True)

        # Load the models and optionally wrap with DDP
        net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

        # Apply decoder / vocoder layer freezes ( for fine-tuning )
        apply_training_freezes(net_g, rank)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    # Built after both branches so the shadow starts from the weights the run
    # is actually beginning with (resumed, pretrained, or fresh).
    ema = None
    # Schedule-free already averages the trajectory (its x iterate); an EMA
    # here would be built from the extrapolated y iterate instead, and
    # stacking a second average lengthens the effective horizon, which makes
    # the overtrain detector worse. So the two are mutually exclusive.
    if use_ema and is_schedule_free(optimizer_choice_g):
        if rank == 0:
            warning(
                "Weight EMA is off: Sched-Free AdamW already averages the "
                "trajectory, and its averaged weights are what gets evaluated "
                "and exported.",
                tag="[EMA]",
            )
    elif use_ema:
        ema = WeightEMA(
            net_g,
            decay=float(getattr(config.train, "ema_decay", WeightEMA.DEFAULT_DECAY)),
        )
        restored = False
        if resumed_g_path:
            blob = torch.load(resumed_g_path, map_location="cpu", weights_only=True)
            restored = ema.load_state_dict(
                blob.get("ema"), net_g.module if hasattr(net_g, "module") else net_g
            )
        if rank == 0:
            info(
                f"Decay {ema.decay}"
                + (
                    f", resumed at {ema.updates} updates."
                    if restored
                    else ", starting from the current weights."
                ),
                tag="[EMA]",
            )

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, ema, resumed_extra_d


def prepare_schedulers(
    optim_g, optim_d,
    use_lr_scheduler, lr_scheduler, exp_decay_gamma,
    total_epoch_count, epoch_str, global_step, train_loader,
    fresh_start=False,
):
    def _horizon_decay(final_ratio, total_units):
        """Exponential decay reaching ``final_ratio`` at the end of the run,
        so the same config gives the same endpoint at any run length. Progress
        is clamped at 1.0 so a run extended past its horizon holds the final
        LR instead of decaying straight through it.
        """
        ratio = min(1.0, max(1e-6, float(final_ratio)))
        total = max(1, int(total_units))

        def scale(unit):
            return ratio ** min(1.0, max(0, unit) / total)

        return scale

    def _horizon_cosine(final_ratio, total_units):
        """Cosine anneal from the starting LR to ``final_ratio`` of it.

        Unlike stock ``CosineAnnealingLR``'s shared absolute ``eta_min``,
        the endpoint here is a fraction of each group's own base LR -- G and D
        start at different rates, and a shared floor would drive their ratio
        to 1.0 by the end of the run, which is what keeps the discriminator
        alive. Clamped past the horizon like ``_horizon_decay``.
        """
        ratio = min(1.0, max(1e-6, float(final_ratio)))
        total = max(1, int(total_units))

        def scale(unit):
            progress = min(1.0, max(0, unit) / total)
            return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return scale

    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    scheduler_resume_epoch = -1 if fresh_start else epoch_str - 1
    scheduler_resume_step = -1 if fresh_start else global_step - 1

    for param_group in optim_g.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    for param_group in optim_d.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler and (
        is_schedule_free(optimizer_choice_g) or is_schedule_free(optimizer_choice_d)
    ):
        # Not an error -- the optimizer survives it, because its averaging
        # weights key off ``lr_max`` rather than the current lr -- but a decay
        # schedule is the thing schedule-free exists to remove, so a run using
        # both is almost certainly a leftover setting.
        warning(
            f"'{lr_scheduler}' is set together with a schedule-free optimizer, "
            "which is designed to run without one. The schedule will still be "
            "applied; set the scheduler to 'none' if that was not intended.",
            tag="[INIT]",
        )

    if use_lr_scheduler:
        scheduler_name = (
            "cosine annealing epoch"
            if lr_scheduler == "cosine annealing"
            else lr_scheduler
        )

        horizon_shapes = {
            "exp decay epoch": _horizon_decay,
            "exp decay step": _horizon_decay,
            "cosine annealing epoch": _horizon_cosine,
        }
        if lr_final_ratio is not None and scheduler_name in horizon_shapes:
            # Only one variant is stepped per optimizer step; the others are
            # stepped per epoch, so the ratio has to land at the end of the run
            # in whichever unit this scheduler counts.
            per_epoch = scheduler_name != "exp decay step"
            total_units = total_epoch_count * (1 if per_epoch else num_batches_per_epoch)
            shape = horizon_shapes[scheduler_name](lr_final_ratio, total_units)
            resume_at = scheduler_resume_epoch if per_epoch else scheduler_resume_step
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(
                optim_g, shape, last_epoch=resume_at
            )
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(
                optim_d, shape, last_epoch=resume_at
            )
        elif scheduler_name == "exp decay epoch":
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch
            )
        elif scheduler_name == "exp decay step":
            scheduler_gamma = (
                exp_decay_gamma
                if exp_decay_step_raw
                else exp_decay_gamma ** (1.0 / num_batches_per_epoch)
            )
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=scheduler_gamma, last_epoch=scheduler_resume_step
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=scheduler_gamma, last_epoch=scheduler_resume_step
            )
        elif scheduler_name == "cosine annealing epoch":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_g, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch
            )
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_d, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch
            )

    return scheduler_g, scheduler_d


def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    if use_custom_ref:
        info("Using custom reference input from 'logs/reference/'.", tag="[REFERENCE]")
        reference_audio = None
        reference_source = reference_path

        phone = torch.FloatTensor(np.repeat(np.load(os.path.join(reference_path, "ref_feats.npy")), 2, axis=0)).unsqueeze(0).to(device)
        pitch = torch.LongTensor(np.load(os.path.join(reference_path, "ref_f0c.npy"))).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(np.load(os.path.join(reference_path, "ref_f0f.npy"))).unsqueeze(0).to(device)

        # Measure lengths
        lengths = [phone.shape[1], pitch.shape[1], pitchf.shape[1]]
        min_len = min(lengths)

        # Trim to min length
        phone = phone[:, :min_len, :]
        pitch = pitch[:, :min_len]
        pitchf = pitchf[:, :min_len]
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)
        sid = torch.LongTensor([0]).to(device)

        # Optional ground truth for the preview; without it the preview
        # degrades to the generated waveform alone. Resampled on load, so it
        # can be at any rate -- one f0 frame is one hop at every configured
        # sample rate, both being 10 ms.
        audio_path = os.path.join(reference_path, "ref_audio.wav")
        if os.path.isfile(audio_path):
            from rvc.lib.utils import load_audio

            wave = load_audio(audio_path, config.data.sample_rate)
            wanted = min_len * config.data.hop_length
            if wave.shape[0] < wanted:
                # Short is survivable -- the figure crops both mels to the
                # frames they share -- but silently comparing less than the
                # reference renders is not, so it is said out loud.
                warning(
                    "ref_audio.wav is "
                    f"{wave.shape[0] / config.data.sample_rate:.2f}s, short of the "
                    f"{wanted / config.data.sample_rate:.2f}s the features render; "
                    "the preview will compare only the overlap.",
                    tag="[REFERENCE]",
                )
            reference_audio = (
                torch.FloatTensor(wave[:wanted]).view(1, 1, -1).to(device)
            )
        else:
            warning(
                "No ref_audio.wav; the preview will show the "
                "generated audio without the mel comparison.",
                tag="[REFERENCE]",
            )

    else:
        info("No custom reference found; fetching from train_loader.", tag="[REFERENCE]")
        batch = next(iter(train_loader))
        # Unpack everything from the loader
        phone, phone_lengths, pitch, pitchf, _, _, reference_audio, _, sid = batch

        # Move only the first sample of the batch to device
        phone = phone[0:1].to(device)
        phone_lengths = phone_lengths[0:1].to(device)
        pitch = pitch[0:1].to(device)
        pitchf = pitchf[0:1].to(device)
        reference_audio = reference_audio[0:1].to(device)
        sid = sid[0:1].to(device)

        batch_indices = []
        for batch in train_loader.batch_sampler:
            batch_indices = batch
            break

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            file_paths = train_loader.dataset.dataset.get_file_paths(batch_indices)
        else:
            file_paths = train_loader.dataset.get_file_paths(batch_indices)

        file_name = os.path.basename(file_paths[0])
        info(f"Origin of the ref: {file_name}", tag="[REFERENCE]")
        reference_source = file_name

    return (
        (phone, phone_lengths, pitch, pitchf, sid, config.train.seed),
        reference_audio,
        reference_source,
    )





def main():
    """
    Main function to start the training process.
    """
    global gpus

    wavs = [wav for wav in glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*")) if wav.endswith((".wav", ".flac"))]
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print_error(
                f"Pretrained model sample rate ({sample_rate} Hz) does not "
                f"match the dataset audio ({sr} Hz).",
                tag="[INIT]",
            )
            os._exit(1)
    else:
        warning("No sliced wav files found.", tag="[INIT]")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        warning(
            "No GPU detected, falling back to the CPU. This will take a very "
            "long time.",
            tag="[INIT]",
        )

    if n_gpus > 1:
        # Use an explicit IPv4 loopback address.  On Windows, ``localhost``
        # may resolve to the machine hostname/IPv6 and collide with a stale
        # worker. Ask the OS for a currently free rendezvous port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as master_socket:
            master_socket.bind(("127.0.0.1", 0))
            master_port = master_socket.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []

        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch_count,
                    epoch_save_frequency,
                    save_weight_models,
                    save_only_latest_net_models,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()
            pid_data["process_pids"].append(subproc.pid)

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        old_session_cleanup(now_dir, model_name)
    start()

def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    config,
    device,
    device_id,
):
    global global_step, warmup_completed, optimizer_choice_g, optimizer_choice_d
    global from_scratch, swap_start_step, swap_completed
    global phase_start_step, phase_step, phase_limit_reached

    if rank == 0:
        configure_logging(tag="[TRAIN]")

    install_stop_handlers()

    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Initial print / session info for console
    print_init_setup(
        warmup_duration,
        rank,
        use_warmup,
        config,
        optimizer_choice_g,
        optimizer_choice_d,
        lr_scheduler,
        exp_decay_gamma,
        spectral_loss,
        lr_final_ratio,
        # Passed from the same variable that drives every ``autocast`` call, so
        # the banner cannot disagree with what the loop actually runs.
        amp_dtype=amp_dtype if use_amp else None,
    )

    # Initial setup
    setup_env_and_distr(rank, n_gpus, device, device_id, config)

    # Dataloading and loaders preparation
    train_loader, holdout_set, probe_set = prepare_dataloaders(
        config, n_gpus, rank, batch_size
    )

    # Spk dim verif
    speaker_layout = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = speaker_layout.embed_dim

    if rank == 0 and warmup_active():
        warmup_tag = "Manual control" if warmup_steps > 0 else f"per epoch x{warmup_duration}"
        info(f"Linear warmup: {effective_warmup_steps(train_loader)} steps ({warmup_tag}).", tag="[INIT]")

    # Spectral loss init
    fn_spectral_loss2 = None
    fn_spectral_loss_ms = None

    # RefineGAN's compressed-mel distance.  A plain L1 has a gradient of
    # constant magnitude no matter how close the reconstruction gets, so the
    # generator settles into a noise-limited equilibrium where the loss creeps
    # down but the gradient norm never anneals.  Huber is L2 below ``beta``
    # (self-annealing as the residual shrinks) and L1 above it (robust to the
    # onset/silence bins that a plain MSE would over-weight).
    mel_distance = str(
        getattr(config.train, "mel_distance", "huber")
    ).lower()
    mel_huber_beta = float(getattr(config.train, "mel_huber_beta", 0.3))
    # Frequency weighting for that distance.  A mean over bins gives every bin
    # the same vote, and past the Huber knee the vote stops scaling with the
    # error, so a badly wrong minority of bins stays wrong: measured at 46% of
    # the mel error against 32% of the gradient below 1 kHz.  1.0 is off, which
    # is what a config that predates this key gets.
    mel_low_emphasis = float(
        getattr(config.train, "mel_low_emphasis", 1.0)
    )
    mel_low_emphasis_hz = float(
        getattr(config.train, "mel_low_emphasis_hz", 1000.0)
    )

    def _make_mel_distance():
        # ``mel_distance``/``mel_low_emphasis`` were only ever wired up for the
        # ChouwaGAN stack; every other vocoder always took this branch
        # regardless of the configured value.  Preserved exactly rather than
        # generalised, since making Huber/MSE/weighting actually apply here
        # would be a behaviour change to what RefineGAN/HiFi-GAN train on.
        base, weighted = torch.nn.L1Loss, False
        if not weighted or mel_low_emphasis == 1.0:
            return base()

        def _weights(num_mels: int):
            return mel_low_frequency_weights(
                num_mels=num_mels,
                sample_rate=config.data.sample_rate,
                mel_fmin=config.data.mel_fmin,
                mel_fmax=config.data.mel_fmax,
                emphasis=mel_low_emphasis,
                cutoff_hz=mel_low_emphasis_hz,
            )

        # The factory is what makes this usable by the multi-scale mel loss,
        # which evaluates the same distance at 5/10/20/40/80/160/320 bands.
        # Without it that mode raised ``Band weights cover 128 mel bins but the
        # loss was handed 5`` on its first step, so selecting "Multi-Scale Mel
        # Loss" in the UI could not run at all with a low-emphasis config.  The
        # weighting is defined by frequency, not by bin count, so rebuilding it
        # per resolution is the same statement about which bands matter.
        return BandWeightedSpectralLoss(
            base(reduction="none"),
            _weights(config.data.n_mel_channels).to(device),
            weight_factory=_weights,
        ).to(device)

    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = _make_mel_distance()
        if swap_l1_to_ms:
            fn_spectral_loss_ms = MultiScaleMelSpectrogramLoss(
                sample_rate=sample_rate,
                safe_log=False,
                loss_fn=_make_mel_distance(),
            )
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=sample_rate,
            safe_log=False,
            loss_fn=_make_mel_distance(),
        )
    elif spectral_loss == "Hybrid L1":
        fn_spectral_loss = _make_mel_distance()
        fn_spectral_loss2 = MultiScaleSTFTLoss()
    else:
        print_error(f"Unknown spectral loss {spectral_loss!r}. Exiting.", tag="[INIT]")
        sys.exit(1)


    # Loading of models and optims
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, ema, resumed_extra_d = load_models_and_optimizers(
        config,
        pretrainG,
        pretrainD,
        vocoder,
        use_checkpointing,
        sample_rate,
        optimizer_choice_g,
        optimizer_choice_d,
        custom_lr_g,
        custom_lr_d,
        use_custom_lr, 
        total_epoch_count,
        train_loader,
        device,
        device_id,
        n_gpus,
        rank,
        speaker_layout.reset_pretrained,
    )

    enable_vocoder_compile(net_g, device, rank)
    enable_discriminator_compile(net_d, config, device, rank)

    # GradScaler for FP16 AMP.  The init_scale is set high to avoid an early
    # underflow that zeros a whole batch of gradients, but not so high that
    # FP16-range overflows on the first step.  The ceiling is well above any
    # healthy operating range; the scaler's own backoff handles overflows, so
    # this is a guard, not a per-step rescaler.
    grad_scaler = (
        torch.amp.GradScaler("cuda", init_scale=2.0 ** 10, growth_interval=2000)
        if use_amp
        else None
    )
    # The scaler carries real state: the current scale and how far it is into the
    # growth interval.  Restarting it at ``init_scale`` on every resume replays
    # the initial overflow-and-back-off search, which throws away a handful of
    # steps each time -- and hides a run that had settled at a much lower scale.
    global amp_skipped_steps
    amp_skipped_steps = int(resumed_extra_d.get("amp_skipped_steps") or 0)
    if grad_scaler is not None:
        scaler_state = resumed_extra_d.get("grad_scaler")
        if scaler_state:
            grad_scaler.load_state_dict(scaler_state)
            if rank == 0:
                info(
                    f"Restored the GradScaler at scale {grad_scaler.get_scale():.0f} "
                    f"({amp_skipped_steps} steps skipped so far).",
                    tag="[RESUME]",
                )
    phase_start_step = global_step
    phase_step = 0
    phase_limit_reached = False
    if finetune_phase:
        epoch_str = 1

    if rank == 0:
        print_model_summary(
            [("Generator", net_g), ("Discriminator", net_d)],
            title=f"RVC {vocoder} {sample_rate} Hz",
        )

    if warmup_active() and global_step >= effective_warmup_steps(train_loader):
        warmup_completed = True

    # Loss swap: kick in from the moment this run resumes
    if swap_l1_to_ms:
        swap_start_step = global_step
        swap_completed = False
        info(f"Loss swap: L1 mel -> multi-scale mel over {swap_duration_steps} steps (starting at step {swap_start_step}).", tag="[TRAIN]")

    # Tensorboard handling
    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400,
            purge_step=global_step + 1
        )
        block_tensorboard_flush_on_exit(writer_eval)

        if global_step != 0:
            info(f"TensorBoard writer initialized; purging logs after step {global_step}.", tag="[INIT]")
        else:
            info("TensorBoard writer initialized.", tag="[INIT]")

    # from-scratch checker ( disables average loss )
    if finetune_phase:
        from_scratch = False
        if rank == 0:
            info(
                f"Fine-tune phase active: max steps={max_steps or 'epoch limit'}, "
                f"starting global step={global_step}.",
                tag="[INIT]",
            )
    elif (pretrainG in ["", "None"] or pretrainD in ["", "None"]) or force_from_scratch:
        from_scratch = True
        if rank == 0:
            warning("No pretrains used: average loss disabled.", tag="[INIT]")
    else:
        from_scratch = False

    # Prepare the schedulers
    scheduler_g, scheduler_d = prepare_schedulers(
        optim_g,
        optim_d,
        use_lr_scheduler,
        lr_scheduler,
        exp_decay_gamma,
        total_epoch_count,
        epoch_str,
        global_step,
        train_loader,
        fresh_start=reset_optimizer_for_run,
    )

    # Reference sample for live-infer
    reference, reference_audio, reference_source = get_reference_sample(
        train_loader,
        device,
        config,
    )

    # Cache for training with " cache " enabled
    cache = []

    # Both of these live across epochs.  ``training_loop`` is called once per
    # epoch, so anything built inside it is silently reconstructed every epoch:
    # the governor kept there lost its ramp progress, its best-headroom
    # backstop and both EMAs at every boundary, which pinned the adversarial
    # ceiling at its starting value forever and left the trend check blind for
    # the first two thousand steps of every epoch.
    #
    # Pretraining ramps on global_step so resuming picks the ramp up where it
    # left off.  Fine-tuning ramps on phase_step -- steps since *this* run
    # started -- because global_step is inherited from the source checkpoint,
    # which would put a fresh fine-tune past the end of its own ramp before the
    # first batch.
    # Read straight from the config rather than from ``training_loop``'s
    # locals, which are out of scope here and are the reason this ended up
    # inside the per-epoch function in the first place.
    #
    # Both the delay and the ramp are fitted to the run: a ramp that outlives
    # its own run leaves the ceiling pinned somewhere below the configured
    # maximum for every step that run ever takes, which reads as "the ceiling
    # is too low" when the real problem is that it was still climbing.  The
    # shares differ because the two mean different things -- the delay is dead
    # time and is worth at most a fifth of a run, while the ramp is the
    # transition itself and can have half of one.
    planned_steps = planned_step_count(total_epoch_count, train_loader, max_steps)

    # The turn from "still learning" to "memorising" happens on the scale of a
    # run, not an epoch.
    overtrain_monitor = None
    if holdout_set is not None and len(holdout_set):
        overtrain_monitor = _OvertrainMonitor(
            patience=int(getattr(config.train, "overtrain_patience", 8)),
            min_delta=float(getattr(config.train, "overtrain_min_delta", 0.001)),
            smoothing=int(getattr(config.train, "overtrain_smoothing", 3)),
            noise_window=int(getattr(config.train, "overtrain_noise_window", 6)),
        )
        interval = int(getattr(config.train, "holdout_interval", 0))
        if interval <= 0:
            interval = max(200, min(2000, len(train_loader)))
        # An explicit ``holdout_interval`` is fitted too.  It is a statement
        # about how often to score, not about how long the run is, and the
        # value that makes the detector inert is just as inert when it was
        # typed in as when it was derived.
        fitted = fit_eval_interval(interval, planned_steps, overtrain_monitor.patience)
        evaluations = planned_steps // fitted if planned_steps else 0
        if fitted != interval:
            info(
                f"Interval {interval} -> {fitted} steps: at {interval} "
                f"this {planned_steps}-step run would fit "
                f"{planned_steps // interval} evaluations against a patience of "
                f"{overtrain_monitor.patience}, and the detector could never fire.",
                tag="[HOLDOUT]",
            )
        interval = fitted
        info(
            f"Evaluating every {interval} steps, "
            f"patience {overtrain_monitor.patience} evaluations"
            + (f" ({evaluations} evaluations planned)." if evaluations else "."),
            tag="[HOLDOUT]",
        )
    else:
        interval = 0

    for epoch in range(epoch_str, total_epoch_count + 1):
        training_loop(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            [writer_eval],
            cache,
            total_epoch_count,
            epoch_save_frequency,
            save_weight_models,
            save_only_latest_net_models,
            device,
            device_id,
            reference,
            reference_audio,
            reference_source,
            fn_spectral_loss,
            n_gpus,
            fn_spectral_loss2,
            fn_spectral_loss_ms,
            holdout_set=holdout_set,
            probe_set=probe_set,
            overtrain_monitor=overtrain_monitor,
            holdout_interval=interval,
            ema=ema,
            grad_scaler=grad_scaler,
        )
        if use_lr_scheduler and (not warmup_active() or warmup_completed):
            if lr_scheduler in ["exp decay epoch", "cosine annealing", "cosine annealing epoch"]:
                scheduler_g.step()
                scheduler_d.step()


def warmup_active():
    """True if the linear warmup should run: CLI flag or manual step control."""
    return use_warmup or warmup_steps > 0


def effective_warmup_steps(train_loader):
    """Step count for the linear warmup ramp: manual `warmup_steps` if set, else warmup_duration epochs in steps."""
    return warmup_steps if warmup_steps > 0 else warmup_duration * len(train_loader)


def apply_linear_warmup(optim_g, optim_d, global_step, warmup_steps, rank):
    """
    Per-step linear LR warmup:
    ramps every param group's LR from ~0 up to its base ( initial_lr ) over `warmup_steps` steps.
    """
    global warmup_completed

    if warmup_completed:
        return

    if warmup_steps <= 0:
        warmup_completed = True
        return

    factor = min(global_step / warmup_steps, 1.0)
    for optim in (optim_g, optim_d):
        for param_group in optim.param_groups:
            param_group["lr"] = param_group["initial_lr"] * factor

    if factor >= 1.0:
        warmup_completed = True
        if rank == 0:
            info(
                f"Warmup completed at step {global_step} "
                f"(lr G {optim_g.param_groups[0]['lr']:.2e}, "
                f"lr D {optim_d.param_groups[0]['lr']:.2e}).",
                tag="[TRAIN]",
            )
        return


def training_loop(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    train_loader,
    writers,
    cache,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    device,
    device_id,
    reference,
    reference_audio,
    reference_source,
    fn_spectral_loss,
    n_gpus,
    fn_spectral_loss2=None,
    fn_spectral_loss_ms=None,
    holdout_set=None,
    probe_set=None,
    overtrain_monitor=None,
    holdout_interval=0,
    ema=None,
    grad_scaler=None,
):
    """Trains and evaluates the model for one epoch."""
    global global_step, warmup_completed, use_lr_scheduler, lr_scheduler, use_warmup, swap_completed
    global phase_step, phase_limit_reached, overtrain_flagged, overtrain_exported
    global amp_skipped_steps

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers if schedulers is not None else (None, None)

    train_loader = train_loader if train_loader is not None else None
    train_loader.batch_sampler.set_epoch(epoch)

    if writers is not None:
        writer = writers[0]

    live_sd_g = None

    net_g.train()
    net_d.train()

    if is_schedule_free(optimizer_choice_g):
        optim_g.train()
    if is_schedule_free(optimizer_choice_d):
        optim_d.train()

    # Partial resume aligning
    current_epoch_start_step = (epoch - 1) * len(train_loader)
    start_batch_idx = (
        0
        if finetune_phase
        else global_step - current_epoch_start_step
    )
    start_batch_idx = max(0, start_batch_idx)

    if start_batch_idx > 0:
        train_loader.batch_sampler.start_index = start_batch_idx

    remaining_batches = len(train_loader) - start_batch_idx
    data_iterator = islice(enumerate(train_loader), remaining_batches)

    epoch_recorder = EpochRecorder()

    if not from_scratch:
        # Tensors init for averaged losses:
        epoch_loss_tensor = torch.zeros(8, device=device)
        num_batches_in_epoch = 0

    avg_rolling_cache = {
        "grad_norm_d": deque(maxlen=rolling_loss_steps),
        "grad_norm_g": deque(maxlen=rolling_loss_steps),
        "loss_disc": deque(maxlen=rolling_loss_steps),
        "loss_disc_real": deque(maxlen=rolling_loss_steps),
        "loss_disc_fake": deque(maxlen=rolling_loss_steps),
        "loss_adv": deque(maxlen=rolling_loss_steps),
        "loss_gen_total": deque(maxlen=rolling_loss_steps),
        "loss_fm": deque(maxlen=rolling_loss_steps),
        "loss_spectral": deque(maxlen=rolling_loss_steps),
        "prior_kl_fast": deque(maxlen=rolling_loss_steps),
        "prior_std_fast": deque(maxlen=rolling_loss_steps),
        "posterior_std_fast": deque(maxlen=rolling_loss_steps),
        "scale_anchor": deque(maxlen=rolling_loss_steps),
        "prior_replacement": deque(maxlen=rolling_loss_steps),
        "prior_replacement_mean": deque(maxlen=rolling_loss_steps),
        "content_rms": deque(maxlen=rolling_loss_steps),
        "posterior_detail_rms": deque(maxlen=rolling_loss_steps),
        "prior_detail_rms": deque(maxlen=rolling_loss_steps),
    }
    avg_rolling_cache["loss_kl"] = deque(maxlen=rolling_loss_steps)
    kl_std_cache = deque(maxlen=rolling_loss_steps)
    kl_mean_cache = deque(maxlen=rolling_loss_steps)
    kl_active_cache = deque(maxlen=rolling_loss_steps)
    last_kl_per_dim = None

    # One 0/1 per step, so the logged rate is "how much of the recent window did
    # FP16 throw away" rather than a lifetime average that a bad first epoch
    # would dominate forever.
    amp_skip_cache = deque(maxlen=rolling_loss_steps)

    diagnostics_interval = max(
        1,
        int(getattr(config.train, "diagnostics_interval", 256)),
    )

    # SAN's direction term, weighted well below the function term: it trains
    # only the unit-norm projection, and at 1.0 it doubles ``loss_disc`` for a
    # quantity the generator never sees.  Clamped to [0, 1] because a negative
    # weight would reward a direction that separates real from fake backwards.
    san_direction_weight = max(
        0.0,
        min(1.0, float(getattr(config.train, "san_direction_weight", 0.25))),
    )
    san_active = bool(
        getattr(
            net_d.module if hasattr(net_d, "module") else net_d, "supports_san", False
        )
    )
    kl_active_threshold = max(
        0.0,
        float(getattr(config.train, "kl_active_threshold", 0.01)),
    )

    # Only read by the "Hybrid L1" spectral loss.  Left at 1.0 so the option
    # behaves exactly as before unless it is raised deliberately.
    ms_stft_weight = max(
        0.0,
        float(getattr(config.train, "ms_stft_weight", 1.0)),
    )

    with progress_task(
        len(train_loader),
        f"Epoch {epoch}/{total_epoch_count}",
        initial=start_batch_idx,
        training=True,
        disable=rank != 0,
    ) as (progress, task_id):
        progress_metrics = ""
        metrics_update_interval = max(1, min(rolling_loss_steps, 8))
        for batch_idx, batch in data_iterator:

            global_step += 1
            phase_step = max(0, global_step - phase_start_step)
            if not from_scratch:
                num_batches_in_epoch += 1

            # Linear warmup: per-step ramp ( manual `warmup_steps` or `warmup_duration` epochs )
            if warmup_active():
                apply_linear_warmup(optim_g, optim_d, global_step, effective_warmup_steps(train_loader), rank)

            # Gradient clipping.  The step-scheduled variant (clip hard early,
            # loosen later) was removed: with RefineGAN the global norm below
            # already bounds both nets, and it never bound anything in practice
            # -- ``grad_clip_hit_rate_g`` sat at 0 with grad norms 4x under the
            # cap.  What remains is the manual override and that global bound.
            if clip_grad_norm_override:
                grad_clip_value_g = clip_grad_norm_override_value_g
                grad_clip_value_d = clip_grad_norm_override_value_d
            else:
                grad_clip_value_g = grad_clip_value_d = float("inf")

            # Device handling
            if device.type == "cuda":
                batch = [tensor.cuda(device_id, non_blocking=True) for tensor in batch]
            elif device.type != "cuda":
                batch = [tensor.to(device) for tensor in batch]

            # Batch unpacking
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = batch

            model_g = net_g.module if hasattr(net_g, "module") else net_g

            # Generator main forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                model_output = net_g(spec, spec_lengths, sid, phone, phone_lengths, pitchf, pitch)

                y_hat, ids_slice, x_mask, z_mask, vae_parts = model_output

                # Gaussian latent samples and parameters used by the VITS ELBO.
                z, z_p, m_p, logs_p, m_q, logs_q = vae_parts

                # Slice the original waveform ( y ) to match the generated slice:
                y = commons.slice_segments(y, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)



            # Discriminator update
            _loss_disc_acc, _loss_disc_real_acc, _loss_disc_fake_acc, _grad_norm_d_acc = [], [], [], []

            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                disc_loss_parts = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                    san_direction_weight=san_direction_weight,
                    normalize=False,
                    per_branch=False,
                )
                loss_disc, loss_disc_real, loss_disc_fake = disc_loss_parts[:3]

            optim_d.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss_disc).backward()
                grad_scaler.unscale_(optim_d)
            else:
                loss_disc.backward()
            grad_norm_d = _clip_or_sample_grad_norm(
                net_d.parameters(),
                grad_clip_value_d,
                global_step,
                metrics_update_interval,
            )
            if grad_scaler is not None:
                grad_scaler.step(optim_d)
            else:
                optim_d.step()

            # Temp accumulation
            _loss_disc_acc.append(loss_disc.detach())
            _loss_disc_real_acc.append(loss_disc_real.detach())
            _loss_disc_fake_acc.append(loss_disc_fake.detach())
            if grad_norm_d is not None:
                _grad_norm_d_acc.append(grad_norm_d)

            # Stack + mean
            loss_disc = torch.stack(_loss_disc_acc).mean()
            loss_disc_real = torch.stack(_loss_disc_real_acc).mean()
            loss_disc_fake = torch.stack(_loss_disc_fake_acc).mean()
            grad_norm_d = (
                torch.stack(_grad_norm_d_acc).mean()
                if _grad_norm_d_acc
                else None
            )
            optim_d.zero_grad(set_to_none=True)

            # Run discriminator on generated output
            discriminator_model = net_d.module if hasattr(net_d, "module") else net_d
            # The generator update reads the discriminator; it never trains it.
            # Its backward still computes a weight gradient for every branch,
            # and that gradient is thrown away -- ``optim_d.zero_grad`` above
            # and the next iteration's discriminator update bracket it, and no
            # ``optim_d.step`` runs in between.  Freezing the parameters skips
            # the weight-gradient half of the discriminator backward while the
            # gradient that *is* wanted, the one flowing back into ``y_hat``,
            # is unchanged.  Paired with ``no_grad_real`` below, measured on an
            # RTX 5060 at batch 8 over 0.4 s (v3 + UnivHD, both passes, forward
            # and backward): 320 ms/step becomes 240, at unchanged peak VRAM.
            discriminator_parameter_states = [
                parameter.requires_grad
                for parameter in discriminator_model.parameters()
            ]
            for parameter in discriminator_model.parameters():
                parameter.requires_grad_(False)
            try:
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    # ``no_grad_real``: the real side is the feature matching
                    # *target* and its logits are discarded, so differentiating
                    # it builds a graph nothing consumes.  See
                    # ``MPD_MSD_Combined.forward``.
                    # The unwrapped module, not ``net_d``: under DDP the
                    # wrapper's forward arms the reducer for a backward that
                    # will never produce a discriminator gradient, and no
                    # gradient sync is wanted here in the first place.
                    _, y_d_hat_g, fmap_r, fmap_g = discriminator_model(
                        y, y_hat, no_grad_real=True
                    )


                # Compute generator losses:
                prior_gap_delta = None
                prior_gap_error = None
                if (
                    rank == 0
                    and m_p is not None
                    and diagnostics_interval > 0
                    and global_step % diagnostics_interval == 0
                    # The flow runs at the prior's frame rate; a batch where the
                    # two rates disagree is not comparable and is skipped rather
                    # than silently trimmed.
                    and m_p.shape[-1] == z.shape[-1]
                ):
                    prior_gap_delta, prior_gap_error = _prior_gap(
                        model_g,
                        m_p,
                        x_mask,
                        ids_slice,
                        config.train.segment_size // config.data.hop_length,
                        pitchf,
                        sid,
                        y,
                        y_hat,
                        config,
                    )
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):

                    # Spectral loss.  The component terms are logged separately
                    # where a mode has more than one, because the combined series
                    # cannot show which half is actually moving -- and for
                    # "Hybrid L1" the balance between them is the thing to tune.
                    loss_spectral_parts: dict[str, torch.Tensor] = {}
                    if spectral_loss == "L1 Mel Loss":
                        y_mel = wave_to_mel(
                            config, y, num_mels=None,
                            for_loss=False,
                        )
                        y_hat_mel = wave_to_mel(
                            config, y_hat, num_mels=None,
                            for_loss=False,
                        )
                        if swap_l1_to_ms and fn_spectral_loss_ms is not None:
                            # Loss swap: L1 mel fades out, Multi-Scale mel fades in over swap_duration_steps
                            swap_progress = min(1.0, max(0.0, (global_step - swap_start_step) / max(1, swap_duration_steps)))
                            swap_alpha = 0.5 * (1.0 - math.cos(math.pi * swap_progress))  # smooth 0->1 ramp
                            loss_l1_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                            loss_ms_mel = fn_spectral_loss_ms(y, y_hat) * config.train.c_mel / 3.0
                            loss_spectral = (1.0 - swap_alpha) * loss_l1_mel + swap_alpha * loss_ms_mel
                            loss_spectral_parts = {
                                "loss_spectral_l1_mel": loss_l1_mel,
                                "loss_spectral_ms_mel": loss_ms_mel,
                            }
                            if swap_progress >= 1.0 and not swap_completed:
                                swap_completed = True
                                success(f"Loss swap complete at step {global_step}; now using multi-scale mel loss.", tag="[TRAIN]")
                        else:
                            loss_spectral = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                    elif spectral_loss == "Multi-Scale Mel Loss":
                        loss_spectral = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0 # * 15
                    elif spectral_loss == "Hybrid L1":
                        # L1 Mel
                        y_mel = wave_to_mel(
                            config, y, num_mels=None,
                            for_loss=False,
                        )
                        y_hat_mel = wave_to_mel(
                            config, y_hat, num_mels=None,
                            for_loss=False,
                        )
                        loss_l1_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel # * 45
                        # MS-STFT.  Weighted, because the mel term it sits beside
                        # carries c_mel (45 by default): at a hardcoded 1.0 the
                        # multi-resolution term -- the only part of this loss that
                        # resolves high frequencies directly -- contributes almost
                        # nothing to the gradient it is there to provide.
                        loss_ms_stft = (
                            fn_spectral_loss2(y_hat.float(), y.float()) * ms_stft_weight
                        )
                        # Loss
                        loss_spectral = loss_l1_mel + loss_ms_stft
                        # Logged post-weight, which is what the sum above actually
                        # sees: the point of these two series is to show whether
                        # ``ms_stft_weight`` is high enough for the
                        # MS-STFT term to matter beside a mel term carrying c_mel.
                        loss_spectral_parts = {
                            "loss_spectral_l1_mel": loss_l1_mel,
                            "loss_spectral_ms_stft": loss_ms_stft,
                        }

                    loss_fm = feature_loss(fmap_r, fmap_g, normalize=False) * 2.0

                    # Generator loss.  ``y_d_hat_g`` comes from the *generator*
                    # update's forward, which never sets ``san_training``, so these
                    # are plain logits and ``san_direction_weight`` is inert -- the
                    # direction output is the discriminator's business alone.
                    loss_adv = generator_loss(
                        y_d_hat_g,
                        normalize=False,
                        san_direction_weight=san_direction_weight,
                        use_softplus=san_active,
                    )

                    loss_kl, raw_kl = kl_loss(
                        z_p,
                        logs_q,
                        m_p,
                        logs_p,
                        z_mask,
                        return_terms=True,
                    )
                    loss_kl = loss_kl * config.train.c_kl

                    # KL diagnostic: per-dimension raw divergence.  Two things
                    # this deliberately does not do.  It does not re-form
                    # ``raw_kl`` -- that is the tensor ``kl_loss`` just built,
                    # handed back detached.  And it does not call ``.item()``:
                    # the three it used to make sat between the generator's
                    # forward and its backward, so each one drained the queue
                    # and gave up the CPU's run-ahead on a step that is
                    # dispatch-bound, for three floats nothing reads until the
                    # next logging interval.  The caches hold device tensors
                    # and are reduced where the series are written.
                    with torch.no_grad():
                        raw_kl_per_dim = (raw_kl * z_mask).sum(dim=(0, 2)) / z_mask.sum(
                            dim=(0, 2)
                        ).clamp(min=1)
                        diagnostic_kl = raw_kl_per_dim.clamp_min(0.0)
                        # ``.float()`` so the deque holds one dtype whatever
                        # autocast handed this step, which is what lets the
                        # window be reduced with a single ``torch.stack``.
                        kl_std_cache.append(diagnostic_kl.std().float())
                        kl_mean_cache.append(diagnostic_kl.mean().float())
                        kl_active_cache.append(
                            (diagnostic_kl > kl_active_threshold).float().mean()
                        )
                        last_kl_per_dim = diagnostic_kl

                    loss_core = loss_spectral + loss_kl
                    loss_gan = loss_adv + loss_fm
                    loss_gen_total = loss_core + loss_gan
                    if rank == 0 and prior_gap_delta is not None:
                        writer.add_scalar(
                            "diag/prior_gap_mel_l1",
                            prior_gap_delta.item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "diag/prior_mel_l1",
                            prior_gap_error.item(),
                            global_step,
                        )

                # Generator backward and update:
                optim_g.zero_grad(set_to_none=True)
                module_grad_metrics = {}
                if grad_scaler is not None:
                    grad_scaler.scale(loss_gen_total).backward()
                    grad_scaler.unscale_(optim_g)
                else:
                    loss_gen_total.backward()
                if global_step % metrics_update_interval == 0:
                    module_grad_metrics = _generator_gradient_metrics(net_g)
                grad_norm_g = _clip_or_sample_grad_norm(
                    net_g.parameters(),
                    grad_clip_value_g,
                    global_step,
                    metrics_update_interval,
                )
                if grad_scaler is not None:
                    grad_scaler.step(optim_g)
                else:
                    optim_g.step()
            finally:
                # ``finally`` and not a plain restore after the step: a
                # skipped batch or a raise anywhere above would otherwise
                # leave every ``net_d`` parameter frozen for the rest of
                # the run, with ``loss_disc`` still logged and
                # ``optim_d.step`` still called -- a discriminator that
                # has silently stopped learning.
                for parameter, requires_grad in zip(
                    discriminator_model.parameters(),
                    discriminator_parameter_states,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)


            if grad_scaler is not None:
                # ``update`` is the only place that reports an overflow, and it
                # reports it by *lowering the scale*: there is no "was the step
                # taken" flag on the scaler.  Reading the scale either side of
                # the call is the sanctioned way to detect it.
                scale_before_update = grad_scaler.get_scale()
                grad_scaler.update()
                if grad_scaler.get_scale() < scale_before_update:
                    amp_skipped_steps += 1
                    amp_skip_cache.append(1.0)
                else:
                    amp_skip_cache.append(0.0)

            # After both optimizer branches, so it always sees post-step
            # weights, and before any of the preview/holdout weight swaps, so
            # it never averages in a set of weights that is not the live one.
            if ema is not None:
                ema.update(net_g)

            # Held-out evaluation.  This is the only number in the loop that can
            # go up while the model is still "improving" on its training data,
            # which is exactly what makes it the overtrain signal.
            if (
                overtrain_monitor is not None
                and holdout_set is not None
                and holdout_interval > 0
                and global_step > 0
                and global_step % (holdout_interval * overtrain_monitor.interval_scale)
                == 0
            ):
                # The latent diagnostic answers a question about the *shape* of
                # the model rather than about this step, and it costs a second
                # decode of every excerpt; the selection metric is what has to
                # be measured every time.
                evaluation_index = len(overtrain_monitor.history)
                latent_every = max(
                    1, int(getattr(config.train, "holdout_latent_interval", 4))
                )
                want_latent = evaluation_index % latent_every == 0
                noise_scale = float(
                    getattr(config.train, "holdout_noise_scale", 0.0)
                )
                probe_score = float("nan")

                # Everything is scored inside the same weight-swap block, so
                # the metric, the probe and the latent diagnostic all describe
                # one set of weights.  They used to disagree: the metric ran
                # under the EMA and the diagnostic under the live model, which
                # put two versions of the same number side by side in
                # TensorBoard.
                if ema is not None:
                    # With an EMA, both halves of the detector get better and
                    # for different reasons.  The curve it scores is the
                    # average rather than a single oscillating step, so the
                    # minimum is far better localised and ``patience`` stops
                    # being mostly a noise margin.  And the weights it keeps
                    # are the average, which for a GAN vocoder is normally
                    # better than any step it was made from -- so the thing
                    # being measured is also the thing worth keeping.
                    with ema.applied(net_g) as averaged:
                        metrics = _holdout_metrics_resilient(
                            averaged,
                            holdout_set,
                            config,
                            device,
                            want_latent=want_latent,
                            noise_scale=noise_scale,
                        )
                        if probe_set is not None:
                            probe_score = _holdout_metrics_resilient(
                                averaged,
                                probe_set,
                                config,
                                device,
                                want_latent=False,
                                noise_scale=noise_scale,
                            )["mel_l1"]
                    improved = overtrain_monitor.update(
                        ema, metrics["mel_l1"], global_step
                    )
                else:
                    # Under a schedule-free optimizer the live weights are the
                    # extrapolated iterate, which is not the point the method
                    # converges.  Both calls sit inside the block: the monitor
                    # clones the weights it is handed, and one line later those
                    # would be the extrapolated iterate again -- which is the
                    # bug the "keep the weights that were actually scored"
                    # contract in ``update`` exists to prevent.
                    with averaged_weights((optimizer_choice_g, optim_g)):
                        metrics = _holdout_metrics_resilient(
                            net_g,
                            holdout_set,
                            config,
                            device,
                            want_latent=want_latent,
                            noise_scale=noise_scale,
                        )
                        if probe_set is not None:
                            probe_score = _holdout_metrics_resilient(
                                net_g,
                                probe_set,
                                config,
                                device,
                                want_latent=False,
                                noise_scale=noise_scale,
                            )["mel_l1"]
                        improved = overtrain_monitor.update(
                            net_g.module if hasattr(net_g, "module") else net_g,
                            metrics["mel_l1"],
                            global_step,
                        )

                holdout_loss = metrics["mel_l1"]
                if rank == 0 and math.isfinite(holdout_loss):
                    writer.add_scalar("holdout/mel_l1", holdout_loss, global_step)
                    writer.add_scalar("holdout/best", overtrain_monitor.best, global_step)
                    writer.add_scalar(
                        "holdout/smoothed", overtrain_monitor.smoothed, global_step
                    )
                    # What the run itself says a real change looks like.  Read
                    # every other movement on these curves against it.
                    writer.add_scalar(
                        "holdout/noise_sigma", overtrain_monitor.sigma, global_step
                    )
                    writer.add_scalar(
                        "holdout/evals_since_progress",
                        overtrain_monitor.since_progress,
                        global_step,
                    )
                    for name in ("latent_gap", "latent_posterior"):
                        if name in metrics and math.isfinite(metrics[name]):
                            writer.add_scalar(
                                f"holdout/{name}", metrics[name], global_step
                            )
                    # ``improved`` earns a success glyph rather than a marker
                    # character: the line is already dense, and the glyph column
                    # is where every other stage puts its outcome.
                    emit = success if improved else info
                    emit(
                        f"step {global_step}: {holdout_loss:.5f}  "
                        f"(smoothed {overtrain_monitor.smoothed:.5f}, best "
                        f"{overtrain_monitor.best:.5f} @ {overtrain_monitor.best_step}, "
                        f"noise {overtrain_monitor.sigma:.5f}, "
                        f"{overtrain_monitor.since_progress}/"
                        f"{overtrain_monitor.patience} since progress)",
                        tag="[HOLDOUT]",
                    )
                    if math.isfinite(probe_score):
                        # The gap, not the two numbers: both sides carry the
                        # same "still learning" trend and only their difference
                        # is generalisation.
                        writer.add_scalar(
                            "holdout/train_probe", probe_score, global_step
                        )
                        writer.add_scalar(
                            "holdout/generalization_gap",
                            holdout_loss - probe_score,
                            global_step,
                        )
                        info(
                            f"train probe {probe_score:.5f}, "
                            f"generalisation gap {holdout_loss - probe_score:+.5f}",
                            tag="[HOLDOUT]",
                        )
                    if "latent_gap" in metrics:
                        info(
                            f"latent gap {metrics['latent_gap']:.5f} "
                            f"(posterior {metrics['latent_posterior']:.5f}, "
                            f"prior {holdout_loss:.5f})",
                            tag="[HOLDOUT]",
                        )
                if overtrain_monitor.overtrained and not overtrain_flagged:
                    overtrain_flagged = True
                    if rank == 0:
                        warning(
                            f"Held-out loss has not improved for "
                            f"{overtrain_monitor.since_progress} evaluations. "
                            f"The last good weights are step "
                            f"{overtrain_monitor.best_step} ({overtrain_monitor.best:.5f}); "
                            f"they will be the ones exported.",
                            tag="[OVERTRAIN]",
                        )
                        if stop_on_overtrain:
                            warning("Stopping (stop_on_overtrain is on).", tag="[OVERTRAIN]")
                        else:
                            scale = overtrain_monitor.backoff()
                            info(
                                f"Continuing; scoring every "
                                f"{holdout_interval * scale} steps from here.",
                                tag="[OVERTRAIN]",
                            )


            # Per step exp lr decay for both optimizers.
            if use_lr_scheduler and (not warmup_active() or warmup_completed) and lr_scheduler == "exp decay step":
                # FP32/BF16 only: no scaler can retract a step, so the
                # schedulers always advance.
                scheduler_g.step()
                scheduler_d.step()



            if not from_scratch:
                # Loss accumulation for epoch-avg
                epoch_loss_tensor[0].add_(loss_disc.detach())
                epoch_loss_tensor[1].add_(loss_disc_real.detach())
                epoch_loss_tensor[2].add_(loss_disc_fake.detach())
                epoch_loss_tensor[3].add_(loss_adv.detach())
                epoch_loss_tensor[4].add_(loss_gen_total.detach())
                epoch_loss_tensor[5].add_(loss_fm.detach())
                epoch_loss_tensor[6].add_(loss_spectral.detach())
                epoch_loss_tensor[7].add_(loss_kl.detach())

            # Loss accumulation for rolling-avg

            # Losses:
            avg_rolling_cache["loss_disc"].append(loss_disc.detach())
            avg_rolling_cache["loss_disc_real"].append(loss_disc_real.detach())
            avg_rolling_cache["loss_disc_fake"].append(loss_disc_fake.detach())
            avg_rolling_cache["loss_adv"].append(loss_adv.detach()) 
            avg_rolling_cache["loss_gen_total"].append(loss_gen_total.detach())
            avg_rolling_cache["loss_fm"].append(loss_fm.detach())
            avg_rolling_cache["loss_spectral"].append(loss_spectral.detach())
            # Present only for the modes that combine two terms; ``setdefault``
            # keeps the series out of the log entirely for the modes that do
            # not, rather than writing a constant zero.
            for part_key, part_value in loss_spectral_parts.items():
                avg_rolling_cache.setdefault(
                    part_key, deque(maxlen=rolling_loss_steps)
                ).append(part_value.detach())
            avg_rolling_cache["loss_kl"].append(loss_kl.detach())

            # D Grads:
            if grad_norm_d is not None:
                if torch.isfinite(grad_norm_d):
                    avg_rolling_cache["grad_norm_d"].append(grad_norm_d)
                else:
                    writer.add_scalar("Grad_Norm_Diag/D_Skipped", 1, global_step)
            # G Grads:
            if grad_norm_g is not None:
                if torch.isfinite(grad_norm_g):
                    avg_rolling_cache["grad_norm_g"].append(grad_norm_g)
                    # Fraction of steps the clip actually fires.  A value pinned
                    # near 1.0 means the threshold sits below the gradient's
                    # normal operating point, so it is rescaling every update
                    # instead of only catching spikes.
                    if math.isfinite(grad_clip_value_g):
                        avg_rolling_cache.setdefault(
                            "grad_clip_hit_rate_g",
                            deque(maxlen=rolling_loss_steps),
                        ).append(
                            grad_norm_g.detach().gt(grad_clip_value_g).float()
                        )
                else:
                    writer.add_scalar("Grad_Norm_Diag/G_Skipped", 1, global_step)
            for key, value in module_grad_metrics.items():
                avg_rolling_cache.setdefault(
                    key,
                    deque(maxlen=rolling_loss_steps),
                ).append(value.detach())

            if rank == 0 and global_step % rolling_loss_steps == 0:
                scalar_dict_rolling = {}

                # Learning rate retrieval for rolling logging
                if from_scratch:
                    scalar_dict_rolling.update({
                        "learning_rate/lr_d": optim_d.param_groups[0]["lr"],
                        "learning_rate/lr_g": optim_g.param_groups[0]["lr"],
                    })

                # AMP health.  ``scale`` is the diagnostic: a healthy FP16 run
                # settles at a scale and grows it back after the occasional
                # overflow, so a scale walking down decade by decade, or a
                # ``skip_rate`` that stops returning to zero, is the run telling
                # you the gradients are overflowing faster than the scaler can
                # back off.  Without these two series that state is invisible --
                # the losses simply stop moving, because the steps are not
                # being applied.
                if grad_scaler is not None:
                    scalar_dict_rolling["AMP/grad_scaler_scale"] = grad_scaler.get_scale()
                    scalar_dict_rolling["AMP/skipped_steps_total"] = amp_skipped_steps
                    if amp_skip_cache:
                        scalar_dict_rolling[
                            f"AMP/skip_rate_{rolling_loss_steps}"
                        ] = sum(amp_skip_cache) / len(amp_skip_cache)

                # logging rolling averages
                for key, queue in avg_rolling_cache.items():
                    if len(queue) > 0:
                        if key.startswith("loss_"):
                            category = "loss"
                        elif (
                            key.startswith("prior_")
                            # The posterior series are the other half of every
                            # prior series and only mean something read side by
                            # side with it, so they have to share a namespace.
                            or key.startswith("posterior_")
                            or key.startswith("usage_")
                            or key.startswith("kl_")
                            or key in ("scale_anchor", "content_rms")
                        ):
                            category = "diag"
                        else:
                            category = "grad"
                        # dynamic labeling
                        label = f"{category}_avg_{rolling_loss_steps}/{key}_{rolling_loss_steps}"
                        # Calculate mean
                        val = torch.stack(list(queue)).mean().item() if torch.is_tensor(queue[0]) else sum(queue)/len(queue)
                        scalar_dict_rolling[label] = val

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)

                # KL diagnostics (diag tab)
                if len(kl_std_cache) > 0:
                    # The caches hold device tensors, so this is where the
                    # window's three numbers cross to the host -- once per
                    # ``rolling_loss_steps``, on a line that is already
                    # synchronising to write TensorBoard.
                    diag_scalars = {
                        "diag/kl_std": _cache_mean(kl_std_cache),
                        "diag/kl_mean_per_dim": _cache_mean(kl_mean_cache),
                        "diag/kl_active_fraction": _cache_mean(kl_active_cache),
                    }
                    summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=diag_scalars,
                    )
                    writer.add_histogram(
                        "diag/kl_per_dim_hist",
                        last_kl_per_dim.cpu(),
                        global_step,
                    )
                flush_writer(writer, rank)

            preview_interval = (
                finetune_preview_interval if finetune_phase else pretrain_preview_interval
            )
            if pretrain_preview and rank == 0 and phase_step % preview_interval == 0:
                with averaged_weights((optimizer_choice_g, optim_g)):
                    o = eval_infer(net_g, reference)
                if reference_audio is not None:
                    eval_original_mel = wave_to_mel(
                        config,
                        reference_audio,
                        num_mels=None,
                    )
                    eval_generated_mel = wave_to_mel(
                        config,
                        o,
                        num_mels=None,
                    )
                    log_validation_preview(
                        writer=writer,
                        experiment_dir=experiment_dir,
                        epoch=epoch,
                        sample_index=0,
                        global_step=global_step,
                        sample_rate=config.data.sample_rate,
                        predicted_mel=eval_generated_mel,
                        target_mel=eval_original_mel,
                        predicted_wave=o,
                        target_wave=reference_audio,
                        source=reference_source,
                        # The time axis is frames * hop / sample_rate.  This
                        # used to fall back to the function's default of 256
                        # while every shipped config uses sample_rate/100
                        # (441 at 44.1 kHz), which labelled the axis 1.7x short.
                        hop_length=config.data.hop_length,
                        # Same story one axis over: the frequency ticks are only
                        # right if they are placed on the mel range the mels
                        # were actually binned with.
                        mel_fmin=config.data.mel_fmin,
                        mel_fmax=config.data.mel_fmax,
                        dpi=validation_preview_dpi,
                        figsize=validation_preview_figsize,
                    )
                else:
                    log_tensorboard_media(
                        writer=writer,
                        namespace=TENSORBOARD_VALIDATION_FALLBACK_NAMESPACE,
                        global_step=global_step,
                        sample_rate=config.data.sample_rate,
                        audio={TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"]: o[0]},
                        text={TENSORBOARD_MEDIA_SOURCE_NAME: reference_source}
                        if reference_source
                        else None,
                    )
                flush_writer(writer, rank)
                torch.cuda.empty_cache()

            if rank == 0 and (
                not progress_metrics
                or (batch_idx + 1) % metrics_update_interval == 0
            ):
                progress_metrics = (
                    f"G={loss_gen_total.detach().float().item():.4f}  "
                    f"D={loss_disc.detach().float().item():.4f}"
                )
            progress.update(
                task_id,
                advance=1,
                metrics=progress_metrics,
            )
            _emit_machine_progress(
                epoch, total_epoch_count, batch_idx + 1, len(train_loader),
                global_step, progress_metrics, rank,
            )

            if stop_was_requested():
                # Nothing is being written at a batch boundary, so this is the
                # cheapest safe point to honour the request.
                finish_stop(writer if rank == 0 else None)

            if max_steps > 0 and phase_step >= max_steps:
                phase_limit_reached = True
                break

        # end of batch train
    # end of Rich progress

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Learning rate retrieval for avg-epoch variation:
        lr_d = optim_d.param_groups[0]["lr"]
        lr_g = optim_g.param_groups[0]["lr"]

        # At each epoch completion
        if global_step % len(train_loader) == 0 and not from_scratch:

            # Calculate the avg epoch loss:
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            # metrics dict
            scalar_dict_avg = {
            "loss_avg/loss_disc": avg_epoch_loss[0].item(),
            "loss_avg/loss_disc_real": avg_epoch_loss[1].item(),
            "loss_avg/loss_disc_fake": avg_epoch_loss[2].item(),
            "loss_avg/loss_adv": avg_epoch_loss[3].item(),
            "loss_avg/loss_gen_total": avg_epoch_loss[4].item(),
            "loss_avg/loss_fm": avg_epoch_loss[5].item(),
            "loss_avg/loss_spectral": avg_epoch_loss[6].item(),
            "loss_avg/loss_kl": avg_epoch_loss[7].item(),
            "learning_rate/lr_d": lr_d,
            "learning_rate/lr_g": lr_g,
            }

            summarize(writer=writer, global_step=global_step, scalars=scalar_dict_avg)
            flush_writer(writer, rank)
            num_batches_in_epoch = 0
            epoch_loss_tensor.zero_()

        # At each epoch save point:
        if epoch % epoch_save_frequency == 0 or phase_limit_reached:

            # Preview whatever this run would actually hand over, so the audio
            # you judge it by is the audio the exported model produces.
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            preview_sd, preview_label = _deliverable_weights(
                overtrain_monitor, ema, model_g
            )
            if preview_label != "live weights":
                live_sd_g = {k: v.detach().clone() for k, v in model_g.state_dict().items()}
                model_g.load_state_dict(preview_sd)
                info(f"Epoch {epoch}: rendering from {preview_label}.", tag="[PREVIEW]")

            # Inferencing on reference sample


            # ``live_sd_g`` set means the preview swapped in EMA or holdout
            # weights, which are already an evaluation point -- converting those
            # to the schedule-free x iterate would apply the transform to
            # weights its ``z`` state does not describe.
            if live_sd_g is None:
                with averaged_weights((optimizer_choice_g, optim_g)):
                    o = eval_infer(net_g, reference)
            else:
                o = eval_infer(net_g, reference)
            if reference_audio is not None:
                eval_original_mel = wave_to_mel(
                    config,
                    reference_audio,
                    num_mels=None,
                )
                eval_generated_mel = wave_to_mel(
                    config,
                    o,
                    num_mels=None,
                )
                log_validation_preview(
                    writer=writer,
                    experiment_dir=experiment_dir,
                    epoch=epoch,
                    sample_index=0,
                    global_step=global_step,
                    sample_rate=config.data.sample_rate,
                    predicted_mel=eval_generated_mel,
                    target_mel=eval_original_mel,
                    predicted_wave=o,
                    target_wave=reference_audio,
                    source=reference_source,
                    # This branch was still on the function's fallbacks, so its
                    # previews carried the hop-256 time axis the other call site
                    # was already fixed for.
                    hop_length=config.data.hop_length,
                    mel_fmin=config.data.mel_fmin,
                    mel_fmax=config.data.mel_fmax,
                )
            else:
                log_tensorboard_media(
                    writer=writer,
                    namespace=TENSORBOARD_VALIDATION_FALLBACK_NAMESPACE,
                    global_step=global_step,
                    sample_rate=config.data.sample_rate,
                    audio={TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"]: o[0]},
                    text={TENSORBOARD_MEDIA_SOURCE_NAME: reference_source}
                    if reference_source
                    else None,
                )

            # Restore live weights immediately ~ checkpoint saving stays raw
            if live_sd_g is not None:
                model_g.load_state_dict(live_sd_g)
                live_sd_g = None

        flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = phase_limit_reached or (overtrain_flagged and stop_on_overtrain)

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | phase_step={phase_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % epoch_save_frequency == 0 or phase_limit_reached:
            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            if save_only_latest_net_models:
                old_files = glob.glob(os.path.join(experiment_dir, "G_*.pth")) + glob.glob(os.path.join(experiment_dir, "D_*.pth"))
                for f in old_files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Both writes sit in one protected region: a stop between them would
            # leave a generator without its matching discriminator.  The
            # schedule-free switch wraps the whole region so the saved weights
            # are the averaged iterate and the saved optimizer state records
            # that it was written in that mode -- ``train_mode`` lives in
            # ``param_groups``, so resuming restores the pairing.
            with averaged_weights(
                (optimizer_choice_g, optim_g), (optimizer_choice_d, optim_d)
            ):
                with uninterruptible_save("checkpoint write"):
                    save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, ema=ema)
                    save_checkpoint(
                        net_d,
                        optim_d,
                        config.train.learning_rate_d,
                        epoch,
                        d_path,
                        extra=_checkpoint_extra(grad_scaler),
                    )


            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_add.append(os.path.join(experiment_dir, weight_model_name))

        # Check completion
        if epoch >= total_epoch_count:
            success(
                f"Training completed: {epoch} epochs, {global_step} steps, "
                f"generator loss {loss_gen_total.item():.3f}.",
                tag="[TRAIN]",
            )
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append(os.path.join(experiment_dir, weight_model_name))
            done = True

        if phase_limit_reached:
            info(
                f"Training phase limit reached at {phase_step} local steps "
                f"({global_step} global steps).",
                tag="[TRAIN]",
            )

        # Emitted once, by name, at the first epoch boundary after the turn is
        # seen.  Latched on a flag rather than on the filename: the monitor can
        # still find a new best afterwards, and keying the guard off
        # ``best_step`` would then write a second copy under a second name.
        if overtrain_flagged and not overtrain_exported and overtrain_monitor is not None:
            overtrain_exported = True
            model_add.append(
                os.path.join(
                    experiment_dir,
                    small_model_naming(
                        f"{model_name}_pre-overtrain", epoch, overtrain_monitor.best_step
                    ),
                )
            )

        if model_add:
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            # ``_deliverable_weights`` can fall through to the live weights, and
            # under a schedule-free optimizer those are the extrapolated
            # iterate.  This is the exported .pth -- the file that gets used for
            # inference -- so it is the last place that read may go unaveraged.
            with averaged_weights((optimizer_choice_g, optim_g)):
                ckpt, ckpt_label = _deliverable_weights(
                    overtrain_monitor, ema, model_g
                )
            success(f"Weights: {ckpt_label}", tag="[EXPORT]")

            for m in model_add:
                if not os.path.exists(m):
                  with uninterruptible_save("weight model export"):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=config,
                        vocoder=vocoder,
                        architecture=architecture,
                    )

        if stop_was_requested():
            finish_stop(writer if rank == 0 else None)

        if done:
            # Clean-up process IDs from memory
            pid_data["process_pids"].clear()  # Clear the PID list when done

            if rank == 0:
                writer.flush()
                writer.close()

            os._exit(0) #2333333

        with torch.no_grad():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
