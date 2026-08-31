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
    envelope_loss,
    local_log_rms_loss,
    peak_headroom_loss,
    mel_low_frequency_weights,
    BandWeightedSpectralLoss,
    MultiScaleSTFTLoss,
)

from mel_processing import MultiScaleMelSpectrogramLoss

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.configs.vocoders import (
    get_architecture_id,
    get_discriminator_id,
    get_vocoder_spec,
    normalize_vocoder,
    uses_chouwagan_stack,
)
from rvc.train.run_spec import TrainRunSpec

# ======== Run spec start region ===============================================
#
# argv[1] is the run spec written by the launcher.  Note this is *not* the same
# indexing as ``core._find_trainer_processes``, which inspects the OS command
# line -- that includes the interpreter, so the script sits at ``cmdline[1]``
# and the spec at ``cmdline[2]``.  Inside the process the interpreter is absent
# and ``sys.argv[0]`` is the script itself.
#
# Everything below is a plain module-level name so the rest of this file reads
# exactly as it did when these came from positional argv -- only the source
# changed.  The ``spawn`` start method re-executes this module in every DDP
# child, so each rank re-reads the same file and arrives at identical values.

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
# Generator and discriminator have always been given the same optimizer; the
# launcher sent one value into two slots.  Kept as two names because the rest
# of the file distinguishes them.
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
# The UI intentionally has no manual phase or step controls.  A pretrained
# source selects fine-tuning; no pretrained source selects pretraining.
training_phase = spec.training_phase
max_steps = 0

# Torch backend config -----
cuda.matmul.allow_tf32 = use_tf32
cudnn.allow_tf32 = use_tf32
cudnn.benchmark = use_benchmark

# Run spec end region ===============================================

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

# Load the config from json
config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

# Tuning knobs that used to ride along as UI flags.  They live in the config
# JSON now: neither is something to reach for per-run.  ``lr_decay`` was
# already present in every shipped config (and unread); ``rolling_loss_steps``
# is deliberately sticky because its value is baked into the TensorBoard tag
# names -- changing it starts a fresh set of series.  Both are read with a
# fallback so a config written before this change still loads.
exp_decay_gamma = float(getattr(config.train, "lr_decay", 0.999875))
# Which reconstruction term the run uses.  This belongs to the vocoder, not to
# the run: a 128-band mel cannot resolve a harmonic comb above ~2 kHz (at 8 kHz
# a bin spans four harmonics at f0=120), so RefineGAN needs the linear-frequency
# MS-STFT term beside it and ships "Hybrid L1", while HiFi-GAN ships the plain
# mel it was designed around.  Leaving it on the UI meant every run re-answered
# a question the vocoder had already answered, and answered it wrong by default.
spectral_loss = str(getattr(config.train, "spectral_loss", "L1 Mel Loss"))
# Horizon-derived decay, and the preferred way to set one: "end at this fraction
# of the starting LR", with the per-epoch gamma derived from the run's real
# length, so changing the epoch count restretches the schedule instead of
# silently changing the decay.  A ratio rather than an absolute target LR
# because G and D start at different rates and that balance is load-bearing.
# ``None`` keeps the old ``lr_decay`` behaviour.
lr_final_ratio = getattr(config.train, "lr_final_ratio", None)
lr_final_ratio = None if lr_final_ratio is None else float(lr_final_ratio)
rolling_loss_steps = int(getattr(config.train, "rolling_loss_steps", 50))

# Validation preview geometry.  ``dpi`` is the resolution knob and scales the
# whole figure, text included; width/height are inches and change how much room
# the panels get relative to the labels.  Pixel size is width * dpi by
# height * dpi -- the defaults give 1608x388.  Every preview is embedded in the
# event file at full size, so raising these grows the TensorBoard log too.
validation_preview_dpi = max(
    10.0, float(getattr(config.train, "validation_preview_dpi", 67))
)
validation_preview_figsize = (
    max(1.0, float(getattr(config.train, "validation_preview_width", 24.0))),
    max(1.0, float(getattr(config.train, "validation_preview_height", 5.8))),
)

# Two independent facts, kept apart because they used to be one flag.
#
# ``chouwagan_stack_active`` is about the *decoder stack*: it turns on the
# per-branch SAN discriminator, the R1 strength controller, the waveform
# reconstruction terms and the retain_graph backward pattern.  Only ChouwaGAN
# uses those.  Applio's RefineGAN is registered with the plain MPD+MSD
# discriminator and so leaves this False -- no SAN, no R1, no per-branch
# driving, which is exactly why this is no longer called ``refinegan_active``.
chouwagan_stack_active = uses_chouwagan_stack(vocoder)
if chouwagan_stack_active and sample_rate != 44100:
    raise ValueError(
        f"{get_vocoder_spec(vocoder)['label']} requires the 44.1 kHz configuration."
    )

# AMP precision / dtype init
# Default: FP32 + TF32 tensor cores, no autocast, no scaler.
#
# ``use_fp16`` enables ``torch.autocast`` at FP16 with ``GradScaler``.  The
# generator, frontend and discriminator autocast-disable wrappers have been
# narrowed to protect only distribution math and the NSF source, so the
# convolutional backbone (the dominant cost) runs in FP16 without inserting
# cast nodes that break Inductor fusions.
#
# BF16 carries FP32's exponent range, so it needs no GradScaler and cannot
# overflow the way FP16 does in the periodic activations, the anti-aliased
# resampling or the Gaussian latent statistics.
#
# History: an earlier measurement with blanket ``autocast(enabled=False)``
# wrapping the entire forward showed FP16 at 0.93x of TF32.  That was cast
# overhead; with the fencing narrowed to distribution math only, the compiled
# decoder graph stays in one dtype end-to-end.
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
# How many optimizer steps the GradScaler has thrown away because the gradients
# came back non-finite.  Under FP16 a skipped step is the *expected* response to
# an overflow, but a run that skips them in series is not training at all -- and
# without this counter that looks exactly like a loss that stopped improving.
# Cumulative across resumes: it rides in the D checkpoint's ``extra``.
amp_skipped_steps = 0



# ========  Advanced / Manual and exp tweaks  ========================
enable_persistent_workers = True

pretrain_preview = True
pretrain_preview_interval = 500  # Measured in steps.
finetune_preview_interval = 100  # Measured in steps.

# How often the per-loss `Grad_Source/*` probes fire, in steps.  Each firing
# costs 8 extra `autograd.grad` calls, 4 of which are full decoder backwards,
# so this is the most expensive diagnostic in the loop by a wide margin.  The
# series it feeds moves on the scale of tens of thousands of steps, so a
# coarser interval loses nothing and hands the time back to training.
grad_source_probe_interval = 200

force_from_scratch = False
strict_load = True

clip_grad_norm_override = False
clip_grad_norm_override_value_g = 100
clip_grad_norm_override_value_d = 100

# linear-warmup in steps. 0 = disabled
warmup_steps = 0

# Loss swap: L1 mel -> multi-scale mel transition over N steps (kicks in at resume)
swap_l1_to_ms = False  # Set True to swap from L1 mel loss to multi-scale mel loss
swap_duration_steps = 500  # Steps over which the transition runs
swap_start_step = 0  # Filled at resume: global_step when the swap was enabled
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
            print(f"[TRAIN] {description} finished; the stop can proceed now.")


def _handle_stop_signal(signum, frame):
    """Record a stop request. Deliberately does not exit."""
    if _stop_requested.is_set():
        return
    _stop_requested.set()
    with _saving_lock:
        mid_write = _saving_depth > 0
    if mid_write:
        print(
            "[TRAIN] Stop requested while writing a checkpoint - "
            "finishing the write first, then exiting.",
            flush=True,
        )
    else:
        print("[TRAIN] Stop requested - exiting at the next safe point.", flush=True)


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
    print("[TRAIN] Stopping cleanly.", flush=True)
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
    """Fixed excerpts of a fixed length, kept in RAM and batched on demand.

    The evaluation used to run through a ``DataLoader`` at batch size 1.  That
    was not a throughput choice: the collate pads every item up to the longest
    one in the batch, and padding silence into the mel puts a constant in the
    metric that moves with whatever else lands beside it -- at batch 1 there is
    nothing to pad against.

    Cropping every excerpt to the same number of frames removes the padding
    instead of the batching.  That matters because the step here is dispatch
    bound rather than compute bound, so a batch of eight costs barely more than
    a batch of one and the evaluation stops spending its wall clock in Python.

    Holds the excerpts rather than the batches so :meth:`shrink` can re-form
    them: the evaluation decodes several seconds per item where a training step
    decodes a fraction of one, so it can be the largest allocation in the run
    even though it stores no gradients.
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
                print(
                    f"[HOLDOUT] {len(holdout_indices)} slices from {sources} source "
                    f"recordings held out of {len(rows)} ~ never trained on."
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
            print(
                f"[HOLDOUT] Scoring {len(holdout_set)} excerpts of "
                f"{holdout_set.seconds(config):.1f}s each, batched "
                f"{holdout_set.batch_size} at a time."
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
    if chouwagan_stack_active:
        # Decoder-specific defaults.  Only the selected decoder's are applied,
        # so a run does not carry another decoder's channel schedule in its
        # config, and only keys the config leaves *absent* are filled in.
        decoder_defaults = {
            # The decoder is depthwise-separable and its blocks narrow as the
            # rate climbs, which is affordable because the excitation U-Net --
            # not the trunk -- carries pitch phase up to the output rate.  The
            # excitation is rendered once at full rate and band-limited down, so
            # every stage sees one phase-consistent source rather than four
            # independently re-rendered harmonic banks.
            "chouwagan": {
                "chouwagan_channels": [256, 160, 80, 40],
                "chouwagan_block_kernels": [[3, 7], [3, 7], [7], [7]],
                "chouwagan_dilations": [1, 3, 5],
                "chouwagan_expansion": [2, 2, 1, 1],
                "chouwagan_istft_hop": 0,
                "chouwagan_harmonics": 128,
                "chouwagan_excitation_unet": True,
                "chouwagan_latent_source_gate": True,
                "chouwagan_remove_output_dc": True,
            },
        }
        decoder_config_defaults = dict(
            decoder_defaults.get(get_vocoder_spec(vocoder)["generator"], {})
        )
        # The id is pinned rather than read from the config: ``net_g`` loads
        # non-strictly, so a checkpoint whose latent or decoder layout differs
        # would otherwise load with the mismatched modules left at their init
        # instead of failing the guard.
        config.model["architecture_id"] = get_architecture_id(
            vocoder, config.model.__dict__
        )
        for key, value in decoder_config_defaults.items():
            if key not in config.model:
                config.model[key] = value
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
    """Build the discriminator and apply the trainer's execution policy.

    ``d_branchwise`` is architecture-adjacent -- it decides what R1 penalises
    and how many ``loss_disc`` series exist -- so it is a constructor argument.
    ``d_generator_branchwise`` is not: it only decides whether the generator
    step walks the branches one at a time, which is a peak-memory-against-step-
    time trade with identical losses and gradients either way.  Setting it here
    keeps it out of every discriminator's signature, and keeps the two
    decisions from being read as one.
    """
    model = _build_d_model(config, vocoder, use_checkpointing)
    model.generator_branchwise = bool(
        getattr(config.model, "d_generator_branchwise", True)
    )
    return model


def _build_d_model(config, vocoder, use_checkpointing):
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

    if discriminator_id == "chouwagan":
        from rvc.lib.algorithm.discriminators.multi import chouwagan as chouwagan_d

        return chouwagan_d.ChouwaGANDiscriminator(
            use_spectral_norm=bool(
                getattr(config.model, "use_spectral_norm", False)
            ),
            use_checkpointing=use_checkpointing,
            sample_rate=config.data.sample_rate,
            # Per-branch driving: R1 rotated one branch at a time under its own
            # strength controller, plus a ``loss_disc`` series per branch.  Off
            # collapses all of that onto one branch; see ``branchwise.py``.
            # This is *not* the generator step's per-branch execution -- that is
            # ``d_generator_branchwise``, applied in ``get_d_model``.
            branchwise=bool(getattr(config.model, "d_branchwise", True)),
            # Sliced-adversarial heads.  Off falls back to plain LSGAN logits,
            # and ``_constant_discriminator_loss`` moves the governor's floor
            # with it -- the two loss forms have different floors.
            use_san=bool(getattr(config.model, "d_use_san", True)),
            # Same two layers as the mpd_msd path below: ``d_use_periods`` /
            # ``d_use_spectrogram`` switch a family off, ``d_periods`` /
            # ``d_spectrogram_specs`` say which members it has when it is on.
            # ``d_use_cqt`` and ``d_use_subband`` are already booleans because
            # each is a single branch rather than a family.
            periods=(
                ()
                if not setting("d_use_periods", True)
                else tuple(setting("d_periods", chouwagan_d.PERIODS))
            ),
            spectrogram_channels=tuple(
                setting("d_spectrogram_channels", chouwagan_d.SPECTROGRAM_CHANNELS)
            ),
            spectrogram_compression=float(
                getattr(config.model, "d_spectrogram_compression", 0.3)
            ),
            spectrogram_specs=(
                ()
                if not setting("d_use_spectrogram", True)
                else tuple(
                    setting("d_spectrogram_specs", chouwagan_d.SPECTROGRAM_SPECS)
                )
            ),
            use_cqt=bool(getattr(config.model, "d_use_cqt", False)),
            cqt_bins_per_octave=int(
                getattr(config.model, "d_cqt_bins_per_octave", 16)
            ),
            cqt_bins=int(getattr(config.model, "d_cqt_bins", 128)),
            cqt_f_min=float(getattr(config.model, "d_cqt_f_min", 80.0)),
            cqt_channels=tuple(setting("d_cqt_channels", chouwagan_d.CQT_CHANNELS)),
            use_subband=bool(getattr(config.model, "d_use_subband", False)),
            subband_bands=int(getattr(config.model, "d_subband_bands", 8)),
            subband_channels=tuple(
                setting("d_subband_channels", chouwagan_d.SUBBAND_CHANNELS)
            ),
        )

    from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined

    # ``mpd_msd`` is Applio's ``v2``; ``mpd_msd_v3`` is the layout Applio picks
    # for RefineGAN specifically -- five period branches instead of eight, plus
    # three multi-resolution spectrogram branches.  The id carries the version
    # so the registry stays the single place a vocoder's discriminator is
    # chosen by default.
    #
    # ``d_version`` overrides it, because the choice is a VRAM decision as much
    # as a quality one.  Measured on an 8 GB card, batch 8 over 0.4 s, decoder
    # plus both discriminator passes:
    #
    #   v2         4.64 GiB    498 ms/step
    #   v3         6.42 GiB   5912 ms/step   (does not fit; the allocator
    #                                         thrashing is the 12x, not the
    #                                         branches, which are worth 1.8x)
    #   v3l        5.97 GiB   1282 ms/step
    #   v3l + checkpointing
    #              3.43 GiB   1024 ms/step
    #   v3l, batch 6
    #              4.72 GiB    640 ms/step
    #
    # ``v3`` is what both RefineGAN configs ship, and it is exact Applio parity.
    #
    # ``v3l`` -- v3's branches with the frequency axis downsampled in the last
    # two layers -- was the default on the strength of the 24% above.  It did
    # not survive re-measurement: on an RTX 5060 it moved neither step time nor
    # peak VRAM against v3.  The branch microbenchmark it was chosen on
    # (81.1 ms / 342 MiB against 107.0 / 396, see the parity tests) is real, but
    # the discriminator is not the bound half of the step, so a saving there
    # does not reach the step -- the same reason the eager step is dispatch- and
    # not kernel-bound.  It stays available as an override; it is just not worth
    # spending parity on by default.
    # ``d_version: "v2"`` is the one that actually buys VRAM back.
    version = "v3" if discriminator_id == "mpd_msd_v3" else "v2"
    version = str(getattr(config.model, "d_version", None) or version)
    # Branch-by-branch overrides on top of the preset, in two layers.
    #
    # ``d_use_*`` is the switch: one boolean per family, all defaulting to on.
    # ``d_periods``/``d_resolutions`` are the *content* of a family that is on --
    # ``None`` keeps the preset's, a list replaces it.  Two keys rather than one
    # because "off" and "which ones" are different questions, and answering the
    # first should not require writing out an empty list.
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
    )


def _lazy_r1_penalty(
    net_d,
    real_audio,
    branch_index,
    segment_size=None,
):
    """Compute an unbiased branch sample of the mean discriminator R1.

    Sampling a window is this function's job; *what the penalty differentiates*
    is the discriminator's, because only it knows what each branch consumes --
    a spectrogram branch is penalised on its spectrogram rather than through
    the divergent ``|X|**0.3`` Jacobian.  This used to differentiate the
    waveform for every branch, which meant the model's own ``r1_penalty`` was
    dead code that only the tests exercised.
    """
    if segment_size is not None and real_audio.shape[-1] > int(segment_size):
        max_start = real_audio.shape[-1] - int(segment_size)
        start = int(
            torch.randint(
                max_start + 1,
                (),
                device=real_audio.device,
            ).item()
        )
        real_audio = real_audio[..., start : start + int(segment_size)]
    model = net_d.module if hasattr(net_d, "module") else net_d
    branch_penalty = getattr(model, "r1_penalty", None)
    if branch_penalty is not None:
        return branch_penalty(real_audio.detach().float(), int(branch_index))
    real_audio = real_audio.detach().float().requires_grad_(True)
    real_logits, _ = net_d(real_audio, branch_index=int(branch_index))
    real_score = real_logits.float().mean()
    real_grad = torch.autograd.grad(
        outputs=real_score,
        inputs=real_audio,
        create_graph=True,
        only_inputs=True,
    )[0]
    return real_grad.square().reshape(real_grad.shape[0], -1).sum(dim=1).mean()



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

    Decodes twice from the same batch -- once from the posterior sample ``z``
    that produced ``full_output``, once from the prior mean pushed through the
    flow -- and returns the mel L1 gap between them.  This is the measurement
    ``diag/kl_*`` cannot make, and the distinction matters because the two
    failure modes are indistinguishable in KL: a rate near zero means "the
    prior predicts the posterior" exactly as often as it means "the posterior
    stopped encoding anything", and only the second one is a problem.

    A large gap with a small KL says the posterior is load-bearing and the flow
    simply has not caught up -- training will close it.  A small gap says the
    decoder would produce the same audio without ever seeing the spectrogram,
    which is the collapse worth acting on.  The prior mean is used rather than
    a sample so the number is not confounded by the prior's variance.

    Diagnostic only: everything here runs under ``no_grad``.  A term of this
    shape has no gradient toward its own stated purpose -- the cheapest way to
    inflate it is to make the latent's scale destructive rather than
    informative -- so it is measured and never optimised.
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


def _decoder_parameters(net_g):
    model = net_g.module if hasattr(net_g, "module") else net_g
    return [parameter for parameter in model.dec.parameters() if parameter.requires_grad]


def _add_decoder_only_gradients(loss, net_g):
    decoder_parameters = _decoder_parameters(net_g)
    if not decoder_parameters:
        return

    decoder_gradients = torch.autograd.grad(
        loss,
        decoder_parameters,
        retain_graph=False,
        allow_unused=True,
    )
    _accumulate_gradients(decoder_parameters, decoder_gradients)


def _add_decoder_only_wave_gradients(y_hat, wave_gradient, net_g):
    """The same accumulation, entered from the waveform instead of a scalar.

    The branchwise generator step never builds a ``loss_gan`` graph -- each
    discriminator branch is backwarded into the detached waveform and thrown
    away -- so what reaches here is ``d loss_gan / d y_hat`` rather than
    ``loss_gan``.  Every path from the GAN objective to the decoder runs
    through ``y_hat``, so seeding the generator's backward with that gradient
    is the chain rule, not an approximation: it produces the same decoder
    gradients the joint path would have.
    """
    decoder_parameters = _decoder_parameters(net_g)
    if not decoder_parameters or wave_gradient is None:
        return

    decoder_gradients = torch.autograd.grad(
        y_hat,
        decoder_parameters,
        grad_outputs=wave_gradient,
        retain_graph=False,
        allow_unused=True,
    )
    _accumulate_gradients(decoder_parameters, decoder_gradients)


def _branchwise_generator_terms(
    discriminator_model,
    y,
    y_hat,
    real_spectrograms,
    *,
    use_amp,
    amp_dtype,
    loss_scale=1.0,
    san_direction_weight=1.0,
    use_softplus=False,
):
    """Adversarial and feature-matching terms, one discriminator branch at a time.

    The joint path holds every branch's feature maps and activation graph alive
    at once, from the forward until the single ``loss_gan`` backward, which is
    at the far end of the step: one copy of the largest intermediate tensors in
    the step per branch, held across every reconstruction loss in between.
    Here each branch is forwarded on a *detached* copy of the waveform,
    differentiated immediately, and dropped before the next one starts, so the
    peak holds one branch instead of all of them.  What survives
    the loop is two waveform-shaped gradients, which are the same size as
    ``y_hat``.

    The two terms are differentiated separately because their weights
    (``adaptive_adv`` and ``adaptive_fm``) are not known yet -- they are decided
    downstream from these very losses -- and the wave gradient is linear in
    both, so keeping them apart is what lets the governors stay exact rather
    than lag a step.

    Normalisation is applied after the loop rather than inside it: ``adv`` is a
    mean over branches and ``fm`` a mean over *all* layer terms of all branches,
    and neither denominator is known until every branch has been seen.  Both
    are single scalars, so scaling the accumulated losses and gradients at the
    end is identical to having divided each contribution as it arrived.

    ``loss_scale`` is the AMP scaler's factor.  The returned gradients carry it
    (they are handed to the optimizer, which unscales) while the returned
    losses do not (they are reported and fed to the controllers).
    """
    branches = discriminator_model.discriminators
    proxy = y_hat.detach().requires_grad_(True)
    adv_wave_grad = torch.zeros_like(proxy)
    fm_wave_grad = torch.zeros_like(proxy)
    adv_total = None
    fm_total = None
    term_count = 0

    for index in range(len(branches)):
        branch = branches[index]
        slot = discriminator_model._spectrogram_index(index)
        real_spectrogram = (
            real_spectrograms[slot]
            if real_spectrograms is not None and slot is not None
            else None
        )
        with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
            with torch.no_grad():
                _, real_features = discriminator_model._forward_one(
                    branch, y, real_spectrogram
                )
            fake_logit, fake_features = discriminator_model._forward_one(
                branch, proxy
            )
            # ``normalize=False`` on both: the denominators are applied once,
            # after the loop.
            branch_adv = generator_loss(
                [fake_logit],
                normalize=False,
                san_direction_weight=san_direction_weight,
                use_softplus=use_softplus,
            )
            branch_fm = feature_loss([real_features], [fake_features])

        term_count += min(len(real_features), len(fake_features))
        adv_total = branch_adv.detach() if adv_total is None else adv_total + branch_adv.detach()
        fm_total = branch_fm.detach() if fm_total is None else fm_total + branch_fm.detach()

        # ``retain_graph`` on the first of the pair only: the second frees this
        # branch's graph, which is the whole point of the loop.
        adv_wave_grad += torch.autograd.grad(
            branch_adv * loss_scale, proxy, retain_graph=True
        )[0]
        fm_wave_grad += torch.autograd.grad(branch_fm * loss_scale, proxy)[0]
        del real_features, fake_features, fake_logit, branch_adv, branch_fm

    branch_count = max(1, len(branches))
    term_count = max(1, term_count)
    zero = y_hat.detach().float().new_zeros(())
    return {
        "loss_scale": float(loss_scale),
        "loss_adv": (adv_total / branch_count) if adv_total is not None else zero,
        "loss_fm": (fm_total / term_count) if fm_total is not None else zero,
        "adv_wave_grad": adv_wave_grad / branch_count,
        "fm_wave_grad": fm_wave_grad / term_count,
    }


# The balance rule measures gradient norms with ``torch.autograd.grad``, and
# that traversal runs *outside* the ``GradScaler`` -- the scaler only covers
# what goes through ``scale(...).backward()``.  Under FP16 the probe therefore
# has no overflow protection at all, and a reconstruction loss in the tens
# overflows the half range on its way back through the decoder.  The overflow
# becomes ``inf``, ``inf - inf`` becomes ``NaN``, and the ratio of two NaN
# norms is NaN -- which is then cached for ``adaptive_adv_interval`` steps and
# multiplied into ``loss_gen_total``.  Measured on the 44.1 kHz FP16 pretrain:
# 50 of 50 generator steps skipped, every loss component finite and only the
# weights NaN, the scaler halving its scale every step from 2**10 to 2**-40.
#
# What makes this free to fix is that the rule is a *ratio*: ``||d(s*L)/dp||/s``
# is the same number for every ``s``, so each term can be measured through
# whatever scale keeps it inside the half range -- smaller when it overflows,
# larger when it underflows to nothing.  In FP32 the first attempt always
# succeeds and ``s`` stays 1.0, so nothing about the existing behaviour moves.
_PROBE_SCALE_ATTEMPTS = 6
_PROBE_SCALE_STEP = 256.0


def _probe_gradient(make_gradient):
    """One balance-rule gradient, measured through a scale that survives FP16.

    ``make_gradient(scale)`` returns ``d(scale * loss) / d(parameter)``, or
    ``None`` where the term does not reach the parameter at all.  Returns
    ``(gradient, norm)`` in FP32 and at unit scale, so callers see exactly the
    numbers they saw before -- the scale exists only inside the measurement.

    Returns the last attempt's values rather than raising when no scale works;
    the caller decides what a failed probe means, because that decision belongs
    where the cached weight lives.
    """
    scale = 1.0
    direction = 0.0
    gradient = None
    norm = None
    finite_gradient = None
    finite_norm = None
    for _ in range(_PROBE_SCALE_ATTEMPTS):
        scaled = make_gradient(scale)
        if scaled is None:
            return None, None
        gradient = scaled.detach().float() / scale
        norm = gradient.norm()
        value = float(norm)
        if value == value and value != float("inf"):
            finite_gradient = gradient
            finite_norm = norm
            if value > 0.0:
                return gradient, norm
            # Exactly zero: either the term genuinely does not move this
            # parameter, or every element underflowed.  One measurement through
            # a larger scale tells the two apart, and a term that is really
            # zero stays zero at any scale.
            step = _PROBE_SCALE_STEP
        elif direction == _PROBE_SCALE_STEP:
            # Growing the scale broke the measurement instead of rescuing it:
            # ``scale`` itself has outgrown the half range, so the multiply
            # that seeds the traversal is now ``inf`` and the zero it was
            # chasing was a real zero.  Keep the zero.
            break
        else:
            step = 1.0 / _PROBE_SCALE_STEP
        if direction and step != direction:
            # The two failure modes point opposite ways.  A term that overflows
            # at one scale and vanishes at the next has no representable
            # measurement, so stop rather than oscillate between them.
            break
        direction = step
        scale *= step
    if finite_norm is not None:
        return finite_gradient, finite_norm
    return gradient, norm


def _wave_parameter_gradient(y_hat, wave_gradient, parameter):
    """``d loss / d parameter`` for a loss that reaches it only through ``y_hat``.

    Scaled like every other balance-rule probe -- this traversal crosses the
    whole decoder in whatever dtype autocast recorded, so it overflows for the
    same reason and is measured the same way.  See ``_probe_gradient``.
    """
    if wave_gradient is None:
        return None
    gradient, _ = _probe_gradient(
        lambda scale: torch.autograd.grad(
            y_hat,
            parameter,
            grad_outputs=wave_gradient * scale,
            retain_graph=True,
            allow_unused=True,
        )[0]
    )
    return gradient


def _accumulate_gradients(parameters, gradients):
    # Accumulate with one multi-tensor kernel rather than one per parameter.
    # The decoder has 332 of them, and the old `parameter.grad + gradient` was
    # also out-of-place, so it allocated 332 tensors it immediately dropped:
    # measured 5.15 ms per step against 0.30 ms here, bitwise identical, on a
    # ~270 ms step.  This is the same in-place accumulation the autograd engine
    # itself does, so it is safe on grads that `backward()` just populated.
    existing_grads = []
    incoming_grads = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            existing_grads.append(parameter.grad)
            incoming_grads.append(gradient)
    if existing_grads:
        torch._foreach_add_(existing_grads, incoming_grads)


def _normalize_san_weights(net_d):
    model = net_d.module if hasattr(net_d, "module") else net_d
    for module in model.modules():
        normalize_weight = getattr(module, "normalize_weight", None)
        if normalize_weight is not None:
            normalize_weight()


def _gradient_norm(loss, targets):
    gradients = torch.autograd.grad(
        loss,
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    terms = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not terms:
        return loss.detach().new_zeros(())
    return torch.sqrt(torch.stack(terms).sum())


def _tensor_gradient_norm(loss, tensor):
    gradient = torch.autograd.grad(
        loss,
        tensor,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return loss.detach().new_zeros(())
    return gradient.detach().float().norm()


def _last_layer_parameter(module):
    """Return the differentiable leaf behind ``module.weight``.

    ``weight_norm`` from ``torch.nn.utils.parametrizations`` turns ``weight``
    into a property that is recomputed on every access, so the tensor handed
    back is *not* the leaf the autograd graph was built on.  Asking
    ``torch.autograd.grad`` for it silently yields ``None`` (with
    ``allow_unused=True``), which would pin the adaptive adversarial weight to
    its lower clamp forever.  Reach for the underlying ``original1`` (the
    direction vector ``v``) instead, which has the same shape as ``weight``.
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is not None and "weight" in parametrizations:
        return parametrizations["weight"].original1
    return module.weight


#: Final projection of each decoder, in the order they are looked up.  The
#: adaptive adversarial weight differentiates the reconstruction and adversarial
#: losses with respect to this layer, so it has to be the *last* one that both
#: reach.  Every decoder here is time-domain, so it is ``conv_post``;
#: ``output_proj`` stays in the lookup for a decoder that emits a spectrogram.
#: ``istft_head.proj`` is dotted because ChouwaGAN's iSTFT head keeps its
#: projection inside the head module.  It comes first: with the head active
#: ``conv_post`` is ``None``, and it is the head's projection that both the
#: reconstruction and the adversarial loss actually reach last.
_DECODER_OUTPUT_LAYERS = ("istft_head.proj", "conv_post", "output_proj")


def _decoder_output_layer(decoder):
    for name in _DECODER_OUTPUT_LAYERS:
        layer = decoder
        for part in name.split("."):
            layer = getattr(layer, part, None)
            if layer is None:
                break
        if layer is not None:
            return layer
    raise AttributeError(
        f"{type(decoder).__name__} has none of {_DECODER_OUTPUT_LAYERS}; the "
        "adaptive adversarial weight has no layer to differentiate."
    )


#: Wall-clock seconds between machine-readable progress lines.  Rich's bar is
#: for a terminal; this is for whatever is reading the pipe.
_MACHINE_PROGRESS_INTERVAL = 1.0
_last_machine_progress = 0.0


def _emit_machine_progress(
    epoch, total_epochs, batch, total_batches, step, metrics, rank
):
    """Print one parseable progress line, for a front-end reading stdout.

    Rich renders nothing at all when stdout is not a terminal -- a run driven
    by a GUI produces no progress output whatsoever -- so there is no bar to
    scrape.  This emits the same numbers in a fixed format instead, and only
    when there is no terminal to clutter, so an interactive run still sees just
    the bar.

    Throttled by wall clock rather than by batch count: batch rate varies by
    two orders of magnitude between configurations, and the reader wants a
    steady update rate, not a steady batch stride.
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


#: How fast ``loss_disc`` may drift upward, per 10k steps, while still counting
#: as a healthy GAN rather than a generator running away from its critic.
#:
#: A GAN's discriminator loss rises as the generator improves -- that is the
#: process working -- so this has to sit well clear of the normal case or noise
#: decides.  It is a property of the run rather than of the model, which is why
#: ``_increment`` scopes the trend gate by headroom instead of leaning on this
#: constant alone.  Raising it is not the fix when the ramp stalls.
HEALTHY_DISC_DRIFT_PER_10K = 0.03


def planned_step_count(total_epoch_count: int, train_loader, max_steps: int = 0) -> int:
    """How many optimizer steps this run will actually take.

    Every schedule below this line is denominated in steps, and a step count
    only means something against the length of the run it is scheduling.  A
    pretrain runs for hundreds of thousands of steps, so a 10k warmup is the
    first 2% of it.  A fine-tune on one speaker's dataset is 8k steps or fewer,
    where that same literal is the whole run: the warmup never reaches full
    weight, the ramp never reaches its ceiling, and the holdout is scored too
    few times for the patience to be reachable.  The values are not wrong, they
    are stated in absolute units against a run length that varies by two orders
    of magnitude.

    Returns 0 when the budget cannot be known, which every caller treats as
    "leave the configured value alone".
    """
    per_epoch = max(1, len(train_loader))
    planned = max(0, int(total_epoch_count)) * per_epoch
    limit = max(0, int(max_steps))
    if limit:
        planned = min(planned, limit) if planned else limit
    return planned


def fit_schedule(configured: int, planned_steps: int, fraction: float, minimum: int = 1) -> int:
    """Shrink a step-denominated schedule to fit inside the run.

    Only ever shrinks.  On a run long enough for the configured value the
    configured value comes back untouched, so pretraining is bit-for-bit
    unaffected and this cannot quietly re-tune a recipe that already works.
    ``fraction`` is the share of the run the schedule is allowed to occupy.
    """
    configured = max(0, int(configured))
    if planned_steps <= 0 or configured <= 0:
        return configured
    return max(minimum, min(configured, int(planned_steps * fraction)))


def fit_eval_interval(configured: int, planned_steps: int, patience: int) -> int:
    """An evaluation interval that lets ``patience`` actually be reached.

    The overtrain detector stops on ``patience`` consecutive evaluations
    without a new best, so the interval and the run length together decide
    whether it can fire at all.  At one evaluation per 2000 steps an 8k
    fine-tune gets four of them against a patience of eight: the detector is
    not strict there, it is *inert*, and a fine-tune is exactly where
    overtraining is most likely because the dataset is smallest.

    Sizing for ~3x patience is what makes it a detector again: enough
    evaluations that the counter can run out with training still left to
    discard, rather than arriving at the last step to report that the run is
    over.  Only ever shrinks, for the same reason as :func:`fit_schedule`.
    """
    configured = max(1, int(configured))
    if planned_steps <= 0:
        return configured
    wanted = planned_steps // max(1, 3 * max(1, int(patience)))
    return max(1, min(configured, wanted)) if wanted else configured


#: ``loss_disc`` for a constant discriminator under the plain LSGAN objective.
CONSTANT_DISCRIMINATOR_LOSS = 0.5


def _constant_discriminator_loss(san_direction_weight: float, san_active: bool) -> float:
    """``loss_disc`` for a discriminator that emits one constant for real and fake.

    This is the operational definition of a dead discriminator: it has stopped
    being a function of its input.  It is a fixed number for a given loss form,
    so the distance to it is an absolute health measure that needs no baseline
    and no reference run.  Both sides are symmetric about the midpoint of the
    real and fake targets, so the best constant is always 0.5.

    Mirrors the two branches of :func:`~rvc.train.losses.discriminator_loss`:
    SAN heads use a squared-softplus surrogate and carry a direction term at
    ``san_direction_weight``, which puts the floor near 2.2 rather than 0.5.
    Leaving it at 0.5 under SAN would tell the adversarial governor that a
    perfectly healthy discriminator had already collapsed, and it would clamp
    the ceiling to its minimum for the whole run.
    """
    if not san_active:
        return CONSTANT_DISCRIMINATOR_LOSS
    per_side = math.log1p(math.exp(0.5)) ** 2
    return (1.0 + max(0.0, float(san_direction_weight))) * 2.0 * per_side


class _AdversarialCeilingGovernor:
    """Advance the adversarial ceiling only while the discriminator holds.

    The bound on the adaptive adversarial weight is raised gradually rather
    than set once.  Early in a run the coarse spectral envelope is what
    matters and adversarial pressure only adds variance; texture -- high
    frequency detail and the aperiodic component -- is what the adversarial
    term is uniquely able to teach, and it is worth little before there is a
    signal to texture.  Ramping keeps the first phase reconstruction-driven
    without capping the second phase forever.  It is cosine-eased, so the
    ceiling never steps and the discriminator is never handed a sudden change
    in the objective it is chasing.

    The ceiling cannot be a pure function of ``global_step``, because the
    balance rule asks for ``balance_target * rec_grad / adv_grad`` and
    ``adv_grad`` *is* the discriminator's strength -- so a weakening
    discriminator makes the rule request a *larger* weight, which weakens it
    further.  The ceiling is the only thing bounding that loop.

    So the schedule is driven by a counter that advances only while the
    discriminator is holding its ground, and rewinds when it is not.  Two
    signals gate it, because they fail in different places:

    * ``loss_disc`` trending up -- fast EMA above slow -- means the generator
      is currently pulling ahead.  This is the early warning, and it fires
      while the discriminator is still useful.
    * headroom -- the distance from ``loss_disc`` to the constant-discriminator
      loss -- falling below an absolute floor.  This is the backstop for the
      case the trend misses: a discriminator that already collapsed and now
      sits flat, where the trend is zero precisely because there is nothing
      left to lose.

    The backstop is an *absolute* floor, not a fraction of the best headroom
    seen.  Headroom peaks early, while the generator is still weak, and falls as
    the generator improves -- which is the GAN working, not failing -- so a
    peak-relative rule holds the ceiling down for the rest of a healthy run.  An
    absolute floor answers the question actually being asked: is this
    discriminator near useless.

    Rewinding is deliberately faster than advancing.  The weight that causes
    the damage stays applied for as long as it takes to notice, so unwinding
    at the same rate would leave it there for twice that again.
    """

    def __init__(
        self,
        start_step: int,
        ramp_steps: int,
        ceiling_start: float,
        ceiling_end: float,
        floor_loss: float,
        fast_momentum: float = 5e-4,
        slow_momentum: float = 1e-4,
        tolerance: float | None = None,
        headroom_floor: float = 0.08,
        collapse_ceiling: float = 0.01,
        trend_gate_headroom: float = 2.0,
    ):
        self.start_step = max(0, int(start_step))
        self.ramp_steps = max(0, int(ramp_steps))
        self.ceiling_start = float(ceiling_start)
        self.ceiling_end = float(ceiling_end)
        self.floor_loss = float(floor_loss)
        # Where the ceiling goes when the discriminator has actually collapsed.
        # It has to be able to retreat *below* ``ceiling_start``: against a
        # discriminator emitting one constant for real and fake the balance rule
        # asks for a huge weight, so a hard bottom of 1.0 would keep full
        # adversarial pressure on a dead discriminator forever.  Retreating to
        # ``adaptive_adv_min`` lets the generator fall back on reconstruction
        # until the discriminator recovers.
        self.collapse_ceiling = min(float(collapse_ceiling), float(ceiling_start))
        self.fast_momentum = float(fast_momentum)
        self.slow_momentum = float(slow_momentum)
        # Derived, not chosen.  Two EMAs following a linear drift of ``r`` per
        # step settle at a fixed separation of ``r * (1/slow - 1/fast)``, because
        # each lags the true value by its own time constant.  So a *healthy*
        # upward drift produces a constant, predictable ``fast - slow``, and a
        # fixed threshold that does not account for it spends most of its budget
        # on the normal case and fires on noise.  Deriving it from a stated drift
        # rate keeps the early warning -- a runaway is fast, not a creep -- while
        # giving the healthy creep room to exist.
        if tolerance is None:
            tolerance = (HEALTHY_DISC_DRIFT_PER_10K / 10_000.0) * (
                1.0 / self.slow_momentum - 1.0 / self.fast_momentum
            )
        self.tolerance = float(tolerance)
        # ~0.055 of real/fake separation in score space: alive, but only just.
        self.headroom_floor = float(headroom_floor)
        # Above this the trend gate is not consulted at all.  The gate is an
        # early warning for a discriminator *on its way* to the floor, and one
        # sitting at several times the floor is not on its way anywhere -- so
        # its drift carries no information the backstop does not already have.
        # See ``_increment``.
        self.trend_gate_headroom = float(trend_gate_headroom) * self.headroom_floor
        # Long enough for the EMAs to mean something, short enough that a
        # finetune's own ramp is not spent entirely inside it.
        self.warmup_steps = min(2000, max(200, self.ramp_steps // 8))
        # Negative progress is the retreat below ``ceiling_start``.  Sized so a
        # sustained collapse (-4 per step) reaches ``collapse_ceiling`` in about
        # 2500 steps: fast enough to matter, slow enough that a brief dip does
        # not throw away the ramp.
        self.retreat_span = max(1.0, self.ramp_steps / 8.0)
        self.progress = 0.0
        self.fast: float | None = None
        self.slow: float | None = None
        self.best_headroom = 0.0
        self.holding = False
        self.seeded = False
        #: Steps *this governor* has observed, which is not ``global_step`` once
        #: a run resumes.  Every guard below counts in these.
        self.observations = 0

    def _seed(self, step: int) -> None:
        """Put the ramp where the *run* is, not where this process is.

        ``progress`` is a counter that advances at most +1 per step and is built
        fresh on every process start, so without this a resume would drop the
        ceiling back to ``ceiling_start`` and make the run re-earn a ramp it had
        already earned -- tens of thousands of steps under less adversarial
        pressure than the step before the resume.

        Seeding assumes the skipped steps were healthy, which forgets any holds
        and retreats in the original run.  That is the right way to be wrong
        here: a hold only pauses the ramp, so the worst case is a ceiling as
        high as an unheld run would have had, and if the discriminator really is
        in trouble the EMAs are seeded from the first ``loss_disc`` of the
        resumed run, so the backstop starts retreating on this same call.
        Starting from zero is wrong in every case including the healthy one.
        """
        self.seeded = True
        if self.ramp_steps > 0 and step > self.start_step:
            self.progress = min(float(self.ramp_steps), float(step - self.start_step))

    def update(self, step: int, loss_disc: float | None) -> float:
        if loss_disc is not None and math.isfinite(loss_disc):
            self.observations += 1
            if self.fast is None:
                self.fast = self.slow = float(loss_disc)
            elif self.observations <= self.warmup_steps:
                # Running mean, not the EMA, until there are enough samples for
                # the EMA to mean anything.  The slow EMA's time constant is
                # 1/slow_momentum = 10k steps, so whatever the *first* batch
                # produces would dominate ``slow`` for tens of thousands of
                # steps -- and on a resume that reads as a false collapse and
                # rewinds a ramp nothing was wrong with.  A running mean has no
                # seed to shake off: after k samples it *is* the mean of those k.
                self.fast += (loss_disc - self.fast) / self.observations
                self.slow += (loss_disc - self.slow) / self.observations
            else:
                self.fast += self.fast_momentum * (loss_disc - self.fast)
                self.slow += self.slow_momentum * (loss_disc - self.slow)
            if self.observations >= self.warmup_steps:
                self.best_headroom = max(self.best_headroom, self.headroom)
        if not self.seeded:
            # After the EMAs above, so the backstop is armed with a real
            # headroom before the seeded progress can be acted on.  A fresh
            # pretrain seeds at step 0 against ``start_step`` and a fresh
            # fine-tune seeds at ``phase_step`` 0, so both are no-ops.
            self._seed(step)
        if self.ramp_steps > 0 and step > self.start_step:
            self.progress = min(
                float(self.ramp_steps),
                max(-self.retreat_span, self.progress + self._increment(step)),
            )
        return self.ceiling

    def _increment(self, step: int) -> float:
        """+1 advance, 0 pause, -4 retreat.

        The middle value is the fix for a measured stall.  A rising ``loss_disc``
        has two causes that this trend check cannot tell apart: the generator
        improving, which is the GAN working, and the generator running away,
        which is what the ramp exists to catch.  Penalising both at -1 makes the
        healthy case *cost* ramp progress, and since the counter only gains +1
        the net rate is ``1 - 2h``: a discriminator merely holding a third of the
        time cancels two thirds of the ramp.

        Pausing instead of rewinding makes the trend gate what it is described
        as -- an early warning that stops the ramp -- rather than a mechanism
        that actively unwinds progress on healthy behaviour.  Retreat is left to
        the headroom backstop, which fires on the operational definition of a
        dead discriminator rather than on the sign of a noisy derivative.

        The gate is additionally scoped to *where it can be right*.
        ``tolerance`` comes from a stated healthy drift rate, and that rate is a
        property of the run rather than of the model, so no constant fits every
        run -- recalibrating it buys one run and loses the next.  This branch
        only matters for a discriminator heading for the floor, so it is
        consulted only while the discriminator is near enough for the answer to
        change anything.  Above ``trend_gate_headroom`` the ramp advances on the
        backstop alone, which is the check with an absolute meaning.
        """
        # Counted in observations, not in ``step``.  ``step`` is ``global_step``
        # during pretraining, so on a resume it is already far past any warmup
        # while these EMAs have seen a single batch -- which is exactly when
        # their verdict is worthless and the guard is needed most.
        if self.fast is None or self.slow is None or self.observations < self.warmup_steps:
            self.holding = False
            return 1.0
        if self.headroom < self.headroom_floor:
            self.holding = True
            return -4.0
        if (
            self.headroom < self.trend_gate_headroom
            and self.fast - self.slow > self.tolerance
        ):
            self.holding = True
            return 0.0
        self.holding = False
        return 1.0

    @property
    def headroom(self) -> float:
        """How far the discriminator is from being a constant.  Higher is better."""
        return 0.0 if self.slow is None else self.floor_loss - self.slow

    @property
    def ceiling(self) -> float:
        if self.ramp_steps <= 0:
            return self.ceiling_end
        if self.progress < 0.0:
            # Retreat below the starting ceiling.  Linear rather than cosine:
            # this branch is an emergency, and easing it would spend the first
            # few hundred steps of a collapse barely moving.
            retreat = min(1.0, -self.progress / self.retreat_span)
            return self.ceiling_start + (self.collapse_ceiling - self.ceiling_start) * retreat
        progress = min(1.0, self.progress / self.ramp_steps)
        eased = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.ceiling_start + (self.ceiling_end - self.ceiling_start) * eased


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
    """Score held-out audio down the inference path, in a single pass.

    ``mel_l1`` is the mel distance between what a listener actually gets and
    the ground truth: the prior path, not the training forward.  The training
    path samples a posterior that has seen the target spectrogram, and a model
    that is memorising scores well there long after it has stopped
    generalising.

    Prior, posterior and the gap between them all come out of the same pass
    over the same weights.  This used to be two functions -- one calling
    ``Synthesizer.infer``, one rebuilding the prior path by hand -- which
    decoded the prior twice per excerpt and, worse, did it under *different*
    weights: the metric under the EMA, the diagnostic under the live model.  So
    the two prior numbers sat side by side in TensorBoard without being
    comparable.  ``latent_prior`` is gone for the same reason: it *is*
    ``mel_l1``.

    ``noise_scale`` is the prior draw.  ``Synthesizer.infer`` hardcodes
    0.66666; at 0 this decodes the prior mean instead, which makes the metric a
    pure function of the weights and removes a sampling floor that was eating
    most of its dynamic range.

    Plain L1 on the log-mel, with no adversarial, feature-matching or KL term:
    those are scored against a discriminator and a schedule that keep moving,
    and the only thing that may move between two evaluations here is the
    generator.
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

    Overtraining is not visible in the training loss -- that is what the word
    means.  The generator keeps getting better at reproducing the slices it is
    shown while getting worse at everything else, so the only way to catch the
    turn is to score audio it has never been trained on and watch for the
    minimum.

    Two things here are judged against a *noise band* rather than a fixed
    threshold, because a fixed threshold means something different in every
    run:

    * The score is median-filtered over ``smoothing`` evaluations before
      anything is decided on it.  A single-point argmin over a flat tail picks
      the luckiest evaluation rather than the best weights: on the run that
      motivated this, the last six scores spanned 0.4% and the winner beat its
      neighbour by 0.1%.
    * ``min_delta`` is only a floor now.  The band is the residual spread of
      the recent scores about a straight line through them, so a curve that is
      still falling fast has a narrow band while a flat noisy one has a wide
      one.  It is what the run itself says a real change looks like.

    The median filter is centred, so the point being judged is one evaluation
    behind the newest and its weights have to still be around when it is
    judged -- which is why this keeps a short window of state dicts rather than
    a single snapshot.

    There are still two questions here, and they still want different answers:

    * *Which weights do we keep?*  The lowest smoothed score.  Ties inside the
      band go to the earlier step: the same measured quality with less
      memorisation behind it, and the later one only won on noise.
    * *Has the run stopped improving?*  The counter ignores anything inside the
      band, so noise cannot reset it forever.
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
        """Residual spread of the recent scores about a straight line.

        A trend line and not a mean: a run that is genuinely improving has a
        large spread about its mean and a small one about its slope, and it is
        the second that says how much of an evaluation-to-evaluation
        difference is real.  Returns 0 until there are enough points to fit,
        which leaves ``min_delta`` as the floor in the meantime.
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
        """``source`` is whatever produced ``value``: a model, or a ``WeightEMA``.

        Returns whether this evaluation moved the best, which is what the
        caller marks in the log.  Keeping the weights that were actually scored
        is the whole contract -- snapshotting the live model while having
        measured the average would deliver a model nobody evaluated -- and with
        a centred filter that means keeping the weights of the middle of the
        window, not of the newest evaluation.
        """
        if not math.isfinite(value):
            return False
        self.history.append((int(step), float(value)))
        self.sigma = self._noise_sigma()

        # A state dict per evaluation is a copy of the model per evaluation, so
        # only evaluations close enough to the running best to still win one
        # are cloned.  A window entry without weights can never be selected,
        # which costs nothing: for it to have won, its neighbours would have to
        # drag a score that far off the best back under it.
        competitive = (
            not math.isfinite(self.best_raw)
            or value <= self.best_raw + 2.0 * self._band(self.best_raw)
        )
        self._window.append(
            (int(step), float(value), _cpu_state_dict(source) if competitive else None)
        )
        self.best_raw = min(self.best_raw, float(value))

        if len(self._window) < self.smoothing:
            # Until the window fills there is nothing to filter with, so the
            # detector behaves like the unsmoothed one rather than reporting a
            # half-formed median -- which on an even window is not a median at
            # all, it is whichever side the tie broke towards.
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
        """Evaluate less often once the run has already been called.

        The detector fires long before a 200-epoch run ends, and with
        ``stop_on_overtrain`` off the run keeps going -- paying full price for
        the evaluation every interval to keep confirming an answer it already
        has.  It still evaluates, because a run can come back, just not as
        often.
        """
        self.interval_scale = max(1, int(self.interval_scale * max(1, int(factor))))
        return self.interval_scale

    @property
    def overtrained(self) -> bool:
        return self.state_dict is not None and self.since_progress >= self.patience


def _deliverable_weights(overtrain_monitor, ema, model_g):
    """The weights this run would hand you if it stopped right now.

    Ordered by how much each source knows.  The holdout monitor scored audio
    the model was never trained on, so it wins outright; when an EMA is also
    running its snapshot *is* the average at its best point.  The EMA comes
    next because averaging the trajectory beats any single point on it, and it
    needs no metric to be fooled by.  Live weights are whatever the last step
    happened to leave behind.

    Returns ``(state_dict, label)``; the label is what gets printed, so a run
    always says which of these it handed over.
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


def _checkpoint_extra(r1_controller, grad_scaler, r1_grad_scaler=None):
    """Training-loop state that a resume cannot re-derive, for the D checkpoint.

    Everything here is plain scalars and plain dicts on purpose: ``extra`` is
    unpickled under ``weights_only=True``, and the ``GradScaler``'s own
    ``state_dict`` is five floats and ints, so it stays inside that contract.
    Returns ``None`` when there is nothing to carry, which is what
    ``save_checkpoint`` already treats as "omit the key".
    """
    extra = {}
    if r1_controller is not None:
        extra["r1_controller"] = r1_controller.state_dict
    if grad_scaler is not None:
        extra["grad_scaler"] = grad_scaler.state_dict()
        extra["amp_skipped_steps"] = int(amp_skipped_steps)
    if r1_grad_scaler is not None:
        extra["r1_grad_scaler"] = r1_grad_scaler.state_dict()
    return extra or None


class _R1StrengthController:
    """Hold the lazy R1 penalty at a fixed share of the discriminator's own update.

    ``r1_gamma`` is an absolute penalty weight, and the quantity that matters is
    a *relative* one: how much of the discriminator's movement is the gradient
    penalty flattening it, versus the discriminative loss sharpening it.  Those
    two are only in a fixed relation if the discriminative gradient is constant,
    and it is not -- it decays over a run as the generator improves.  So a fixed
    gamma is a rising share, and the stabiliser grows into the thing it is
    supposed to be stabilising.

    The R1 step runs its own backward and its own ``optim_d.step()``, so one
    step in ``r1_interval`` moves the discriminator on the penalty alone.  Under
    a fixed gamma that share climbs as the run goes on, and the symptom is
    ``loss_disc`` rising while ``grad_norm_d`` falls -- a converging
    discriminator does the opposite, so that pattern is one being flattened.

    Targeting the ratio directly makes the stabilisation constant in the units
    that matter and removes the constant that has to be re-picked per run.  The
    update is multiplicative on ``log scale`` and symmetric in relative error,
    matching ``RefineGANSVAE._kl_beta`` -- the same controller shape for the same
    reason, and overshooting by 2x costs what undershooting by 2x costs.

    The scale applied to an R1 event is the one earned by the events before it:
    the penalty's gradient norm is only known after its backward, and choosing
    the strength before it is the whole point.

    The default target of 1.0 is where a healthy run sat, not a round number.

    Do not read the target as "smaller is safer".  R1 is a *stabiliser*, and
    this model has already died once from a frozen discriminator (see the SAN
    direction-loss fix), so driving the share toward zero trades one failure
    mode for its opposite.  Lowering the *bounds* is a different change from
    lowering the target; see ``minimum`` and ``maximum`` below.
    """

    def __init__(
        self,
        branch_names=(),
        target_ratio: float = 1.0,
        lr: float = 0.05,
        momentum: float = 0.05,
        minimum: float = 1e-4,
        maximum: float = 10000.0,
    ):
        #: The bounds are wide because the *absolute* strength each branch needs
        #: to hold the same share spans ~700x, and that spread is legitimate: it
        #: tracks the transforms, not the regularisation.  A branch whose penalty
        #: gradient is small against its own discriminative gradient needs a
        #: large scale to carry the target share, and vice versa.  An absolute
        #: scale is therefore never comparable across branches -- only the ratio
        #: is -- which is the whole reason this controller targets one.
        #:
        #: Both bounds are places the controller loses authority, and neither is
        #: visible in the aggregate ``GAN/r1_*`` series, which are means over the
        #: branches.  So both are *detected* rather than merely bounded: see
        #: ``newly_saturated`` and ``newly_floored``.  Read ``R1_Branch/scale_*``
        #: to tell a branch that reached its target from one that ran to a bound
        #: and stayed.
        #:
        #: The floor binds for the mirror-image reason the ceiling does: a branch
        #: that separates real from fake easily has a *small* discriminative
        #: gradient, and that gradient is the denominator of the share, so the
        #: strength it needs falls with its own success.  The strongest branches
        #: arrive there first.
        #:
        #: Widening a bound is not lowering the target -- at target the penalty
        #: is still 100% of the branch's own movement.  The class docstring's
        #: warning about the target still stands.
        #:
        #: Do **not** "fix" a branch that needs an outlying strength by making it
        #: look like its neighbours.  For ``stft_2048`` in particular the stride,
        #: kernel and width are deliberate -- it is the only branch that can
        #: resolve a harmonic comb; see ``SPECTROGRAM_SPECS``.  Its large
        #: absolute strength is that design working, not a symptom.
        #:
        #: One controller per branch, because ``uses_branchwise_r1`` means each
        #: R1 event hits exactly one of them.  A single branch name (or none)
        #: degrades to the global behaviour this class shipped with.
        self.names = tuple(branch_names) or ("_global",)
        self.target_ratio = max(0.0, float(target_ratio))
        self.lr = max(0.0, float(lr))
        # Faster than the KL controller's 0.01 because this one is fed once per
        # ``r1_interval`` steps, not once per step: at interval 16 a momentum of
        # 0.05 is a ~320-step window, close to the KL controller's 100.  Per
        # branch that is 16x longer again -- a branch's own R1 fires once every
        # ``r1_interval * num_branches`` steps -- which is the price of asking
        # the question per branch rather than in aggregate.
        self.momentum = min(1.0, max(1e-6, float(momentum)))
        self.minimum = max(1e-6, float(minimum))
        self.maximum = max(self.minimum, float(maximum))
        self.log_scale = {name: 0.0 for name in self.names}
        self.disc_ema: dict[str, float | None] = {name: None for name in self.names}
        self.r1_ema: dict[str, float | None] = {name: None for name in self.names}
        # Consecutive R1 events a branch has spent pinned at ``maximum`` while
        # still short of target -- the state in which it has no authority left.
        self._pinned: dict[str, int] = {name: 0 for name in self.names}
        self._warned: set[str] = set()
        # The same state at the other bound: pinned at ``minimum`` while still
        # *over* target.  Counted separately because it is a separate failure
        # with a separate remedy, and because ``newly_saturated`` is documented
        # and tested as being about the ceiling alone.
        self._floored: dict[str, int] = {name: 0 for name in self.names}
        self._warned_floor: set[str] = set()

    @property
    def active(self) -> bool:
        return self.target_ratio > 0.0

    def _key(self, branch) -> str:
        if isinstance(branch, int):
            return self.names[branch % len(self.names)]
        return branch if branch in self.log_scale else self.names[0]

    def scale_for(self, branch) -> float:
        return math.exp(self.log_scale[self._key(branch)])

    def ratio_for(self, branch) -> float:
        """The measured share for one branch, or 0.0 before both EMAs exist."""
        key = self._key(branch)
        disc, r1 = self.disc_ema[key], self.r1_ema[key]
        return 0.0 if not disc or r1 is None else r1 / disc

    @property
    def scale(self) -> float:
        """Mean strength, for the aggregate series only."""
        values = [math.exp(v) for v in self.log_scale.values()]
        return sum(values) / len(values)

    @property
    def ratio(self) -> float:
        """Mean of the branches that have measured a share, or 0.0 before any has.

        Branches with no R1 event yet are excluded rather than counted as zero,
        which would drag the aggregate down for the first
        ``r1_interval * num_branches`` steps of a run and read as a controller
        that is failing.
        """
        live = [
            value
            for value in (self.ratio_for(name) for name in self.names)
            if value > 0.0
        ]
        return sum(live) / len(live) if live else 0.0

    def observe_discriminator(self, branch, grad_norm: float) -> None:
        """The reference this branch's R1 is measured against.

        Fed from the *discriminative* step and restricted to this branch's own
        parameters.  Using the whole discriminator's norm -- which is what the
        global version did -- compares a branch's penalty against a total it
        contributes a fraction of, and the fractions differ by two orders of
        magnitude across branch types.
        """
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            return
        key = self._key(branch)
        current = self.disc_ema[key]
        self.disc_ema[key] = (
            float(grad_norm)
            if current is None
            else current + self.momentum * (float(grad_norm) - current)
        )

    def observe_r1(self, branch, grad_norm: float) -> None:
        """After the R1 backward: measure this branch's share, correct its strength."""
        if not self.active or not math.isfinite(grad_norm) or grad_norm <= 0.0:
            return
        key = self._key(branch)
        current = self.r1_ema[key]
        self.r1_ema[key] = (
            float(grad_norm)
            if current is None
            else current + self.momentum * (float(grad_norm) - current)
        )
        if not self.disc_ema[key]:
            return
        # Every branch starts at strength 1.0 and *walks*.  Seeding it from the
        # branch's first measurement looks like the obvious speed-up -- the share
        # is linear in the strength, so one sample appears to name the answer --
        # but that linearity holds only at fixed weights, and the first R1 event
        # lands with the discriminator still at its initialisation, where the
        # gain has no relation to the trained one.  It was tried and it seeded
        # every branch wrong, several by orders of magnitude and the one it
        # existed for in the wrong direction.
        #
        # The asymmetry is the lesson: this controller exists because the gain
        # *moves* over a run, so no single early measurement can name a
        # strength.  Under-regularised early is mild and self-correcting;
        # over-regularised early flattens the discriminator.  Walk.
        #
        # Relative error, clamped so a single outlying event -- the R1 series is
        # spiky -- moves the strength by at most ``lr`` in log space rather than
        # by its full excursion.
        error = (self.ratio_for(key) - self.target_ratio) / self.target_ratio
        error = max(-1.0, min(1.0, error))
        self.log_scale[key] = max(
            math.log(self.minimum),
            min(math.log(self.maximum), self.log_scale[key] - self.lr * error),
        )
        # Pinned at a bound while the error still points past it is the one
        # failure this controller cannot correct, and it is indistinguishable
        # from health in every aggregate.  So it says so.  At the ceiling that
        # is ``error < 0``: still short of target with no strength left to add.
        if self.log_scale[key] >= math.log(self.maximum) - 1e-9 and error < 0.0:
            self._pinned[key] += 1
        else:
            self._pinned[key] = 0
            self._warned.discard(key)
        # And at the floor, ``error > 0``: the branch is still asking to be
        # regularised less than the bound allows.
        if self.log_scale[key] <= math.log(self.minimum) + 1e-9 and error > 0.0:
            self._floored[key] += 1
        else:
            self._floored[key] = 0
            self._warned_floor.discard(key)

    def newly_saturated(self, minimum_events: int = 200):
        """Branches that have just run out of authority, reported once each.

        Counted in the branch's *own* R1 events, not steps: a branch fires once
        every ``r1_interval * num_branches`` steps, so 200 events is roughly
        22k steps at interval 16 over 7 branches -- long enough that a branch
        still climbing toward a reachable target does not trip it.
        """
        fresh = []
        for name in self.names:
            if self._pinned[name] >= minimum_events and name not in self._warned:
                self._warned.add(name)
                fresh.append((name, self.ratio_for(name)))
        return fresh

    def newly_floored(self, minimum_events: int = 200):
        """Branches stuck at ``minimum`` and still over target, reported once each.

        The floor's counterpart to ``newly_saturated``.  Both bounds are places
        this controller loses authority, and neither is visible in
        ``GAN/r1_scale`` or ``GAN/r1_to_disc_ratio``, which are means over the
        branches.  Counted in the branch's own R1 events on the same budget as
        the ceiling.
        """
        fresh = []
        for name in self.names:
            if self._floored[name] >= minimum_events and name not in self._warned_floor:
                self._warned_floor.add(name)
                fresh.append((name, self.ratio_for(name)))
        return fresh

    @property
    def state_dict(self) -> dict:
        return {
            "version": 2,
            "log_scale": dict(self.log_scale),
            "disc_ema": dict(self.disc_ema),
            "r1_ema": dict(self.r1_ema),
        }

    def load_state_dict(self, state: dict | None) -> bool:
        """Restore across a resume.

        Worth persisting rather than re-deriving: this controller is fed once
        per ``r1_interval`` steps, so at interval 16 an unseeded resume spends
        thousands of steps re-earning a strength it already had -- the same
        mistake ``_AdversarialCeilingGovernor._seed`` exists to undo.

        Accepts the flat, pre-per-branch layout as well.  Those checkpoints hold
        one strength that *was* applied to every branch, so seeding every branch
        from it restores exactly the state the run was in; the branches then
        diverge from there.  The scalar EMAs are deliberately **not** carried
        over: they were norms over the whole discriminator, and the per-branch
        controller measures each branch's own parameters, so the two are not the
        same quantity and reusing them would seed every ratio wrong.
        """
        if not isinstance(state, dict) or "log_scale" not in state:
            return False
        stored = state["log_scale"]
        if not isinstance(stored, dict):
            try:
                seed = float(stored)
            except (TypeError, ValueError):
                return False
            self.log_scale = {name: seed for name in self.names}
            self.disc_ema = {name: None for name in self.names}
            self.r1_ema = {name: None for name in self.names}
            return True

        def _restore(key: str, target: dict) -> None:
            values = state.get(key) or {}
            if not isinstance(values, dict):
                return
            for name in self.names:
                if name in values:
                    value = values[name]
                    target[name] = None if value is None else float(value)

        # A branch absent from the checkpoint -- ``periods`` changed, the
        # sub-band branch was turned on -- keeps its cold default rather than
        # inheriting an unrelated branch's strength.
        for name in self.names:
            if name in stored:
                self.log_scale[name] = float(stored[name])
        _restore("disc_ema", self.disc_ema)
        _restore("r1_ema", self.r1_ema)
        return True


def _branch_discriminative_norms(net_d):
    """Per-branch gradient norms of the discriminative loss, in branch order.

    Called while ``loss_disc``'s gradients are still on the parameters, i.e.
    after its backward and before ``optim_d.step()``.  ``max_norm=inf`` makes
    ``clip_grad_norm_`` a pure measurement: the coefficient it computes is
    infinite and then clamped to 1.0, so nothing is scaled.

    Sampled rather than run every step -- see the call site.
    """
    model = net_d.module if hasattr(net_d, "module") else net_d
    # The grouping is the discriminator's to state: with per-branch driving off
    # it reports one group covering every branch, which is the aggregate the
    # single controller is steering.  Zipping this against ``controller.names``
    # would otherwise silently truncate to the first branch's norm.
    groups = getattr(model, "branch_parameter_groups", None)
    if groups is None:
        groups = lambda: [branch.parameters() for branch in model.discriminators]
    return [float(clip_grad_norm_(group, float("inf"))) for group in groups()]


def _adaptive_adversarial_weight(
    loss_reconstruction,
    loss_adversarial,
    parameter,
    balance_target: float = 0.5,
    minimum: float = 0.01,
    maximum: float = 1.0,
    adversarial_gradient=None,
):
    """Balance adversarial pressure against reconstruction on one decoder layer.

    Returns ``(applied, reconstruction_norm, adversarial_norm, requested)``.
    ``requested`` is the unclamped ratio: it is what the balance rule asked
    for, and logging it is the only way to see that the ceiling is binding.
    Reporting just the clamped value hides a saturated weight as a flat line at
    the ceiling, which is indistinguishable from a healthy constant.
    """
    zero = loss_reconstruction.detach().float().new_zeros(())
    _, reconstruction_norm = _probe_gradient(
        lambda scale: torch.autograd.grad(
            loss_reconstruction * scale,
            parameter,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
    )
    if adversarial_gradient is None:
        # ``None`` where the branchwise step has already produced this by the
        # chain rule from the waveform: there the adversarial loss is a
        # detached scalar with no graph left to differentiate.
        _, adversarial_norm = _probe_gradient(
            lambda scale: torch.autograd.grad(
                loss_adversarial * scale,
                parameter,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
        )
    else:
        # Already measured through the probe scale by
        # ``_wave_parameter_gradient``, which is where that traversal happens.
        adversarial_norm = adversarial_gradient.detach().float().norm()

    if reconstruction_norm is None:
        reconstruction_norm = zero
    if adversarial_norm is None:
        adversarial_norm = zero
    requested = (
        balance_target * reconstruction_norm / (adversarial_norm + 1e-6)
    ).detach()
    adaptive_weight = requested.clamp(min=minimum, max=maximum).detach()
    return adaptive_weight, reconstruction_norm, adversarial_norm, requested


def _adaptive_feature_match_weight(
    loss_feature_match,
    parameter,
    reconstruction_norm,
    balance_target: float,
    minimum: float = 0.1,
    maximum: float = 1.0,
    feature_gradient=None,
):
    """The same balance rule, applied to the other discriminator-derived loss.

    ``_adaptive_adversarial_weight`` governs ``loss_adv`` and nothing else, but
    ``loss_fm`` is also computed from the discriminator's feature maps and grows
    with it -- and it was carrying a fixed weight.  So the balance rule held one
    half of the GAN objective at target while the other half was free to grow,
    which is the loophole rather than a second opinion.

    Left ungoverned, feature matching grows into the largest single source of
    gradient into the waveform -- larger than the adversarial term the governor
    is busy restraining -- while the adversarial weight falls to hold its own
    half at target.

    ``balance_target`` is the fm-to-reconstruction gradient ratio to hold at the
    last decoder layer, so it is directly comparable to
    ``adv_balance_target``.  Its default is where a healthy run sat.

    ``maximum`` is 1.0 on purpose, and it means this can only ever *reduce*
    ``fm_weight``, never amplify it.  The configured weight stays the
    ceiling because the observed failure is overgrowth; letting a controller
    push feature matching above what the config asked for would be a second,
    unproven change riding along with this one.
    """
    zero = loss_feature_match.detach().float().new_zeros(())
    if feature_gradient is None:
        # A detached loss with no supplied gradient means feature matching is
        # not reaching the decoder at all; a detached loss *with* one is the
        # branchwise step, which computed it from the waveform instead.
        if not loss_feature_match.requires_grad:
            return loss_feature_match.detach().new_ones(()), zero, zero

        _, feature_match_norm = _probe_gradient(
            lambda scale: torch.autograd.grad(
                loss_feature_match * scale,
                parameter,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
        )
        if feature_match_norm is None:
            feature_match_norm = zero
    else:
        # Already measured through the probe scale by
        # ``_wave_parameter_gradient``.
        feature_match_norm = feature_gradient.detach().float().norm()
    requested = (
        balance_target * reconstruction_norm / (feature_match_norm + 1e-6)
    ).detach()
    return requested.clamp(min=minimum, max=maximum).detach(), feature_match_norm, requested


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
    lazy_reg_interval = None
    if chouwagan_stack_active and float(getattr(config.train, "r1_gamma", 0.0)) > 0:
        lazy_reg_interval = int(getattr(config.train, "r1_interval", 16))
    optim_d = _make_optimizer(
        net_d,
        optimizer_choice_d,
        lr_d,
        num_epochs=total_epoch_count,
        num_batches=num_batches,
        lazy_reg_interval=lazy_reg_interval,
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
            print("[INIT] Frontend: nothing frozen")
        return

    model = net_g.module if hasattr(net_g, "module") else net_g
    frozen_params = 0
    for name, param in model.named_parameters():
        if not name.startswith("dec.") and not name.startswith("emb_g.") and param.requires_grad:
            param.requires_grad = False
            frozen_params += param.numel()

    if rank == 0:
        print(f"[INIT] Frontend frozen: everything but dec/emb_g ({frozen_params:,} params)")


def apply_training_freezes(net_g, rank):
    apply_frontend_freeze(net_g, rank)


def apply_resume_lr_override(optim_g, optim_d=None):
    """Re-anchor G and/or D param groups to `resume_lr` after loading
    optimizer state.

    Must run AFTER load_checkpoint (so the saved lr / initial_lr are present)
    and BEFORE prepare_schedulers (so the schedulers snapshot the new
    initial_lr as their base). Each group's saved decay ratio ( current /
    initial ) is preserved, so the LR schedule keeps its position while the
    value restarts at resume_lr x its per-group lr_scale ( baked in by
    build_decoder_param_groups; 1.0 fallback for the default single group ).
    `resume_lr_target` picks which side(s) it applies to: "full", "g", or "d".
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
    print(f"[OVERRIDE] Resume LR override: base {resume_lr:.2e} -> " + " | ".join(parts))


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
            print(VOCODER_COMPILE_NO_CUDA)
        return False

    cache_dir = os.path.join(current_dir, "logs", ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    mode = torch_compile_mode
    if chouwagan_stack_active:
        # The RefineGAN loop backwards through the decoder graph more than once
        # per step: `loss_core.backward(retain_graph=True)` is followed by the
        # decoder-only `autograd.grad` for the GAN term, plus the adaptive-adv
        # and Grad_Source probes.  AOTAutograd's donated-buffer optimisation
        # rejects `retain_graph=True` outright, so it must be off for every
        # compiled mode.  The HiFi-GAN path does a single backward and keeps it.
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
    if chouwagan_stack_active and mode == "reduce-overhead":
        # `reduce-overhead` records CUDA graphs, which cannot survive the extra
        # backward passes above: the Grad_Source probes fire on step 50 and trip
        # "graph recording observed an input tensor deallocate during graph
        # recording".  No config flag fixes it, so fall back to the fastest
        # cudagraph-free mode instead of crashing 50 steps in.  Measured on the
        # real backward pattern, 'default' beats 'max-autotune-no-cudagraphs'
        # (1.30x vs 1.26x) and warms up far quicker.
        mode = "default"
        if rank == 0:
            print(
                "[INIT] 'reduce-overhead' is incompatible with the VITS-latent "
                "multi-backward loop (CUDA graphs); using "
                f"'{mode}' instead."
            )

    model = net_g.module if hasattr(net_g, "module") else net_g
    enabled = model.enable_decoder_compile(mode=mode)
    if not enabled and rank == 0:
        print(VOCODER_COMPILE_NOT_SUPPORTED)
    if enabled and rank == 0:
        print(VOCODER_COMPILE_ENABLED.format(mode=mode))
    return enabled


def enable_discriminator_compile(net_d, config, device, rank):
    """Compile the discriminator, driven by ``compile_discriminator``.

    Config-driven rather than a run-spec flag: it is a property of the
    RefineGAN discriminator's shape contract, not a choice a user makes per
    run.  It travels with the architecture that makes it valid.

    ``torch_compile_mode`` is deliberately not consulted.  That option exists
    for the decoder, and the one mode a user would reach for --
    ``reduce-overhead`` -- records CUDA graphs, which this loop cannot support
    (see :func:`enable_vocoder_compile`) and which an 8 GiB card has no room
    for besides.  Plain fusion is the whole benefit here, so the mode is fixed.
    """
    if not chouwagan_stack_active:
        return False
    if not bool(getattr(config.train, "compile_discriminator", False)):
        return False
    if device.type != "cuda":
        if rank == 0:
            print(DISCRIMINATOR_COMPILE_NO_CUDA)
        return False

    cache_dir = os.path.join(current_dir, "logs", ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

    model = net_d.module if hasattr(net_d, "module") else net_d
    enable = getattr(model, "enable_compile", None)
    if enable is None:
        if rank == 0:
            print(DISCRIMINATOR_COMPILE_NOT_SUPPORTED)
        return False
    enabled = enable(mode="default")
    if enabled and rank == 0:
        print(DISCRIMINATOR_COMPILE_ENABLED.format(mode="default"))
    return enabled


def collect_source_path_metrics(net_g):
    """How hard the pitch template is pushed into each decoder stage.

    Read off the weights, not the activations, so it costs nothing and can be
    sampled at the rolling-log cadence.

    Why this is worth a series of its own: dissecting the stock RVC v2 HiFi-GAN
    pretrains showed that the one thing separating them from a short reproduction
    was the *strength of the source injection at the high-rate stages*.  Their
    `noise_convs` sit at 1.4-2.0x of init on the first (lowest-rate) stage and at
    0.4-8% of init on the rest -- the harmonic source enters once, heavily
    filtered, and the network synthesises the rest.

    ChouwaGAN's excitation U-Net makes the same quantity directly readable.
    Every decoder stage opens by concatenating its upsampled activation with the
    excitation skip taken at that resolution, and that skip is a pure function of
    the NSF source: the excitation encoder sees nothing else.  ``fusion_proj`` is
    the 1x1 that decides the mix, so the ratio of its skip-half column norms to
    its main-half column norms answers "how much of this stage's output is
    excitation rather than latent".

    With ``chouwagan_latent_source_gate`` on that ratio stops being the whole
    answer: the skip is scaled by ``2 * sigmoid(gate(latent))`` before it
    reaches ``fusion_proj``, so the mixing weight and the gate multiply, and
    reporting the weight alone silently understates or overstates the share.
    Four series come out of it, answering different questions:

    ``inject_ratio_stage_i``
        The raw ``fusion_proj`` ratio, unchanged, so a gated run stays
        comparable against one without the gate.
    ``latent_gate_mean_stage_i``
        The gate's mean over the last forward's batch, read from the buffer the
        decoder writes.  1.0 at initialisation.
    ``latent_gate_saturation_stage_i``
        The share of gate elements pinned at the ceiling.  A saturated gate is a
        hard mask, not a multiplier, and this is what says which of the two is
        being reported.
    ``latent_gate_sensitivity_stage_i``
        The norm of the gate's weight: how much the multiplier varies with the
        latent.  Exactly 0.0 at initialisation, so this is the series that says
        whether the pitch path has become conditional on ``z`` at all, which is
        the whole point of the gate.
    ``inject_ratio_effective_stage_i``
        ``inject_ratio * latent_gate_mean``: the excitation share the stage
        really sees.

    The mean is measured rather than derived, and the reason is worth keeping
    because the derivation looked convincing.  ``2 * sigmoid(bias)`` -- the
    gate's value when the latent contributes nothing -- is free to compute here
    and equals ``E[2*sigmoid(b + w.z)]`` to first order whenever ``w.z`` is
    small and centred.  Neither condition survives training.  Measured on
    ``G_8139`` of the first gated run, over 48 real clips: the logits are wide
    enough to pin ~30% of the elements at the ceiling and 6-21% at the floor, so
    the gate is a hard mask whose mean is set by the *shape* of that split, not
    by its centre.  The bias-only estimate read 1.00 at the output-rate stage
    where the true mean is 0.41 -- wrong by 2.4x, and wrong in the direction
    that hid the result, since the real number says the gate closes the
    excitation down at exactly the stage where spectral mirroring lives.

    So the decoder measures it in ``forward`` and parks it in a non-persistent
    buffer, and this function reads that buffer.  It costs two reductions per
    stage and one stacked buffer write per forward, which is the price of a
    series that is a measurement instead of a guess.

    Only the U-Net excitation path is readable this way.  With
    ``chouwagan_excitation_unet`` off the source is folded in additively through
    ``source_gates`` instead of concatenated, and RefineGAN concatenates its
    downsampled excitation straight into a ParallelResBlock, so in both cases
    there is no mixing weight to split and ``{}`` is the honest answer.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    decoder = getattr(model, "dec", None)
    stages = getattr(decoder, "fusion_proj", None)
    # Skips are built highest-rate first and consumed reversed, so reversing
    # here puts index ``i`` on the stage that actually concatenates it.  Reading
    # the width from the decoder rather than inferring it from the channel count
    # is what keeps a changed schedule from attributing part of the main path to
    # the excitation.
    skip_widths = tuple(reversed(getattr(decoder, "exc_skip_channels", ()) or ()))
    if not stages or not skip_widths:
        return {}

    gates = getattr(decoder, "latent_gates", None)
    # Written by the decoder's last forward.  Absent only before the first one,
    # where the initialised buffers already hold the right answer (an open gate
    # that is not yet saturated).
    measured_mean = getattr(decoder, "latent_gate_mean", None)
    measured_saturation = getattr(decoder, "latent_gate_saturation", None)
    if measured_mean is None or measured_saturation is None:
        gates = None

    metrics = {}
    ratios = []
    effective_ratios = []
    for index, stage in enumerate(stages):
        weight = stage.weight
        # cat order is (upsampled, skip), so the main path owns the leading
        # columns.
        skip = int(skip_widths[index])
        main = weight[:, :-skip].norm()
        excitation = weight[:, -skip:].norm()
        ratio = (excitation / main.clamp_min(1e-8)).item()
        metrics[f"Source/inject_ratio_stage_{index}"] = ratio
        ratios.append(ratio)

        if gates is None or index >= len(gates):
            continue
        gate = gates[index]
        with torch.no_grad():
            sensitivity = float(gate.weight.norm())
            gate_mean = float(measured_mean[index])
            saturation = float(measured_saturation[index])
        metrics[f"Source/latent_gate_mean_stage_{index}"] = gate_mean
        metrics[f"Source/latent_gate_saturation_stage_{index}"] = saturation
        metrics[f"Source/latent_gate_sensitivity_stage_{index}"] = sensitivity
        effective = ratio * gate_mean
        metrics[f"Source/inject_ratio_effective_stage_{index}"] = effective
        effective_ratios.append(effective)

    if ratios:
        # The last stage runs at the output rate: it is the one whose excitation
        # share lands directly in the top octave, so it gets a flat alias.
        metrics["Source/inject_ratio_output_stage"] = ratios[-1]
    if effective_ratios:
        metrics["Source/inject_ratio_effective_output_stage"] = effective_ratios[-1]
    return metrics


def _assert_resumable_architecture(net_g, checkpoint_path):
    """Refuse to resume from a checkpoint built for a different architecture.

    The equivalent guard already exists on the *pretrained* path and in
    ``generate_config``; this is the third door into the same room and was the
    one left open.  A checkpoint written before the id existed reports ``None``
    and is treated as a mismatch rather than as "nothing to check", for the same
    reason ``generate_config`` does: the runs that predate the key are exactly
    the ones whose layout has since changed.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    expected = getattr(model, "architecture_id", None)
    if not expected or expected == "vits_gaussian_v1":
        return
    found = torch.load(checkpoint_path, map_location="cpu", weights_only=True).get(
        "architecture_id"
    )
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
        print("[INIT] Starting the training ...")

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
            generator_strict_load = strict_load and not chouwagan_stack_active
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
                print("[INIT] Fine-tune optimizer state reset.")

            #epoch_str += 1
            #global_step = (epoch_str - 1) * len(train_loader)

            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            print(f"[RESUMING] (G) & (D) at global_step: {global_step} and epoch count: {epoch_str - 1}")

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
                print(f"[ ] Loading pretrained (G) '{pretrainG}'")
            checkpoint = torch.load(pretrainG, map_location="cpu", weights_only=True)
            expected_architecture = getattr(
                net_g.module if hasattr(net_g, "module") else net_g,
                "architecture_id",
                None,
            )
            checkpoint_architecture = checkpoint.get("architecture_id")
            # An *absent* id is not a mismatch here.  The resume guard treats it
            # as one on purpose -- a `G_*.pth` in the experiment folder that
            # predates the key is one of this fork's own old runs, and those are
            # exactly the layouts that have since changed.  A pretrain is the
            # opposite case: it comes from upstream, where the key never
            # existed.  Every stock RVC v2 checkpoint carries `iteration` and
            # `learning_rate` and nothing else, the bundled
            # `hifi-gan/f0G{32,40,48}k.pth` included, so this branch rejected
            # the shipped pretrains for the shipped vocoder -- HiFi-GAN
            # fine-tuning could not start at all.
            #
            # It went unnoticed because HiFi-GAN is the only architecture the
            # check can actually reach: the VITS-latent vocoders both report
            # `vits_gaussian_v1`, and that opt-out exists for *precisely* this
            # reason (an Applio checkpoint carries no id either).  HiFi-GAN, the
            # oldest path, was the one left out of it.
            #
            # What verifies an id-less pretrain instead is the load below, which
            # is strict for everything but the ChouwaGAN stack: a state dict
            # from another architecture raises on its missing and unexpected
            # keys.  An id that is present and disagrees is still refused --
            # that is a checkpoint making a claim about itself.
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
            # to other speakers, so ``emb_g`` was sized for *this* dataset and
            # the pretrained rows would not even fit.  Substituting the model's
            # own freshly initialised tensor rather than dropping the key keeps
            # the load strict for the vocoders that load strictly -- a missing
            # key would be an error there, and a silent one is exactly what
            # this branch must not produce.
            if reset_pretrained_embeddings and "emb_g.weight" in state_dict:
                state_dict = substitute_speaker_embeddings(state_dict, net_g)
                if rank == 0:
                    print(
                        "[ ] Pretrained speaker embeddings discarded: "
                        f"starting {state_dict['emb_g.weight'].shape[0]} fresh ones."
                    )

            net_g.load_state_dict(
                state_dict,
                strict=not chouwagan_stack_active,
            )

        # Loading the pretrained Discriminator model
        if pretrainD not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (D) '{pretrainD}'")
            checkpoint = torch.load(pretrainD, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_d.load_state_dict(state_dict, strict=True)

        # Load the models and optionally wrap with DDP
        net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

        # Apply decoder / vocoder layer freezes ( for fine-tuning )
        apply_training_freezes(net_g, rank)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    # Built after both branches so the shadow starts from whatever weights the
    # run is actually beginning with -- resumed, pretrained, or fresh -- rather
    # than from the initialisation ``get_g_model`` handed back.
    ema = None
    # Schedule-free already maintains an average of the trajectory -- that is
    # what its x iterate is -- and the EMA would be built from the extrapolated
    # y iterate, which is not a point the method claims is good.  Stacking a
    # second average on top of the first also lengthens the effective horizon,
    # and a long horizon is precisely what makes the overtrain detector worse
    # (see the EMA decay choice).  So the two are mutually exclusive here.
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
            print(
                f"[EMA] decay {ema.decay}"
                + (
                    f", resumed at {ema.updates} updates."
                    if restored
                    else ", starting from the current weights."
                )
            )

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, ema, resumed_extra_d


def prepare_schedulers(
    optim_g, optim_d,
    use_lr_scheduler, lr_scheduler, exp_decay_gamma,
    total_epoch_count, epoch_str, global_step, train_loader,
    fresh_start=False,
):
    def _horizon_decay(final_ratio, total_units):
        """Multiplier that reaches ``final_ratio`` at the end of the run.

        ``ratio ** (unit / total)`` is plain exponential decay written in terms
        of where it lands rather than of its per-step gamma, so the same config
        gives the same endpoint at any run length.

        The progress term is clamped at 1.0, which gives the schedule a floor:
        extending a run past its configured horizon holds the final LR instead
        of decaying straight through it.  Exponential decay has no floor of its
        own, and at these gammas a run continued well past its planned end
        would otherwise drift into the 1e-6 range unnoticed.
        """
        ratio = min(1.0, max(1e-6, float(final_ratio)))
        total = max(1, int(total_units))

        def scale(unit):
            return ratio ** min(1.0, max(0, unit) / total)

        return scale

    def _horizon_cosine(final_ratio, total_units):
        """Cosine anneal from the starting LR down to ``final_ratio`` of it.

        The stock ``CosineAnnealingLR`` takes ``eta_min`` -- one absolute floor
        shared by every optimizer it is given.  G and D start at different
        rates here, so a shared floor does not just decay them by different
        factors (0.30x and 0.16x at the shipped 3e-5), it drives the ratio
        between them to exactly 1.0 by the end of the run.  That ratio is the
        thing that keeps the discriminator alive, so it has to survive the
        schedule.  Expressing the endpoint as a fraction of each group's own
        base LR is what makes that happen.

        Clamped past the horizon for the same reason as ``_horizon_decay``:
        continuing a run should hold the final LR, and an unclamped cosine
        would turn back upward.
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
            # No announcement here.  What this branch does -- decay to
            # ``lr_final_ratio`` by the end of the run -- is what the settings
            # panel already prints for the scheduler, and the half of the old
            # line that named the superseded ``lr_decay``/``eta_min`` only had
            # to exist because the panel was printing that dead number as though
            # it were live.  It no longer is.
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
        print("[REFERENCE] Using custom reference input from 'logs\\reference\\'")
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

        # ---- Optional ground truth for the preview -------------------------
        # Without it the preview degrades to the generated waveform alone: the
        # three-panel figure needs something to subtract, so a custom reference
        # used to cost exactly the comparison it was set up to make stable.
        # The wav is resampled on load, so it can be at any rate; what has to
        # line up is the *time span*, and it does by construction -- one f0
        # frame is one hop at every configured sample rate, both being 10 ms.
        audio_path = os.path.join(reference_path, "ref_audio.wav")
        if os.path.isfile(audio_path):
            from rvc.lib.utils import load_audio

            wave = load_audio(audio_path, config.data.sample_rate)
            wanted = min_len * config.data.hop_length
            if wave.shape[0] < wanted:
                # Short is survivable -- the figure crops both mels to the
                # frames they share -- but silently comparing less than the
                # reference renders is not, so it is said out loud.
                print(
                    "[REFERENCE] ref_audio.wav is "
                    f"{wave.shape[0] / config.data.sample_rate:.2f}s, short of the "
                    f"{wanted / config.data.sample_rate:.2f}s the features render; "
                    "the preview will compare only the overlap."
                )
            reference_audio = (
                torch.FloatTensor(wave[:wanted]).view(1, 1, -1).to(device)
            )
        else:
            print(
                "[REFERENCE] No ref_audio.wav; the preview will show the "
                "generated audio without the mel comparison."
            )

    else:
        print("[REFERENCE] No custom reference found. Fetching from train_loader.")
        info = next(iter(train_loader))
        # Unpack everything from the loader
        phone, phone_lengths, pitch, pitchf, _, _, reference_audio, _, sid = info

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
        print(f"[REFERENCE] Origin of the ref: {file_name}")
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
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice_g, optimizer_choice_d
    global from_scratch, swap_start_step, swap_completed
    global phase_start_step, phase_step, phase_limit_reached

    if rank == 0:
        configure_logging()

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
        print(f"[INIT] linear warmup: {effective_warmup_steps(train_loader)} steps  -  ({warmup_tag})")

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
        if not chouwagan_stack_active or mel_distance in ("l1", "mae"):
            base, weighted = torch.nn.L1Loss, chouwagan_stack_active
        elif mel_distance == "huber":
            base, weighted = (
                lambda **kw: torch.nn.SmoothL1Loss(beta=mel_huber_beta, **kw)
            ), True
        elif mel_distance in ("mse", "l2"):
            base, weighted = torch.nn.MSELoss, True
        else:
            raise ValueError(
                f"Unknown mel_distance {mel_distance!r}: "
                "expected 'huber', 'l1' or 'mse'."
            )
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
        if rank == 0 and chouwagan_stack_active:
            print(
                f"[INIT] Mel distance: {mel_distance}"
                + (f" (beta={mel_huber_beta})" if mel_distance == "huber" else "")
            )
        if swap_l1_to_ms:
            fn_spectral_loss_ms = MultiScaleMelSpectrogramLoss(
                sample_rate=sample_rate,
                safe_log=chouwagan_stack_active,
                loss_fn=_make_mel_distance(),
            )
    elif spectral_loss == "Multi-Scale Mel Loss":
        # Uses the same configured distance as the single-scale mel loss.  It
        # used to hardcode L1, which silently ignored ``mel_distance``
        # -- selecting this mode quietly discarded the Huber setting.
        #
        # One caveat worth knowing: ``mel_huber_beta`` was tuned
        # against ``wave_to_mel(for_loss=True)``, while this operates on
        # ``log1p(mel * log_scale)``.  The two scales are similar but not
        # identical, so beta is a slightly different L2/L1 crossover here.
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=sample_rate,
            safe_log=chouwagan_stack_active,
            loss_fn=_make_mel_distance(),
        )
        if rank == 0 and chouwagan_stack_active:
            print(
                f"[INIT] Multi-scale mel distance: {mel_distance}"
                + (f" (beta={mel_huber_beta})" if mel_distance == "huber" else "")
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
    # The lazy R1 step is a *second* optimizer iteration on ``optim_d`` inside
    # the same training step, and a GradScaler tracks its state per optimizer
    # per iteration: once ``step(optim_d)`` has been called, a further
    # ``unscale_(optim_d)`` before ``update()`` raises "unscale_() is being
    # called after step()".  It did, on the first R1 step of every FP16 run.
    #
    # A second scaler is the fix rather than an extra ``update()`` because the
    # two backwards genuinely want different loss scales: R1 is a double
    # backward of a squared gradient norm, so its magnitude has no relation to
    # the discriminative loss's, and forcing them to share one scale means
    # whichever overflows first drags the other down with it.
    r1_grad_scaler = (
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
    if r1_grad_scaler is not None:
        r1_scaler_state = resumed_extra_d.get("r1_grad_scaler")
        if r1_scaler_state:
            r1_grad_scaler.load_state_dict(r1_scaler_state)

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
        print(f"[TRAIN] LOSS SWAP: L1 mel -> Multi-Scale mel over {swap_duration_steps} steps (starting step {swap_start_step})")

    # Tensorboard handling
    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400,
            purge_step=global_step + 1
        )
        block_tensorboard_flush_on_exit(writer_eval)

        if global_step != 0:
            print(f"[INIT] TensorBoard writer initialized. Purging logs after step: {global_step}")
        else:
            print(f"[INIT] TensorBoard writer initialized.")

    # from-scratch checker ( disables average loss )
    if finetune_phase:
        from_scratch = False
        if rank == 0:
            print(
                f"[INIT] Fine-tune phase active: max steps={max_steps or 'epoch limit'}, "
                f"starting global step={global_step}."
            )
    elif (pretrainG in ["", "None"] or pretrainD in ["", "None"]) or force_from_scratch:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")
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
    adv_ramp_start = fit_schedule(
        0
        if finetune_phase
        else max(0, int(getattr(config.train, "adaptive_adv_ramp_start", 20000))),
        planned_steps,
        0.2,
    )
    adv_ramp_steps = fit_schedule(
        max(
            0,
            int(
                getattr(config.train, "adaptive_adv_ramp_steps_finetune", 2000)
                if finetune_phase
                else getattr(config.train, "adaptive_adv_ramp_steps", 80000)
            ),
        ),
        planned_steps,
        0.5 if not finetune_phase else 0.25,
    )
    # Read off the built discriminator rather than off the config: the config
    # key only *requests* SAN heads, and a discriminator that does not implement
    # them would leave the governor's floor describing a loss form nothing
    # computes.
    san_active = bool(
        getattr(net_d.module if hasattr(net_d, "module") else net_d, "supports_san", False)
    )
    san_direction_weight = max(
        0.0,
        min(1.0, float(getattr(config.train, "san_direction_weight", 0.25))),
    )
    adversarial_ceiling_governor = _AdversarialCeilingGovernor(
        start_step=adv_ramp_start,
        ramp_steps=adv_ramp_steps,
        ceiling_start=1.0,
        ceiling_end=max(
            max(0.0, float(getattr(config.train, "adaptive_adv_min", 0.01))),
            float(getattr(config.train, "adaptive_adv_max", 8.0)),
        ),
        floor_loss=_constant_discriminator_loss(san_direction_weight, san_active),
        collapse_ceiling=max(
            0.0, float(getattr(config.train, "adaptive_adv_min", 0.01))
        ),
    )

    # Built here rather than in ``training_loop``, which is re-entered once per
    # epoch: a controller rebuilt there would throw away its strength and both
    # EMAs every 8141 steps and spend the next epoch re-earning them.
    r1_controller = _R1StrengthController(
        branch_names=getattr(
            net_d.module if hasattr(net_d, "module") else net_d, "branch_names", ()
        ),
        target_ratio=float(getattr(config.train, "r1_target_ratio", 1.0)),
        maximum=float(getattr(config.train, "r1_max_scale", 10000.0)),
        minimum=float(getattr(config.train, "r1_min_scale", 1e-4)),
    )
    if r1_controller.load_state_dict(resumed_extra_d.get("r1_controller")):
        info(
            f"Restored the R1 strength controller over {len(r1_controller.names)} "
            f"branches (mean scale {r1_controller.scale:.3f}, "
            f"mean share {r1_controller.ratio:.3f}).",
            tag="[RESUME]",
        )

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
            print(
                f"[HOLDOUT] Interval {interval} -> {fitted} steps: at {interval} "
                f"this {planned_steps}-step run would fit "
                f"{planned_steps // interval} evaluations against a patience of "
                f"{overtrain_monitor.patience}, and the detector could never fire."
            )
        interval = fitted
        print(
            f"[HOLDOUT] Evaluating every {interval} steps, "
            f"patience {overtrain_monitor.patience} evaluations"
            + (f" ({evaluations} evaluations planned)." if evaluations else ".")
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
            adversarial_ceiling_governor=adversarial_ceiling_governor,
            r1_controller=r1_controller,
            grad_scaler=grad_scaler,
            r1_grad_scaler=r1_grad_scaler,
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
    adversarial_ceiling_governor=None,
    r1_controller=None,
    grad_scaler=None,
    r1_grad_scaler=None,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        config (object): Configuration object containing training parameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, net_d].
        train_loader: training dataloader.
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        device (torch.device): The device to use for training (CPU or GPU).
        reference (list): Contains reference sample. Either custom or from train loader.
        reference_audio (torch.Tensor): Original waveform for the evaluation reference, when available.
        fn_spectral_loss: spectral loss;  can be l1, multi-scale or ms-stft.
        fn_spectral_loss2: 2nd spectral loss
    """
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
    if chouwagan_stack_active:
        # Waveform-domain reconstruction terms, logged unweighted so the series
        # stay readable when their weights change.
        for key in ("loss_envelope", "loss_rms", "loss_peak", "loss_waveform"):
            avg_rolling_cache[key] = deque(maxlen=rolling_loss_steps)
    avg_rolling_cache["loss_kl"] = deque(maxlen=rolling_loss_steps)
    kl_std_cache = deque(maxlen=rolling_loss_steps)
    kl_mean_cache = deque(maxlen=rolling_loss_steps)
    kl_active_cache = deque(maxlen=rolling_loss_steps)
    last_kl_per_dim = None

    r1_interval = max(1, int(getattr(config.train, "r1_interval", 16)))
    r1_gamma = float(getattr(config.train, "r1_gamma", 1.0))
    # Only fires once every ``r1_interval`` steps, so its window is scaled down
    # to cover the same wall-clock span as the per-step series.
    avg_rolling_cache["grad_norm_d_r1"] = deque(
        maxlen=max(2, rolling_loss_steps // r1_interval)
    )

    if r1_controller is None:
        # Inert stand-in rather than a guard at each call site: a zero target
        # makes ``active`` false, every ``scale_for`` exactly 1.0 and
        # ``observe_r1`` a no-op, so the R1 step behaves as it did before the
        # controller existed.
        r1_controller = _R1StrengthController(target_ratio=0.0)

    # Per-head discriminator losses.  Kept out of ``avg_rolling_cache`` because
    # that dict is keyed by scalar name and auto-namespaced by prefix, while
    # this is one ``(heads, 2)`` tensor per step -- stacking it whole means the
    # 50-step window costs a single mean and a single device transfer instead
    # of 2 * heads of each.
    disc_branch_cache = deque(maxlen=rolling_loss_steps)
    # One 0/1 per step, so the logged rate is "how much of the recent window did
    # FP16 throw away" rather than a lifetime average that a bad first epoch
    # would dominate forever.
    amp_skip_cache = deque(maxlen=rolling_loss_steps)
    disc_branch_names = ()
    if chouwagan_stack_active:
        disc_branch_names = tuple(
            getattr(net_d.module if hasattr(net_d, "module") else net_d, "branch_names", ())
        )

    diagnostics_interval = max(
        1,
        int(getattr(config.train, "diagnostics_interval", 256)),
    )

    r1_segment_size = max(
        1,
        int(getattr(config.train, "r1_segment_size", config.train.segment_size)),
    )
    # The generator's gradient norm is dominated by ``dec.conv_post`` (a 40->1
    # output conv that sums the waveform-domain loss gradient over B*T samples),
    # so its natural operating point sits in the hundreds.  The clip is a spike
    # guard, not a per-step rescaler -- keep the ceiling well above that range.
    global_clip_g = max(
        0.0,
        float(getattr(config.train, "global_clip_norm_g", 500.0)),
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
    adv_warmup_steps = 0 if finetune_phase else max(
        0, int(getattr(config.train, "adv_warmup_steps", 2000))
    )
    kl_active_threshold = max(
        0.0,
        float(getattr(config.train, "kl_active_threshold", 0.01)),
    )
    adaptive_adv_interval = max(
        1,
        int(getattr(config.train, "adaptive_adv_interval", 8)),
    )

    # ---- Adversarial balance --------------------------------------------
    # The adaptive weight equalises the adversarial and reconstruction
    # gradients measured on the decoder's last layer, scaled by the balance
    # target.  The ceiling has to sit well above 1.0 or it is a cap rather than
    # a balance: with c_mel at 45 the reconstruction gradient runs one to two
    # orders of magnitude above the adversarial one, so the rule asks for tens.
    # Adversarial pressure is the only term that teaches high-frequency detail
    # and aperiodic texture -- a mel distance is nearly blind to both -- so
    # capping it there caps the ceiling on final quality.
    adaptive_adv_balance = max(
        0.0,
        float(getattr(config.train, "adv_balance_target", 0.5)),
    )
    adaptive_adv_min = max(
        0.0,
        float(getattr(config.train, "adaptive_adv_min", 0.01)),
    )
    # The same rule for the other discriminator-derived loss -- see
    # ``_adaptive_feature_match_weight`` for the measurements that motivate it.
    # 0.0 disables the governor and restores the fixed ``fm_weight``.
    adaptive_fm_balance = max(
        0.0,
        float(getattr(config.train, "fm_balance_target", 0.33)),
    )
    adaptive_fm_min = max(
        0.0,
        float(getattr(config.train, "fm_balance_min", 0.1)),
    )
    # The ceiling on the feature-matching weight.  1.0 -- the historical value,
    # and still the default -- means the governor can only ever *reduce*
    # ``fm_weight``, because the failure it was written for was overgrowth.
    # That makes ``fm_balance_target`` unreachable from below: a run whose
    # feature matching sits under target has the governor pinned at the ceiling
    # asking for more and unable to take it.  Raise this to let the rule work
    # in both directions.
    adaptive_fm_max = max(
        adaptive_fm_min,
        float(getattr(config.train, "fm_balance_max", 1.0)),
    )
    # 8.0 is a judgement call, not a derived constant: it is roughly a quarter
    # of the balance the rule currently requests, an ~8x increase on the old
    # ceiling, and bounded so a collapsing discriminator cannot run the weight
    # away.  Raise it if the high-frequency deficit persists once the ramp has
    # finished; lower it if loss_disc_real starts falling well below 1.0.
    adaptive_adv_max = max(
        adaptive_adv_min,
        float(getattr(config.train, "adaptive_adv_max", 8.0)),
    )
    adaptive_adv_ramp_start = max(
        0,
        int(getattr(config.train, "adaptive_adv_ramp_start", 20000)),
    )
    adaptive_adv_ramp_steps = max(
        0,
        int(getattr(config.train, "adaptive_adv_ramp_steps", 80000)),
    )
    # A fine-tune never reaches the pretraining ramp's horizon -- a few thousand
    # steps is typical -- so it gets its own, short one.  It still gets a ramp
    # rather than the full ceiling at once: the loaded G and D were equilibrated
    # under whatever weight trained them, and an 8x step change in the objective
    # on the first batch is how a fine-tune picks up artefacts it never had.
    adaptive_adv_ramp_steps_finetune = max(
        0,
        int(getattr(config.train, "adaptive_adv_ramp_steps_finetune", 2000)),
    )

    # Feature matching is the other channel that carries texture, and it was
    # hardcoded to 1.0 while being normalised by branch count -- roughly half
    # of HiFi-GAN's convention of 2.0 unnormalised, on a term already three
    # orders of magnitude below the spectral loss.
    fm_weight = max(
        0.0,
        float(getattr(config.train, "fm_weight", 2.0)),
    )

    # Only read by the "Hybrid L1" spectral loss.  Left at 1.0 so the option
    # behaves exactly as before unless it is raised deliberately.
    ms_stft_weight = max(
        0.0,
        float(getattr(config.train, "ms_stft_weight", 1.0)),
    )

    # ---- Waveform-domain reconstruction terms ---------------------------
    # The compressed-mel loss constrains magnitude per frame but says nothing
    # about the local level envelope, which is where a bounded output head
    # tends to drift.  Both terms below are relative (log ratios / max-pool
    # differences), so neither assumes a particular dataset normalisation.
    envelope_loss_weight = max(
        0.0, float(getattr(config.train, "envelope_loss_weight", 3.0))
    )
    envelope_kernel = max(
        2, int(getattr(config.train, "envelope_kernel", 100))
    )
    envelope_stride = max(
        1, int(getattr(config.train, "envelope_stride", 50))
    )
    # Amplitude below which the companded envelope stops resolving detail.
    # 1e-3 is about -60 dBFS, under any decay tail worth reproducing.
    envelope_floor = max(
        1e-8, float(getattr(config.train, "envelope_floor", 1e-3))
    )
    rms_loss_weight = max(
        0.0, float(getattr(config.train, "rms_loss_weight", 5.0))
    )
    rms_window_size = max(
        16, int(getattr(config.train, "rms_window", 1024))
    )
    rms_hop_size = max(1, int(getattr(config.train, "rms_hop", 256)))
    # One-sided and absolute: only useful when the dataset is peak-normalised
    # below the head's threshold.  Off by default.
    peak_headroom_weight = max(
        0.0, float(getattr(config.train, "peak_headroom_weight", 0.0))
    )
    peak_headroom_threshold = float(
        getattr(config.train, "peak_headroom_threshold", 0.85)
    )

    # Counts refreshes whose gradient probe had no representable measurement.
    # Logged, because the failure it guards against was invisible from the
    # losses: every component reads finite and only the weights are NaN.
    adaptive_probe_failures = 0
    cached_adaptive_adv = None
    cached_rec_grad = None
    cached_adv_grad = None
    cached_adv_requested = None
    cached_adaptive_fm = None
    cached_fm_grad = None
    cached_fm_requested = None

    with progress_task(
        len(train_loader),
        f"Epoch {epoch}/{total_epoch_count}",
        initial=start_batch_idx,
        training=True,
        disable=rank != 0,
    ) as (progress, task_id):
        progress_metrics = ""
        metrics_update_interval = max(1, min(rolling_loss_steps, 8))
        for batch_idx, info in data_iterator:

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

            if chouwagan_stack_active and global_clip_g > 0:
                grad_clip_value_g = min(grad_clip_value_g, global_clip_g)
                grad_clip_value_d = min(grad_clip_value_d, global_clip_g)


            # Device handling
            if device.type == "cuda":
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]

            # Batch unpacking
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = info

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
            main_real_spectrograms = None
            if chouwagan_stack_active:
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                with torch.no_grad(), autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    main_real_spectrograms = (
                        discriminator_model.prepare_spectrograms(y)
                    )

            d_updates = [(y, y_hat.detach(), main_real_spectrograms)]

            # The lazy R1 step runs its own backward and its own optimizer step
            # with a penalty scaled by ``r1_gamma * r1_interval``.  Averaging its
            # gradient norm into the adversarial one hides which of the two is
            # actually growing, so they are tracked as separate series.
            _loss_disc_acc, _loss_disc_real_acc, _loss_disc_fake_acc, _grad_norm_d_acc = [], [], [], []
            _disc_branch_acc = []
            grad_norm_d_r1 = None

            for y_d_real, y_d_fake, real_spectrograms in d_updates:
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    if chouwagan_stack_active:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real,
                            y_d_fake,
                            real_spectrograms=real_spectrograms,
                            pair_batches=True,
                            # Only the *discriminator* update splits the heads:
                            # the direction output exists to train the
                            # projection against this loss, and a discriminator
                            # without SAN heads ignores the flag.
                            san_training=True,
                        )
                    else:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real, y_d_fake
                        )

                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    disc_loss_parts = discriminator_loss(
                        y_d_hat_r,
                        y_d_hat_g,
                        san_direction_weight=san_direction_weight,
                        normalize=chouwagan_stack_active,
                        per_branch=bool(disc_branch_names),
                    )
                    loss_disc, loss_disc_real, loss_disc_fake = disc_loss_parts[:3]
                    if disc_branch_names:
                        _disc_branch_acc.append(disc_loss_parts[3])

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
                _normalize_san_weights(net_d)

                # Temp accumulation
                _loss_disc_acc.append(loss_disc.detach())
                _loss_disc_real_acc.append(loss_disc_real.detach())
                _loss_disc_fake_acc.append(loss_disc_fake.detach())
                if grad_norm_d is not None:
                    _grad_norm_d_acc.append(grad_norm_d)
                # The reference each branch's R1 share is measured against.  Read
                # here because ``optim_d.step()`` leaves the gradients in place --
                # only the next iteration's ``zero_grad`` clears them -- and they
                # are the *discriminative* ones, so the penalty cannot move its
                # own denominator.
                #
                # Sampled at ``r1_interval`` rather than every step: nine extra
                # norm reductions per sample is ~0.3% of a 7127-launch step at
                # this cadence, and every branch is measured on each sample, so
                # each EMA still advances 16x more often than the branch's own R1
                # fires.
                if r1_controller.active and global_step % r1_interval == 0:
                    for name, norm in zip(
                        r1_controller.names, _branch_discriminative_norms(net_d)
                    ):
                        r1_controller.observe_discriminator(name, norm)

            if (
                chouwagan_stack_active
                and r1_gamma > 0
                and global_step % r1_interval == 0
            ):
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                r1_event = max(0, global_step // r1_interval - 1)
                r1_branch = r1_event % discriminator_model.num_branches
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    r1_penalty = _lazy_r1_penalty(
                        net_d,
                        y,
                        r1_branch,
                        r1_segment_size,
                    )
                    loss_r1 = r1_penalty * (
                        r1_gamma * r1_interval * 0.5
                        * r1_controller.scale_for(r1_branch)
                    )

                optim_d.zero_grad(set_to_none=True)
                if r1_grad_scaler is not None:
                    r1_grad_scaler.scale(loss_r1).backward()
                    r1_grad_scaler.unscale_(optim_d)
                else:
                    loss_r1.backward()
                grad_norm_d_r1 = _clip_or_sample_grad_norm(
                    net_d.parameters(),
                    grad_clip_value_d,
                    global_step,
                    metrics_update_interval,
                )
                if grad_norm_d_r1 is not None:
                    r1_controller.observe_r1(r1_branch, float(grad_norm_d_r1))
                if r1_grad_scaler is not None:
                    r1_grad_scaler.step(optim_d)
                    # Closed here rather than with the main scaler at the end of
                    # the step: this is a complete iteration of its own, and it
                    # only runs every ``r1_interval`` steps.
                    r1_grad_scaler.update()
                else:
                    optim_d.step()
                _normalize_san_weights(net_d)

            # Stack + mean
            loss_disc = torch.stack(_loss_disc_acc).mean()
            loss_disc_real = torch.stack(_loss_disc_real_acc).mean()
            loss_disc_fake = torch.stack(_loss_disc_fake_acc).mean()
            grad_norm_d = (
                torch.stack(_grad_norm_d_acc).mean()
                if _grad_norm_d_acc
                else None
            )
            if _disc_branch_acc:
                disc_branch_cache.append(torch.stack(_disc_branch_acc).mean(dim=0))

            optim_d.zero_grad(set_to_none=True)

            # Run discriminator on generated output
            discriminator_parameter_states = None
            branchwise_generator = False
            branch_terms = None
            if chouwagan_stack_active:
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                discriminator_parameter_states = [
                    parameter.requires_grad
                    for parameter in discriminator_model.parameters()
                ]
                for parameter in discriminator_model.parameters():
                    parameter.requires_grad_(False)

                # ``d_generator_branchwise``, which is a memory decision and
                # deliberately not ``d_branchwise``: the R1 layout is a
                # separate question.  Off, this is the joint forward it always
                # was.
                branchwise_generator = bool(
                    getattr(discriminator_model, "generator_branchwise", False)
                )
                if branchwise_generator:
                    branch_terms = _branchwise_generator_terms(
                        discriminator_model,
                        y,
                        y_hat,
                        main_real_spectrograms,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        loss_scale=(
                            grad_scaler.get_scale()
                            if grad_scaler is not None
                            else 1.0
                        ),
                        san_direction_weight=san_direction_weight,
                        use_softplus=san_active,
                    )
                else:
                    with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                        with torch.no_grad():
                            _, fmap_r = discriminator_model._forward_audio(
                                y, main_real_spectrograms
                            )
                        y_d_hat_g, fmap_g = discriminator_model._forward_audio(y_hat)
            else:
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)


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
                        for_loss=chouwagan_stack_active,
                    )
                    y_hat_mel = wave_to_mel(
                        config, y_hat, num_mels=None,
                        for_loss=chouwagan_stack_active,
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
                            print(f"[TRAIN] LOSS SWAP complete at step {global_step} - now using Multi-Scale mel loss")
                    else:
                        loss_spectral = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                elif spectral_loss == "Multi-Scale Mel Loss":
                    loss_spectral = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0 # * 15
                elif spectral_loss == "Hybrid L1":
                    # L1 Mel
                    y_mel = wave_to_mel(
                        config, y, num_mels=None,
                        for_loss=chouwagan_stack_active,
                    )
                    y_hat_mel = wave_to_mel(
                        config, y_hat, num_mels=None,
                        for_loss=chouwagan_stack_active,
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

                loss_envelope = torch.zeros((), device=y.device)
                loss_rms = torch.zeros((), device=y.device)
                loss_peak = torch.zeros((), device=y.device)
                if chouwagan_stack_active:
                    if envelope_loss_weight > 0.0:
                        loss_envelope = envelope_loss(
                            y.float(),
                            y_hat.float(),
                            kernel_size=envelope_kernel,
                            stride=envelope_stride,
                            floor=envelope_floor,
                        )
                    if rms_loss_weight > 0.0:
                        loss_rms = local_log_rms_loss(
                            y,
                            y_hat,
                            window_size=rms_window_size,
                            hop_size=rms_hop_size,
                        )
                    if peak_headroom_weight > 0.0:
                        loss_peak = peak_headroom_loss(
                            y_hat, threshold=peak_headroom_threshold
                        )
                loss_waveform = (
                    envelope_loss_weight * loss_envelope
                    + rms_loss_weight * loss_rms
                    + peak_headroom_weight * loss_peak
                )

                # Feature Matching loss.  Branchwise, both this and the
                # adversarial term below were already summed over branches
                # inside the loop that differentiated them; only the configured
                # weight is still outstanding.
                if branchwise_generator:
                    loss_fm = branch_terms["loss_fm"] * fm_weight
                else:
                    loss_fm = feature_loss(
                        fmap_r,
                        fmap_g,
                        normalize=chouwagan_stack_active,
                    ) * (fm_weight if chouwagan_stack_active else 2.0)

                # Generator loss.  ``y_d_hat_g`` comes from the *generator*
                # update's forward, which never sets ``san_training``, so these
                # are plain logits and ``san_direction_weight`` is inert -- the
                # direction output is the discriminator's business alone.
                #
                # The squared-softplus surrogate is SAN's, and it is one-sided:
                # LSGAN's ``(1 - d) ** 2`` also punishes the generator for
                # scoring *above* the real target, which is not a defect worth
                # a gradient.
                if branchwise_generator:
                    loss_adv = branch_terms["loss_adv"]
                else:
                    loss_adv = generator_loss(
                        y_d_hat_g,
                        normalize=chouwagan_stack_active,
                        san_direction_weight=san_direction_weight,
                        use_softplus=san_active,
                    )

                loss_kl = kl_loss(
                    z_p,
                    logs_q,
                    m_p,
                    logs_p,
                    z_mask,
                ) * config.train.c_kl

                # KL diagnostic: per-dimension raw divergence.
                with torch.no_grad():
                    raw_kl = logs_p - logs_q - 0.5 + 0.5 * (
                        (z_p - m_p) ** 2
                    ) * torch.exp(-2 * logs_p)
                    raw_kl_per_dim = (raw_kl * z_mask).sum(dim=(0, 2)) / z_mask.sum(
                        dim=(0, 2)
                    ).clamp(min=1)
                    diagnostic_kl = raw_kl_per_dim.clamp_min(0.0)
                    kl_std_cache.append(diagnostic_kl.std().item())
                    kl_mean_cache.append(diagnostic_kl.mean().item())
                    kl_active_cache.append(
                        (diagnostic_kl > kl_active_threshold)
                        .float()
                        .mean()
                        .item()
                    )
                    last_kl_per_dim = diagnostic_kl

                adaptive_adv = torch.ones((), device=y.device)
                adaptive_fm = torch.ones((), device=y.device)
                last_layer_rec_grad = torch.zeros((), device=y.device)
                last_layer_adv_grad = torch.zeros((), device=y.device)
                last_layer_fm_grad = torch.zeros((), device=y.device)
                adaptive_adv_requested = torch.zeros((), device=y.device)
                adaptive_fm_requested = torch.zeros((), device=y.device)
                # Gated on discriminator health rather than on the step count
                # alone -- see ``_AdversarialCeilingGovernor``.  ``loss_disc``
                # is this step's, since the discriminator update ran above.
                adaptive_adv_ceiling = adversarial_ceiling_governor.update(
                    phase_step if finetune_phase else global_step,
                    loss_disc.item() if chouwagan_stack_active else None,
                )
                if chouwagan_stack_active:
                    # Each refresh costs two extra autograd traversals (one of
                    # them through the whole discriminator), measured at +16% of
                    # step time.  The weight is a slowly-varying scalar that is
                    # clamped to [0.01, 1.0], so recomputing it periodically and
                    # reusing it in between buys that back.
                    if (
                        cached_adaptive_adv is None
                        or global_step % adaptive_adv_interval == 0
                    ):
                        last_layer = _last_layer_parameter(
                            _decoder_output_layer(model_g.dec)
                        )
                        # Branchwise, the GAN losses are detached scalars: their
                        # gradient at the last layer comes from the waveform
                        # gradients the branch loop accumulated, unscaled first
                        # because the balance rule is a *ratio* against a
                        # reconstruction gradient that was never scaled.
                        branch_adv_parameter_grad = None
                        branch_fm_parameter_grad = None
                        if branchwise_generator:
                            inverse_scale = 1.0 / branch_terms["loss_scale"]
                            branch_adv_parameter_grad = _wave_parameter_gradient(
                                y_hat,
                                branch_terms["adv_wave_grad"] * inverse_scale,
                                last_layer,
                            )
                            branch_fm_parameter_grad = _wave_parameter_gradient(
                                y_hat,
                                branch_terms["fm_wave_grad"]
                                * (inverse_scale * fm_weight),
                                last_layer,
                            )
                        (
                            probe_adaptive_adv,
                            probe_rec_grad,
                            probe_adv_grad,
                            probe_adv_requested,
                        ) = _adaptive_adversarial_weight(
                            # The waveform terms are part of the reconstruction
                            # objective, so the GAN balance has to see them too.
                            loss_spectral + loss_waveform,
                            loss_adv,
                            last_layer,
                            balance_target=adaptive_adv_balance,
                            minimum=adaptive_adv_min,
                            maximum=adaptive_adv_ceiling,
                            adversarial_gradient=branch_adv_parameter_grad,
                        )
                        # Shares the reconstruction gradient that call just
                        # measured, so governing the second half of the GAN
                        # objective costs one traversal, not three.
                        probe_adaptive_fm = None
                        probe_fm_grad = None
                        probe_fm_requested = None
                        if adaptive_fm_balance > 0.0:
                            (
                                probe_adaptive_fm,
                                probe_fm_grad,
                                probe_fm_requested,
                            ) = _adaptive_feature_match_weight(
                                loss_fm,
                                last_layer,
                                probe_rec_grad,
                                balance_target=adaptive_fm_balance,
                                minimum=adaptive_fm_min,
                                maximum=adaptive_fm_max,
                                feature_gradient=branch_fm_parameter_grad,
                            )
                        # ``_probe_gradient`` returns its last attempt rather
                        # than raising, so a term with no representable
                        # measurement still arrives here -- and a weight is not
                        # one bad step, it is reused for
                        # ``adaptive_adv_interval`` steps and multiplied into
                        # ``loss_gen_total``.  One non-finite value reaching
                        # the cache is a run that never takes another generator
                        # step: the loss is NaN, every step is skipped, and no
                        # amount of scaler backoff recovers it because the probe
                        # does not go through the scaler.  So the cache only
                        # ever takes finite values, and a failed probe keeps the
                        # last good ones instead.
                        probe_finite = bool(
                            torch.isfinite(
                                torch.stack(
                                    [
                                        value.detach().float().reshape(())
                                        for value in (
                                            probe_adaptive_adv,
                                            probe_rec_grad,
                                            probe_adv_grad,
                                            probe_adv_requested,
                                            probe_adaptive_fm,
                                            probe_fm_grad,
                                            probe_fm_requested,
                                        )
                                        if value is not None
                                    ]
                                )
                            )
                            .all()
                            .item()
                        )
                        if probe_finite:
                            cached_adaptive_adv = probe_adaptive_adv
                            cached_rec_grad = probe_rec_grad
                            cached_adv_grad = probe_adv_grad
                            cached_adv_requested = probe_adv_requested
                            if probe_adaptive_fm is not None:
                                cached_adaptive_fm = probe_adaptive_fm
                                cached_fm_grad = probe_fm_grad
                                cached_fm_requested = probe_fm_requested
                        else:
                            adaptive_probe_failures += 1
                            if cached_adaptive_adv is None:
                                # Nothing good to fall back to yet.  The floor
                                # is the conservative end of the rule's own
                                # range, so the step still happens and the next
                                # refresh can correct it.
                                zero_probe = loss_adv.detach().float().new_zeros(())
                                cached_adaptive_adv = zero_probe + adaptive_adv_min
                                cached_rec_grad = zero_probe
                                cached_adv_grad = zero_probe
                                cached_adv_requested = zero_probe
                            if (
                                adaptive_fm_balance > 0.0
                                and cached_adaptive_fm is None
                            ):
                                zero_probe = loss_fm.detach().float().new_zeros(())
                                cached_adaptive_fm = zero_probe + adaptive_fm_max
                                cached_fm_grad = zero_probe
                                cached_fm_requested = zero_probe
                    # The cached weight was clamped against the ceiling in force
                    # when it was computed; re-clamp so a ceiling that moved in
                    # between takes effect on the very next step.
                    adaptive_adv = cached_adaptive_adv.clamp(
                        min=adaptive_adv_min, max=adaptive_adv_ceiling
                    )
                    last_layer_rec_grad = cached_rec_grad
                    last_layer_adv_grad = cached_adv_grad
                    adaptive_adv_requested = cached_adv_requested
                    if cached_adaptive_fm is not None:
                        adaptive_fm = cached_adaptive_fm
                        last_layer_fm_grad = cached_fm_grad
                        adaptive_fm_requested = cached_fm_requested
                    if adv_warmup_steps:
                        adv_progress = min(1.0, global_step / adv_warmup_steps)
                    else:
                        adv_progress = 1.0
                    gan_weight = adv_progress
                else:
                    gan_weight = 1.0

                loss_core = (
                    loss_spectral
                    + loss_waveform
                    + loss_kl
                )
                loss_gan = (
                    adaptive_adv * loss_adv + adaptive_fm * loss_fm
                ) * gan_weight
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
            if (
                chouwagan_stack_active
                and rank == 0
                and global_step % grad_source_probe_interval == 0
            ):
                decoder_parameters = _decoder_parameters(net_g)
                grad_source_losses = {"spectral": loss_spectral}
                if loss_waveform.requires_grad:
                    grad_source_losses["waveform"] = loss_waveform
                # The two GAN terms are probed from a loss graph on the joint
                # path and from their waveform gradients on the branchwise one.
                # Both express the same quantity: the wave series *is* the norm
                # of that gradient, and the decoder series is it propagated one
                # step further back.
                grad_source_waves = {}
                if branchwise_generator:
                    inverse_scale = 1.0 / branch_terms["loss_scale"]
                    grad_source_waves["adv"] = (
                        branch_terms["adv_wave_grad"] * (inverse_scale * gan_weight)
                    )
                    grad_source_waves["fm"] = branch_terms["fm_wave_grad"] * (
                        inverse_scale * fm_weight * float(adaptive_fm) * gan_weight
                    )
                else:
                    grad_source_losses["adv"] = loss_adv * gan_weight
                    # Weighted, unlike ``adv``: the fm weight is now governed,
                    # so the unweighted probe would report a term the optimizer
                    # never sees and hide the correction working.
                    grad_source_losses["fm"] = adaptive_fm * loss_fm * gan_weight
                for name, source_loss in grad_source_losses.items():
                    writer.add_scalar(
                        f"Grad_Source/decoder_{name}",
                        _gradient_norm(source_loss, decoder_parameters).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f"Grad_Source/wave_{name}",
                        _tensor_gradient_norm(source_loss, y_hat).item(),
                        global_step,
                    )
                for name, wave_gradient in grad_source_waves.items():
                    decoder_gradients = torch.autograd.grad(
                        y_hat,
                        decoder_parameters,
                        grad_outputs=wave_gradient,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    terms = [
                        gradient.detach().float().square().sum()
                        for gradient in decoder_gradients
                        if gradient is not None
                    ]
                    writer.add_scalar(
                        f"Grad_Source/decoder_{name}",
                        torch.sqrt(torch.stack(terms).sum()).item() if terms else 0.0,
                        global_step,
                    )
                    writer.add_scalar(
                        f"Grad_Source/wave_{name}",
                        wave_gradient.detach().float().norm().item(),
                        global_step,
                    )
            if chouwagan_stack_active:
                # Branchwise, ``loss_gan`` is a number rather than a graph; the
                # gradient it stands for was accumulated in wave space and is
                # already carrying the AMP scale, exactly like the scaled
                # scalar the joint path hands over.
                gan_wave_gradient = None
                if branchwise_generator:
                    gan_wave_gradient = (
                        branch_terms["adv_wave_grad"] * float(adaptive_adv)
                        + branch_terms["fm_wave_grad"] * (float(adaptive_fm) * fm_weight)
                    ) * gan_weight
                if grad_scaler is not None:
                    grad_scaler.scale(loss_core).backward(retain_graph=True)
                    if branchwise_generator:
                        _add_decoder_only_wave_gradients(
                            y_hat, gan_wave_gradient, net_g
                        )
                    else:
                        _add_decoder_only_gradients(
                            grad_scaler.scale(loss_gan), net_g
                        )
                    grad_scaler.unscale_(optim_g)
                else:
                    loss_core.backward(retain_graph=True)
                    if branchwise_generator:
                        _add_decoder_only_wave_gradients(
                            y_hat, gan_wave_gradient, net_g
                        )
                    else:
                        _add_decoder_only_gradients(loss_gan, net_g)
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

                if rank == 0 and global_step % 50 == 0:
                    writer.add_scalar(
                        "GAN/adaptive_adv_weight",
                        adaptive_adv.item(),
                        global_step,
                    )
                    # The applied weight alone cannot show that the ceiling is
                    # binding: a saturated weight logs as a flat line, which
                    # reads exactly like a healthy constant.  These three make
                    # the difference visible -- when `requested` sits far above
                    # `ceiling`, the balance rule is being overruled, and
                    # `saturation` is the fraction of it that survives.
                    if chouwagan_stack_active:
                        writer.add_scalar(
                            "GAN/adaptive_adv_requested",
                            adaptive_adv_requested.item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "GAN/adaptive_adv_ceiling",
                            adaptive_adv_ceiling,
                            global_step,
                        )
                        # Flat at zero on a healthy run.  Anything else means
                        # the balance rule is running on a stale weight because
                        # its gradient probe had no representable measurement --
                        # which under FP16 used to be a silent NaN instead.
                        writer.add_scalar(
                            "GAN/adaptive_probe_failures",
                            float(adaptive_probe_failures),
                            global_step,
                        )
                        writer.add_scalar(
                            "GAN/adaptive_adv_saturation",
                            min(
                                1.0,
                                adaptive_adv.item()
                                / max(1e-6, adaptive_adv_requested.item()),
                            ),
                            global_step,
                        )
                    if r1_controller.active:
                        # The share is the controlled variable and the scale is
                        # the actuator.  Logging only the scale would leave a
                        # controller that is failing to reach its target
                        # indistinguishable from one that has reached it.
                        #
                        # Per branch as well as in aggregate: the branches are
                        # what is being controlled, and the mean is exactly the
                        # view that hid the 9.9x spread the per-branch
                        # controller exists to remove.  A branch pinned at
                        # ``minimum`` has lost authority and only its own series
                        # shows it.
                        writer.add_scalar(
                            "GAN/r1_scale", r1_controller.scale, global_step
                        )
                        writer.add_scalar(
                            "GAN/r1_to_disc_ratio", r1_controller.ratio, global_step
                        )
                        for _name in r1_controller.names:
                            writer.add_scalar(
                                f"R1_Branch/scale_{_name}",
                                r1_controller.scale_for(_name),
                                global_step,
                            )
                            writer.add_scalar(
                                f"R1_Branch/ratio_{_name}",
                                r1_controller.ratio_for(_name),
                                global_step,
                            )
                        for _name, _share in r1_controller.newly_saturated():
                            warning(
                                f"'{_name}' has been pinned at the R1 ceiling "
                                f"({r1_controller.maximum:g}) with a share of "
                                f"{_share:.3f} against a target of "
                                f"{r1_controller.target_ratio:.2f}. The controller "
                                f"has no authority left over this branch; raise "
                                f"r1_max_scale.",
                                tag="[R1]",
                            )
                        for _name, _share in r1_controller.newly_floored():
                            warning(
                                f"'{_name}' has been pinned at the R1 floor "
                                f"({r1_controller.minimum:g}) with a share of "
                                f"{_share:.3f} against a target of "
                                f"{r1_controller.target_ratio:.2f}. The branch is "
                                f"over-regularised and the controller cannot ease "
                                f"it further; lower r1_min_scale.",
                                tag="[R1]",
                            )
                    writer.add_scalar(
                        "GAN/last_layer_rec_grad",
                        last_layer_rec_grad.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "GAN/last_layer_adv_grad",
                        last_layer_adv_grad.item(),
                        global_step,
                    )
                    # The quantity being held at ``adv_balance_target``,
                    # on the same terms as ``GAN/fm_to_rec_ratio`` below: *after*
                    # the adaptive weight, so it is what the optimizer receives
                    # rather than what the adversarial term asked for.  The two
                    # sit side by side and are read as comparable, and until now
                    # they were not -- this one was the raw gradient ratio, which
                    # reads as tens against the other's fractions and makes a
                    # balance rule sitting exactly on target look broken.
                    #
                    # Nothing is lost by weighting it: the raw ratio is
                    # ``last_layer_adv_grad / last_layer_rec_grad``, and both are
                    # logged above.  A run that spans this change has a step
                    # discontinuity in the series at the resume.
                    writer.add_scalar(
                        "GAN/adv_to_rec_ratio",
                        (
                            adaptive_adv
                            * last_layer_adv_grad
                            / (last_layer_rec_grad + 1e-6)
                        ).item(),
                        global_step,
                    )
                    if adaptive_fm_balance > 0.0:
                        writer.add_scalar(
                            "GAN/adaptive_fm_weight", adaptive_fm.item(), global_step
                        )
                        writer.add_scalar(
                            "GAN/adaptive_fm_requested",
                            adaptive_fm_requested.item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "GAN/last_layer_fm_grad",
                            last_layer_fm_grad.item(),
                            global_step,
                        )
                        # The quantity being held at ``fm_balance_target``.
                        # Reported *after* the weight, so it is what the optimizer
                        # actually receives rather than what fm asked for.
                        writer.add_scalar(
                            "GAN/fm_to_rec_ratio",
                            (
                                adaptive_fm
                                * last_layer_fm_grad
                                / (last_layer_rec_grad + 1e-6)
                            ).item(),
                            global_step,
                        )
                    # How far the discriminator is from emitting one constant
                    # for real and fake.  Falling toward zero is the failure
                    # the governor exists to catch, and reading it needs no
                    # baseline: zero is dead, and bigger is better.
                    writer.add_scalar(
                        "GAN/disc_headroom",
                        adversarial_ceiling_governor.headroom,
                        global_step,
                    )
                    writer.add_scalar(
                        "GAN/disc_headroom_best",
                        adversarial_ceiling_governor.best_headroom,
                        global_step,
                    )
                    # 1 while the governor is rewinding the ramp.  A series
                    # that sits at 1 means the adversarial weight is as high as
                    # this discriminator can support.
                    writer.add_scalar(
                        "GAN/ceiling_holding",
                        float(adversarial_ceiling_governor.holding),
                        global_step,
                    )
            else:
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

            if discriminator_parameter_states is not None:
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
                    marker = "*" if improved else " "
                    print(
                        f"[HOLDOUT]{marker} step {global_step}: {holdout_loss:.5f}  "
                        f"(smoothed {overtrain_monitor.smoothed:.5f}, best "
                        f"{overtrain_monitor.best:.5f} @ {overtrain_monitor.best_step}, "
                        f"noise {overtrain_monitor.sigma:.5f}, "
                        f"{overtrain_monitor.since_progress}/"
                        f"{overtrain_monitor.patience} since progress)"
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
                        print(
                            f"[HOLDOUT]  train probe {probe_score:.5f}, "
                            f"generalisation gap {holdout_loss - probe_score:+.5f}"
                        )
                    if "latent_gap" in metrics:
                        print(
                            f"[HOLDOUT]  latent gap {metrics['latent_gap']:.5f}  "
                            f"(posterior {metrics['latent_posterior']:.5f}, "
                            f"prior {holdout_loss:.5f})"
                        )
                if overtrain_monitor.overtrained and not overtrain_flagged:
                    overtrain_flagged = True
                    if rank == 0:
                        print(
                            f"\n[OVERTRAIN] Held-out loss has not improved for "
                            f"{overtrain_monitor.since_progress} evaluations. "
                            f"The last good weights are step "
                            f"{overtrain_monitor.best_step} ({overtrain_monitor.best:.5f}); "
                            f"they will be the ones exported."
                        )
                        if stop_on_overtrain:
                            print("[OVERTRAIN] Stopping (stop_on_overtrain is on).")
                        else:
                            scale = overtrain_monitor.backoff()
                            print(
                                f"[OVERTRAIN] Continuing; scoring every "
                                f"{holdout_interval * scale} steps from here."
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
            if chouwagan_stack_active:
                avg_rolling_cache["loss_envelope"].append(loss_envelope.detach())
                avg_rolling_cache["loss_rms"].append(loss_rms.detach())
                avg_rolling_cache["loss_peak"].append(loss_peak.detach())
                avg_rolling_cache["loss_waveform"].append(loss_waveform.detach())
            avg_rolling_cache["loss_kl"].append(loss_kl.detach())

            # D Grads:
            if grad_norm_d is not None:
                if torch.isfinite(grad_norm_d):
                    avg_rolling_cache["grad_norm_d"].append(grad_norm_d)
                else:
                    writer.add_scalar("Grad_Norm_Diag/D_Skipped", 1, global_step)
            if grad_norm_d_r1 is not None and torch.isfinite(grad_norm_d_r1):
                avg_rolling_cache["grad_norm_d_r1"].append(grad_norm_d_r1)
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

                if disc_branch_names and disc_branch_cache:
                    # ``loss_disc`` is normalized by head count, so the governor's
                    # ``floor_loss`` -- the loss of a head that emits one constant
                    # for real and fake -- is already a *per-head* quantity.  The
                    # same subtraction therefore gives each head the metric
                    # ``GAN/disc_headroom`` reports for the ensemble, and the mean
                    # of these series reproduces it exactly.  Reading them side by
                    # side is the point: the aggregate cannot distinguish nine
                    # heads losing ground evenly from one collapsed head being
                    # carried by eight healthy ones.
                    branch_means = torch.stack(list(disc_branch_cache)).mean(dim=0).cpu()
                    floor_loss = (
                        adversarial_ceiling_governor.floor_loss
                        if adversarial_ceiling_governor is not None
                        else None
                    )
                    branch_scalars = {}
                    for index, name in enumerate(disc_branch_names):
                        if index >= branch_means.shape[0]:
                            break
                        real_loss = branch_means[index, 0].item()
                        fake_loss = branch_means[index, 1].item()
                        branch_scalars[f"Disc_Branch/real_{name}"] = real_loss
                        branch_scalars[f"Disc_Branch/fake_{name}"] = fake_loss
                        if floor_loss is not None:
                            branch_scalars[f"Disc_Branch/headroom_{name}"] = (
                                floor_loss - (real_loss + fake_loss)
                            )
                    summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=branch_scalars,
                    )

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)

                if chouwagan_stack_active:
                    source_scalars = collect_source_path_metrics(net_g)
                    if source_scalars:
                        summarize(
                            writer=writer,
                            global_step=global_step,
                            scalars=source_scalars,
                        )

                # KL diagnostics (diag tab)
                if len(kl_std_cache) > 0:
                    diag_scalars = {
                        "diag/kl_std": sum(kl_std_cache) / len(kl_std_cache),
                        "diag/kl_mean_per_dim": sum(kl_mean_cache) / len(kl_mean_cache),
                        "diag/kl_active_fraction": sum(kl_active_cache) / len(kl_active_cache),
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
                print(f"[PREVIEW] epoch {epoch}: rendering from {preview_label}")

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
                        extra=_checkpoint_extra(r1_controller, grad_scaler, r1_grad_scaler),
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
            print(
                f"Training phase limit reached at {phase_step} local steps "
                f"({global_step} global steps)."
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
            print(f"[EXPORT] weights: {ckpt_label}")

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
