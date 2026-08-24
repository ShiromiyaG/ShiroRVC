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
from random import randint
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
from rvc.configs.vocoders import normalize_vocoder
from rvc.lib.algorithm.chouwagan_svae import (
    ARCHITECTURE_ID as CHOUWAGAN_ARCHITECTURE_ID,
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
# a bin spans four harmonics at f0=120), so ChouwaGAN needs the linear-frequency
# MS-STFT term beside it and ships "Hybrid L1", while HiFi-GAN ships the plain
# mel it was designed around.  Leaving it on the UI meant every run re-answered
# a question the vocoder had already answered, and answered it wrong by default.
spectral_loss = str(getattr(config.train, "spectral_loss", "L1 Mel Loss"))
# Horizon-derived decay, and the preferred way to set one.
#
# ``lr_decay`` is a per-epoch gamma, so the decay a run actually receives is a
# function of how many epochs it turns out to have.  The shipped 0.999875
# delivers 0.94x over 500 epochs and 0.99x over 60 -- effectively no decay at
# either length, which is how the 44.1 kHz pretrain ran 20k steps with a flat
# LR without anything looking wrong.  Worse, the value has to be recomputed by
# hand every time the planned length changes, and nothing catches it when that
# is forgotten.
#
# ``lr_final_ratio`` states the intent instead -- "end at this fraction of the
# starting LR" -- and the gamma is derived from the run's real length.  Change
# the epoch count and the schedule restretches itself.  It is a ratio rather
# than an absolute target LR because G and D start at different rates and the
# balance between them is load-bearing: a shared ratio preserves it, a shared
# endpoint would collapse it.
#
# ``None`` keeps the old ``lr_decay`` behaviour, so existing configs are
# unaffected.
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

chouwagan_active = vocoder == "chouwagan"
if chouwagan_active and sample_rate != 44100:
    raise ValueError("ChouwaGAN requires the 44.1 kHz configuration.")

# AMP precision / dtype init
# BF16 carries FP32's exponent range, so it needs no GradScaler and cannot
# overflow the way FP16 does in the periodic activations, the anti-aliased
# resampling or the Gaussian latent statistics.  It takes precedence when both
# flags are set.
# Training runs in FP32 throughout.  BF16's 8-bit mantissa was measured to
# cost far too much gradient fidelity for this model: against a true-FP32
# reference the gradient cosine was 0.77 at init and 0.82 after training,
# versus 0.9997 for TF32 and 0.99 for FP16 (both 11-bit mantissa).  The
# oscillatory NSF source and the L1-on-log-mel loss produce heavily cancelling
# sums, which amplify BF16 rounding: `m_source.l_linear.weight` even flipped
# sign (+94.13 -> -3.30).  TF32 already provides tensor cores at 11 bits with
# no autocast and no scaler, so it is the mixed-precision path here.

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
# ( ChouwaGAN: enc_p + chouwagan_discrete / legacy VITS: enc_p + enc_q + flow )
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

    # Batch size 1: the collate pads to the longest item, and padding silence
    # into the mel would put a constant in the metric that moves with whatever
    # else lands in the batch.  A single item has nothing to pad against.
    holdout_batches = None
    if holdout_dataset is not None and rank == 0:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # Materialised once and kept: the point of a holdout is that every
        # evaluation scores the exact same audio, and re-reading it through a
        # loader each time would also cost more than the forward pass.
        holdout_batches = list(holdout_loader)

    return train_loader, holdout_batches

def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    if vocoder == "chouwagan":
        chouwagan_defaults = {
            "chouwagan_hierarchical": True,
            "chouwagan_architecture_id": CHOUWAGAN_ARCHITECTURE_ID,
            "chouwagan_content_channels": 128,
            "chouwagan_detail_channels": 64,
            "chouwagan_detail_gate_init": 0.0,
            "chouwagan_late_detail_fusion": True,
            "chouwagan_posterior_channels": 160,
            "chouwagan_prior_hidden_channels": 128,
            "chouwagan_shared_prior_trunk": True,
            "chouwagan_content_speaker_conditioning": False,
            "chouwagan_content_path_dropout": 0.05,
            "chouwagan_svae_slow_latent_channels": 32,
            "chouwagan_svae_fast_latent_channels": 64,
            "chouwagan_svae_prior_blocks": 4,
            "chouwagan_svae_prior_heads": 4,
            "chouwagan_svae_prior_kernel_size": 31,
            "chouwagan_svae_posterior_blocks": 4,
            "chouwagan_svae_posterior_slow_blocks": 2,
            # A fraction of the 0.15 rate target, not most of it.  These were
            # briefly 0.10 -- 63% of the budget -- on the reasoning that a
            # higher floor stops dimensions between the floor and the target
            # from being pushed at the prior.  It does, and it is the wrong
            # trade: the controller pins the per-dim *mean* at the target, so
            # raising the floor does not add rate, it only decides how much of
            # the fixed budget is spent before allocation starts.  At 0.10 only
            # 0.05 nats/dim were allocatable and latent collapse became the
            # optimum -- measured at step 75.6k, 62 of 64 fast dims parked
            # exactly at the floor while two carried everything, and
            # ``kl_effective_dims_fast`` fell from 51.7 to 17.6.  The rate
            # controller cannot see that: the mean reads as satisfied either
            # way.  Do not "fix" a collapse by raising these.
            "chouwagan_svae_free_bits_slow": 0.05,
            "chouwagan_svae_free_bits_fast": 0.03,
            "chouwagan_svae_kl_scale_anchor": 1.0,
            "chouwagan_svae_feature_scale_anchor": 1.0,
            # The decoder only learns to make good audio from prior latents on
            # the replaced fraction of the batch, and inference is 100% prior.
            # At 0.25 that gap stays wide in low-level material, where ContentVec
            # and an unvoiced f0 give the prior almost nothing to predict from.
            #
            # The start and ramp are early on purpose.  The reconstruction
            # gradient is the only pressure that makes the prior *informative*
            # rather than merely cheap to match, and at the old 8000/20000 it
            # arrived around step 28k -- after the window in which the latent
            # collapses had already closed.
            "chouwagan_prior_replacement_max": 0.7,
            "chouwagan_prior_replacement_start": 2000,
            "chouwagan_prior_replacement_ramp": 12000,
            "chouwagan_prior_uses_logs": True,
            "chouwagan_excitation_unet": True,
            "chouwagan_excitation_kernel": 7,
            "chouwagan_noise_injection": False,
            "chouwagan_output_head_threshold": 0.85,
            "chouwagan_output_head_ceiling": 1.0,
            "chouwagan_remove_output_dc": True,
        }
        # The Chouwa frontend is intentionally replaced by the continuous
        # SVAE architecture; old FSQ identifiers must not select the legacy
        # implementation for a new run.
        config.model["chouwagan_architecture_id"] = CHOUWAGAN_ARCHITECTURE_ID
        for key, value in chouwagan_defaults.items():
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
    vocoder = normalize_vocoder(vocoder)
    if vocoder == "chouwagan":
        from rvc.lib.algorithm.discriminators.multi import ChouwaGANDiscriminator

        return ChouwaGANDiscriminator(
            use_spectral_norm=False,
            use_san=bool(getattr(config.model, "chouwagan_use_san", True)),
            use_checkpointing=use_checkpointing,
            sample_rate=config.data.sample_rate,
            periods=tuple(getattr(config.model, "chouwagan_d_periods", (2, 3, 5, 7, 11))),
            spectrogram_channels=tuple(
                getattr(config.model, "chouwagan_d_spectrogram_channels", (32, 64, 96))
            ),
            spectrogram_compression=float(
                getattr(config.model, "chouwagan_d_spectrogram_compression", 0.3)
            ),
            use_subband=bool(getattr(config.model, "chouwagan_d_use_subband", False)),
            subband_bands=int(getattr(config.model, "chouwagan_d_subband_bands", 8)),
            subband_channels=tuple(
                getattr(config.model, "chouwagan_d_subband_channels", (64, 128, 192))
            ),
        )

    from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined

    return MPD_MSD_Combined(
        config.model.use_spectral_norm,
        use_checkpointing=use_checkpointing,
    )


def _lazy_r1_penalty(
    net_d,
    real_audio,
    branch_index,
    segment_size=None,
):
    """Compute an unbiased branch sample of the mean discriminator R1."""
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


def _chouwagan_ablation_margin(
    model,
    discrete_model,
    discrete_parts,
    ids_slice,
    segment_size,
    pitchf,
    sid,
    target,
    full_output,
    branch,
    dimension,
    margin,
):
    """Measure one latent dimension: is the decoder actually using it?

    Returns ``(loss, delta, ablated_error)``.  ``delta`` is the measurement and
    is the reason to call this at all: mel L1 with the dimension zeroed minus
    mel L1 with it intact, so positive means zeroing the dimension hurt, i.e.
    it was load-bearing.  It is the only signal here that distinguishes a dead
    dimension from one the prior simply predicts well -- per-dimension KL
    cannot, because KL measures ``posterior || prior`` with both learned, and a
    low value means "the prior predicts it" at least as often as it means
    "nothing is there".  Measured over this pretrain the rank correlation
    between the two is 0.24.

    ``loss`` is kept for callers that still want it, but it should be weighted
    at zero, and the reasons are worth stating so it does not get switched back
    on by intuition:

    * It has no gradient toward its stated purpose.  ``ablated_error`` is
      detached -- and computed under ``no_grad`` besides -- so the only
      derivative is ``d/d full_error = 1``.  The term therefore adds *generic
      reconstruction pressure*, scaled by how useless the probed dimension was.
      It cannot make that dimension carry information, which is the thing it is
      named for.  The detach is not the bug: without it the cheapest way to
      raise ``ablated_error`` is to inflate the dimension's output scale, which
      makes ablation more destructive without making it more informative.
    * Even with a correct gradient it would be too dilute to matter.  One
      dimension is drawn uniformly per interval out of 96, so a given dimension
      is touched about once every ``96 * interval`` steps -- roughly 35 times
      across an entire 434k-step run at an interval of 128.  That is ample for
      a histogram and nowhere near enough to shape a parameter.

    The dilution argument is what makes this a diagnostic rather than a loss: a
    measurement only has to be dense enough to average, while a gradient has to
    be dense enough to steer.
    """
    with torch.no_grad():
        ablated_detail, slow_detail, fast_detail = discrete_model.ablate_dimension(
            discrete_parts,
            branch,
            dimension,
            discrete_parts["content"].shape[-1],
        )
    content_slice = commons.slice_segments(
        discrete_parts["content"], ids_slice, segment_size, dim=3
    )
    detail_slice = commons.slice_segments(
        ablated_detail, ids_slice, segment_size, dim=3
    )
    slow_slice = commons.slice_segments(
        slow_detail, ids_slice, segment_size, dim=3
    )
    fast_slice = commons.slice_segments(
        fast_detail, ids_slice, segment_size, dim=3
    )
    pitchf_slice = commons.slice_segments(pitchf, ids_slice, segment_size, dim=2)
    speaker = model.emb_g(sid).unsqueeze(-1)

    with torch.no_grad():
        ablated_output = model.dec(
            torch.cat((content_slice, detail_slice), dim=1),
            pitchf_slice,
            g=speaker,
            content_latent=content_slice,
            detail_latent=detail_slice,
            slow_detail_latent=slow_slice,
            fast_detail_latent=fast_slice,
        )

    target_mel = wave_to_mel(config, target)
    full_mel = wave_to_mel(config, full_output)
    ablated_mel = wave_to_mel(config, ablated_output)
    full_error = F.l1_loss(full_mel.float(), target_mel.float())
    ablated_error = F.l1_loss(ablated_mel.float(), target_mel.float())
    delta = ablated_error - full_error.detach()
    loss = F.relu(full_error + float(margin) - ablated_error.detach())
    return loss, delta.detach(), ablated_error.detach()


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
    # Accumulate with one multi-tensor kernel rather than one per parameter.
    # The decoder has 332 of them, and the old `parameter.grad + gradient` was
    # also out-of-place, so it allocated 332 tensors it immediately dropped:
    # measured 5.15 ms per step against 0.30 ms here, bitwise identical, on a
    # ~270 ms step.  This is the same in-place accumulation the autograd engine
    # itself does, so it is safe on grads that `backward()` just populated.
    existing_grads = []
    incoming_grads = []
    for parameter, gradient in zip(decoder_parameters, decoder_gradients):
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


#: Defaults for the two prior-loss knobs.  Named because the "was this set on
#: purpose?" test compares against them, and a literal repeated at the
#: comparison would drift from the literal in the ``getattr`` fallback.
PRIOR_LOSS_WEIGHT_DEFAULT = 1.0
PRIOR_WARMUP_STEPS_DEFAULT = 10000

#: How fast ``loss_disc`` may drift upward, per 10k steps, while still counting
#: as a healthy GAN rather than a generator running away from its critic.
#:
#: A GAN's discriminator loss rises as the generator improves -- that is the
#: process working.  Measured on the 44.1 kHz ChouwaGAN pretrain, that drift was
#: 0.0022 per 10k steps over 80k steps with the discriminator's headroom holding
#: at 0.11, well above the 0.08 collapse floor.  4.5x that is the allowance
#: here: generous enough that the normal case never trips the ramp, far short of
#: the runaway the ramp exists to catch.
#:
#: Re-measured 2026-08-22 and raised from 0.01.  A later 44.1 kHz pretrain drifts
#: at 0.0095 per 10k -- 4.3x the run this was calibrated against, and *healthy*:
#: headroom held at 0.19, 2.4x the collapse floor, with the holdout still setting
#: records.  Sitting a threshold on top of the normal case lets noise decide, and
#: it did: ``ceiling_holding`` averaged 0.36 after ``ramp_start`` and the ceiling
#: went backwards from 1.380 to 1.170 while the balance rule asked for 2.49.  The
#: allowance is now 3x the measured healthy drift rather than 1.05x it.
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


def _constant_discriminator_loss(san_direction_weight: float, san_active: bool) -> float:
    """``loss_disc`` for a discriminator that emits one constant for real and fake.

    This is the operational definition of a dead discriminator: it has stopped
    being a function of its input.  It is a fixed number for a given loss form,
    so the distance to it is an absolute health measure that needs no baseline
    and no reference run.  Both sides are symmetric about the midpoint of the
    real and fake targets, so the best constant is always 0.5.

    Mirrors the two branches of :func:`~rvc.train.losses.discriminator_loss`:
    SAN heads use a squared-softplus surrogate and carry a direction term at
    ``san_direction_weight``; everything else is plain LSGAN.
    """
    if not san_active:
        return 0.5
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

    The ceiling used to be a pure function of ``global_step``: it rose on a
    fixed cosine schedule regardless of what the discriminator was doing.  That
    is unsafe in one specific, observed way.  The balance rule asks for
    ``balance_target * rec_grad / adv_grad``, and ``adv_grad`` *is* the
    discriminator's strength -- so a weakening discriminator makes the rule
    request a *larger* weight, which weakens it further.  A pretrain run rode
    that loop from a real/fake separation of 0.13 to 0.04, 97% of the way to a
    discriminator emitting one constant for both, while the requested weight
    climbed past 6 and the ceiling was the only thing bounding it.

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

    The backstop was first written against a *fraction of the best headroom
    seen*, and a live run showed that to be wrong.  Headroom peaks early, while
    the generator is still weak, and then falls as the generator improves --
    which is the GAN working, not failing.  Measured on the 44.1 kHz pretrain:
    headroom peaked at 0.247 around step 9k and was 0.099 by step 17k, a level
    at which the discriminator still separated real from fake by 0.094 in score
    space and was plainly alive.  A peak-relative rule would have held the
    ceiling down for the rest of the run over that.  An absolute floor answers
    the question actually being asked -- is this discriminator near useless --
    and the run that did collapse sat at headroom 0.060 with a margin of 0.040,
    well under it.

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
    ):
        self.start_step = max(0, int(start_step))
        self.ramp_steps = max(0, int(ramp_steps))
        self.ceiling_start = float(ceiling_start)
        self.ceiling_end = float(ceiling_end)
        self.floor_loss = float(floor_loss)
        # Where the ceiling goes when the discriminator has actually collapsed.
        # ``ceiling_start`` used to be the hard bottom, which left the one case
        # the backstop exists for only half-answered: against a discriminator
        # emitting one constant for real and fake, the balance rule asks for a
        # huge weight, the request saturates the ceiling, and the ceiling could
        # not retreat below 1.0 -- so full adversarial pressure stayed applied
        # to a dead discriminator for the rest of the run.  Retreating to the
        # configured ``adaptive_adv_min`` lets the generator fall back on
        # reconstruction until the discriminator recovers.
        self.collapse_ceiling = min(float(collapse_ceiling), float(ceiling_start))
        self.fast_momentum = float(fast_momentum)
        self.slow_momentum = float(slow_momentum)
        # Derived, not chosen.  Two EMAs following a linear drift of ``r`` per
        # step settle at a fixed separation of ``r * (1/slow - 1/fast)``, because
        # each lags the true value by its own time constant.  So a *healthy*
        # upward drift produces a constant, predictable ``fast - slow``, and a
        # threshold that does not account for it spends most of its budget on
        # the normal case and fires on noise.
        #
        # That is what the 44.1 kHz pretrain did.  Its ``loss_disc`` drifted up
        # at 0.0022 per 10k steps -- the generator improving, exactly what the
        # docstring above says is the GAN working -- which with these momenta
        # gives a steady separation of 0.0018, or 59% of the old fixed 3e-3.
        # Noise covered the rest: ``ceiling_holding`` sat near 0.5, and since
        # the counter moves +1 when clear and -1 when holding, the ramp was
        # stationary.  80k steps of ramp delivered a ceiling of 1.12 out of a
        # configured 3.0, while the balance rule was asking for 4-9.
        #
        # Deriving the threshold from a stated drift rate keeps the early
        # warning the docstring describes -- a runaway is fast, not a creep --
        # while giving the healthy creep room to exist.  The headroom floor is
        # untouched and remains the backstop against an actual collapse.
        if tolerance is None:
            tolerance = (HEALTHY_DISC_DRIFT_PER_10K / 10_000.0) * (
                1.0 / self.slow_momentum - 1.0 / self.fast_momentum
            )
        self.tolerance = float(tolerance)
        # ~0.055 of real/fake separation in score space: alive, but only just.
        self.headroom_floor = float(headroom_floor)
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

        ``progress`` is a counter that advances at most +1 per step, and it is
        built fresh on every process start.  The class docstring says the
        pretraining ramp follows ``global_step`` "so resuming picks the ramp up
        where it left off", but nothing implemented that: resuming reset the
        counter to zero, dropped the ceiling back to ``ceiling_start`` and made
        the run re-earn a ramp it had already earned.  On the 44.1 kHz pretrain
        that meant a resume at step 168k would spend ~3.5k steps under *less*
        adversarial pressure than the step before it, and 64k steps getting
        back.

        Seeding assumes the skipped steps were healthy, which is the optimistic
        reading -- holds and retreats in the original run are forgotten.  That
        is the right way to be wrong here.  A hold only pauses the ramp, so the
        worst case is a ceiling as high as an unheld run would have had; and if
        the discriminator really is in trouble, the EMAs are seeded from the
        first ``loss_disc`` of the resumed run, so the headroom backstop sees a
        true headroom and starts retreating on this very same call.  Starting
        from zero, by contrast, is wrong in every case including the healthy
        one.
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
                # the EMA to mean anything.
                #
                # The slow EMA has a time constant of 1/slow_momentum = 10k
                # steps, so whatever the *first* batch happens to produce
                # dominates ``slow`` for tens of thousands of steps after it is
                # seeded.  On a resume that is a live hazard rather than a
                # cosmetic one: the first batch of a resumed run reads high,
                # ``headroom`` therefore reads far below its true value, and the
                # collapse backstop rewinds a ramp that nothing was wrong with.
                # Measured on the 44.1 kHz pretrain at the 187013 resume --
                # ``ceiling_holding`` pinned at 1.0 for every step, the ceiling
                # walking 3.0 -> 2.82 on the -4/step retreat, while ``loss_disc``
                # of 2.2604 put the true headroom at 0.1117, well clear of the
                # 0.08 floor.
                #
                # A running mean has no memory of the seed to shake off: after k
                # samples it *is* the mean of those k.  It hands over to the EMA
                # at the end of warmup with an estimate worth acting on.
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
        time cancels two thirds of the ramp.  Measured on the 44.1 kHz pretrain
        with h = 0.36, the ceiling reached 1.184 of a configured 3.0 after 55.6k
        steps and was *falling* while the balance rule asked for 2.49.

        Pausing instead of rewinding makes the trend gate what it is described
        as -- an early warning that stops the ramp -- rather than a mechanism
        that actively unwinds progress on healthy behaviour.  Retreat is left to
        the headroom backstop, which fires on the operational definition of a
        dead discriminator rather than on the sign of a noisy derivative.
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
        if self.fast - self.slow > self.tolerance:
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


def _holdout_spectral_loss(net_g, batches, config, device):
    """Mel distance between the *inference* path and ground truth, held out.

    Deliberately the inference path (``infer``) and not the training forward.
    On ChouwaGAN the training path samples the posterior, which has seen the
    target spectrogram, and a model that is memorising scores well there long
    after it has stopped generalising.  ``infer`` runs prior-only, which is
    what inference actually does, and the same call works for the HiFi-GAN
    synthesizer -- so one metric covers both vocoders and every existing
    pretrain.

    Plain L1 on the log-mel, with no adversarial, feature-matching or KL term:
    those are scored against a discriminator and a schedule that keep moving,
    and the only thing that may move between two evaluations here is the
    generator.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0

    # The metric has to be a pure function of the weights, and ``infer`` draws
    # noise on both paths: the ChouwaGAN branch consumes RNG even when
    # ``deterministic`` zeroes the scale, and the VITS/HiFi-GAN branch ignores
    # the flag entirely and always samples at 0.66666 (see
    # ``Synthesizer.infer``).  Left alone that would put fresh sampling noise
    # into every evaluation, which on the HiFi-GAN path is the whole signal.
    # Pinning the seed makes each evaluation see identical noise; restoring the
    # state afterwards keeps a variable number of draws from shifting the
    # training stream underneath the run.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(0x5EED)
    if cuda_rng_state is not None:
        torch.cuda.manual_seed_all(0x5EED)

    try:
        with torch.no_grad():
            for batch in batches:
                phone, phone_lengths, pitch, pitchf, _spec, _spec_lengths, wave, wave_lengths, sid = batch
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
                pitch = pitch.to(device, non_blocking=True)
                pitchf = pitchf.to(device, non_blocking=True)
                sid = sid.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)

                generated, *_ = model.infer(phone, phone_lengths, pitch, pitchf, sid, 0)
                # ``infer`` rebuilds the waveform from frame-rate features, so
                # its length lands within a hop of the target rather than on it.
                length = min(generated.shape[-1], wave.shape[-1], int(wave_lengths.min()))
                if length <= config.data.filter_length:
                    continue
                target_mel = wave_to_mel(config, wave[..., :length], num_mels=None)
                output_mel = wave_to_mel(config, generated[..., :length], num_mels=None)
                total += float(F.l1_loss(output_mel, target_mel))
                count += 1
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            model.train()
    return total / count if count else float("nan")


class _OvertrainMonitor:
    """Find the last point where held-out quality was still improving.

    Overtraining is not visible in the training loss -- that is what the word
    means.  The generator keeps getting better at reproducing the slices it is
    shown while getting worse at everything else, so the only way to catch the
    turn is to score audio it has never been trained on and watch for the
    minimum.

    Keeps the weights from that minimum.  ``patience`` is counted in
    evaluations rather than steps so it does not silently change meaning when
    the interval does, and ``min_delta`` is relative so it means the same thing
    at any loss scale.

    There are two questions here and they want opposite thresholds, which is
    why they are tracked separately:

    * *Which weights do we keep?*  The lowest held-out loss, full stop.  A
      threshold here throws away a real improvement for being small.  It did:
      the 44.1 kHz pretrain's true minimum was 0.70210 at step 166k, but a
      single ``min_delta`` gate had frozen ``best`` at 0.70250 from step 140k
      because the improvement was 0.06% against a required 0.1%, and
      :func:`_deliverable_weights` hands out ``state_dict`` in preference to
      everything else -- so the run would have exported weights it had itself
      measured as worse.
    * *Has the run stopped improving?*  Here a threshold is the whole point.
      Without one, noise resets the counter forever and the detector never
      fires; the counter has to ignore improvements too small to be real.

    Using one number for both means picking which of the two to get wrong.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.best = float("inf")
        self.best_step: int | None = None
        self.state_dict: dict | None = None
        # The last value good enough to count as progress.  Separate from
        # ``best`` because it moves on a coarser ratchet.
        self.patience_reference = float("inf")
        self.since_best = 0
        self.history: list[tuple[int, float]] = []

    def update(self, source, value: float, step: int) -> bool:
        """``source`` is whatever produced ``value``: a model, or a ``WeightEMA``.

        Returns whether this is a new best, which is what the caller marks in
        the log.  Keeping the weights that were actually scored is the whole
        contract -- snapshotting the live model while having measured the
        average would deliver a model nobody evaluated.
        """
        if not math.isfinite(value):
            return False
        self.history.append((int(step), float(value)))
        improved = value < self.best
        if improved:
            self.best = float(value)
            self.best_step = int(step)
            self.state_dict = _cpu_state_dict(source)
        if value < self.patience_reference * (1.0 - self.min_delta):
            self.patience_reference = float(value)
            self.since_best = 0
        else:
            self.since_best += 1
        return improved

    @property
    def overtrained(self) -> bool:
        return self.state_dict is not None and self.since_best >= self.patience


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


def _adaptive_adversarial_weight(
    loss_reconstruction,
    loss_adversarial,
    parameter,
    balance_target: float = 0.5,
    minimum: float = 0.01,
    maximum: float = 1.0,
):
    """Balance adversarial pressure against reconstruction on one decoder layer.

    Returns ``(applied, reconstruction_norm, adversarial_norm, requested)``.
    ``requested`` is the unclamped ratio: it is what the balance rule asked
    for, and logging it is the only way to see that the ceiling is binding.
    Reporting just the clamped value hides a saturated weight as a flat line at
    the ceiling, which is indistinguishable from a healthy constant.
    """
    reconstruction_gradient = torch.autograd.grad(
        loss_reconstruction,
        parameter,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    adversarial_gradient = torch.autograd.grad(
        loss_adversarial,
        parameter,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]

    zero = loss_reconstruction.detach().new_zeros(())
    reconstruction_norm = (
        reconstruction_gradient.detach().float().norm()
        if reconstruction_gradient is not None
        else zero
    )
    adversarial_norm = (
        adversarial_gradient.detach().float().norm()
        if adversarial_gradient is not None
        else zero
    )
    requested = (
        balance_target * reconstruction_norm / (adversarial_norm + 1e-6)
    ).detach()
    adaptive_weight = requested.clamp(min=minimum, max=maximum).detach()
    return adaptive_weight, reconstruction_norm, adversarial_norm, requested


def _generator_gradient_metrics(net_g):
    """Return gradient and gradient-to-parameter norms for generator subsystems."""
    model = net_g.module if hasattr(net_g, "module") else net_g
    groups = {
        "content_encoder": [],
        "frontend_content": [],
        "coarse_spectral": [],
        "posterior": [],
            "latent": [],
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
        elif name.startswith("chouwagan_discrete.content"):
            group = "frontend_content"
        elif name.startswith("chouwagan_discrete.coarse_spectral"):
            group = "coarse_spectral"
        elif (
            name.startswith("chouwagan_discrete.posterior_slow_head")
            or name.startswith("chouwagan_discrete.posterior_fast_head")
            or name.startswith("chouwagan_discrete.fast_to_")
            or name.startswith("chouwagan_discrete.slow_to_")
        ):
            group = "latent"
        elif name.startswith("chouwagan_discrete.posterior"):
            group = "posterior"
        elif name.startswith("chouwagan_discrete.prior"):
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
    if vocoder == "chouwagan" and float(getattr(config.train, "r1_gamma", 0.0)) > 0:
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

    Everything outside `dec.` and `emb_g.` is frozen, which on ChouwaGAN means
    `enc_p` plus the SVAE prior/posterior under `chouwagan_discrete`.
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


def apply_finetune_freezes(net_g, rank):
    """Keep the learned content/code distribution stable during Chouwa fine-tuning."""
    if not finetune_phase:
        return

    model = net_g.module if hasattr(net_g, "module") else net_g
    discrete = getattr(model, "chouwagan_discrete", None)
    if discrete is None:
        return

    frozen_modules = [
        getattr(model, "enc_p", None),
        getattr(discrete, "content_input", None),
        getattr(discrete, "content_blocks", None),
        getattr(discrete, "content_attention", None),
        getattr(discrete, "posterior_input", None),
        getattr(discrete, "posterior_content", None),
        getattr(discrete, "posterior_speaker", None),
        getattr(discrete, "posterior_blocks", None),
        getattr(discrete, "posterior_slow_head", None),
        getattr(discrete, "posterior_fast_head", None),
        getattr(discrete, "posterior_slow_precondition", None),
        getattr(discrete, "posterior_fast_precondition", None),
        getattr(discrete, "coarse_spectral_head", None),
    ]
    frozen_params = 0
    discrete.content_path_dropout = 0.0
    for module in frozen_modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen_params += parameter.numel()

    if rank == 0:
        print(
            f"[INIT] Chouwa fine-tune frontend frozen: {frozen_params:,} params; "
            "decoder, speaker embedding, prior and continuous latent adapters remain trainable."
        )


def apply_training_freezes(net_g, rank):
    apply_frontend_freeze(net_g, rank)
    apply_finetune_freezes(net_g, rank)


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
    if chouwagan_active:
        # The ChouwaGAN loop backwards through the decoder graph more than once
        # per step: `loss_core.backward(retain_graph=True)` is followed by the
        # decoder-only `autograd.grad` for the GAN term, plus the adaptive-adv
        # and Grad_Source probes.  AOTAutograd's donated-buffer optimisation
        # rejects `retain_graph=True` outright, so it must be off for every
        # compiled mode.  The HiFi-GAN path does a single backward and keeps it.
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
    if chouwagan_active and mode == "reduce-overhead":
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
                "[INIT] 'reduce-overhead' is incompatible with the ChouwaGAN "
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


def checkpoint_step_from_path(path):
    if not path or path in ("", "None"):
        return 0
    match = re.search(r"(?:^|[\\/])G_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else 0


def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, sample_rate, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    resumed_g_path = None
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

            # Load the model and optim states
            generator_strict_load = strict_load and not chouwagan_active
            _, _, _, epoch_str, _ = load_checkpoint(
                g_checkpoint_path,
                net_g,
                None if reset_optimizer_for_run else optim_g,
                generator_strict_load,
            )
            _, _, _, epoch_str, _ = load_checkpoint(
                d_checkpoint_path,
                net_d,
                None if reset_optimizer_for_run else optim_d,
                strict_load,
            )

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
            if (
                expected_architecture
                and expected_architecture != "vits_gaussian_v1"
                and checkpoint.get("architecture_id") != expected_architecture
            ):
                raise ValueError(
                    f"Pretrained generator architecture mismatch: expected '{expected_architecture}'."
                )
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_g.load_state_dict(
                state_dict,
                strict=not chouwagan_active,
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

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, ema


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
    )

    # Initial setup
    setup_env_and_distr(rank, n_gpus, device, device_id, config)

    # Dataloading and loaders preparation
    train_loader, holdout_batches = prepare_dataloaders(config, n_gpus, rank, batch_size)

    # Spk dim verif
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = spk_dim

    if rank == 0 and warmup_active():
        warmup_tag = "Manual control" if warmup_steps > 0 else f"per epoch x{warmup_duration}"
        print(f"[INIT] linear warmup: {effective_warmup_steps(train_loader)} steps  -  ({warmup_tag})")

    # Spectral loss init
    fn_spectral_loss2 = None
    fn_spectral_loss_ms = None

    # ChouwaGAN's compressed-mel distance.  A plain L1 has a gradient of
    # constant magnitude no matter how close the reconstruction gets, so the
    # generator settles into a noise-limited equilibrium where the loss creeps
    # down but the gradient norm never anneals.  Huber is L2 below ``beta``
    # (self-annealing as the residual shrinks) and L1 above it (robust to the
    # onset/silence bins that a plain MSE would over-weight).
    mel_distance = str(
        getattr(config.train, "chouwagan_mel_distance", "huber")
    ).lower()
    mel_huber_beta = float(getattr(config.train, "chouwagan_mel_huber_beta", 0.3))
    # Frequency weighting for that distance.  A mean over bins gives every bin
    # the same vote, and past the Huber knee the vote stops scaling with the
    # error, so a badly wrong minority of bins stays wrong: measured at 46% of
    # the mel error against 32% of the gradient below 1 kHz.  1.0 is off, which
    # is what a config that predates this key gets.
    mel_low_emphasis = float(
        getattr(config.train, "chouwagan_mel_low_emphasis", 1.0)
    )
    mel_low_emphasis_hz = float(
        getattr(config.train, "chouwagan_mel_low_emphasis_hz", 1000.0)
    )

    def _make_mel_distance():
        if not chouwagan_active or mel_distance in ("l1", "mae"):
            base, weighted = torch.nn.L1Loss, chouwagan_active
        elif mel_distance == "huber":
            base, weighted = (
                lambda **kw: torch.nn.SmoothL1Loss(beta=mel_huber_beta, **kw)
            ), True
        elif mel_distance in ("mse", "l2"):
            base, weighted = torch.nn.MSELoss, True
        else:
            raise ValueError(
                f"Unknown chouwagan_mel_distance {mel_distance!r}: "
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
        if rank == 0 and chouwagan_active:
            print(
                f"[INIT] Mel distance: {mel_distance}"
                + (f" (beta={mel_huber_beta})" if mel_distance == "huber" else "")
            )
        if swap_l1_to_ms:
            fn_spectral_loss_ms = MultiScaleMelSpectrogramLoss(
                sample_rate=sample_rate,
                safe_log=chouwagan_active,
                loss_fn=_make_mel_distance(),
            )
    elif spectral_loss == "Multi-Scale Mel Loss":
        # Uses the same configured distance as the single-scale mel loss.  It
        # used to hardcode L1, which silently ignored ``chouwagan_mel_distance``
        # -- selecting this mode quietly discarded the Huber setting.
        #
        # One caveat worth knowing: ``chouwagan_mel_huber_beta`` was tuned
        # against ``wave_to_mel(for_loss=True)``, while this operates on
        # ``log1p(mel * log_scale)``.  The two scales are similar but not
        # identical, so beta is a slightly different L2/L1 crossover here.
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=sample_rate,
            safe_log=chouwagan_active,
            loss_fn=_make_mel_distance(),
        )
        if rank == 0 and chouwagan_active:
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
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, ema = load_models_and_optimizers(
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
        rank
    )

    enable_vocoder_compile(net_g, device, rank)

    phase_start_step = global_step
    phase_step = 0
    phase_limit_reached = False
    if finetune_phase and vocoder == "chouwagan":
        model_g = net_g.module if hasattr(net_g, "module") else net_g
        discrete = getattr(model_g, "chouwagan_discrete", None)
        if discrete is not None:
            discrete.usage_loss_weight *= max(
                0.0,
                float(getattr(config.train, "chouwagan_finetune_usage_scale", 0.25)),
            )
            discrete.coarse_spectral_loss_weight *= max(
                0.0,
                float(getattr(config.train, "chouwagan_finetune_coarse_scale", 0.5)),
            )
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
        else max(0, int(getattr(config.train, "chouwagan_adaptive_adv_ramp_start", 20000))),
        planned_steps,
        0.2,
    )
    adv_ramp_steps = fit_schedule(
        max(
            0,
            int(
                getattr(config.train, "chouwagan_adaptive_adv_ramp_steps_finetune", 2000)
                if finetune_phase
                else getattr(config.train, "chouwagan_adaptive_adv_ramp_steps", 80000)
            ),
        ),
        planned_steps,
        0.5 if not finetune_phase else 0.25,
    )
    adversarial_ceiling_governor = _AdversarialCeilingGovernor(
        start_step=adv_ramp_start,
        ramp_steps=adv_ramp_steps,
        ceiling_start=1.0,
        ceiling_end=max(
            max(0.0, float(getattr(config.train, "chouwagan_adaptive_adv_min", 0.01))),
            float(getattr(config.train, "chouwagan_adaptive_adv_max", 8.0)),
        ),
        floor_loss=_constant_discriminator_loss(
            max(0.0, min(1.0, float(getattr(config.train, "chouwagan_san_direction_weight", 0.25)))),
            bool(getattr(config.model, "chouwagan_use_san", True)),
        ),
        collapse_ceiling=max(
            0.0, float(getattr(config.train, "chouwagan_adaptive_adv_min", 0.01))
        ),
    )

    # The turn from "still learning" to "memorising" happens on the scale of a
    # run, not an epoch.
    overtrain_monitor = None
    if holdout_batches:
        overtrain_monitor = _OvertrainMonitor(
            patience=int(getattr(config.train, "overtrain_patience", 8)),
            min_delta=float(getattr(config.train, "overtrain_min_delta", 0.001)),
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
            holdout_batches=holdout_batches,
            overtrain_monitor=overtrain_monitor,
            holdout_interval=interval,
            ema=ema,
            adversarial_ceiling_governor=adversarial_ceiling_governor,
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
    holdout_batches=None,
    overtrain_monitor=None,
    holdout_interval=0,
    ema=None,
    adversarial_ceiling_governor=None,
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
        "loss_prior": deque(maxlen=rolling_loss_steps),
        "loss_prior_slow": deque(maxlen=rolling_loss_steps),
        "loss_prior_fast": deque(maxlen=rolling_loss_steps),
        "prior_kl_slow": deque(maxlen=rolling_loss_steps),
        "prior_kl_fast": deque(maxlen=rolling_loss_steps),
        "prior_std_slow": deque(maxlen=rolling_loss_steps),
        "prior_std_fast": deque(maxlen=rolling_loss_steps),
        "posterior_std_slow": deque(maxlen=rolling_loss_steps),
        "posterior_std_fast": deque(maxlen=rolling_loss_steps),
        "scale_anchor": deque(maxlen=rolling_loss_steps),
        "prior_replacement": deque(maxlen=rolling_loss_steps),
        "content_rms": deque(maxlen=rolling_loss_steps),
        "posterior_detail_rms": deque(maxlen=rolling_loss_steps),
        "prior_detail_rms": deque(maxlen=rolling_loss_steps),
    }
    if chouwagan_active:
        # Waveform-domain reconstruction terms, logged unweighted so the series
        # stay readable when their weights change.
        for key in ("loss_envelope", "loss_rms", "loss_peak", "loss_waveform"):
            avg_rolling_cache[key] = deque(maxlen=rolling_loss_steps)
    else:
        # ``loss_kl`` is the Gaussian VITS ELBO term; the ChouwaGAN frontend
        # reports its divergence through ``loss_prior_*`` instead and leaves
        # this at a constant zero.
        avg_rolling_cache["loss_kl"] = deque(maxlen=rolling_loss_steps)
    kl_std_cache = deque(maxlen=rolling_loss_steps)
    kl_mean_cache = deque(maxlen=rolling_loss_steps)
    kl_active_cache = deque(maxlen=rolling_loss_steps)
    last_kl_per_dim = None
    discrete_vector_cache = {}

    r1_interval = max(1, int(getattr(config.train, "r1_interval", 16)))
    r1_gamma = float(getattr(config.train, "r1_gamma", 1.0))
    # Only fires once every ``r1_interval`` steps, so its window is scaled down
    # to cover the same wall-clock span as the per-step series.
    avg_rolling_cache["grad_norm_d_r1"] = deque(
        maxlen=max(2, rolling_loss_steps // r1_interval)
    )

    chouwagan_diagnostics_interval = max(
        1,
        int(getattr(config.train, "chouwagan_diagnostics_interval", 256)),
    )

    # Same correction for everything sourced from ``discrete_diagnostics``.
    # Those are produced once every ``chouwagan_diagnostics_interval`` steps, so
    # a deque sized in *steps* holds `maxlen * interval` steps of history: at the
    # defaults that is 50 x 256 = 12800 steps, which made the KL/prior series a
    # near-cumulative average of the whole run rather than a recent one. They
    # read low for thousands of steps after the underlying value had moved.
    diagnostic_window = max(
        2, rolling_loss_steps // max(1, chouwagan_diagnostics_interval)
    )
    for key in (
        "prior_kl_slow",
        "prior_kl_fast",
        "prior_std_slow",
        "prior_std_fast",
        "posterior_std_slow",
        "posterior_std_fast",
        "scale_anchor",
        "prior_replacement",
        "content_rms",
        "posterior_detail_rms",
        "prior_detail_rms",
        "kl_beta_slow",
        "kl_beta_fast",
        # Collapse detectors.  The rate controller targets the per-dim *mean*,
        # which a collapsed latent satisfies exactly, so none of the series
        # above can show the latent concentrating into a handful of dimensions.
        "kl_effective_dims_slow",
        "kl_effective_dims_fast",
        "kl_above_floor_slow",
        "kl_above_floor_fast",
        "kl_median_slow",
        "kl_median_fast",
    ):
        avg_rolling_cache[key] = deque(maxlen=diagnostic_window)
    r1_segment_size = max(
        1,
        int(getattr(config.train, "r1_segment_size", config.train.segment_size)),
    )
    # The generator's gradient norm is dominated by ``dec.conv_post`` (a 40->1
    # output conv that sums the waveform-domain loss gradient over B*T samples),
    # so its natural operating point sits in the hundreds.  The clip is a spike
    # guard, not a per-step rescaler -- keep the ceiling well above that range.
    chouwagan_global_clip_g = max(
        0.0,
        float(getattr(config.train, "chouwagan_global_clip_norm_g", 500.0)),
    )
    chouwagan_san_direction_weight = max(
        0.0,
        min(1.0, float(getattr(config.train, "chouwagan_san_direction_weight", 0.25))),
    )
    adv_warmup_steps = 0 if finetune_phase else max(
        0, int(getattr(config.train, "adv_warmup_steps", 2000))
    )
    chouwagan_discrete = None
    if chouwagan_active:
        model_g = net_g.module if hasattr(net_g, "module") else net_g
        chouwagan_discrete = getattr(model_g, "chouwagan_discrete", None)
    chouwagan_prior_loss_weight = max(
        0.0,
        float(getattr(config.train, "chouwagan_prior_loss_weight", PRIOR_LOSS_WEIGHT_DEFAULT)),
    )
    chouwagan_prior_warmup_steps = 0 if finetune_phase else max(
        0,
        int(getattr(config.train, "chouwagan_prior_warmup_steps", PRIOR_WARMUP_STEPS_DEFAULT)),
    )
    # When ``chouwagan_svae_kl_target_*`` is set, the rate controller owns the
    # KL weight and discards both settings above (see the
    # ``kl_rate_control_active`` test at the loss site).  Silently ignored
    # config is worse than no config: a run tuned by turning these knobs would
    # have produced identical training, and the lever that actually moves the
    # prior/posterior balance is the KL target.
    #
    # Only when they differ from their defaults, though.  The shipped config
    # carries both keys at their default values, so testing for the key's
    # presence fired this on every ChouwaGAN run and warned about a choice
    # nobody made -- which is how a warning becomes something people scroll
    # past.  A value equal to the default says nothing about intent.
    if (
        rank == 0
        and chouwagan_discrete is not None
        and chouwagan_discrete.kl_rate_control_active
    ):
        overridden = []
        if chouwagan_prior_loss_weight != PRIOR_LOSS_WEIGHT_DEFAULT:
            overridden.append(f"chouwagan_prior_loss_weight={chouwagan_prior_loss_weight}")
        if (
            not finetune_phase
            and chouwagan_prior_warmup_steps != PRIOR_WARMUP_STEPS_DEFAULT
        ):
            overridden.append(f"chouwagan_prior_warmup_steps={chouwagan_prior_warmup_steps}")
        if overridden:
            warning(
                f"KL rate control is active (targets slow="
                f"{chouwagan_discrete.kl_target_slow}, fast="
                f"{chouwagan_discrete.kl_target_fast}), so it sets the KL weight "
                f"itself and ignores {', '.join(overridden)}. Change "
                "chouwagan_svae_kl_target_slow/_fast to move the "
                "prior/posterior balance.",
                tag="[INIT]",
            )
    chouwagan_ablation_interval = 0 if finetune_phase else max(
        0,
        int(getattr(config.train, "chouwagan_ablation_interval", 0)),
    )
    chouwagan_ablation_weight = max(
        0.0,
        float(getattr(config.train, "chouwagan_ablation_loss_weight", 0.0)),
    )
    chouwagan_ablation_margin = max(
        0.0,
        float(getattr(config.train, "chouwagan_ablation_margin", 0.01)),
    )
    chouwagan_usage_loss_interval = max(
        1,
        int(getattr(config.train, "chouwagan_usage_loss_interval", 16)),
    )
    kl_active_threshold = max(
        0.0,
        float(getattr(config.train, "kl_active_threshold", 0.01)),
    )
    adaptive_adv_interval = max(
        1,
        int(getattr(config.train, "chouwagan_adaptive_adv_interval", 8)),
    )

    # ---- Adversarial balance --------------------------------------------
    # The adaptive weight equalises the adversarial and reconstruction
    # gradients measured on the decoder's last layer, scaled by the balance
    # target.  Its ceiling used to be 1.0, which made it a cap rather than a
    # balance: with c_mel at 45 the reconstruction gradient runs one to two
    # orders of magnitude above the adversarial one, so the rule asked for ~30
    # and was handed 1 for the entire run.  Adversarial pressure is the only
    # term that teaches high-frequency detail and aperiodic texture -- a mel
    # distance is nearly blind to both -- so capping it there caps the ceiling
    # on final quality.
    adaptive_adv_balance = max(
        0.0,
        float(getattr(config.train, "chouwagan_adv_balance_target", 0.5)),
    )
    adaptive_adv_min = max(
        0.0,
        float(getattr(config.train, "chouwagan_adaptive_adv_min", 0.01)),
    )
    # 8.0 is a judgement call, not a derived constant: it is roughly a quarter
    # of the balance the rule currently requests, an ~8x increase on the old
    # ceiling, and bounded so a collapsing discriminator cannot run the weight
    # away.  Raise it if the high-frequency deficit persists once the ramp has
    # finished; lower it if loss_disc_real starts falling well below 1.0.
    adaptive_adv_max = max(
        adaptive_adv_min,
        float(getattr(config.train, "chouwagan_adaptive_adv_max", 8.0)),
    )
    adaptive_adv_ramp_start = max(
        0,
        int(getattr(config.train, "chouwagan_adaptive_adv_ramp_start", 20000)),
    )
    adaptive_adv_ramp_steps = max(
        0,
        int(getattr(config.train, "chouwagan_adaptive_adv_ramp_steps", 80000)),
    )
    # A fine-tune never reaches the pretraining ramp's horizon -- a few thousand
    # steps is typical -- so it gets its own, short one.  It still gets a ramp
    # rather than the full ceiling at once: the loaded G and D were equilibrated
    # under whatever weight trained them, and an 8x step change in the objective
    # on the first batch is how a fine-tune picks up artefacts it never had.
    adaptive_adv_ramp_steps_finetune = max(
        0,
        int(getattr(config.train, "chouwagan_adaptive_adv_ramp_steps_finetune", 2000)),
    )

    # Feature matching is the other channel that carries texture, and it was
    # hardcoded to 1.0 while being normalised by branch count -- roughly half
    # of HiFi-GAN's convention of 2.0 unnormalised, on a term already three
    # orders of magnitude below the spectral loss.
    chouwagan_fm_weight = max(
        0.0,
        float(getattr(config.train, "chouwagan_fm_weight", 2.0)),
    )

    # Only read by the "Hybrid L1" spectral loss.  Left at 1.0 so the option
    # behaves exactly as before unless it is raised deliberately.
    ms_stft_weight = max(
        0.0,
        float(getattr(config.train, "chouwagan_ms_stft_weight", 1.0)),
    )

    # ---- Waveform-domain reconstruction terms ---------------------------
    # The compressed-mel loss constrains magnitude per frame but says nothing
    # about the local level envelope, which is where a bounded output head
    # tends to drift.  Both terms below are relative (log ratios / max-pool
    # differences), so neither assumes a particular dataset normalisation.
    envelope_loss_weight = max(
        0.0, float(getattr(config.train, "chouwagan_envelope_loss_weight", 3.0))
    )
    envelope_kernel = max(
        2, int(getattr(config.train, "chouwagan_envelope_kernel", 100))
    )
    envelope_stride = max(
        1, int(getattr(config.train, "chouwagan_envelope_stride", 50))
    )
    # Amplitude below which the companded envelope stops resolving detail.
    # 1e-3 is about -60 dBFS, under any decay tail worth reproducing.
    envelope_floor = max(
        1e-8, float(getattr(config.train, "chouwagan_envelope_floor", 1e-3))
    )
    rms_loss_weight = max(
        0.0, float(getattr(config.train, "chouwagan_rms_loss_weight", 5.0))
    )
    rms_window_size = max(
        16, int(getattr(config.train, "chouwagan_rms_window", 1024))
    )
    rms_hop_size = max(1, int(getattr(config.train, "chouwagan_rms_hop", 256)))
    # One-sided and absolute: only useful when the dataset is peak-normalised
    # below the head's threshold.  Off by default.
    peak_headroom_weight = max(
        0.0, float(getattr(config.train, "chouwagan_peak_headroom_weight", 0.0))
    )
    peak_headroom_threshold = float(
        getattr(config.train, "chouwagan_peak_headroom_threshold", 0.85)
    )

    cached_adaptive_adv = None
    cached_rec_grad = None
    cached_adv_grad = None
    cached_adv_requested = None

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
            # loosen later) was removed: with ChouwaGAN the global norm below
            # already bounds both nets, and it never bound anything in practice
            # -- ``grad_clip_hit_rate_g`` sat at 0 with grad norms 4x under the
            # cap.  What remains is the manual override and that global bound.
            if clip_grad_norm_override:
                grad_clip_value_g = clip_grad_norm_override_value_g
                grad_clip_value_d = clip_grad_norm_override_value_d
            else:
                grad_clip_value_g = grad_clip_value_d = float("inf")

            if chouwagan_active and chouwagan_global_clip_g > 0:
                grad_clip_value_g = min(grad_clip_value_g, chouwagan_global_clip_g)
                grad_clip_value_d = min(grad_clip_value_d, chouwagan_global_clip_g)


            # Device handling
            if device.type == "cuda":
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]

            # Batch unpacking
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = info

            model_g = net_g.module if hasattr(net_g, "module") else net_g
            if hasattr(model_g, "set_training_step"):
                model_g.set_training_step(global_step)

            # Generator main forward pass:
            with autocast(device_type="cuda", enabled=False):
                model_output = net_g(spec, spec_lengths, sid, phone, phone_lengths, pitchf, pitch)

                y_hat, ids_slice, x_mask, z_mask, vae_parts = model_output

                discrete_parts = None
                if chouwagan_discrete is not None:
                    discrete_parts = vae_parts
                    z = z_p = m_q = logs_q = None
                    m_p = None
                    logs_p = None
                else:
                    # Gaussian latent samples and parameters used by the VITS ELBO.
                    z, z_p, m_p, logs_p, m_q, logs_q = vae_parts

                # Slice the original waveform ( y ) to match the generated slice:
                y = commons.slice_segments(y, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)



            # Discriminator update
            main_real_spectrograms = None
            if chouwagan_active:
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                with torch.no_grad(), autocast(device_type="cuda", enabled=False):
                    main_real_spectrograms = (
                        discriminator_model.prepare_spectrograms(y)
                    )

            d_updates = [(y, y_hat.detach(), main_real_spectrograms)]

            # The lazy R1 step runs its own backward and its own optimizer step
            # with a penalty scaled by ``r1_gamma * r1_interval``.  Averaging its
            # gradient norm into the adversarial one hides which of the two is
            # actually growing, so they are tracked as separate series.
            _loss_disc_acc, _loss_disc_real_acc, _loss_disc_fake_acc, _grad_norm_d_acc = [], [], [], []
            grad_norm_d_r1 = None

            for y_d_real, y_d_fake, real_spectrograms in d_updates:
                with autocast(device_type="cuda", enabled=False):
                    if chouwagan_active:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real,
                            y_d_fake,
                            real_spectrograms=real_spectrograms,
                            pair_batches=True,
                            san_training=True,
                        )
                    else:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real, y_d_fake
                        )

                with autocast(device_type="cuda", enabled=False):
                    loss_disc, loss_disc_real, loss_disc_fake = discriminator_loss(
                        y_d_hat_r,
                        y_d_hat_g,
                        san_direction_weight=chouwagan_san_direction_weight
                        if chouwagan_active
                        else 1.0,
                        normalize=chouwagan_active,
                    )

                optim_d.zero_grad(set_to_none=True)
                loss_disc.backward()
                grad_norm_d = _clip_or_sample_grad_norm(
                    net_d.parameters(),
                    grad_clip_value_d,
                    global_step,
                    metrics_update_interval,
                )
                optim_d.step()
                _normalize_san_weights(net_d)

                # Temp accumulation
                _loss_disc_acc.append(loss_disc.detach())
                _loss_disc_real_acc.append(loss_disc_real.detach())
                _loss_disc_fake_acc.append(loss_disc_fake.detach())
                if grad_norm_d is not None:
                    _grad_norm_d_acc.append(grad_norm_d)

            if (
                chouwagan_active
                and r1_gamma > 0
                and global_step % r1_interval == 0
            ):
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                r1_event = max(0, global_step // r1_interval - 1)
                r1_branch = r1_event % discriminator_model.num_branches
                with autocast(device_type="cuda", enabled=False):
                    r1_penalty = _lazy_r1_penalty(
                        net_d,
                        y,
                        r1_branch,
                        r1_segment_size,
                    )
                    loss_r1 = r1_penalty * (r1_gamma * r1_interval * 0.5)

                optim_d.zero_grad(set_to_none=True)
                loss_r1.backward()
                grad_norm_d_r1 = _clip_or_sample_grad_norm(
                    net_d.parameters(),
                    grad_clip_value_d,
                    global_step,
                    metrics_update_interval,
                )
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

            optim_d.zero_grad(set_to_none=True)

            # Run discriminator on generated output
            discriminator_parameter_states = None
            if chouwagan_active:
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                discriminator_parameter_states = [
                    parameter.requires_grad
                    for parameter in discriminator_model.parameters()
                ]
                for parameter in discriminator_model.parameters():
                    parameter.requires_grad_(False)

                with autocast(device_type="cuda", enabled=False):
                    with torch.no_grad():
                        _, fmap_r = discriminator_model._forward_audio(
                            y, main_real_spectrograms
                        )
                    y_d_hat_g, fmap_g = discriminator_model._forward_audio(y_hat)
            else:
                with autocast(device_type="cuda", enabled=False):
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)


            # Compute generator losses:
            loss_ablation = torch.zeros((), device=y.device)
            ablation_delta = None
            ablation_branch = None
            ablation_dimension = None
            with autocast(device_type="cuda", enabled=False):

                # Spectral loss.  The component terms are logged separately
                # where a mode has more than one, because the combined series
                # cannot show which half is actually moving -- and for
                # "Hybrid L1" the balance between them is the thing to tune.
                loss_spectral_parts: dict[str, torch.Tensor] = {}
                if spectral_loss == "L1 Mel Loss":
                    y_mel = wave_to_mel(
                        config, y, num_mels=None,
                        for_loss=chouwagan_active,
                    )
                    y_hat_mel = wave_to_mel(
                        config, y_hat, num_mels=None,
                        for_loss=chouwagan_active,
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
                        for_loss=chouwagan_active,
                    )
                    y_hat_mel = wave_to_mel(
                        config, y_hat, num_mels=None,
                        for_loss=chouwagan_active,
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
                    # ``chouwagan_ms_stft_weight`` is high enough for the
                    # MS-STFT term to matter beside a mel term carrying c_mel.
                    loss_spectral_parts = {
                        "loss_spectral_l1_mel": loss_l1_mel,
                        "loss_spectral_ms_stft": loss_ms_stft,
                    }

                loss_envelope = torch.zeros((), device=y.device)
                loss_rms = torch.zeros((), device=y.device)
                loss_peak = torch.zeros((), device=y.device)
                if chouwagan_active:
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

                # The probe is instrumentation first and a loss second, so it
                # is gated on the interval alone.  It used to also require
                # ``chouwagan_ablation_weight > 0``, which meant the only way to
                # stop paying for a term that does nothing was to also go blind
                # to the one measurement that says whether a latent dimension is
                # load-bearing.  ``loss_ablation`` is still scaled by the weight
                # below, so a weight of zero costs exactly the forward pass.
                if (
                    chouwagan_discrete is not None
                    and chouwagan_ablation_interval > 0
                    and global_step % chouwagan_ablation_interval == 0
                ):
                    slow_dimension_count = len(chouwagan_discrete.slow_levels)
                    fast_dimension_count = len(chouwagan_discrete.fast_levels)
                    total_dimension_count = slow_dimension_count + fast_dimension_count
                    selected_dimension = int(
                        torch.randint(
                            total_dimension_count,
                            (),
                            device=y.device,
                        ).item()
                    )
                    if selected_dimension < slow_dimension_count:
                        ablation_branch = "slow"
                        ablation_dimension = selected_dimension
                    else:
                        ablation_branch = "fast"
                        ablation_dimension = selected_dimension - slow_dimension_count
                    ablation_raw, ablation_delta, _ = _chouwagan_ablation_margin(
                        model_g,
                        chouwagan_discrete,
                        discrete_parts,
                        ids_slice,
                        config.train.segment_size // config.data.hop_length,
                        pitchf,
                        sid,
                        y,
                        y_hat,
                        ablation_branch,
                        ablation_dimension,
                        chouwagan_ablation_margin,
                    )
                    loss_ablation = ablation_raw * chouwagan_ablation_weight

                # Feature Matching loss
                loss_fm = feature_loss(
                    fmap_r,
                    fmap_g,
                    normalize=chouwagan_active,
                ) * (chouwagan_fm_weight if chouwagan_active else 2.0)

                # Generator loss
                loss_adv = generator_loss(
                    y_d_hat_g,
                    normalize=chouwagan_active,
                    san_direction_weight=(
                        chouwagan_san_direction_weight if chouwagan_active else 1.0
                    ),
                    use_softplus=chouwagan_active,
                )

                loss_prior_slow = torch.zeros((), device=y.device)
                loss_prior_fast = torch.zeros((), device=y.device)
                loss_prior = torch.zeros((), device=y.device)
                loss_coarse_spectral = torch.zeros((), device=y.device)
                loss_usage = torch.zeros((), device=y.device)
                usage_diagnostics = {}
                discrete_diagnostics = {}
                if chouwagan_discrete is not None:
                    loss_prior_slow, loss_prior_fast, raw_prior_loss = (
                        chouwagan_discrete.prior_losses(discrete_parts, x_mask)
                    )
                    if chouwagan_discrete.kl_rate_control_active:
                        # The rate controller owns the KL weight: applying the
                        # fixed weight and warmup on top would fight its own
                        # feedback loop.
                        loss_prior = raw_prior_loss
                    else:
                        if chouwagan_prior_warmup_steps:
                            prior_progress = min(1.0, global_step / chouwagan_prior_warmup_steps)
                        else:
                            prior_progress = 1.0
                        loss_prior = raw_prior_loss * chouwagan_prior_loss_weight * prior_progress
                    loss_coarse_spectral = (
                        discrete_parts["coarse_spectral_loss"]
                        * chouwagan_discrete.coarse_spectral_loss_weight
                    )
                    if (
                        chouwagan_discrete.usage_loss_weight > 0.0
                        and global_step % chouwagan_usage_loss_interval == 0
                    ):
                        loss_usage, usage_diagnostics = chouwagan_discrete.usage_regularization(
                            discrete_parts,
                            x_mask,
                        )
                    if global_step % chouwagan_diagnostics_interval == 0:
                        discrete_diagnostics = chouwagan_discrete.diagnostics(
                            discrete_parts,
                            x_mask,
                        )
                    loss_kl = torch.zeros((), device=y.device)
                else:
                    loss_kl = kl_loss(
                        z_p,
                        logs_q,
                        m_p,
                        logs_p,
                        z_mask,
                    ) * config.train.c_kl

                    # KL diagnostic: per-dimension raw divergence for the
                    # Gaussian VITS path.  The discrete Chouwa path does not
                    # enter this branch.
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
                last_layer_rec_grad = torch.zeros((), device=y.device)
                last_layer_adv_grad = torch.zeros((), device=y.device)
                adaptive_adv_requested = torch.zeros((), device=y.device)
                # Gated on discriminator health rather than on the step count
                # alone -- see ``_AdversarialCeilingGovernor``.  ``loss_disc``
                # is this step's, since the discriminator update ran above.
                adaptive_adv_ceiling = adversarial_ceiling_governor.update(
                    phase_step if finetune_phase else global_step,
                    loss_disc.item() if chouwagan_active else None,
                )
                if chouwagan_active:
                    # Each refresh costs two extra autograd traversals (one of
                    # them through the whole discriminator), measured at +16% of
                    # step time.  The weight is a slowly-varying scalar that is
                    # clamped to [0.01, 1.0], so recomputing it periodically and
                    # reusing it in between buys that back.
                    if (
                        cached_adaptive_adv is None
                        or global_step % adaptive_adv_interval == 0
                    ):
                        last_layer = _last_layer_parameter(model_g.dec.conv_post)
                        (
                            cached_adaptive_adv,
                            cached_rec_grad,
                            cached_adv_grad,
                            cached_adv_requested,
                        ) = _adaptive_adversarial_weight(
                            # The waveform terms are part of the reconstruction
                            # objective, so the GAN balance has to see them too.
                            loss_spectral + loss_waveform,
                            loss_adv,
                            last_layer,
                            balance_target=adaptive_adv_balance,
                            minimum=adaptive_adv_min,
                            maximum=adaptive_adv_ceiling,
                        )
                    # The cached weight was clamped against the ceiling in force
                    # when it was computed; re-clamp so a ceiling that moved in
                    # between takes effect on the very next step.
                    adaptive_adv = cached_adaptive_adv.clamp(
                        min=adaptive_adv_min, max=adaptive_adv_ceiling
                    )
                    last_layer_rec_grad = cached_rec_grad
                    last_layer_adv_grad = cached_adv_grad
                    adaptive_adv_requested = cached_adv_requested
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
                    + loss_prior
                    + loss_coarse_spectral
                    + loss_usage
                    + loss_ablation
                )
                loss_gan = (
                    adaptive_adv * loss_adv + loss_fm
                ) * gan_weight
                loss_gen_total = loss_core + loss_gan
                if rank == 0 and ablation_delta is not None:
                    writer.add_scalar(
                        f"diag/ablation_delta/{ablation_branch}_dim_{ablation_dimension}",
                        ablation_delta.item(),
                        global_step,
                    )

            # Generator backward and update:
            optim_g.zero_grad(set_to_none=True)
            module_grad_metrics = {}
            if (
                chouwagan_active
                and rank == 0
                and global_step % grad_source_probe_interval == 0
            ):
                decoder_parameters = _decoder_parameters(net_g)
                grad_source_losses = {
                    "spectral": loss_spectral,
                    "adv": loss_adv * gan_weight,
                    "fm": loss_fm * gan_weight,
                }
                if loss_waveform.requires_grad:
                    grad_source_losses["waveform"] = loss_waveform
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
            if chouwagan_active:
                loss_core.backward(retain_graph=True)
                _add_decoder_only_gradients(loss_gan, net_g)
                if global_step % metrics_update_interval == 0:
                    module_grad_metrics = _generator_gradient_metrics(net_g)
                grad_norm_g = _clip_or_sample_grad_norm(
                    net_g.parameters(),
                    grad_clip_value_g,
                    global_step,
                    metrics_update_interval,
                )
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
                    if chouwagan_active:
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
                        writer.add_scalar(
                            "GAN/adaptive_adv_saturation",
                            min(
                                1.0,
                                adaptive_adv.item()
                                / max(1e-6, adaptive_adv_requested.item()),
                            ),
                            global_step,
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
                    writer.add_scalar(
                        "GAN/adv_to_rec_ratio",
                        (
                            last_layer_adv_grad
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
                loss_gen_total.backward() # Loss backward
                if global_step % metrics_update_interval == 0:
                    module_grad_metrics = _generator_gradient_metrics(net_g)
                grad_norm_g = _clip_or_sample_grad_norm(
                    net_g.parameters(),
                    grad_clip_value_g,
                    global_step,
                    metrics_update_interval,
                )
                optim_g.step() # Optim step

            if discriminator_parameter_states is not None:
                for parameter, requires_grad in zip(
                    discriminator_model.parameters(),
                    discriminator_parameter_states,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)


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
                and holdout_interval > 0
                and global_step > 0
                and global_step % holdout_interval == 0
            ):
                # With an EMA, both halves of the detector get better and for
                # different reasons.  The curve it scores is the average rather
                # than a single oscillating step, so the minimum is far better
                # localised and ``patience`` stops being mostly a noise margin.
                # And the weights it keeps are the average, which for a GAN
                # vocoder is normally better than any step it was made from --
                # so the thing being measured is also the thing worth keeping.
                if ema is not None:
                    with ema.applied(net_g) as averaged:
                        holdout_loss = _holdout_spectral_loss(
                            averaged, holdout_batches, config, device
                        )
                    improved = overtrain_monitor.update(
                        ema, holdout_loss, global_step
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
                        holdout_loss = _holdout_spectral_loss(
                            net_g, holdout_batches, config, device
                        )
                        improved = overtrain_monitor.update(
                            net_g.module if hasattr(net_g, "module") else net_g,
                            holdout_loss,
                            global_step,
                        )
                if rank == 0 and math.isfinite(holdout_loss):
                    writer.add_scalar("holdout/mel_l1", holdout_loss, global_step)
                    writer.add_scalar("holdout/best", overtrain_monitor.best, global_step)
                    writer.add_scalar(
                        "holdout/evals_since_best",
                        overtrain_monitor.since_best,
                        global_step,
                    )
                    marker = "*" if improved else " "
                    print(
                        f"[HOLDOUT]{marker} step {global_step}: {holdout_loss:.5f}  "
                        f"(best {overtrain_monitor.best:.5f} @ {overtrain_monitor.best_step}, "
                        f"{overtrain_monitor.since_best}/{overtrain_monitor.patience} since)"
                    )
                if overtrain_monitor.overtrained and not overtrain_flagged:
                    overtrain_flagged = True
                    if rank == 0:
                        print(
                            f"\n[OVERTRAIN] Held-out loss has not improved for "
                            f"{overtrain_monitor.since_best} evaluations. "
                            f"The last good weights are step "
                            f"{overtrain_monitor.best_step} ({overtrain_monitor.best:.5f}); "
                            f"they will be the ones exported."
                        )
                        if stop_on_overtrain:
                            print("[OVERTRAIN] Stopping (stop_on_overtrain is on).")


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
            avg_rolling_cache["loss_prior"].append(loss_prior.detach())
            avg_rolling_cache["loss_prior_slow"].append(loss_prior_slow.detach())
            avg_rolling_cache["loss_prior_fast"].append(loss_prior_fast.detach())
            if chouwagan_active:
                avg_rolling_cache["loss_envelope"].append(loss_envelope.detach())
                avg_rolling_cache["loss_rms"].append(loss_rms.detach())
                avg_rolling_cache["loss_peak"].append(loss_peak.detach())
                avg_rolling_cache["loss_waveform"].append(loss_waveform.detach())
            else:
                avg_rolling_cache["loss_kl"].append(loss_kl.detach())
            if discrete_diagnostics:
                for key in (
                    "prior_kl_slow",
                    "prior_kl_fast",
                    "prior_std_slow",
                    "prior_std_fast",
                    "posterior_std_slow",
                    "posterior_std_fast",
                    "scale_anchor",
                    "prior_replacement",
                    "content_rms",
                    "posterior_detail_rms",
                    "prior_detail_rms",
                    "kl_effective_dims_slow",
                    "kl_effective_dims_fast",
                    "kl_above_floor_slow",
                    "kl_above_floor_fast",
                    "kl_median_slow",
                    "kl_median_fast",
                ):
                    if key in discrete_diagnostics:
                        avg_rolling_cache[key].append(
                            discrete_diagnostics[key].detach()
                        )
                for key in ("kl_beta_slow", "kl_beta_fast"):
                    if key in discrete_diagnostics:
                        avg_rolling_cache.setdefault(
                            key,
                            deque(maxlen=rolling_loss_steps),
                        ).append(discrete_diagnostics[key].detach())
                for key, value in discrete_diagnostics.items():
                    if key.endswith("_per_dim"):
                        discrete_vector_cache.setdefault(
                            key,
                            deque(maxlen=rolling_loss_steps),
                        ).append(value.detach())

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

                for key, values in discrete_vector_cache.items():
                    if not values:
                        continue
                    vector = torch.stack(list(values)).mean(dim=0)
                    for dimension, value in enumerate(vector):
                        writer.add_scalar(
                            f"diag/{key}/dim_{dimension}",
                            value.item(),
                            global_step,
                        )
                discrete_vector_cache.clear()

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
                    save_checkpoint(net_d, optim_d, config.train.learning_rate_d, epoch, d_path)


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
