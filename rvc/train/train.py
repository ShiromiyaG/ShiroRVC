import os
import glob
import math
import re
import socket
import sys

from collections import deque
from contextlib import nullcontext

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

# ``expandable_segments`` lets the caching allocator grow one virtual segment
# instead of handing out fixed-size blocks, which is what stops a large
# transient request from failing against a heap that has enough free memory but
# none of it contiguous.  The requests that hit this are cuDNN's benchmark
# workspaces: the collate pads to the per-batch max, so nearly every step is a
# new input shape, cuDNN re-autotunes for each one, and autotuning probes
# algorithms whose workspaces run to gigabytes.  Without this the allocator
# recovers by flushing its whole cache and retrying -- training survives, but
# pays a full re-warm each time.
#
# Linux-only: the backend is unimplemented on Windows, where setting it makes
# the allocator raise on the first allocation rather than fall back.  An
# existing value is left alone so the setting stays overridable from outside.
if sys.platform.startswith("linux") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cuda, cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
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
from rvc.train.overtrain import (
    OvertrainMonitor,
    carve_holdout,
    deliverable_weights,
    holdout_metrics_resilient,
    materialise_holdout,
)
from rvc.train.diagnostics import (
    branch_separation,
    cache_mean,
    clip_or_sample_grad_norm,
    generator_gradient_metrics,
    prior_gap,
    split_branch_outputs,
)
from rvc.train.progress import EpochRecorder, emit_machine_progress
from rvc.train.schedules import (
    fit_eval_interval,
    planned_step_count,
    prepare_schedulers,
)
from rvc.train.setup import (
    apply_resume_lr_override,
    apply_training_freezes,
    assert_resumable_architecture,
    checkpoint_step_from_path,
    enable_discriminator_compile,
    enable_frontend_compile,
    enable_vocoder_compile,
    get_d_model,
    get_g_model,
    get_optimizers,
    normalize_san_weights,
    setup_models_for_training,
)
from rvc.train.stop import (
    finish_stop,
    install_stop_handlers,
    stop_was_requested,
    uninterruptible_save,
)
from rvc.train.optimizers import (
    averaged_weights,
    is_schedule_free,
)
from rvc.train.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_FALLBACK_NAMESPACE,
    TENSORBOARD_MEDIA_SOURCE_NAME,
)

install_rich_print()

from utils import (
    summarize,
    assert_decoder_layout_matches,
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
    mel_frequency_tilt_weights,
    BandWeightedSpectralLoss,
    HighFrequencyFloorNegative,
)

from mel_processing import build_ms_mel_loss, spectrogram_torch

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.configs.vocoders import normalize_vocoder
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
freeze_mode = spec.freeze_mode
c_kl_scale = spec.c_kl_scale
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
# Belongs to the vocoder, not the run: a single-scale mel cannot resolve a
# harmonic comb far above ~2 kHz, so RefineGAN wants the multi-scale mel while
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
# Prior draw for the preview render.  Absent means the model's own
# ``prior_noise_scale``, which is what inference decodes with.
preview_noise_scale = getattr(config.train, "preview_noise_scale", None)
preview_noise_scale = (
    None if preview_noise_scale is None else float(preview_noise_scale)
)
# Posterior draw for the stage-1 reconstruction preview.  0 decodes ``m_q``,
# which is what the holdout scorer already does (``holdout_noise_scale``);
# raise it to see what the latent's own noise adds to the render.
preview_posterior_noise_scale = float(
    getattr(config.train, "preview_posterior_noise_scale", 0.0)
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
# True when this run continues from a G_/D_ checkpoint in the experiment folder
# rather than starting from the pretrains (or from scratch).
resumed_run = False
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
# ( enc_p + enc_q + flow ).  Legacy spelling of `freeze_mode = "frontend"`,
# kept because it is edited here by hand; when True it wins over the spec's
# `freeze_mode`.  For a staged pretrain use the spec field instead -- see
# `rvc.train.setup.FREEZE_MODES` and `tools/pretrain_stage1_vocoder.py`.
freeze_vae = False # If true, lets only vocoder ( dec ), spk embedding and discriminator learn

# ----  Global LR scales  ----
# Multipliers of the base LR ( 0.1 = 10%, 1.0 = 100% ).
# These four are edited here by hand *or* carried on the run spec, whichever
# is set: the spec wins when it names a value, so a launcher can set them
# per-run ( see `tools/pretrain_stage3_endtoend.py` ) without this file being
# edited, and an edit here still works when the spec leaves them unset.
dec_lr_scale = None  # everything under `dec.` ( the decoder/vocoder )
vae_lr_scale = None  # everything else ( frontend + emb_g ), see `freeze_vae` above

# ----  Resume LR override  ----
resume_lr = None  # e.g. 5e-5 ( None = Override disabled. )
resume_lr_target = "full"  # Pick what you want it applied to: "g", "d" or "full" where full refers to both G/D

if spec.dec_lr_scale is not None:
    dec_lr_scale = spec.dec_lr_scale
if spec.vae_lr_scale is not None:
    vae_lr_scale = spec.vae_lr_scale
if spec.resume_lr is not None:
    resume_lr = spec.resume_lr
    resume_lr_target = spec.resume_lr_target

# True = Gamma is applied as-is each step.
# False = Per-epoch budget, re-binned to per-step so the total decay per epoch equals "exp decay epoch" ( VITS-style ) ~ Default
exp_decay_step_raw = False

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        model = net_g.module if hasattr(net_g, "module") else net_g
        o, *_ = model.infer(*reference, noise_scale=preview_noise_scale)
    net_g.train()
    return o


def eval_reconstruct(net_g, reference, reference_audio, config):
    """Render the preview through the posterior -- the path the stage optimises.

    ``eval_infer`` decodes ``flow(m_p + exp(logs_p) * randn * noise_scale)``,
    so it reads ``enc_p`` and ``flow``.  Stage 1 (``freeze_mode="vocoder"``)
    freezes exactly those two, and from a scratch pretrain they are still at
    their initialisation: the preview then showed a decoder fed an untrained
    prior's draw, while the loss the stage minimises goes through
    ``enc_q(spec)``.  Two different inputs, so the image and ``loss_spectral``
    described different things -- and the mottle the prior draw leaves between
    the harmonics is the prior's, not the decoder's.  ``source_gain``
    multiplies the excitation by an envelope projected from ``z``, which is
    what turns a per-frame-white ``z`` into a sideband on every partial.

    Rendered unsliced, unlike the training forward, which decodes one
    ``segment_size`` slice: a defect that depends on render length is invisible
    in 0.4 s.

    Decoded from ``m_q``, like the holdout scorer, unless
    ``preview_posterior_noise_scale`` says otherwise.  The draw is independent
    per frame, so at 1.0 it lands in the image as broadband flutter -- the
    decoder's own defects and the latent's noise, in one picture.

    The reconstruction target is ``reference_audio``, so this needs one; the
    custom-reference branch without ``ref_audio.wav`` has no target mel to
    compare against either and keeps the ``infer`` render.
    """
    # ``phone``/``pitch`` are the prior's inputs and are deliberately unused:
    # this render does not consult ``enc_p`` at all.
    _phone, _phone_lengths, _pitch, pitchf, sid, seed = reference

    net_g.eval()
    with torch.no_grad():
        model = net_g.module if hasattr(net_g, "module") else net_g
        # Only bites when ``preview_posterior_noise_scale`` is non-zero: a
        # preview that moves for two reasons cannot be compared against the
        # previous epoch's.
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # The spectrogram ``enc_q`` was trained on, built the way the
        # dataloader builds it (``rvc/train/data_utils.py``): same window, same
        # ``center=False``, so one frame is one hop here as well.
        spec = spectrogram_torch(
            reference_audio.squeeze(1),
            config.data.filter_length,
            config.data.hop_length,
            config.data.win_length,
            center=False,
        )
        # f0 and the spectrogram are cropped independently -- the custom
        # reference trims to the shortest of its three files, and a short
        # ``ref_audio.wav`` is only warned about -- so decode the prefix they
        # share rather than trusting the two to agree.
        frames = min(int(spec.shape[-1]), int(pitchf.shape[-1]))
        spec = spec[..., :frames]
        pitchf = pitchf[..., :frames]
        spec_lengths = torch.LongTensor([frames]).to(spec.device)

        g = model.emb_g(sid).unsqueeze(-1)
        _z, m_q, logs_q, spec_mask = model.enc_q(spec, spec_lengths, g=g)
        # The draw is independent per frame, and this decoder renders it as
        # bursts between the harmonics -- the same thing ``prior_noise_scale``
        # was lowered for on the prior side.
        if preview_posterior_noise_scale:
            z = m_q + (
                torch.randn_like(m_q)
                * torch.exp(logs_q)
                * preview_posterior_noise_scale
            )
        else:
            z = m_q
        o = model.dec(z * spec_mask, pitchf, g=g)
    net_g.train()
    return o


def eval_preview(net_g, reference, reference_audio, config):
    """Render the preview by whichever path this stage's loss describes.

    Stage 1 holds ``enc_p`` and ``flow``, so ``infer`` measures neither what it
    trains nor what it scores -- see ``eval_reconstruct``.  Every other stage
    trains the prior, and there the ``infer`` render is the honest one: it is
    what conversion will actually run.
    """
    if freeze_mode == "vocoder" and reference_audio is not None:
        if not getattr(eval_preview, "_announced", False):
            info(
                "freeze_mode='vocoder': previews render from the posterior "
                "(enc_q -> dec), which is what this stage optimises. The "
                "infer path reads the frozen, untrained prior.",
                tag="[PREVIEW]",
            )
            eval_preview._announced = True
        return eval_reconstruct(net_g, reference, reference_audio, config)
    return eval_infer(net_g, reference)


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
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data, n_mel_bins=config.model.inter_channels)

    # Carve a held-out set out of the dataset before anything else sees it;
    # everything that scores it lives in ``rvc.train.overtrain``.
    holdout_dataset = carve_holdout(
        config, train_dataset, rank, enabled=overtrain_detector
    )

    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    # Quantise the padded batch length to a grid instead of the exact per-batch
    # maximum.  The buckets above are 100 frames wide, so a 32-frame grid turns
    # each of them into ~3 shapes rather than one per batch -- enough for
    # ``cudnn.benchmark`` and ``torch.compile`` to stay warm, while the padding
    # it adds averages 16 frames against bucket lengths of hundreds.  The true
    # lengths still ride along in the batch, so nothing downstream sees the
    # difference.
    collate_fn = TextAudioCollateMultiNSFsid(
        pad_multiple=int(getattr(config.data, "pad_multiple", 32)),
        hop_length=config.data.hop_length,
    )
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

    holdout_set = None
    probe_set = None
    if holdout_dataset is not None and rank == 0:
        holdout_set, probe_set = materialise_holdout(
            config, train_dataset, holdout_dataset
        )

    return train_loader, holdout_set, probe_set


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


def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, sample_rate, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank, reset_pretrained_embeddings=False):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    resumed_g_path = None
    # Training-loop controller state travelling with the D checkpoint.  Empty on
    # a fresh run, on a pretrained start, and on every checkpoint written before
    # the key existed -- all three of which mean "start the controller cold".
    resumed_extra_d = {}
    global reset_optimizer_for_run, resumed_run
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
            # A checkpoint of this run's own is a resume, pretrains or not: the
            # fine-tune reset (fresh optimizer, epoch 1) belongs to the run that
            # starts from the pretrain, not to every restart after it.
            resumed_run = True
            reset_optimizer_for_run = False

            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

            # Apply decoder / frontend layer freezes for the selected phase.
            apply_training_freezes(net_g, rank, freeze_vae=freeze_vae, freeze_mode=freeze_mode)

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, dec_lr_scale=dec_lr_scale, vae_lr_scale=vae_lr_scale)

            # Resuming loads the generator non-strictly for the VITS-latent
            # vocoders, so nothing downstream would complain if the checkpoint
            # in this folder belonged to a different decoder: the frontend
            # weights match either way, the decoder is silently left at its
            # init, and the run restarts at the old step count with a fully
            # trained discriminator against a random generator.  The vocoders
            # share a config shape and a folder layout, which makes that one
            # edited ``vocoder`` field away.
            assert_resumable_architecture(net_g, g_checkpoint_path)

            # Load the model and optim states
            generator_strict_load = strict_load
            _, _, _, epoch_str, _ = load_checkpoint(
                g_checkpoint_path,
                net_g,
                optim_g,
                generator_strict_load,
            )
            _, _, _, _, extra_d = load_checkpoint(
                d_checkpoint_path,
                net_d,
                optim_d,
                strict_load,
            )
            resumed_extra_d = extra_d or {}

            # resume_lr re-anchors G and/or D to the given base LR.
            apply_resume_lr_override(
                optim_g, optim_d, resume_lr=resume_lr, resume_lr_target=resume_lr_target
            )

            # As in Applio: checkpoints are written at the end of an epoch, so
            # the run continues with the next one.  The step comes from the
            # filename rather than from ``epoch * len(train_loader)``: in a
            # fine-tune it also counts the pretrain's steps.
            epoch_str += 1
            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
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
            assert_decoder_layout_matches(
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
        apply_training_freezes(net_g, rank, freeze_vae=freeze_vae, freeze_mode=freeze_mode)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, dec_lr_scale=dec_lr_scale, vae_lr_scale=vae_lr_scale)

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


def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    # The reference is embedder-specific, and nothing about the filename says
    # which embedder wrote it.  Handing 768-wide features to a 256-wide text
    # encoder used to reach ``F.linear`` and die there -- "mat1 and mat2 shapes
    # cannot be multiplied" -- a few thousand steps into a run, at the first
    # preview rather than at startup.  Checked here instead, and the run
    # continues on a reference taken from the dataset, which is right by
    # construction.
    if use_custom_ref:
        expected_dim = int(getattr(config.model, "text_enc_hidden_dim", 768))
        features = np.load(os.path.join(reference_path, "ref_feats.npy"))
        if features.ndim != 2 or features.shape[1] != expected_dim:
            found = "x".join(str(size) for size in features.shape)
            warning(
                f"logs/reference/ref_feats.npy is {found} but this model's text "
                f"encoder takes {expected_dim}-wide features; it was made with a "
                f"different embedder. Falling back to a reference from the "
                f"dataset. To use your own, rebuild it with the embedder this "
                f"model trains on: python tools/make_reference.py <audio> "
                f"--embedder <name>",
                tag="[REFERENCE]",
            )
            use_custom_ref = False

    if use_custom_ref:
        info("Using custom reference input from 'logs/reference/'.", tag="[REFERENCE]")
        reference_audio = None
        reference_source = reference_path

        phone = torch.FloatTensor(np.repeat(features, 2, axis=0)).unsqueeze(0).to(device)
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
    # Undoes part of the mel scale's bin density, which is what decides how the
    # L1's gradient is split across frequency: 0-2 kHz holds 45% of the bins
    # and 12.5% of the spectrum, and that ratio is the same at every scale in
    # the multi-scale set.  0.0 is off, and off is what a config predating this
    # key gets -- see ``mel_frequency_tilt_weights``.
    mel_frequency_tilt = float(getattr(config.train, "mel_frequency_tilt", 0.0))
    mel_frequency_tilt_max_ratio = float(
        getattr(config.train, "mel_frequency_tilt_max_ratio", 8.0)
    )

    def _make_mel_distance():
        # ``mel_distance`` was only ever wired up for the ChouwaGAN stack;
        # every other vocoder always took the plain-L1 branch regardless of the
        # configured value.  Preserved rather than generalised, since making
        # Huber/MSE actually apply here would be a behaviour change to what
        # RefineGAN/HiFi-GAN train on.
        #
        # The band weighting is the one part that is reachable now, because it
        # is the only lever that reaches the multi-scale mel's frequency
        # allocation at all: the scale set cannot, the clamp floor is not what
        # is binding, and both weightings are normalised to a mean of 1 so
        # turning one on does not move the loss scale the adversarial balance
        # reads.  Both default to their neutral value, so a config that does
        # not mention them trains exactly as before.
        base = torch.nn.L1Loss
        weighted = mel_low_emphasis != 1.0 or mel_frequency_tilt != 0.0
        if not weighted:
            return base()

        def _weights(num_mels: int):
            weights = mel_frequency_tilt_weights(
                num_mels=num_mels,
                sample_rate=config.data.sample_rate,
                mel_fmin=config.data.mel_fmin,
                mel_fmax=config.data.mel_fmax,
                tilt=mel_frequency_tilt,
                max_ratio=mel_frequency_tilt_max_ratio,
            )
            if mel_low_emphasis != 1.0:
                # Multiplied, then renormalised: the two answer different
                # questions -- one says the bottom matters more than the mel
                # bin count says, the other says the top does -- and composing
                # them keeps either usable on its own.
                weights = weights * mel_low_frequency_weights(
                    num_mels=num_mels,
                    sample_rate=config.data.sample_rate,
                    mel_fmin=config.data.mel_fmin,
                    mel_fmax=config.data.mel_fmax,
                    emphasis=mel_low_emphasis,
                    cutoff_hz=mel_low_emphasis_hz,
                )
                weights = weights / weights.mean()
            return weights

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


    def _make_ms_mel_loss():
        return build_ms_mel_loss(sample_rate, loss_fn=_make_mel_distance())

    # Synthetic negative for the discriminator: real audio with its
    # high-frequency noise floor pulled down, scored as fake.  0.0 is off, and
    # off is what a config predating this key gets.  See
    # ``HighFrequencyFloorNegative`` for what it is for and what was measured
    # without it; the weight is below 1.0 because this is a regulariser on the
    # discriminator's opinion and not its task, and it is a starting point
    # rather than a measured optimum.
    hf_floor_weight = max(
        0.0, float(getattr(config.train, "hf_floor_negative_weight", 0.0))
    )
    fn_hf_floor_negative = None
    if hf_floor_weight > 0.0:
        fn_hf_floor_negative = HighFrequencyFloorNegative(
            sample_rate=sample_rate,
            gamma_range=tuple(
                getattr(config.train, "hf_floor_negative_gamma", (1.3, 2.0))
            ),
            cutoff_range=tuple(
                getattr(config.train, "hf_floor_negative_cutoff", (6000.0, 11000.0))
            ),
        ).to(device)

    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = _make_mel_distance()
        if swap_l1_to_ms:
            fn_spectral_loss_ms = _make_ms_mel_loss()
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = _make_ms_mel_loss()
    elif spectral_loss == "Hybrid L1":
        # Removed 2026-09-09.  It was a single-scale mel plus an MS-STFT term,
        # and it read as "the multi-scale mel, with high-frequency resolution
        # added" when it was really "the multi-scale mel replaced by an 80-bin
        # one".  At 32 kHz that mel is 561 Hz wide at 6 kHz -- 1.9 harmonics at
        # f0 297, so blind to harmonic contrast exactly where the MS-STFT term
        # was supposed to help -- and the two carried separate weights
        # (``c_mel`` at 45 against ``ms_stft_weight`` at 1.0) that had to be
        # balanced by hand.  "Multi-Scale Mel Loss" reaches 640 bins at window
        # 4096 and needs none of that: measured by overfitting a single clip,
        # it drives harmonic contrast to within 0.2 dB of the target in every
        # band up to 13 kHz.
        print_error(
            "'Hybrid L1' was removed on 2026-09-09: its single-scale 80-bin "
            "mel is coarser at the top of the band than the multi-scale mel "
            "it replaced, and the MS-STFT term beside it was reintroducing "
            "resolution that 'Multi-Scale Mel Loss' already has. Use "
            "'Multi-Scale Mel Loss'. Exiting.",
            tag="[INIT]",
        )
        sys.exit(1)
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

    # Before the compile: the fence is a plain attribute read inside
    # ``forward``, so Dynamo guards on it and a policy set afterwards would
    # only take effect on a recompile.

    enable_vocoder_compile(
        net_g, device, rank, enabled=compile_vocoder, mode=torch_compile_mode
    )
    enable_frontend_compile(net_g, config, device, rank)
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
    # Where this phase began: the pretrain's step for a fine-tune, 0 from
    # scratch.  Derived rather than taken from ``global_step`` so a resume
    # carries on counting ``phase_step`` (and ``max_steps``) instead of
    # restarting them.
    phase_start_step = checkpoint_step_from_path(pretrainG) if finetune_phase else 0
    phase_step = max(0, global_step - phase_start_step)
    phase_limit_reached = False
    if finetune_phase and not resumed_run:
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
        # Steps into this phase: a fine-tune's schedule starts at the pretrain,
        # not at step 0 of whatever run produced it.
        global_step - phase_start_step,
        train_loader,
        fresh_start=reset_optimizer_for_run,
        optimizer_choice_g=optimizer_choice_g,
        optimizer_choice_d=optimizer_choice_d,
        lr_final_ratio=lr_final_ratio,
        exp_decay_step_raw=exp_decay_step_raw,
    )

    # Reference sample for live-infer
    reference, reference_audio, reference_source = get_reference_sample(
        train_loader,
        device,
        config,
    )

    # Cache for training with " cache " enabled
    cache = []

    # Logged here rather than in ``training_loop``: the weights are constants of
    # the assembled discriminator, and the loop runs once per epoch, so saying
    # it there reprinted an [INIT] line after every epoch's save.
    if rank == 0:
        model_d = net_d.module if hasattr(net_d, "module") else net_d
        if getattr(model_d, "uses_branch_weights", False):
            weighted = ", ".join(
                f"{label} {weight:g}"
                for label, weight in zip(model_d.branch_labels, model_d.branch_weights)
                if weight != 1.0
            )
            info(f"Discriminator branch weights: {weighted}.", tag="[INIT]")

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
        overtrain_monitor = OvertrainMonitor(
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
            fn_spectral_loss_ms,
            holdout_set=holdout_set,
            fn_hf_floor_negative=fn_hf_floor_negative,
            hf_floor_weight=hf_floor_weight,
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
    fn_spectral_loss_ms=None,
    fn_hf_floor_negative=None,
    hf_floor_weight=0.0,
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

    # Checkpoints are only written at epoch boundaries, so every epoch -- a
    # resumed one included -- starts at its first batch.
    data_iterator = enumerate(train_loader)

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
    # One separation per head per step.  Rank 0 only: it is a diagnostic, and
    # the other ranks never write to the summary.
    branch_disc_cache = deque(maxlen=rolling_loss_steps)
    branch_neg_cache = deque(maxlen=rolling_loss_steps)
    # Per-head share of ``loss_adv``, which ``disc_sep`` cannot stand in for:
    # the generator's term saturates in the logit gap, so a head separating an
    # order of magnitude better than the rest is not paying an order of
    # magnitude more of the objective.  This is the series a branch weight is
    # actually tuned against.
    branch_adv_cache = deque(maxlen=rolling_loss_steps)
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
    # Per-branch loss weights, read once: they are constants of the assembled
    # discriminator, not of the step.  ``None`` on a discriminator that weights
    # nothing, so an unweighted run takes exactly the path it took before
    # weighting existed.
    model_d = net_d.module if hasattr(net_d, "module") else net_d
    branch_weights = (
        tuple(model_d.branch_weights)
        if getattr(model_d, "uses_branch_weights", False)
        else None
    )
    kl_active_threshold = max(
        0.0,
        float(getattr(config.train, "kl_active_threshold", 0.01)),
    )

    with progress_task(
        len(train_loader),
        f"Epoch {epoch}/{total_epoch_count}",
        initial=0,
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
                # The only pass that asks for the direction output: it is the
                # one whose gradient trains the discriminator.
                # ``combine_inputs``: the real and the fake side go through
                # every branch as one batch of ``2B``.  Same numbers, half the
                # kernel launches -- 149 ms -> 136 on a v4 at batch 8 over
                # 0.4 s, at unchanged peak VRAM.  Only this pass can ask for
                # it; the generator's runs the real side under ``no_grad``,
                # and half a batch cannot be.
                #
                # The synthetic negative rides in the same tensor rather than
                # in a second call: the real side is then computed once for
                # both, so the pass is ``3B`` instead of ``2B`` and not ``4B``.
                fake = y_hat.detach()
                if fn_hf_floor_negative is not None:
                    fake = torch.cat((fake, fn_hf_floor_negative(y)), dim=0)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(
                    y, fake, san_training=san_active, combine_inputs=True
                )
                y_d_hat_neg = None
                if fn_hf_floor_negative is not None:
                    y_d_hat_g, y_d_hat_neg = split_branch_outputs(
                        y_d_hat_g, (y_hat.shape[0], y.shape[0])
                    )

            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                disc_loss_parts = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                    san_direction_weight=san_direction_weight,
                    normalize=False,
                    per_branch=False,
                    branch_weights=branch_weights,
                )
                loss_disc, loss_disc_real, loss_disc_fake = disc_loss_parts[:3]
                if y_d_hat_neg is not None:
                    # Only the fake half of the second call is wanted: the real
                    # half is the same logits already charged for above, and
                    # counting them twice would double the weight on one side
                    # of a two-sided loss.
                    loss_hf_floor = discriminator_loss(
                        y_d_hat_r,
                        y_d_hat_neg,
                        san_direction_weight=san_direction_weight,
                        normalize=False,
                        per_branch=False,
                        branch_weights=branch_weights,
                    )[2]
                    loss_disc = loss_disc + hf_floor_weight * loss_hf_floor

            optim_d.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss_disc).backward()
                grad_scaler.unscale_(optim_d)
            else:
                loss_disc.backward()
            grad_norm_d = clip_or_sample_grad_norm(
                net_d.parameters(),
                grad_clip_value_d,
                global_step,
                metrics_update_interval,
            )
            if grad_scaler is not None:
                grad_scaler.step(optim_d)
            else:
                optim_d.step()
            if san_active:
                normalize_san_weights(net_d)

            # Per-head separation, which the aggregate loss cannot show: nine
            # heads are summed into ``loss_disc``, and a head that has stopped
            # separating and a head that has learned the *opposite* of its job
            # both leave that sum where it was.  The loss halves are no
            # substitute -- each is a "how wrong" measure, so a head that is
            # confidently wrong about fakes and a head that is right about them
            # move the same term in opposite directions for the same reason.
            # What answers the question is the raw logits: how much higher this
            # head scores real audio than the generator's output.
            if rank == 0:
                branch_disc_cache.append(branch_separation(y_d_hat_r, y_d_hat_g))
                if y_d_hat_neg is not None:
                    # The series this switch exists to move: without it the
                    # heads that can see the floor sat at -0.13 and -0.82.
                    branch_neg_cache.append(
                        branch_separation(y_d_hat_r, y_d_hat_neg)
                    )

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

            # Run discriminator on generated output.
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
            # ``Module.requires_grad_`` *is* the loop over ``parameters()``,
            # and on ``net_d`` and not the unwrapped module because DDP holds
            # the module as a submodule and adds no parameters of its own --
            # the same 165 tensors either way.  The unwrap below is for the
            # *call*, which is a different question.
            net_d.requires_grad_(False)
            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                # ``no_grad_real``: the real side is the feature matching
                # *target* and its logits are discarded, so differentiating
                # it builds a graph nothing consumes.  See
                # ``MPD_MSD_Combined.forward``.
                #
                # The unwrapped module, and here the unwrap is load
                # bearing: ``DistributedDataParallel._post_forward`` calls
                # ``reducer.prepare_for_backward`` whenever grad is
                # enabled, arming an allreduce that this backward can never
                # complete -- every parameter is frozen, so no gradient
                # hook fires.  The next iteration's discriminator forward
                # then raises "Expected to have finished reduction in the
                # prior iteration before starting a new one".  No gradient
                # sync is wanted here in the first place.
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
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
                prior_gap_delta, prior_gap_error = prior_gap(
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
                # cannot show which half is actually moving.
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
                        loss_ms_mel = fn_spectral_loss_ms(y, y_hat) * config.train.c_mel
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
                    loss_spectral = fn_spectral_loss(y, y_hat) * config.train.c_mel
                loss_fm = (
                    feature_loss(
                        fmap_r,
                        fmap_g,
                        normalize=False,
                        branch_weights=branch_weights,
                    )
                    * 2.0
                )

                # Generator loss.  ``y_d_hat_g`` comes from the *generator*
                # update's forward, which never sets ``san_training``, so these
                # are plain logits and ``san_direction_weight`` is inert -- the
                # direction output is the discriminator's business alone.
                loss_adv, branch_adv = generator_loss(
                    y_d_hat_g,
                    normalize=False,
                    san_direction_weight=san_direction_weight,
                    use_softplus=san_active,
                    branch_weights=branch_weights,
                    per_branch=True,
                )
                if rank == 0:
                    # Detached scalars the loss has already formed, so this is
                    # a stack of nine numbers and not a second pass.
                    branch_adv_cache.append(branch_adv)

                loss_kl, raw_kl = kl_loss(
                    z_p,
                    logs_q,
                    m_p,
                    logs_p,
                    z_mask,
                    return_terms=True,
                )
                # ``c_kl_scale`` is the launch's multiplier on the config's
                # weight, so a staged pretrain can run stage 1 at a low KL
                # without editing ``config.json`` -- which a later resume
                # would then inherit silently.
                loss_kl = loss_kl * config.train.c_kl * c_kl_scale

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
                    # Masked mean sigma, prior beside posterior because neither
                    # reads alone.  Both deques existed and were never filled,
                    # so the tag stage 1 tells you to watch was never written.
                    avg_rolling_cache["posterior_std_fast"].append(
                        (torch.exp(logs_q.float()) * z_mask).sum()
                        / (z_mask.sum().clamp(min=1) * logs_q.shape[1])
                    )
                    avg_rolling_cache["prior_std_fast"].append(
                        (torch.exp(logs_p.float()) * x_mask).sum()
                        / (x_mask.sum().clamp(min=1) * logs_p.shape[1])
                    )

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
                module_grad_metrics = generator_gradient_metrics(net_g)
            grad_norm_g = clip_or_sample_grad_norm(
                net_g.parameters(),
                grad_clip_value_g,
                global_step,
                metrics_update_interval,
            )
            if grad_scaler is not None:
                grad_scaler.step(optim_g)
            else:
                optim_g.step()
            # Unwind the freeze.  A plain restore and not a ``finally``: the
            # batch loop catches nothing, ``run`` is the process target, so an
            # exception anywhere above ends this process rather than reaching
            # another step that a frozen discriminator could spoil.
            #
            # Unconditionally ``True`` and not a captured state list:
            # ``apply_frontend_freeze`` is the only other ``requires_grad``
            # write in the codebase and it operates on ``net_g``, so every
            # discriminator parameter is trainable at every step.  A freeze
            # that ever reaches ``net_d`` -- a frozen-D stage, a partial
            # pretrained load -- makes both paragraphs false, and this is the
            # line to change.
            net_d.requires_grad_(True)


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
                        metrics = holdout_metrics_resilient(
                            averaged,
                            holdout_set,
                            config,
                            device,
                            want_latent=want_latent,
                            noise_scale=noise_scale,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                        )
                        if probe_set is not None:
                            probe_score = holdout_metrics_resilient(
                                averaged,
                                probe_set,
                                config,
                                device,
                                want_latent=False,
                                noise_scale=noise_scale,
                                use_amp=use_amp,
                                amp_dtype=amp_dtype,
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
                        metrics = holdout_metrics_resilient(
                            net_g,
                            holdout_set,
                            config,
                            device,
                            want_latent=want_latent,
                            noise_scale=noise_scale,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                        )
                        if probe_set is not None:
                            probe_score = holdout_metrics_resilient(
                                net_g,
                                probe_set,
                                config,
                                device,
                                want_latent=False,
                                noise_scale=noise_scale,
                                use_amp=use_amp,
                                amp_dtype=amp_dtype,
                            )["mel_l1"]
                        improved = overtrain_monitor.update(
                            net_g.module if hasattr(net_g, "module") else net_g,
                            metrics["mel_l1"],
                            global_step,
                        )

                holdout_loss = metrics["mel_l1"]
                if rank == 0 and math.isfinite(holdout_loss):
                    writer.add_scalar("holdout/mel_l1", holdout_loss, global_step)
                    # Read these against ``mel_l1``, not instead of it: the
                    # deficits say where the reconstruction is wrong and the L1
                    # says how much it costs.  A spectral reweighting that is
                    # buying the top bands with the bottom ones shows up as the
                    # top deficits closing while ``mel_l1`` fails to move.
                    for key, value in metrics.items():
                        if key.startswith("band_deficit_") and math.isfinite(value):
                            writer.add_scalar(
                                f"holdout/{key}", value, global_step
                            )
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
                            f"they will be exported as an extra '_pre-overtrain' "
                            f"model alongside the regular saves.",
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
            # Same guard as the two norms above, and for the same reason: on
            # an AMP overflow step the unscaled grads are ``inf``, the scaler
            # throws the step away, but the per-module metric is still measured
            # from them.  These are sampled once every
            # ``metrics_update_interval`` steps into a ``rolling_loss_steps``
            # window, so a single poisoned sample turns the logged average NaN
            # for ``metrics_update_interval * rolling_loss_steps`` steps --
            # 400, against the 50 the G/D series would lose.
            for key, value in module_grad_metrics.items():
                value = value.detach()
                if not torch.isfinite(value):
                    continue
                avg_rolling_cache.setdefault(
                    key,
                    deque(maxlen=rolling_loss_steps),
                ).append(value)

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

                # Per-head separation: the mean logit on real audio minus the
                # mean on the generator's output.  Positive is a head doing its
                # job, ~0 is one that has stopped separating, and *negative* is
                # a head that scores the generator above real audio -- which
                # does not merely stop helping, it pays the generator to keep
                # whatever that head is reading.  The run this was added for
                # spent 87k steps with every head between -0.05 and +0.02 on
                # exactly the spectral defect it never fixed, the three
                # spectrogram heads significantly on the wrong side of zero,
                # while ``loss_disc`` sat at 20.0 throughout.
                labels = getattr(
                    net_d.module if hasattr(net_d, "module") else net_d,
                    "branch_labels",
                    (),
                )
                for prefix, cache in (
                    (f"disc_sep_{rolling_loss_steps}", branch_disc_cache),
                    (f"disc_sep_floor_{rolling_loss_steps}", branch_neg_cache),
                    # Post-weight, so the series sums to ``loss_adv`` and a
                    # head's share can be read off it directly.
                    (f"adv_sep_{rolling_loss_steps}", branch_adv_cache),
                ):
                    if not cache:
                        continue
                    branch_mean = torch.stack(list(cache)).mean(0)
                    for index, label in enumerate(labels[: branch_mean.shape[0]]):
                        scalar_dict_rolling[f"{prefix}/{label}"] = branch_mean[
                            index
                        ].item()

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)

                # KL diagnostics (diag tab)
                if len(kl_std_cache) > 0:
                    # The caches hold device tensors, so this is where the
                    # window's three numbers cross to the host -- once per
                    # ``rolling_loss_steps``, on a line that is already
                    # synchronising to write TensorBoard.
                    diag_scalars = {
                        "diag/kl_std": cache_mean(kl_std_cache),
                        "diag/kl_mean_per_dim": cache_mean(kl_mean_cache),
                        "diag/kl_active_fraction": cache_mean(kl_active_cache),
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
                # The EMA when there is one: it is what the holdout scores and
                # what the export ships.  The live weights of a GAN generator
                # swing from step to step and read the latent's noise far more
                # strongly -- measured 2026-09-11 on ``pretrain-contentvec`` at
                # 218k, the seed-driven bursts between the harmonics came out
                # 1.90 dB RMS on the live weights against 1.17 on the EMA at
                # the same draw, so the preview showed a defect the delivered
                # model mostly does not have.  Schedule-free runs carry no EMA
                # and keep reading their averaged ``x`` iterate.
                if ema is not None:
                    with ema.applied(net_g):
                        o = eval_preview(net_g, reference, reference_audio, config)
                else:
                    with averaged_weights((optimizer_choice_g, optim_g)):
                        o = eval_preview(net_g, reference, reference_audio, config)
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
                        # (320 at 32 kHz), which labelled the axis 1.7x short.
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
            emit_machine_progress(
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
            preview_sd, preview_label, _preview_step = deliverable_weights(
                overtrain_monitor, ema, model_g, use_holdout=False
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
                    o = eval_preview(net_g, reference, reference_audio, config)
            else:
                o = eval_preview(net_g, reference, reference_audio, config)
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
                    # The generator is written as if there were no EMA: the
                    # average goes in ``model`` and no ``ema`` key is kept, so
                    # the checkpoint is a plain one for anything that reads it
                    # and is a pretrain as it stands, like
                    # ``tools/clean_pretrain.py`` makes.  A resume therefore
                    # continues from the average, and the shadow restarts from
                    # it -- see the ``ema.load_state_dict`` fallback.
                    with ema.applied(net_g) if ema is not None else nullcontext():
                        save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path)
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
                model_add.append((os.path.join(experiment_dir, weight_model_name), False))

        # Check completion
        if epoch >= total_epoch_count:
            success(
                f"Training completed: {epoch} epochs, {global_step} steps, "
                f"generator loss {loss_gen_total.item():.3f}.",
                tag="[TRAIN]",
            )
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append((os.path.join(experiment_dir, weight_model_name), False))
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
        # This is the only export that carries the holdout best; the periodic
        # ones keep the current weights, so this file is an extra rather than
        # a replacement for them.
        if overtrain_flagged and not overtrain_exported and overtrain_monitor is not None:
            overtrain_exported = True
            model_add.append(
                (
                    os.path.join(
                        experiment_dir,
                        small_model_naming(
                            f"{model_name}_pre-overtrain", epoch, overtrain_monitor.best_step
                        ),
                    ),
                    True,
                )
            )

        if model_add:
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            for m, use_holdout in model_add:
                if os.path.exists(m):
                    continue
                # ``deliverable_weights`` can fall through to the live weights,
                # and under a schedule-free optimizer those are the extrapolated
                # iterate.  This is the exported .pth -- the file that gets used
                # for inference -- so it is the last place that read may go
                # unaveraged.
                with averaged_weights((optimizer_choice_g, optim_g)):
                    ckpt, ckpt_label, ckpt_step = deliverable_weights(
                        overtrain_monitor, ema, model_g, use_holdout=use_holdout
                    )
                success(
                    f"{os.path.basename(m)} <- {ckpt_label}", tag="[EXPORT]"
                )
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
                        weights_step=ckpt_step,
                        weights_source=ckpt_label,
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
