import os
import signal
import datetime
import glob
import math
import re
import sys

from itertools import islice
from collections import deque
from distutils.util import strtobool
from random import randint
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

from torch.backends import cuda, cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from rvc.lib.terminal import (
    configure_logging,
    install_rich_print,
    print_model_summary,
    progress_task,
)
from rvc.train.messages import (
    VOCODER_COMPILE_ENABLED,
    VOCODER_COMPILE_NO_CUDA,
    VOCODER_COMPILE_NOT_SUPPORTED,
)

install_rich_print()

from utils import (
    plot_spectrogram_to_numpy,
    plot_mel_comparison_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    load_config_from_json,
    flush_writer,
    block_tensorboard_flush_on_exit,
    trim_audio_preview_events,
    write_audio_preview,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    print_init_setup,
    train_loader_safety,
    verify_spk_dim,
    early_stopper
)

from losses import (
    discriminator_loss,
    generator_loss,
    feature_loss,
    kl_loss,
    MultiScaleSTFTLoss,
)

from mel_processing import spec_to_mel_torch, MultiScaleMelSpectrogramLoss

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.configs.vocoders import normalize_vocoder

# ======== Parse command line arguments start region ===========================

model_name = sys.argv[1]
epoch_save_frequency = int(sys.argv[2])
total_epoch_count = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest_net_models = strtobool(sys.argv[9])
save_weight_models = strtobool(sys.argv[10])
use_warmup = strtobool(sys.argv[11])
warmup_duration = int(sys.argv[12])
cleanup = strtobool(sys.argv[13])
vocoder = normalize_vocoder(sys.argv[14])
architecture = sys.argv[15]
architecture = "RVC"
optimizer_choice_g = sys.argv[16]
optimizer_choice_d = sys.argv[17]
use_checkpointing = strtobool(sys.argv[18])
use_tf32 = bool(strtobool(sys.argv[19]))
use_benchmark = bool(strtobool(sys.argv[20]))
use_deterministic = bool(strtobool(sys.argv[21]))
spectral_loss = sys.argv[22]
lr_scheduler = sys.argv[23]
exp_decay_gamma = float(sys.argv[24])
use_kl_annealing = strtobool(sys.argv[25])
kl_annealing_cycle_duration = int(sys.argv[26])
rolling_loss_steps = int(sys.argv[27])

grad_clip_scheduling = bool(strtobool(sys.argv[28]))
grad_clip_steps_duration = int(sys.argv[29])
grad_clip_value_g_cap, grad_clip_value_d_cap = (int(sys.argv[30]), int(sys.argv[31]))
grad_clip_value_g_release, grad_clip_value_d_release = (int(sys.argv[32]), int(sys.argv[33]))

use_custom_lr = strtobool(sys.argv[34])
custom_lr_g, custom_lr_d = (float(sys.argv[35]), float(sys.argv[36])) if use_custom_lr else (None, None)
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

use_2_sample_kl = bool(strtobool(sys.argv[37]))
use_best_step = bool(strtobool(sys.argv[38]))
double_d_updates = bool(strtobool(sys.argv[39]))
compile_vocoder = (
    bool(strtobool(sys.argv[40])) if len(sys.argv) > 40 else False
)
torch_compile_mode = sys.argv[41] if len(sys.argv) > 41 else "default"

# Torch backend config -----
cuda.matmul.allow_tf32 = use_tf32
cudnn.allow_tf32 = use_tf32
cudnn.benchmark = use_benchmark
cudnn.deterministic = use_deterministic

# Parse command line arguments end region ===========================

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

# Load the config from json
config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

chouwagan_active = vocoder == "chouwagan"
if chouwagan_active and sample_rate != 44100:
    raise ValueError("ChouwaGAN requires the 44.1 kHz configuration.")

# AMP precision / dtype init
train_dtype = torch.float16 if config.train.fp16_run else torch.float32

# Globals ( Do not alter these )
global_step = 0
warmup_completed = False
from_scratch = False
use_lr_scheduler = lr_scheduler != "none"



# ========  Advanced / Manual and exp tweaks  ========================
enable_persistent_workers = True

pretrain_preview = True
pretrain_preview_interval = 1000  # Measured in steps.
finetune_preview_interval = 100  # Measured in steps.

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

# Freezes the entire VAE frontend ( enc_p / enc_q / flow )
freeze_vae = False # If true, lets only vocoder ( dec ), spk embedding and discriminator learn

# ----  Decoder / vocoder per-layer tuning ( freezing + differential LR )  ----
decoder_plan = {
    "vocoder": None,  # "VOCODER" or None ( applies to any vocoder )
    "conv_pre":        {"freeze": False, "lr": 1.0},
    "cond":            {"freeze": False, "lr": 1.0},
    "ups":             {"freeze": False, "lr": 1.0},
    "exc_proj":        {"freeze": False, "lr": 1.0},
    "fusion_proj":     {"freeze": False, "lr": 1.0},
    "adain_resblocks": {"freeze": False, "lr": 1.0},
    "conv_post":       {"freeze": False, "lr": 1.0},
}

# ----  Global LR scales  ----
# Multipliers of the base LR ( 0.1 = 10%, 1.0 = 100% ).
dec_lr_scale = None  # the entire decoder/vocoder 
vae_lr_scale = None  # the whole VAE frontend ( enc_p / enc_q / flow / emb_g )

# ----  Resume LR override  ----
resume_lr = None  # e.g. 5e-5 ( None = Override disabled. )
resume_lr_target = "full"  # Pick what you want it applied to: "g", "d" or "full" where full refers to both G/D

# True = Gamma is applied as-is each step.
# False = Per-epoch budget, re-binned to per-step so the total decay per epoch equals "exp decay epoch" ( VITS-style ) ~ Default
exp_decay_step_raw = False


# ----  EXPERIMENTAL  ----
use_sid_swap = False  # Allows to finetune using a different base-speaker
custom_sid = 1

# Vocoder-only training: bypasses the VAE frontend entirely ( no cvec feats ), uses mel specs.
train_voc_only = False

##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


class EarlyStopSignalHandler:
    def __init__(self):
        self.stop_triggered = False
        signal.signal(signal.SIGINT, self._handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._handler)

    def _handler(self, signum, frame):
        self.stop_triggered = True
        print(f"\n[TRAINING] Early Stopping signal received! Finishing current step and saving...")


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
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

def endless_loader(loader):
    while True:
        for batch in loader:
            yield batch

def prepare_dataloaders(config, n_gpus, rank, batch_size, build_extra_d_loader=False):
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data, voc_only=train_voc_only, n_mel_bins=config.model.inter_channels)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    collate_fn = TextAudioCollateMultiNSFsid(voc_only=train_voc_only)
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

    extra_d_loader = None
    if build_extra_d_loader:
        extra_d_sampler = DistributedBucketSampler(
            train_dataset,
            batch_size * n_gpus,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True
        )
        extra_d_loader = DataLoader(
            train_dataset,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=extra_d_sampler,
            persistent_workers=enable_persistent_workers,
            prefetch_factor=2
        )

    return train_loader, extra_d_loader

def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
        use_2_sample_kl = use_2_sample_kl,
        train_voc_only = train_voc_only,
    )

def get_d_model(config, vocoder, use_checkpointing):
    vocoder = normalize_vocoder(vocoder)
    if vocoder == "chouwagan":
        from rvc.lib.algorithm.discriminators.multi import ChouwaGANDiscriminator

        return ChouwaGANDiscriminator(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            sample_rate=config.data.sample_rate,
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


def _add_decoder_only_gradients(loss, net_g, scale=None):
    decoder_parameters = _decoder_parameters(net_g)
    if not decoder_parameters:
        return

    scaled_loss = loss if scale is None else loss * scale
    decoder_gradients = torch.autograd.grad(
        scaled_loss,
        decoder_parameters,
        retain_graph=False,
        allow_unused=True,
    )
    for parameter, gradient in zip(decoder_parameters, decoder_gradients):
        if gradient is None:
            continue
        parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient


def _make_optimizer(
    model,
    choice,
    lr,
    num_epochs=None,
    num_batches=None,
    param_groups=None,
    lazy_reg_interval=None,
):
    params = param_groups if param_groups is not None else filter(lambda p: p.requires_grad, model.parameters())

    lazy_scale = 1.0
    if lazy_reg_interval is not None:
        interval = max(1, int(lazy_reg_interval))
        lazy_scale = interval / (interval + 1.0)
    lr = lr * lazy_scale

    def lazy_betas(betas):
        return tuple(float(beta) ** lazy_scale for beta in betas)

    if choice == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=lr, betas=lazy_betas((0.8, 0.99)), eps=1e-9, weight_decay=0.01, fused=True)

    elif choice == "RAdam":
        optimizer = torch.optim.RAdam(params, lr=lr, betas=lazy_betas((0.8, 0.99)), eps=1e-9, weight_decay=0.01, decoupled_weight_decay=True)

    elif choice == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        ranger_kw = dict(
            num_epochs=num_epochs, num_batches_per_epoch=num_batches,
            use_madgrad=False, use_warmup=False, warmdown_active=False,
            use_cheb=False, lookahead_active=True, normloss_active=False,
            normloss_factor=1e-4, softplus=False,
            use_adaptive_gradient_clipping=True, agc_clipping_value=0.01,
            agc_eps=1e-3, using_gc=True, gc_conv_only=True, using_normgc=False,
        )
        optimizer = Ranger21(params, lr=lr, betas=lazy_betas((0.8, 0.99)), eps=1e-9, weight_decay=0.0, **ranger_kw)

    elif choice == "AdaBelief":
        from rvc.train.custom_optimizers.adabelief import AdaBelief
        optimizer = AdaBelief(params, lr=lr, betas=lazy_betas((0.8, 0.999)), eps=1e-16, weight_decay=0, rectify=False)

    elif choice == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(params, lr=lr, betas=lazy_betas((0.8, 0.99)), eps=1e-9, weight_decay=0.01, warmup_steps=0)

    elif choice == "Sched-Free RAdam":
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(params, lr=lr, betas=lazy_betas((0.8, 0.99)), eps=1e-9, weight_decay=0.0, r=0.0, weight_lr_power=2.0, foreach=False, silent_sgd_phase=False)
    else:
        raise ValueError(f"Unknown optimizer choice: {choice}")

    for group in optimizer.param_groups:
        group["lazy_reg_scale"] = lazy_scale
    return optimizer


def active_decoder_plan():
    """
    Return the decoder_plan entries if the plan targets the current training vocoder,
    else an empty dict.
    """
    target = decoder_plan.get("vocoder")
    if target is not None and target != vocoder:
        return {}
    return {k: v for k, v in decoder_plan.items() if k != "vocoder"}


def build_decoder_param_groups(net_g, base_lr):
    """
    Build optimizer param groups with differential LR for decoder components.

    Per-part LR scales come from `decoder_plan`
    any decoder layer NOT listed there falls back to `dec_lr_scale`.
    Returns None if no LR scales are configured.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return None

    plan = active_decoder_plan()

    any_scale = (
        dec_lr_scale is not None
        or vae_lr_scale is not None
        or any(spec.get("lr") is not None and spec["lr"] != 1.0 for spec in plan.values())
    )
    if not any_scale:
        return None

    def scale_for(name):
        """Effective LR scale for a decoder parameter name."""
        for part, spec in plan.items():
            if name.startswith(f"{part}.") and spec.get("lr") is not None:
                return spec["lr"]
        return dec_lr_scale if dec_lr_scale is not None else 1.0

    # scale -> [params]  ( same scale == same effective LR )
    groups = {}

    def add(param, scale):
        groups.setdefault(scale, []).append(param)

    # Decoder params, each bucketed by its effective scale.
    for name, param in dec.named_parameters():
        if param.requires_grad:
            add(param, scale_for(name))

    # Everything outside `dec.` (enc_p / enc_q / flow / emb_g).
    rest_scale = vae_lr_scale if vae_lr_scale is not None else 1.0
    for name, param in model.named_parameters():
        if not name.startswith("dec.") and param.requires_grad:
            add(param, rest_scale)

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


def apply_decoder_freezes(net_g, rank):
    """
    Apply decoder / vocoder layer freezing for fine-tuning.

    Drives everything off `decoder_plan` and the standalone `freeze_vae` flag.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return

    plan = active_decoder_plan()

    frozen_parts = []
    frozen_params = 0

    for attr_name, spec in plan.items():
        if spec.get("freeze") and hasattr(dec, attr_name):
            module = getattr(dec, attr_name)
            for param in module.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_parts.append(attr_name)

    # Freeze the VAE frontend ( enc_p / enc_q / flow ), keep spk embedding trainable.
    if freeze_vae:
        for name, param in model.named_parameters():
            if not name.startswith("dec.") and not name.startswith("emb_g.") and param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        frozen_parts.append("VAE (enc_p/enc_q/flow, emb_g kept trainable)")

    if rank == 0:
        if frozen_parts:
            print(f"[INIT] Decoder frozen: {', '.join(frozen_parts)} ({frozen_params:,} params)")
        else:
            print("[INIT] Decoder: no layers frozen")


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

    model = net_g.module if hasattr(net_g, "module") else net_g
    enabled = model.enable_decoder_compile(mode=torch_compile_mode)
    if not enabled and rank == 0:
        print(VOCODER_COMPILE_NOT_SUPPORTED)
    if enabled and rank == 0:
        print(VOCODER_COMPILE_ENABLED.format(mode=torch_compile_mode))
    return enabled


def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, sample_rate, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing)
    net_d = get_d_model(config, vocoder, use_checkpointing)
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
        if g_checkpoint_path and d_checkpoint_path:

            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus)

            # Apply decoder / vocoder layer freezes ( for fine-tuning )
            apply_decoder_freezes(net_g, rank)

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

            # Load the model and optim states
            _, _, _, epoch_str, gradscaler_dict_g = load_checkpoint(g_checkpoint_path, net_g, optim_g, strict_load)
            _, _, _, epoch_str, gradscaler_dict_d = load_checkpoint(d_checkpoint_path, net_d, optim_d, strict_load)

            # resume_lr re-anchors G and/or D to the given base LR.
            apply_resume_lr_override(optim_g, optim_d)

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
        global_step = 0
        gradscaler_dict_g = {}
        gradscaler_dict_d = {}

        # Loading the pretrained Generator model
        if pretrainG not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (G) '{pretrainG}'")
            checkpoint = torch.load(pretrainG, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_g.load_state_dict(state_dict, strict=True)

            if use_sid_swap and custom_sid != 0:
                total_sids = net_g.emb_g.weight.size(0)

                if custom_sid >= total_sids:
                    print(f"[SID SWAP] {custom_sid} is out of bounds!")
                    print(f"[SID SWAP] Currently chosen pretrains only support SIDs from 0 to {total_sids - 1}.")
                    sys.exit("Invalid SID Selection. Please choose a lower custom_sid.")
                if rank == 0:
                    print(f"[SID SWAP] Swapping SID: 0 with SID: {custom_sid}")

                with torch.no_grad():
                    temp_sid_0 = net_g.emb_g.weight[0].clone()
                    net_g.emb_g.weight[0].copy_(net_g.emb_g.weight[custom_sid])
                    net_g.emb_g.weight[custom_sid].copy_(temp_sid_0)
                if rank == 0:
                    print("[SID SWAP] Swap successful. Model is ready for fine-tuning.")

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
        apply_decoder_freezes(net_g, rank)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice_g, optimizer_choice_d, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict_g, gradscaler_dict_d


def prepare_schedulers(
    optim_g, optim_d,
    use_lr_scheduler, lr_scheduler, exp_decay_gamma,
    total_epoch_count, epoch_str, global_step, train_loader
):
    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    scheduler_resume_epoch = epoch_str - 1
    scheduler_resume_step = global_step - 1

    for param_group in optim_g.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    for param_group in optim_d.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler:
        scheduler_name = (
            "cosine annealing epoch"
            if lr_scheduler == "cosine annealing"
            else lr_scheduler
        )

        if scheduler_name == "exp decay epoch":
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

    if train_voc_only:
        print("[REFERENCE] Vocoder-only mode: fetching mel + f0 reference from train_loader.")
        info = next(iter(train_loader))
        _, _, pitch, pitchf, spec, _, reference_audio, _, sid = info

        pitch = pitch[0:1].to(device)
        pitchf = pitchf[0:1].to(device)
        spec = spec[0:1].to(device)
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

        return (
            (None, None, pitch, pitchf, sid, config.train.seed, spec),
            reference_audio,
        )

    if use_custom_ref:
        print("[REFERENCE] Using custom reference input from 'logs\\reference\\'")
        reference_audio = None

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

    return (
        (phone, phone_lengths, pitch, pitchf, sid, config.train.seed),
        reference_audio,
    )





def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    wavs = [wav for wav in glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*")) if wav.endswith((".wav", ".flac"))]
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz).")
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("No GPU detected, fallback to CPU. This will take a very long time ...")

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
    global global_step, warmup_completed, optimizer_choice_g, optimizer_choice_d, from_scratch, swap_start_step, swap_completed

    if rank == 0:
        configure_logging()

    stopper = EarlyStopSignalHandler()

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
        use_kl_annealing,
        kl_annealing_cycle_duration,
        spectral_loss,
    )

    # Initial setup
    setup_env_and_distr(rank, n_gpus, device, device_id, config)

    # Dataloading and loaders preparation
    train_loader, extra_d_loader = prepare_dataloaders(config, n_gpus, rank, batch_size, build_extra_d_loader=double_d_updates)

    # Spk dim verif
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = spk_dim

    if rank == 0 and warmup_active():
        warmup_tag = "Manual control" if warmup_steps > 0 else f"per epoch x{warmup_duration}"
        print(f"[INIT] linear warmup: {effective_warmup_steps(train_loader)} steps  -  ({warmup_tag})")

    # Spectral loss init
    fn_spectral_loss2 = None
    fn_spectral_loss_ms = None

    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
        if swap_l1_to_ms:
            fn_spectral_loss_ms = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
    elif spectral_loss == "Hybrid L1":
        fn_spectral_loss = torch.nn.L1Loss()
        fn_spectral_loss2 = MultiScaleSTFTLoss()
    else:
        print("ERROR: Chosen spectral loss is undefined. Exiting.")
        sys.exit(1)


    # Loading of models and optims
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict_g, gradscaler_dict_d = load_models_and_optimizers(
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
        audio_preview_log_dir = os.path.join(experiment_dir, "eval", "audio_previews")
        trim_audio_preview_events(audio_preview_log_dir, keep=10)
        block_tensorboard_flush_on_exit(writer_eval)

        if global_step != 0:
            print(f"[INIT] TensorBoard writer initialized. Purging logs after step: {global_step}")
        else:
            print(f"[INIT] TensorBoard writer initialized.")

    # from-scratch checker ( disables average loss )
    if (pretrainG in ["", "None"] or pretrainD in ["", "None"]) or force_from_scratch:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")

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
        train_loader
    )

    # GradScaler for FP16 training
    gradscaler_g = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    gradscaler_d = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))

    if len(gradscaler_dict_g) > 0 and len(gradscaler_dict_d) > 0:
        gradscaler_g.load_state_dict(gradscaler_dict_g)
        gradscaler_d.load_state_dict(gradscaler_dict_d)
        print("[INIT] Loading G/D gradscaler state dicts")
    else:
        print("[INIT] G/D gradscaler state dicts not found - Fresh initialization")

    # Reference sample for live-infer
    reference, reference_audio = get_reference_sample(train_loader, device, config)

    # Cache for training with " cache " enabled
    cache = []

    for epoch in range(epoch_str, total_epoch_count + 1):
        if extra_d_loader is not None:
            extra_d_loader.batch_sampler.set_epoch(epoch)
        should_stop = training_loop(
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
            fn_spectral_loss,
            n_gpus,
            gradscaler_g,
            gradscaler_d,
            fn_spectral_loss2,
            fn_spectral_loss_ms,
            stopper=stopper,
            extra_d_loader=extra_d_loader,
        )
        if should_stop:
            break

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
            print(f"[TRAIN] Warmup completed at step: {global_step}")
            print(f"[TRAIN] LR G: {optim_g.param_groups[0]['lr']}")
            print(f"[TRAIN] LR D: {optim_d.param_groups[0]['lr']}")
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
    fn_spectral_loss,
    n_gpus,
    gradscaler_g,
    gradscaler_d,
    fn_spectral_loss2=None,
    fn_spectral_loss_ms=None,
    stopper=None,
    extra_d_loader=None
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
        gradscaler_g: gradscaler for fp16 - Used for Generator
        gradscaler_d: gradscaler for fp16 - Used for Discriminator
    """
    global global_step, warmup_completed, use_lr_scheduler, lr_scheduler, use_warmup, use_best_step, swap_completed

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers if schedulers is not None else (None, None)

    train_loader = train_loader if train_loader is not None else None
    train_loader.batch_sampler.set_epoch(epoch)

    extra_d_train_loader = endless_loader(extra_d_loader) if extra_d_loader is not None else None

    if writers is not None:
        writer = writers[0]

    audio_preview_log_dir = os.path.join(
        experiment_dir,
        "eval",
        "audio_previews",
    )

    # Best in-epoch step tracking
    if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and use_best_step:
        use_best_step = False
        if rank == 0:
            print("[ ATTENTION ] Best in-epoch step disabled ~ Cannot be used alongside Schedule-Free Optimizers.")
    if use_best_step:
        best_loss_g = float('inf')
        best_state_dict_g = None
        live_sd_g = None

    net_g.train()
    net_d.train()

    if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
        optim_g.train()
    if optimizer_choice_d in ("Sched-Free AdamW", "Sched-Free RAdam"):
        optim_d.train()

    # Partial resume aligning
    current_epoch_start_step = (epoch - 1) * len(train_loader)
    start_batch_idx = global_step - current_epoch_start_step
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
        "loss_kl": deque(maxlen=rolling_loss_steps),
    }
    kl_std_cache = deque(maxlen=rolling_loss_steps)
    last_kl_per_dim = None

    use_amp = config.train.fp16_run and device.type == "cuda"
    r1_interval = max(1, int(getattr(config.train, "r1_interval", 16)))
    r1_gamma = float(getattr(config.train, "r1_gamma", 1.0))
    r1_segment_size = max(
        1,
        int(getattr(config.train, "r1_segment_size", config.train.segment_size)),
    )
    adv_warmup_steps = max(0, int(getattr(config.train, "adv_warmup_steps", 0)))
    kl_warmup_steps = max(0, int(getattr(config.train, "kl_warmup_steps", 0)))
    kl_start_weight = float(getattr(config.train, "kl_start_weight", 0.25))

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
            if not from_scratch:
                num_batches_in_epoch += 1

            # Linear warmup: per-step ramp ( manual `warmup_steps` or `warmup_duration` epochs )
            if warmup_active():
                apply_linear_warmup(optim_g, optim_d, global_step, effective_warmup_steps(train_loader), rank)

            # Clip scheduling
            if not clip_grad_norm_override:
                if grad_clip_scheduling and grad_clip_steps_duration > 0:
                    if global_step < grad_clip_steps_duration:
                        # Clip
                        grad_clip_value_g = grad_clip_value_g_cap if grad_clip_value_g_cap != 0 else float("inf")
                        grad_clip_value_d = grad_clip_value_d_cap if grad_clip_value_d_cap != 0 else float("inf")
                    else:
                        # Release ( or 2nd clip phase )
                        grad_clip_value_g = grad_clip_value_g_release if grad_clip_value_g_release != 0 else float("inf")
                        grad_clip_value_d = grad_clip_value_d_release if grad_clip_value_d_release != 0 else float("inf")
                else:
                    grad_clip_value_g = grad_clip_value_d = float("inf") # Default: No Clipping
            else:
                grad_clip_value_g = clip_grad_norm_override_value_g
                grad_clip_value_d = clip_grad_norm_override_value_d


            # Device handling
            if device.type == "cuda":
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]

            if double_d_updates:
                info_extra = next(extra_d_train_loader)
                # Device handling
                if device.type == "cuda":
                    info_extra = [t.cuda(device_id, non_blocking=True) for t in info_extra]
                elif device.type != "cuda":
                    info_extra = [t.to(device) for t in info_extra]


            # Batch unpacking ( Main )
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid) = info

            # Extra batch unpacking ( Used for additional disc update )
            if double_d_updates:
                (phone_ex, phone_lengths_ex, pitch_ex, pitchf_ex, spec_ex, spec_lengths_ex, y_ex, y_lengths_ex, sid_ex) = info_extra


            # Generator extra forward pass:
            if double_d_updates:
                with torch.no_grad(), autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if train_voc_only:
                        model_output_ex = net_g(spec_ex, spec_lengths_ex, sid_ex, None, None, pitchf_ex, pitch_ex)
                    else:
                        model_output_ex = net_g(spec_ex, spec_lengths_ex, sid_ex, phone_ex, phone_lengths_ex, pitchf_ex, pitch_ex)

                    y_hat_ex, ids_slice_ex, *_ = model_output_ex
                y_ex_sliced = commons.slice_segments(y_ex, ids_slice_ex * config.data.hop_length, config.train.segment_size, dim=3)


            # Generator main forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                if train_voc_only:
                    model_output = net_g(spec, spec_lengths, sid, None, None, pitchf, pitch)
                else:
                    model_output = net_g(spec, spec_lengths, sid, phone, phone_lengths, pitchf, pitch)

                y_hat, ids_slice, x_mask, z_mask, vae_parts = model_output

                # latent samples + Gaussian params for ELBO
                z, z_p, z_p2, m_p, logs_p, m_q, logs_q = vae_parts

                # Slice the original waveform ( y ) to match the generated slice:
                y = commons.slice_segments(y, ids_slice * config.data.hop_length, config.train.segment_size, dim=3)



            # Discriminator updates (independent batches when double)
            main_real_spectrograms = None
            extra_real_spectrograms = None
            if chouwagan_active:
                discriminator_model = (
                    net_d.module if hasattr(net_d, "module") else net_d
                )
                with torch.no_grad(), autocast(device_type="cuda", enabled=False):
                    main_real_spectrograms = (
                        discriminator_model.prepare_spectrograms(y)
                    )
                    if double_d_updates:
                        extra_real_spectrograms = (
                            discriminator_model.prepare_spectrograms(y_ex_sliced)
                        )

            d_updates = [(y, y_hat.detach(), main_real_spectrograms)]
            if double_d_updates:
                d_updates.insert(
                    0,
                    (y_ex_sliced, y_hat_ex.detach(), extra_real_spectrograms),
                )

            _loss_disc_acc, _loss_disc_real_acc, _loss_disc_fake_acc, _grad_norm_d_acc = [], [], [], []

            for y_d_real, y_d_fake, real_spectrograms in d_updates:
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if chouwagan_active:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real,
                            y_d_fake,
                            real_spectrograms=real_spectrograms,
                            pair_batches=True,
                        )
                    else:
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(
                            y_d_real, y_d_fake
                        )

                with autocast(device_type="cuda", enabled=False):
                    loss_disc, loss_disc_real, loss_disc_fake = discriminator_loss(y_d_hat_r, y_d_hat_g)

                optim_d.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler_d.scale(loss_disc).backward()
                    gradscaler_d.unscale_(optim_d)
                    scale_d = gradscaler_d.get_scale()
                    grad_norm_d = _clip_or_sample_grad_norm(
                        net_d.parameters(),
                        grad_clip_value_d,
                        global_step,
                        metrics_update_interval,
                    )
                    gradscaler_d.step(optim_d)
                    gradscaler_d.update()
                    skip_lr_sched_d = (scale_d > gradscaler_d.get_scale())
                else:
                    loss_disc.backward()
                    grad_norm_d = _clip_or_sample_grad_norm(
                        net_d.parameters(),
                        grad_clip_value_d,
                        global_step,
                        metrics_update_interval,
                    )
                    optim_d.step()
                    skip_lr_sched_d = False

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
                if train_dtype == torch.float16:
                    gradscaler_d.scale(loss_r1).backward()
                    gradscaler_d.unscale_(optim_d)
                    grad_norm_d_r1 = _clip_or_sample_grad_norm(
                        net_d.parameters(),
                        grad_clip_value_d,
                        global_step,
                        metrics_update_interval,
                    )
                    gradscaler_d.step(optim_d)
                    gradscaler_d.update()
                else:
                    loss_r1.backward()
                    grad_norm_d_r1 = _clip_or_sample_grad_norm(
                        net_d.parameters(),
                        grad_clip_value_d,
                        global_step,
                        metrics_update_interval,
                    )
                    optim_d.step()
                if grad_norm_d_r1 is not None:
                    _grad_norm_d_acc.append(grad_norm_d_r1)

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

                with autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    with torch.no_grad():
                        _, fmap_r = discriminator_model._forward_audio(
                            y, main_real_spectrograms
                        )
                    y_d_hat_g, fmap_g = discriminator_model._forward_audio(y_hat)
            else:
                with autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)


            # Compute generator losses:
            with autocast(device_type="cuda", enabled=False):

                # Spectral loss
                if spectral_loss == "L1 Mel Loss":
                    y_mel = wave_to_mel(config, y, half=train_dtype, num_mels=config.model.inter_channels if train_voc_only else None)
                    y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype, num_mels=config.model.inter_channels if train_voc_only else None)
                    if swap_l1_to_ms and fn_spectral_loss_ms is not None:
                        # Loss swap: L1 mel fades out, Multi-Scale mel fades in over swap_duration_steps
                        swap_progress = min(1.0, max(0.0, (global_step - swap_start_step) / max(1, swap_duration_steps)))
                        swap_alpha = 0.5 * (1.0 - math.cos(math.pi * swap_progress))  # smooth 0->1 ramp
                        loss_l1_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                        loss_ms_mel = fn_spectral_loss_ms(y, y_hat) * config.train.c_mel / 3.0
                        loss_spectral = (1.0 - swap_alpha) * loss_l1_mel + swap_alpha * loss_ms_mel
                        if swap_progress >= 1.0 and not swap_completed:
                            swap_completed = True
                            print(f"[TRAIN] LOSS SWAP complete at step {global_step} - now using Multi-Scale mel loss")
                    else:
                        loss_spectral = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                elif spectral_loss == "Multi-Scale Mel Loss":
                    loss_spectral = fn_spectral_loss(y, y_hat) * config.train.c_mel / 3.0 # * 15
                elif spectral_loss == "Hybrid L1":
                    # L1 Mel
                    y_mel = wave_to_mel(config, y, half=train_dtype, num_mels=config.model.inter_channels if train_voc_only else None)
                    y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype, num_mels=config.model.inter_channels if train_voc_only else None)
                    loss_l1_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel # * 45
                    # MS-STFT
                    loss_ms_stft = fn_spectral_loss2(y_hat.float(), y.float()) * 1.0
                    # Loss
                    loss_spectral = loss_l1_mel + loss_ms_stft

                # Feature Matching loss
                loss_fm = feature_loss(fmap_r, fmap_g) * 2.0

                # Generator loss
                loss_adv = generator_loss(y_d_hat_g)

                # ChouwaGAN uses monotonic KL protection instead of cyclic decay.
                if chouwagan_active and not train_voc_only:
                    if kl_warmup_steps:
                        kl_progress = min(1.0, global_step / kl_warmup_steps)
                    else:
                        kl_progress = 1.0
                    kl_beta = kl_start_weight + (1.0 - kl_start_weight) * kl_progress
                elif use_kl_annealing:
                    annealing_cycle_steps = len(train_loader) * kl_annealing_cycle_duration
                    kl_beta = 0.5 * (1 - math.cos((global_step % annealing_cycle_steps) * (math.pi / annealing_cycle_steps)))
                else:
                    kl_beta = 1.0

                # KL ( Kullback–Leibler divergence ) loss
                if train_voc_only:
                    # Vocoder-only: no VAE frontend, no KL.
                    loss_kl = torch.zeros((), device=y.device)
                else:
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask, z_p2) * config.train.c_kl

                    # KL diagnostic: per-dim std (raw, without the non-negativity clamp)
                    with torch.no_grad():
                        if z_p2 is not None:
                            raw_kl = (logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
                                      + logs_p - logs_q - 0.5 + 0.5 * ((z_p2 - m_p) ** 2) * torch.exp(-2 * logs_p)) * 0.5
                        else:
                            raw_kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
                        raw_kl_per_dim = (raw_kl * z_mask).sum(dim=(0, 2)) / z_mask.sum(dim=(0, 2)).clamp(min=1)
                        kl_std_cache.append(raw_kl_per_dim.std().item())
                        last_kl_per_dim = raw_kl_per_dim.detach()

                if chouwagan_active:
                    if adv_warmup_steps:
                        adv_progress = min(1.0, global_step / adv_warmup_steps)
                    else:
                        adv_progress = 1.0
                    gan_weight = adv_progress
                else:
                    gan_weight = 1.0

                loss_core = loss_spectral + loss_kl * kl_beta
                loss_gan = (loss_adv + loss_fm) * gan_weight
                loss_gen_total = loss_core + loss_gan

            # Generator backward and update:
            optim_g.zero_grad(set_to_none=True)
            if chouwagan_active:
                if train_dtype == torch.float16:
                    scale_g = gradscaler_g.get_scale()
                    gradscaler_g.scale(loss_core).backward(retain_graph=True)
                    _add_decoder_only_gradients(loss_gan, net_g, scale=scale_g)
                    gradscaler_g.unscale_(optim_g)
                    grad_norm_g = _clip_or_sample_grad_norm(
                        net_g.parameters(),
                        grad_clip_value_g,
                        global_step,
                        metrics_update_interval,
                    )
                    gradscaler_g.step(optim_g)
                    gradscaler_g.update()
                    skip_lr_sched_g = scale_g > gradscaler_g.get_scale()
                else:
                    loss_core.backward(retain_graph=True)
                    _add_decoder_only_gradients(loss_gan, net_g)
                    grad_norm_g = _clip_or_sample_grad_norm(
                        net_g.parameters(),
                        grad_clip_value_g,
                        global_step,
                        metrics_update_interval,
                    )
                    optim_g.step()
                    skip_lr_sched_g = False
            elif train_dtype == torch.float16:
                gradscaler_g.scale(loss_gen_total).backward() # Scale and backward of the loss
                gradscaler_g.unscale_(optim_g) # Unscale
                scale_g = gradscaler_g.get_scale() # To retrieve current gradscaler's scaling
                grad_norm_g = _clip_or_sample_grad_norm(
                    net_g.parameters(),
                    grad_clip_value_g,
                    global_step,
                    metrics_update_interval,
                )
                gradscaler_g.step(optim_g) # Optim step
                gradscaler_g.update() # Scaler update, to prepare the scaling for the next iteration
                skip_lr_sched_g = (scale_g > gradscaler_g.get_scale())
            else:
                loss_gen_total.backward() # Loss backward
                grad_norm_g = _clip_or_sample_grad_norm(
                    net_g.parameters(),
                    grad_clip_value_g,
                    global_step,
                    metrics_update_interval,
                )
                optim_g.step() # Optim step
                skip_lr_sched_g = False

            if discriminator_parameter_states is not None:
                for parameter, requires_grad in zip(
                    discriminator_model.parameters(),
                    discriminator_parameter_states,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)


            # Track best step in this epoch (FM + Spectral)
            if use_best_step:
                loss_val = loss_gen_total.detach() if from_scratch else (loss_fm + loss_spectral).detach()
                if loss_val < best_loss_g:
                    best_loss_g = loss_val
                    model_g = net_g.module if hasattr(net_g, "module") else net_g
                    best_state_dict_g = {k: v.detach().clone() for k, v in model_g.state_dict().items()}


            # Per step exp lr decay for both optimizers.
            if use_lr_scheduler and (not warmup_active() or warmup_completed) and lr_scheduler == "exp decay step":
                if not skip_lr_sched_g:  # Skip when the generator scaler found NaN/Inf gradients.
                    scheduler_g.step()
                if not skip_lr_sched_d:  # Skip when the discriminator scaler found NaN/Inf gradients.
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
                else:
                    writer.add_scalar("Grad_Norm_Diag/G_Skipped", 1, global_step)

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
                        # determine loss or grad category
                        category = "loss" if "loss" in key else "grad"
                        # dynamic labeling
                        label = f"{category}_avg_{rolling_loss_steps}/{key}_{rolling_loss_steps}"
                        # Calculate mean
                        val = torch.stack(list(queue)).mean().item() if torch.is_tensor(queue[0]) else sum(queue)/len(queue)
                        scalar_dict_rolling[label] = val

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)

                # KL diagnostics (diag tab)
                if len(kl_std_cache) > 0:
                    diag_scalar = sum(kl_std_cache) / len(kl_std_cache)
                    summarize(writer=writer, global_step=global_step, scalars={"diag/kl_std": diag_scalar})
                    writer.add_histogram("diag/kl_per_dim_hist", last_kl_per_dim.cpu(), global_step)
                flush_writer(writer, rank)

            preview_interval = (
                pretrain_preview_interval if from_scratch else finetune_preview_interval
            )
            if pretrain_preview and rank == 0 and global_step % preview_interval == 0:
                if optimizer_choice_g in ("AdamWScheduleFree", "RAdamScheduleFree"):
                    optim_g.eval()
                o = eval_infer(net_g, reference)
                if optimizer_choice_g in ("AdamWScheduleFree", "RAdamScheduleFree"):
                    optim_g.train()
                audio_dict = {"generated": o[0, :, :]}
                image_dict = {}
                if reference_audio is not None:
                    audio_dict["original"] = reference_audio[0]
                    eval_original_mel = wave_to_mel(
                        config,
                        reference_audio,
                        half=train_dtype,
                        num_mels=config.model.inter_channels if train_voc_only else None,
                    )
                    eval_generated_mel = wave_to_mel(
                        config,
                        o,
                        half=train_dtype,
                        num_mels=config.model.inter_channels if train_voc_only else None,
                    )
                    image_dict["eval/pretrain_mel_comparison"] = plot_mel_comparison_to_numpy(
                        eval_original_mel[0].detach().float().cpu().numpy(),
                        eval_generated_mel[0].detach().float().cpu().numpy(),
                    )
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                )
                write_audio_preview(
                    audio_preview_log_dir,
                    global_step,
                    audio_dict,
                    config.data.sample_rate,
                    category="pretrain",
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

            if early_stopper(
                stopper, rank, global_step, epoch, architecture, 
                [net_g, net_d], [optim_g, optim_d], config, 
                experiment_dir, gradscaler_g, gradscaler_d, save_weight_models,
                model_name, vocoder, n_gpus
            ):
                return True

        # end of batch train
    # end of Rich progress

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Used for tensorboard chart - all/mel
        if train_voc_only:
            # The dataloader's spec slot already holds the 192-bin mel spectrogram.
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sample_rate,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )

        # For fp16 we need to .half() the mel spec
        if train_dtype == torch.float16:
            mel = mel.half()

        # Used for tensorboard mel charts
        y_mel = commons.slice_segments(mel, ids_slice, config.train.segment_size // config.data.hop_length, dim=3) # slice/mel_org
        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype, num_mels=config.model.inter_channels if train_voc_only else None) # slice/mel_gen

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

        # Determine the plot data type
        if train_dtype == torch.float16:
            plot_dtype = torch.float16
        else:
            plot_dtype = torch.float32

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().to(plot_dtype).numpy()),
        }

        # At each epoch save point:
        if epoch % epoch_save_frequency == 0:

            # Swap to best-step weights for eval_infer preview
            if use_best_step and best_state_dict_g is not None:
                model_g = net_g.module if hasattr(net_g, "module") else net_g
                live_sd_g = {k: v.detach().clone() for k, v in model_g.state_dict().items()}
                model_g.load_state_dict(best_state_dict_g)

            # Inferencing on reference sample


            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and not (use_best_step and best_state_dict_g is not None):
                optim_g.eval()
            o = eval_infer(net_g, reference)
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam") and not (use_best_step and best_state_dict_g is not None):
                optim_g.train()
            audio_dict = {"generated": o[0, :, :]} # Eval-infer samples
            if reference_audio is not None:
                audio_dict["original"] = reference_audio[0]

                eval_original_mel = wave_to_mel(
                    config,
                    reference_audio,
                    half=train_dtype,
                    num_mels=config.model.inter_channels if train_voc_only else None,
                )
                eval_generated_mel = wave_to_mel(
                    config,
                    o,
                    half=train_dtype,
                    num_mels=config.model.inter_channels if train_voc_only else None,
                )
                image_dict["eval/mel_comparison"] = plot_mel_comparison_to_numpy(
                    eval_original_mel[0].detach().float().cpu().numpy(),
                    eval_generated_mel[0].detach().float().cpu().numpy(),
                )

            # Restore live weights immediately ~ checkpoint saving stays raw
            if use_best_step and live_sd_g is not None:
                model_g.load_state_dict(live_sd_g)
                live_sd_g = None

            # Logging
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            write_audio_preview(
                audio_preview_log_dir,
                global_step,
                audio_dict,
                config.data.sample_rate,
                category="eval",
            )
            flush_writer(writer, rank)
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % epoch_save_frequency == 0:
            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            if save_only_latest_net_models:
                old_files = glob.glob(os.path.join(experiment_dir, "G_*.pth")) + glob.glob(os.path.join(experiment_dir, "D_*.pth"))
                for f in old_files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Switch to eval mode for Schedule-Free optims before saving (uses averaged params)
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_g.eval()
            if optimizer_choice_d in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_d.eval()


            # Save Generator checkpoint (live weights; averaging was restored above)
            save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, gradscaler_g)

            # Save Discriminator checkpoint
            save_checkpoint(net_d, optim_d, config.train.learning_rate_d, epoch, d_path, gradscaler_d)

            # Switch back to train mode after saving
            if optimizer_choice_g in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_g.train()
            if optimizer_choice_d in ("Sched-Free AdamW", "Sched-Free RAdam"):
                optim_d.train()


            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_add.append(os.path.join(experiment_dir, weight_model_name))

        # Check completion
        if epoch >= total_epoch_count:
            print(f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_total.item(), 3)} loss gen.")
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append(os.path.join(experiment_dir, weight_model_name))
            done = True

        if model_add:
            model_g = net_g.module if hasattr(net_g, "module") else net_g
            ckpt = best_state_dict_g if (use_best_step and best_state_dict_g is not None) else model_g.state_dict()

            for m in model_add:
                if not os.path.exists(m):
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
