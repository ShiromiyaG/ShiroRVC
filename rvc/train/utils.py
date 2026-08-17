import os
import glob
import json
import signal
import sys
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import soundfile as sf

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm


MATPLOTLIB_FLAG = False

debug_save_load = False

from itertools import chain
from mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.lib.terminal import print_settings_panel

def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def remap_optimizer_state(optimizer, model, opt_state):
    """Prune a saved optimizer state dict so it fits an optimizer whose parameter set changed
        ( e.g. layers were frozen between runs ).

    The moments of params that still exist are kept and removed params' state is dropped.
    Returns None if nothing can be salvaged.
    """
    model_params = list(model.parameters())
    old_index = {id(p): i for i, p in enumerate(model_params)}
    new_params = [p for g in optimizer.param_groups for p in g["params"]]
    new_index = {id(p): j for j, p in enumerate(new_params)}

    state = {}
    for old_i, group_state in opt_state.get("state", {}).items():
        if isinstance(old_i, int) and old_i < len(model_params):
            p = model_params[old_i]
            if id(p) in new_index:
                state[new_index[id(p)]] = group_state

    saved_groups = opt_state.get("param_groups", [])
    if not saved_groups:
        return None

    param_groups = []
    for i, group in enumerate(optimizer.param_groups):
        new_group = {}
        for key, value in saved_groups[min(i, len(saved_groups) - 1)].items():
            if key != "params":
                new_group[key] = value
        new_group["params"] = group["params"]
        if "lr_scale" in group:
            new_group["lr_scale"] = group["lr_scale"]
        param_groups.append(new_group)

    if not state and not param_groups:
        return None
    return {"state": state, "param_groups": param_groups}


def load_checkpoint(checkpoint_path, model, optimizer=None, strict_load=True):
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state = model.module if hasattr(model, "module") else model
    model_state.load_state_dict(checkpoint_dict["model"], strict=strict_load)

    if optimizer:
        opt_state = checkpoint_dict.get("optimizer")
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
                print("Loaded optimizer state.")
            except ValueError:
                print("[WARNING] Optimizer parameter set changed ( e.g. layers were frozen ).")
                print("[WARNING] Pruning saved optimizer state to the surviving params; LR re-anchored to the saved value.")
                pruned = remap_optimizer_state(optimizer, model_state, opt_state)
                if pruned is not None:
                    optimizer.load_state_dict(pruned)
                    print("Loaded optimizer state ( pruned to surviving params ).")
                else:
                    print("[WARNING] Could not remap optimizer state; starting optimizer fresh.")
        else:
            if strict_load:
                raise ValueError(f"[ERROR] Missing optimizer state...")
            else:
                print("[WARNING] No optimizer state found in checkpoint, starting optimizer fresh.")


    print(f"Loaded checkpoint '{checkpoint_path}' (iteration {checkpoint_dict['iteration']})")
    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        checkpoint_dict.get("gradscaler", {})
    )

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, gradscaler=None):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    if gradscaler is not None:
        checkpoint_data["gradscaler"] = gradscaler.state_dict()

    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def _audio_preview_event_files(log_dir):
    if not os.path.isdir(log_dir):
        return []
    return sorted(
        (
            path
            for path in glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            if os.path.isfile(path)
        ),
        key=os.path.getmtime,
    )


def trim_audio_preview_events(log_dir, keep=10):
    """Keep only the newest audio-preview event files in the dedicated log dir."""
    keep = max(1, int(keep))
    event_files = _audio_preview_event_files(log_dir)
    for path in event_files[:-keep]:
        try:
            os.remove(path)
        except OSError:
            continue


def write_audio_preview(
    log_dir,
    global_step,
    audios,
    audio_sample_rate,
    category,
    keep=10,
):
    """Write one grouped audio preview and retain only the newest previews."""
    if not audios:
        return

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=log_dir,
        flush_secs=1,
        filename_suffix=f".{category}.{int(global_step)}.{time.time_ns()}",
    )
    try:
        for name, audio in audios.items():
            if torch.is_tensor(audio):
                audio = audio.detach().float().cpu()
            tag = f"audio/{category}/step_{int(global_step):08d}/{name}"
            writer.add_audio(tag, audio, global_step, audio_sample_rate)
        writer.flush()
    finally:
        writer.close()

    trim_audio_preview_events(log_dir, keep=keep)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]
    plt.close(fig)
    return data


def plot_mel_comparison_to_numpy(original_mel, generated_mel):
    """Render original mel, generated mel, and absolute mel error as one image."""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    original_mel = np.asarray(original_mel, dtype=np.float32)
    generated_mel = np.asarray(generated_mel, dtype=np.float32)

    if original_mel.ndim != 2 or generated_mel.ndim != 2:
        raise ValueError("Mel comparison expects two-dimensional arrays.")

    mel_bins = min(original_mel.shape[0], generated_mel.shape[0])
    frames = min(original_mel.shape[1], generated_mel.shape[1])
    original_mel = original_mel[:mel_bins, :frames]
    generated_mel = generated_mel[:mel_bins, :frames]
    error_mel = np.abs(generated_mel - original_mel)

    mel_values = np.concatenate((original_mel.ravel(), generated_mel.ravel()))
    mel_values = mel_values[np.isfinite(mel_values)]
    if mel_values.size:
        mel_min = float(mel_values.min())
        mel_max = float(mel_values.max())
    else:
        mel_min, mel_max = 0.0, 1.0
    if mel_min == mel_max:
        mel_max = mel_min + 1.0

    error_values = error_mel[np.isfinite(error_mel)]
    error_max = float(error_values.max()) if error_values.size else 1.0
    if error_max <= 0:
        error_max = 1.0

    fig = plt.figure(figsize=(12, 7.5))
    layout = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)
    panels = (
        (original_mel, "Original mel", "magma", mel_min, mel_max),
        (generated_mel, "Generated mel", "magma", mel_min, mel_max),
        (error_mel, "Absolute mel error", "viridis", 0.0, error_max),
    )

    axes = []
    for index, (mel, title, cmap, vmin, vmax) in enumerate(panels):
        axis = fig.add_subplot(layout[index])
        image = axis.imshow(
            mel,
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_ylabel("Mel bins")
        if index == len(panels) - 1:
            axis.set_xlabel("Frames")
        fig.colorbar(image, ax=axis, pad=0.01)
        axes.append(axis)

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        top=0.96,
        bottom=0.07,
        hspace=0.55,
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Load filepaths and associated text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
    """
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config_from_json(config_save_path):
    try:
        with open(config_save_path, "r") as f:
            config = json.load(f)
        config = HParams(**config)
        return config
    except FileNotFoundError:
        print(
            f"Model config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)

def flush_writer(writer, rank):
    if rank == 0 and writer is not None:
        writer.flush()


# Currently has no use, kept for future ig.
def flush_writer_grad(writer, rank, global_step):
    if rank == 0 and writer is not None and global_step % 10 == 0:
        writer.flush()


def block_tensorboard_flush_on_exit(writer):
    def handler(signum, frame):
        print("[Warning] Training interrupted. Skipping flush to avoid partial logs.")
        try:
            writer.close()
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def si_sdr(preds, target, eps=1e-8):
    """Scale-Invariant SDR"""
    preds = preds - preds.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    scaling_factor = (preds * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scaling_factor * target

    noise = preds - projection

    si_sdr_value = 10 * torch.log10((projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + eps)

    return si_sdr_value.mean()


def wave_to_mel(config, waveform, half, num_mels=None):
    mel_spec = mel_spectrogram_torch(
        waveform.float().squeeze(1),
        config.data.filter_length,
        num_mels if num_mels is not None else config.data.n_mel_channels,
        config.data.sample_rate,
        config.data.hop_length,
        config.data.win_length,
        config.data.mel_fmin,
        config.data.mel_fmax,
    )
    if half == torch.float16:
        mel_spec = mel_spec.half()
    elif half == torch.float32 or half == torch.bfloat16:
        pass

    return mel_spec


def small_model_naming(model_name, epoch, global_step):
    return f"{model_name}_{epoch}e_{global_step}s.pth"


def old_session_cleanup(now_dir, model_name):
    for root, dirs, files in os.walk(os.path.join(now_dir, "logs", model_name), topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension == ".0"
                or (file_name.startswith("D_") and file_extension == ".pth")
                or (file_name.startswith("G_") and file_extension == ".pth")
                or (file_name.startswith("added") and file_extension == ".index")
            ):
                os.remove(file_path)
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                os.rmdir(folder_path)

    print("[INIT] Cleanup done!")


def print_init_setup(
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
):
    if rank != 0:
        return

    tf32_enabled = (
        torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.allow_tf32
    )
    if config.train.fp16_run:
        precision = "TF32 / FP16 - AMP" if tf32_enabled else "FP32 / FP16 - AMP"
    else:
        precision = "TF32" if tf32_enabled else "FP32"

    rows = [
        ("PRECISION", precision),
        ("cudnn.benchmark", torch.backends.cudnn.benchmark),
        ("cudnn.deterministic", torch.backends.cudnn.deterministic),
        ("Optimizer (G)", optimizer_choice_g),
        ("Optimizer (D)", optimizer_choice_d),
        ("Spectral loss", spectral_loss),
    ]

    def scheduler_value(name, gamma):
        if name == "none":
            return "Disabled"
        if name in ("cosine annealing", "cosine annealing epoch"):
            return "cosine annealing"
        return f"{name}, gamma: {gamma}"

    rows.extend(
        [
            ("LR scheduler (G/D)", scheduler_value(lr_scheduler, exp_decay_gamma)),
        ]
    )

    if use_warmup:
        rows.append(("Warmup", f"{warmup_duration} epochs"))
    if use_kl_annealing:
        rows.append(("KL loss annealing", f"{kl_annealing_cycle_duration} epochs"))

    print_settings_panel(rows)

def train_loader_safety(train_loader):
    if len(train_loader) < 3:
        print("Not enough data present in the training set. Perhaps you didn't slice the audio files? ( Preprocessing step )")
        os._exit(1)


def verify_spk_dim(
    config,
    model_info_path,
    experiment_dir,
    latest_checkpoint_path,
    rank,
    pretrainG
):
    embedder_name = "contentvec"  # Default embedder
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            spk_dim = model_info["speakers_id"]
    except Exception as e:
        print(f"Could not load model info file: {e}. Using defaults.")

    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

    if rank == 0:
        print(f"[INIT] Initializing the generator with: {spk_dim} speakers.")

    return spk_dim


def early_stopper(
    stopper, 
    rank, 
    global_step, 
    epoch, 
    architecture, 
    nets, 
    optims, 
    config, 
    experiment_dir, 
    gradscaler_g,
    gradscaler_d,
    save_weight_models,
    model_name,
    vocoder,
    n_gpus
):
    if stopper is not None and stopper.stop_triggered:
        net_g, net_d = nets
        optim_g, optim_d = optims

        if rank == 0:
            print(f"[TRAINING] Saving the models at steps: '{global_step}' in progress ...")

            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            # Save Generator checkpoint
            save_checkpoint(net_g, optim_g, config.train.learning_rate_g, epoch, g_path, gradscaler_g)
            # Save Discriminator checkpoint
            save_checkpoint(net_d, optim_d, config.train.learning_rate_d, epoch, d_path, gradscaler_d)

            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_path = os.path.join(experiment_dir, weight_model_name)

                ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                extract_model(
                    ckpt=ckpt, 
                    sr=config.data.sample_rate, 
                    name=model_name, 
                    model_path=model_path, 
                    epoch=epoch, 
                    step=global_step, 
                    hps=config, 
                    vocoder=vocoder, 
                    architecture=architecture, 
                )
                print(f"[TRAINING] All finished .. You're good to go.")
        if n_gpus > 1:
            dist.barrier()
        return True
    return False
