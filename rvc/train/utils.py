import os
import glob
import json
import signal
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

import numpy as np
import soundfile as sf

from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rvc.train.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_AUDIO_TAG,
    TENSORBOARD_VALIDATION_AXIS_X,
    TENSORBOARD_VALIDATION_AXIS_Y,
    TENSORBOARD_VALIDATION_DB_LABEL,
    TENSORBOARD_VALIDATION_DIFFERENCE_LABEL,
    TENSORBOARD_VALIDATION_FOOTER,
    TENSORBOARD_VALIDATION_MEL_TAG,
    TENSORBOARD_VALIDATION_MEL_TITLES,
    TENSORBOARD_VALIDATION_PREVIEW_DIR,
    TENSORBOARD_VALIDATION_SOURCE_TAG,
)


MATPLOTLIB_FLAG = False
#: Defaults for the validation preview figure.  Overridable per model through
#: ``validation_preview_dpi`` / ``_width`` / ``_height`` in the config JSON.
#:
#: The two knobs are not interchangeable.  ``figsize`` is in inches and dpi
#: converts to pixels, so raising **dpi** scales the whole figure -- panels and
#: text together -- and is what "make it sharper" means.  Raising **figsize**
#: at a fixed dpi gives the panels more room while the text stays the same
#: physical size, so labels shrink relative to the plot.  67 x 24in lands on
#: 1608px wide, which is the historical output.
VALIDATION_PREVIEW_DPI = 67
VALIDATION_PREVIEW_FIGSIZE = (24.0, 5.8)

debug_save_load = False

from itertools import chain
from mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.lib.terminal import (
    error as print_error,
    info,
    print_settings_panel,
    warning,
)

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


def load_checkpoint(checkpoint_path, model, optimizer=None, strict_load=True, ema=None):
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state = model.module if hasattr(model, "module") else model
    expected_architecture = getattr(model_state, "architecture_id", None)
    checkpoint_architecture = checkpoint_dict.get("architecture_id")
    if expected_architecture and expected_architecture != "vits_gaussian_v1":
        if checkpoint_architecture != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch: expected '{expected_architecture}', "
                f"received '{checkpoint_architecture or 'unknown'}'."
            )
    model_state.load_state_dict(checkpoint_dict["model"], strict=strict_load)

    if ema is not None:
        # Absent from every checkpoint written before the EMA existed.  Seeding
        # the shadow from the restored weights is the correct restart: it is
        # what the average would be if this were step zero, and it costs only
        # the averaging already done.
        if ema.load_state_dict(checkpoint_dict.get("ema"), model_state):
            info(f"Loaded EMA state ({ema.updates} updates).", tag="[RESUME]")
        else:
            warning(
                "No usable EMA state in the checkpoint; seeding it from the "
                "weights.",
                tag="[RESUME]",
            )

    if optimizer:
        opt_state = checkpoint_dict.get("optimizer")
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
                info("Loaded optimizer state.", tag="[RESUME]")
            except ValueError:
                warning(
                    "The optimizer's parameter set changed (layers were frozen, "
                    "for instance); pruning the saved state to the surviving "
                    "params, with the LR re-anchored to the saved value.",
                    tag="[RESUME]",
                )
                pruned = remap_optimizer_state(optimizer, model_state, opt_state)
                if pruned is not None:
                    optimizer.load_state_dict(pruned)
                    info(
                        "Loaded optimizer state (pruned to the surviving params).",
                        tag="[RESUME]",
                    )
                else:
                    warning(
                        "Could not remap the optimizer state; starting the "
                        "optimizer fresh.",
                        tag="[RESUME]",
                    )
        else:
            if strict_load:
                raise ValueError(f"[ERROR] Missing optimizer state...")
            else:
                warning(
                    "No optimizer state in the checkpoint; starting the "
                    "optimizer fresh.",
                    tag="[RESUME]",
                )


    info(
        f"Loaded '{os.path.basename(checkpoint_path)}' at iteration "
        f"{checkpoint_dict['iteration']}.",
        tag="[RESUME]",
    )
    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        # Retained so existing FP16-era checkpoints keep unpacking; training is
        # FP32 only now, so nothing consumes it.
        {},
    )

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, ema=None):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    model_instance = model.module if hasattr(model, "module") else model

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    architecture_id = getattr(model_instance, "architecture_id", None)
    if architecture_id is not None:
        checkpoint_data["architecture_id"] = architecture_id
    # Additive key.  "model" still holds the live weights, so anything that
    # reads these checkpoints without knowing about the EMA -- older code, the
    # extractor, the blender -- keeps seeing exactly what it saw before.
    if ema is not None:
        checkpoint_data["ema"] = ema.state_dict()

    torch.save(checkpoint_data, checkpoint_path)
    info(f"Saved '{os.path.basename(checkpoint_path)}'.", tag="[SAVE]")

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


def limit_audio_peak(audio, max_peak=0.98):
    """Return a finite mono preview whose peak stays inside TensorBoard's range."""
    if torch.is_tensor(audio):
        preview = audio.detach().float().cpu()
        preview = torch.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        preview = preview.reshape(-1)
        if preview.numel() == 0:
            return preview
        peak = preview.abs().amax()
        if torch.isfinite(peak) and peak.item() > max_peak:
            preview = preview * (max_peak / peak)
        return preview.clamp(-max_peak, max_peak)

    preview = np.asarray(audio, dtype=np.float32)
    preview = np.nan_to_num(
        preview,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=True,
    ).reshape(-1)
    if preview.size == 0:
        return preview
    peak = float(np.max(np.abs(preview)))
    if np.isfinite(peak) and peak > max_peak:
        preview *= max_peak / peak
    return np.clip(preview, -max_peak, max_peak).astype(np.float32, copy=False)


def _mel_to_numpy(mel):
    if torch.is_tensor(mel):
        mel = mel.detach().float().cpu().numpy()
    mel = np.nan_to_num(np.asarray(mel, dtype=np.float32), copy=True)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError("Validation mel spectrograms must be two-dimensional.")
    return mel


def plot_validation_preview_to_figure(
    predicted_mel,
    target_mel,
    epoch,
    global_step,
    sample_index=0,
    sample_rate=22050,
    hop_length=256,
    dpi=None,
    figsize=None,
):
    """Create a dark TensorBoard-style three-panel validation report.

    ``hop_length`` sets the time axis and must match the config the mels were
    produced with -- the default is a fallback, not a good guess.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    target_mel = _mel_to_numpy(target_mel)
    predicted_mel = _mel_to_numpy(predicted_mel)
    mel_bins = min(target_mel.shape[0], predicted_mel.shape[0])
    frames = min(target_mel.shape[1], predicted_mel.shape[1])
    if mel_bins <= 0 or frames <= 0:
        raise ValueError("Validation mel spectrograms must not be empty.")

    target_mel = target_mel[:mel_bins, :frames]
    predicted_mel = predicted_mel[:mel_bins, :frames]
    difference_mel = predicted_mel - target_mel

    shared_values = np.concatenate((target_mel.ravel(), predicted_mel.ravel()))
    shared_low = float(np.quantile(shared_values, 0.01))
    shared_high = float(np.quantile(shared_values, 0.99))
    if not np.isfinite(shared_low) or not np.isfinite(shared_high):
        shared_low, shared_high = 0.0, 1.0
    if shared_low >= shared_high:
        shared_high = shared_low + 1.0

    difference_limit = max(
        1e-5,
        float(np.quantile(np.abs(difference_mel), 0.99)),
    )
    time_axis = np.arange(frames, dtype=np.float32) * float(hop_length) / float(sample_rate)
    # The mel filterbank spans DC to Nyquist, not DC to the sample rate. Using
    # the sample rate as the extent labelled every frequency tick at twice its
    # true value.
    top_frequency = float(sample_rate) / 2.0
    figure, axes = plt.subplots(
        1,
        3,
        figsize=tuple(figsize) if figsize else VALIDATION_PREVIEW_FIGSIZE,
        dpi=float(dpi) if dpi else VALIDATION_PREVIEW_DPI,
        facecolor="#10161f",
        gridspec_kw={
            "left": 0.045,
            "right": 0.955,
            "wspace": 0.16,
            "bottom": 0.25,
            "top": 0.86,
        },
    )
    figure.patch.set_facecolor("#10161f")
    panels = (predicted_mel, target_mel, difference_mel)
    panel_cmaps = ("inferno", "inferno", "turbo")
    panel_norms = (
        (shared_low, shared_high),
        (shared_low, shared_high),
        (-difference_limit, difference_limit),
    )
    for axis, mel, title, cmap, (vmin, vmax) in zip(
        axes,
        panels,
        TENSORBOARD_VALIDATION_MEL_TITLES,
        panel_cmaps,
        panel_norms,
    ):
        axis.set_facecolor("#10161f")
        rendered = axis.imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(time_axis[0], time_axis[-1], 0, top_frequency),
        )
        axis.set_title(title, color="#f4f4f4", fontsize=17, fontweight="bold", pad=12)
        axis.set_xlabel(TENSORBOARD_VALIDATION_AXIS_X, color="#f4f4f4", fontsize=11)
        axis.set_ylabel(TENSORBOARD_VALIDATION_AXIS_Y, color="#f4f4f4", fontsize=11)
        frequency_ticks = np.linspace(0.0, top_frequency, 5)
        axis.set_yticks(frequency_ticks)
        axis.set_yticklabels(
            [
                f"{frequency / 1000:g}k" if frequency else "0"
                for frequency in frequency_ticks
            ]
        )
        axis.tick_params(colors="#f4f4f4", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#c9cdd3")
        colorbar = figure.colorbar(
            rendered,
            ax=axis,
            orientation="horizontal",
            fraction=0.055,
            pad=0.12,
        )
        colorbar.set_label(
            TENSORBOARD_VALIDATION_DIFFERENCE_LABEL
            if title == TENSORBOARD_VALIDATION_MEL_TITLES[-1]
            else TENSORBOARD_VALIDATION_DB_LABEL,
            color="#d7dbe1",
            fontsize=8,
        )
        colorbar.ax.tick_params(colors="#d7dbe1", labelsize=8)
        colorbar.outline.set_edgecolor("#c9cdd3")

    figure.text(
        0.5,
        0.055,
        TENSORBOARD_VALIDATION_FOOTER.format(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            n_mels=int(mel_bins),
            epoch=int(epoch),
            step=int(global_step),
        ),
        color="#bfc5ce",
        fontsize=10,
        ha="center",
    )
    return figure


def log_validation_preview(
    writer,
    experiment_dir,
    epoch,
    sample_index,
    global_step,
    sample_rate,
    predicted_mel,
    target_mel,
    predicted_wave=None,
    target_wave=None,
    source=None,
    hop_length=256,
    dpi=None,
    figsize=None,
):
    """Save and log one organized validation preview with mel difference."""
    sample_stem = f"sample_{int(sample_index):02d}"
    preview_dir = os.path.join(
        str(experiment_dir),
        TENSORBOARD_VALIDATION_PREVIEW_DIR,
        f"epoch_{int(epoch):04d}",
    )
    mel_dir = os.path.join(preview_dir, "mel")
    audio_dir = os.path.join(preview_dir, "audio")
    os.makedirs(mel_dir, exist_ok=True)

    figure = plot_validation_preview_to_figure(
        predicted_mel=predicted_mel,
        target_mel=target_mel,
        epoch=epoch,
        global_step=global_step,
        sample_index=sample_index,
        sample_rate=sample_rate,
        hop_length=hop_length,
        dpi=dpi,
        figsize=figsize,
    )
    image_path = os.path.join(mel_dir, f"{sample_stem}.png")
    try:
        figure.canvas.draw()
        composite = np.asarray(
            figure.canvas.buffer_rgba(),
            dtype=np.uint8,
        )[..., :3].copy()
        # ``figure.dpi`` rather than the module default: the figure may have
        # been built at an overridden dpi, and saving at a different one would
        # write a PNG that does not match what TensorBoard received.
        figure.savefig(image_path, dpi=figure.dpi)

        if writer is not None:
            writer.add_image(
                TENSORBOARD_VALIDATION_MEL_TAG.format(sample=sample_stem),
                torch.from_numpy(composite).permute(2, 0, 1),
                global_step,
                dataformats="CHW",
            )
            if source:
                writer.add_text(
                    TENSORBOARD_VALIDATION_SOURCE_TAG.format(sample=sample_stem),
                    str(source),
                    global_step=global_step,
                )
    finally:
        plt.close(figure)

    if predicted_wave is None or target_wave is None:
        return image_path

    os.makedirs(audio_dir, exist_ok=True)
    predicted_wave = limit_audio_peak(predicted_wave)
    target_wave = limit_audio_peak(target_wave)
    audio_length = min(predicted_wave.numel(), target_wave.numel())
    if audio_length <= 0:
        return image_path
    predicted_wave = predicted_wave[:audio_length]
    target_wave = target_wave[:audio_length]

    generated_path = os.path.join(audio_dir, f"{sample_stem}_generated.wav")
    original_path = os.path.join(audio_dir, f"{sample_stem}_original.wav")
    sf.write(generated_path, predicted_wave.numpy(), int(sample_rate), subtype="PCM_16")
    sf.write(original_path, target_wave.numpy(), int(sample_rate), subtype="PCM_16")

    if writer is not None:
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"],
            ),
            predicted_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["original"],
            ),
            target_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )

    return image_path


def log_tensorboard_media(
    writer,
    namespace,
    global_step,
    sample_rate,
    figure=None,
    audio=None,
    text=None,
):
    """Write one sample's text, mel figure, and audio under a single namespace."""
    if writer is None:
        return

    for name, value in (text or {}).items():
        writer.add_text(
            f"{namespace}/{name}",
            str(value),
            global_step=global_step,
        )
    if figure is not None:
        writer.add_figure(
            f"{namespace}/mel",
            figure,
            global_step=global_step,
        )
    for name, value in (audio or {}).items():
        writer.add_audio(
            f"{namespace}/audio/{name}",
            limit_audio_peak(value),
            global_step=global_step,
            sample_rate=sample_rate,
        )


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


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


#: Application root, used to resolve the relative paths in a filelist.  Taken
#: from this file's location so it holds regardless of the working directory
#: the trainer was launched with.
APPLICATION_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_filepaths_and_text(filename, split="|", path_columns=(0, 1, 2, 3), root=None):
    """
    Load filepaths and associated text from a file.

    Paths are stored relative to the application root (see
    ``rvc/train/extract/preparing_files.py``) and are resolved back to absolute
    here.  ``os.path.join`` returns an absolute second argument unchanged, so
    filelists written by older versions -- which stored absolute paths -- keep
    loading without a migration step.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
        path_columns (tuple, optional): Which columns hold paths. The trailing
            speaker id must not be resolved, so the columns are explicit.
        root (str, optional): Base for relative paths. Defaults to the
            application root.
    """
    base = root or APPLICATION_ROOT
    rows = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(split)
            for column in path_columns:
                if column < len(fields):
                    fields[column] = os.path.normpath(
                        os.path.join(base, fields[column])
                    )
            rows.append(fields)
    return rows


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
        print_error(
            f"No model config at {config_save_path}. Run preprocessing and "
            "feature extraction first.",
            tag="[INIT]",
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
        warning(
            "Training interrupted; skipping the flush to avoid partial logs.",
            tag="[TRAIN]",
        )
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


def wave_to_mel(config, waveform, num_mels=None, for_loss=False):
    mel_spec = mel_spectrogram_torch(
        waveform.float().squeeze(1),
        config.data.filter_length,
        num_mels if num_mels is not None else config.data.n_mel_channels,
        config.data.sample_rate,
        config.data.hop_length,
        config.data.win_length,
        config.data.mel_fmin,
        config.data.mel_fmax,
        log_compression=not for_loss,
    )
    # Mel/STFT values and their gradients remain in FP32.  Converting this
    # tensor to FP16 after the transform makes the spectral loss a low
    # precision loss even when its autocast region is disabled.
    mel_spec = mel_spec.float()
    if for_loss:
        mel_spec = torch.log1p(mel_spec * 1000.0)
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

    info("Cleanup done.", tag="[INIT]")


def print_init_setup(
    warmup_duration,
    rank,
    use_warmup,
    config,
    optimizer_choice_g,
    optimizer_choice_d,
    lr_scheduler,
    exp_decay_gamma,
    spectral_loss,
):
    if rank != 0:
        return

    tf32_enabled = (
        torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.allow_tf32
    )
    # FP32 master weights throughout; TF32 only changes how cuDNN/cuBLAS run
    # the convolutions internally (11-bit mantissa, no autocast, no scaler).
    precision = "FP32 (TF32 matmul/conv)" if tf32_enabled else "FP32"

    rows = [
        ("PRECISION", precision),
        ("cudnn.benchmark", torch.backends.cudnn.benchmark),
        ("cudnn.deterministic", torch.backends.cudnn.deterministic),
        (
            "Optimizer (G/D)",
            optimizer_choice_g
            if optimizer_choice_g == optimizer_choice_d
            else f"G: {optimizer_choice_g} | D: {optimizer_choice_d}",
        ),
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

    print_settings_panel(rows)

def train_loader_safety(train_loader):
    if len(train_loader) < 3:
        print_error(
            "Not enough data in the training set. Did the preprocessing step "
            "slice the audio files?",
            tag="[INIT]",
        )
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
        warning(f"Could not read the model info file ({e}); using defaults.", tag="[INIT]")

    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        warning(
            f"Could not read the checkpoint ({e}); using the default speaker count.",
            tag="[INIT]",
        )

    if rank == 0:
        info(f"Initializing the generator with {spk_dim} speakers.", tag="[INIT]")

    return spk_dim
