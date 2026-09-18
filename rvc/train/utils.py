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

from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

debug_save_load = False

# Moved into their own modules; re-exported so ``from utils import ...`` in
# train.py and ``rvc.train.utils`` elsewhere keep working.
from rvc.train.checkpoints import (  # noqa: F401
    replace_keys_in_dict,
    optimizer_param_names,
    remap_optimizer_state,
    load_checkpoint,
    excitation_source,
    assert_excitation_matches,
    LEGACY_UPSAMPLE_RATES,
    LEGACY_UPSAMPLE_FILTER,
    upsample_filter,
    decoder_layout,
    assert_decoder_layout_matches,
    discriminator_periods,
    discriminator_has_msd,
    assert_msd_matches,
    assert_periods_match,
    save_checkpoint,
)
from rvc.train.previews import (  # noqa: F401
    VALIDATION_PREVIEW_DPI,
    VALIDATION_PREVIEW_FIGSIZE,
    VALIDATION_PREVIEW_FREQUENCY_TICKS,
    limit_audio_peak,
    _mel_to_numpy,
    _mel_axis_positions,
    _format_frequency,
    plot_validation_preview_to_figure,
    _write_atomically,
    log_validation_preview,
    log_tensorboard_media,
)


from itertools import chain
from mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.lib.terminal import (
    error as print_error,
    info,
    print_settings_panel,
    warning,
)

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
        except Exception:
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


#: Schedulers whose decay is defined by an endpoint when ``lr_final_ratio`` is
#: set, which makes ``lr_decay``/``exp_decay_gamma`` dead config for them.  Kept
#: next to the panel because the panel is the only place that has to know.
_ENDPOINT_SCHEDULERS = frozenset(
    {"exp decay epoch", "exp decay step", "cosine annealing", "cosine annealing epoch"}
)


def print_init_setup(
    warmup_duration,
    rank,
    use_warmup,
    config,
    optimizer_choice_g,
    optimizer_choice_d,
    lr_scheduler_g,
    lr_scheduler_d,
    exp_decay_gamma,
    spectral_loss,
    lr_final_ratio=None,
    amp_dtype=None,
):
    if rank != 0:
        return

    tf32_enabled = (
        torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.allow_tf32
    )
    # TF32 is orthogonal to autocast: it only changes how cuDNN/cuBLAS run the
    # convolutions internally (11-bit mantissa), and it applies to the FP32 ops
    # that remain under autocast just as much as it does without it.  So it is
    # reported as a qualifier on either line rather than as a precision of its
    # own.
    #
    # ``amp_dtype`` has to be passed in rather than inferred here.  This row read
    # the TF32 flags alone until 2026-08-26, so it printed "FP32 (TF32
    # matmul/conv)" on every run -- including FP16 ones, where autocast and the
    # GradScaler were both live.  The banner prints well before the "AMP enabled"
    # line, so it was the only precision statement most runs ever saw.
    #
    # Master weights stay FP32 in every case, so the row names only what
    # changes: the autocast dtype, whether a GradScaler is live, and TF32.  The
    # panel's value column is narrow enough that a longer string is silently
    # truncated, which is how this row came to be misleading in the first place.
    if amp_dtype is None:
        precision = "FP32 (TF32 matmul/conv)" if tf32_enabled else "FP32"
    else:
        name = {torch.float16: "FP16", torch.bfloat16: "BF16"}.get(
            amp_dtype, str(amp_dtype).rsplit(".", 1)[-1].upper()
        )
        # BF16 carries FP32's exponent range, so it needs no loss scaling.
        scaler = " + GradScaler" if amp_dtype == torch.float16 else ""
        tf32_note = " (TF32 conv)" if tf32_enabled else ""
        precision = f"{name} autocast{scaler}{tf32_note}"

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
        # ``lr_final_ratio`` decides the decay by its endpoint and supersedes
        # the per-step/per-epoch gamma entirely, so printing the gamma there
        # advertises a number that has no effect -- and invites tuning it.  Show
        # the endpoint that is actually in force instead; the gamma comes back
        # the moment there is no ratio to override it.
        if lr_final_ratio is not None and name in _ENDPOINT_SCHEDULERS:
            shown = "cosine annealing" if name.startswith("cosine annealing") else name
            return f"{shown}, to {lr_final_ratio:g}x over the run"
        if name in ("cosine annealing", "cosine annealing epoch"):
            return "cosine annealing"
        return f"{name}, gamma: {gamma}"

    shown_g = scheduler_value(lr_scheduler_g, exp_decay_gamma)
    shown_d = scheduler_value(lr_scheduler_d, exp_decay_gamma)
    rows.extend(
        [
            (
                "LR scheduler (G/D)",
                shown_g if shown_g == shown_d else f"G: {shown_g} | D: {shown_d}",
            ),
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


def substitute_speaker_embeddings(state_dict, net_g):
    """Swap a pretrain's ``emb_g.weight`` for the model's own fresh rows.

    Substituting rather than dropping the key on purpose: the VITS-latent
    vocoders load their pretrained generator *strictly*, where a missing key is
    an error, and a load that silently skips a tensor is exactly what this path
    must not produce.  Returns a shallow copy -- the caller's checkpoint dict is
    left alone, because it may still be read for other keys.
    """

    if "emb_g.weight" not in state_dict:
        return state_dict
    target = net_g.module if hasattr(net_g, "module") else net_g
    updated = dict(state_dict)
    updated["emb_g.weight"] = target.emb_g.weight.detach().cpu().clone()
    return updated


class SpeakerLayout(NamedTuple):
    """How many rows ``emb_g`` gets, and whether a pretrain's rows are usable.

    ``embed_dim`` is what the synthesizer is built with.  ``reset_pretrained``
    says the pretrained checkpoint's speaker table describes *other* speakers
    and must not be inherited -- see :func:`verify_spk_dim`.
    """

    embed_dim: int
    reset_pretrained: bool


def verify_spk_dim(
    config,
    model_info_path,
    experiment_dir,
    latest_checkpoint_path,
    rank,
    pretrainG
):
    """Resolve ``spk_embed_dim`` and decide whether to keep a pretrain's ``emb_g``.

    Three sources, most specific last: the config's default, the dataset's own
    count from ``model_info.json``, and a checkpoint's row count.

    The last one used to win unconditionally, which is what made a multispeaker
    fine-tune start with *the pretrain's* speaker table.  That is not a harmless
    initialisation.  Measured on ``f0G40k``: ``dec.cond`` reads the directions
    the trained embedding occupies with 1.53x the gain it gives random ones, so
    at step 0 the decoder renders each of your speakers as a specific, confident
    pretrain identity, and the run has to move away from a wrong answer rather
    than toward a right one.  On a small dataset it does not move far enough and
    the pretrain's timbre stays audible -- the leak people report.  Dropping the
    table costs nothing in separation: after ``dec.cond`` four freshly
    initialised speakers sit 62.3 apart against 41.0 for four inherited ones.

    The rule is the shape, because the shape is what actually carries the
    question.  Rows that disagree with the dataset's speaker count cannot be
    this dataset's speakers, so they go.  Rows that agree almost certainly are
    -- that is staged pretraining handing its own table forward -- so they stay.
    Single-speaker runs keep the old behaviour outright: there is no
    cross-speaker leak to fix, and row 0 is a working starting timbre.

    A resume is never touched: ``G_*.pth`` in the experiment folder *is* this
    run's table and its width is not negotiable.
    """

    embedder_name = "contentvec"  # Default embedder
    spk_dim = config.model.spk_embed_dim  # 109 default speakers
    dataset_speakers = None

    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            dataset_speakers = int(model_info["speakers_id"])
            spk_dim = dataset_speakers
    except Exception as e:
        if rank == 0:
            warning(
                f"Could not read the model info file ({e}); using defaults.",
                tag="[INIT]",
            )

    reset_pretrained = False
    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            checkpoint_speakers = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
            if (
                last_g is None
                and dataset_speakers is not None
                and dataset_speakers > 1
                and checkpoint_speakers != dataset_speakers
            ):
                spk_dim = dataset_speakers
                reset_pretrained = True
            else:
                spk_dim = checkpoint_speakers
    except Exception as e:
        # Rank-gated like every other message here: without it each DDP rank
        # prints the same warning, which reads as several different failures.
        if rank == 0:
            warning(
                f"Could not read the checkpoint ({e}); using the default speaker count.",
                tag="[INIT]",
            )

    if rank == 0:
        info(f"Initializing the generator with {spk_dim} speakers.", tag="[INIT]")
        if reset_pretrained:
            info(
                "The pretrained speaker table describes a different set of "
                "speakers and will not be inherited; this run's embeddings "
                "start fresh.",
                tag="[INIT]",
            )

    return SpeakerLayout(spk_dim, reset_pretrained)
