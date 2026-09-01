"""Everything the GUI needs to populate a dropdown.

All of it is filesystem or JSON work -- deliberately no torch -- so the views
can refresh a list on the UI thread without stalling.  Anything that needs a
checkpoint actually loaded goes through :mod:`gui.services.engine` instead.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import paths

# Kept in sync with core.py's click choices.  These are small closed sets that
# the backend validates anyway, so mirroring them costs nothing and keeps the
# GUI's first paint free of backend imports.
F0_METHODS = ["rmvpe", "crepe", "crepe-tiny", "fcpe"]
EMBEDDER_MODELS = ["contentvec", "spin_v1", "spin_v2", "custom"]
EXPORT_FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"]
#: Mirrors rvc.train.optimizers.OPTIMIZER_CHOICES; first entry is the default.
OPTIMIZERS = ["AdamW", "Sched-Free AdamW", "Muon", "Lion"]
LR_SCHEDULERS = ["exp decay step", "exp decay epoch", "cosine annealing", "none"]
INDEX_ALGORITHMS = ["Auto", "Faiss", "KMeans"]
#: How the index ranks neighbours.  "l2" is the default and is what upstream RVC
#: writes; "cosine" compares direction only, which suits embeddings whose
#: magnitude tracks loudness rather than content.  First entry is the default the
#: combo lands on.
INDEX_METRICS = ["l2", "cosine"]
#: Storage dtype for the extracted embeddings.  Mirrors
#: rvc.train.extract.extract.FEATURE_PRECISIONS.
FEATURE_PRECISIONS = ["fp32", "fp16"]
#: Mirrors rvc.train.messages.TORCH_COMPILE_MODES.
TORCH_COMPILE_MODES = [
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",
]
CUT_PREPROCESS = ["Skip", "Simple", "Automatic"]
NORMALIZATION_MODES = ["none", "post_peak", "post_peak_rvc", "post_rms"]
LOADING_RESAMPLING = ["librosa", "ffmpeg"]
DATASET_FORMATS = ["WAV", "FLAC", "MP3", "OGG", "M4A"]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma")

#: Starting values, copied from the Gradio tabs so the two interfaces produce
#: the same output for an untouched form.  They are genuinely different per
#: context -- single-file inference, folder batch and TTS each ship their own
#: numbers upstream -- so they are kept apart here rather than averaged into
#: one "sensible" set that would match none of them.
#:
#: ``filter_radius`` is the odd one.  The single-file tab declares it as a
#: 0-1 float at 0.006 and then hides the control (``interactive=False,
#: visible=False``); batch and TTS expose it as an integer 0-7 at 3.  The GUI
#: mirrors that, hidden control included: making it editable here would let a
#: value through that no Gradio user can produce.
INFERENCE_DEFAULTS: dict[str, dict] = {
    "single": {
        "pitch": 0,
        "index_rate": 0.5,
        "index_k": 8,
        "index_power": 2.0,
        "index_continuity": 0.5,
        "volume_envelope": 1.0,
        "silence_gate_db": -60.0,
        "protect": 0.33,
        "filter_radius": 0.006,
        "filter_radius_range": (0.0, 1.0, 0.001, 3),
        "filter_radius_visible": False,
        "clean_audio": False,
        "clean_strength": 0.3,
        "split_audio": False,
        "f0_autotune": False,
        "f0_autotune_strength": 1.0,
        "formant_shifting": False,
        "seed": 0,
    },
    "batch": {
        "pitch": 0,
        "index_rate": 0.5,
        "index_k": 8,
        "index_power": 2.0,
        "index_continuity": 0.5,
        "volume_envelope": 1.0,
        "silence_gate_db": -60.0,
        "protect": 0.3,
        "filter_radius": 3,
        "filter_radius_range": (0, 7, 1, 0),
        "filter_radius_visible": True,
        "clean_audio": False,
        "clean_strength": 0.5,
        "split_audio": False,
        "f0_autotune": False,
        "f0_autotune_strength": 1.0,
        "formant_shifting": False,
        "seed": 0,
    },
    "tts": {
        "pitch": 0,
        "index_rate": 0.75,
        "index_k": 8,
        "index_power": 2.0,
        "index_continuity": 0.5,
        "volume_envelope": 1.0,
        "silence_gate_db": -60.0,
        "protect": 0.5,
        "filter_radius": 3,
        "filter_radius_range": (0, 7, 1, 0),
        "filter_radius_visible": True,
        "clean_audio": True,
        "clean_strength": 0.5,
        "split_audio": False,
        "f0_autotune": False,
        "f0_autotune_strength": 1.0,
        "formant_shifting": False,
        "seed": 0,
    },
}

#: Suffix the Gradio tabs give a converted file, from ``output_path_fn``.
OUTPUT_SUFFIX = "_output"
TTS_RAW_NAME = "tts_output.wav"
TTS_CONVERTED_NAME = "tts_rvc_output.wav"


def default_output_path(input_path: str) -> str:
    """Where a conversion lands, matching the Gradio tab exactly.

    ``<input stem>_output.wav`` in ``assets/audios``.  Same rule as
    ``tabs/inference/inference.py:output_path_fn`` -- a different name here
    would mean the two interfaces quietly write to different files, and the
    "clear _output files" button in the Gradio tab would miss these.
    """
    stem = Path(input_path).name.rsplit(".", 1)[0]
    return str(paths.AUDIO_DIR / f"{stem}{OUTPUT_SUFFIX}.wav")

_EPOCH_RE = re.compile(r"_e(\d+)_s(\d+)", re.IGNORECASE)


def _walk_models(root: Path):
    """``os.walk`` with the training-artifact directories pruned.

    Reuses the backend's own skip list rather than duplicating it: a single
    trained model leaves hundreds of thousands of files under ``extracted/``
    and ``f0/``, which is the difference between an instant refresh and a
    multi-second freeze.
    """
    try:
        from rvc.lib.model_bundle import walk_models

        yield from walk_models(root)
    except Exception:
        # The backend is missing or unimportable -- degrade to a plain walk so
        # the GUI still opens and can explain itself.
        skip = {
            "sliced_audios", "sliced_audios_16k", "extracted", "f0", "f0_voiced",
            "eval", "validation_samples", "zips", "__pycache__", ".torchinductor",
        }
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            yield dirpath, dirnames, filenames


def sort_key(path: str) -> tuple:
    """Order checkpoints by training progress, newest last.

    Filenames carry ``_e<epoch>_s<step>``; sorting on that instead of
    lexicographically is what stops ``_e100`` from landing before ``_e20``.
    """
    match = _EPOCH_RE.search(Path(path).name)
    if match:
        return (Path(path).parent.name.lower(), int(match.group(1)), int(match.group(2)))
    return (Path(path).parent.name.lower(), -1, -1)


def list_models() -> list[str]:
    """Voice checkpoints and model bundles under ``logs/``, repo-relative."""
    found = []
    for dirpath, _, filenames in _walk_models(paths.LOGS_DIR):
        for name in filenames:
            if name.lower().endswith((".pth", ".srvc")) and "G_" not in name and "D_" not in name:
                found.append(paths.relative(Path(dirpath) / name))
    return sorted(found, key=sort_key)


def list_indexes() -> list[str]:
    """Faiss indexes under ``logs/``, repo-relative."""
    found = []
    for dirpath, _, filenames in _walk_models(paths.LOGS_DIR):
        for name in filenames:
            if name.lower().endswith(".index") and "trained" not in name:
                found.append(paths.relative(Path(dirpath) / name))
    return sorted(found)


def guess_index_for(model_path: str) -> str:
    """The index sitting next to a checkpoint, if there is exactly one.

    Picking it automatically removes the single most common source of "why does
    my output sound nothing like the model" -- a forgotten index dropdown.
    """
    if not model_path:
        return ""
    folder = (paths.ROOT / model_path).parent
    candidates = [p for p in folder.glob("*.index") if "trained" not in p.name]
    return paths.relative(candidates[0]) if len(candidates) == 1 else ""


def list_audios() -> list[str]:
    """Audio files sitting directly in ``assets/audios``."""
    if not paths.AUDIO_DIR.is_dir():
        return []
    return sorted(
        paths.relative(p)
        for p in paths.AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def list_training_models() -> list[str]:
    """Model folders under ``logs/`` that look like a training run."""
    if not paths.LOGS_DIR.is_dir():
        return []
    names = []
    for entry in paths.LOGS_DIR.iterdir():
        if entry.is_dir() and not entry.name.startswith((".", "mute", "reference")):
            names.append(entry.name)
    return sorted(names)


def list_custom_embedders() -> list[str]:
    """Folders holding a user-supplied embedder."""
    root = paths.CUSTOM_EMBEDDER_DIR
    if not root.is_dir():
        return []
    return sorted(paths.relative(p) for p in root.iterdir() if p.is_dir())


def list_custom_pretraineds(kind: str) -> list[str]:
    """User-supplied pretrained weights, ``kind`` being ``"G"`` or ``"D"``.

    Matches what the Gradio tab offers: every ``.pth`` under the custom
    pretraineds folder whose filename starts with the generator or
    discriminator letter.  Sorted so the list does not reshuffle between
    refreshes.
    """
    root = paths.CUSTOM_PRETRAINED_DIR
    if not root.is_dir():
        return []
    wanted = kind.upper()
    found = [
        path for path in root.rglob("*.pth")
        if path.is_file() and path.name.upper().startswith(wanted)
    ]
    return sorted(paths.relative(path) for path in found)


def list_dataset_folders() -> list[str]:
    """Folders under ``assets/datasets`` that actually contain audio.

    Offering every directory would suggest the empty ones a fresh install
    creates, which is a suggestion that cannot lead anywhere.
    """
    root = paths.DATASET_DIR
    if not root.is_dir():
        return []
    found = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        if any(
            entry.suffix.lower() in AUDIO_EXTENSIONS
            for entry in candidate.rglob("*")
            if entry.is_file()
        ):
            found.append(paths.relative(candidate))
    return sorted(found)


def vocoders() -> list[tuple[str, str]]:
    """``(label, id)`` pairs from the backend's vocoder registry."""
    registry = _vocoder_registry()
    return [(spec.get("label", key), key) for key, spec in registry.items()]


def sample_rates_for(vocoder: str) -> list[int]:
    """Sample rates a vocoder actually ships a config for."""
    spec = _vocoder_registry().get(vocoder, {})
    return [int(rate) for rate in spec.get("sample_rates", [])]


def supports_smartcutter(vocoder: str) -> bool:
    return bool(_vocoder_registry().get(vocoder, {}).get("supports_smartcutter"))


def _vocoder_registry() -> dict:
    """Read ``rvc/configs/vocoders.json`` directly.

    Importing ``rvc.configs.vocoders`` would work too, but reading the JSON
    keeps this module free of backend imports -- the registry is the file, not
    the module wrapped around it.
    """
    path = paths.ROOT / "rvc" / "configs" / "vocoders.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def tts_voices() -> list[tuple[str, str]]:
    """``(display, short_name)`` for every Edge-TTS voice."""
    path = paths.ROOT / "rvc" / "lib" / "tools" / "tts_voices.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    voices = []
    for voice in data:
        short = voice.get("ShortName", "")
        gender = voice.get("Gender", "")
        locale = voice.get("Locale", "")
        voices.append((f"{short}  ({locale}, {gender})", short))
    return sorted(voices)


def latest_run_dir(model_name: str) -> Path | None:
    """The TensorBoard event directory for a training run, if it exists."""
    candidate = paths.LOGS_DIR / model_name / "eval"
    return candidate if candidate.is_dir() else None


def _has_events(directory: Path) -> bool:
    try:
        return any(
            entry.name.startswith("events.out.tfevents")
            for entry in directory.iterdir()
            if entry.is_file()
        )
    except OSError:
        return False


def list_runs() -> list[tuple[str, str, float]]:
    """Every run with TensorBoard events, newest first.

    Returns ``(display, path, mtime)``.  Runs are usually at
    ``logs/<model>/eval``, but a checkpoint copied in from elsewhere can leave
    them directly under ``logs/<model>``, so both shapes are accepted -- the
    monitor should be able to open whatever is actually on disk, not only what
    this application wrote.
    """
    if not paths.LOGS_DIR.is_dir():
        return []

    found: list[tuple[str, str, float]] = []
    for entry in paths.LOGS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        for directory, label in ((entry / "eval", entry.name), (entry, f"{entry.name} (root)")):
            if directory.is_dir() and _has_events(directory):
                try:
                    stamp = max(
                        item.stat().st_mtime
                        for item in directory.iterdir()
                        if item.name.startswith("events.out.tfevents")
                    )
                except (OSError, ValueError):
                    stamp = 0.0
                found.append((label, str(directory), stamp))
                break  # eval/ wins when both exist; it is what the trainer writes

    found.sort(key=lambda row: row[2], reverse=True)
    return found


def describe_age(stamp: float) -> str:
    """How long ago a run last wrote, for the picker's second line."""
    if not stamp:
        return ""
    delta = max(0.0, time.time() - stamp)
    if delta < 90:
        return "active now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    return f"{int(delta // 86400)} d ago"
