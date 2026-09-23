"""Everything the GUI needs to populate a dropdown.

All of it is filesystem or JSON work -- deliberately no torch -- so the views
can refresh a list on the UI thread without stalling.  Anything that needs a
checkpoint actually loaded goes through :mod:`gui.services.engine` instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import paths
from ..i18n import _

# Kept in sync with core.py's click choices.  These are small closed sets that
# the backend validates anyway, so mirroring them costs nothing and keeps the
# GUI's first paint free of backend imports.
F0_METHODS = ["rmvpe", "crepe", "crepe-tiny", "fcpe"]
EMBEDDER_MODELS = ["contentvec", "spin_v2"]
TRAINING_EMBEDDER_MODELS = ["contentvec", "spin_v2"]
EXPORT_FORMATS = ["WAV", "MP3", "FLAC", "OGG", "M4A"]
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
CUT_PREPROCESS = ["Skip", "Simple", "Automatic", "New Automatic"]
#: ``post_rms``, ``post_peak_rvc`` and ``post_loudness`` are gone from the
#: list but not from the code: an experiment whose config names one re-runs
#: unchanged.  All three level every *slice* independently, which flattens the
#: dynamics between phrases; ``post_rms`` additionally has a -40 dBFS gate that
#: is scale-dependent.  See ``rvc/train/preprocess/loudness.py``.
#:
#: The ``pre_`` prefix is about the *scope of the gain*, not about running
#: earlier: one factor per source recording, so recordings match each other and
#: the dynamics inside one survive.  Both listed modes run after slicing like
#: everything else.
#:
#: ``pre_peak_rvc`` is the default: stock RVC's peak blend, at recording scope.
#: It anchors the *peak*, so the loudness it lands on is whatever the crest
#: factor of the material allows -- two datasets normalised this way match each
#: other in headroom, not in perceived volume.
#:
#: ``pre_loudness`` anchors BS.1770 loudness instead, which is what perceived
#: volume follows and what makes recordings match each other by ear.  Prefer it
#: when levels have to be consistent across a many-speaker set; the peak then
#: lands wherever the crest factor puts it, which for compressed material is
#: well below the ceiling and is not a fault.
NORMALIZATION_MODES = ["none", "post_peak", "pre_peak_rvc", "pre_loudness"]
LOADING_RESAMPLING = ["ffmpeg", "librosa"]
DATASET_FORMATS = ["WAV", "FLAC", "MP3", "OGG", "M4A"]

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

TTS_RAW_NAME = "tts_output.wav"
TTS_CONVERTED_NAME = "tts_rvc_output.wav"


def _shared():
    """``rvc.lib.catalog``, imported on first use: a window that cannot find the
    application should still open and say so."""
    from rvc.lib import catalog

    return catalog


def __getattr__(name: str):
    if name in ("AUDIO_EXTENSIONS", "OUTPUT_SUFFIX"):
        return getattr(_shared(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_output_path(input_path: str) -> str:
    return _shared().default_output_path(input_path, paths.AUDIO_DIR)


def conversion_outputs(output_path: str, export_format: str) -> list[str]:
    """Every file a conversion to ``output_path`` writes.

    The path itself, which the pipeline always writes as WAV, and the copy it
    makes for any other export format -- named by the same ``.wav`` substitution
    ``core.run_infer_script`` and ``core.run_tts_script`` use.
    """
    converted = output_path.replace(".wav", f".{export_format.lower()}")
    return [output_path] if converted == output_path else [output_path, converted]


def list_models() -> list[str]:
    return _shared().list_models(paths.LOGS_DIR)


def list_bundles() -> list[str]:
    return _shared().list_bundles(paths.LOGS_DIR)


def list_indexes() -> list[str]:
    return _shared().list_indexes(paths.LOGS_DIR)


def guess_index_for(model_path: str) -> str:
    return _shared().guess_index_for(model_path)


def list_audios() -> list[str]:
    return _shared().list_audios(paths.AUDIO_DIR)


def list_training_models() -> list[str]:
    return _shared().list_training_models(paths.LOGS_DIR)


def list_experiment_speakers(model_name: str) -> list[int]:
    return _shared().list_experiment_speakers(model_name, paths.LOGS_DIR)


def list_custom_pretraineds(kind: str) -> list[str]:
    return _shared().list_custom_pretraineds(kind, paths.CUSTOM_PRETRAINED_DIR)


def list_dataset_folders() -> list[str]:
    return _shared().list_dataset_folders(paths.DATASET_DIR)


def vocoders() -> list[tuple[str, str]]:
    """``(label, id)`` pairs from the backend's vocoder registry, minus disabled ones."""
    registry = _vocoder_registry()
    return [
        (spec.get("label", key), key)
        for key, spec in registry.items()
        if spec.get("enabled", True)
    ]


def description_for(vocoder: str) -> str:
    """The registry's ``description`` for a vocoder, or ``""``."""
    return str(_vocoder_registry().get(vocoder, {}).get("description", ""))


def sample_rates_for(vocoder: str) -> list[int]:
    """Sample rates a vocoder actually ships a config for."""
    spec = _vocoder_registry().get(vocoder, {})
    return [int(rate) for rate in spec.get("sample_rates", [])]


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
    path = paths.ROOT / "rvc" / "lib" / "extras" / "tts_voices.json"
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
        return _("active now")
    if delta < 3600:
        return _("{count} min ago").format(count=int(delta // 60))
    if delta < 86400:
        return _("{count} h ago").format(count=int(delta // 3600))
    return _("{count} d ago").format(count=int(delta // 86400))
