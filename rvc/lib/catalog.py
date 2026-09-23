"""What is on disk: the models, indexes, audio and datasets the interfaces list.

Filesystem only, no torch, so both the Gradio tabs and the Qt interface can
call it to fill a dropdown.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rvc.lib.model_bundle import is_model_bundle, is_model_file, walk_models
from rvc.lib.paths import (
    AUDIO_DIR,
    CUSTOM_PRETRAINED_DIR,
    DATASET_DIR,
    LOGS_DIR,
    ROOT,
)

AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4",
    ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3",
)

#: Suffix a converted file gets, and what tells outputs apart from inputs.
OUTPUT_SUFFIX = "_output"

_EPOCH_RE = re.compile(r"_(\d+)e_(\d+)s", re.IGNORECASE)
_TRAINING_DIR_PREFIXES = (".", "mute", "reference")


def is_audio_file(name: str | os.PathLike[str]) -> bool:
    return str(name).lower().endswith(AUDIO_EXTENSIONS)


def relative(path: str | os.PathLike[str]) -> str:
    """``path`` relative to the application root when it is inside it.

    ``abspath`` rather than ``resolve``, so a symlinked ``logs/`` still lists
    as ``logs/...``.
    """
    try:
        return str(Path(os.path.abspath(path)).relative_to(ROOT))
    except ValueError:
        return str(path)


def sort_key(path: str) -> tuple:
    """Order checkpoints by ``_<epoch>e_<step>s``, so ``_100e`` follows ``_20e``."""
    name = Path(path).name
    folder = Path(path).parent.name.lower()
    match = _EPOCH_RE.search(name)
    if match:
        return (folder, int(match.group(1)), int(match.group(2)), name.lower())
    return (folder, -1, -1, name.lower())


def list_models(
    logs_dir: Path = LOGS_DIR,
    *,
    bundles: bool = True,
    training_checkpoints: bool = False,
) -> list[str]:
    """Voice models under ``logs_dir``, ordered by training progress.

    ``training_checkpoints`` adds the trainer's own ``G_*``/``D_*`` files.
    """
    found = []
    for dirpath, _dirnames, filenames in walk_models(logs_dir):
        for name in filenames:
            if not is_model_file(name):
                continue
            if not bundles and is_model_bundle(name):
                continue
            if not training_checkpoints and name.startswith(("G_", "D_")):
                continue
            found.append(relative(os.path.join(dirpath, name)))
    return sorted(found, key=sort_key)


def list_bundles(logs_dir: Path = LOGS_DIR) -> list[str]:
    return [path for path in list_models(logs_dir) if is_model_bundle(path)]


def list_indexes(logs_dir: Path = LOGS_DIR) -> list[str]:
    """Retrieval indexes under ``logs_dir``, without the ``trained_*`` intermediates."""
    found = []
    for dirpath, _dirnames, filenames in walk_models(logs_dir):
        for name in filenames:
            if name.endswith(".index") and "trained" not in name:
                found.append(relative(os.path.join(dirpath, name)))
    return sorted(found)


def guess_index_for(model_path: str) -> str:
    """The index that belongs to a checkpoint, or ``""``.

    The only index in the model's folder, or else the one whose name contains
    the model's name up to the first underscore.
    """
    if not model_path or is_model_bundle(model_path):
        return ""
    folder = (ROOT / model_path).parent
    try:
        candidates = sorted(
            p for p in folder.glob("*.index") if "trained" not in p.name
        )
    except OSError:
        return ""
    if len(candidates) == 1:
        return relative(candidates[0])
    stem = Path(model_path).stem.split("_")[0].lower()
    for candidate in candidates:
        if stem and stem in candidate.name.lower():
            return relative(candidate)
    return ""


def list_audios(audio_dir: Path = AUDIO_DIR) -> list[str]:
    """Inputs sitting directly in ``audio_dir``, conversion results left out."""
    if not audio_dir.is_dir():
        return []
    return sorted(
        relative(p)
        for p in audio_dir.iterdir()
        if p.is_file() and is_audio_file(p.name) and OUTPUT_SUFFIX not in p.stem
    )


def list_outputs(audio_dir: Path = AUDIO_DIR) -> list[Path]:
    """Conversion results in ``audio_dir`` and below."""
    if not audio_dir.is_dir():
        return []
    return [
        p for p in audio_dir.rglob("*")
        if p.is_file() and is_audio_file(p.name) and OUTPUT_SUFFIX in p.name
    ]


def default_output_path(input_path: str, audio_dir: Path = AUDIO_DIR) -> str:
    """``<audio_dir>/<input stem>_output.wav``."""
    stem = Path(input_path).name.rsplit(".", 1)[0]
    return str(audio_dir / f"{stem}{OUTPUT_SUFFIX}.wav")


def list_training_models(logs_dir: Path = LOGS_DIR) -> list[str]:
    """Experiment folders under ``logs_dir``."""
    if not logs_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in logs_dir.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(_TRAINING_DIR_PREFIXES)
        and entry.name != "zips"
    )


def list_experiment_speakers(model_name: str, logs_dir: Path = LOGS_DIR) -> list[int]:
    """Speaker ids with extracted features in ``logs/<model_name>/``.

    Read from the ``<sid>_<file>_<slice>`` feature filenames, so faiss and
    scikit-learn stay out of a dropdown refresh.  Mirrors
    ``rvc.train.process.extract_index.available_speakers``.
    """
    if not model_name:
        return []
    feature_dir = logs_dir / model_name / "extracted"
    if not feature_dir.is_dir():
        return []
    found = set()
    for entry in feature_dir.iterdir():
        if entry.suffix != ".npy":
            continue
        try:
            found.add(int(entry.name.split("_", 1)[0]))
        except ValueError:
            continue
    return sorted(found)


def list_custom_pretraineds(kind: str, root: Path = CUSTOM_PRETRAINED_DIR) -> list[str]:
    """User-supplied pretrained weights, ``kind`` being ``"G"`` or ``"D"``.

    ``kind`` has to stand as its own letter in the filename, which takes
    ``G_48k.pth`` and ``f0G40k.pth`` but not ``D_Giga.pth``.
    """
    if not root.is_dir():
        return []
    pattern = re.compile(rf"(?<![a-z]){re.escape(kind)}(?![a-z])", re.IGNORECASE)
    return sorted(
        relative(path)
        for path in root.rglob("*.pth")
        if path.is_file() and pattern.search(path.stem)
    )


def list_dataset_folders(root: Path = DATASET_DIR) -> list[str]:
    """Folders directly under ``root`` with audio anywhere inside them.

    Subfolders are speakers of that dataset to the preprocessor, so they are
    not offered as datasets of their own.
    """
    if not root.is_dir():
        return []
    return sorted(
        relative(candidate)
        for candidate in root.iterdir()
        if candidate.is_dir()
        and any(
            entry.is_file() and is_audio_file(entry.name)
            for entry in candidate.rglob("*")
        )
    )
