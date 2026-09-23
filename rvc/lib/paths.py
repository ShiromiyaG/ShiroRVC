"""Filesystem layout of the application.

Derived from this file's location rather than the working directory, so the
paths hold however the process was launched.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOGS_DIR = ROOT / "logs"
ASSETS_DIR = ROOT / "assets"
CONFIG_PATH = ASSETS_DIR / "config.json"
AUDIO_DIR = ASSETS_DIR / "audios"
DATASET_DIR = ASSETS_DIR / "datasets"
TRAINING_PRESET_DIR = ASSETS_DIR / "training_presets"
INFERENCE_PRESET_DIR = ASSETS_DIR / "inference_presets"
FORMANT_DIR = ASSETS_DIR / "formant_shift"
INFER_PID_PATH = ASSETS_DIR / "infer_pid.txt"
MODELS_DIR = ROOT / "rvc" / "models"
CUSTOM_PRETRAINED_DIR = MODELS_DIR / "pretraineds" / "custom"


def relative(path: str | os.PathLike[str]) -> str:
    """``path`` relative to :data:`ROOT` when it is inside it, else unchanged."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)
