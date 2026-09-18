"""Inference presets, in the same folder and format as the Gradio tab's.

The format itself -- which keys, which values are valid -- is
``rvc.lib.inference_presets``, the module the Gradio tab uses too, so the two
interfaces cannot drift into writing files the other half-reads.  It is plain
JSON, so it runs here on the UI thread with no trip through the worker.

Imported on first use, like the backend helpers in ``catalog``: a window that
cannot find the application should still open and say so.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import paths


def _lib():
    from rvc.lib import inference_presets

    return inference_presets


def list_presets() -> list[str]:
    return _lib().list_presets(paths.INFERENCE_PRESET_DIR)


def is_valid_name(name: str) -> bool:
    return _lib().is_valid_name(name)


def exists(name: str) -> bool:
    return (paths.INFERENCE_PRESET_DIR / f"{name.strip()}.json").is_file()


def load(name: str) -> dict:
    """The preset's valid settings.  Raises ``OSError`` or ``ValueError``."""
    return _lib().read_preset(paths.INFERENCE_PRESET_DIR, name)


def save(name: str, values: dict) -> str:
    """Write ``values`` as the preset ``name``; returns the file written."""
    return _lib().write_preset(paths.INFERENCE_PRESET_DIR, name, values)


def import_file(source: str) -> str:
    """Copy a preset file into the presets folder; returns its preset name.

    Parsed first, so a file that is not a preset is refused here rather than
    landing in the folder and failing each time it is picked.
    """
    _lib().parse_preset(source)
    name = Path(source).stem
    if not is_valid_name(name):
        raise ValueError(f"{name!r} is not a usable preset name")
    paths.INFERENCE_PRESET_DIR.mkdir(parents=True, exist_ok=True)
    target = paths.INFERENCE_PRESET_DIR / f"{name}.json"
    if not (target.exists() and os.path.samefile(source, target)):
        shutil.copyfile(source, target)
    return name
