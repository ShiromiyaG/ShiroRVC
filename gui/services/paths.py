"""Filesystem layout of the host application.

Every path is derived from this file's own location rather than from the
process working directory, which file dialogs and the odd third-party library
are free to change out from under us.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_DIR.parent

CORE_SCRIPT = ROOT / "core.py"
LOGS_DIR = ROOT / "logs"
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
AUDIO_DIR = ASSETS_DIR / "audios"
DATASET_DIR = ASSETS_DIR / "datasets"
PRESET_DIR = ASSETS_DIR / "training_presets"
FORMANT_DIR = ASSETS_DIR / "formant_shift"
CUSTOM_EMBEDDER_DIR = ROOT / "rvc" / "models" / "embedders" / "embedders_custom"
#: Where the Gradio tab drops user-supplied pretrained weights, and therefore
#: where the native interface has to look for them too -- the two must offer
#: the same files or "the one I downloaded" is missing from one of them.
CUSTOM_PRETRAINED_DIR = ROOT / "rvc" / "models" / "pretraineds" / "custom"
RESOURCE_DIR = PACKAGE_DIR / "resources"

#: Where the GUI keeps its own state.  Deliberately outside ``assets/`` so that
#: wiping the GUI's preferences never touches user audio.
STATE_DIR = PACKAGE_DIR / ".state"


def python_executable() -> str:
    """Interpreter to spawn the backend worker with.

    ``pythonw.exe`` is what actually launches the GUI on Windows, and it has no
    usable stdout -- spawning the worker with it would silently discard every
    log line.  Swap back to the console interpreter for children.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(exe)


def ensure_dirs() -> None:
    """Create the directories the GUI writes into.  Safe to call repeatedly."""
    for path in (LOGS_DIR, AUDIO_DIR, DATASET_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def output_dir() -> Path:
    """Default destination for converted audio."""
    return AUDIO_DIR


def relative(path: str | os.PathLike[str]) -> str:
    """Path relative to the application root when it is inside it.

    The backend resolves several arguments against its own working directory,
    so handing it a repo-relative path is both shorter to display and less
    likely to be misread than an absolute one.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


def is_backend_available() -> bool:
    """Whether the host application is present next to this package."""
    return CORE_SCRIPT.is_file()
