"""Constants and build-time release metadata for the bootstrapper.

The workflow writes ``release_info.json`` next to this file before PyInstaller
runs, which is how a setup executable knows which tag it belongs to.  Without
it -- a local build, say -- the installer falls back to the repository's latest
release, so the file is an optimisation rather than a requirement.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "ShiroRVC"
DEFAULT_REPO = "ShiromiyaG/ShiroRVC"

#: python-build-standalone release that uv will fetch.  Matches the version the
#: repository's own conda installer pins.
PYTHON_VERSION = "3.12"

#: uv is the whole reason this install is minutes rather than tens of minutes:
#: it resolves and installs the torch stack in parallel with a warm global
#: cache.  ``latest`` avoids pinning a version that goes stale between releases.
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

TORCH_INDEXES = {
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

#: Used only when requirements.txt cannot be parsed for them.
FALLBACK_TORCH_PINS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "torchaudio": "2.11.0",
}

#: Roughly what a CUDA install occupies once the wheels are unpacked, plus room
#: for the prerequisite models.  Checked before anything is downloaded.
REQUIRED_DISK_BYTES = 14 * 1024**3


def bundle_dir() -> Path:
    """Directory holding this module's data, frozen or not."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def release_info() -> dict:
    """Repository and tag this installer was built for."""
    path = bundle_dir() / "release_info.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"repo": os.environ.get("SHIRORVC_REPO", DEFAULT_REPO), "tag": ""}


def source_url() -> str:
    """Where to fetch the application source from."""
    info = release_info()
    explicit = info.get("source_url")
    if explicit:
        return explicit
    repo = info.get("repo", DEFAULT_REPO)
    tag = info.get("tag")
    if tag:
        return f"https://github.com/{repo}/releases/download/{tag}/{APP_NAME}-source-{tag}.zip"
    # No tag baked in: take whatever the default branch currently is.
    return f"https://github.com/{repo}/archive/refs/heads/main.zip"


def default_install_dir() -> Path:
    """Somewhere writable without elevation, and not inside OneDrive.

    ``%LOCALAPPDATA%`` is the right answer on Windows: Program Files needs
    admin rights for the model downloads that happen later, and a synced
    Documents folder would try to upload several gigabytes of wheels.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "Programs" / APP_NAME
    return Path.home() / f".local/share/{APP_NAME.lower()}"
