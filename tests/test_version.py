"""The version must have exactly one definition.

Three things read it -- the Gradio interface, the Qt interface and the release
workflow -- and the failure they would otherwise produce is silent: a release
tagged v1.2.0 whose window says v1.1.0.  Cheaper to assert than to notice.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"


def declared_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def test_version_file_exists_and_is_well_formed():
    assert VERSION_FILE.is_file(), "VERSION is the source of truth and must exist"
    version = declared_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?", version), (
        f"'{version}' is not a version the release workflow will accept"
    )


def test_version_module_reports_the_file():
    result = subprocess.run(
        [sys.executable, "version.py"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == declared_version()


def test_gui_reports_the_same_version():
    """The Qt package reads VERSION directly, without importing version.py."""
    result = subprocess.run(
        [sys.executable, "-c", "import gui; print(gui.__version__)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == declared_version()


def test_no_hardcoded_version_literals():
    """Nobody should be re-declaring the number in Python."""
    version = declared_version()
    pattern = re.compile(
        r"""(APP_VERSION|__version__|VERSION)\s*=\s*["']""" + re.escape(version)
    )
    offenders = []
    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT).parts
        if {"env", ".venv", "build", "dist", "__pycache__", "tests"} & set(parts):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert not offenders, (
        f"These files hardcode the version instead of reading VERSION: {offenders}"
    )
