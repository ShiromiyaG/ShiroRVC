"""The version must have exactly one definition.

Three things read it -- the Gradio interface, the Qt interface and the release
workflow -- and the failure they would otherwise produce is silent: a release
tagged v1.2.0 whose window says v1.1.0.  Cheaper to assert than to notice.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "assets" / "config.json"


def declared_version() -> str:
    payload = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return str(payload["version"]).strip()


def test_config_carries_a_well_formed_version():
    assert CONFIG_FILE.is_file(), (
        "assets/config.json is the source of truth and must exist"
    )
    version = declared_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?", version), (
        f"'{version}' is not a version the release workflow will accept"
    )


def test_version_module_reports_the_config():
    result = subprocess.run(
        [sys.executable, "version.py"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == declared_version()


def test_gui_reports_the_same_version():
    """The Qt package reads the config directly, without importing version.py.

    Two readers of one file, which is the arrangement this test exists to
    police: they can only disagree by drifting, and drifting is silent.
    """
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
        "These files hardcode the version instead of reading "
        f"assets/config.json: {offenders}"
    )


def test_a_config_without_a_version_falls_back_rather_than_raising(tmp_path,
                                                                  monkeypatch):
    """The config is a settings file four tabs rewrite at runtime.

    A user who corrupts it should get an application that starts and says
    0.0.0, not an ImportError from a module half the tree imports for
    ``APP_NAME``.  The release workflow is what refuses to publish 0.0.0.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import version as version_module
    finally:
        sys.path.pop(0)

    broken = tmp_path / "config.json"
    for payload in ("{ not json", "{}", '{"version": null}', '{"version": "  "}'):
        broken.write_text(payload, encoding="utf-8")
        monkeypatch.setattr(version_module, "CONFIG_FILE", broken)
        assert version_module.read_version() == version_module.FALLBACK_VERSION

    monkeypatch.setattr(version_module, "CONFIG_FILE", tmp_path / "absent.json")
    assert version_module.read_version() == version_module.FALLBACK_VERSION
