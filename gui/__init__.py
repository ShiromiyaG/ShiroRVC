"""Native Qt front-end for ShiroRVC.

This package is strictly additive: nothing outside ``gui/`` imports it, so the
directory can be deleted without breaking the Gradio interface or the CLI.
The only code allowed to reach into the host application lives in
``gui.services`` -- see ``gui/README.md`` for the rule and why it exists.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ShiroRVC"

#: Read from the application's VERSION file rather than imported from
#: ``version.py``, so that this package needs nothing on ``sys.path`` to know
#: what it is -- and so the number still has exactly one definition.
_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"

try:
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip() or "0.0.0"
except OSError:
    __version__ = "0.0.0"
