"""Native Qt front-end for ShiroRVC.

Strictly additive: nothing outside ``gui/`` imports it, so the directory can
be deleted without breaking the Gradio interface or the CLI. Only
``gui.services`` may reach into the host application -- see ``gui/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

APP_NAME = "ShiroRVC"

#: Read from ``assets/config.json`` rather than imported from ``version.py`` so
#: this package needs nothing on ``sys.path``. ``tests/test_version.py`` keeps
#: this reader and ``version.read_version`` from drifting apart.
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "assets" / "config.json"

try:
    _payload = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    __version__ = str(_payload.get("version", "")).strip() or "0.0.0"
except (OSError, ValueError, AttributeError):
    __version__ = "0.0.0"
