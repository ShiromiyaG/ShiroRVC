"""Native Qt front-end for ShiroRVC.

This package is strictly additive: nothing outside ``gui/`` imports it, so the
directory can be deleted without breaking the Gradio interface or the CLI.
The only code allowed to reach into the host application lives in
``gui.services`` -- see ``gui/README.md`` for the rule and why it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

APP_NAME = "ShiroRVC"

#: Read from ``assets/config.json`` rather than imported from ``version.py``, so
#: that this package needs nothing on ``sys.path`` to know what it is -- and so
#: the number still has exactly one definition.  ``tests/test_version.py`` is
#: what keeps this reader and ``version.read_version`` from drifting apart.
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "assets" / "config.json"

try:
    _payload = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    __version__ = str(_payload.get("version", "")).strip() or "0.0.0"
except (OSError, ValueError, AttributeError):
    __version__ = "0.0.0"
