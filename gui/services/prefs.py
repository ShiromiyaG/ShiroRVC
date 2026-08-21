"""Persisted GUI state.

Kept in a plain JSON file under ``gui/.state`` rather than in the registry or
``~/.config`` so that a portable install stays portable: copy the folder, keep
your settings.  Deleting ``gui/`` takes its preferences with it, which is the
whole promise of this package.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import paths

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_FILE = paths.STATE_DIR / "preferences.json"

DEFAULTS: dict[str, Any] = {
    #: First run is dark. After that the stored value wins, and it is written
    #: the moment the user switches rather than at shutdown.
    "theme": "dark",
    "accent": "violet",
    "window_geometry": None,
    "console_visible": True,
    #: Whether the training monitor is folded away.  Worth remembering: on a
    #: laptop screen the answer is the same every session.
    "metrics_collapsed": False,
    "console_max_lines": 5000,
    "warmup_on_launch": True,
    #: Interface language, as a tag from ``rvc.lib.i18n.LANGUAGES``.  ``None``
    #: means "follow the operating system" and is *not* the same as ``"en"``:
    #: somebody who has never touched the switch keeps tracking their Windows
    #: display language, while somebody who deliberately picked English stays
    #: on English when that machine is later switched to Spanish.
    "language": None,
    #: Compositor backdrop behind the window: "none", "mica" or "acrylic".
    #: Off by default -- it puts the page background on the wallpaper, which is
    #: a taste, not an improvement, and not one to make on someone's behalf.
    "backdrop": "none",
}


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        data = dict(DEFAULTS)
        try:
            with open(_FILE, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                data.update(stored)
        except (OSError, ValueError):
            pass  # first run, or a file we cannot parse: defaults are correct
        _CACHE = data
    return _CACHE


def get(key: str, fallback: Any = None) -> Any:
    return _load().get(key, DEFAULTS.get(key, fallback))


def set(key: str, value: Any) -> None:  # noqa: A001 - reads better than set_value
    _load()[key] = value


def save() -> None:
    """Write the current state.  Called on close; failures are not fatal."""
    with _LOCK:
        data = _load()
        try:
            paths.STATE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = _FILE.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            tmp.replace(_FILE)
        except OSError:
            pass


def remember(scope: str, values: dict[str, Any]) -> None:
    """Store a view's field values under its own namespace."""
    _load().setdefault("views", {})[scope] = values


def recall(scope: str) -> dict[str, Any]:
    return _load().get("views", {}).get(scope, {})
