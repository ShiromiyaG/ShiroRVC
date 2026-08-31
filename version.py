"""The single place the application's version is defined.

The number lives under ``"version"`` in ``assets/config.json``, not in a Python
literal, so that every consumer reads the same bytes:

* ``app.py`` -- the Gradio interface's title
* ``gui/`` -- the native interface's window title and about text
* ``.github/workflows/release.yml`` -- what decides whether to cut a release

It used to live in a plain-text ``VERSION`` file, for the release workflow's
benefit: that workflow triggers on a path and reads the number with ``tr``, and
a bare file needs no parsing. ``assets/config.json`` already carried a
``"version"`` key alongside it, so the number had two definitions that nothing
kept in step. Consolidating on the JSON costs the workflow one ``python3 -c``
-- it already shells out to this module for ``APP_NAME`` -- and removes the
copy that could drift.

The file is a *settings* file: the theme, language, model-author and FP16
tabs rewrite it at runtime. They all read-modify-write, so ``"version"``
survives them, but that is the property to preserve when adding another writer.
"""

from __future__ import annotations

import json
from pathlib import Path

APP_NAME = "ShiroRVC"
APP_DESCRIPTION = "RVC voice conversion and training by Shiromiya."
REPOSITORY = "ShiromiyaG/ShiroRVC"

#: Used when the config is missing, unreadable or carries no version -- a
#: partial copy of the tree, or an installed layout that dropped it.  Never the
#: source of truth.
FALLBACK_VERSION = "0.0.0"

CONFIG_FILE = Path(__file__).resolve().parent / "assets" / "config.json"


def read_version() -> str:
    """The ``"version"`` key of ``assets/config.json``, or the fallback.

    Every failure lands on the fallback rather than raising: this module is
    imported for ``APP_NAME`` on paths that do not care about the number, and a
    settings file a user has just corrupted should not stop the application from
    starting to tell them so.
    """
    try:
        payload = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return FALLBACK_VERSION
    if not isinstance(payload, dict):
        return FALLBACK_VERSION
    value = payload.get("version")
    if not isinstance(value, str):
        return FALLBACK_VERSION
    return value.strip() or FALLBACK_VERSION


__version__ = read_version()
APP_TITLE = f"{APP_NAME} v{__version__}"


if __name__ == "__main__":
    # `python version.py` is how CI and the build scripts ask for the number.
    print(__version__)
