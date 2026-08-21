"""The single place the application's version is defined.

The number itself lives in the plain-text ``VERSION`` file next to this module,
not in a Python literal, so that every consumer reads the same bytes:

* ``app.py`` -- the Gradio interface's title
* ``gui/`` -- the native interface's window title and about text
* ``.github/workflows/release.yml`` -- what decides whether to cut a release

That last one is why it is a file rather than an assignment: the workflow
compares ``VERSION`` between two commits with plain ``git show``, and a version
buried in Python would need parsing to do the same job.

Bumping the version is therefore a one-line edit, and pushing that edit to
``main`` is what publishes a release.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ShiroRVC"
APP_DESCRIPTION = "RVC voice conversion and training by Shiromiya."
REPOSITORY = "ShiromiyaG/ShiroRVC"

#: Used when the VERSION file is missing -- a partial copy of the tree, or an
#: installed layout that dropped it.  Never the source of truth.
FALLBACK_VERSION = "0.0.0"

VERSION_FILE = Path(__file__).resolve().parent / "VERSION"


def read_version() -> str:
    """The contents of the VERSION file, or the fallback."""
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return FALLBACK_VERSION
    return value or FALLBACK_VERSION


__version__ = read_version()
APP_TITLE = f"{APP_NAME} v{__version__}"


if __name__ == "__main__":
    # `python version.py` is how CI and the build scripts ask for the number.
    print(__version__)
