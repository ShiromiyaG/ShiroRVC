"""Frozen entry point for the setup executable.

PyInstaller runs its entry script as ``__main__`` with no package context, so
``bootstrap.py``'s relative imports cannot resolve when it is frozen directly.
This wrapper uses absolute imports instead, which both PyInstaller's analysis
and the interpreter are happy with, and leaves ``bootstrap.py`` runnable as
``python -m installer.bootstrap`` during development.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not getattr(sys, "frozen", False):
        # Running from source: make the repository root importable so that
        # `python installer/setup_entry.py` behaves like the frozen build.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

    from installer.bootstrap import main as run_installer

    return run_installer()


if __name__ == "__main__":
    raise SystemExit(main())
