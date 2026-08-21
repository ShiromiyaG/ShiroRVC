"""Entry point: ``python -m gui``.

Kept thin on purpose.  The one thing it does beyond calling :func:`gui.app.run`
is turn an import failure into a message a user can act on, because the most
likely reason this fails is a missing PySide6 rather than a bug.
"""

from __future__ import annotations

import os
import sys


def _fail(message: str) -> int:
    """Report a startup failure on whatever channel exists.

    Launched from ``pythonw.exe`` there is no console to print to, so fall back
    to a native message box before giving up.
    """
    sys.stderr.write(message + "\n")
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "ShiroRVC", 0x10)
        except Exception:
            pass
    return 1


def main() -> int:
    # The package needs its parent (the application root) on sys.path so that
    # `python -m gui` works from any working directory.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from gui.app import run
    except ImportError as error:
        return _fail(
            f"ShiroRVC could not start: {error}\n\n"
            "The Qt interface needs its own dependencies. Install them with:\n"
            "    pip install -r gui/requirements-gui.txt"
        )

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
