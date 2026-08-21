"""The stub compiled into ``ShiroRVC.exe``.

Freezing the application itself is not viable: torch's wheels are several
gigabytes of native extensions that PyInstaller mis-detects, and a ``--onefile``
build would unpack all of it to a temporary directory on every launch.  So the
executable is a ~15 MB shim whose only job is to start the real interpreter
next to it, and to say something useful when it cannot.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "ShiroRVC"


def message_box(text: str, title: str = APP_NAME) -> None:
    """Report a failure without a console.

    A windowed build has no stderr anyone will see, so an early failure would
    otherwise be a silent non-launch.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
            return
        except Exception:
            pass
    sys.stderr.write(text + "\n")


def application_root() -> Path:
    """The install directory: where this executable lives."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def interpreter(root: Path) -> Path | None:
    """The environment's windowed interpreter, if the install is intact.

    Two layouts are possible.  The installer builds a uv venv, which puts the
    interpreter in ``env/Scripts/``; the repository's older conda installer
    puts it directly in ``env/``.  Checking both means this executable works in
    either kind of install rather than only the one it shipped with.
    """
    candidates = [
        root / "env" / "Scripts" / "pythonw.exe",
        root / "env" / "Scripts" / "python.exe",
        root / "env" / "pythonw.exe",
        root / "env" / "python.exe",
        root / "env" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    root = application_root()

    if not (root / "core.py").is_file():
        message_box(
            f"{APP_NAME} could not find its files.\n\n"
            f"Expected core.py in:\n{root}\n\n"
            "Keep this executable inside the install folder."
        )
        return 1

    python = interpreter(root)
    if python is None:
        message_box(
            f"{APP_NAME} could not find its Python environment.\n\n"
            f"Expected it in:\n{root / 'env'}\n\n"
            "Re-run the installer to repair it."
        )
        return 1

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment.setdefault("PYTHONUTF8", "1")

    creation_flags = 0
    if sys.platform == "win32":
        # CREATE_NO_WINDOW rather than DETACHED_PROCESS (they are mutually
        # exclusive): both keep a console window off the screen, but a detached
        # process has no console at all, and then anything the application
        # spawns that does want one is handed a fresh window instead.  A new
        # process group so a Ctrl+C in whatever started this cannot reach it.
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    try:
        subprocess.Popen(
            [str(python), "-m", "gui"],
            cwd=str(root),
            env=environment,
            creationflags=creation_flags,
            close_fds=True,
        )
    except OSError as error:
        message_box(f"{APP_NAME} could not start:\n\n{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
