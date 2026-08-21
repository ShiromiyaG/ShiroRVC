"""The GUI must stay deletable, and it must stay behind one seam.

Both properties are easy to state and easy to break with one convenient
import, so they are checked mechanically rather than by review.  Run with::

    python -m pytest tests/test_gui_isolation.py

The test uses ``ast`` rather than a text search so that a mention of ``gui`` in
a docstring or a path string does not count as an import.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUI = ROOT / "gui"

#: Directories that are not part of the application source.
SKIP_DIRS = {
    ".git", "env", ".venv", "__pycache__", "logs", "assets", ".tmp",
    "build", "dist", "node_modules", ".state",
}

#: Modules belonging to the host application.
BACKEND_ROOTS = {"core", "rvc", "tabs", "app"}

#: Files outside gui/ that are allowed to name it: the launchers and the
#: packaging pipeline have to.
ALLOWED_REFERENCES = {
    "start-gui.bat",
    "start-gui.sh",
}

#: The test suite is exempt wholesale. It is not shipped, nothing imports it at
#: runtime, and testing the GUI necessarily means importing it -- the property
#: being protected is that the *application* keeps working without gui/, which
#: a test file cannot affect.
EXEMPT_DIRS = ("tests/",)

#: The single file inside gui/ allowed to import from the application.
#:
#: ``gui/i18n.py`` re-exports ``rvc.lib.i18n`` so that all three interfaces
#: share one set of ``.po`` files.  The seam this test defends is the *backend*
#: one -- torch, ``core.py``, the training code -- and the catalog is none of
#: those: stdlib ``gettext`` and a path, importing nothing and starting nothing.
#: Duplicating it inside gui/ to satisfy the rule would produce two catalogs
#: that drift, which is a worse outcome than the exemption.
BACKEND_IMPORT_EXEMPT = {"gui/i18n.py"}


def _python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names imported by a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # A relative import never names a top-level package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_nothing_outside_gui_imports_gui():
    """Deleting gui/ must not break the Gradio app or the CLI."""
    offenders = []
    for path in _python_files(ROOT):
        relative = path.relative_to(ROOT).as_posix()
        if (
            relative.startswith("gui/")
            or relative.startswith(EXEMPT_DIRS)
            or relative in ALLOWED_REFERENCES
        ):
            continue
        if "gui" in _imported_roots(path):
            offenders.append(relative)

    assert not offenders, (
        "These files import the GUI, which breaks the promise that gui/ can be "
        f"deleted: {offenders}"
    )


def test_only_services_reach_the_backend():
    """Views and widgets go through gui.services, never straight to core."""
    offenders = []
    for path in _python_files(GUI):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("gui/services/") or relative in BACKEND_IMPORT_EXEMPT:
            continue
        leaked = _imported_roots(path) & BACKEND_ROOTS
        if leaked:
            offenders.append(f"{relative} -> {sorted(leaked)}")

    assert not offenders, (
        "Only gui/services/ may import the backend; move the call behind the "
        f"engine or the catalog: {offenders}"
    )


def test_widgets_do_not_depend_on_services():
    """Widgets stay presentation-only so they can be built without a worker."""
    offenders = []
    for path in _python_files(GUI / "widgets"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] == "services":
                    offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"Widgets must not import services: {offenders}"


def test_gui_imports_without_the_backend(monkeypatch):
    """Importing the GUI must not drag torch in.

    This is what keeps startup under a second; a stray ``import core`` at
    module scope in a view would silently undo it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gui.app, gui.views.training, gui.views.inference; "
            "import sys; "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'core' not in sys.modules, 'core was imported'; "
            "print('clean')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
