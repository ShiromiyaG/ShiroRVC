"""The application name must have exactly one value.

``APP_NAME`` is declared in more than one module on purpose -- ``gui/`` has to
keep working when nothing else is importable.  What is not optional is that
they all say the same thing.  This is the version-literal test's argument
(``test_version.py``) applied to the other identifier a release is built out of.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Directories that are not ours to police.
EXCLUDED = {"env", ".venv", "build", "dist", "__pycache__", ".git", "tests"}

_LITERAL = re.compile(r"""^APP_NAME\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def _sources() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.py")
        if not EXCLUDED & set(path.relative_to(ROOT).parts)
    ]


def declared_names() -> dict[str, str]:
    """Every module-level ``APP_NAME = "..."`` in the tree."""
    found = {}
    for path in _sources():
        match = _LITERAL.search(path.read_text(encoding="utf-8", errors="ignore"))
        if match:
            found[path.relative_to(ROOT).as_posix()] = match.group(1)
    return found


def test_every_app_name_literal_agrees():
    names = declared_names()
    assert names, "APP_NAME should be declared somewhere"
    distinct = set(names.values())
    assert len(distinct) == 1, (
        f"APP_NAME disagrees across the tree: {names}; they must all say the "
        "same thing."
    )


def test_workflows_do_not_hardcode_the_app_name():
    """CI must read the name from the tree, not restate it.

    A literal in a workflow is the one copy no Python test can reach, and it is
    exactly the copy that names the release asset.
    """
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        return

    sys.path.insert(0, str(ROOT))
    import version

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in workflows.glob("*.yml")
        if re.search(
            rf"""APP_NAME\s*:\s*["']?{re.escape(version.APP_NAME)}["']?\s*$""",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert not offenders, (
        f"These workflows hardcode the application name: {offenders}. Read it "
        "from version.py instead so the archive name cannot drift from the tree."
    )
