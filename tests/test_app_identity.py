"""The application name must have exactly one value.

``APP_NAME`` is declared in several modules on purpose -- ``installer/`` is
standalone by design, since it runs before the application is on disk, and
``gui/`` has to keep working when nothing else is importable.  What is not
optional is that they all say the same thing, because two of them are load
bearing in a way nothing else checks:

* the release workflow names the source archive ``<APP_NAME>-source-<tag>.zip``;
* ``installer.config.source_url`` builds the URL the wizard downloads from the
  same pattern.

Disagree on those two and every install fails with a 404 at run time, while the
build stays green -- the smoke test only proves the wizard opens a window, not
that the thing it will later fetch exists.  This is the version-literal test's
argument (``test_version.py``) applied to the other identifier a release is
built out of.
"""

from __future__ import annotations

import re
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
        f"APP_NAME disagrees across the tree: {names}. The release workflow and "
        "installer.config.source_url build the same filename from it, so a "
        "mismatch is a 404 for every install."
    )


def test_release_asset_name_matches_what_the_installer_fetches():
    """The workflow's archive name and the wizard's URL must be the same string."""
    import sys

    sys.path.insert(0, str(ROOT))
    from installer import build_windows, config

    tag = "v9.9.9"
    expected = f"{config.APP_NAME}-source-{tag}.zip"
    url = f"https://github.com/owner/repo/releases/download/{tag}/{expected}"

    assert build_windows.APP_NAME == config.APP_NAME
    # Mirrors the branch of source_url() that a tagged release takes.
    assert url.endswith(expected)


def test_workflows_do_not_hardcode_the_app_name():
    """CI must read the name from the tree, not restate it.

    A literal in a workflow is the one copy no Python test can reach, and it is
    exactly the copy that names the release asset.
    """
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        return

    from installer import config

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in workflows.glob("*.yml")
        if re.search(
            rf"""APP_NAME\s*:\s*["']?{re.escape(config.APP_NAME)}["']?\s*$""",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert not offenders, (
        f"These workflows hardcode the application name: {offenders}. Read it "
        "from version.py instead so the archive name and the installer's URL "
        "cannot drift apart."
    )
