"""The installer must not write itself into somebody else's folder.

It extracts the whole application, a Python environment and its own ``_tools``
directory straight into the folder it is given.  Pointed at a Downloads folder
or a drive root that interleaves irrecoverably with whatever was already there:
no uninstall could separate the two again.

Refusing every non-empty folder would be wrong in the other direction, because
reinstalling over an existing install is the normal upgrade path.  So the rule
is "empty, or ours".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from installer.config import APP_NAME  # noqa: E402
from installer.steps import (  # noqa: E402
    MANIFEST_NAME,
    InstallError,
    TargetState,
    check_target,
    inspect_target,
    suggest_target,
)


# -- the subfolder the picker's result becomes ------------------------------


def test_browsing_appends_the_application_name(tmp_path):
    """A folder picker returns the parent the user has in mind, not the target."""
    assert suggest_target(tmp_path) == tmp_path / APP_NAME


def test_a_busy_folder_becomes_an_empty_subfolder(tmp_path):
    """The whole point: picking Downloads must not be a dead end."""
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    for name in ("installer.zip", "invoice.pdf"):
        (downloads / name).write_text("x", encoding="utf-8")

    assert _state(downloads) is TargetState.OCCUPIED
    assert _state(suggest_target(downloads)) is TargetState.EMPTY


def test_a_drive_root_becomes_a_folder_on_it(tmp_path):
    root = Path(tmp_path.anchor or "/")
    suggested = suggest_target(root)
    assert suggested == root / APP_NAME
    assert _state(suggested) is not TargetState.DRIVE_ROOT


def test_suggesting_twice_does_not_nest(tmp_path):
    """Browse, then Browse again into the same place, must not stack copies."""
    once = suggest_target(tmp_path)
    assert suggest_target(once) == once


def test_the_name_check_ignores_case(tmp_path):
    lowercase = tmp_path / APP_NAME.lower()
    lowercase.mkdir()
    assert suggest_target(lowercase) == lowercase


def test_the_default_location_is_already_suggested(tmp_path):
    """``default_install_dir`` ends in the app name, so Browse-free is stable."""
    from installer import config

    default = config.default_install_dir()
    assert suggest_target(default) == default


def _state(path) -> TargetState:
    return inspect_target(Path(path))[0]


def _allows(path) -> bool:
    try:
        check_target(Path(path))
    except InstallError:
        return False
    return True


# -- accepted ---------------------------------------------------------------


def test_a_folder_that_does_not_exist_yet(tmp_path):
    target = tmp_path / "ShiroRVC"
    assert _state(target) is TargetState.EMPTY
    assert _allows(target)


def test_an_empty_folder(tmp_path):
    target = tmp_path / "empty"
    target.mkdir()
    assert _state(target) is TargetState.EMPTY
    assert _allows(target)


@pytest.mark.parametrize("junk", ["desktop.ini", "Thumbs.db", ".DS_Store"])
def test_shell_droppings_still_count_as_empty(tmp_path, junk):
    """Explorer leaves these behind; refusing over one would read as a bug."""
    target = tmp_path / "empty"
    target.mkdir()
    (target / junk).write_text("x", encoding="utf-8")
    assert _state(target) is TargetState.EMPTY
    assert _allows(target)


def test_a_previous_install_is_an_upgrade_not_a_refusal(tmp_path):
    """The manifest is what tells our folder from anyone else's."""
    target = tmp_path / "ShiroRVC"
    target.mkdir()
    (target / MANIFEST_NAME).write_text(json.dumps({"app": "ShiroRVC"}), encoding="utf-8")
    (target / "core.py").write_text("x", encoding="utf-8")
    (target / "env").mkdir()

    assert _state(target) is TargetState.EXISTING_INSTALL
    assert _allows(target), "reinstalling over an existing install must work"
    assert "update it in place" in inspect_target(target)[1]


# -- refused ----------------------------------------------------------------


def test_a_folder_with_unrelated_files(tmp_path):
    target = tmp_path / "Documents"
    target.mkdir()
    for name in ("taxes.pdf", "holiday.jpg", "notes.txt"):
        (target / name).write_text("x", encoding="utf-8")

    assert _state(target) is TargetState.OCCUPIED
    assert not _allows(target)


def test_the_refusal_names_what_it_found(tmp_path):
    """"Not empty" alone leaves the user guessing which folder they picked."""
    target = tmp_path / "Documents"
    target.mkdir()
    for name in ("taxes.pdf", "holiday.jpg", "notes.txt", "song.mp3", "cv.docx"):
        (target / name).write_text("x", encoding="utf-8")

    explanation = inspect_target(target)[1]
    # Sorted, so the sample is stable rather than filesystem-order dependent.
    assert "cv.docx" in explanation
    assert "and 2 more" in explanation


def test_a_folder_holding_only_a_subfolder_is_still_occupied(tmp_path):
    target = tmp_path / "projects"
    target.mkdir()
    (target / "something").mkdir()
    assert _state(target) is TargetState.OCCUPIED
    assert not _allows(target)


def test_a_drive_root(tmp_path):
    root = Path(tmp_path.anchor or "/")
    assert _state(root) is TargetState.DRIVE_ROOT
    assert not _allows(root)


def test_a_path_that_is_a_file(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("x", encoding="utf-8")
    assert _state(target) is TargetState.UNUSABLE
    assert not _allows(target)


@pytest.mark.parametrize("raw", ["", "   ", "."])
def test_an_empty_field_asks_for_a_folder(raw):
    state, explanation = inspect_target(Path(raw))
    assert state is TargetState.UNUSABLE
    assert "Choose a folder" in explanation


@pytest.mark.parametrize("raw", ["ShiroRVC", "sub/folder", "./here"])
def test_a_relative_path_is_refused(raw):
    """It would resolve against wherever the setup exe was launched from."""
    state, explanation = inspect_target(Path(raw))
    assert state is TargetState.UNUSABLE
    assert "full path" in explanation


# -- the pipeline enforces it too -------------------------------------------


def test_install_refuses_before_touching_the_disk(tmp_path, monkeypatch):
    """The wizard checks, but the CLI entry point reaches ``install`` directly.

    This must fail before ``mkdir``, before the disk-space probe and long
    before anything is downloaded.
    """
    from installer import steps

    target = tmp_path / "Documents"
    target.mkdir()
    (target / "taxes.pdf").write_text("x", encoding="utf-8")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the install started work on a refused folder")

    monkeypatch.setattr(steps, "check_disk_space", explode)
    monkeypatch.setattr(steps, "detect_gpu", explode)

    with pytest.raises(InstallError, match="not empty"):
        steps.install(steps.Options(install_dir=target, variant="cpu"))


# -- the wizard --------------------------------------------------------------

pytest.importorskip("PySide6", reason="the setup wizard is a Qt application")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def wizard(app, monkeypatch):
    """A built wizard with GPU probing stubbed out."""
    from installer import bootstrap, steps as steps_module

    monkeypatch.setattr(
        steps_module, "detect_gpu", lambda: {"cuda": False, "reason": "test"}
    )
    window = bootstrap.Installer()
    yield window
    window.deleteLater()


def test_the_install_button_follows_the_folder(app, wizard, tmp_path):
    """Greyed out rather than failing on click.

    Validating only on the click would let someone pick their Downloads
    folder, read an error a second later, and have no idea which of the two
    choices on the page was wrong.
    """
    occupied = tmp_path / "Downloads"
    occupied.mkdir()
    (occupied / "installer.zip").write_text("x", encoding="utf-8")
    empty = tmp_path / "fresh"
    empty.mkdir()

    wizard.choice_page.path_edit.setText(str(occupied))
    app.processEvents()
    assert not wizard.next_button.isEnabled()
    assert "not empty" in wizard.choice_page.location_status.text()

    wizard.choice_page.path_edit.setText(str(empty))
    app.processEvents()
    assert wizard.next_button.isEnabled()


def test_an_existing_install_keeps_the_button_live(app, wizard, tmp_path):
    target = tmp_path / "ShiroRVC"
    target.mkdir()
    (target / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    wizard.choice_page.path_edit.setText(str(target))
    app.processEvents()
    assert wizard.next_button.isEnabled()
    assert "update it in place" in wizard.choice_page.location_status.text()


def test_the_default_location_is_installable(app, wizard):
    """Whatever ``default_install_dir`` returns must pass its own check."""
    assert wizard.next_button.isEnabled()
    assert wizard.choice_page.target_is_usable()


def test_browsing_a_busy_folder_lands_in_a_subfolder(app, wizard, tmp_path, monkeypatch):
    """The end-to-end shape: picking Downloads is not an error, it is a subfolder.

    Typing that same folder in by hand still refuses -- the appending is a
    convenience of the picker, not a silent rewrite of what the user wrote.
    """
    from PySide6.QtWidgets import QFileDialog

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "installer.zip").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(downloads))
    )
    wizard.choice_page._browse()
    app.processEvents()

    assert Path(wizard.choice_page.path_edit.text()) == downloads / APP_NAME
    assert wizard.next_button.isEnabled()
    assert str(downloads / APP_NAME) in wizard.choice_page.location_status.text()

    wizard.choice_page.path_edit.setText(str(downloads))
    app.processEvents()
    assert not wizard.next_button.isEnabled()
