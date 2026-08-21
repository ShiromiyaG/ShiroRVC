"""Native-interface affordances the Gradio tabs already had.

Each of these is a place where the two interfaces offered different amounts of
help for the same task: TF32 chosen from the hardware, the files already
sitting in the datasets and custom-pretraineds folders, and keeping a copy of a
conversion somewhere other than where it was written.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6", reason="the Qt interface is optional")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    QThreadPool.globalInstance().waitForDone(5000)


def _device(capability: str) -> dict:
    return {"index": 0, "name": "Test", "total_vram": 2**33, "capability": capability}


# -- TF32 --------------------------------------------------------------------


@pytest.mark.parametrize(
    "capability,expected",
    [("8.0", True), ("8.9", True), ("9.0", True), ("12.0", True),
     ("7.5", False), ("6.1", False)],
)
def test_tf32_follows_compute_capability(app, capability, expected):
    """Ampere (8.0) and up have the units; older cards would ignore the flag.

    The same rule as ``microarchitecture_capability_checker``, which is what
    the Gradio tab uses to both tick and enable its box.
    """
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        page._apply_tf32_support([_device(capability)])
        assert page.tf32.isChecked() is expected
        assert page.tf32.isEnabled() is expected
    finally:
        page.deleteLater()


def test_tf32_starts_off_until_the_backend_answers(app):
    """A tick before the GPU is known would claim something unverified."""
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        assert not page.tf32.isChecked()
        assert not page.tf32.isEnabled()
    finally:
        page.deleteLater()


def test_tf32_is_off_with_no_gpu_at_all(app):
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        page._apply_tf32_support([])
        assert not page.tf32.isChecked()
        assert not page.tf32.isEnabled()
    finally:
        page.deleteLater()


def test_a_mixed_pair_of_cards_takes_the_capable_one(app):
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        page._apply_tf32_support([_device("7.5"), _device("8.6")])
        assert page.tf32.isChecked()
    finally:
        page.deleteLater()


# -- path suggestions --------------------------------------------------------


def test_pretrained_listing_splits_generator_from_discriminator(tmp_path, monkeypatch):
    from gui.services import catalog, paths

    root = tmp_path / "custom"
    root.mkdir()
    for name in ("G_48k.pth", "G_other.pth", "D_48k.pth", "readme.txt", "notes.md"):
        (root / name).write_bytes(b"x")
    monkeypatch.setattr(paths, "CUSTOM_PRETRAINED_DIR", root)

    generators = catalog.list_custom_pretraineds("G")
    discriminators = catalog.list_custom_pretraineds("D")

    assert len(generators) == 2 and all("G_" in p for p in generators)
    assert len(discriminators) == 1
    assert not any(p.endswith(".txt") or p.endswith(".md")
                   for p in generators + discriminators)
    assert generators == sorted(generators), "order must be stable between refreshes"


def test_pretrained_listing_survives_a_missing_folder(tmp_path, monkeypatch):
    from gui.services import catalog, paths

    monkeypatch.setattr(paths, "CUSTOM_PRETRAINED_DIR", tmp_path / "nope")
    assert catalog.list_custom_pretraineds("G") == []


def test_dataset_listing_skips_folders_with_no_audio(tmp_path, monkeypatch):
    """A fresh install creates empty folders; suggesting them leads nowhere."""
    from gui.services import catalog, paths

    root = tmp_path / "datasets"
    (root / "with_audio").mkdir(parents=True)
    (root / "with_audio" / "take.wav").write_bytes(b"x")
    (root / "nested").mkdir()
    (root / "nested" / "deeper").mkdir()
    (root / "nested" / "deeper" / "take.flac").write_bytes(b"x")
    (root / "empty").mkdir()
    (root / "just_text").mkdir()
    (root / "just_text" / "notes.txt").write_bytes(b"x")
    monkeypatch.setattr(paths, "DATASET_DIR", root)

    found = catalog.list_dataset_folders()
    assert any(name.endswith("with_audio") for name in found)
    assert any(name.endswith("nested") for name in found), "audio further down counts"
    assert not any(name.endswith("empty") for name in found)
    assert not any(name.endswith("just_text") for name in found)


def test_the_path_fields_offer_what_is_on_disk(app, tmp_path, monkeypatch):
    from gui.services import paths
    from gui.views.training import TrainingPage

    pretraineds = tmp_path / "custom"
    pretraineds.mkdir()
    (pretraineds / "G_48k.pth").write_bytes(b"x")
    (pretraineds / "D_48k.pth").write_bytes(b"x")
    datasets = tmp_path / "datasets"
    (datasets / "voice").mkdir(parents=True)
    (datasets / "voice" / "take.wav").write_bytes(b"x")

    monkeypatch.setattr(paths, "CUSTOM_PRETRAINED_DIR", pretraineds)
    monkeypatch.setattr(paths, "DATASET_DIR", datasets)

    page = TrainingPage()
    try:
        page.refresh_suggestions()
        assert page.pretrained_g._completer.model().rowCount() == 1
        assert page.pretrained_d._completer.model().rowCount() == 1
        assert page.dataset._completer.model().rowCount() == 1
    finally:
        page.deleteLater()


def test_a_field_with_no_suggestions_has_no_completer(app):
    from gui.widgets.forms import PathPicker

    picker = PathPicker()
    try:
        picker.set_suggestions(["a.pth"])
        assert picker.edit.completer() is not None
        picker.set_suggestions([])
        assert picker.edit.completer() is None, "an empty popup is worse than none"
    finally:
        picker.deleteLater()


# -- the optimizer moved -----------------------------------------------------


def test_the_optimizer_lives_under_advanced(app):
    """It sat beside the GPU picker as though equally routine."""
    from gui.views.training import TrainingPage
    from gui.widgets.forms import Collapsible

    page = TrainingPage()
    try:
        holder = page.optimizer
        while holder is not None and not isinstance(holder, Collapsible):
            holder = holder.parentWidget()
        assert isinstance(holder, Collapsible), "the optimizer is not inside a Collapsible"
        assert holder is page.train_advanced
    finally:
        page.deleteLater()


# -- saving a copy of a conversion -------------------------------------------


def test_the_save_icon_sits_beside_show_in_folder(app):
    """It acts on the loaded file, so it belongs on the player's transport."""
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        player = page.output_player
        transport = player.layout().itemAt(1).layout()
        order = [
            transport.itemAt(index).widget()
            for index in range(transport.count())
            if transport.itemAt(index).widget() is not None
        ]
        assert player.save_button in order
        assert order.index(player.save_button) == order.index(player.open_button) - 1
        assert not player.save_button.icon().isNull(), "the icon never got painted"
    finally:
        page.deleteLater()


def test_only_the_output_player_offers_saving(app):
    """Offering to keep a copy of the file the user just picked is noise."""
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        assert not page.output_player.save_button.isHidden()
        assert page.input_player.save_button.isHidden()
    finally:
        page.deleteLater()


def test_save_a_copy_is_dead_until_there_is_a_result(app):
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        assert not page.output_player.save_button.isEnabled()
        assert page._last_output == ""
    finally:
        page.deleteLater()


def test_a_conversion_arms_the_icon(app, tmp_path):
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"x")

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        assert page.output_player.save_button.isEnabled()
        assert page._last_output == str(produced)
    finally:
        page.deleteLater()


def test_unloading_disarms_the_icon(app, tmp_path):
    """Clearing the player leaves nothing to copy."""
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"x")

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        page.output_player.clear()
        assert not page.output_player.save_button.isEnabled()
    finally:
        page.deleteLater()


def test_pressing_the_icon_asks_the_owner(app, tmp_path, monkeypatch):
    """The player must not open a dialog of its own; it reports and defers."""
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"audio")
    destination = tmp_path / "Music"
    destination.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(destination))
    )

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        page.output_player.save_button.click()
        assert (destination / "result.wav").read_bytes() == b"audio"
    finally:
        page.deleteLater()


def test_save_a_copy_writes_where_it_was_told(app, tmp_path, monkeypatch):
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"audio")
    destination = tmp_path / "Music"
    destination.mkdir()

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(destination))
    )

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        page._save_copy()
        assert (destination / "result.wav").read_bytes() == b"audio"
        assert produced.exists(), "the original must stay where it was written"
    finally:
        page.deleteLater()


def test_a_second_copy_does_not_overwrite_the_first(app, tmp_path, monkeypatch):
    """Two takes of the same input land in one folder; neither should vanish."""
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"first")
    destination = tmp_path / "Music"
    destination.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(destination))
    )

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        page._save_copy()
        produced.write_bytes(b"second")
        page._save_copy()

        assert (destination / "result.wav").read_bytes() == b"first"
        assert (destination / "result (2).wav").read_bytes() == b"second"
    finally:
        page.deleteLater()


def test_cancelling_the_dialog_writes_nothing(app, tmp_path, monkeypatch):
    from gui.views.inference import InferencePage

    produced = tmp_path / "result.wav"
    produced.write_bytes(b"x")
    destination = tmp_path / "Music"
    destination.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "")
    )

    page = InferencePage()
    try:
        page._on_converted({"preview": str(produced)})
        page._save_copy()
        assert list(destination.iterdir()) == []
    finally:
        page.deleteLater()
