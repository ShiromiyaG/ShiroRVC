"""Behavioural checks for the Qt interface.

Skipped when PySide6 is absent, so the isolation and version tests still run in
a bare CI environment.  These need a real widget tree: the wheel guard is an
event filter, and the only honest way to check an event filter is to send it an
event.
"""

from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError)

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.services import catalog  # noqa: E402
from gui.widgets import scrollguard  # noqa: E402


@pytest.fixture(scope="module")
def app():
    # Offscreen: CI runners have no display, and none of this needs one.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    yield application


def _wheel(widget) -> QWheelEvent:
    """A one-notch scroll-down event positioned over ``widget``."""
    return QWheelEvent(
        QPointF(widget.rect().center()),
        QPointF(widget.mapToGlobal(widget.rect().center())),
        QPoint(0, -120),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


@pytest.fixture
def guarded_page(app):
    """A slider and a combo inside a scroll area, with the guard installed."""
    guard = scrollguard.WheelGuard()
    app.installEventFilter(guard)

    area = QScrollArea()
    inner = QWidget()
    layout = QVBoxLayout(inner)

    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    layout.addWidget(slider)

    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    layout.addWidget(combo)

    # Tall enough that the scroll area actually has somewhere to scroll to.
    filler = QWidget()
    filler.setMinimumHeight(4000)
    layout.addWidget(filler)

    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.resize(400, 300)
    area.show()

    yield area, slider, combo

    app.removeEventFilter(guard)
    area.close()


def test_wheel_does_not_move_a_slider(app, guarded_page):
    _area, slider, _combo = guarded_page
    before = slider.value()
    QApplication.sendEvent(slider, _wheel(slider))
    assert slider.value() == before, "the wheel edited a slider it merely passed over"


def test_wheel_does_not_change_a_combo(app, guarded_page):
    _area, _slider, combo = guarded_page
    before = combo.currentIndex()
    QApplication.sendEvent(combo, _wheel(combo))
    assert combo.currentIndex() == before, "the wheel changed a dropdown selection"


def test_wheel_still_scrolls_the_page(app, guarded_page):
    """Swallowing the event is only half the job; the page has to move."""
    area, slider, _combo = guarded_page
    scrollbar = area.verticalScrollBar()
    scrollbar.setValue(0)
    QApplication.sendEvent(slider, _wheel(slider))
    assert scrollbar.value() > 0, "the wheel was swallowed instead of scrolling the page"


# -- defaults --------------------------------------------------------------


@pytest.mark.parametrize("profile", ["single", "batch", "tts"])
def test_every_profile_defines_the_same_keys(profile):
    """A missing key here is a KeyError at window construction."""
    reference = set(catalog.INFERENCE_DEFAULTS["single"])
    assert set(catalog.INFERENCE_DEFAULTS[profile]) == reference


def test_defaults_match_the_gradio_tabs():
    """Spot-check the values that differ between the three tabs upstream.

    These are the ones an averaged "sensible default" would quietly get wrong,
    which is exactly the divergence this table exists to prevent.
    """
    single = catalog.INFERENCE_DEFAULTS["single"]
    batch = catalog.INFERENCE_DEFAULTS["batch"]
    tts = catalog.INFERENCE_DEFAULTS["tts"]

    assert (single["index_rate"], batch["index_rate"], tts["index_rate"]) == (0.5, 0.5, 0.75)
    assert (single["protect"], batch["protect"], tts["protect"]) == (0.33, 0.3, 0.5)
    assert (single["clean_strength"], batch["clean_strength"], tts["clean_strength"]) == (
        0.3, 0.5, 0.5,
    )
    # TTS is the only tab that denoises by default.
    assert (single["clean_audio"], batch["clean_audio"], tts["clean_audio"]) == (
        False, False, True,
    )
    # Single-file keeps the hidden 0.006 float; the others expose an integer 3.
    assert single["filter_radius"] == 0.006 and not single["filter_radius_visible"]
    assert batch["filter_radius"] == 3 and batch["filter_radius_visible"]


def test_output_path_matches_the_gradio_naming():
    """``output_path_fn`` in tabs/inference/inference.py builds this name."""
    produced = catalog.default_output_path("/somewhere/My Song.flac")
    assert Path(produced).name == "My Song_output.wav"
    assert Path(produced).parent == catalog.paths.AUDIO_DIR


# -- the metrics panel in a small window -----------------------------------


def _spin(app, milliseconds: int) -> None:
    """Let queued work and animations run for a while."""
    deadline = time.monotonic() + milliseconds / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _squeezed(widget, width: int, height: int):
    """Lay a widget out inside a host too small for it, and return the host.

    A top-level widget cannot be resized below its own minimum -- Qt clamps it
    -- so a panel tested on its own never reproduces what a cramped window
    does to it.  As a child of a fixed-size host it does.
    """
    host = QWidget()
    host.setFixedSize(width, height)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    host.show()
    QApplication.processEvents()
    layout.activate()
    QApplication.processEvents()
    return host


def test_metrics_panel_fits_a_laptop_window(app):
    """Its minimum has to fit the monitor column of a 1130x760 window.

    When a QVBoxLayout cannot meet its children's minimums it does not clip
    them, it overlaps them -- which is how the window/smoothing row ended up
    painted across the chart's legend.
    """
    from gui.widgets.metrics import MetricsPanel

    panel = MetricsPanel()
    try:
        assert panel.minimumSizeHint().height() <= 560
    finally:
        panel.deleteLater()


def test_metrics_rows_do_not_overlap_when_narrow(app):
    """The real symptom: two children of the panel sharing pixels."""
    from gui.widgets.metrics import MetricsPanel

    panel = MetricsPanel()
    host = _squeezed(panel, 330, 560)
    try:
        parts = [panel.chart, panel.controls, panel.presets, panel.picker]
        for first, second in zip(parts, parts[1:]):
            assert not first.geometry().intersects(second.geometry()), (
                f"{first.__class__.__name__} overlaps {second.__class__.__name__}: "
                f"{first.geometry()} vs {second.geometry()}"
            )
    finally:
        host.close()
        host.deleteLater()


def test_metrics_collapse_leaves_the_header(app):
    """Something has to stay clickable, or the panel cannot be brought back."""
    from gui.widgets.metrics import MetricsPanel

    panel = MetricsPanel()
    host = _squeezed(panel, 400, 600)
    try:
        panel.set_collapsed(True)
        QApplication.processEvents()
        assert not panel.chart.isVisible() and not panel.picker.isVisible()
        assert panel.collapse_button.isVisible()

        panel.set_collapsed(False)
        QApplication.processEvents()
        assert panel.chart.isVisible() and panel.picker.isVisible()
    finally:
        host.close()
        host.deleteLater()


# -- a run outlives the page that started it -------------------------------


def test_progress_card_can_be_floated_and_reclaimed(app):
    """The dock borrows the card; the page must get the same widget back."""
    from gui.widgets.dock import FloatingDock
    from gui.widgets.progress import TrainingProgress

    host = QWidget()
    host.resize(900, 600)
    dock = FloatingDock(host)
    card = TrainingProgress()
    holder = QWidget()
    QVBoxLayout(holder).addWidget(card)

    host.show()
    QApplication.processEvents()

    dock.adopt(card)
    QApplication.processEvents()
    assert card.parentWidget() is dock
    assert dock.isVisible()
    # Pinned to the bottom of the host, not floating in the middle of it.
    assert dock.geometry().bottom() <= host.height()
    assert dock.geometry().bottom() > host.height() * 0.5

    assert dock.release() is card
    assert not dock.isVisible()

    host.close()
    host.deleteLater()


def test_showing_the_page_does_not_rebuild_the_event_reader(app, tmp_path):
    """Rebuilding it re-reads the run from byte zero, on the UI thread.

    ``_attach_reader`` runs on every show of the page and on every change of
    the model name, so a reader that is thrown away each time turns a tab
    switch into a full parse of the event file -- half a second on a 36 MB
    run, and the reason this page used to freeze on the way in.
    """
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        first_run = tmp_path / "run-a"
        second_run = tmp_path / "run-b"
        first_run.mkdir()
        second_run.mkdir()

        page._attach_reader(str(first_run))
        reader = page._reader
        assert reader is not None

        page._attach_reader(str(first_run))
        assert page._reader is reader, "the same run was attached from scratch"

        page._attach_reader(str(second_run))
        assert page._reader is not reader, "a different run kept the old reader"
    finally:
        page.close()
        page.deleteLater()


def test_the_card_grows_in_and_collapses_away(app):
    """Start and finish are movements, not a widget blinking in and out."""
    from gui.widgets import QWIDGETSIZE_MAX
    from gui.widgets.progress import TrainingProgress

    card = TrainingProgress()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(card)
    layout.addStretch(1)
    host.resize(420, 400)
    host.show()
    QApplication.processEvents()
    card.hide()
    QApplication.processEvents()

    card.begin(50)
    card.reveal()
    # It starts from nothing and is on its way up, rather than already there.
    assert card.isVisible()
    assert card.maximumHeight() < card.sizeHint().height()

    _spin(app, 600)
    assert card.height() > 40, "the card never grew"
    assert card.maximumHeight() == QWIDGETSIZE_MAX, (
        "the ceiling used for the animation was not given back, so the detail "
        "line can no longer wrap"
    )

    gone = []
    card.dismiss(on_done=lambda: gone.append(True))
    _spin(app, 700)
    assert gone == [True]
    assert not card.isVisible()
    assert card.maximumHeight() == QWIDGETSIZE_MAX

    host.close()
    host.deleteLater()


def test_dismissing_a_card_that_never_showed_is_still_reported(app):
    """The window waits for that callback before it drops the floating dock."""
    from gui.widgets.progress import TrainingProgress

    card = TrainingProgress()
    gone = []
    card.dismiss(on_done=lambda: gone.append(True))
    assert gone == [True]
    card.deleteLater()


def test_the_card_travels_between_its_two_homes(app):
    """It moves rather than teleports, and leaves nothing behind."""
    from PySide6.QtCore import QRect
    from gui.widgets.dock import fly

    host = QWidget()
    host.resize(800, 600)
    host.show()
    source = QLabel("progress")
    source.resize(240, 90)

    landed = []
    start, end = QRect(40, 40, 240, 90), QRect(460, 470, 320, 100)
    animation = fly(host, source.grab(), start, end,
                    duration=120, on_finished=lambda: landed.append(True))

    ghost = animation.targetObject()
    assert ghost.isVisible() and ghost.geometry() == start

    _spin(app, 500)
    assert landed == [True], "the flight never reported landing"
    # The stand-in deletes itself; anything left visible would sit on top of
    # the page for the rest of the session.
    assert not [child for child in host.findChildren(QLabel) if child.isVisible()]

    host.close()
    host.deleteLater()


def test_a_flight_cut_short_still_reports_landing(app):
    """Switching tabs twice quickly must not leave the card hidden.

    ``stop()`` does not emit ``finished``, so a handler on that signal alone
    would never re-show the widget the ghost was standing in for.
    """
    from PySide6.QtCore import QRect
    from gui.widgets.dock import fly

    host = QWidget()
    host.resize(800, 600)
    host.show()
    source = QLabel("progress")
    source.resize(240, 90)

    landed = []
    animation = fly(host, source.grab(), QRect(0, 0, 240, 90), QRect(400, 400, 320, 100),
                    duration=5000, on_finished=lambda: landed.append(True))
    animation.stop()
    assert landed == [True]

    host.close()
    host.deleteLater()


def test_stop_replaces_start_while_training(app):
    """Only one of the two is ever the thing to press."""
    from gui.widgets.progress import TrainingProgress

    card = TrainingProgress()
    card.show()
    QApplication.processEvents()

    card.set_stoppable(True)
    assert card.stop_button.isVisible() and card.stop_button.isEnabled()

    stopped = []
    card.stopRequested.connect(lambda: stopped.append(True))
    card.stop_button.click()
    assert stopped, "the card's stop button did not ask for a stop"
    assert not card.stop_button.isEnabled(), "a second stop can be requested"

    card.finish("Training finished.")
    assert not card.stop_button.isVisible()

    card.close()
    card.deleteLater()


# -- results outlive a tab switch ------------------------------------------


def test_audio_player_keeps_its_file_when_hidden(app, tmp_path):
    """Switching tabs hides the page; it must not unload the result."""
    pytest.importorskip("PySide6.QtMultimedia", reason="Qt Multimedia is optional", exc_type=ImportError)
    from gui.widgets.audio import AudioPlayer

    sample = tmp_path / "beep.wav"
    with wave.open(str(sample), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x01" * 4000)

    player = AudioPlayer()
    player.show()
    QApplication.processEvents()
    player.load(str(sample))
    QApplication.processEvents()
    assert player.play_button.isEnabled()

    player.hide()          # what a tab switch does
    QApplication.processEvents()
    player.show()
    QApplication.processEvents()
    assert player._path == str(sample), "the loaded audio was dropped while hidden"
    assert player.play_button.isEnabled()

    # ...and the one thing that is meant to unload it.
    player.clear_button.click()
    assert player._path == ""
    assert not player.play_button.isEnabled()

    player.close()
    player.deleteLater()
