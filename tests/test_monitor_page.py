"""The diagnostics page, and the media discovery under it.

Two halves are worth pinning separately.  ``services.media`` is pure path
logic and is tested against a synthetic tree, so it can assert the awkward
cases -- an unpadded epoch number, a sample with audio but no image -- that a
real run will not conveniently provide.  The page itself is checked for the
things that break silently: the sidebar and the stack falling out of step, and
the metrics panel filtering away the per-dimension tags this screen exists to
show.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui.services import media  # noqa: E402  - no Qt, safe to import first


# -- media discovery --------------------------------------------------------


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def run_tree(tmp_path):
    """A log directory shaped like a real run's."""
    root = tmp_path / "logs" / "demo"
    previews = root / media.PREVIEW_DIR_NAME
    for epoch in (1, 2, 10):
        _write(previews / f"epoch_{epoch:04d}" / "mel" / "sample_00.png")
        _write(previews / f"epoch_{epoch:04d}" / "audio" / "sample_00_generated.wav")
        _write(previews / f"epoch_{epoch:04d}" / "audio" / "sample_00_original.wav")
    (root / "eval").mkdir(parents=True, exist_ok=True)
    return root


def test_previews_are_found_from_the_event_directory(run_tree):
    """The monitor knows a run by ``logs/<model>/eval``, one level down."""
    assert media.preview_root(run_tree / "eval") == run_tree / media.PREVIEW_DIR_NAME


def test_previews_are_found_from_the_model_directory(run_tree):
    """A run copied in from elsewhere has its events directly under the model."""
    assert media.preview_root(run_tree) == run_tree / media.PREVIEW_DIR_NAME


def test_missing_run_yields_nothing():
    assert media.preview_root(None) is None
    assert media.preview_root("/definitely/not/here") is None
    assert media.list_previews(None) == []


def test_epochs_sort_numerically_not_lexically(run_tree):
    """``epoch_10`` sorts before ``epoch_9`` as a string; it must not here."""
    epochs = [preview.epoch for preview in media.list_previews(run_tree / "eval")]
    assert epochs == [1, 2, 10]


def test_a_sample_carries_all_three_artifacts(run_tree):
    latest = media.latest_preview(run_tree / "eval")
    assert latest.epoch == 10
    (sample,) = latest.samples
    assert sample.mel is not None
    assert sample.generated is not None and sample.original is not None
    assert sample.has_audio


def test_audio_without_an_image_still_registers(tmp_path):
    """Preview writing is two steps; a run can be interrupted between them."""
    previews = tmp_path / media.PREVIEW_DIR_NAME
    _write(previews / "epoch_0003" / "audio" / "sample_00_generated.wav")
    (epoch,) = media.list_previews(tmp_path)
    assert epoch.mel_count == 0
    assert epoch.audio_count == 1
    assert epoch.samples[0].mel is None


def test_unrelated_files_are_ignored(tmp_path):
    previews = tmp_path / media.PREVIEW_DIR_NAME
    _write(previews / "epoch_0001" / "mel" / "sample_00.png")
    _write(previews / "epoch_0001" / "mel" / "notes.txt")
    _write(previews / "epoch_0001" / "audio" / "scratch.tmp")
    _write(previews / "not_an_epoch" / "mel" / "sample_00.png")
    previews_found = media.list_previews(tmp_path)
    assert len(previews_found) == 1
    assert len(previews_found[0].samples) == 1


def test_multiple_samples_per_epoch(tmp_path):
    previews = tmp_path / media.PREVIEW_DIR_NAME
    for index in (0, 1, 2):
        _write(previews / "epoch_0001" / "mel" / f"sample_{index:02d}.png")
    (epoch,) = media.list_previews(tmp_path)
    assert [s.index for s in epoch.samples] == [0, 1, 2]


def test_human_bytes_is_readable():
    assert media.human_bytes(512) == "512 B"
    assert media.human_bytes(2048) == "2 KB"
    assert media.human_bytes(5 * 1024 * 1024) == "5.0 MB"


# -- the page ---------------------------------------------------------------

pytest.importorskip("PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError)

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    # Peak scans outlive the widgets that asked for them otherwise.
    QThreadPool.globalInstance().waitForDone(5000)


def test_sidebar_and_stack_stay_in_step(app):
    """``_on_nav`` indexes ``self.pages`` with the sidebar row.

    Adding a NAV entry without a page makes every row past it open the wrong
    screen, and the last one raise IndexError.
    """
    from gui.app import NAV, MainWindow

    window = MainWindow()
    try:
        assert len(window.pages) == len(NAV)
        shortcuts = [entry[3] for entry in NAV]
        assert shortcuts == sorted(shortcuts), "shortcuts must follow row order"
        assert len(set(shortcuts)) == len(shortcuts), "duplicate shortcut"
    finally:
        window.deleteLater()


def test_diagnostics_page_shows_the_per_dimension_tags(app):
    """The training panel hides these; this page is why they exist."""
    from gui.views.monitor import MonitorPage
    from gui.widgets.metrics import NOISY_PREFIXES

    page = MonitorPage()
    try:
        noisy = f"{NOISY_PREFIXES[0]}dim_0"
        page.metrics.set_available_tags(["loss_avg_50/loss_spectral_50", noisy])
        assert noisy in page.metrics._available
    finally:
        page.deleteLater()


def test_training_panel_still_hides_them(app):
    """The filter is opt-in; the default must not have moved."""
    from gui.widgets.metrics import NOISY_PREFIXES, MetricsPanel

    panel = MetricsPanel()
    try:
        noisy = f"{NOISY_PREFIXES[0]}dim_0"
        panel.set_available_tags(["loss_avg_50/loss_spectral_50", noisy])
        assert noisy not in panel._available
    finally:
        panel.deleteLater()


def test_include_noisy_re_filters_without_a_re_read(app):
    from gui.widgets.metrics import NOISY_PREFIXES, MetricsPanel

    panel = MetricsPanel()
    try:
        noisy = f"{NOISY_PREFIXES[0]}dim_0"
        panel.set_available_tags(["loss_avg_50/loss_spectral_50", noisy])
        assert noisy not in panel._available
        panel.include_noisy = True
        assert noisy in panel._available, "flipping the flag must re-apply the filter"
        panel.include_noisy = False
        assert noisy not in panel._available
    finally:
        panel.deleteLater()


def test_preview_fits_the_tab_it_was_not_built_on(app, tmp_path, run_tree):
    """The image must fill the viewport after switching to the Previews tab.

    The gallery lives on a tab that is not current when the page is built, so
    the pixmap is first scaled against a placeholder geometry.  The resize that
    gives it the real size arrives on the scroll area's *viewport*, not on the
    widget, so watching only ``resizeEvent`` left the image stuck at roughly a
    third of its width until the window itself was resized.
    """
    from PySide6.QtGui import QPixmap

    from gui.views.monitor import MonitorPage

    # A real PNG, wide and short like the trainer writes.
    image_path = run_tree / media.PREVIEW_DIR_NAME / "epoch_0010" / "mel" / "sample_00.png"
    QPixmap(1608, 388).save(str(image_path))

    page = MonitorPage()
    try:
        page.resize(1200, 700)
        page.show()
        page._load_media(str(run_tree / "eval"))
        page.tabs.setCurrentIndex(1)  # the user's path: switch after showing
        app.processEvents()

        viewport = page.gallery.image._scroll.viewport()
        shown = page.gallery.image._label.pixmap()
        assert not shown.isNull(), "no image was displayed"
        assert shown.width() >= viewport.width() - 4, (
            f"image is {shown.width()}px in a {viewport.width()}px viewport"
        )
    finally:
        page.deleteLater()


def test_the_step_badge_reports_progress(app):
    """The badge beside "Live metrics" showed its placeholder dash forever.

    The training page fills it after every read; the diagnostics page did not,
    so a chart plainly full of data sat under a badge reading "no data".
    """
    from gui.services.tbreader import RunReader
    from gui.views.monitor import MonitorPage

    page = MonitorPage()
    try:
        placeholder = page.metrics.step_badge.text()

        reader = RunReader(".")
        reader.series["loss_avg_50/loss_spectral_50"] = ([100, 200, 38550], [3.0, 2.0, 1.0])
        page._reader = reader
        page._pending_initial = True
        page._on_metrics_read(reader, True)

        assert page.metrics.step_badge.text() != placeholder
        assert "38,550" in page.metrics.step_badge.text()
    finally:
        page.deleteLater()


def test_clearing_drops_a_stale_step(app):
    """Switching runs must not leave the previous run's step on screen."""
    from gui.widgets.metrics import MetricsPanel

    panel = MetricsPanel()
    try:
        placeholder = panel.step_badge.text()
        panel.set_run("", "step 38,550")
        assert panel.step_badge.text() == "step 38,550"
        panel.clear()
        assert panel.step_badge.text() == placeholder
    finally:
        panel.deleteLater()


def test_page_survives_a_run_with_no_previews(app, tmp_path):
    """A fresh run has an event file long before it has written any media."""
    from gui.views.monitor import MonitorPage

    page = MonitorPage()
    try:
        page._load_media(str(tmp_path))
        assert page._previews == []
        assert page.audio_picker.count() == 0
        assert "not written validation previews" in page.preview_note.text()
    finally:
        page.deleteLater()
