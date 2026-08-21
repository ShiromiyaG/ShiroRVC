"""Two screens watching one run must parse it once.

A ``RunReader`` is only cheap after its first read; that one walks the whole
event file, which is half a second on a long pretrain.  The training page and
the diagnostics page both watch runs, so without sharing, opening the second
one repeats the expensive read and the two views drift apart between polls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6", reason="the Qt interface is optional")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.services import runwatch  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    QThreadPool.globalInstance().waitForDone(5000)


@pytest.fixture(autouse=True)
def clean_pool():
    runwatch.forget_readers()
    yield
    runwatch.forget_readers()


def test_the_same_run_yields_the_same_reader(tmp_path):
    first = runwatch.reader_for(tmp_path)
    second = runwatch.reader_for(tmp_path)
    assert first is second


def test_path_spelling_does_not_split_the_cache(tmp_path):
    """``logs/x/eval`` and ``logs/x/./eval`` are one run, not two."""
    first = runwatch.reader_for(tmp_path)
    second = runwatch.reader_for(str(tmp_path) + os.sep + "." )
    assert first is second


def test_different_runs_get_different_readers(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert runwatch.reader_for(a) is not runwatch.reader_for(b)


def test_the_cache_is_bounded(tmp_path):
    """Each reader holds every series it has read; they cannot accumulate."""
    for index in range(runwatch.READER_CACHE_LIMIT + 3):
        directory = tmp_path / f"run_{index}"
        directory.mkdir()
        runwatch.reader_for(directory)
    assert len(runwatch._readers) <= runwatch.READER_CACHE_LIMIT


def test_the_oldest_run_is_evicted_first(tmp_path):
    directories = []
    for index in range(runwatch.READER_CACHE_LIMIT):
        directory = tmp_path / f"run_{index}"
        directory.mkdir()
        directories.append(directory)
        runwatch.reader_for(directory)

    oldest = runwatch.reader_for(directories[0])
    # Touching it makes it the most recent, so the *second* is now the oldest.
    runwatch.reader_for(directories[0])
    extra = tmp_path / "extra"
    extra.mkdir()
    runwatch.reader_for(extra)

    assert runwatch.reader_for(directories[0]) is oldest, "recently used was evicted"


def test_a_second_poll_joins_the_first_instead_of_reparsing(app, tmp_path):
    """Both callers hear the result; only one read runs.

    The failure this guards against is subtle: if the second caller were simply
    dropped it would never receive ``done``, so its "reading" flag would stay
    set and its timer would stop asking -- a screen that silently goes stale.
    """
    reader = runwatch.reader_for(tmp_path)
    first = runwatch.ReadSignals()
    second = runwatch.ReadSignals()

    heard = []
    first.done.connect(lambda *_: heard.append("first"))
    second.done.connect(lambda *_: heard.append("second"))

    started_first = runwatch.start_poll(reader, first)
    # Registered before the pool thread finishes: this is the coalescing case.
    joined = runwatch.start_poll(reader, second)

    assert started_first is True
    assert joined is False, "the second caller started its own parse"

    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()

    assert set(heard) == {"first", "second"}, f"only {heard} was told"


def test_a_later_poll_starts_a_fresh_read(app, tmp_path):
    reader = runwatch.reader_for(tmp_path)
    signals = runwatch.ReadSignals()
    assert runwatch.start_poll(reader, signals) is True
    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()
    # The in-flight entry must be cleared, or the reader is polled once ever.
    assert runwatch.start_poll(reader, signals) is True


def test_both_pages_share_one_reader(app):
    """The end-to-end point of all of the above."""
    from gui.services import catalog
    from gui.views.monitor import MonitorPage
    from gui.views.training import TrainingPage

    runs = catalog.list_runs()
    if not runs:
        pytest.skip("no training runs on disk to open")
    run_dir = runs[0][1]

    monitor = MonitorPage()
    training = TrainingPage()
    try:
        monitor._attach(run_dir)
        training._attach_reader(run_dir)
        assert monitor._reader is training._reader
    finally:
        monitor.deleteLater()
        training.deleteLater()
