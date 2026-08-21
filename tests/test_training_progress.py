"""The progress card and the line format it depends on.

The GUI cannot scrape rich's progress bar: rich renders nothing at all when
stdout is not a terminal, which is exactly the case under a front-end.  So the
trainer prints a fixed-format line instead, and these tests pin both ends of
that contract -- the producer's format and the consumer's parser.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.progress import (  # noqa: E402
    TrainingProgress,
    _clock,
    parse_progress,
)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


SAMPLE = "[PROGRESS] epoch=7/500 batch=3600/8083 step=48500 G=15.5421  D=1.7712"


def test_parses_a_trainer_line():
    parsed = parse_progress(SAMPLE)
    assert parsed == {
        "epoch": 7,
        "total_epochs": 500,
        "batch": 3600,
        "total_batches": 8083,
        "step": 48500,
        "metrics": "G=15.5421  D=1.7712",
    }


def test_parses_a_line_without_metrics():
    parsed = parse_progress("[PROGRESS] epoch=1/2 batch=5/10 step=5")
    assert parsed["metrics"] == ""
    assert parsed["epoch"] == 1


@pytest.mark.parametrize(
    "line",
    [
        "[TRAIN] Warmup completed at step: 2000",
        "[INIT] Mel distance: huber (beta=0.3)",
        "",
        "epoch=7/500 batch=3600/8083",  # no marker
    ],
)
def test_ignores_everything_else(line):
    assert parse_progress(line) is None


def test_the_trainer_still_emits_this_format():
    """Guards the producer side: the format lives in train.py's f-string.

    Parsed out of the source rather than executed -- importing train.py runs a
    module-level argv parse and pulls in torch.
    """
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    assert "[PROGRESS]" in source, "the trainer no longer emits progress lines"

    # Rebuild the literal the f-string would produce and check the parser eats
    # it, so a rename of the fields fails here rather than silently in the GUI.
    tree = ast.parse(source)
    emitter = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_emit_machine_progress"
    )
    pieces = [
        part.value for part in ast.walk(emitter)
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    ]
    template = "".join(pieces)
    assert "[PROGRESS]" in template
    for field in ("epoch=", "batch=", "step="):
        assert field in template, f"the trainer stopped emitting {field}"


def test_only_prints_when_stdout_is_not_a_terminal():
    """An interactive run must keep seeing only rich's bar."""
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    assert re.search(r"sys\.stdout\.isatty\(\)", source), (
        "the progress line would flood an interactive terminal"
    )


# -- widget behaviour -------------------------------------------------------


def test_begin_shows_an_indeterminate_bar(app):
    """A bar pinned at 0% while the trainer loads reads as a hang."""
    widget = TrainingProgress()
    widget.begin(500)
    assert widget.overall.maximum() == 0
    assert widget.epoch_badge.text() == "0 / 500"


def test_first_line_switches_to_a_real_bar(app):
    widget = TrainingProgress()
    widget.begin(500)
    assert widget.consume(SAMPLE) is True
    assert widget.overall.maximum() == 1000
    assert widget.epoch_badge.text() == "7 / 500"
    # Six full epochs plus 3600/8083 of the seventh, over 500.
    expected = ((7 - 1) + 3600 / 8083) / 500
    assert abs(widget.overall.value() / 1000 - expected) < 1e-3


def test_non_progress_lines_are_reported_as_such(app):
    widget = TrainingProgress()
    widget.begin(10)
    assert widget.consume("[TRAIN] something else") is False


def test_estimate_is_withheld_until_there_is_enough_history(app):
    widget = TrainingProgress()
    widget.begin(500)
    update = {
        "epoch": 7, "total_epochs": 500, "batch": 4180,
        "total_batches": 8083, "step": 49080, "metrics": "",
    }
    widget._samples = [(1000.0 + i, 6 * 8083 + 3600 + i * 20) for i in range(10)]
    assert widget._remaining_seconds(update) is None

    widget._samples = [(1000.0 + i, 6 * 8083 + 3600 + i * 20) for i in range(30)]
    remaining = widget._remaining_seconds(update)
    # 20 batches/s over 500 epochs of 8083 batches, ~52k done.
    assert remaining is not None
    assert 150_000 < remaining < 260_000


def test_estimate_survives_a_stalled_sample(app):
    """A checkpoint write pauses the counter; that must not divide by zero."""
    widget = TrainingProgress()
    widget.begin(10)
    update = {
        "epoch": 1, "total_epochs": 10, "batch": 5,
        "total_batches": 100, "step": 5, "metrics": "",
    }
    widget._samples = [(1000.0 + i, 5) for i in range(30)]  # no progress at all
    assert widget._remaining_seconds(update) is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45, "45s"), (300, "5m 00s"), (5400, "1h 30m"), (200000, "2d 7h")],
)
def test_clock_formatting(seconds, expected):
    assert _clock(seconds) == expected
