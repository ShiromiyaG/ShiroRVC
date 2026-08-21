"""The legend must fit in the space the chart asked the layout for.

A ``QVBoxLayout`` that cannot meet its children's minimums does not clip or
scroll -- it overlaps them.  The chart's minimum height was a fixed 190 px that
knew nothing about the legend, so three series at a narrow width became two
legend rows, the panel came up 17 px short, and the controls row was drawn over
``loss_gen_total``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6", reason="the Qt interface is optional")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.chart import MIN_PLOT_HEIGHT, LiveChart  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _chart(app, count: int, width: int) -> LiveChart:
    """A chart sized exactly as a layout honouring heightForWidth would."""
    chart = LiveChart()
    steps = list(range(200))
    values = [1.0 / (index + 1) for index in range(200)]
    for index in range(count):
        chart.set_series(f"loss_avg_50/metric_{index}_50", steps, values)
    chart.resize(width, chart.heightForWidth(width))
    app.processEvents()
    return chart


@pytest.mark.parametrize("count", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("width", [1600, 1100, 900, 620, 420, 300])
def test_legend_never_leaves_the_chart(app, count, width):
    """At the chart's own minimum height, every legend row is inside it."""
    chart = _chart(app, count, width)
    try:
        chart.grab()  # a real paint, which is what fills _legend_rows
        assert chart._legend_rows, "the legend was not drawn"
        lowest = max(box.bottom() for box, _ in chart._legend_rows)
        assert lowest <= chart.height(), (
            f"{count} series at {width}px: legend reaches {lowest:.0f} "
            f"in a {chart.height()}px chart"
        )
    finally:
        chart.deleteLater()


@pytest.mark.parametrize("count", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("width", [1600, 900, 420])
def test_plot_keeps_its_floor(app, count, width):
    """The legend must not eat the plot; the minimum covers both."""
    chart = _chart(app, count, width)
    try:
        chart.grab()
        assert chart._plot.height() >= MIN_PLOT_HEIGHT - 1
    finally:
        chart.deleteLater()


def test_a_wide_chart_uses_one_legend_row(app):
    """Three series across 1600 px was two rows before; the width was wasted."""
    chart = _chart(app, 3, 1600)
    try:
        assert chart._legend_columns(3) == 3
    finally:
        chart.deleteLater()


def test_a_narrow_chart_falls_back_to_one_column(app):
    chart = _chart(app, 3, 300)
    try:
        assert chart._legend_columns(3) == 1
    finally:
        chart.deleteLater()


@pytest.mark.parametrize("count", [1, 3, 6])
def test_height_for_width_tracks_the_legend(app, count):
    """Narrower means more legend rows means a taller ask.

    Asked of ``heightForWidth`` rather than ``minimumHeight`` deliberately:
    a pushed minimum is only correct once ``resizeEvent`` has been delivered,
    which leaves a frame where the layout works from a stale number and
    overlaps the widget below.  ``heightForWidth`` is answered during the
    layout pass itself, so there is no such window.
    """
    chart = _chart(app, count, 1600)
    try:
        rows_wide = math.ceil(count / chart._legend_columns(count, 1600))
        rows_narrow = math.ceil(count / chart._legend_columns(count, 300))
        if rows_narrow > rows_wide:
            assert chart.heightForWidth(300) > chart.heightForWidth(1600)
        assert chart.heightForWidth(1600) >= MIN_PLOT_HEIGHT
        assert chart.hasHeightForWidth()
        assert chart.sizePolicy().hasHeightForWidth(), (
            "the layout only calls heightForWidth when the policy says to"
        )
    finally:
        chart.deleteLater()


def test_panel_children_never_overlap(app):
    """The bug as reported: the controls row drawn on top of the legend."""
    from gui.widgets.metrics import MetricsPanel

    panel = MetricsPanel()
    try:
        steps = list(range(200))
        values = [1.0 / (index + 1) for index in range(200)]
        tags = [f"loss_avg_50/metric_{index}_50" for index in range(3)]
        panel.set_available_tags(tags)
        panel._set_checked(tags)
        for tag in tags:
            panel.update_series(tag, steps, values)

        for width, height in ((1050, 460), (900, 420), (1400, 700)):
            panel.resize(width, height)
            app.processEvents()
            panel.grab()

            previous = None
            for name in ("chart", "controls", "presets", "picker"):
                rect = getattr(panel, name).geometry()
                if previous is not None:
                    assert rect.top() > previous, (
                        f"{width}x{height}: {name} starts at {rect.top()} but the "
                        f"widget above ends at {previous}"
                    )
                previous = rect.bottom()
    finally:
        panel.deleteLater()
