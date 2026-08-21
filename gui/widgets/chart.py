"""A live line chart drawn with QPainter.

Deliberately not pyqtgraph or matplotlib.  This has to render a handful of
series that grow every few seconds for days, follow the application theme, and
not add a plotting stack to an install that is already several gigabytes.

What it has to get right, because a training chart is nearly useless without
them:

* **A legend carrying the current value.**  The question being asked is "what
  is the loss now, and is it still falling" -- which a colour key alone cannot
  answer.
* **A real log axis.**  Reconstruction loss falls from ~60 to ~14 in the first
  2% of a run; on a linear axis the remaining 98% is a flat line at the bottom.
  Ticks are labelled with actual values, not exponents.
* **A window on the x-axis.**  Once a run is 300k steps long, the interesting
  part is the last few thousand.
* **Decimation that keeps extremes**, so a spike is never sampled away.

Performance
-----------
The grid and the series lines are rendered once into a pixmap and blitted on
subsequent paints; only the legend and the hover crosshair are redrawn live.
Without that split, every mouse move re-ran decimation, smoothing and ~24k
line segments: measured at 146 ms per paint on a 20k-point six-series run,
which is a 7 fps chart under the cursor.  The cached path makes hovering
independent of run length.
"""

from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

#: Distinguishable at a glance and colour-blind-safe enough for six series,
#: which is more than a training chart should show at once anyway.
SERIES_COLOURS = [
    "#8b7cf6", "#2dd4bf", "#f59e0b", "#fb7185", "#38bdf8", "#a3e635",
]

#: Hard ceiling on points per series after decimation.  The effective budget is
#: derived from the plot width -- drawing more segments than the widget has
#: pixels costs time and shows nothing.
_MAX_POINTS = 2000
_MIN_POINTS = 400

#: Width below which a legend entry cannot show its swatch, label, reading and
#: trend arrow without eliding the label into uselessness.
MIN_LEGEND_ENTRY = 210
#: Plot area worth drawing at all.  The chart asks the layout for this plus
#: whatever the legend needs.
MIN_PLOT_HEIGHT = 96

#: How many trailing steps each window preset keeps.  ``None`` means everything.
WINDOWS: list[tuple[str, int | None]] = [
    ("All", None),
    ("50k", 50_000),
    ("10k", 10_000),
    ("2k", 2_000),
]


def _decimate(steps: Sequence[float], values: Sequence[float], budget: int = _MAX_POINTS):
    """Thin a series while keeping its extremes.

    Plain stride sampling drops the spikes, which on a loss curve are the only
    interesting part.  This keeps the min and max of every bucket, so a
    divergence still shows up at any zoom level.
    """
    count = len(values)
    if count <= budget:
        return list(steps), list(values)

    bucket = math.ceil(count / (budget / 2))
    out_steps: list[float] = []
    out_values: list[float] = []
    for start in range(0, count, bucket):
        chunk = values[start:start + bucket]
        chunk_steps = steps[start:start + bucket]
        if not chunk:
            continue
        low = min(range(len(chunk)), key=chunk.__getitem__)
        high = max(range(len(chunk)), key=chunk.__getitem__)
        first, second = (low, high) if low <= high else (high, low)
        out_steps.append(chunk_steps[first])
        out_values.append(chunk[first])
        if second != first:
            out_steps.append(chunk_steps[second])
            out_values.append(chunk[second])
    return out_steps, out_values


def _smooth(values: Sequence[float], weight: float) -> list[float]:
    """TensorBoard's exponential smoothing, with its debias correction.

    Without the debias term the first points of a smoothed curve are dragged
    towards zero, which on a loss that starts at 40 looks like a phantom warmup.
    """
    if weight <= 0 or len(values) < 2:
        return list(values)
    smoothed = []
    last = 0.0
    debias = 0.0
    for value in values:
        last = last * weight + (1 - weight) * value
        debias = debias * weight + (1 - weight)
        smoothed.append(last / debias if debias else value)
    return smoothed


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Tick positions on round numbers rather than on even fractions.

    Axis labels like 13.87 / 27.74 / 41.61 are technically correct and unusable;
    people read a chart against 10 / 20 / 30.
    """
    if high <= low:
        return [low]
    raw = (high - low) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if raw <= step:
            break
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.001:
        # Re-rounding kills the float dust that turns 0.30000000000000004 into
        # a four-decimal label.
        ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


def _log_ticks(low: float, high: float) -> list[float]:
    """Decade ticks, with 2/5 subdivisions when the span is narrow."""
    ticks: list[float] = []
    start = math.floor(low)
    end = math.ceil(high)
    subdivide = (end - start) <= 3
    for exponent in range(int(start), int(end) + 1):
        for mantissa in ((1, 2, 5) if subdivide else (1,)):
            value = math.log10(mantissa) + exponent
            if low - 1e-9 <= value <= high + 1e-9:
                ticks.append(value)
    return ticks or [low, high]


class LiveChart(QWidget):
    """Multi-series step/value chart with a legend and a hover crosshair."""

    #: Emitted when the user clicks a legend row, so the panel can drop a series.
    seriesClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # Tracked by :meth:`_sync_minimum_height`, which asks for exactly the
        # plot floor plus however many rows the legend currently needs.  A
        # fixed number here was wrong in both directions: too tall for one
        # legend row, and -- because a box layout that cannot meet its
        # children's minimums overlaps them rather than scrolling -- too short
        # to stop the controls below from being drawn over the legend.
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Without this the layout never calls heightForWidth, and the legend
        # goes back to being drawn under the controls at narrow widths.
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMouseTracking(True)
        # Nothing shows through and nothing behind needs painting first: this
        # skips one full-widget background fill per repaint.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._series: dict[str, tuple[list[float], list[float]]] = {}
        self._order: list[str] = []
        self._hover: QPoint | None = None
        self._legend_rows: list[tuple[QRectF, str]] = []
        self._plot = QRectF()

        self._smoothing = 0.6
        self._log_scale = False
        self._window_steps: int | None = None
        self._show_raw = True

        # Cache state.  `_revision` changes when the data does; `_theme` when
        # the palette does.  Together with the geometry and the view options
        # they key the rendered scene.
        self._revision = 0
        self._theme_revision = 0
        self._scene: QPixmap | None = None
        self._scene_key: tuple | None = None
        self._prepared: dict[str, tuple[list[float], list[float], list[float]]] = {}
        self._bounds: tuple[float, float, float, float] | None = None

        self.colours = {
            "bg": QColor("#16161c"),
            "grid": QColor("#2e2e3a"),
            "axis": QColor("#3d3d4d"),
            "text": QColor("#9a9aab"),
            "text_strong": QColor("#e8e8ef"),
            "faint": QColor("#6b6b7d"),
        }


    # -- view options ------------------------------------------------------
    # Properties rather than plain attributes so that changing one invalidates
    # the cached scene.  A silently stale chart is worse than a slow one.

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float) -> None:
        if value != self._smoothing:
            self._smoothing = value
            self._invalidate()

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool) -> None:
        if value != self._log_scale:
            self._log_scale = value
            self._invalidate()

    @property
    def window_steps(self) -> int | None:
        return self._window_steps

    @window_steps.setter
    def window_steps(self, value: int | None) -> None:
        if value != self._window_steps:
            self._window_steps = value
            self._invalidate()

    @property
    def show_raw(self) -> bool:
        return self._show_raw

    @show_raw.setter
    def show_raw(self, value: bool) -> None:
        if value != self._show_raw:
            self._show_raw = value
            self._invalidate()

    def _invalidate(self) -> None:
        self._scene_key = None
        # The legend's row count follows the series count, so the answer this
        # widget gives to heightForWidth just changed and the layout has to be
        # told to ask again.
        self.updateGeometry()
        self.update()

    # -- theming -----------------------------------------------------------

    def apply_theme(self, tokens: dict[str, str]) -> None:
        self.colours["bg"] = QColor(tokens["input"])
        self.colours["grid"] = QColor(tokens["border"])
        self.colours["axis"] = QColor(tokens["border_strong"])
        self.colours["text"] = QColor(tokens["text_dim"])
        self.colours["text_strong"] = QColor(tokens["text"])
        self.colours["faint"] = QColor(tokens["text_faint"])
        self._theme_revision += 1
        self._invalidate()

    # -- data --------------------------------------------------------------

    def set_series(self, name: str, steps: Sequence[float], values: Sequence[float]) -> None:
        existing = self._series.get(name)
        # The training view re-pushes selected series on every poll. Skipping
        # an unchanged one keeps the cached scene alive instead of re-rendering
        # the whole chart three times a second for nothing.
        if (
            existing is not None
            and len(existing[1]) == len(values)
            and (not values or existing[1][-1] == values[-1])
        ):
            return
        if name not in self._order:
            self._order.append(name)
        self._series[name] = (list(steps), list(values))
        self._revision += 1
        self._invalidate()

    def remove_series(self, name: str) -> None:
        if name not in self._series:
            return
        self._series.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        self._revision += 1
        self._invalidate()

    def clear(self) -> None:
        if not self._series and not self._order:
            return
        self._series.clear()
        self._order.clear()
        self._revision += 1
        self._invalidate()

    def series_names(self) -> list[str]:
        return list(self._order)

    def colour_for(self, name: str) -> str:
        if name not in self._order:
            return SERIES_COLOURS[0]
        return SERIES_COLOURS[self._order.index(name) % len(SERIES_COLOURS)]

    def latest(self, name: str) -> float | None:
        entry = self._series.get(name)
        return entry[1][-1] if entry and entry[1] else None

    def trend(self, name: str, span: int = 40) -> float | None:
        """Change over the last ``span`` points, as a fraction of the older value.

        This is the number that answers "is it still improving", which is the
        actual question a loss chart exists to answer.
        """
        entry = self._series.get(name)
        if not entry or len(entry[1]) < 4:
            return None
        values = entry[1]
        window = min(span, len(values) // 2)
        recent = sum(values[-window:]) / window
        earlier = sum(values[-2 * window:-window]) / window
        if earlier == 0:
            return None
        return (recent - earlier) / abs(earlier)

    # -- painting ----------------------------------------------------------

    def _chart_font(self) -> QFont:
        font = QFont(self.font())
        font.setPointSize(8)
        return font

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        font = self._chart_font()
        painter.setFont(font)
        metrics = QFontMetrics(font)

        self._rebuild(metrics)

        if not self._prepared or self._scene is None:
            painter.fillRect(self.rect(), self.colours["bg"])
            painter.setPen(self.colours["faint"])
            painter.drawText(
                self.rect(), Qt.AlignCenter, "Select a metric below to plot it"
            )
            return

        # Grid and series come from the cache; only the two interactive layers
        # below are drawn per paint.
        painter.drawPixmap(0, 0, self._scene)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_legend(painter, metrics)
        if self._hover and self._plot.contains(QPointF(self._hover)):
            self._draw_crosshair(painter, metrics)

    def _rebuild(self, metrics: QFontMetrics) -> None:
        """Recompute the prepared series and re-render the scene, if needed."""
        key = (
            self.width(), self.height(), self._revision, self._theme_revision,
            self._smoothing, self._log_scale, self._window_steps, self._show_raw,
        )
        if key == self._scene_key:
            return
        self._scene_key = key

        active = [
            name for name in self._order
            if self._series.get(name, ([], []))[1]
        ]
        legend_height = self._legend_height(metrics, len(active))
        self._plot = QRectF(
            58, 14,
            max(1.0, self.width() - 74.0),
            max(1.0, self.height() - 40.0 - legend_height),
        )
        # One point per pixel of plot width: anything finer is drawn on top of
        # itself. Halving the old budget halves the segment count, which is
        # what the rebuild actually spends its time on.
        budget = int(max(_MIN_POINTS, min(_MAX_POINTS, self._plot.width())))
        self._prepared = self._prepare(budget)

        if not self._prepared:
            self._scene = None
            self._bounds = None
            return

        self._bounds = self._bounds_of(self._prepared)

        ratio = self.devicePixelRatioF()
        scene = QPixmap(
            max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        )
        scene.setDevicePixelRatio(ratio)
        scene.fill(self.colours["bg"])

        painter = QPainter(scene)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self._chart_font())
        self._draw_grid(painter, metrics)
        self._draw_series(painter)
        painter.end()
        self._scene = scene

    def _prepare(self, budget: int) -> dict[str, tuple[list[float], list[float], list[float]]]:
        """``{name: (steps, smoothed, raw)}`` after windowing and decimation."""
        prepared = {}
        for name in self._order:
            steps, values = self._series.get(name, ([], []))
            if not values:
                continue

            if self._window_steps is not None and steps:
                cutoff = steps[-1] - self._window_steps
                start = 0
                for index, step in enumerate(steps):
                    if step >= cutoff:
                        start = index
                        break
                steps, values = steps[start:], values[start:]
                if not values:
                    continue

            steps, values = _decimate(steps, values, budget)
            smoothed = _smooth(values, self._smoothing)

            if self._log_scale:
                # Non-positive samples have no place on a log axis; dropping
                # the whole triple keeps the arrays aligned.
                triples = [
                    (s, math.log10(sm), math.log10(r))
                    for s, sm, r in zip(steps, smoothed, values)
                    if sm > 0 and r > 0
                ]
                if not triples:
                    continue
                steps = [t[0] for t in triples]
                smoothed = [t[1] for t in triples]
                values = [t[2] for t in triples]

            prepared[name] = (steps, smoothed, values)
        return prepared

    @staticmethod
    def _bounds_of(prepared: dict) -> tuple[float, float, float, float]:
        x_min = min(min(entry[0]) for entry in prepared.values())
        x_max = max(max(entry[0]) for entry in prepared.values())
        # Bounds follow the smoothed line: letting raw spikes set the scale
        # squashes the curve everyone is actually looking at.
        y_min = min(min(entry[1]) for entry in prepared.values())
        y_max = max(max(entry[1]) for entry in prepared.values())
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            padding = abs(y_max) * 0.1 or 1.0
            y_min, y_max = y_min - padding, y_max + padding
        else:
            padding = (y_max - y_min) * 0.1
            y_min, y_max = y_min - padding, y_max + padding
        return x_min, x_max, y_min, y_max

    def _point(self, step, value) -> QPointF:
        plot = self._plot
        x_min, x_max, y_min, y_max = self._bounds
        x = plot.left() + (step - x_min) / (x_max - x_min) * plot.width()
        y = plot.bottom() - (value - y_min) / (y_max - y_min) * plot.height()
        return QPointF(x, y)

    def _polyline(self, steps: Sequence[float], values: Sequence[float]) -> QPolygonF:
        """Map a series into device coordinates in one pass.

        The scale factors are hoisted out of the loop and the polygon is handed
        to Qt whole: ``drawPolyline`` on a prepared QPolygonF is markedly
        cheaper than a QPainterPath built with a ``lineTo`` per point, and each
        of those crosses the Python/C++ boundary.
        """
        plot = self._plot
        x_min, x_max, y_min, y_max = self._bounds
        x_scale = plot.width() / (x_max - x_min)
        y_scale = plot.height() / (y_max - y_min)
        left = plot.left()
        bottom = plot.bottom()
        return QPolygonF(
            [
                QPointF(left + (step - x_min) * x_scale, bottom - (value - y_min) * y_scale)
                for step, value in zip(steps, values)
            ]
        )

    def _draw_grid(self, painter, metrics) -> None:
        plot = self._plot
        x_min, x_max, y_min, y_max = self._bounds
        y_ticks = _log_ticks(y_min, y_max) if self._log_scale else _nice_ticks(y_min, y_max)

        painter.setPen(QPen(self.colours["grid"], 1, Qt.DotLine))
        for value in y_ticks:
            y = self._point(x_min, value).y()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        painter.setPen(self.colours["faint"])
        for value in y_ticks:
            y = self._point(x_min, value).y()
            shown = 10 ** value if self._log_scale else value
            painter.drawText(
                QRectF(0, y - 8, plot.left() - 9, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                self._format(shown),
            )

        for value in _nice_ticks(x_min, x_max, 5):
            x = self._point(value, y_min).x()
            if not (plot.left() - 1 <= x <= plot.right() + 1):
                continue
            label = self._format(value, integral=True)
            width = metrics.horizontalAdvance(label)
            painter.setPen(QPen(self.colours["grid"], 1, Qt.DotLine))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(self.colours["faint"])
            painter.drawText(
                QRectF(x - width / 2 - 6, plot.bottom() + 5, width + 12, 15),
                Qt.AlignCenter, label,
            )

        painter.setPen(QPen(self.colours["axis"], 1))
        painter.drawLine(
            QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom())
        )
        # The axis caption goes in the y-label gutter, not at the right edge,
        # where it collided with the last tick label.
        painter.setPen(self.colours["faint"])
        painter.drawText(
            QRectF(0, plot.bottom() + 5, plot.left() - 9, 15),
            Qt.AlignRight | Qt.AlignVCenter, "step",
        )

    def _draw_series(self, painter) -> None:
        plot = self._plot
        painter.save()
        painter.setClipRect(plot.adjusted(-1, -1, 1, 1))

        for name, (steps, smoothed, raw) in self._prepared.items():
            colour = QColor(self.colour_for(name))

            if self._show_raw and self._smoothing > 0.05:
                # The unsmoothed trace as a faint ghost: it is the only way to
                # see how noisy a metric is once smoothing is turned up.  Drawn
                # at half resolution -- it is a texture, not a curve to read,
                # and it is half the segments in the whole chart.
                ghost = self._polyline(steps[::2], raw[::2])
                faded = QColor(colour)
                faded.setAlpha(48)
                painter.setPen(QPen(faded, 1.0))
                painter.drawPolyline(ghost)

            polyline = self._polyline(steps, smoothed)
            line = QPainterPath()
            line.addPolygon(polyline)

            # A single-series chart gets a gradient fill; several would just
            # muddy each other, so the fill is dropped once there is company.
            if len(self._prepared) == 1:
                area = QPainterPath(line)
                area.lineTo(QPointF(self._point(steps[-1], 0).x(), plot.bottom()))
                area.lineTo(QPointF(self._point(steps[0], 0).x(), plot.bottom()))
                area.closeSubpath()
                gradient = QLinearGradient(0, plot.top(), 0, plot.bottom())
                top = QColor(colour)
                top.setAlpha(64)
                bottom = QColor(colour)
                bottom.setAlpha(0)
                gradient.setColorAt(0.0, top)
                gradient.setColorAt(1.0, bottom)
                painter.setPen(Qt.NoPen)
                painter.setBrush(gradient)
                painter.drawPath(area)
                painter.setBrush(Qt.NoBrush)

            painter.setPen(QPen(colour, 1.8))
            painter.drawPath(line)

            # A dot on the newest sample: the eye needs somewhere to land when
            # the chart updates every three seconds.
            if steps:
                head = self._point(steps[-1], smoothed[-1])
                painter.setPen(Qt.NoPen)
                painter.setBrush(colour)
                painter.drawEllipse(head, 3.0, 3.0)
                painter.setBrush(Qt.NoBrush)

        painter.restore()

    def _legend_columns(self, count: int, width: float | None = None) -> int:
        """How many legend entries fit side by side at ``width``.

        Fixed at two before, which wasted most of a wide chart and cost a whole
        row -- three series became two rows, and those rows are subtracted from
        the plot.  An entry needs room for its swatch, an elided label, the
        reading and the trend arrow; below that it is not worth a column.
        """
        if count <= 0:
            return 1
        usable = max(1.0, (self.width() if width is None else width) - 24.0)
        return max(1, min(int(usable // MIN_LEGEND_ENTRY), count))

    def _legend_height(
        self, metrics: QFontMetrics, count: int, width: float | None = None
    ) -> float:
        if not count:
            return 0.0
        rows = math.ceil(count / self._legend_columns(count, width))
        return 8 + rows * (metrics.height() + 8)

    def _active_count(self) -> int:
        return sum(1 for name in self._order if self._series.get(name, ([], []))[1])

    # Height genuinely depends on width here: a narrower chart fits fewer
    # legend entries per row, needs more rows, and each row comes out of the
    # plot.  ``heightForWidth`` is the mechanism for that, and unlike pushing
    # ``setMinimumHeight`` from ``resizeEvent`` it is answered *during* the
    # layout pass -- so there is never a frame where the layout is working from
    # a stale minimum and overlapping the widget below.

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt's spelling
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt's spelling
        metrics = QFontMetrics(self._chart_font())
        return int(
            math.ceil(
                40 + MIN_PLOT_HEIGHT
                + self._legend_height(metrics, self._active_count(), float(width))
            )
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's spelling
        # The floor a layout falls back on when it ignores heightForWidth: one
        # legend row's worth, never less than the plot is worth drawing at.
        metrics = QFontMetrics(self._chart_font())
        rows = 1 if self._active_count() else 0
        legend = 8 + rows * (metrics.height() + 8) if rows else 0
        return QSize(240, int(math.ceil(40 + MIN_PLOT_HEIGHT + legend)))

    def _draw_legend(self, painter, metrics) -> None:
        self._legend_rows = []
        if not self._prepared:
            return

        row_height = metrics.height() + 8
        columns = self._legend_columns(len(self._prepared))
        legend_height = self._legend_height(metrics, len(self._prepared))
        top = self.height() - legend_height + 2
        column_width = (self.width() - 24) / columns

        for index, name in enumerate(self._prepared):
            column = index % columns
            row = index // columns
            box = QRectF(
                12 + column * column_width, top + row * row_height,
                column_width - 8, row_height - 2,
            )
            self._legend_rows.append((box, name))

            hovered = self._hover is not None and box.contains(QPointF(self._hover))
            if hovered:
                painter.setPen(Qt.NoPen)
                wash = QColor(self.colours["grid"])
                wash.setAlpha(120)
                painter.setBrush(wash)
                painter.drawRoundedRect(box, 6, 6)
                painter.setBrush(Qt.NoBrush)

            colour = QColor(self.colour_for(name))
            painter.setPen(QPen(colour, 2.4))
            middle = box.center().y()
            painter.drawLine(QPointF(box.left() + 6, middle), QPointF(box.left() + 20, middle))

            value = self.latest(name)
            trend = self.trend(name)
            label = name.split("/")[-1]
            # The rolling-average suffix is on every tag and carries no
            # information once it is on every legend row too.
            label = label.removesuffix("_50")

            painter.setPen(self.colours["text_strong"])
            reading = self._format(value) if value is not None else "—"
            reading_width = metrics.horizontalAdvance(reading) + 6
            arrow_width = 34
            painter.drawText(
                QRectF(box.right() - reading_width - arrow_width, box.top(),
                       reading_width, box.height()),
                Qt.AlignRight | Qt.AlignVCenter, reading,
            )

            # Below half a percent the movement is batch noise, and rendering
            # it as a rounded "0%" reads as a broken number.
            if trend is not None and abs(trend) >= 0.005:
                falling = trend < 0
                magnitude = abs(trend) * 100
                painter.setPen(QColor("#4ade80") if falling else QColor("#fbbf24"))
                painter.drawText(
                    QRectF(box.right() - arrow_width, box.top(), arrow_width, box.height()),
                    Qt.AlignRight | Qt.AlignVCenter,
                    f"{'▾' if falling else '▴'}{magnitude:.0f}%" if magnitude >= 1
                    else f"{'▾' if falling else '▴'}{magnitude:.1f}%",
                )

            painter.setPen(self.colours["text"])
            available = box.width() - 26 - reading_width - arrow_width
            painter.drawText(
                QRectF(box.left() + 26, box.top(), max(20.0, available), box.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                metrics.elidedText(label, Qt.ElideMiddle, int(max(20.0, available))),
            )

    def _draw_crosshair(self, painter, metrics):
        plot = self._plot
        x_min, x_max, _y_min, _y_max = self._bounds
        x = self._hover.x()
        step = x_min + (x - plot.left()) / plot.width() * (x_max - x_min)

        rows = []
        for name, (steps, smoothed, _raw) in self._prepared.items():
            index = min(range(len(steps)), key=lambda i: abs(steps[i] - step))
            value = smoothed[index]
            rows.append((
                name,
                self._format(10 ** value if self._log_scale else value),
                self.colour_for(name),
                self._point(steps[index], value),
            ))
        if not rows:
            return

        painter.setPen(QPen(self.colours["axis"], 1, Qt.DashLine))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        for _name, _value, colour, point in rows:
            painter.setPen(QPen(QColor(colour), 1.4))
            painter.setBrush(self.colours["bg"])
            painter.drawEllipse(point, 3.4, 3.4)
        painter.setBrush(Qt.NoBrush)

        line_height = metrics.height() + 4
        width = 30 + max(
            metrics.horizontalAdvance(f"{name.split('/')[-1]}   {value}")
            for name, value, _c, _p in rows
        )
        height = line_height * (len(rows) + 1) + 12
        box_x = x + 16 if x + 16 + width < plot.right() else x - 16 - width
        box_y = max(plot.top() + 4, min(self._hover.y() - height / 2, plot.bottom() - height))
        box = QRectF(box_x, box_y, width, height)

        backdrop = QColor(self.colours["bg"])
        backdrop.setAlpha(242)
        painter.setPen(QPen(self.colours["axis"], 1))
        painter.setBrush(backdrop)
        painter.drawRoundedRect(box, 8, 8)
        painter.setBrush(Qt.NoBrush)

        painter.setPen(self.colours["faint"])
        painter.drawText(
            QRectF(box.left() + 11, box.top() + 6, box.width() - 22, line_height),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"step {self._format(step, integral=True)}",
        )
        for index, (name, value, colour, _point) in enumerate(rows):
            top = box.top() + 6 + line_height * (index + 1)
            painter.setPen(QPen(QColor(colour), 2.2))
            painter.drawLine(
                QPointF(box.left() + 11, top + line_height / 2),
                QPointF(box.left() + 22, top + line_height / 2),
            )
            painter.setPen(self.colours["text"])
            painter.drawText(
                QRectF(box.left() + 28, top, box.width() - 38, line_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                name.split("/")[-1].removesuffix("_50"),
            )
            painter.setPen(self.colours["text_strong"])
            painter.drawText(
                QRectF(box.left() + 28, top, box.width() - 39, line_height),
                Qt.AlignRight | Qt.AlignVCenter, value,
            )

    @staticmethod
    def _format(value: float, integral: bool = False) -> str:
        if integral:
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
            if abs(value) >= 1_000:
                return f"{value / 1_000:.1f}k".replace(".0k", "k")
            return f"{value:.0f}"
        magnitude = abs(value)
        if magnitude == 0:
            return "0"
        if magnitude < 1e-3 or magnitude >= 1e6:
            return f"{value:.2e}"
        # Axis ticks land on round numbers by design; "100.0" instead of "100"
        # makes them look like measurements rather than gridlines.
        if abs(value - round(value)) < 1e-9 and magnitude >= 1:
            return f"{round(value):,}"
        if magnitude >= 1000:
            return f"{value:,.0f}"
        if magnitude >= 100:
            return f"{value:.1f}"
        if magnitude >= 1:
            return f"{value:.3g}"
        return f"{value:.4g}"

    # -- interaction -------------------------------------------------------

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        position = event.position().toPoint()
        if self._hover == position:
            return
        was_inside = self._hover is not None and self._plot.contains(QPointF(self._hover))
        self._hover = position
        over_legend = any(box.contains(event.position()) for box, _ in self._legend_rows)
        self.setCursor(Qt.PointingHandCursor if over_legend else Qt.CrossCursor)
        # The scene is cached, so this only redraws the legend and crosshair.
        if was_inside or self._plot.contains(QPointF(position)) or over_legend:
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        for box, name in self._legend_rows:
            if box.contains(event.position()):
                self.seriesClicked.emit(name)
                return

    def leaveEvent(self, _event) -> None:  # noqa: N802
        if self._hover is None:
            return
        self._hover = None
        self.unsetCursor()
        self.update()
