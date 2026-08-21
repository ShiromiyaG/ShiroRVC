"""Sidebar navigation with an animated selection.

A stylesheet can colour the selected row but it cannot move between rows: QSS
has no transitions, so the highlight teleports.  Painting the selection here
instead lets it slide, which is the difference between a list that changes and
one that responds.

The rest of the row -- icon, label, hover wash -- is still left to QSS, so the
theme stays in one file.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QListWidget


class NavList(QListWidget):
    """A list whose selection indicator animates between rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indicator_top = 0.0
        self._indicator_height = 0.0
        self._pill = QColor("#8b7cf6")
        self._pill.setAlpha(38)
        self._accent = QColor("#8b7cf6")

        self._animation = QPropertyAnimation(self, b"indicatorTop", self)
        self._animation.setDuration(190)
        # Slight overshoot-free ease: the indicator should feel attached to the
        # click, not springy.
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

        self.currentRowChanged.connect(self._retarget)

    # -- animated property -------------------------------------------------

    def _get_indicator_top(self) -> float:
        return self._indicator_top

    def _set_indicator_top(self, value: float) -> None:
        self._indicator_top = value
        self.viewport().update()

    indicatorTop = Property(float, _get_indicator_top, _set_indicator_top)

    # -- theming -----------------------------------------------------------

    def apply_theme(self, accent: str, wash_alpha: int = 38) -> None:
        self._accent = QColor(accent)
        self._pill = QColor(accent)
        self._pill.setAlpha(wash_alpha)
        self.viewport().update()

    # -- geometry ----------------------------------------------------------

    def _retarget(self, row: int) -> None:
        rect = self.visualItemRect(self.item(row)) if 0 <= row < self.count() else None
        if rect is None or rect.isEmpty():
            return
        self._indicator_height = rect.height()
        if self._indicator_top == 0.0 and self._animation.state() != QPropertyAnimation.Running:
            # First selection: land on the row rather than sliding in from the
            # top of the list.
            self._set_indicator_top(float(rect.top()))
            return
        self._animation.stop()
        self._animation.setStartValue(self._indicator_top)
        self._animation.setEndValue(float(rect.top()))
        self._animation.start()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Rows move when the list is resized; the indicator has to follow
        # without animating, or it chases the layout.
        row = self.currentRow()
        if 0 <= row < self.count():
            rect = self.visualItemRect(self.item(row))
            self._indicator_height = rect.height()
            self._animation.stop()
            self._set_indicator_top(float(rect.top()))

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing)

        if self._indicator_height > 0 and self.currentRow() >= 0:
            box = QRectF(
                6.0, self._indicator_top, self.viewport().width() - 12.0,
                self._indicator_height,
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._pill)
            painter.drawRoundedRect(box, 9, 9)

            # A short accent bar on the leading edge: it is what makes the
            # selection read as a position rather than just a tint.
            bar = QRectF(box.left(), box.top() + box.height() * 0.22, 3.0,
                         box.height() * 0.56)
            painter.setBrush(self._accent)
            painter.drawRoundedRect(bar, 1.5, 1.5)

        painter.end()
        # The items paint on top of the indicator, so the label and icon stay
        # legible against it.
        super().paintEvent(event)
