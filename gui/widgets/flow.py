"""A row of controls that wraps instead of overflowing.

``QHBoxLayout`` has one answer when its children do not fit: squeeze them past
their minimum and let them overlap.  On the metrics panel that is exactly what
happened -- at 1130 px the window/smoothing/scale row landed on top of the
chart's legend.  A row that reflows onto a second line has no such failure
mode, and it is the only reason this module exists.

The layout is the one from Qt's own flow-layout example, with the parts this
application does not need left out.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Left-to-right layout that starts a new line when it runs out of width."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout plumbing --------------------------------------------------

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    # -- geometry ----------------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        # The widest single item, not the sum: a row that can wrap is allowed
        # to become one item per line, and claiming otherwise is what puts a
        # horizontal scrollbar on a panel that does not need one.
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect: QRect, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, line_height = area.x(), area.y(), 0
        spacing = self.spacing()

        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + spacing
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + spacing
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + margins.bottom()


class FlowRow(QWidget):
    """Widget wrapper for :class:`FlowLayout`.

    A bare layout nested in a ``QVBoxLayout`` does not get asked for its
    height-for-width in every Qt version; a widget always does.
    """

    def __init__(self, spacing: int = 6, parent: QWidget | None = None):
        super().__init__(parent)
        self._flow = FlowLayout(self, spacing=spacing)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def add(self, *widgets: QWidget) -> None:
        for widget in widgets:
            self._flow.addWidget(widget)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._flow.heightForWidth(width)

    def sizeHint(self) -> QSize:  # noqa: N802
        width = self.width() or self._flow.minimumSize().width()
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._flow.minimumSize()
