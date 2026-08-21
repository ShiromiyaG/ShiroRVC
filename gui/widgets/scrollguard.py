"""Stop the wheel from editing values while scrolling a page.

Qt's default is that a wheel event over a combo box, spin box or slider changes
it.  On a settings-heavy page that is a trap: scrolling from the top of a form
to the button at the bottom silently rewrites every control the cursor passes
over, and nothing announces it.  The damage is invisible until the conversion
comes out wrong.

So the wheel never edits.  These widgets take keyboard and drag input as
before; the wheel is reserved for the scroll area behind them, which is what
the gesture means everywhere else on the page.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QSlider,
    QWidget,
)

#: Widgets whose wheel handling is taken away.  Text areas, lists and trees are
#: deliberately absent: scrolling is their whole purpose.
GUARDED = (QComboBox, QAbstractSpinBox, QSlider)


class WheelGuard(QObject):
    """Application-wide filter that redirects wheel events to the page."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Wheel or not isinstance(watched, GUARDED):
            return False

        # A combo box's popup is a separate window; scrolling that list has to
        # keep working, and it is not what this filter is aimed at.
        if isinstance(watched, QComboBox) and watched.view().isVisible():
            return False

        scroller = self._enclosing_scroll_area(watched)
        if scroller is not None:
            QApplication.sendEvent(scroller.viewport(), event)
        # Swallowed either way: with no scroll area behind it, the correct
        # outcome is still that the value does not move.
        return True

    @staticmethod
    def _enclosing_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                return parent
            parent = parent.parentWidget()
        return None


def install(application: QApplication) -> WheelGuard:
    """Attach the guard for the lifetime of the application.

    Parented to the application so it outlives every widget it filters; a
    garbage-collected filter would silently stop working.
    """
    guard = WheelGuard(application)
    application.installEventFilter(guard)
    return guard
