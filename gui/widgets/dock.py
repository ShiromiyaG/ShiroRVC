"""A panel that floats over the page area.

Used for the training progress card.  A run outlives the page that started it
-- it is a process in the backend, and switching to Inference to convert
something while it trains is the normal thing to do -- but the card reporting
it lives on the Training page, so leaving it there means the run becomes
invisible the moment you look away from it.

Deliberately a child widget rather than a ``Qt.Tool`` window: a real window
would appear in the taskbar, float above other applications, and need its own
title bar and theming.  This is the page's own bottom-right corner.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QObject,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .effects import elevate


class FloatingDock(QWidget):
    """Holds one widget pinned to the bottom-right of its host."""

    #: Gap from the host's edges.  Also the room the drop shadow needs, which
    #: is why it is the layout's margin rather than part of the geometry.
    MARGIN = 16

    #: Widest the panel gets.  A progress card stretched across a 2560 px
    #: window is a banner, not a status readout.
    MAX_WIDTH = 620

    def __init__(self, host: QWidget):
        super().__init__(host)
        self._host = host
        self._panel: QWidget | None = None
        host.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN
        )
        layout.setSpacing(0)
        # Floating means it has to read as *above* the page, in both themes,
        # which is the one place a shadow earns its cost unconditionally.
        elevate(self, radius=38, y_offset=12, alpha=150)
        self.hide()

    # -- api ---------------------------------------------------------------

    def adopt(self, panel: QWidget) -> None:
        """Take a widget out of wherever it lives and float it here."""
        if panel.parentWidget() is not self:
            self.layout().addWidget(panel)
        if self._panel is not panel:
            if self._panel is not None:
                self._panel.removeEventFilter(self)
            panel.installEventFilter(self)
            self._panel = panel
        panel.show()
        self.show()
        self.raise_()
        self._reposition()

    def release(self) -> QWidget | None:
        """Hand the panel back.  The caller re-homes it into its own layout."""
        self.hide()
        if self._panel is not None:
            self._panel.removeEventFilter(self)
            self._panel = None
        layout = self.layout()
        item = layout.takeAt(0) if layout.count() else None
        return item.widget() if item is not None else None

    # -- placement ---------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        # The host moving is obvious; the panel resizing matters just as much,
        # because the dock is anchored to a corner rather than to a size.  It
        # is what lets the card animate its own height -- collapsing away when
        # a run ends -- and have the dock collapse with it instead of leaving a
        # shadow around a shrinking card.
        if event.type() in (QEvent.Resize, QEvent.Show):
            if watched is self._host or watched is self._panel:
                self._reposition()
        return False

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.layout().sizeHint()

    def _reposition(self) -> None:
        if not self.isVisible() or self._host.width() <= 0:
            return
        width = min(self.MAX_WIDTH, max(280, self._host.width() - 2 * self.MARGIN))
        height = self.sizeHint().height()
        self.setGeometry(
            self._host.width() - width - self.MARGIN,
            self._host.height() - height - self.MARGIN,
            width,
            height,
        )


def fly(
    host: QWidget,
    snapshot: QPixmap,
    start: QRect,
    end: QRect,
    *,
    duration: int = 260,
    on_finished: Callable[[], None] | None = None,
) -> QPropertyAnimation:
    """Carry a picture of a widget from one rectangle to another.

    A widget cannot be animated between two parents -- it is in one layout or
    the other, and the change is instant.  So the move is played by a snapshot
    that belongs to neither: the real widget is hidden, its image flies across
    the host, and it reappears where it landed.  Which is also why the pixmap
    is scaled rather than centred -- the card is a different width in its two
    homes, and stretching the image between them is what makes the two look
    like the same object.

    The caller owns the real widget's visibility; ``on_finished`` runs when the
    flight ends, including when it is cut short by a later one.
    """
    ghost = QLabel(host)
    ghost.setPixmap(snapshot)
    ghost.setScaledContents(True)
    # It is a picture, not a control: clicks during the flight belong to
    # whatever is underneath.
    ghost.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    ghost.setGeometry(start)
    ghost.show()
    ghost.raise_()

    animation = QPropertyAnimation(ghost, b"geometry", ghost)
    animation.setDuration(duration)
    animation.setStartValue(start)
    animation.setEndValue(end)
    # Ease at both ends: the card is a thing being moved, not thrown.
    animation.setEasingCurve(QEasingCurve.InOutCubic)

    def on_state(new_state, _old_state) -> None:
        # stateChanged rather than finished: an animation that is stop()ed to
        # make way for a newer one never emits finished, and the widget it was
        # standing in for would stay hidden for good.
        if new_state != QAbstractAnimation.Stopped:
            return
        ghost.hide()
        ghost.deleteLater()
        if on_finished is not None:
            on_finished()

    animation.stateChanged.connect(on_state)
    animation.start()
    return animation
