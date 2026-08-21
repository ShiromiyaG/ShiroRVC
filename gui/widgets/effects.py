"""Depth and surfaces: the two things that are not just colour.

``QGraphicsDropShadowEffect`` is not free: a widget carrying one is rendered
into an offscreen pixmap and blurred on every repaint, and Qt disables some
fast paths for its whole subtree.  On a card it is worth it -- that is the
elevation the layout is built around.  On the couple of hundred controls
inside the cards it would be ruinous, so nothing here is applied wholesale.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPaintEvent
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def elevate(
    widget: QWidget,
    *,
    radius: int = 26,
    y_offset: int = 6,
    alpha: int = 90,
    colour: str = "#000000",
) -> QGraphicsDropShadowEffect:
    """Give a widget a soft drop shadow.

    Defaults are tuned for a card on a dark surface: wide and faint, so it
    reads as ambient occlusion rather than as a hard offset.
    """
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(radius)
    shadow.setOffset(0, y_offset)
    tint = QColor(colour)
    tint.setAlpha(alpha)
    shadow.setColor(tint)
    widget.setGraphicsEffect(shadow)
    return shadow


class ChromePanel(QWidget):
    """A surface of the window frame: the sidebar, the status bar.

    Paints its own fill rather than taking one from the stylesheet, because the
    alpha has to change when the compositor backdrop is switched on and off,
    and a stylesheet change means re-polishing every widget in the window --
    0.3 s of frozen window for what should be a repaint.

    Subclasses pick the two palette tokens the fill runs between and which edge
    carries the divider.
    """

    #: Palette tokens for the top and bottom of the fill.  The same token twice
    #: is a flat colour, which is what a 34 px strip wants.
    TOP_TOKEN = "bg_alt"
    BOTTOM_TOKEN = "bg"

    #: Which side gets the 1 px divider: "right", "top" or "" for none.
    DIVIDER_EDGE = ""

    #: Opacity of the fill once a backdrop is behind the window.  Windows' own
    #: navigation panes go fully transparent, but they sit on Mica, which is
    #: barely more than a tint.  Over Acrylic, 11 px labels in ``text_faint``
    #: need something to sit on, and this is the point where the blur is still
    #: unmistakable and the text has not started to fight the wallpaper.
    BACKDROP_ALPHA = 160

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._top = QColor(Qt.transparent)
        self._bottom = QColor(Qt.transparent)
        self._divider = QColor(Qt.transparent)
        self._translucent = False

    def apply_chrome(self, tokens: dict[str, str]) -> None:
        """Take the fill and divider colours from the palette."""
        self._top = QColor(tokens[self.TOP_TOKEN])
        self._bottom = QColor(tokens[self.BOTTOM_TOKEN])
        self._divider = QColor(tokens["border"])
        self.update()

    def set_translucent(self, translucent: bool) -> None:
        """Whether a compositor backdrop is showing behind the window."""
        if translucent == self._translucent:
            return
        self._translucent = translucent
        self.update()

    def _divider_rect(self) -> QRect:
        if self.DIVIDER_EDGE == "right":
            return QRect(self.width() - 1, 0, 1, self.height())
        if self.DIVIDER_EDGE == "top":
            return QRect(0, 0, self.width(), 1)
        return QRect()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        top, bottom = QColor(self._top), QColor(self._bottom)
        if self._translucent:
            top.setAlpha(self.BACKDROP_ALPHA)
            bottom.setAlpha(self.BACKDROP_ALPHA)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, top)
        gradient.setColorAt(1, bottom)

        painter = QPainter(self)
        # Source rather than the default blend: with a backdrop on, these
        # pixels have to *become* semi-transparent.  Blending a translucent
        # fill over what the window painted underneath would only stack the two
        # and end up opaque again.
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(event.rect(), gradient)
        # The divider stays fully opaque either way: a hairline that the
        # backdrop shows through does not read as an edge.
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.fillRect(self._divider_rect(), self._divider)
