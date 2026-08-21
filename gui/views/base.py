"""Shared scaffolding for the screens.

Every view is a scrollable page with a heading and a set of cards, and every
view runs backend commands the same way: disable the trigger, show what is
happening, re-enable on either outcome.  Doing that once here is what keeps a
forgotten ``setEnabled(True)`` from leaving a button dead after an error.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..services import engine

from ..i18n import _



class Page(QWidget):
    """Base class for the screens in the sidebar."""

    #: ``(active, description)`` -- drives the status bar.
    busy = Signal(bool, str)
    #: ``(level, message)`` with level in ``info``/``success``/``error``.
    notify = Signal(str, str)
    #: A line the view wants in the console even though the backend did not say it.
    log = Signal(str)

    title = ""
    subtitle = ""

    #: Whether the whole page scrolls as one column.  Views that put a
    #: fixed-height panel beside a long form turn this off: inside a scroll
    #: area every child is stretched to the tallest one, which would give a
    #: chart the height of the form next to it.
    scrollable = True

    #: Widest the form column is allowed to get, in pixels.  Ignored when
    #: :attr:`scrollable` is off, since those pages lay themselves out.
    max_width = 1180

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if self.scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            outer.addWidget(scroll)

            holder = QWidget()
            holder.setObjectName("Root")
            scroll.setWidget(holder)

            # A form is capped and centred rather than filling the window: on a
            # wide monitor an uncapped card turns a 0-1 slider into 1400 px of
            # travel, which is both hard to aim and hard to read the label of.
            column = QWidget()
            column.setMaximumWidth(self.max_width)
            centring = QHBoxLayout(holder)
            centring.setContentsMargins(0, 0, 0, 0)
            # The column takes the stretch so it grows with the window; the
            # maximum width is what stops it, and the side stretches only
            # collect whatever is left over. Giving the column zero stretch
            # instead collapses it to its minimum and centres a narrow strip.
            centring.addStretch(0)
            centring.addWidget(column, 1)
            centring.addStretch(0)
            surface = column
        else:
            holder = QWidget()
            holder.setObjectName("Root")
            outer.addWidget(holder)
            surface = holder

        self.content = QVBoxLayout(surface)
        self.content.setContentsMargins(28, 24, 28, 28)
        self.content.setSpacing(18)

        if self.title:
            heading = QLabel(_(self.title))
            heading.setObjectName("PageTitle")
            self.content.addWidget(heading)
        if self.subtitle:
            sub = QLabel(_(self.subtitle))
            sub.setObjectName("PageSubtitle")
            sub.setWordWrap(True)
            self.content.addWidget(sub)

    @staticmethod
    def scroll_area(inner: QWidget) -> QScrollArea:
        """Wrap one widget in a scroll area styled like the page's own."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(inner)
        return area

    # -- backend -----------------------------------------------------------

    @property
    def engine(self):
        return engine.instance()

    def run(
        self,
        cmd: str,
        args: dict[str, Any],
        *,
        busy_text: str,
        on_result: Callable[[dict], None] | None = None,
        buttons: list[QWidget] | None = None,
        success_text: str | None = None,
    ) -> None:
        """Dispatch a command with the usual busy/enable/report cycle."""
        buttons = buttons or []
        for button in buttons:
            button.setEnabled(False)
        self.busy.emit(True, busy_text)

        def finish() -> None:
            for button in buttons:
                button.setEnabled(True)
            self.busy.emit(False, "")

        def handle_result(data: dict) -> None:
            finish()
            if on_result:
                on_result(data)
            message = success_text or data.get("message")
            if message:
                self.notify.emit("success", str(message))

        def handle_error(error: str) -> None:
            finish()
            self.notify.emit("error", error)

        self.engine.call(cmd, args, on_result=handle_result, on_error=handle_error)

    def require(self, **fields: Any) -> bool:
        """Report the first empty required field.  Returns whether all are set."""
        for name, value in fields.items():
            if value in (None, "", []):
                self.notify.emit(
                    "error", _("{field} is required.").format(field=name)
                )
                return False
        return True

    # -- lifecycle ---------------------------------------------------------

    def on_shown(self) -> None:
        """Called when the view becomes visible.  Refresh lists here."""

    def on_hidden(self) -> None:
        """Called when another view takes over.  Stop timers here.

        A page that keeps polling while off-screen is paying for work nobody
        can see, and on this application that means re-reading an event file
        and repainting a chart every few seconds behind another tab.
        """

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """Hook for custom-painted children that QSS cannot reach.

        Walks the tree instead of relying on each view to forward by hand.  A
        widget shared across every view -- ``SearchableCombo``'s refresh icon is
        the case that forced this -- otherwise needs a line in every override,
        and the one that gets forgotten keeps last theme's colours until the
        window is reopened.  Overrides must call ``super()``.
        """
        for child in self.findChildren(QWidget):
            hook = getattr(child, "apply_theme", None)
            if callable(hook):
                hook(tokens)
