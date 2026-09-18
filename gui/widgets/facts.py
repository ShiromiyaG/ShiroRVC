"""A labelled summary with callouts under it.

For a handful of facts about one thing that want reading at a glance: which
value, and whether it is a problem.  The experiment config panel used to show
the same facts as a paragraph of bold labels and inline code, where "absent"
and "unrecognised" looked exactly like a healthy value.

Colours come from ``style.qss`` (``#FactsPanel``, ``#Pill``, ``#Notice``) keyed
on a ``tone`` property, so the theme switch reaches them with no code here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme


@dataclass(frozen=True)
class Fact:
    """One row: what it is, its value, and a line of context."""

    label: str
    value: str
    #: ``None`` shows the value as plain text.  A tone -- ``neutral``,
    #: ``accent``, ``success``, ``warning`` or ``danger`` -- shows it as a pill
    #: in that colour.
    tone: str | None = None
    note: str = ""
    #: Monospaced, for identifiers read out of a file.
    code: bool = False


def set_tone(widget: QWidget, tone: str) -> None:
    """Restyle a ``Pill`` or ``Notice`` for ``tone``.

    Qt does not watch dynamic properties for style invalidation, so changing
    one on a widget already shown needs the explicit repolish.
    """
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def notice(text: str, tone: str = "info") -> QLabel:
    """A callout: ``info``, ``success``, ``warning`` or ``danger``."""
    label = QLabel(text)
    label.setObjectName("Notice")
    label.setProperty("tone", tone)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


class FactsPanel(QWidget):
    """Rows of label, value and note in a framed block, callouts underneath."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

    def set_facts(
        self,
        facts: Iterable[Fact],
        notices: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Show ``facts``, then one callout per ``(tone, text)``."""
        frame = QFrame()
        frame.setObjectName("FactsPanel")
        grid = QGridLayout(frame)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(9)
        # The note takes the leftover width, so labels and values line up in
        # two tidy columns however long a note runs.
        grid.setColumnStretch(2, 1)

        for row, fact in enumerate(facts):
            label = QLabel(fact.label)
            label.setObjectName("FactLabel")
            grid.addWidget(label, row, 0, Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(self._value(fact), row, 1, Qt.AlignLeft | Qt.AlignVCenter)
            if fact.note:
                note = QLabel(fact.note)
                note.setObjectName("FactNote")
                note.setWordWrap(True)
                grid.addWidget(note, row, 2, Qt.AlignVCenter)

        self._replace([frame, *(notice(text, tone) for tone, text in notices)])

    def set_message(self, text: str, tone: str = "info") -> None:
        """Nothing to list: a single callout where the facts would be."""
        self._replace([notice(text, tone)])

    def _replace(self, widgets: list[QWidget]) -> None:
        # Hidden before they go: ``deleteLater`` waits for the event loop, and
        # until then the old rows would still be drawn under the new ones.
        while self._layout.count():
            old = self._layout.takeAt(0).widget()
            if old is not None:
                old.hide()
                old.deleteLater()
        for widget in widgets:
            self._layout.addWidget(widget)

    @staticmethod
    def _value(fact: Fact) -> QLabel:
        value = QLabel(fact.value)
        value.setTextFormat(Qt.PlainText)
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if fact.tone is None:
            value.setObjectName("FactValue")
        else:
            value.setObjectName("Pill")
            value.setProperty("tone", fact.tone)
        if fact.code:
            value.setFont(theme.monospace_font(9))
        # A pill hugs its text rather than stretching across the column.
        value.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        return value
