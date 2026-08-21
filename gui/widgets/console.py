"""Log console for backend output.

Two things make this more than a text box.  First, the backend is a torch
application: it emits ANSI from ``rich``, progress bars that rewrite their line
with ``\\r``, and warnings that arrive faster than a widget can repaint.
Second, a training run produces output for days, so the buffer has to be
bounded and appends have to be batched or the UI thread spends its life in
``appendPlainText``.
"""

from __future__ import annotations

import re
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import monospace_font
from .forms import ghost_button

from ..i18n import _

#: CSI sequences plus the handful of non-CSI escapes rich emits.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
#: Progress bars redraw by returning to column zero; keep only the final state.
_CARRIAGE = re.compile(r"^.*\r(?!\n)", re.MULTILINE)


def clean(text: str) -> str:
    """Strip terminal control codes, keeping the last state of rewritten lines."""
    text = _ANSI.sub("", text)
    text = _CARRIAGE.sub("", text)
    return text.replace("\x00", "")


class LogConsole(QWidget):
    """Bounded, batched, searchable view of the backend's stdout."""

    #: Emitted for each cleaned line, so views can watch for progress markers
    #: without re-parsing the widget's contents.
    lineLogged = Signal(str)

    def __init__(self, max_lines: int = 5000, parent: QWidget | None = None):
        super().__init__(parent)
        self._pending: list[str] = []
        self._autoscroll = True

        self.view = QPlainTextEdit()
        self.view.setObjectName("Console")
        self.view.setReadOnly(True)
        self.view.setFont(monospace_font(9))
        self.view.setMaximumBlockCount(max_lines)
        self.view.setWordWrapMode(QTextOption.NoWrap)
        self.view.setFrameShape(QPlainTextEdit.NoFrame)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(_("Filter…"))
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setFixedWidth(200)
        self.filter_edit.textChanged.connect(self._refilter)

        self.autoscroll_button = ghost_button(_("Follow ✓"))
        self.autoscroll_button.setCheckable(True)
        self.autoscroll_button.setChecked(True)
        self.autoscroll_button.toggled.connect(self._set_autoscroll)

        copy_button = ghost_button(_("Copy"))
        copy_button.clicked.connect(self._copy_all)
        save_button = ghost_button(_("Save…"))
        save_button.clicked.connect(self._save)
        clear_button = ghost_button(_("Clear"))
        clear_button.clicked.connect(self.clear)

        bar = QWidget()
        bar.setObjectName("ConsoleBar")
        # Otherwise the stylesheet's fill is never painted and the bar is a
        # hole in the console; see the note in gui/app.py's Sidebar.
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 6, 8, 6)
        bar_layout.setSpacing(6)
        title = QLabel(_("Backend log"))
        title.setObjectName("CardTitle")
        bar_layout.addWidget(title)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self.filter_edit)
        bar_layout.addWidget(self.autoscroll_button)
        bar_layout.addWidget(copy_button)
        bar_layout.addWidget(save_button)
        bar_layout.addWidget(clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(bar)
        layout.addWidget(self.view, 1)

        # Everything the backend writes lands in a list and is flushed on a
        # timer.  A trainer logging a few hundred lines a second would
        # otherwise repaint the widget a few hundred times a second.
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(80)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

        #: Retained separately from the widget so the filter can be cleared
        #: without having lost the hidden lines.
        self._history: list[str] = []
        self._history_limit = max_lines

    # -- input -------------------------------------------------------------

    def append(self, text: str) -> None:
        for line in clean(text).splitlines() or [""]:
            self._pending.append(line)
            self.lineLogged.emit(line)

    def append_notice(self, text: str) -> None:
        """A GUI-originated message, marked so it is not mistaken for backend output."""
        stamp = time.strftime("%H:%M:%S")
        self._pending.append(f"[{stamp}] ── {text}")

    def clear(self) -> None:
        self._pending.clear()
        self._history.clear()
        self.view.clear()

    # -- internals ---------------------------------------------------------

    def _flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self._history.extend(batch)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]

        needle = self.filter_edit.text().lower()
        if needle:
            batch = [line for line in batch if needle in line.lower()]
            if not batch:
                return

        scrollbar = self.view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.view.appendPlainText("\n".join(batch))
        if self._autoscroll or at_bottom:
            self.view.moveCursor(QTextCursor.End)
            scrollbar.setValue(scrollbar.maximum())

    def _refilter(self, needle: str) -> None:
        needle = needle.lower()
        lines = self._history if not needle else [
            line for line in self._history if needle in line.lower()
        ]
        self.view.setPlainText("\n".join(lines))
        self.view.moveCursor(QTextCursor.End)

    def _set_autoscroll(self, enabled: bool) -> None:
        self._autoscroll = enabled
        self.autoscroll_button.setText("Follow ✓" if enabled else "Follow")
        if enabled:
            self.view.moveCursor(QTextCursor.End)

    def _copy_all(self) -> None:
        self.view.selectAll()
        self.view.copy()
        cursor = self.view.textCursor()
        cursor.clearSelection()
        self.view.setTextCursor(cursor)

    def _save(self) -> None:
        target, _chosen_filter = QFileDialog.getSaveFileName(
            self, "Save log", "shirorvc-log.txt", "Text files (*.txt)"
        )
        if not target:
            return
        try:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._history))
        except OSError as error:
            self.append_notice(_("Could not save the log: {error}").format(error=error))
