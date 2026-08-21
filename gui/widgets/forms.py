"""Form primitives shared by every view.

Qt gives you a spin box and a slider; it does not give you the labelled,
hint-carrying, consistently-spaced pairing that a settings-heavy application
needs on several hundred controls.  These wrap that pattern once so the views
stay a description of *what* is configurable rather than of how it is laid out.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from PySide6.QtCore import QEvent, QLocale, QSize, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..i18n import _
from . import icons


def _label(text: str, object_name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(True)
    return label


class Card(QFrame):
    """A titled surface.  The content goes into :attr:`body`."""

    def __init__(self, title: str = "", subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        if title:
            header = QVBoxLayout()
            header.setSpacing(2)
            header.addWidget(_label(title, "CardTitle"))
            if subtitle:
                header.addWidget(_label(subtitle, "CardSubtitle"))
            outer.addLayout(header)

        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body)

    def add(self, *widgets: QWidget) -> None:
        for widget in widgets:
            self.body.addWidget(widget)

    def add_row(self, *widgets: QWidget, stretch: Sequence[int] | None = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        for index, widget in enumerate(widgets):
            row.addWidget(widget, stretch[index] if stretch else 1)
        self.body.addLayout(row)
        return row


class Field(QWidget):
    """Label, optional hint, and the control itself, stacked."""

    def __init__(
        self,
        label: str,
        widget: QWidget,
        hint: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.widget = widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(_label(label, "FieldLabel"))
        layout.addWidget(widget)
        if hint:
            hint_label = _label(hint, "FieldHint")
            layout.addWidget(hint_label)
            # The hint doubles as the tooltip: it is the same sentence, and on a
            # narrow window the inline copy is the first thing to get elided.
            widget.setToolTip(hint)

        # Pin the contents to the top of whatever height the row ends up with.
        # ``Card.add_row`` puts Fields in a QHBoxLayout, which gives every child
        # the full row height; a Field carrying a hint is taller than one
        # without, so the shorter Field's label and control were spread over the
        # extra space and sat lower than its neighbour's.  That is the visible
        # step between "Vocoder" and the control beside it.
        #
        # Skipped when the control wants the space itself -- a log view or a
        # file list is meant to grow, and a stretch here would pin it to its
        # minimum height instead.
        if widget.sizePolicy().verticalPolicy() not in (
            QSizePolicy.Expanding,
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Ignored,
        ):
            layout.addStretch(1)


class Collapsible(QWidget):
    """A disclosure section, the way the Gradio tabs use ``gr.Accordion``.

    Settings-heavy forms need a floor and a ceiling: the controls someone
    touches on every run stay in front of them, and the two dozen that exist
    for one unusual case stay reachable without being in the way.  Collapsed by
    default, because a section that opens itself is not hiding anything.

    Content goes into :attr:`body`.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        expanded: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._expanded = expanded

        self.header = QPushButton()
        self.header.setObjectName("Disclosure")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self._on_clicked)
        self.header.setIconSize(QSize(14, 14))
        self._title = title
        self._subtitle = subtitle
        # A drawn chevron rather than "›"/"⌄" in the label, for the reason the
        # transport controls already gave up on glyph characters: they come
        # from whatever font on the machine carries them, at whatever weight
        # and baseline that font chose, beside an otherwise hand-drawn set.
        self._icon_colour = theme.tokens()["text_dim"]

        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(2, 10, 2, 2)
        self.body.setSpacing(12)
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self._refresh_header()

    def _on_clicked(self) -> None:
        self.set_expanded(self.header.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded

        # Showing or hiding the body relaids out everything below it, and the
        # header text changes at the same time.  Left alone, Qt repaints once
        # per change and the section visibly flashes -- worse inside a scroll
        # area, where the whole viewport is involved.  Freezing updates on the
        # top-level window collapses it into a single repaint.
        window = self.window()
        was_enabled = window.updatesEnabled()
        window.setUpdatesEnabled(False)
        try:
            self.header.setChecked(expanded)
            self.content.setVisible(expanded)
            self._refresh_header()
            # Settle the geometry while painting is still frozen, so the first
            # frame the user sees is the finished layout.
            self.content.updateGeometry()
            layout = window.layout()
            if layout is not None:
                layout.activate()
        finally:
            window.setUpdatesEnabled(was_enabled)

        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def _refresh_header(self) -> None:
        suffix = f"    {self._subtitle}" if self._subtitle and not self._expanded else ""
        self.header.setText(f" {self._title}{suffix}")
        self.header.setIcon(
            icons.icon(
                "chevron" if self._expanded else "chevron_right",
                self._icon_colour,
                size=14,
            )
        )

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """Recolour the drawn chevron; QSS cannot reach a rendered pixmap."""
        self._icon_colour = tokens["text_dim"]
        self._refresh_header()

    def add(self, *widgets: QWidget) -> None:
        for widget in widgets:
            self.body.addWidget(widget)

    def add_row(self, *widgets: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        for widget in widgets:
            row.addWidget(widget, 1)
        self.body.addLayout(row)
        return row

    def add_group(self, title: str) -> None:
        """A small heading inside the section, for its sub-topics."""
        self.body.addWidget(SectionHeader(title))


class SectionHeader(QWidget):
    """A small all-caps rule used to group fields inside a card."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)
        heading = _label(text.upper(), "SectionHeader")
        # A two-word all-caps heading has no business wrapping; the rule beside
        # it should give up its width instead.
        heading.setWordWrap(False)
        layout.addWidget(heading, 0)
        rule = QFrame()
        rule.setObjectName("Separator")
        rule.setFrameShape(QFrame.HLine)
        rule.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(rule, 1)


class SliderSpin(QWidget):
    """A slider and a spin box driving one value.

    Both are load-bearing: the slider is how you explore a parameter, the spin
    box is how you reproduce a setting someone gave you.  Keeping only one of
    them is the usual mistake.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        step: float = 0.01,
        decimals: int = 2,
        value: float | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(round(minimum * self._scale)))
        self.slider.setMaximum(int(round(maximum * self._scale)))
        self.slider.setSingleStep(max(1, int(round(step * self._scale))))
        self.slider.setPageStep(max(1, int(round(step * self._scale * 10))))

        if decimals:
            self.spin: QAbstractSpinBox = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(float(minimum), float(maximum))
            self.spin.setSingleStep(float(step))
        else:
            # QSpinBox is int-typed all the way through; handing it the floats
            # the caller wrote for the slider raises rather than truncating.
            self.spin = QSpinBox()
            self.spin.setRange(int(round(minimum)), int(round(maximum)))
            self.spin.setSingleStep(max(1, int(round(step))))
        self.spin.setFixedWidth(84)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        # Force a decimal point rather than the system separator: every value
        # here is also written into a config file, quoted in a log line or
        # pasted from someone else's settings, and all of those use a dot.
        self.spin.setLocale(QLocale(QLocale.C))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(minimum if value is None else value)

    def _from_slider(self, raw: int) -> None:
        value = raw / self._scale
        if abs(self.spin.value() - value) > 1e-9:
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _from_spin(self, value: float) -> None:
        raw = int(round(value * self._scale))
        if self.slider.value() != raw:
            self.slider.blockSignals(True)
            self.slider.setValue(raw)
            self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float) -> None:  # noqa: N802 - matches Qt naming
        self.spin.setValue(value if self._decimals else int(round(value)))


class SearchableCombo(QWidget):
    """A combo box with type-ahead filtering and a refresh button.

    Model lists routinely run to hundreds of checkpoints; scrolling a plain
    dropdown to find ``_e320_s41600`` is not a workable interaction.
    """

    currentTextChanged = Signal(str)
    refreshRequested = Signal()

    def __init__(self, editable: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.combo = QComboBox()
        self.combo.setEditable(editable)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        if editable:
            completer = self.combo.completer()
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)

        # A drawn icon rather than the "↻" character: the glyph came from
        # whatever font happened to have it, at whatever weight and baseline
        # that font chose, so it sat off-centre and looked nothing like the
        # rest of the icon set.
        self.refresh_button = QPushButton()
        self.refresh_button.setObjectName("IconButton")
        self.refresh_button.setFixedWidth(34)
        self.refresh_button.setIconSize(QSize(15, 15))
        self.refresh_button.setToolTip(_("Rescan"))
        self.refresh_button.clicked.connect(self.refreshRequested)
        self._icon_colour = theme.tokens()["text_dim"]
        self._paint_icon()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.refresh_button, 0)

        self.combo.currentTextChanged.connect(self.currentTextChanged)

    def _paint_icon(self) -> None:
        self.refresh_button.setIcon(
            icons.icon("refresh", self._icon_colour, size=15)
        )

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """Recolour the drawn icon; QSS cannot reach a rendered pixmap."""
        self._icon_colour = tokens["text_dim"]
        self._paint_icon()

    def set_items(self, items: Iterable[str], keep_current: bool = True) -> None:
        """Replace the list, preserving the selection when it still exists."""
        current = self.combo.currentText()
        items = list(items)
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(items)
        if keep_current and current in items:
            self.combo.setCurrentText(current)
        elif items:
            self.combo.setCurrentIndex(0)
        else:
            self.combo.setCurrentText("")
        self.combo.blockSignals(False)
        self.currentTextChanged.emit(self.combo.currentText())

    def set_pairs(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Populate with ``(display, value)`` pairs."""
        current = self.value()
        self.combo.blockSignals(True)
        self.combo.clear()
        for display, value in pairs:
            self.combo.addItem(display, value)
        index = self.combo.findData(current)
        self.combo.setCurrentIndex(max(0, index))
        self.combo.blockSignals(False)

    def value(self) -> str:
        data = self.combo.currentData()
        return data if data is not None else self.combo.currentText()

    def text(self) -> str:
        return self.combo.currentText()

    def set_text(self, text: str) -> None:
        self.combo.setCurrentText(text)


class PathPicker(QWidget):
    """A path line edit with a browse button that also accepts drops."""

    pathChanged = Signal(str)

    def __init__(
        self,
        mode: str = "open",
        filters: str = "All files (*.*)",
        placeholder: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.filters = filters
        self._completer: QCompleter | None = None

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder or "Drop a file here or browse…")
        self.edit.textChanged.connect(self.pathChanged)
        self.edit.installEventFilter(self)

        self.button = QPushButton(_("Browse"))
        self.button.setFixedWidth(84)
        self.button.clicked.connect(self._browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button, 0)

        self.setAcceptDrops(True)

    # -- suggestions -------------------------------------------------------

    def set_suggestions(self, paths: Iterable[str]) -> None:
        """Offer known-good paths from the field itself.

        The Gradio tabs present these as a dropdown; here the equivalent is a
        completer that opens on click rather than only after typing, so the
        files already sitting in the custom-pretraineds or datasets folder are
        discoverable without knowing they are there.  Browsing still works, and
        anything can still be typed -- this only saves the user from having to
        remember a path.
        """
        items = [str(path) for path in paths]
        if not items:
            self.edit.setCompleter(None)
            self._completer = None
            return

        completer = QCompleter(items, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        # Unfiltered so the first click shows everything available; typing then
        # narrows it. The default mode shows nothing until a prefix matches,
        # which for a path means nothing until the user already knows it.
        completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        completer.setMaxVisibleItems(12)
        self.edit.setCompleter(completer)
        self._completer = completer

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        if (
            watched is self.edit
            and self._completer is not None
            and event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn)
            and not self.edit.text().strip()
        ):
            # Only while the field is empty: popping the list over a path the
            # user is editing would hide what they are working on.
            self._completer.complete()
        return super().eventFilter(watched, event)

    def _browse(self) -> None:
        start = self.edit.text() or os.getcwd()
        if self.mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "Select folder", start)
        elif self.mode == "save":
            chosen, _filter = QFileDialog.getSaveFileName(self, "Save as", start, self.filters)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, "Select file", start, self.filters)
        if chosen:
            self.edit.setText(os.path.normpath(chosen))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.edit.setText(os.path.normpath(urls[0].toLocalFile()))
            event.acceptProposedAction()

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(value or "")


class Toggle(QCheckBox):
    """A checkbox that carries its explanation in the tooltip."""

    def __init__(self, text: str, hint: str = "", checked: bool = False, parent=None):
        super().__init__(text, parent)
        self.setChecked(checked)
        if hint:
            self.setToolTip(hint)


def primary_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Primary")
    button.setMinimumHeight(38)
    button.setCursor(Qt.PointingHandCursor)
    return button


def danger_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Danger")
    button.setMinimumHeight(38)
    button.setCursor(Qt.PointingHandCursor)
    return button


def ghost_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Ghost")
    button.setCursor(Qt.PointingHandCursor)
    return button


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    return line


def spacer(height: int = 8) -> QWidget:
    widget = QWidget()
    widget.setFixedHeight(height)
    return widget


def icon_size() -> QSize:
    return QSize(16, 16)
