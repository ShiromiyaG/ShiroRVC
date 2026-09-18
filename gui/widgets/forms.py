"""Form primitives shared by every view.

Qt gives you a spin box and a slider; it does not give you the labelled,
hint-carrying, consistently-spaced pairing that a settings-heavy application
needs on several hundred controls.  These wrap that pattern once so the views
stay a description of *what* is configurable rather than of how it is laid out.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Sequence

from PySide6.QtCore import QElapsedTimer, QEvent, QLocale, QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyledItemDelegate,
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
    """A titled surface.  The content goes into :attr:`body`.

    ``icon`` names a glyph from :mod:`icons`, shown in a tinted badge beside
    the title.
    """

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        parent: QWidget | None = None,
        icon: str = "",
    ):
        super().__init__(parent)
        self.setObjectName("Card")
        self._icon = icon
        self._badge: QLabel | None = None
        # No outer margins: the divider under the header runs edge to edge, so
        # the header and the body carry their own.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if title:
            header = QHBoxLayout()
            header.setContentsMargins(18, 14, 18, 14)
            header.setSpacing(12)
            if icon:
                self._badge = QLabel()
                self._badge.setObjectName("CardIcon")
                self._badge.setFixedSize(32, 32)
                self._badge.setAlignment(Qt.AlignCenter)
                header.addWidget(self._badge, 0, Qt.AlignVCenter)
            text = QVBoxLayout()
            text.setSpacing(2)
            text.addWidget(_label(title, "CardTitle"))
            if subtitle:
                text.addWidget(_label(subtitle, "CardSubtitle"))
            header.addLayout(text, 1)
            outer.addLayout(header)
            outer.addWidget(separator())

        self.body = QVBoxLayout()
        self.body.setContentsMargins(18, 16 if title else 18, 18, 18)
        self.body.setSpacing(12)
        outer.addLayout(self.body, 1)
        self._paint_badge(theme.tokens()["accent"])

    def _paint_badge(self, colour: str) -> None:
        if self._badge is not None:
            self._badge.setPixmap(icons.pixmap(self._icon, colour, 18))

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """Recolour the badge glyph; QSS cannot reach a rendered pixmap."""
        self._paint_badge(tokens["accent"])

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
        self.spin.setFixedWidth(72)
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


class _PopupDelegate(QStyledItemDelegate):
    """Popup rows with a tick on the entry whose text is ``current()``.

    The highlight follows the mouse, so without the tick nothing in an open
    list says which entry is the one already chosen.  Matched by text rather
    than row because a completer's rows are a filtered subset.
    """

    def __init__(self, current: Callable[[], str], parent: QWidget):
        super().__init__(parent)
        self._current = current

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        if index.data(Qt.DisplayRole) != self._current():
            return
        colour = option.palette.color(QPalette.Text).name()
        tick = icons.pixmap("check", colour, 14, 2.2)
        # Inside the right padding that ``::item`` reserves for it in the QSS.
        painter.drawPixmap(option.rect.right() - 24, option.rect.center().y() - 7, tick)


#: ``popup.qss`` for the current theme, and the tokens it was built from;
#: see :func:`restyle_popups`.
_popup_sheet = ""
_popup_tokens: dict[str, str] = {}


def _current_popup_sheet() -> str:
    global _popup_sheet
    if not _popup_sheet:
        _popup_sheet = theme.popup_stylesheet(theme.tokens())
    return _popup_sheet


class _PopupBackground(QObject):
    """Paints a completer popup's fill and border, which QSS cannot.

    The completer's list is itself the translucent popup window, and Qt never
    paints a translucent window's styled background.  A dropdown's list sits
    inside a container window instead, so its QSS fill does show.
    """

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        if event.type() == QEvent.Paint:
            tokens = _popup_tokens or theme.tokens()
            painter = QPainter(watched)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(tokens["border_strong"]), 1))
            painter.setBrush(QColor(tokens["surface"]))
            # The radius popup.qss gives the dropdown lists.
            painter.drawRoundedRect(QRectF(watched.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
            painter.end()
        return False


def _style_popup(window: QWidget) -> None:
    """Give a popup window its own sheet, with a rounded border for a shape."""
    window.setWindowFlag(Qt.NoDropShadowWindowHint, True)
    window.setAttribute(Qt.WA_TranslucentBackground)
    window.setStyleSheet(_current_popup_sheet())


def restyle_popups(root: QWidget, tokens: dict[str, str]) -> None:
    """Re-apply ``popup.qss`` to every list popup under ``root``.

    Popups carry their own sheet, so a theme change has to reach them here
    rather than through the window's.
    """
    global _popup_sheet, _popup_tokens
    _popup_sheet = theme.popup_stylesheet(tokens)
    _popup_tokens = dict(tokens)
    for combo in root.findChildren(QComboBox):
        combo.view().window().setStyleSheet(_popup_sheet)
    for edit in root.findChildren(QLineEdit):
        completer = edit.completer()
        if completer is not None:
            completer.popup().setStyleSheet(_popup_sheet)


def polish_combo(combo: QComboBox) -> None:
    """The application's dropdown: a list below the field, ticked on the value.

    Needs ``combobox-popup: 0`` from the QSS as well; the default menu-style
    popup ignores the ``::item`` rules and shows three rows with scroll arrows.
    """
    combo.setItemDelegate(
        _PopupDelegate(lambda: combo.itemText(combo.currentIndex()), combo)
    )
    combo.setMaxVisibleItems(12)
    _style_popup(combo.view().window())


def polish_completer(completer: QCompleter, current: Callable[[], str]) -> None:
    """Style a completer's list like the dropdowns, ticked on ``current()``.

    Its own delegate is a ``QItemDelegate``, which ignores the QSS ``::item``
    rules, so it is replaced.
    """
    popup = completer.popup()
    popup.setItemDelegate(_PopupDelegate(current, popup))
    popup.installEventFilter(_PopupBackground(popup))
    _style_popup(popup)


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
        self._list_closed = QElapsedTimer()
        if editable:
            completer = self.combo.completer()
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            polish_completer(
                completer, lambda: self.combo.itemText(self.combo.currentIndex())
            )
            # A click anywhere on the text opens the list, not only the arrow;
            # typing then narrows the same list.
            self.combo.lineEdit().installEventFilter(self)
            completer.popup().installEventFilter(self)
        polish_combo(self.combo)

        # A drawn icon rather than the "↻" character: the glyph came from
        # whatever font happened to have it, at whatever weight and baseline
        # that font chose, so it sat off-centre and looked nothing like the
        # rest of the icon set.
        self.refresh_button = QPushButton()
        self.refresh_button.setObjectName("IconButton")
        # Square, at the height of the combo beside it.
        self.refresh_button.setFixedSize(34, 34)
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

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        completer = self.combo.completer()
        if completer is None:
            return super().eventFilter(watched, event)
        popup = completer.popup()
        if watched is popup and event.type() == QEvent.Hide:
            self._list_closed.start()
        elif (
            watched is self.combo.lineEdit()
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
            and not popup.isVisible()
            # The click that dismissed the list can land here as well, and
            # reopening on it would make clicking away impossible.
            and not (self._list_closed.isValid() and self._list_closed.elapsed() < 250)
        ):
            self._open_list(completer)
            return True
        return super().eventFilter(watched, event)

    def _open_list(self, completer: QCompleter) -> None:
        """Show every entry, the current one highlighted, text ready to replace."""
        line = self.combo.lineEdit()
        line.setFocus(Qt.MouseFocusReason)
        completer.setCompletionPrefix("")
        completer.complete()
        row = self.combo.currentIndex()
        if row >= 0:
            completer.popup().setCurrentIndex(completer.completionModel().index(row, 0))
        # After the highlight, which rewrites the text: selected, the first
        # keystroke replaces it and starts a search instead of appending.
        line.selectAll()

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
        self.edit.setPlaceholderText(placeholder or _("Drop a file here or browse…"))
        self.edit.textChanged.connect(self.pathChanged)
        self.edit.installEventFilter(self)

        # The suggestions' dropdown arrow, for when the field is not empty and
        # a click no longer opens them.  Shown only while there are any.
        self._icon_colour = theme.tokens()["text_dim"]
        self._list_action = self.edit.addAction(
            icons.icon("chevron", self._icon_colour, size=14), QLineEdit.TrailingPosition
        )
        self._list_action.setToolTip(_("Show the list"))
        self._list_action.setVisible(False)
        self._list_action.triggered.connect(self._open_list)
        self._list_closed = QElapsedTimer()

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
        if self._completer is not None:
            self._completer.deleteLater()
        self._list_action.setVisible(bool(items))
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
        polish_completer(completer, self.edit.text)
        completer.popup().installEventFilter(self)
        self._completer = completer

    def _open_list(self) -> None:
        """Every suggestion, the current path highlighted."""
        # The click that dismissed the list can trigger the arrow as well.
        if self._completer is None or (
            self._list_closed.isValid() and self._list_closed.elapsed() < 250
        ):
            return
        self._completer.setCompletionPrefix("")
        self._completer.complete()
        model = self._completer.completionModel()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if index.data() == self.edit.text():
                self._completer.popup().setCurrentIndex(index)
                break

    def apply_theme(self, tokens: dict[str, str]) -> None:
        """Recolour the drawn arrow; QSS cannot reach an action's icon."""
        self._icon_colour = tokens["text_dim"]
        self._list_action.setIcon(icons.icon("chevron", self._icon_colour, size=14))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        if (
            self._completer is not None
            and watched is self._completer.popup()
            and event.type() == QEvent.Hide
        ):
            self._list_closed.start()
        elif (
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
            chosen = QFileDialog.getExistingDirectory(self, _("Select folder"), start)
        elif self.mode == "save":
            chosen, _filter = QFileDialog.getSaveFileName(self, _("Save as"), start, self.filters)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, _("Select file"), start, self.filters)
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


class FileList(QWidget):
    """Several files of the given suffixes: add, remove, or drop them in."""

    changed = Signal()

    def __init__(
        self,
        suffixes: tuple[str, ...],
        filters: str = "All files (*.*)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._suffixes = tuple(suffix.lower() for suffix in suffixes)
        self.filters = filters

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(96)

        add = QPushButton(_("Add…"))
        add.clicked.connect(self._browse)
        remove = QPushButton(_("Remove"))
        remove.clicked.connect(self._remove_selected)
        buttons = QVBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        self.setAcceptDrops(True)

    def paths(self) -> list[str]:
        return [self.list.item(row).text() for row in range(self.list.count())]

    def add_paths(self, paths: Iterable[str]) -> None:
        present = set(self.paths())
        for path in paths:
            path = os.path.normpath(path)
            if path.lower().endswith(self._suffixes) and path not in present:
                self.list.addItem(path)
                present.add(path)
        self.changed.emit()

    def clear(self) -> None:
        self.list.clear()
        self.changed.emit()

    def _remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.changed.emit()

    def _browse(self) -> None:
        start = os.path.dirname(self.paths()[-1]) if self.list.count() else os.getcwd()
        chosen, _filter = QFileDialog.getOpenFileNames(self, _("Select files"), start, self.filters)
        self.add_paths(chosen)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.add_paths(url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile())
        event.acceptProposedAction()


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
