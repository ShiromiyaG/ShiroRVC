"""The training monitor: chart, controls and a metric picker.

A run writes on the order of 180 scalar tags, most of them per-dimension
diagnostics.  A flat checkbox list of that is not a picker, it is a haystack,
so the tags are grouped by their prefix, searchable, and fronted by presets
that answer the questions people actually open this panel to ask.

Presentation only: it is handed data and emits signals.  The view owns the
event-file reader.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from .chart import WINDOWS, LiveChart
from .flow import FlowRow
from .forms import SliderSpin, ghost_button

from ..i18n import _

#: Curated views over the tag set, keyed on the tag's leaf name with any
#: rolling-average suffix removed.  Matching is exact rather than by substring:
#: "loss_disc" as a substring also catches loss_disc_real and loss_disc_fake,
#: which silently turns a three-line preset into a five-line one.
#:
#: Only tags the run actually wrote are offered, so a HiFi-GAN run never sees
#: the ChouwaGAN-only prior metrics.
PRESETS: dict[str, list[str]] = {
    "Health": ["loss_spectral", "loss_gen_total", "loss_disc"],
    "Adversarial": ["loss_disc_real", "loss_disc_fake", "loss_adv", "loss_fm"],
    # Whether the adaptive weight is actually being applied. `saturation` below
    # 1 means the ceiling is overruling the balance rule, which is invisible in
    # `adaptive_adv_weight` alone -- it just reads as a flat line.
    "Balance": [
        "adaptive_adv_saturation", "adaptive_adv_weight",
        "adaptive_adv_requested", "adaptive_adv_ceiling", "adv_to_rec_ratio",
    ],
    "Gradients": [
        "grad_norm_g", "grad_norm_d", "grad_norm_d_r1", "grad_clip_hit_rate_g",
    ],
    "Latent": [
        "loss_prior_slow", "loss_prior_fast", "prior_kl_slow", "prior_kl_fast",
        "prior_replacement",
    ],
    "Schedule": ["lr_g", "lr_d", "kl_beta_slow", "kl_beta_fast"],
}


def leaf(tag: str) -> str:
    """The comparable part of a tag: no group prefix, no ``_50`` suffix."""
    return tag.split("/")[-1].removesuffix("_50")

#: The palette carries six distinguishable colours; past that a chart stops
#: being readable regardless of how many the user ticks.
MAX_SERIES = 6

#: Per-dimension and ablation diagnostics: hundreds of tags that belong in
#: TensorBoard, not in a picker meant to be scanned.
NOISY_PREFIXES = ("diag/kl_", "diag/ablation_")


class MetricsPanel(QWidget):
    """Chart plus everything needed to drive it."""

    #: The set of tags the user wants plotted has changed.
    selectionChanged = Signal(list)
    #: The user asked for a fresh scan of the log directory.
    rescanRequested = Signal()
    #: A different run was chosen.  Carries its event directory.
    runChanged = Signal(str)
    #: The panel was folded away, or brought back.
    collapsedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._available: list[str] = []
        #: What the host last handed over, before filtering -- kept so
        #: :attr:`include_noisy` can be flipped without a re-read.
        self._supplied: list[str] = []
        self._checked: list[str] = []
        self._runs: list[tuple[str, str, float]] = []
        self._collapsed = False
        self._include_noisy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addLayout(self._build_header())
        self.chart = LiveChart()
        self.chart.seriesClicked.connect(self._toggle_tag)
        self.controls = self._build_controls()
        self.presets = self._build_presets()
        self.picker = self._build_picker()

        layout.addWidget(self.chart, 5)
        layout.addWidget(self.controls)
        layout.addWidget(self.presets)
        layout.addWidget(self.picker, 3)

        #: Everything the collapse button folds away.  The header stays, so
        #: there is always something left to click to bring it back.
        self._body = [self.chart, self.controls, self.presets, self.picker]

    # -- construction ------------------------------------------------------

    def _build_header(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        heading = QLabel(_("Live metrics"))
        heading.setObjectName("CardTitle")
        self.step_badge = QLabel("—")
        self.step_badge.setObjectName("Badge")
        # Folding the panel away is worth a button on a laptop screen: the
        # chart and its picker are the tallest thing in the window, and the
        # steps beside them are what someone setting up a run is reading.
        self.collapse_button = ghost_button(_("Hide"))
        self.collapse_button.setToolTip(_("Fold the chart away and give the width to the steps."))
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        title_row.addWidget(heading)
        title_row.addWidget(self.step_badge)
        title_row.addStretch(1)
        title_row.addWidget(self.collapse_button)
        column.addLayout(title_row)

        # The run being watched is its own choice, not a side effect of the
        # model name above: people routinely compare a finished run against the
        # one currently training, and want to keep watching one while setting
        # up the next.
        self.run_row = QWidget()
        run_row = QHBoxLayout(self.run_row)
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(6)
        self.run_combo = QComboBox()
        self.run_combo.setToolTip(_("Which training log to follow."))
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        rescan = ghost_button(_("Rescan"))
        rescan.clicked.connect(self.rescanRequested)
        run_row.addWidget(self.run_combo, 1)
        run_row.addWidget(rescan)
        column.addWidget(self.run_row)

        # Kept so callers that only know a run's name still have something to
        # set; the combo carries the visible identity.
        self.run_label = QLabel("")
        self.run_label.hide()
        return column

    def _build_controls(self) -> FlowRow:
        # A wrapping row, not a QHBoxLayout: this is eight controls in a column
        # that is 480 px wide on a laptop, and a box layout that cannot fit its
        # children squeezes them past their minimum until they overlap -- which
        # is what used to land this row on top of the chart's legend.
        row = FlowRow(spacing=6)

        window_label = QLabel(_("Window"))
        window_label.setObjectName("FieldHint")
        row.add(window_label)

        self._window_group = QButtonGroup(self)
        self._window_group.setExclusive(True)
        for index, (label, span) in enumerate(WINDOWS):
            button = QPushButton(label)
            button.setObjectName("Segment")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedWidth(46)
            button.clicked.connect(lambda _checked, s=span: self._set_window(s))
            self._window_group.addButton(button, index)
            row.add(button)

        smoothing_label = QLabel(_("Smoothing"))
        smoothing_label.setObjectName("FieldHint")
        row.add(smoothing_label)

        self.smoothing = SliderSpin(0, 0.99, 0.01, decimals=2, value=0.6)
        self.smoothing.setMinimumWidth(150)
        self.smoothing.setMaximumWidth(210)
        self.smoothing.valueChanged.connect(self._set_smoothing)
        row.add(self.smoothing)

        self.log_button = ghost_button(_("Log"))
        self.log_button.setCheckable(True)
        self.log_button.setToolTip(
            _("Log scale. Worth turning on for losses: the first 2% of a run "
            "covers most of the range, and flattens everything after it.")
        )
        self.log_button.toggled.connect(self._set_log)
        row.add(self.log_button)

        self.raw_button = ghost_button(_("Raw"))
        self.raw_button.setCheckable(True)
        self.raw_button.setChecked(True)
        self.raw_button.setToolTip(_("Show the unsmoothed trace behind each line."))
        self.raw_button.toggled.connect(self._set_raw)
        row.add(self.raw_button)
        return row

    def _build_presets(self) -> FlowRow:
        row = FlowRow(spacing=6)
        self._preset_buttons: dict[str, QPushButton] = {}
        for name in PRESETS:
            button = QPushButton(name)
            button.setObjectName("Chip")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, n=name: self._apply_preset(n))
            button.setEnabled(False)
            self._preset_buttons[name] = button
            row.add(button)

        clear = QPushButton(_("Clear"))
        clear.setObjectName("Chip")
        clear.setCursor(Qt.PointingHandCursor)
        clear.clicked.connect(lambda: self._set_checked([]))
        row.add(clear)
        return row

    def _build_picker(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Search metrics…"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._rebuild_tree)
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.NoSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(14)
        self.tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Low enough that the panel still fits in a 700 px window.  It takes
        # the layout's stretch, so on a normal screen it is far taller than
        # this; the minimum only decides what happens when there is no room.
        self.tree.setMinimumHeight(72)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)

        self.hint = QLabel("")
        self.hint.setObjectName("FieldHint")
        self.hint.setWordWrap(True)
        # Hidden until there is something to say: an empty label still claims a
        # line, and vertical space is what this panel is short of.
        self.hint.hide()
        layout.addWidget(self.hint)
        return holder

    # -- public api --------------------------------------------------------

    def set_runs(self, runs: list[tuple[str, str, str]], select: str | None = None) -> None:
        """Populate the run picker with ``(label, path, age)`` rows.

        Keeps whatever was selected if it is still present, so a rescan while
        watching a run does not yank the chart to a different one.
        """
        self._runs = runs
        current = select or self.current_run()

        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for label, path, age in runs:
            self.run_combo.addItem(f"{label}   ·   {age}" if age else label, path)
        if not runs:
            self.run_combo.addItem("No training logs found", "")
        index = self.run_combo.findData(current) if current else -1
        self.run_combo.setCurrentIndex(index if index >= 0 else 0)
        self.run_combo.blockSignals(False)

        chosen = self.current_run()
        if chosen != current:
            self.runChanged.emit(chosen)

    def current_run(self) -> str:
        data = self.run_combo.currentData()
        return data or ""

    def set_run(self, title: str, subtitle: str = "") -> None:
        self.run_label.setText(title)
        self.step_badge.setText(subtitle or "—")

    def _on_run_selected(self, _index: int) -> None:
        self.runChanged.emit(self.current_run())

    def set_status(self, text: str) -> None:
        self.hint.setText(text)
        self.hint.setVisible(bool(text) and not self._collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold everything but the title row away, or bring it back."""
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        for widget in (self.run_row, *self._body):
            widget.setVisible(not collapsed)
        self.collapse_button.setText("Metrics" if collapsed else "Hide")
        self.collapse_button.setToolTip(
            "Bring the chart back." if collapsed
            else "Fold the chart away and give the width to the steps."
        )
        self.collapsedChanged.emit(collapsed)

    @property
    def include_noisy(self) -> bool:
        """Whether the per-dimension diagnostics are offered.

        Off for the training page, where this panel is a sidebar and the ~180
        ``diag/kl_*`` series would bury the dozen tags worth glancing at.  On
        for the diagnostics page, whose whole purpose is the complete set.
        """
        return self._include_noisy

    @include_noisy.setter
    def include_noisy(self, value: bool) -> None:
        value = bool(value)
        if value == self._include_noisy:
            return
        self._include_noisy = value
        # Re-run the filter over whatever the caller last supplied.
        supplied, self._supplied = self._supplied, []
        self.set_available_tags(supplied)

    def set_run_picker_visible(self, visible: bool) -> None:
        """Hide the built-in run picker for hosts that provide their own."""
        self.run_row.setVisible(bool(visible))

    def set_available_tags(self, tags: list[str]) -> None:
        """Replace the tag list, keeping any current selection that survives."""
        self._supplied = list(tags)
        filtered = [
            tag for tag in tags
            if self._include_noisy or not tag.startswith(NOISY_PREFIXES)
        ]
        if filtered == self._available:
            return
        self._available = filtered

        leaves = {leaf(tag) for tag in filtered}
        for name, wanted in PRESETS.items():
            self._preset_buttons[name].setEnabled(bool(leaves & set(wanted)))

        self._checked = [tag for tag in self._checked if tag in filtered]
        self._rebuild_tree()

        if not self._checked and filtered:
            # Open on something useful rather than an empty chart.
            for name in PRESETS:
                if self._preset_buttons[name].isEnabled():
                    self._apply_preset(name)
                    break

    def update_series(self, tag: str, steps, values) -> None:
        self.chart.set_series(tag, steps, values)

    def checked_tags(self) -> list[str]:
        return list(self._checked)

    def clear(self) -> None:
        self._available = []
        self._supplied = []
        self._checked = []
        self.chart.clear()
        self.tree.clear()
        # Back to the placeholder: this is called when the run changes, and a
        # badge left on the previous run's step is worse than no badge -- it
        # reads as current.
        self.set_run("", "")
        for button in self._preset_buttons.values():
            button.setEnabled(False)

    def apply_theme(self, tokens: dict[str, str]) -> None:
        self.chart.apply_theme(tokens)

    # -- internals ---------------------------------------------------------

    def _set_window(self, span: int | None) -> None:
        self.chart.window_steps = span
        self.chart.update()

    def _set_smoothing(self, value: float) -> None:
        self.chart.smoothing = value
        self.chart.update()

    def _set_log(self, enabled: bool) -> None:
        self.chart.log_scale = enabled
        self.chart.update()

    def _set_raw(self, enabled: bool) -> None:
        self.chart.show_raw = enabled
        self.chart.update()

    def _apply_preset(self, name: str) -> None:
        wanted = PRESETS[name]
        # Ordered by the preset rather than by the tag list, so the colours a
        # given preset assigns stay the same between runs.
        by_leaf = {leaf(tag): tag for tag in self._available}
        chosen = [by_leaf[key] for key in wanted if key in by_leaf]
        self._set_checked(chosen[:MAX_SERIES])

    def _toggle_tag(self, tag: str) -> None:
        """Clicking a legend row removes that series."""
        self._set_checked([name for name in self._checked if name != tag])

    def _set_checked(self, tags: list[str]) -> None:
        self._checked = tags[:MAX_SERIES]
        self._sync_tree_checks()
        for name in self.chart.series_names():
            if name not in self._checked:
                self.chart.remove_series(name)
        self.selectionChanged.emit(list(self._checked))
        self._update_hint()

    def _update_hint(self) -> None:
        if len(self._checked) >= MAX_SERIES:
            self.hint.setText(
                f"{MAX_SERIES} metrics is the limit — untick one, or click a "
                "line in the legend to drop it."
            )
        elif not self._checked:
            self.hint.setText(_("Pick a preset above, or tick metrics below."))
        else:
            self.hint.setText("")

    def _rebuild_tree(self) -> None:
        needle = self.search.text().strip().lower()
        visible = [
            tag for tag in self._available
            if not needle or needle in tag.lower()
        ]

        self.tree.blockSignals(True)
        self.tree.clear()

        groups: dict[str, list[str]] = {}
        for tag in visible:
            group = tag.split("/")[0] if "/" in tag else "other"
            groups.setdefault(group, []).append(tag)

        for group, tags in sorted(groups.items()):
            parent = QTreeWidgetItem([f"{group}  ({len(tags)})"])
            parent.setFlags(Qt.ItemIsEnabled)
            parent.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(parent)
            for tag in sorted(tags):
                child = QTreeWidgetItem([tag.split("/", 1)[-1].removesuffix("_50")])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, tag)
                child.setToolTip(0, tag)
                child.setCheckState(
                    0, Qt.Checked if tag in self._checked else Qt.Unchecked
                )
                parent.addChild(child)
            # Expanded while searching so matches are visible without a click;
            # collapsed otherwise so the list stays scannable.
            parent.setExpanded(bool(needle) or group.startswith("loss"))

        self.tree.blockSignals(False)

    def _sync_tree_checks(self) -> None:
        self.tree.blockSignals(True)
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            tag = item.data(0, Qt.UserRole)
            if tag is not None:
                item.setCheckState(
                    0, Qt.Checked if tag in self._checked else Qt.Unchecked
                )
            iterator += 1
        self.tree.blockSignals(False)

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        tag = item.data(0, Qt.UserRole)
        if tag is None:
            return
        checked = item.checkState(0) == Qt.Checked
        if checked and tag not in self._checked:
            if len(self._checked) >= MAX_SERIES:
                self.tree.blockSignals(True)
                item.setCheckState(0, Qt.Unchecked)
                self.tree.blockSignals(False)
                self._update_hint()
                return
            self._set_checked(self._checked + [tag])
        elif not checked and tag in self._checked:
            self._set_checked([name for name in self._checked if name != tag])
