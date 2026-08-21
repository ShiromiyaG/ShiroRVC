"""Everything a run wrote, without starting TensorBoard.

The training page already charts a run, but deliberately narrowly: it is a
sidebar to the thing you are doing, so its picker hides the per-dimension
diagnostics (``metrics.NOISY_PREFIXES``) and it shows no media at all.  This
page is the other half -- the full tag set including the ~180 per-dimension
series, and the validation previews and audio that until now existed only
inside the event file.

Media comes from the loose copies on disk rather than from the event file; see
``services.media`` for why.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .base import Page
from ..services import catalog, media
from ..services import runwatch
from ..services.runwatch import ReadSignals
from ..services.tbreader import RunReader
from ..widgets.audio import AudioPlayer
from ..widgets.forms import Card, ghost_button
from ..widgets.gallery import PreviewGallery
from ..widgets.metrics import MetricsPanel

from ..i18n import _, N_


class MonitorPage(Page):
    """Full metrics, previews and audio for any run under ``logs/``."""

    title = N_("Diagnostics")
    subtitle = N_(
        "Every scalar, preview image and audio sample a training run wrote. "
        "Reads the same files TensorBoard would, without starting a server."
    )
    # The chart and the preview both want the window's height; inside a scroll
    # area every child is stretched to the tallest, which would leave the chart
    # as tall as the tag tree beside it.
    scrollable = False

    #: Slower than the training page's 3 s: this page is opened to study a run,
    #: not to watch one tick over.
    POLL_INTERVAL_MS = 5000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._reader: RunReader | None = None
        self._reading = False
        self._pending_initial = False
        self._previews: list[media.EpochPreviews] = []
        self._run_dir = ""

        self._read_signals = ReadSignals()
        self._read_signals.done.connect(self._on_metrics_read)
        self._read_signals.failed.connect(self._on_metrics_failed)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_metrics)

        self.content.addLayout(self._build_run_row())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_metrics_tab(), _("Metrics"))
        self.tabs.addTab(self._build_previews_tab(), _("Previews"))
        self.tabs.addTab(self._build_audio_tab(), _("Audio"))
        self.content.addWidget(self.tabs, 1)

    # -- construction ------------------------------------------------------

    def _build_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.run_picker = QComboBox()
        self.run_picker.setMinimumWidth(280)
        self.run_picker.currentIndexChanged.connect(self._on_run_selected)

        self.refresh_button = ghost_button(_("Rescan"))
        self.refresh_button.clicked.connect(lambda: self.refresh(keep_selection=True))

        self.run_status = QLabel()
        self.run_status.setObjectName("PageSubtitle")
        self.run_status.setWordWrap(True)

        row.addWidget(QLabel(_("Run")))
        row.addWidget(self.run_picker)
        row.addWidget(self.refresh_button)
        row.addWidget(self.run_status, 1)
        return row

    def _build_metrics_tab(self) -> QWidget:
        self.metrics = MetricsPanel()
        # This page has its own picker above the tabs; the panel's built-in one
        # would be a second control for the same choice.
        self.metrics.set_run_picker_visible(False)
        self.metrics.collapse_button.hide()
        self.metrics.selectionChanged.connect(self._on_selection_changed)
        # Everything, including the per-dimension diagnostics the training
        # page filters out -- that is the point of this screen.
        self.metrics.include_noisy = True

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.addWidget(self.metrics)
        # Scrolled rather than squeezed.  The chart's minimum height grows with
        # the legend -- six series in a narrow window is six legend rows -- and
        # a QVBoxLayout that cannot meet its children's minimums overlaps them
        # instead of clipping, which is what drew the controls over the legend.
        # With a scroll area the panel always gets the height it asks for.
        return self.scroll_area(holder)

    def _build_previews_tab(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        self.gallery = PreviewGallery()
        self.gallery.epochChanged.connect(self._show_epoch)
        layout.addWidget(self.gallery, 1)

        self.preview_note = QLabel()
        self.preview_note.setObjectName("PageSubtitle")
        self.preview_note.setWordWrap(True)
        layout.addWidget(self.preview_note)
        return holder

    def _build_audio_tab(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.audio_picker = QComboBox()
        self.audio_picker.setMinimumWidth(220)
        self.audio_picker.currentIndexChanged.connect(self._on_audio_epoch)
        row.addWidget(QLabel(_("Epoch")))
        row.addWidget(self.audio_picker)
        row.addStretch(1)
        layout.addLayout(row)

        # Generated above original: the question this screen answers is "what
        # does it sound like now", and the reference is what you compare to.
        card = Card(_("Validation sample"))
        self.audio_generated = AudioPlayer(_("Generated"))
        self.audio_original = AudioPlayer(_("Original"))
        card.add(self.audio_generated, self.audio_original)
        layout.addWidget(card)

        self.audio_note = QLabel()
        self.audio_note.setObjectName("PageSubtitle")
        self.audio_note.setWordWrap(True)
        layout.addWidget(self.audio_note)
        layout.addStretch(1)
        return holder

    # -- run selection -----------------------------------------------------

    def refresh(self, keep_selection: bool = False) -> None:
        """Rescan ``logs/`` and repopulate the picker."""
        wanted = self._run_dir if keep_selection else None
        runs = catalog.list_runs()

        blocked = self.run_picker.blockSignals(True)
        self.run_picker.clear()
        for label, path, stamp in runs:
            age = catalog.describe_age(stamp)
            self.run_picker.addItem(f"{label}  ·  {age}" if age else label, path)
        self.run_picker.blockSignals(blocked)

        if not runs:
            self._attach("")
            return

        index = 0
        if wanted:
            found = self.run_picker.findData(wanted)
            if found >= 0:
                index = found
        self.run_picker.setCurrentIndex(index)
        self._attach(self.run_picker.itemData(index) or "")

    def _on_run_selected(self, index: int) -> None:
        self._attach(self.run_picker.itemData(index) or "")

    def _attach(self, run_dir: str) -> None:
        self._run_dir = run_dir
        self._load_media(run_dir)

        if not run_dir or not os.path.isdir(run_dir):
            self._reader = None
            self._poll_timer.stop()
            self.metrics.clear()
            self.run_status.setText(
                _("No training logs found under logs/. They appear once a run "
                  "starts writing TensorBoard events.")
            )
            return

        # Shared with the training page's monitor: the same run watched from
        # two screens is parsed once.
        self._reader = runwatch.reader_for(run_dir)
        self.metrics.clear()
        self.metrics.set_status(_("Reading history…"))
        self.run_status.setText("")
        self._poll_metrics(initial=True)
        self._poll_timer.start()

    # -- metrics -----------------------------------------------------------

    def _poll_metrics(self, initial: bool = False) -> None:
        if self._reader is None or self._reading:
            return
        self._reading = True
        self._pending_initial = self._pending_initial or initial
        runwatch.start_poll(self._reader, self._read_signals)

    def _on_metrics_read(self, reader: object, fresh: bool) -> None:
        self._reading = False
        if reader is not self._reader:
            # The run changed while that read was in flight.
            self._pending_initial = False
            self._poll_metrics(initial=True)
            return

        initial, self._pending_initial = self._pending_initial, False
        if not fresh and not initial:
            return

        tags = self._reader.tags()
        self.metrics.set_available_tags(tags)
        self.metrics.set_status(
            _("{tags} metrics · {points} points").format(
                tags=len(tags),
                points=sum(len(v[1]) for v in self._reader.series.values()),
            )
        )
        # The badge beside "Live metrics" is how far the run has got.  Without
        # this it stays on its placeholder dash, which reads as "no data" next
        # to a chart that plainly has some.
        last_step = max(
            (steps[-1] for steps, _ in self._reader.series.values() if steps),
            default=0,
        )
        self.metrics.set_run("", _("step {step}").format(step=f"{last_step:,}"))
        self._push_series()

    def _on_metrics_failed(self, message: str) -> None:
        self._reading = False
        self.metrics.set_status(_("Could not read the event file: {error}").format(error=message))

    def _on_selection_changed(self, _tags: list) -> None:
        self._push_series()

    def _push_series(self) -> None:
        if self._reader is None:
            return
        wanted = set(self.metrics.checked_tags())
        for name in list(self.metrics.chart.series_names()):
            if name not in wanted:
                self.metrics.chart.remove_series(name)
        for tag in wanted:
            entry = self._reader.series.get(tag)
            if entry:
                self.metrics.update_series(tag, entry[0], entry[1])

    # -- media -------------------------------------------------------------

    def _load_media(self, run_dir: str) -> None:
        self._previews = media.list_previews(run_dir)
        self.gallery.set_count(len(self._previews))

        blocked = self.audio_picker.blockSignals(True)
        self.audio_picker.clear()
        for epoch in self._previews:
            if epoch.audio_count:
                self.audio_picker.addItem(epoch.label, epoch.epoch)
        self.audio_picker.blockSignals(blocked)

        if not self._previews:
            self.gallery.image.clear()
            self.gallery.set_caption("")
            self.preview_note.setText(
                _("This run has not written validation previews yet. They appear "
                  "every few hundred steps once training starts.")
            )
            self.audio_generated.clear()
            self.audio_original.clear()
            self.audio_note.setText("")
            return

        size = media.human_bytes(media.total_bytes(self._previews))
        self.preview_note.setText(
            _("{count} epochs with previews · {size} on disk, under "
              "logs/<model>/validation_samples/.").format(
                count=len(self._previews), size=size
            )
        )
        # Land on the newest, which is what someone opening this wants to see.
        self.gallery.set_index(len(self._previews) - 1)
        self._show_epoch(len(self._previews) - 1)
        if self.audio_picker.count():
            self.audio_picker.setCurrentIndex(self.audio_picker.count() - 1)
            self._on_audio_epoch(self.audio_picker.count() - 1)

    def _show_epoch(self, index: int) -> None:
        if not (0 <= index < len(self._previews)):
            return
        epoch = self._previews[index]
        sample = epoch.samples[0] if epoch.samples else None
        self.gallery.image.show_path(sample.mel if sample else None)
        native = self.gallery.image.native_size()
        detail = f" · {native[0]}x{native[1]}" if native else ""
        self.gallery.set_caption(
            _("{label} · {n} sample(s){detail}").format(
                label=epoch.label, n=len(epoch.samples), detail=detail
            )
        )

    def _on_audio_epoch(self, index: int) -> None:
        number = self.audio_picker.itemData(index)
        epoch = next((e for e in self._previews if e.epoch == number), None)
        sample = next((s for s in epoch.samples if s.has_audio), None) if epoch else None
        if sample is None:
            self.audio_generated.clear()
            self.audio_original.clear()
            self.audio_note.setText("")
            return

        if sample.generated:
            self.audio_generated.load(str(sample.generated))
        else:
            self.audio_generated.clear()
        if sample.original:
            self.audio_original.load(str(sample.original))
        else:
            self.audio_original.clear()
        self.audio_note.setText(
            _("{label}, {sample} — the generated clip is the model's own "
              "inference path, so it is what conversion will sound like.").format(
                label=epoch.label, sample=sample.label
            )
        )

    # -- lifecycle ---------------------------------------------------------

    def on_shown(self) -> None:
        self.refresh(keep_selection=bool(self._run_dir))
        if self._reader is not None:
            self._poll_timer.start()

    def on_hidden(self) -> None:
        self._poll_timer.stop()
        # Audio keeps playing behind a hidden page otherwise, which is
        # disorienting when the sound has no visible source.
        self.audio_generated.stop()
        self.audio_original.stop()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        super().apply_theme(tokens)
        for widget in (
            self.metrics, self.audio_generated, self.audio_original,
        ):
            if hasattr(widget, "apply_theme"):
                widget.apply_theme(tokens)
