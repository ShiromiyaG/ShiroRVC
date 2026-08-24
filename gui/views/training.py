"""The training pipeline: preprocess, extract, train, index.

Presented as four ordered steps rather than four independent forms, because
they are strictly sequential and running them out of order is the single most
common way a first training attempt fails.  The live chart on the right reads
the same event files TensorBoard would, so a run can be watched without
starting a second server.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets import QWIDGETSIZE_MAX

from ..i18n import _, N_


# Shared with the diagnostics page, which reads the same event files.
from ..services import runwatch
from ..services.runwatch import ReadSignals as _ReadSignals

from ..services import catalog, prefs
from ..services.tbreader import RunReader
from ..widgets.metrics import MetricsPanel
from ..widgets.progress import TrainingProgress
from ..widgets.forms import (
    Card,
    Collapsible,
    Field,
    PathPicker,
    SearchableCombo,
    SectionHeader,
    SliderSpin,
    Toggle,
    danger_button,
    ghost_button,
    primary_button,
)
from .base import Page


class TrainingPage(Page):
    title = N_("Training")
    subtitle = N_("Prepare a dataset and train a voice, one step at a time.")

    #: The progress card has something to show, or has finished showing it.
    #: The window uses this to float the card over whatever page the user
    #: wanders off to -- which outlasts the run itself, by the few seconds the
    #: finished card stays up.
    progressActive = Signal(bool)

    #: How long a finished run stays on screen before the card packs itself
    #: away.  Long enough to read the final epoch count and elapsed time;
    #: short enough that the panel is not left holding a stale run.
    FINISHED_HOLD_MS = 8000

    # The steps column scrolls on its own; the monitor beside it stays pinned
    # to the window so the chart keeps a sane height.
    scrollable = False

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._reader: RunReader | None = None
        #: A background read is in flight; the poll timer skips its tick rather
        #: than queueing a second read of the same file.
        self._reading = False
        self._pending_initial = False
        # Owned by the page, and outlived by any read still in flight when the
        # page goes: the job checks for that rather than emitting into a
        # deleted object.
        self._read_signals = _ReadSignals(self)
        self._read_signals.done.connect(self._on_metrics_read)
        self._read_signals.failed.connect(self._on_metrics_read_failed)
        self._training = False
        #: Whether a run has ever been started in this session.  The progress
        #: card stays up after one finishes -- the final counters are worth
        #: reading -- but there is nothing to show before the first.
        self._progress_used = False
        #: Set once the user picks a run by hand, after which the model-name
        #: field stops steering the monitor.
        self._run_locked = False

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self.scroll_area(self._build_steps()))
        self._splitter.addWidget(self._build_monitor())
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([680, 480])
        self.content.addWidget(self._splitter, 1)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(3000)
        self._poll_timer.timeout.connect(self._poll_metrics)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._dismiss_progress)

        self.engine.log.connect(self._on_engine_log)

        # Both of these fan out into widgets built by later cards -- the
        # vocoder drives the preprocess controls, and populating the model list
        # attaches the metrics reader -- so they run once everything exists.
        self._on_vocoder_changed()
        self._refresh_models()
        self._refresh_runs()
        if prefs.get("metrics_collapsed", False):
            self.metrics.set_collapsed(True)

    # -- step column -------------------------------------------------------

    def _build_steps(self) -> QWidget:
        holder = QWidget()
        holder.setObjectName("Root")
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 14, 4)
        layout.setSpacing(18)

        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_preprocess_card())
        layout.addWidget(self._build_extract_card())
        layout.addWidget(self._build_train_card())
        layout.addWidget(self._build_index_card())
        layout.addStretch(1)
        return holder

    def _build_model_card(self) -> Card:
        card = Card(_("1 · Model"), _("Names the run and decides the architecture."))

        self.model_name = SearchableCombo()
        self.model_name.refreshRequested.connect(self._refresh_models)
        self.model_name.currentTextChanged.connect(self._on_model_changed)

        self.vocoder = SearchableCombo(editable=False)
        self.vocoder.refresh_button.hide()
        self.vocoder.set_pairs(catalog.vocoders())
        self.vocoder.combo.currentIndexChanged.connect(self._on_vocoder_changed)

        self.sample_rate = SearchableCombo(editable=False)
        self.sample_rate.refresh_button.hide()

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Model name"), self.model_name, _("Creates logs/<name>/. Reuse a name to continue a run.")), 2)
        row.addWidget(Field(_("Vocoder"), self.vocoder, _("ChouwaGAN is 44.1 kHz only.")), 1)
        row.addWidget(Field(_("Sample rate"), self.sample_rate, ""), 1)
        card.body.addLayout(row)

        return card

    def _build_preprocess_card(self) -> Card:
        card = Card(_("2 · Preprocess"), _("Slice and normalise the dataset."))

        self.dataset = PathPicker(mode="dir", placeholder=_("Folder with your training audio"))
        card.add(Field(_("Dataset folder"), self.dataset, _("Every audio file below this folder is used.")))

        self.cut_preprocess = SearchableCombo(editable=False)
        self.cut_preprocess.refresh_button.hide()
        self.cut_preprocess.set_items(catalog.CUT_PREPROCESS)
        self.cut_preprocess.set_text("Simple")

        self.chunk_len = SliderSpin(0.5, 5.0, 0.1, decimals=1, value=3.0)
        self.overlap_len = SliderSpin(0.0, 0.4, 0.01, decimals=2, value=0.36)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Slicing"), self.cut_preprocess, _("Automatic detects silence; Simple cuts on a fixed grid.")))
        row.addWidget(Field(_("Chunk length (s)"), self.chunk_len, ""))
        row.addWidget(Field(_("Overlap (s)"), self.overlap_len, ""))
        card.body.addLayout(row)

        advanced = self.preprocess_advanced = Collapsible(
            _("Advanced settings"), _("normalisation, cleanup, format, threads")
        )
        card.add(advanced)

        self.normalization = SearchableCombo(editable=False)
        self.normalization.refresh_button.hide()
        self.normalization.set_items(catalog.NORMALIZATION_MODES)
        self.normalization.set_text("post_peak")

        self.rms_db = SliderSpin(-32, -6, 0.5, decimals=1, value=-18.0)

        self.resampling = SearchableCombo(editable=False)
        self.resampling.refresh_button.hide()
        self.resampling.set_items(catalog.LOADING_RESAMPLING)

        advanced.add_row(
            Field(_("Normalisation"), self.normalization, ""),
            Field(_("Target RMS (dB)"), self.rms_db, _("Only used by the RMS modes.")),
            Field(_("Resampler"), self.resampling, ""),
        )

        self.process_effects = Toggle(_("Trim and de-click"), _("Standard cleanup pass on each slice."), checked=True)
        self.noise_reduction = Toggle(_("Noise reduction"), _("Only worth it on genuinely noisy source material."))
        self.clean_strength = SliderSpin(0, 1, 0.01, decimals=2, value=0.5)
        self.smart_cutter = Toggle(_("SmartCutter"), _("Model-guided slicing. Not available at 44.1 kHz."))
        advanced.add(self.process_effects, self.noise_reduction,
                     Field(_("Noise reduction strength"), self.clean_strength, ""),
                     self.smart_cutter)

        self.dataset_format = SearchableCombo(editable=False)
        self.dataset_format.refresh_button.hide()
        self.dataset_format.set_items(catalog.DATASET_FORMATS)
        self.preprocess_threads = SliderSpin(1, max(1, os.cpu_count() or 8), 1, decimals=0,
                                             value=max(1, (os.cpu_count() or 8) // 2))
        advanced.add_row(
            Field(_("Dataset format"), self.dataset_format, ""),
            Field(_("CPU threads"), self.preprocess_threads, ""),
        )

        self.preprocess_button = primary_button(_("Run preprocessing"))
        self.preprocess_button.clicked.connect(self._preprocess)
        card.add(self.preprocess_button)

        self.noise_reduction.toggled.connect(self.clean_strength.setEnabled)
        self.clean_strength.setEnabled(False)
        return card

    def _build_extract_card(self) -> Card:
        card = Card(_("3 · Extract features"), _("Pitch curves and content embeddings."))

        self.extract_f0 = SearchableCombo(editable=False)
        self.extract_f0.refresh_button.hide()
        self.extract_f0.set_items(catalog.F0_METHODS)

        self.extract_embedder = SearchableCombo(editable=False)
        self.extract_embedder.refresh_button.hide()
        self.extract_embedder.set_items(catalog.EMBEDDER_MODELS)

        self.extract_custom_embedder = SearchableCombo()
        self.extract_custom_embedder.set_items(catalog.list_custom_embedders())
        self.extract_custom_embedder.refreshRequested.connect(
            lambda: self.extract_custom_embedder.set_items(catalog.list_custom_embedders())
        )
        self.extract_custom_field = Field(_("Custom embedder"), self.extract_custom_embedder, "")
        self.extract_custom_field.hide()
        self.extract_embedder.currentTextChanged.connect(
            lambda value: self.extract_custom_field.setVisible(value == "custom")
        )

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Pitch algorithm"), self.extract_f0, _("Must match what you will use at inference time.")))
        row.addWidget(Field(_("Embedder"), self.extract_embedder, ""))
        card.body.addLayout(row)
        card.add(self.extract_custom_field)

        self.extract_gpu = SearchableCombo(editable=False)
        self.extract_gpu.refresh_button.hide()
        self.extract_gpu.set_items(["0"])
        card.add(Field(_("GPU"), self.extract_gpu, _("Comma-separated for multi-GPU.")))

        advanced = self.extract_advanced = Collapsible(
            _("Advanced settings"), _("threads, mute samples, precision, cleanup")
        )
        card.add(advanced)

        self.extract_threads = SliderSpin(1, max(1, os.cpu_count() or 8), 1, decimals=0,
                                          value=max(1, (os.cpu_count() or 8) // 2))
        self.include_mutes = SliderSpin(0, 10, 1, decimals=0, value=2)
        advanced.add_row(
            Field(_("CPU threads"), self.extract_threads, ""),
            Field(_("Mute samples"), self.include_mutes, _("Silent references that teach the model to stay quiet.")),
        )

        self.feature_precision = SearchableCombo(editable=False)
        self.feature_precision.refresh_button.hide()
        self.feature_precision.set_items(catalog.FEATURE_PRECISIONS)
        advanced.add(
            Field(
                _("Feature precision"),
                self.feature_precision,
                _("fp32 doubles the feature cache on disk but keeps the retrieval "
                "index free of a quantisation floor; fp16 halves it. Either is "
                "readable without re-extracting."),
            )
        )

        self.remove_16k = Toggle(_("Delete 16 kHz slices afterwards"), _("Frees disk once the features exist."))
        advanced.add(self.remove_16k)

        self.extract_button = primary_button(_("Run extraction"))
        self.extract_button.clicked.connect(self._extract)
        card.add(self.extract_button)
        return card

    def _build_train_card(self) -> Card:
        card = Card(_("4 · Train"), _("The long part."))

        self.total_epochs = SliderSpin(1, 10000, 1, decimals=0, value=500)
        self.batch_size = SliderSpin(1, 64, 1, decimals=0, value=8)
        self.save_every = SliderSpin(1, 100, 1, decimals=0, value=1)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Total epochs"), self.total_epochs, ""))
        row.addWidget(Field(_("Batch size"), self.batch_size, _("Raise until VRAM is nearly full, then stop.")))
        row.addWidget(Field(_("Save every N epochs"), self.save_every, ""))
        card.body.addLayout(row)

        self.train_gpu = SearchableCombo(editable=False)
        self.train_gpu.refresh_button.hide()
        self.train_gpu.set_items(["0"])
        card.add(Field(_("GPU"), self.train_gpu, _("Comma-separated for multi-GPU.")))

        self.pretrained = Toggle(
            _("Start from a pretrained model"),
            _("Off means training from scratch, which needs far more data."),
            checked=True,
        )
        card.add(self.pretrained)

        advanced = self.train_advanced = Collapsible(
            _("Advanced settings"),
            _("checkpoints, schedule, precision, compilation, gradient clipping"),
        )
        card.add(advanced)

        # -- optimiser ------------------------------------------------------
        # Advanced rather than on the main card: AdamW is right for almost
        # every run, and the choice sat beside the GPU picker as though it were
        # an equally routine decision.
        advanced.add_group("Optimizer")
        self.optimizer = SearchableCombo(editable=False)
        self.optimizer.refresh_button.hide()
        self.optimizer.set_items(catalog.OPTIMIZERS)
        advanced.add(Field(
            _("Optimizer"), self.optimizer,
            _("AdamW unless you have a reason. The others change the schedule too."),
        ))

        # -- pretrained -----------------------------------------------------
        advanced.add_group("Pretrained")
        self.custom_pretrained = Toggle(_("Use my own pretrained files"), "")
        self.pretrained_g = PathPicker(filters="Generator (*.pth)")
        self.pretrained_d = PathPicker(filters="Discriminator (*.pth)")
        advanced.add(
            self.custom_pretrained,
            Field(
                _("Generator (G)"), self.pretrained_g,
                _("Click the field to pick one already in the custom pretraineds folder."),
            ),
            Field(_("Discriminator (D)"), self.pretrained_d, ""),
        )

        # -- checkpoints ----------------------------------------------------
        advanced.add_group("Checkpoints")
        self.save_latest_only = Toggle(_("Keep only the latest G/D"), _("Saves a lot of disk on long runs."), checked=True)
        self.save_weights = Toggle(_("Also export inference weights each save"), "", checked=True)
        self.cleanup = Toggle(_("Discard previous training state"), _("Restarts the run instead of resuming it."))
        advanced.add(self.save_latest_only, self.save_weights, self.cleanup)

        # -- which weights you end up with -----------------------------------
        advanced.add_group("Model selection")
        self.use_ema = Toggle(
            _("Average the weights (EMA)"),
            _("A GAN generator oscillates against its discriminator; the average is usually "
            "better than any single step. Costs one extra copy of it in VRAM."),
            checked=True,
        )
        self.overtrain_detector = Toggle(
            _("Detect overtraining"),
            _("Holds whole source recordings out of training and scores them as it goes. "
            "Training loss cannot see overtraining. Exports the last good weights."),
        )
        self.stop_on_overtrain = Toggle(
            _("Stop once overtrained"),
            _("The pre-overtrain model is exported either way; this only ends the run."),
        )
        advanced.add(
            self.use_ema,
            self.overtrain_detector,
            self.stop_on_overtrain,
        )
        # Nothing to stop on without the signal that produces it, so the
        # control goes away rather than sitting there greyed out.
        self.overtrain_detector.toggled.connect(self.stop_on_overtrain.setVisible)
        self.stop_on_overtrain.setVisible(self.overtrain_detector.isChecked())

        # -- schedule -------------------------------------------------------
        advanced.add_group("Schedule")
        self.use_warmup = Toggle(_("Warm up the learning rate"), "")
        self.warmup_duration = SliderSpin(1, 100, 1, decimals=0, value=5)
        self.lr_scheduler = SearchableCombo(editable=False)
        self.lr_scheduler.refresh_button.hide()
        self.lr_scheduler.set_items(catalog.LR_SCHEDULERS)
        self.lr_scheduler.set_text("exp decay epoch")

        advanced.add(self.use_warmup, Field(_("Warmup epochs"), self.warmup_duration, ""))
        advanced.add(Field(_("LR scheduler"), self.lr_scheduler, ""))

        self.use_custom_lr = Toggle(_("Override the learning rates"), "")
        self.custom_lr_g = SliderSpin(0.00001, 0.001, 0.00001, decimals=5, value=0.0001)
        self.custom_lr_d = SliderSpin(0.00001, 0.001, 0.00001, decimals=5, value=0.0001)
        advanced.add(self.use_custom_lr)
        advanced.add_row(
            Field(_("Generator LR"), self.custom_lr_g, ""),
            Field(_("Discriminator LR"), self.custom_lr_d, ""),
        )

        # -- performance ----------------------------------------------------
        advanced.add_group("Performance")
        self.checkpointing = Toggle(_("Gradient checkpointing"), _("Trades speed for a much smaller VRAM footprint."))
        # Left off and disabled until the backend reports what the card is;
        # ``_apply_tf32_support`` turns it on when the hardware has the units.
        # Ticking it by default on a card without them would send a flag that
        # does nothing and read as though it were doing something.
        self.tf32 = Toggle(_("TF32 matmuls"), _("Faster on Ampere and newer, marginally less precise."))
        self.tf32.setEnabled(False)
        self.benchmark = Toggle(_("cuDNN benchmark"), _("Faster once shapes settle."), checked=True)
        self.compile_vocoder = Toggle(
            _("torch.compile the vocoder"), _("Slow first epoch, faster afterwards.")
        )

        self.torch_compile_mode = SearchableCombo(editable=False)
        self.torch_compile_mode.refresh_button.hide()
        self.torch_compile_mode.set_items(catalog.TORCH_COMPILE_MODES)
        self.torch_compile_mode_field = Field(
            _("Compile mode"), self.torch_compile_mode,
            _("max-autotune spends longer compiling for a faster steady state."),
        )
        # Revealed by the toggle above, matching the Gradio tab: the mode is
        # meaningless while compilation is off.
        self.torch_compile_mode_field.setVisible(False)
        self.compile_vocoder.toggled.connect(self.torch_compile_mode_field.setVisible)

        advanced.add(
            self.checkpointing, self.tf32, self.benchmark,
            self.compile_vocoder, self.torch_compile_mode_field,
        )

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.train_button = primary_button(_("Start training"))
        self.train_button.clicked.connect(self._start_training)
        # The stop button takes the start button's place rather than sitting
        # beside it: only one of the two is ever the thing to press, and a
        # permanently greyed-out Stop is just noise on the card.
        self.stop_button = danger_button(_("Stop training"))
        self.stop_button.clicked.connect(self._stop_training)
        buttons.addWidget(self.train_button, 1)
        buttons.addWidget(self.stop_button, 1)
        card.body.addLayout(buttons)
        self.stop_button.setVisible(False)

        for toggle, dependants in (
            (self.custom_pretrained, [self.pretrained_g, self.pretrained_d]),
            (self.use_warmup, [self.warmup_duration]),
            (self.use_custom_lr, [self.custom_lr_g, self.custom_lr_d]),
        ):
            toggle.toggled.connect(
                lambda checked, widgets=dependants: [w.setEnabled(checked) for w in widgets]
            )
            for widget in dependants:
                widget.setEnabled(toggle.isChecked())
        return card

    def _build_index_card(self) -> Card:
        card = Card(_("5 · Index"), _("Builds the retrieval index used at inference time."))
        self.index_algorithm = SearchableCombo(editable=False)
        self.index_algorithm.refresh_button.hide()
        self.index_algorithm.set_items(catalog.INDEX_ALGORITHMS)
        self.index_metric = SearchableCombo(editable=False)
        self.index_metric.refresh_button.hide()
        self.index_metric.set_items(catalog.INDEX_METRICS)
        card.add_row(
            Field(_("Algorithm"), self.index_algorithm, _("Auto picks by dataset size.")),
            Field(
                _("Similarity"),
                self.index_metric,
                _("l2 is what upstream RVC builds. Cosine compares direction only, "
                "so a quiet and a loud take of the same sound match equally."),
            ),
        )
        self.index_button = primary_button(_("Generate index"))
        self.index_button.clicked.connect(self._build_index)
        card.add(self.index_button)
        return card

    # -- monitor column ----------------------------------------------------

    def _build_monitor(self) -> QWidget:
        holder = self._monitor = QWidget()
        self._monitor_layout = QVBoxLayout(holder)
        self._monitor_layout.setContentsMargins(12, 0, 0, 0)
        self._monitor_layout.setSpacing(12)

        # Only present while a run is going: an empty progress card sitting
        # above the chart the rest of the time is furniture.
        self.progress = TrainingProgress()
        self.progress.hide()
        self.progress.stopRequested.connect(self._stop_training)
        self._monitor_layout.addWidget(self.progress, 0)

        self._metrics_card = Card()
        self.metrics = MetricsPanel()
        self.metrics.rescanRequested.connect(self._refresh_runs)
        self.metrics.runChanged.connect(self._on_run_changed)
        self.metrics.selectionChanged.connect(self._on_metric_selection)
        self.metrics.collapsedChanged.connect(self._on_metrics_collapsed)
        # Scrolled, so that a window too short for the whole panel scrolls it
        # rather than overlapping its rows -- which is what a box layout does
        # when it cannot meet the minimums.  With room to spare the scroll area
        # stretches its widget instead, so on a normal screen this is invisible.
        self._metrics_scroll = self.scroll_area(self.metrics)
        self._metrics_card.body.addWidget(self._metrics_scroll, 1)

        self._monitor_layout.addWidget(self._metrics_card, 1)
        # Collects the space when the card is folded away, so a collapsed
        # panel is a strip at the top rather than a stretched empty card.
        self._monitor_layout.addStretch(0)
        return holder

    #: Width the monitor column keeps when the chart is folded away: enough for
    #: the title, the step badge and the button that brings it back.
    COLLAPSED_MONITOR_WIDTH = 280

    def _on_metrics_collapsed(self, collapsed: bool) -> None:
        """Give the width to the steps while the chart is folded away."""
        # The card stops stretching, so a collapsed panel is a strip at the top
        # of the column rather than a title floating in an empty card.
        self._monitor_layout.setStretch(1, 0 if collapsed else 1)
        self._monitor_layout.setStretch(2, 1 if collapsed else 0)

        total = sum(self._splitter.sizes()) or 1160
        if collapsed:
            self._expanded_sizes = self._splitter.sizes()
            # A maximum, not just smaller sizes: QSplitter will not shrink a
            # pane below its minimum size hint, and the chart's is wide.
            self._monitor.setMaximumWidth(self.COLLAPSED_MONITOR_WIDTH)
            self._splitter.setSizes(
                [total - self.COLLAPSED_MONITOR_WIDTH, self.COLLAPSED_MONITOR_WIDTH]
            )
        else:
            self._monitor.setMaximumWidth(QWIDGETSIZE_MAX)
            self._splitter.setSizes(getattr(self, "_expanded_sizes", None)
                                    or [int(total * 0.6), int(total * 0.4)])
        prefs.set("metrics_collapsed", collapsed)

    @property
    def has_progress(self) -> bool:
        """Whether the progress card has anything on it worth moving about."""
        return self._progress_used

    def reclaim_progress(self) -> None:
        """Take the progress card back from the window's floating dock."""
        if self.progress.parentWidget() is not self._monitor_layout.parentWidget():
            self._monitor_layout.insertWidget(0, self.progress)
        self.progress.setVisible(self._progress_used)

    def _on_engine_log(self, line: str) -> None:
        """Watch the backend's output for the trainer's progress lines."""
        if self._training:
            self.progress.consume(line)

    # -- reactions ---------------------------------------------------------

    def _refresh_models(self) -> None:
        self.model_name.set_items(catalog.list_training_models())

    def _on_vocoder_changed(self) -> None:
        vocoder = self.vocoder.value()
        rates = catalog.sample_rates_for(vocoder)
        self.sample_rate.set_items([str(rate) for rate in rates])
        supported = catalog.supports_smartcutter(vocoder)
        self.smart_cutter.setEnabled(supported)
        if not supported:
            self.smart_cutter.setChecked(False)
            self.smart_cutter.setToolTip(
                f"SmartCutter has no {vocoder} configuration; the backend refuses this combination."
            )

    def _on_model_changed(self, name: str) -> None:
        # Follow the model being configured, but only until the user picks a
        # run explicitly -- after that, changing the model name here must not
        # yank the chart away from what they chose to watch.
        if self._run_locked:
            self._refresh_runs(select=None)
            return
        run_dir = catalog.latest_run_dir(name.strip()) if name.strip() else None
        self._refresh_runs(select=str(run_dir) if run_dir else None)

    # -- metrics -----------------------------------------------------------

    def _refresh_runs(self, select: str | None = None) -> None:
        """Rescan logs/ for runs and hand the list to the picker."""
        runs = [
            (label, path, catalog.describe_age(stamp))
            for label, path, stamp in catalog.list_runs()
        ]
        self.metrics.set_runs(runs, select=select)
        # set_runs only signals when the selection actually moves; attach here
        # so a rescan that keeps the same run still refreshes its data.
        self._attach_reader(self.metrics.current_run())

    def _on_run_changed(self, run_dir: str) -> None:
        self._run_locked = True
        self._attach_reader(run_dir)

    @staticmethod
    def _same_run(reader: RunReader | None, run_dir: str) -> bool:
        if reader is None or not run_dir:
            return False
        return os.path.normcase(os.path.abspath(reader.run_dir)) == os.path.normcase(
            os.path.abspath(run_dir)
        )

    def _attach_reader(self, run_dir: str) -> None:
        # Keep the reader when the run has not changed.  It tails the event
        # file from where it left off, so a poll costs a millisecond; building
        # a new one re-reads the file from the start, which on a 36 MB pretrain
        # is half a second.  This is called on every show of this page and on
        # every keystroke in the model name, so throwing the reader away was
        # most of what made switching to this tab feel like a hang.
        if self._same_run(self._reader, run_dir):
            self._poll_timer.start()
            self._poll_metrics()
            return

        self.metrics.clear()

        if not run_dir or not os.path.isdir(run_dir):
            self._reader = None
            self._poll_timer.stop()
            self.metrics.set_status(
                "No training logs found under logs/. They appear once a run "
                "starts writing TensorBoard events."
            )
            return

        # Shared with the diagnostics page, so a run open on both is parsed once.
        self._reader = runwatch.reader_for(run_dir)
        self.metrics.set_status("Reading history…")
        self._poll_metrics(initial=True)
        self._poll_timer.start()

    def _poll_metrics(self, initial: bool = False) -> None:
        """Ask for new events.  The reading itself happens off this thread."""
        if self._reader is None or self._reading:
            return
        self._reading = True
        self._pending_initial = self._pending_initial or initial
        runwatch.start_poll(self._reader, self._read_signals)

    def _on_metrics_read(self, reader: object, fresh: bool) -> None:
        self._reading = False
        if reader is not self._reader:
            # The run was switched while that read was in flight; its data
            # belongs to a chart that is no longer on screen.
            self._pending_initial = False
            self._poll_metrics(initial=True)
            return

        initial, self._pending_initial = self._pending_initial, False
        if not fresh and not initial:
            return

        self.metrics.set_status(
            "" if self._reader.series
            else "This run has not written any scalar events yet."
        )
        self.metrics.set_available_tags(self._reader.tags())
        self._push_series(self.metrics.checked_tags())

        last_step = max(
            (steps[-1] for steps, _ in self._reader.series.values() if steps), default=0
        )
        self.metrics.set_run("", f"step {last_step:,}")

    def _on_metrics_read_failed(self, error: str) -> None:
        self._reading = False
        self._pending_initial = False
        self.metrics.set_status(f"Could not read the event file: {error}")

    def _on_metric_selection(self, tags: list[str]) -> None:
        self._push_series(tags)

    def _push_series(self, tags: list[str]) -> None:
        if self._reader is None:
            return
        for tag in tags:
            steps, values = self._reader.series.get(tag, ([], []))
            if values:
                self.metrics.update_series(tag, steps, values)

    # -- actions -----------------------------------------------------------

    def _preprocess(self) -> None:
        if not self.require(**{
            "A model name": self.model_name.text().strip(),
            "A dataset folder": self.dataset.path(),
        }):
            return
        if not os.path.isdir(self.dataset.path()):
            self.notify.emit("error", _("The dataset folder does not exist."))
            return

        self.run(
            "preprocess",
            {
                "model_name": self.model_name.text().strip(),
                "dataset_path": self.dataset.path(),
                "sample_rate": int(self.sample_rate.text()),
                "cpu_threads": int(self.preprocess_threads.value()),
                "cut_preprocess": self.cut_preprocess.text(),
                "process_effects": self.process_effects.isChecked(),
                "noise_reduction": self.noise_reduction.isChecked(),
                "clean_strength": self.clean_strength.value(),
                "chunk_len": self.chunk_len.value(),
                "overlap_len": self.overlap_len.value(),
                "normalization_mode": self.normalization.text(),
                "loading_resampling": self.resampling.text(),
                "use_smart_cutter": self.smart_cutter.isChecked(),
                "dataset_format": self.dataset_format.text(),
                "rms_norm_db": self.rms_db.value(),
            },
            busy_text=_("Preprocessing dataset…"),
            buttons=[self.preprocess_button],
        )

    def _extract(self) -> None:
        if not self.require(**{"A model name": self.model_name.text().strip()}):
            return
        self.run(
            "extract",
            {
                "model_name": self.model_name.text().strip(),
                "f0_method": self.extract_f0.text(),
                "cpu_threads": int(self.extract_threads.value()),
                "gpu": self.extract_gpu.text(),
                "sample_rate": int(self.sample_rate.text()),
                "vocoder_arch": self.vocoder.value(),
                "embedder_model": self.extract_embedder.text(),
                "embedder_model_custom": self.extract_custom_embedder.text() or None,
                "include_mutes": int(self.include_mutes.value()),
                "remove_16k_slices": self.remove_16k.isChecked(),
                "feature_precision": self.feature_precision.text(),
            },
            busy_text=_("Extracting features…"),
            buttons=[self.extract_button],
        )

    def train_args(self) -> dict:
        """Every argument ``core.run_train_script`` takes, from the form.

        Split out from :meth:`_start_training` so the test suite can compare
        these keys against the backend's signature.  A parameter that exists in
        core but has no control here is invisible otherwise -- it just silently
        keeps its default, which is how torch_compile_mode and the gradient
        clipping schedule went missing in the first place.
        """
        return {
            "model_name": self.model_name.text().strip(),
            "epoch_save_frequency": int(self.save_every.value()),
            "save_only_latest_net_models": self.save_latest_only.isChecked(),
            "save_weight_models": self.save_weights.isChecked(),
            "total_epoch_count": int(self.total_epochs.value()),
            "sample_rate": int(self.sample_rate.text()),
            "batch_size": int(self.batch_size.value()),
            "gpu": self.train_gpu.text(),
            "use_warmup": self.use_warmup.isChecked(),
            "warmup_duration": int(self.warmup_duration.value()),
            "pretrained": self.pretrained.isChecked(),
            "cleanup": self.cleanup.isChecked(),
            "index_algorithm": self.index_algorithm.text(),
            "custom_pretrained": self.custom_pretrained.isChecked(),
            "g_pretrained_path": self.pretrained_g.path() or None,
            "d_pretrained_path": self.pretrained_d.path() or None,
            "vocoder": self.vocoder.value(),
            "optimizer_choice": self.optimizer.text(),
            "use_checkpointing": self.checkpointing.isChecked(),
            "use_tf32": self.tf32.isChecked(),
            "use_benchmark": self.benchmark.isChecked(),
            "lr_scheduler": self.lr_scheduler.text(),
            "use_custom_lr": self.use_custom_lr.isChecked(),
            "custom_lr_g": self.custom_lr_g.value(),
            "custom_lr_d": self.custom_lr_d.value(),
            "use_ema": self.use_ema.isChecked(),
            "overtrain_detector": self.overtrain_detector.isChecked(),
            "stop_on_overtrain": self.stop_on_overtrain.isChecked(),
            "compile_vocoder": self.compile_vocoder.isChecked(),
            "torch_compile_mode": self.torch_compile_mode.text(),
        }

    def _start_training(self) -> None:
        if not self.require(**{"A model name": self.model_name.text().strip()}):
            return

        args = self.train_args()

        self._training = True
        self._set_training_controls(True)
        self.busy.emit(True, "Training…")
        # A previous run's card may still be on its way out.
        self._dismiss_timer.stop()
        self.progress.begin(int(self.total_epochs.value()))
        self._progress_used = True
        self.progress.reveal()
        self.progressActive.emit(True)
        # A run you just started is the one worth watching, so this overrides
        # an earlier explicit pick.
        self._run_locked = False
        started = catalog.latest_run_dir(args["model_name"])
        self._refresh_runs(select=str(started) if started else None)

        def finish(_data=None) -> None:
            self._training = False
            self._set_training_controls(False)
            self.busy.emit(False, "")
            # One last poll so the chart ends on the final checkpoint rather
            # than wherever the 3 s timer happened to leave it.
            self._poll_metrics()

        def on_result(data: dict) -> None:
            finish()
            message = str(data.get("message", "Training finished."))
            self.progress.finish(message)
            self._hold_then_dismiss()
            self.notify.emit("success", message)

        def on_error(error: str) -> None:
            finish()
            self.progress.finish(f"Stopped: {error}")
            self._hold_then_dismiss()
            self.notify.emit("error", error)

        self.engine.call("train", args, on_result=on_result, on_error=on_error)

    def _set_training_controls(self, training: bool) -> None:
        """Swap Start for Stop."""
        self.train_button.setVisible(not training)
        self.stop_button.setVisible(training)
        self.stop_button.setEnabled(training)
        self.stop_button.setText(_("Stop training"))
        self.progress.set_stoppable(training)

    # -- the progress card's own lifetime ----------------------------------

    def _hold_then_dismiss(self) -> None:
        """Leave the finished card up for a while, then animate it away.

        Not immediate: the final epoch count and elapsed time are the only
        place those numbers are shown, and taking them off screen the instant
        the run ends is how a finished run becomes a run you have to go and
        look up.  Not permanent either -- the panel should not be left holding
        a stale run for the rest of the session.
        """
        self._dismiss_timer.start(self.FINISHED_HOLD_MS)

    def _dismiss_progress(self) -> None:
        if self._training:  # a new run beat the timer to it
            return
        self._progress_used = False
        self.progress.dismiss(on_done=lambda: self.progressActive.emit(False))

    def _stop_training(self) -> None:
        # Not hidden: the run keeps going until the trainer reaches a
        # checkpoint, and putting Start back before it has actually stopped
        # invites a second run on top of the first.
        self.stop_button.setEnabled(False)
        self.stop_button.setText(_("Stopping…"))
        self.log.emit("Stop requested; waiting for the current checkpoint to finish writing.")
        self.engine.call(
            "stop_train",
            {},
            on_result=lambda data: self.notify.emit("info", str(data.get("message", "Stop requested."))),
            on_error=lambda error: self.notify.emit("error", error),
        )

    def _build_index(self) -> None:
        if not self.require(**{"A model name": self.model_name.text().strip()}):
            return
        self.run(
            "index",
            {
                "model_name": self.model_name.text().strip(),
                "index_algorithm": self.index_algorithm.text(),
                "index_metric": self.index_metric.text(),
            },
            busy_text=_("Building index…"),
            buttons=[self.index_button],
        )

    # -- lifecycle ---------------------------------------------------------

    def on_shown(self) -> None:
        self._refresh_models()
        self._refresh_runs()
        # Rescanned on every show rather than once at build: files dropped into
        # the datasets or custom pretraineds folder while the window is open
        # are exactly when someone goes looking for them here.
        self.refresh_suggestions()

    def on_hidden(self) -> None:
        # Keep polling only while a run is actually going: a training run left
        # unattended on another tab still wants its chart current when the user
        # comes back, but an idle page has nothing to read.
        if not self._training:
            self._poll_timer.stop()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        super().apply_theme(tokens)

    def populate_gpus(self, devices: list[dict]) -> None:
        """Fill the device pickers once the backend has queried torch."""
        self._apply_tf32_support(devices)
        if not devices:
            return
        labels = [f"{device['index']}" for device in devices]
        self.extract_gpu.set_items(labels)
        self.train_gpu.set_items(labels)
        for combo in (self.extract_gpu, self.train_gpu):
            for position, device in enumerate(devices):
                combo.combo.setItemData(
                    position,
                    f"{device['name']} · {device['total_vram'] / 2**30:.0f} GB",
                    Qt.ToolTipRole,
                )

    def _apply_tf32_support(self, devices: list[dict]) -> None:
        """Enable TF32 when the hardware has the tensor cores for it.

        Matches the Gradio tab, which both ticks and enables the box from
        ``microarchitecture_capability_checker`` -- compute capability 8.0 and
        up, meaning Ampere onwards.  The capability comes back from
        ``cmd_gpu_info`` as "major.minor".
        """
        supported = False
        for device in devices:
            try:
                major = int(str(device.get("capability", "0")).split(".")[0])
            except (TypeError, ValueError):
                continue
            if major >= 8:
                supported = True
                break

        self.tf32.setEnabled(supported)
        self.tf32.setChecked(supported)
        self.tf32.setToolTip(
            _("Faster on Ampere and newer, marginally less precise.")
            if supported
            else _("This GPU has no TF32 units, so the setting would do nothing.")
        )

    def refresh_suggestions(self) -> None:
        """Offer what is already on disk from the path fields themselves."""
        self.pretrained_g.set_suggestions(catalog.list_custom_pretraineds("G"))
        self.pretrained_d.set_suggestions(catalog.list_custom_pretraineds("D"))
        self.dataset.set_suggestions(catalog.list_dataset_folders())
