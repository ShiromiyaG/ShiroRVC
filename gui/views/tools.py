"""Model inspection, blending, downloads and dataset analysis."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..services import catalog, experiments, paths
from ..theme import monospace_font
from ..widgets.forms import (
    Card,
    Field,
    FileList,
    PathPicker,
    SearchableCombo,
    SliderSpin,
    Toggle,
    primary_button,
)
from .base import Page

from ..i18n import _, N_

_MODEL_FILTER = "Models (*.pth *.srvc);;All files (*.*)"


def _details_label() -> QLabel:
    """A job's per-item report, selectable so a path can be copied out."""
    label = QLabel("")
    label.setObjectName("FieldHint")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _markdown(text: str, object_name: str = "") -> QLabel:
    """A wrapping, selectable label that renders the backend's Markdown."""
    label = QLabel(text)
    label.setTextFormat(Qt.MarkdownText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if object_name:
        label.setObjectName(object_name)
    return label


class ToolsPage(Page):
    title = N_("Utilities")
    subtitle = N_(
        "Inspect, blend and fetch models; check a dataset before training on it."
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        tabs = QTabWidget()
        tabs.addTab(self._build_info(), "Model info")
        tabs.addTab(self._build_blender(), "Blender")
        tabs.addTab(self._build_download(), "Download")
        tabs.addTab(self._build_analyzer(), "Audio analyzer")
        tabs.addTab(self._build_prerequisites(), "Prerequisites")
        tabs.addTab(self._build_experiment_config(), _("Experiment Config"))
        tabs.addTab(self._build_bundles(), _("Model Bundles"))
        self.content.addWidget(tabs, 1)

    # -- model info --------------------------------------------------------

    def _build_info(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(_("Model information"), _("Reads the metadata stored in a checkpoint."), icon="search")
        self.info_path = PathPicker(filters=_MODEL_FILTER, placeholder=_("Drop a .pth or .srvc here"))
        card.add(Field(_("Model file"), self.info_path, ""))
        self.info_button = primary_button(_("Inspect"))
        self.info_button.clicked.connect(self._inspect)
        card.add(self.info_button)

        self.info_output = QPlainTextEdit()
        self.info_output.setReadOnly(True)
        self.info_output.setFont(monospace_font(9))
        self.info_output.setMinimumHeight(280)
        self.info_output.setPlaceholderText(_("Details appear here."))
        card.add(self.info_output)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _inspect(self) -> None:
        if not self.require(**{"A model file": self.info_path.path()}):
            return
        self.run(
            "model_info",
            {"pth_path": self.info_path.path()},
            busy_text=_("Reading checkpoint…"),
            buttons=[self.info_button],
            on_result=lambda data: self.info_output.setPlainText(data.get("info", "")),
            success_text=_("Checkpoint read."),
        )

    # -- blender -----------------------------------------------------------

    def _build_blender(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(
            _("Model blender"),
            _("Interpolates two checkpoints. Both must share an architecture and sample rate."),
            icon="sliders",
        )
        self.blend_name = SearchableCombo()
        self.blend_name.refresh_button.hide()
        self.blend_name.combo.setEditable(True)
        card.add(Field(_("Output name"), self.blend_name, _("Written to logs/ under this name.")))

        self.blend_a = PathPicker(filters=_MODEL_FILTER, placeholder=_("First model"))
        self.blend_b = PathPicker(filters=_MODEL_FILTER, placeholder=_("Second model"))
        card.add(Field(_("Model A"), self.blend_a, ""), Field(_("Model B"), self.blend_b, ""))

        self.blend_ratio = SliderSpin(0, 1, 0.01, decimals=2, value=0.5)
        card.add(Field(_("Ratio"), self.blend_ratio, _("0 is all of model A, 1 is all of model B.")))

        self.blend_button = primary_button(_("Blend"))
        self.blend_button.clicked.connect(self._blend)
        card.add(self.blend_button)

        self.blend_result = QLabel("")
        self.blend_result.setObjectName("FieldHint")
        self.blend_result.setWordWrap(True)
        card.add(self.blend_result)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _blend(self) -> None:
        if not self.require(**{
            "An output name": self.blend_name.text().strip(),
            "Model A": self.blend_a.path(),
            "Model B": self.blend_b.path(),
        }):
            return
        self.run(
            "blend",
            {
                "model_name": self.blend_name.text().strip(),
                "pth_path_1": self.blend_a.path(),
                "pth_path_2": self.blend_b.path(),
                "ratio": self.blend_ratio.value(),
            },
            busy_text=_("Blending models…"),
            buttons=[self.blend_button],
            on_result=lambda data: self.blend_result.setText(
                f"{data.get('message', '')}\n{data.get('output', '')}".strip()
            ),
        )

    # -- download ----------------------------------------------------------

    def _build_download(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(
            _("Download a model"),
            _("Accepts Hugging Face, Google Drive, Pixeldrain and direct links."),
            icon="download",
        )
        self.download_url = SearchableCombo()
        self.download_url.refresh_button.hide()
        self.download_url.combo.setEditable(True)
        self.download_url.combo.lineEdit().setPlaceholderText(_("https://…"))
        card.add(Field(_("Link"), self.download_url, _("The archive is unpacked into logs/.")))

        self.download_button = primary_button(_("Download"))
        self.download_button.clicked.connect(self._download)
        card.add(self.download_button)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _download(self) -> None:
        link = self.download_url.text().strip()
        if not self.require(**{"A link": link}):
            return
        self.run(
            "download",
            {"model_link": link},
            busy_text=_("Downloading…"),
            buttons=[self.download_button],
        )

    # -- analyzer ----------------------------------------------------------

    def _build_analyzer(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(_("Audio analyzer"), _("Spectrum and level report for one file."), icon="waveform")
        self.analyze_path = PathPicker(
            filters="Audio (*.wav *.mp3 *.flac *.ogg *.m4a);;All files (*.*)"
        )
        card.add(Field(_("Audio file"), self.analyze_path, ""))
        self.analyze_button = primary_button(_("Analyze"))
        self.analyze_button.clicked.connect(self._analyze)
        card.add(self.analyze_button)

        self.analyze_text = QPlainTextEdit()
        self.analyze_text.setReadOnly(True)
        self.analyze_text.setFont(monospace_font(9))
        self.analyze_text.setMaximumHeight(140)
        card.add(self.analyze_text)

        self.analyze_plot = QLabel(_("The plot appears here."))
        self.analyze_plot.setObjectName("FieldHint")
        self.analyze_plot.setMinimumHeight(240)
        self.analyze_plot.setScaledContents(False)
        card.add(self.analyze_plot)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _analyze(self) -> None:
        if not self.require(**{"An audio file": self.analyze_path.path()}):
            return
        target = str(paths.LOGS_DIR / "audio_analysis.png")
        self.run(
            "analyze",
            {"input_path": self.analyze_path.path(), "save_plot_path": target},
            busy_text=_("Analyzing…"),
            buttons=[self.analyze_button],
            on_result=self._show_analysis,
            success_text=_("Analysis complete."),
        )

    def _show_analysis(self, data: dict) -> None:
        self.analyze_text.setPlainText(data.get("info", ""))
        plot = data.get("plot")
        if plot and os.path.isfile(plot):
            pixmap = QPixmap(plot)
            if not pixmap.isNull():
                # The plot is a chart, not a photo: smooth scaling is what keeps
                # thin axis lines from dropping out entirely.
                self.analyze_plot.setPixmap(
                    pixmap.scaledToWidth(
                        max(320, self.analyze_plot.width()), Qt.SmoothTransformation
                    )
                )

    # -- prerequisites -----------------------------------------------------

    def _build_prerequisites(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(
            _("Prerequisites"),
            _("Downloads the shared models the application needs. Safe to re-run; "
            "anything already present is skipped."),
            icon="check",
        )
        self.pre_pretrained = Toggle(_("HiFi-GAN pretrained models"), _("Needed to fine-tune rather than train from scratch."), checked=True)
        self.pre_models = Toggle(_("Embedders and pitch models"), _("contentvec, RMVPE, FCPE."), checked=True)
        self.pre_exe = Toggle(_("ffmpeg and ffprobe"), "", checked=True)
        card.add(self.pre_pretrained, self.pre_models, self.pre_exe)

        self.pre_button = primary_button(_("Download selected"))
        self.pre_button.clicked.connect(self._prerequisites)
        card.add(self.pre_button)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _prerequisites(self) -> None:
        self.run(
            "prerequisites",
            {
                "pretraineds_hifigan": self.pre_pretrained.isChecked(),
                "models": self.pre_models.isChecked(),
                "exe": self.pre_exe.isChecked(),
            },
            busy_text=_("Downloading prerequisites…"),
            buttons=[self.pre_button],
        )

    # -- experiment config -------------------------------------------------

    def _build_experiment_config(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(_("Experiment Config"), icon="chip")
        card.add(_markdown(
            _(
                "Rewrite an extracted experiment's `config.json` for a different "
                "vocoder, without re-running preprocessing or feature extraction. "
                "The new config is the shipped one for that vocoder at this "
                "experiment's sample rate, so anything hand-tuned in the old file "
                "is replaced rather than merged."
            ),
            "PageSubtitle",
        ))

        self.exp_name = SearchableCombo(editable=False)
        self.exp_name.refreshRequested.connect(self._refresh_experiments)
        self.exp_name.currentTextChanged.connect(self._describe_experiment)
        self.exp_vocoder = SearchableCombo(editable=False)
        self.exp_vocoder.refresh_button.hide()
        self.exp_vocoder.set_pairs(experiments.vocoder_choices())
        self._select_vocoder(experiments.default_vocoder())
        card.add_row(
            Field(_("Experiment"), self.exp_name,
                  _("A folder under logs/ that already has a config.json.")),
            Field(_("Vocoder / Architecture"), self.exp_vocoder,
                  _("The architecture the rebuilt config will be written for.")),
        )

        self.exp_details = _markdown(
            _("Select an experiment to see what its config was written for.")
        )
        card.add(self.exp_details)

        self.exp_backup = Toggle(
            _("Keep a backup of the current config"),
            _("Saved beside it as config.json.<old architecture>.bak."),
            checked=True,
        )
        self.exp_move = Toggle(
            _("Move existing checkpoints aside"),
            _(
                "G_*.pth / D_*.pth from the old architecture are moved into a "
                "subfolder instead of being deleted."
            ),
        )
        card.add_row(self.exp_backup, self.exp_move)

        self.exp_button = primary_button(_("Rebuild config"))
        self.exp_button.clicked.connect(self._rebuild_config)
        card.add(self.exp_button)

        self.exp_report = _markdown("")
        self.exp_report.hide()
        card.add(self.exp_report)

        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _refresh_experiments(self) -> None:
        self.exp_name.set_items(experiments.list_experiments())

    def _select_vocoder(self, vocoder: str) -> None:
        index = self.exp_vocoder.combo.findData(vocoder)
        if index >= 0:
            self.exp_vocoder.combo.setCurrentIndex(index)

    def _describe_experiment(self, experiment: str) -> None:
        if not experiment:
            self.exp_details.setText(
                _("Select an experiment to see what its config was written for.")
            )
            return
        text, current = experiments.describe(experiment)
        self.exp_details.setText(text)
        # Unrecognised or disabled: leave the target where the user put it.
        if current:
            self._select_vocoder(current)

    def _rebuild_config(self) -> None:
        experiment = self.exp_name.value()
        if not experiment:
            self.notify.emit("error", _("Select an experiment first."))
            return
        try:
            report = experiments.rebuild(
                experiment,
                self.exp_vocoder.value(),
                self.exp_backup.isChecked(),
                self.exp_move.isChecked(),
            )
        except Exception as error:  # noqa: BLE001 - refusal or I/O, both shown
            message = str(error)
            self.exp_report.setText(message)
            self.exp_report.show()
            self.notify.emit("error", message.replace("`", ""))
            return
        self.exp_report.setText(report)
        self.exp_report.show()
        self._describe_experiment(experiment)
        self.notify.emit("success", _("Config rebuilt."))

    # -- model bundles -----------------------------------------------------

    def _build_bundles(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        create = Card(
            _("Create a bundle"),
            _("Combines .pth models, and their indexes, into one compressed .srvc file."),
            icon="download",
        )
        self.bundle_pths = FileList((".pth",), "Models (*.pth);;All files (*.*)")
        self.bundle_indexes = FileList((".index",), "Indexes (*.index);;All files (*.*)")
        create.add_row(
            Field(_("Models"), self.bundle_pths,
                  _("Speaker names come from the .pth file names.")),
            Field(_("Indexes (optional)"), self.bundle_indexes,
                  _("Paired by name: voz_10e_850s.pth takes voz_10e_850s.index, "
                    "else voz.index, else a lone voz_spk<N>.index.")),
        )
        self.bundle_single_index = Toggle(
            _("Single index"), _("Attach one uploaded index to every model.")
        )
        self.bundle_compression = SliderSpin(1, 22, 1, decimals=0, value=3)
        create.add_row(
            self.bundle_single_index,
            Field(_("Compression level"), self.bundle_compression,
                  _("3 balances size and speed.")),
        )
        self.bundle_output = PathPicker(
            mode="save", filters="Model bundles (*.srvc)", placeholder=_("Empty saves to logs using the first model name.")
        )
        create.add(Field(_("Output path"), self.bundle_output, ""))
        self.bundle_create_button = primary_button(_("Create model bundle"))
        self.bundle_create_button.clicked.connect(self._create_bundle)
        create.add(self.bundle_create_button)
        self.bundle_create_result = _details_label()
        create.add(self.bundle_create_result)
        layout.addWidget(create)

        extract = Card(
            _("Extract a bundle"),
            _("Writes each model back out as <name>/<name>.pth and .index, the layout Applio pairs by name."),
            icon="folder",
        )
        self.bundle_file = PathPicker(filters="Model bundles (*.srvc);;All files (*.*)")
        self.bundle_extract_dir = PathPicker(
            mode="dir", placeholder=_("Empty extracts to logs/<bundle>_extracted.")
        )
        extract.add_row(
            Field(_("Bundle"), self.bundle_file, _("A .srvc under logs/, or type a path.")),
            Field(_("Output folder"), self.bundle_extract_dir, ""),
        )
        self.bundle_overwrite = Toggle(_("Overwrite existing files"))
        extract.add(self.bundle_overwrite)
        self.bundle_extract_button = primary_button(_("Extract bundle"))
        self.bundle_extract_button.clicked.connect(self._extract_bundle)
        extract.add(self.bundle_extract_button)
        self.bundle_extract_result = _details_label()
        extract.add(self.bundle_extract_result)
        layout.addWidget(extract)

        layout.addStretch(1)
        return page

    def _create_bundle(self) -> None:
        pth_paths = self.bundle_pths.paths()
        if not pth_paths:
            self.notify.emit("error", _("Add at least one .pth file."))
            return
        self.run(
            "bundle_create",
            {
                "pth_paths": pth_paths,
                "index_paths": self.bundle_indexes.paths(),
                "output_path": self.bundle_output.path(),
                "logs_dir": str(paths.LOGS_DIR),
                "single_index": self.bundle_single_index.isChecked(),
                "compression": int(self.bundle_compression.value()),
            },
            busy_text=_("Creating model bundle…"),
            buttons=[self.bundle_create_button],
            on_result=self._on_bundle_created,
        )

    def _on_bundle_created(self, data: dict) -> None:
        self.bundle_create_result.setText("\n".join(data.get("details", [])))
        self.bundle_file.set_suggestions(catalog.list_bundles())

    def _extract_bundle(self) -> None:
        bundle = self.bundle_file.path()
        if not self.require(**{"A bundle": bundle}):
            return
        self.run(
            "bundle_extract",
            {
                "bundle": bundle,
                "output_dir": self.bundle_extract_dir.path(),
                "logs_dir": str(paths.LOGS_DIR),
                "overwrite": self.bundle_overwrite.isChecked(),
            },
            busy_text=_("Extracting model bundle…"),
            buttons=[self.bundle_extract_button],
            on_result=lambda data: self.bundle_extract_result.setText(
                "\n".join(data.get("details", []))
            ),
        )

    def on_shown(self) -> None:
        self.bundle_file.set_suggestions(catalog.list_bundles())
        self._refresh_experiments()
        self.blend_name.set_items(catalog.list_training_models())
