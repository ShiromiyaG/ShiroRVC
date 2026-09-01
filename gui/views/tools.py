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

from ..services import catalog, paths
from ..theme import monospace_font
from ..widgets.forms import (
    Card,
    Field,
    PathPicker,
    SearchableCombo,
    SliderSpin,
    Toggle,
    primary_button,
)
from .base import Page

from ..i18n import _, N_

_MODEL_FILTER = "Models (*.pth *.srvc);;All files (*.*)"


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
        self.content.addWidget(tabs, 1)

    # -- model info --------------------------------------------------------

    def _build_info(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        card = Card(_("Model information"), _("Reads the metadata stored in a checkpoint."))
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

        card = Card(_("Audio analyzer"), _("Spectrum and level report for one file."))
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

    def on_shown(self) -> None:
        self.blend_name.set_items(catalog.list_training_models())
