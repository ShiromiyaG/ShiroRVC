"""Voice conversion: single file and folder batch."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..services import catalog, paths, prefs
from ..widgets.audio import AudioPlayer
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
    primary_button,
)
from .base import Page

from ..i18n import _, N_, ngettext

_AUDIO_FILTER = "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.opus);;All files (*.*)"


class ModelSelector(QWidget):
    """Model, index and speaker, kept consistent with each other.

    Picking the model fills in the index and reloads the speaker list, because
    those three are one choice presented as three controls -- and the two
    silent failure modes of this screen are a stale index and a stale speaker
    id pointing at a different checkpoint.
    """

    def __init__(self, page: Page, parent: QWidget | None = None):
        super().__init__(parent)
        self._page = page

        self.model = SearchableCombo()
        self.model.refreshRequested.connect(self.refresh)
        self.model.currentTextChanged.connect(self._on_model_changed)

        self.index = SearchableCombo()
        self.index.refreshRequested.connect(self.refresh)

        self.speaker = SearchableCombo(editable=False)
        self.speaker.refresh_button.hide()

        self.submodel = SearchableCombo(editable=False)
        self.submodel.refresh_button.hide()
        self.submodel_field = Field(
            _("Bundle voice"),
            self.submodel,
            _("Which voice inside the .srvc bundle to use."),
        )
        self.submodel_field.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Voice model"), self.model, _("Checkpoint (.pth) or bundle (.srvc) under logs/.")), 3)
        row.addWidget(Field(_("Speaker"), self.speaker, _("Speaker id inside a multi-speaker model.")), 1)
        layout.addLayout(row)
        layout.addWidget(Field(_("Index"), self.index, _("Faiss index. Picked automatically when there is only one next to the model.")))
        layout.addWidget(self.submodel_field)

        self.refresh()

    def refresh(self) -> None:
        self.model.set_items(catalog.list_models())
        self.index.set_items([""] + catalog.list_indexes())

    def _on_model_changed(self, model: str) -> None:
        if not model:
            return
        guess = catalog.guess_index_for(model)
        if guess and not self.index.text():
            self.index.set_text(guess)
        elif guess:
            # Switching models with a stale index selected is the classic way
            # to get output that sounds like the previous voice.
            current_owner = Path(self.index.text()).parent.name
            if current_owner != Path(model).parent.name:
                self.index.set_text(guess)

        is_bundle = model.lower().endswith(".srvc")
        self.submodel_field.setVisible(is_bundle)
        if is_bundle:
            self._page.engine.call(
                "bundle_models",
                {"model": model},
                on_result=lambda data: self.submodel.set_items(data.get("names", [])),
                on_error=lambda _error: self.submodel.set_items([]),
            )
        self._reload_speakers()

    def _reload_speakers(self) -> None:
        model = self.model.text()
        if not model:
            return
        self._page.engine.call(
            "speakers",
            {"model": model, "sub_model": self.submodel.text() or None},
            on_result=lambda data: self.speaker.set_items(
                [str(value) for value in data.get("speakers", [0])]
            ),
            on_error=lambda _error: self.speaker.set_items(["0"]),
        )

    def values(self) -> dict:
        return {
            "pth_path": self.model.text(),
            "index_path": self.index.text(),
            "sid": int(self.speaker.text() or 0),
            "bundle_submodel": self.submodel.text() or None,
        }


class ConversionSettings(QWidget):
    """The parameter block shared by single, batch and TTS conversion.

    ``profile`` selects which of the Gradio tabs' defaults to start from; the
    three differ, and matching them is what keeps an untouched form in this
    interface producing the same audio as an untouched form in that one.
    """

    def __init__(self, profile: str = "single", parent: QWidget | None = None):
        super().__init__(parent)
        self.profile = profile
        defaults = catalog.INFERENCE_DEFAULTS[profile]

        self.pitch = SliderSpin(-24, 24, 1, decimals=0, value=defaults["pitch"])
        self.index_rate = SliderSpin(0, 1, 0.01, decimals=2, value=defaults["index_rate"])
        self.index_k = SliderSpin(1, 32, 1, decimals=0, value=defaults["index_k"])
        self.index_power = SliderSpin(
            0, 8, 0.25, decimals=2, value=defaults["index_power"]
        )
        self.index_continuity = SliderSpin(
            0, 4, 0.1, decimals=2, value=defaults["index_continuity"]
        )
        self.volume_envelope = SliderSpin(
            0, 1, 0.01, decimals=2, value=defaults["volume_envelope"]
        )
        self.protect = SliderSpin(0, 0.5, 0.01, decimals=2, value=defaults["protect"])

        low, high, step, decimals = defaults["filter_radius_range"]
        self.filter_radius = SliderSpin(
            low, high, step, decimals=decimals, value=defaults["filter_radius"]
        )

        self.f0_method = SearchableCombo(editable=False)
        self.f0_method.refresh_button.hide()
        self.f0_method.set_items(catalog.F0_METHODS)

        self.embedder = SearchableCombo(editable=False)
        self.embedder.refresh_button.hide()
        self.embedder.set_items(catalog.EMBEDDER_MODELS)
        self.embedder.currentTextChanged.connect(self._on_embedder_changed)

        self.custom_embedder = SearchableCombo()
        self.custom_embedder.refreshRequested.connect(
            lambda: self.custom_embedder.set_items(catalog.list_custom_embedders())
        )
        self.custom_embedder.set_items(catalog.list_custom_embedders())
        self.custom_embedder_field = Field(
            _("Custom embedder"), self.custom_embedder, _("Folder under rvc/models/embedders/embedders_custom.")
        )
        self.custom_embedder_field.hide()

        self.export_format = SearchableCombo(editable=False)
        self.export_format.refresh_button.hide()
        self.export_format.set_items(catalog.EXPORT_FORMATS)

        self.split_audio = Toggle(
            _("Split long audio"), _("Cut into segments before converting. Helps on long files."),
            checked=defaults["split_audio"],
        )
        self.clean_audio = Toggle(
            _("Noise reduction"), _("Denoise the output. Recommended for speech, not for singing."),
            checked=defaults["clean_audio"],
        )
        self.clean_strength = SliderSpin(
            0, 1, 0.01, decimals=2, value=defaults["clean_strength"]
        )
        self.autotune = Toggle(
            _("Autotune"), _("Snap pitch to the chromatic grid. For singing."),
            checked=defaults["f0_autotune"],
        )
        self.autotune_strength = SliderSpin(
            0, 1, 0.01, decimals=2, value=defaults["f0_autotune_strength"]
        )
        self.formant = Toggle(
            _("Formant shifting"),
            _("Shift formants independently of pitch. For cross-gender conversion."),
            checked=defaults["formant_shifting"],
        )
        self.formant_quefrency = SliderSpin(0, 16, 0.1, decimals=1, value=1.0)
        self.formant_timbre = SliderSpin(0, 16, 0.1, decimals=1, value=1.0)

        self.f0_file = PathPicker(filters="F0 curve (*.f0 *.txt);;All files (*.*)",
                                  placeholder=_("Optional external pitch curve"))
        self.seed = SliderSpin(0, 2**16, 1, decimals=0, value=defaults["seed"])
        self.deterministic = Toggle(
            _("Deterministic"), _("Reproducible output for a given seed."),
            checked=defaults["deterministic"],
        )
        self.temperature = SliderSpin(
            0, 2, 0.01, decimals=2, value=defaults["latent_temperature"]
        )

        self.filter_radius_field = Field(
            _("Filter radius"), self.filter_radius,
            _("Smooths the extracted pitch curve."),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # -- always visible: the six controls a conversion actually turns on --
        layout.addWidget(Field(_("Pitch (semitones)"), self.pitch, _("12 = one octave up, -12 = one octave down.")))
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Search feature ratio"), self.index_rate, _("How much of the index to blend in. Higher is closer to the training voice.")))
        row.addWidget(Field(_("Protect"), self.protect, _("Shields consonants and breaths from index artefacts.")))
        layout.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Volume envelope"), self.volume_envelope, _("1 keeps the output's own dynamics; 0 copies the input's.")))
        row.addWidget(Field(_("Pitch algorithm"), self.f0_method, _("rmvpe is the recommended default.")))
        layout.addLayout(row)
        layout.addWidget(Field(_("Embedder"), self.embedder, _("Model used to extract speaker-independent content features.")))
        layout.addWidget(self.custom_embedder_field)

        # -- everything else, folded away like the Gradio tab's accordion --
        self.advanced = Collapsible(
            _("Advanced settings"), _("output format, post-processing, formants, seed")
        )
        layout.addWidget(self.advanced)

        self.advanced.add_group("Output")
        self.advanced.add_row(
            Field(_("Export format"), self.export_format, ""),
            self.filter_radius_field,
        )
        # Hidden for single-file conversion because the Gradio tab hides it
        # there too; exposing it would let this interface produce settings that
        # one cannot.  Only once the field has a parent: showing a parentless
        # widget makes it a top-level window, and it flashes on screen for the
        # frame it takes to be reparented into this layout.
        self.filter_radius_field.setVisible(defaults["filter_radius_visible"])

        self.advanced.add_group("Index retrieval")
        self.advanced.add_row(
            Field(
                _("Neighbours"),
                self.index_k,
                _("Frames averaged per match. Fewer keeps what is idiosyncratic "
                "about the training voice; more averages toward its mean."),
            ),
            Field(
                _("Sharpness"),
                self.index_power,
                _("How strongly closer matches outweigh further ones. 0 averages "
                "them equally."),
            ),
        )
        self.advanced.add(
            Field(
                _("Continuity"),
                self.index_continuity,
                _("Favours matches that continue the previous frame's, instead of "
                "jumping between unrelated parts of the dataset. Needs an index "
                "built by this fork."),
            ),
        )

        self.advanced.add_group("Post-processing")
        self.advanced.add(
            self.split_audio,
            self.clean_audio,
            Field(_("Cleaning strength"), self.clean_strength, ""),
            self.autotune,
            Field(_("Autotune strength"), self.autotune_strength, ""),
            self.formant,
        )
        self.advanced.add_row(
            Field(_("Formant quefrency"), self.formant_quefrency, ""),
            Field(_("Formant timbre"), self.formant_timbre, ""),
        )

        self.advanced.add_group("Generation")
        self.advanced.add(
            Field(_("F0 curve file"), self.f0_file, _("Use a pre-computed pitch curve instead of extracting one.")),
        )
        self.advanced.add_row(
            Field(_("Seed"), self.seed, _("0 picks a fresh seed each run.")),
            Field(_("Latent temperature"), self.temperature, _("ChouwaGAN only. Above 1 adds variation, below 1 flattens it.")),
        )
        self.advanced.add(self.deterministic)

        # Dependent controls start in the state their toggle implies rather
        # than waiting for the first click.
        for toggle, dependants in (
            (self.clean_audio, [self.clean_strength]),
            (self.autotune, [self.autotune_strength]),
            (self.formant, [self.formant_quefrency, self.formant_timbre]),
        ):
            toggle.toggled.connect(
                lambda checked, widgets=dependants: [w.setEnabled(checked) for w in widgets]
            )
            for widget in dependants:
                widget.setEnabled(toggle.isChecked())

    def _on_embedder_changed(self, value: str) -> None:
        self.custom_embedder_field.setVisible(value == "custom")

    def values(self) -> dict:
        return {
            "pitch": int(self.pitch.value()),
            "index_rate": self.index_rate.value(),
            "index_k": int(self.index_k.value()),
            "index_power": self.index_power.value(),
            "index_continuity": self.index_continuity.value(),
            "volume_envelope": self.volume_envelope.value(),
            "protect": self.protect.value(),
            # Not coerced to int: the single-file profile's 0.006 is a float
            # upstream, and rounding it here would change the pitch smoothing.
            "filter_radius": self.filter_radius.value(),
            "f0_method": self.f0_method.text(),
            "embedder_model": self.embedder.text(),
            "embedder_model_custom": self.custom_embedder.text() or None,
            "export_format": self.export_format.text(),
            "split_audio": self.split_audio.isChecked(),
            "clean_audio": self.clean_audio.isChecked(),
            "clean_strength": self.clean_strength.value(),
            "f0_autotune": self.autotune.isChecked(),
            "f0_autotune_strength": self.autotune_strength.value(),
            "formant_shifting": self.formant.isChecked(),
            "formant_qfrency": self.formant_quefrency.value(),
            "formant_timbre": self.formant_timbre.value(),
            "f0_file": self.f0_file.path() or None,
            "seed": int(self.seed.value()),
            "deterministic": self.deterministic.isChecked(),
            "latent_temperature": self.temperature.value(),
        }


class InferencePage(Page):
    title = N_("Inference")
    subtitle = N_("Convert a recording to a trained voice.")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        #: Last file a conversion produced, for "Save a copy…".
        self._last_output: str = ""

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_single(), "Single file")
        self.tabs.addTab(self._build_batch(), "Batch folder")
        self.content.addWidget(self.tabs)
        self.content.addStretch(1)

    # -- single ------------------------------------------------------------

    def _build_single(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        source = Card(_("Source"), _("The audio you want converted."))
        self.input_path = PathPicker(filters=_AUDIO_FILTER, placeholder=_("Drop an audio file here or browse…"))
        self.input_path.pathChanged.connect(self._on_input_changed)
        source.add(Field(_("Input file"), self.input_path, ""))

        self.input_player = AudioPlayer()
        source.add(self.input_player)
        layout.addWidget(source)

        model_card = Card(_("Voice"), _("Which model to convert into."))
        self.selector = ModelSelector(self)
        model_card.add(self.selector)
        layout.addWidget(model_card)

        settings_card = Card(_("Settings"), _("Everything below has a working default."))
        self.settings = ConversionSettings("single")
        settings_card.add(self.settings)
        layout.addWidget(settings_card)

        output_card = Card(_("Output"))
        self.output_path = PathPicker(mode="save", filters=_AUDIO_FILTER)
        self.output_path.set_path(str(paths.AUDIO_DIR / f"filename{catalog.OUTPUT_SUFFIX}.wav"))
        output_card.add(Field(_("Save to"), self.output_path, ""))

        self.convert_button = primary_button(_("Convert"))
        self.convert_button.clicked.connect(self._convert)
        output_card.add(self.convert_button)

        # The conversion already wrote to "Save to", but that path is decided
        # before hearing the result.  Keeping a copy somewhere else afterwards
        # -- the equivalent of the Gradio tab's download button -- otherwise
        # means finding the file in assets/audios by hand.  It sits on the
        # player beside "Show in folder" because it acts on that file, not on
        # the form above it.
        self.output_player = AudioPlayer()
        self.output_player.enable_saving()
        self.output_player.saveRequested.connect(self._save_copy)
        output_card.add(self.output_player)
        layout.addWidget(output_card)

        return page

    def _save_copy(self) -> None:
        """Copy the last result to a folder the user picks."""
        source = self.output_player.path() or self._last_output
        if not source or not os.path.isfile(source):
            self.notify.emit("error", _("There is no converted file to save yet."))
            return

        folder = QFileDialog.getExistingDirectory(
            self, _("Save a copy to"), str(prefs.get("last_export_dir", ""))
            or str(Path.home()),
        )
        if not folder:
            return

        destination = Path(folder) / Path(source).name
        # Never silently replace: the obvious name collision here is copying
        # two takes of the same input into one folder.
        if destination.exists() and os.path.abspath(destination) != os.path.abspath(source):
            stem, suffix = destination.stem, destination.suffix
            index = 2
            while destination.exists():
                destination = destination.with_name(f"{stem} ({index}){suffix}")
                index += 1

        try:
            shutil.copy2(source, destination)
        except OSError as error:
            self.notify.emit("error", _("Could not save the copy: {error}").format(error=error))
            return

        prefs.set("last_export_dir", folder)
        self.notify.emit("success", _("Saved to {path}").format(path=destination))

    def _on_input_changed(self, path: str) -> None:
        if path and os.path.isfile(path):
            self.input_player.load(path)
            # Exactly the name the Gradio tab would choose. Diverging here
            # would put the two interfaces' results in different files, and
            # would hide these from that tab's "clear _output files" button.
            self.output_path.set_path(catalog.default_output_path(path))
        else:
            self.input_player.clear()

    def _convert(self) -> None:
        values = self.selector.values()
        if not self.require(**{
            "An input file": self.input_path.path(),
            "A voice model": values["pth_path"],
            "An output path": self.output_path.path(),
        }):
            return
        if not os.path.isfile(self.input_path.path()):
            self.notify.emit("error", _("The input file does not exist."))
            return

        args = {
            **values,
            **self.settings.values(),
            "input_path": self.input_path.path(),
            "output_path": self.output_path.path(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args["output_path"])) or ".", exist_ok=True)

        self.run(
            "infer",
            args,
            busy_text=_("Converting…"),
            buttons=[self.convert_button],
            on_result=self._on_converted,
        )

    def _on_converted(self, data: dict) -> None:
        preview = data.get("preview") or self.output_path.path()
        if preview and os.path.isfile(preview):
            # The player arms its own save icon on load.
            self.output_player.load(preview)
            self._last_output = preview

    # -- batch -------------------------------------------------------------

    def _build_batch(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        folders = Card(_("Folders"), _("Every audio file in the input folder is converted."))
        self.batch_input = PathPicker(mode="dir", placeholder=_("Folder with audio to convert"))
        self.batch_output = PathPicker(mode="dir", placeholder=_("Where to write the results"))
        self.batch_output.set_path(str(paths.AUDIO_DIR / "batch"))
        folders.add(
            Field(_("Input folder"), self.batch_input, ""),
            Field(_("Output folder"), self.batch_output, ""),
        )
        layout.addWidget(folders)

        model_card = Card(_("Voice"))
        self.batch_selector = ModelSelector(self)
        model_card.add(self.batch_selector)
        layout.addWidget(model_card)

        settings_card = Card(_("Settings"))
        self.batch_settings = ConversionSettings("batch")
        settings_card.add(self.batch_settings)
        layout.addWidget(settings_card)

        run_card = Card()
        self.batch_button = primary_button(_("Convert folder"))
        self.batch_button.clicked.connect(self._convert_batch)
        self.batch_hint = QLabel("")
        self.batch_hint.setObjectName("FieldHint")
        run_card.add(self.batch_button, self.batch_hint)
        layout.addWidget(run_card)

        self.batch_input.pathChanged.connect(self._count_batch)
        return page

    def _count_batch(self, folder: str) -> None:
        if not folder or not os.path.isdir(folder):
            self.batch_hint.setText("")
            return
        count = sum(
            1 for name in os.listdir(folder)
            if name.lower().endswith(catalog.AUDIO_EXTENSIONS)
        )
        self.batch_hint.setText(
            ngettext("{count} audio file found.", "{count} audio files found.", count)
            .format(count=count)
        )

    def _convert_batch(self) -> None:
        values = self.batch_selector.values()
        if not self.require(**{
            "An input folder": self.batch_input.path(),
            "An output folder": self.batch_output.path(),
            "A voice model": values["pth_path"],
        }):
            return

        settings = self.batch_settings.values()
        # run_batch_infer_script takes the same parameters as the single-file
        # one except for bundle_submodel; passing it would be a TypeError at
        # the other end rather than a harmless extra.
        values.pop("bundle_submodel", None)

        args = {
            **values,
            **settings,
            "input_folder": self.batch_input.path(),
            "output_folder": self.batch_output.path(),
        }
        os.makedirs(args["output_folder"], exist_ok=True)

        self.run(
            "batch_infer",
            args,
            busy_text=_("Converting folder…"),
            buttons=[self.batch_button],
        )

    # -- lifecycle ---------------------------------------------------------

    def on_shown(self) -> None:
        self.selector.refresh()
        self.batch_selector.refresh()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        super().apply_theme(tokens)
