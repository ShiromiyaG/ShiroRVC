"""Voice conversion: single file and folder batch."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..services import catalog, paths, prefs, presets
from ..widgets.audio import AudioPlayer
from ..widgets.forms import (
    Card,
    Collapsible,
    Field,
    PathPicker,
    SearchableCombo,
    SectionHeader,
    SliderSpin,
    TabPage,
    Toggle,
    danger_button,
    fit_to_current_tab,
    ghost_button,
    primary_button,
)
from .base import Page

from ..i18n import _, N_, ngettext

_AUDIO_FILTER = "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.opus);;All files (*.*)"

#: Inputs converted from this window, newest first.  The GUI uses files where
#: they are rather than copying them into assets/audios as the Gradio tab does,
#: so without this they would never show up among the previous audios.
_RECENT_INPUTS = "recent_inputs"
_RECENT_LIMIT = 15


def _input_suggestions() -> list[str]:
    """Recent inputs that still exist, then the uploads in assets/audios."""
    items = [
        path
        for path in prefs.get(_RECENT_INPUTS, [])
        if (paths.ROOT / path).is_file()
    ]
    items += [path for path in catalog.list_audios() if path not in items]
    return items


def _remember_input(path: str) -> None:
    path = paths.relative(path)
    recent = [item for item in prefs.get(_RECENT_INPUTS, []) if item != path]
    prefs.set(_RECENT_INPUTS, [path, *recent][:_RECENT_LIMIT])


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
        #: The model the speaker list was last requested for.  See
        #: :meth:`_on_model_changed`.
        self._model_for_speakers: str | None = None

        self.model = SearchableCombo()
        self.model.refreshRequested.connect(self.rescan)
        self.model.currentTextChanged.connect(self._on_model_changed)

        self.index = SearchableCombo()
        self.index.refreshRequested.connect(self.rescan)

        self.speaker = SearchableCombo(editable=False)
        self.speaker.refresh_button.hide()

        self.submodel = SearchableCombo(editable=False)
        self.submodel.refresh_button.hide()
        self.submodel.currentTextChanged.connect(self._on_submodel_changed)
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

        page.engine.ready.connect(self._on_engine_ready)
        self.refresh()

    def _on_engine_ready(self) -> None:
        """Ask again for what was asked before there was a backend to answer.

        The first :meth:`refresh` runs while the window is being built, and
        the backend is only started after its first paint -- so the speaker
        request for the model selected at launch was refused on the spot, and
        a multi-speaker model offered speaker 0 alone until another model was
        picked and this one picked back.  Also covers a restarted backend.
        Nothing is sent for a model whose answer already arrived.
        """
        self._on_model_changed(self.model.text())

    def refresh(self) -> None:
        self.model.set_items(catalog.list_models())
        self.index.set_items([""] + catalog.list_indexes())

    def rescan(self) -> None:
        """The refresh buttons: rescan, and re-read the model's speakers too.

        Unlike a page visit, this is someone asking because something on disk
        changed -- a checkpoint rewritten under the same name included.
        """
        self._model_for_speakers = None
        self.refresh()

    def _on_model_changed(self, model: str) -> None:
        if not model:
            return
        # ``set_items`` reports the selection on every refresh, and the page
        # refreshes on every visit.  Asking the worker again for a model it has
        # already answered for put two jobs in its queue per visit -- behind
        # whatever it was running.  Only an answer counts: a request refused
        # because the backend was still starting is retried on the next visit.
        if model == self._model_for_speakers:
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
        if not is_bundle:
            # Hidden is not enough: ``values`` would still hand the previous
            # bundle's voice name along with a plain checkpoint.
            self.submodel.set_items([])
            self._reload_speakers()
            return

        def names_arrived(names: list[str]) -> None:
            if self.model.text() == model:  # not switched away meanwhile
                self.submodel.set_items(names)

        # A bundle's speakers belong to one of its voices, so they are read
        # once the names are in and one is selected -- by
        # ``_on_submodel_changed``, which ``set_items`` always reaches.
        self._page.engine.call(
            "bundle_models",
            {"model": model},
            on_result=lambda data: names_arrived(data.get("names", [])),
            on_error=lambda _error: names_arrived([]),
        )

    def _on_submodel_changed(self, _name: str) -> None:
        """Speakers follow the voice picked inside a bundle.

        Nothing reloaded them before: they were read once, when the bundle was
        picked -- before its voice names had even arrived -- and stayed the
        first voice's whatever was chosen after.
        """
        if self.model.text().lower().endswith(".srvc"):
            self._reload_speakers()

    def _reload_speakers(self) -> None:
        model = self.model.text()
        if not model:
            return

        def arrived(data: dict) -> None:
            self._model_for_speakers = model
            self.speaker.set_items([str(value) for value in data.get("speakers", [0])])

        def failed(_error: str) -> None:
            self._model_for_speakers = None
            self.speaker.set_items(["0"])

        self._page.engine.call(
            "speakers",
            {"model": model, "sub_model": self.submodel.text() or None},
            on_result=arrived,
            on_error=failed,
        )

    def values(self) -> dict:
        return {
            "pth_path": self.model.text(),
            "index_path": self.index.text(),
            "sid": int(self.speaker.text() or 0),
            "bundle_submodel": self.submodel.text() or None,
        }


class PresetBar(QWidget):
    """Pick, save and import inference presets.

    Same folder and format as the Gradio tab (``services.presets``), so a
    preset made in either interface is offered by both.
    """

    notify = Signal(str, str)

    def __init__(
        self,
        collect: Callable[[], dict],
        apply: Callable[[dict], None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._collect = collect
        self._apply = apply
        #: The preset last applied or saved here, which the picker keeps
        #: showing across rescans.
        self._current = ""

        self.picker = SearchableCombo(editable=False)
        self.picker.combo.setPlaceholderText(_("Choose a preset…"))
        self.picker.refreshRequested.connect(self.refresh)
        # ``activated``, not ``currentTextChanged``: only a pick by the user
        # applies a preset.  A rescan re-selects the current entry, and that
        # must not put back values edited since.
        self.picker.combo.activated.connect(self._on_activated)

        save_button = ghost_button(_("Save…"))
        save_button.setToolTip(_("Save the current settings as a preset."))
        save_button.clicked.connect(self._save)
        import_button = ghost_button(_("Import…"))
        import_button.setToolTip(_("Copy a preset file into the presets folder and apply it."))
        import_button.clicked.connect(self._import)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.picker, 1)
        layout.addWidget(save_button)
        layout.addWidget(import_button)

        self.refresh()

    def refresh(self) -> None:
        self.picker.set_items(presets.list_presets())
        # ``set_items`` falls back to the first entry, which would show a
        # preset that was never applied.  Nothing picked shows the placeholder.
        combo = self.picker.combo
        combo.blockSignals(True)
        combo.setCurrentIndex(combo.findText(self._current) if self._current else -1)
        combo.blockSignals(False)

    def _on_activated(self, index: int) -> None:
        name = self.picker.combo.itemText(index)
        if name:
            self._load(name)

    def _load(self, name: str) -> None:
        try:
            values = presets.load(name)
        except (OSError, ValueError) as error:
            self.notify.emit("error", _("Could not read that preset file: {}").format(error))
            return
        self._apply(values)
        self._current = name
        self.notify.emit("success", _("Preset applied: {name}").format(name=name))

    def _save(self) -> None:
        name, accepted = QInputDialog.getText(
            self, _("Save preset"), _("Preset name:"), text=self._current
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            self.notify.emit("error", _("Enter a preset name first."))
            return
        if not presets.is_valid_name(name):
            self.notify.emit("error", _('A preset name cannot contain \\ / : * ? " < > |'))
            return
        if presets.exists(name) and not self._confirm_replace(name, _("Save preset")):
            return
        try:
            presets.save(name, self._collect())
        except (OSError, ValueError) as error:
            self.notify.emit("error", _("Could not save the preset: {error}").format(error=error))
            return
        self._current = name
        self.refresh()
        self.notify.emit("success", _("Preset saved: {}").format(name))

    def _import(self) -> None:
        source, _filter = QFileDialog.getOpenFileName(
            self, _("Import preset"), str(Path.home()),
            "Presets (*.json);;All files (*.*)",
        )
        if not source:
            return
        target = paths.INFERENCE_PRESET_DIR / f"{Path(source).stem}.json"
        if (
            target.exists()
            and not os.path.samefile(source, target)
            and not self._confirm_replace(target.stem, _("Import preset"))
        ):
            return
        try:
            name = presets.import_file(source)
        except (OSError, ValueError) as error:
            self.notify.emit("error", _("Could not read that preset file: {}").format(error))
            return
        self._current = name
        self.refresh()
        self._load(name)

    def _confirm_replace(self, name: str, title: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            _('Replace the preset "{name}"?').format(name=name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes


class ConversionSettings(QWidget):
    """The parameter block shared by single, batch and TTS conversion.

    ``profile`` selects which of the Gradio tabs' defaults to start from; the
    three differ, and matching them is what keeps an untouched form in this
    interface producing the same audio as an untouched form in that one.
    """

    #: The preset bar's reports, for the page to forward to the window.
    notify = Signal(str, str)

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
        self.silence_gate_db = SliderSpin(
            -120, 0, 1, decimals=0, value=defaults["silence_gate_db"]
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

        self.filter_radius_field = Field(
            _("Filter radius"), self.filter_radius,
            _("Smooths the extracted pitch curve."),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.presets = PresetBar(self.preset_values, self.apply_preset)
        self.presets.notify.connect(self.notify)
        layout.addWidget(Field(
            _("Preset"),
            self.presets,
            _("Saved settings, shared with the web interface. Picking one applies "
              "it; the model, index and file paths are not part of a preset."),
        ))

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
        row.addWidget(Field(_("Silence gate"), self.silence_gate_db, _("Fades out the output where the input is quieter than this, in dBFS. Silence has no level for the content encoder, so the model fills it with hiss. -120 turns the gate off.")))
        row.addWidget(Field(_("Pitch algorithm"), self.f0_method, _("rmvpe is the recommended default.")))
        layout.addLayout(row)
        layout.addWidget(Field(_("Embedder"), self.embedder, _("Model used to extract speaker-independent content features.")))

        # -- everything else, folded away like the Gradio tab's accordion --
        self.advanced = Collapsible(
            _("Advanced settings"), _("output format, post-processing, formants, seed")
        )
        layout.addWidget(self.advanced)

        self.advanced.add_group(_("Output"))
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

        self.advanced.add_group(_("Index retrieval"))
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

        self.advanced.add_group(_("Post-processing"))
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

        self.advanced.add_group(_("Generation"))
        self.advanced.add(
            Field(_("F0 curve file"), self.f0_file, _("Use a pre-computed pitch curve instead of extracting one.")),
        )
        self.advanced.add_row(
            Field(_("Seed"), self.seed, _("0 picks a fresh seed each run.")),
        )

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

    def values(self) -> dict:
        return {
            "pitch": int(self.pitch.value()),
            "index_rate": self.index_rate.value(),
            "index_k": int(self.index_k.value()),
            "index_power": self.index_power.value(),
            "index_continuity": self.index_continuity.value(),
            "volume_envelope": self.volume_envelope.value(),
            "silence_gate_db": self.silence_gate_db.value(),
            "protect": self.protect.value(),
            # Not coerced to int: the single-file profile's 0.006 is a float
            # upstream, and rounding it here would change the pitch smoothing.
            "filter_radius": self.filter_radius.value(),
            "f0_method": self.f0_method.text(),
            "embedder_model": self.embedder.text(),
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
        }

    def _preset_controls(self) -> dict[str, QWidget]:
        """Preset key -> the control holding it.

        The keys are the preset file's (``rvc.lib.inference_presets``), not
        :meth:`values`' -- the two name the envelope blend and the autotune
        pair differently, and the file's names are the ones the Gradio tab
        reads.
        """
        return {
            "export_format": self.export_format,
            "seed": self.seed,
            "split_audio": self.split_audio,
            "autotune": self.autotune,
            "autotune_strength": self.autotune_strength,
            "clean_audio": self.clean_audio,
            "clean_strength": self.clean_strength,
            "formant_shifting": self.formant,
            "formant_qfrency": self.formant_quefrency,
            "formant_timbre": self.formant_timbre,
            "pitch": self.pitch,
            "index_rate": self.index_rate,
            "index_k": self.index_k,
            "index_power": self.index_power,
            "index_continuity": self.index_continuity,
            "rms_mix_rate": self.volume_envelope,
            "protect": self.protect,
            "silence_gate_db": self.silence_gate_db,
            "f0_method": self.f0_method,
            "embedder_model": self.embedder,
        }

    def preset_values(self) -> dict:
        values = {}
        for key, control in self._preset_controls().items():
            if isinstance(control, Toggle):
                values[key] = control.isChecked()
            elif isinstance(control, SliderSpin):
                value = control.value()
                values[key] = int(value) if control.is_integral() else value
            else:
                values[key] = control.text()
        return values

    def apply_preset(self, values: dict) -> None:
        """Set every control the preset names; the rest keep their values.

        The values arrive validated, choices included, so a combo is only ever
        handed an entry it has.  The toggles' own signals then enable or
        disable their strength sliders, as a click would.
        """
        controls = self._preset_controls()
        for key, value in values.items():
            control = controls.get(key)
            if isinstance(control, Toggle):
                control.setChecked(bool(value))
            elif isinstance(control, SliderSpin):
                control.setValue(float(value))
            elif control is not None:
                control.set_text(str(value))


class InferencePage(Page):
    title = N_("Inference")
    subtitle = N_("Convert a recording to a trained voice.")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        #: Last file a conversion produced, for "Save a copy…".
        self._last_output: str = ""

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_single(), _("Single file"))
        self.tabs.addTab(self._build_batch(), _("Batch folder"))
        fit_to_current_tab(self.tabs)
        self.content.addWidget(self.tabs)
        self.content.addStretch(1)

    # -- single ------------------------------------------------------------

    def _build_single(self) -> QWidget:
        page = TabPage()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        source = Card(_("Source"), _("The audio you want converted."), icon="waveform")
        self.input_path = PathPicker(filters=_AUDIO_FILTER, placeholder=_("Drop an audio file here or browse…"))
        self.input_path.pathChanged.connect(self._on_input_changed)
        source.add(Field(
            _("Input file"),
            self.input_path,
            _("The arrow lists audios converted before and the ones in assets/audios."),
        ))

        self.input_player = AudioPlayer()
        self.input_player.accept_drops(catalog.AUDIO_EXTENSIONS)
        self.input_player.fileDropped.connect(self.input_path.set_path)
        source.add(self.input_player)
        layout.addWidget(source)

        model_card = Card(_("Voice"), _("Which model to convert into."), icon="mic")
        self.selector = ModelSelector(self)
        model_card.add(self.selector)
        layout.addWidget(model_card)

        settings_card = Card(_("Settings"), _("Everything below has a working default."), icon="sliders")
        self.settings = ConversionSettings("single")
        self.settings.notify.connect(self.notify)
        settings_card.add(self.settings)
        layout.addWidget(settings_card)

        output_card = Card(_("Output"), icon="download")
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
        # A QTabWidget gives every tab the tallest tab's height, and without a
        # stretch the extra goes to the cards: the last one grew into a tall
        # empty panel under its button.
        layout.addStretch(1)

        return page

    def _save_copy(self) -> None:
        """Copy the last result to a file the user names."""
        source = self.output_player.path() or self._last_output
        if not source or not os.path.isfile(source):
            self.notify.emit("error", _("There is no converted file to save yet."))
            return

        folder = Path(str(prefs.get("last_export_dir", "")) or str(Path.home()))
        suffix = Path(source).suffix
        # Suggest a name that is free: the obvious collision here is copying
        # two takes of the same input into one folder, and a default that
        # overwrites is one Enter away from losing the first.  Replacing an
        # existing file is still possible, but only by picking it, and the
        # dialog asks before it does.
        suggested = folder / Path(source).name
        index = 2
        while suggested.exists():
            suggested = suggested.with_name(f"{Path(source).stem} ({index}){suffix}")
            index += 1

        extension = suffix.lstrip(".").lower()
        filters = f"{extension.upper()} (*.{extension});;{_('All files')} (*)" if extension else ""
        chosen, _chosen_filter = QFileDialog.getSaveFileName(
            self, _("Save a copy as"), str(suggested), filters
        )
        if not chosen:
            return

        destination = Path(chosen)
        # This is a copy, not a conversion: the bytes stay in the source's
        # format, so the name has to say so.
        if suffix and destination.suffix.lower() != suffix.lower():
            destination = destination.with_name(destination.name + suffix)

        if os.path.abspath(destination) != os.path.abspath(source):
            try:
                shutil.copy2(source, destination)
            except OSError as error:
                self.notify.emit("error", _("Could not save the copy: {error}").format(error=error))
                return

        prefs.set("last_export_dir", str(destination.parent))
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
            _("An input file"): self.input_path.path(),
            _("A voice model"): values["pth_path"],
            _("An output path"): self.output_path.path(),
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
        # Out of the players before the backend rewrites it: converting the
        # same input again targets the file the output player still holds, and
        # an output path pointed at the input would do the same to the other.
        written = catalog.conversion_outputs(args["output_path"], args["export_format"])
        self.output_player.release(*written)
        self.input_player.release(*written)
        _remember_input(args["input_path"])
        self.input_path.set_suggestions(_input_suggestions())

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
        page = TabPage()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        folders = Card(_("Folders"), _("Every audio file in the input folder is converted."), icon="folder")
        self.batch_input = PathPicker(mode="dir", placeholder=_("Folder with audio to convert"))
        self.batch_output = PathPicker(mode="dir", placeholder=_("Where to write the results"))
        self.batch_output.set_path(str(paths.AUDIO_DIR / "batch"))
        folders.add(
            Field(_("Input folder"), self.batch_input, ""),
            Field(_("Output folder"), self.batch_output, ""),
        )
        layout.addWidget(folders)

        model_card = Card(_("Voice"), icon="mic")
        self.batch_selector = ModelSelector(self)
        model_card.add(self.batch_selector)
        layout.addWidget(model_card)

        settings_card = Card(_("Settings"), icon="sliders")
        self.batch_settings = ConversionSettings("batch")
        self.batch_settings.notify.connect(self.notify)
        settings_card.add(self.batch_settings)
        layout.addWidget(settings_card)

        run_card = Card()
        self.batch_button = primary_button(_("Convert folder"))
        self.batch_button.clicked.connect(self._convert_batch)
        self.batch_hint = QLabel("")
        self.batch_hint.setObjectName("FieldHint")
        run_card.add(self.batch_button, self.batch_hint)
        layout.addWidget(run_card)
        # See ``_build_single``: this tab is the shorter one, so it is the one
        # whose last card was stretched to the single-file tab's height.
        layout.addStretch(1)

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
            _("An input folder"): self.batch_input.path(),
            _("An output folder"): self.batch_output.path(),
            _("A voice model"): values["pth_path"],
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
        self.input_path.set_suggestions(_input_suggestions())
        # A preset saved from the other tab, or from the web interface, since.
        self.settings.presets.refresh()
        self.batch_settings.presets.refresh()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        super().apply_theme(tokens)
