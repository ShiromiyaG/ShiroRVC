"""Rectified flow: conversion and training, apart from the RVC screens.

Mirrors the Gradio "Rectified" tab: the flow model and its vocoder are
separate files, and training runs at the recipe's single rate of 44.1 kHz.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from ..services import catalog, paths
from ..widgets.audio import AudioPlayer
from ..widgets.forms import (
    Card,
    Collapsible,
    Field,
    PathPicker,
    SearchableCombo,
    SliderSpin,
    TabPage,
    Toggle,
    danger_button,
    fit_to_current_tab,
    ghost_button,
    primary_button,
)
from ..widgets.progress import TrainingProgress, progress_button
from .base import Page
from .inference import _AUDIO_FILTER, _input_suggestions, _remember_input, save_copy
from .training import _DEFAULT_CPU_THREADS, _MAX_CPU_THREADS, apply_precision_support

from ..i18n import _, N_

_EXPORT_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}


def _combo(items: list[str], value: str | None = None) -> SearchableCombo:
    combo = SearchableCombo(editable=False)
    combo.refresh_button.hide()
    combo.set_items(items)
    if value is not None:
        combo.set_text(value)
    return combo


def _checkpoint_field() -> tuple[SearchableCombo, Field]:
    """What a run keeps of its training checkpoints."""
    combo = SearchableCombo(editable=False)
    combo.refresh_button.hide()
    combo.set_pairs([
        (_("Latest only"), "latest"),
        (_("All"), "all"),
        (_("Don't save"), "none"),
    ])
    field = Field(
        _("Keep checkpoints"), combo,
        _("Training checkpoints hold the optimizer, so a stopped run resumes from them; "
          "the flow's are close to 1 GB each. Without them only the exports are saved, "
          "and a stopped run starts over. Exported models are always kept."),
    )
    return combo, field


def _index_for(flow_path: str) -> str:
    """The RVC index of the experiment a flow export was trained in, or ""."""
    if not flow_path:
        return ""
    folder = os.path.dirname(os.path.dirname(os.path.abspath(flow_path)))
    return catalog.guess_index_for(os.path.join(folder, os.path.basename(flow_path)))


class RectifiedPage(Page):
    title = N_("Rectified flow")
    subtitle = N_("Convert with and train the rectified-flow voice model and its mel vocoder.")

    #: As ``TrainingPage.progressActive``: the window floats the card elsewhere.
    progressActive = Signal(bool)
    FINISHED_HOLD_MS = 8000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        #: The flow export or bundle, and the flow inside it, whose speakers
        #: and vocoder were last read.
        self._flow_for_info: str | None = None
        self._submodel_for_info: str | None = None
        #: ``"flow"`` or ``"vocoder"`` while a run is going; core tracks one.
        self._training_part: str | None = None
        #: Whether the progress card has a run on it, live or just finished.
        self._progress_used = False

        # A layout of its own, so the card goes back to the same place above
        # the tabs after floating over another page.
        self._progress_slot = QVBoxLayout()
        self._progress_slot.setContentsMargins(0, 0, 0, 0)
        self.progress = TrainingProgress()
        self.progress.hide()
        self.progress.stopRequested.connect(self._stop_training)
        self._progress_slot.addWidget(self.progress)
        self.content.addLayout(self._progress_slot)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._dismiss_progress)
        self.engine.log.connect(self._on_engine_log)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_inference(), _("Inference"))
        self.tabs.addTab(self._build_training(), _("Training"))
        fit_to_current_tab(self.tabs)
        self.content.addWidget(self.tabs)
        self.content.addStretch(1)

        self.engine.ready.connect(self._on_engine_ready)
        self._refresh_inference()
        self._refresh_training()

    @property
    def _training(self) -> bool:
        """Read by the window before closing."""
        return self._training_part is not None

    # -- inference ---------------------------------------------------------

    def _build_inference(self) -> QWidget:
        page = TabPage()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)

        source = Card(_("Source"), _("The audio you want converted."), icon="waveform")
        self.input_path = PathPicker(filters=_AUDIO_FILTER, placeholder=_("Drop an audio file here or browse…"))
        self.input_path.pathChanged.connect(self._on_input_changed)
        source.add(Field(_("Input file"), self.input_path, ""))
        self.input_player = AudioPlayer()
        self.input_player.accept_drops(catalog.AUDIO_EXTENSIONS)
        self.input_player.fileDropped.connect(self.input_path.set_path)
        source.add(self.input_player)
        layout.addWidget(source)

        models = Card(_("Models"), _("The flow model and the vocoder that renders its mel."), icon="mic")
        self.flow_model = SearchableCombo()
        self.flow_model.refreshRequested.connect(self._rescan_inference)
        self.flow_model.currentTextChanged.connect(self._on_flow_changed)
        self.vocoder_model = SearchableCombo()
        self.vocoder_model.refreshRequested.connect(self._rescan_inference)
        self.index_file = SearchableCombo()
        self.index_file.refreshRequested.connect(self._rescan_inference)
        self.speaker = _combo(["0"])
        self.flow_submodel = _combo([])
        self.flow_submodel.currentTextChanged.connect(self._on_submodel_changed)
        self.flow_submodel_field = Field(
            _("Bundle model"), self.flow_submodel, _("The flow to use inside the bundle.")
        )
        models.add_row(
            Field(_("Flow model"), self.flow_model,
                  _("Rectified-flow export, or a model bundle (.srvc) holding one.")),
            Field(_("Speaker"), self.speaker, _("Speaker id inside a multi-speaker model.")),
            stretch=[3, 1],
        )
        models.add(
            self.flow_submodel_field,
            Field(_("Vocoder model"), self.vocoder_model,
                  _("Rectified NSF-BigVGAN export or OpenVPI NSF-HiFiGAN checkpoint. "
                    "Follows the vocoder the flow was trained with when a file of that name is here.")),
            Field(_("Index"), self.index_file,
                  _("Optional RVC index over the voice model's training features. "
                    "A bundle's own index is used when it has one.")),
        )
        self.flow_submodel_field.setVisible(False)
        self.unload_button = ghost_button(_("Unload the models"))
        self.unload_button.clicked.connect(self._unload)
        models.add(self.unload_button)
        layout.addWidget(models)

        settings = Card(_("Settings"), _("Everything below has a working default."), icon="sliders")
        self._build_inference_settings(settings)
        layout.addWidget(settings)

        output = Card(_("Output"), icon="download")
        self.output_path = PathPicker(mode="save", filters=_AUDIO_FILTER)
        self.output_path.set_path(str(paths.AUDIO_DIR / f"filename{catalog.OUTPUT_SUFFIX}.wav"))
        output.add(Field(_("Save to"), self.output_path,
                         _("A file name, or a folder. The extension picks the export format.")))
        self.convert_button = primary_button(_("Convert"))
        self.convert_button.clicked.connect(self._convert)
        output.add(self.convert_button)
        self.output_player = AudioPlayer()
        self.output_player.enable_saving()
        self.output_player.saveRequested.connect(
            lambda: save_copy(self, self.output_player.path())
        )
        output.add(self.output_player)
        layout.addWidget(output)
        layout.addStretch(1)
        return page

    def _build_inference_settings(self, card: Card) -> None:
        self.pitch = SliderSpin(-24, 24, 1, decimals=0, value=0)
        self.f0_method = _combo(catalog.F0_METHODS, "rmvpe")
        self.steps = SliderSpin(1, 64, 1, decimals=0, value=16)
        self.cfg_scale = SliderSpin(1.0, 5.0, 0.1, decimals=1, value=1.0)
        self.index_rate = SliderSpin(0, 1, 0.01, decimals=2, value=0.5)
        self.protect = SliderSpin(0, 0.5, 0.01, decimals=2, value=0.33)

        card.add(Field(_("Pitch (semitones)"), self.pitch, _("12 = one octave up, -12 = one octave down.")))
        card.add_row(
            Field(_("Pitch algorithm"), self.f0_method, _("rmvpe is the recommended default.")),
            Field(_("Steps"), self.steps, _("ODE steps from noise to mel. More is slower and usually cleaner.")),
        )
        card.add_row(
            Field(_("Speaker guidance"), self.cfg_scale,
                  _("Classifier-free guidance on the speaker. 1 turns it off.")),
            Field(_("Search feature ratio"), self.index_rate, _("Index influence.")),
            Field(_("Protect"), self.protect, _("Shields voiceless consonants from the index.")),
        )

        advanced = Collapsible(_("Advanced settings"), _("pitch, sampling, guidance, index, output"))
        card.add(advanced)

        advanced.add_group(_("Pitch"))
        self.autotune = Toggle(_("Autotune"), _("Snap pitch to the chromatic grid. For singing."))
        self.autotune_strength = SliderSpin(0, 1, 0.01, decimals=2, value=1.0)
        self.formant_shift = SliderSpin(-5, 5, 0.5, decimals=1, value=0.0)
        self.f0_median = SliderSpin(0, 10, 1, decimals=0, value=0)
        self.tension_strength = SliderSpin(0, 1, 0.05, decimals=2, value=1.0)
        self.f0_octave_fix = Toggle(
            _("Fix octave errors"),
            _("Folds pitch that jumps an octave away from its surroundings back into place."),
        )
        advanced.add(self.autotune, Field(_("Autotune strength"), self.autotune_strength, ""))
        advanced.add_row(
            Field(_("Formant shift"), self.formant_shift,
                  _("Moves the formants by semitones, apart from the pitch.")),
            Field(_("Pitch median filter"), self.f0_median,
                  _("10 ms frames either side of each voiced frame, against jitter. 0 turns it off.")),
        )
        advanced.add(
            Field(_("Tension"), self.tension_strength,
                  _("How far the input's tension carries over. 0 leaves the voice at its own. "
                    "Only for models trained with the tension input.")),
            self.f0_octave_fix,
        )

        advanced.add_group(_("Sampling"))
        self.sampler = _combo(catalog.RECTIFIED_SAMPLERS)
        self.schedule = _combo(catalog.RECTIFIED_SCHEDULES)
        self.noise_temperature = SliderSpin(0.0, 1.5, 0.05, decimals=2, value=1.0)
        self.flow_start = SliderSpin(0.0, 0.95, 0.05, decimals=2, value=0.0)
        advanced.add_row(
            Field(_("Sampler"), self.sampler, _("Heun costs two model passes per step.")),
            Field(_("Step schedule"), self.schedule,
                  _("Sway puts more steps early; logit-normal matches the times trained on most.")),
        )
        advanced.add_row(
            Field(_("Noise temperature"), self.noise_temperature,
                  _("Scale of the starting noise. 1 is what the model was trained on.")),
            Field(_("Flow start"), self.flow_start,
                  _("Flow time sampling starts from, on the aux decoder's mel. 0 uses the model's own.")),
        )

        advanced.add_group(_("Guidance"))
        self.content_guidance = SliderSpin(0.0, 1.0, 0.05, decimals=2, value=0.0)
        self.guidance_from = SliderSpin(0.0, 1.0, 0.05, decimals=2, value=0.0)
        self.guidance_until = SliderSpin(0.0, 1.0, 0.05, decimals=2, value=1.0)
        self.guidance_rescale = SliderSpin(0.0, 1.0, 0.05, decimals=2, value=0.7)
        self.rescale_mode = _combo(catalog.RECTIFIED_RESCALE_MODES)
        advanced.add(Field(_("Content guidance"), self.content_guidance,
                           _("Pushes away from a blurred copy of the content. 0 turns it off.")))
        advanced.add_row(
            Field(_("Guidance from"), self.guidance_from, _("Flow time (0 noise, 1 mel) guidance starts at.")),
            Field(_("Guidance until"), self.guidance_until, _("Flow time guidance stops at.")),
        )
        advanced.add_row(
            Field(_("Guidance rescale"), self.guidance_rescale,
                  _("Pulls guided output back to the unguided level.")),
            Field(_("Rescale over"), self.rescale_mode,
                  _("Global measures the whole pass, silences included; frame measures per frame.")),
        )

        advanced.add_group(_("Index retrieval"))
        self.index_k = SliderSpin(1, 32, 1, decimals=0, value=8)
        self.index_power = SliderSpin(0, 8, 0.25, decimals=2, value=2.0)
        self.index_continuity = SliderSpin(0, 4, 0.1, decimals=2, value=0.5)
        advanced.add_row(
            Field(_("Neighbours"), self.index_k, _("Frames averaged per match.")),
            Field(_("Sharpness"), self.index_power, _("How strongly closer matches outweigh further ones.")),
            Field(_("Continuity"), self.index_continuity,
                  _("Favours matches that continue the previous frame's. Needs an index built by this fork.")),
        )

        advanced.add_group(_("Output"))
        self.export_format = _combo(catalog.EXPORT_FORMATS)
        self.seed = SliderSpin(0, 2**16, 1, decimals=0, value=0)
        self.split_audio = Toggle(_("Split long audio"), _("Split the input at silences."))
        self.silence_gate_db = SliderSpin(-120, 0, 1, decimals=0, value=-60)
        self.content_context = SliderSpin(0.0, 10.0, 0.5, decimals=1, value=2.0)
        advanced.add_row(
            Field(_("Export format"), self.export_format, ""),
            Field(_("Seed"), self.seed, _("0 picks a fresh seed each run.")),
        )
        advanced.add(self.split_audio)
        advanced.add_row(
            Field(_("Silence gate"), self.silence_gate_db,
                  _("Fades the output out where the input is quieter than this, in dBFS. -120 turns it off.")),
            Field(_("Content context"), self.content_context,
                  _("Seconds the content encoder sees either side of each 30 s pass.")),
        )

        self.autotune.toggled.connect(self.autotune_strength.setEnabled)
        self.autotune_strength.setEnabled(False)

    def inference_args(self) -> dict:
        """Every argument ``core.run_rectified_infer_script`` takes but the paths."""
        return {
            "flow_path": self.flow_model.text(),
            "flow_submodel": self.flow_submodel.text() if self.flow_submodel_field.isVisibleTo(self) else "",
            "vocoder_path": self.vocoder_model.text(),
            "index_path": self.index_file.text(),
            "sid": int(self.speaker.text() or 0),
            "pitch": int(self.pitch.value()),
            "f0_method": self.f0_method.text(),
            "steps": int(self.steps.value()),
            "sampler": self.sampler.text(),
            "cfg_scale": self.cfg_scale.value(),
            "f0_autotune": self.autotune.isChecked(),
            "f0_autotune_strength": self.autotune_strength.value(),
            "seed": int(self.seed.value()),
            "export_format": self.export_format.text(),
            "index_rate": self.index_rate.value(),
            "index_k": int(self.index_k.value()),
            "index_power": self.index_power.value(),
            "index_continuity": self.index_continuity.value(),
            "protect": self.protect.value(),
            "formant_shift": self.formant_shift.value(),
            "content_guidance": self.content_guidance.value(),
            "guidance_rescale": self.guidance_rescale.value(),
            "split_audio": self.split_audio.isChecked(),
            "silence_gate_db": self.silence_gate_db.value(),
            "noise_temperature": self.noise_temperature.value(),
            "flow_start": self.flow_start.value(),
            "guidance_from": self.guidance_from.value(),
            "guidance_until": self.guidance_until.value(),
            "rescale_mode": self.rescale_mode.text(),
            "schedule": self.schedule.text(),
            "f0_median": int(self.f0_median.value()),
            "f0_octave_fix": self.f0_octave_fix.isChecked(),
            "content_context": self.content_context.value(),
            "tension_strength": self.tension_strength.value(),
        }

    def _refresh_inference(self) -> None:
        self.flow_model.set_items(catalog.list_rectified_exports("flow"))
        self.vocoder_model.set_items(catalog.list_rectified_exports("vocoder"))
        self.index_file.set_items([""] + catalog.list_indexes())
        self.input_path.set_suggestions(_input_suggestions())

    def _rescan_inference(self) -> None:
        self._flow_for_info = self._submodel_for_info = None
        self._refresh_inference()

    def _on_engine_ready(self) -> None:
        self._on_flow_changed(self.flow_model.text())

    def _on_flow_changed(self, flow: str) -> None:
        """A bundle's flows, the speakers, paired vocoder and index follow the
        flow model."""
        if not flow or flow == self._flow_for_info:
            return
        guess = _index_for(flow)
        if guess:
            self.index_file.set_text(guess)

        def arrived(data: dict) -> None:
            if self.flow_model.text() != flow:
                return
            self._flow_for_info = flow
            # Set first, so the picker's own change signal below is not taken
            # for a new choice.
            self._submodel_for_info = data.get("submodel") or ""
            submodels = data.get("submodels") or []
            self.flow_submodel.set_items(submodels)
            self.flow_submodel_field.setVisible(bool(submodels))
            self._apply_flow_info(data)

        def failed(_error: str) -> None:
            self._flow_for_info = None
            self.speaker.set_items(["0"])

        self.engine.call("rectified_flow_info", {"flow_path": flow}, on_result=arrived, on_error=failed)

    def _on_submodel_changed(self, name: str) -> None:
        flow = self.flow_model.text()
        if not name or name == self._submodel_for_info or flow != self._flow_for_info:
            return
        self._submodel_for_info = name

        def arrived(data: dict) -> None:
            if self.flow_model.text() == flow and self.flow_submodel.text() == name:
                self._apply_flow_info(data)

        self.engine.call(
            "rectified_flow_info", {"flow_path": flow, "sub_model": name},
            on_result=arrived, on_error=lambda _error: self.speaker.set_items(["0"]),
        )

    def _apply_flow_info(self, data: dict) -> None:
        self.speaker.set_items([str(sid) for sid in data.get("speakers", [0])])
        if data.get("vocoder"):
            self.vocoder_model.set_text(data["vocoder"])

    def _on_input_changed(self, path: str) -> None:
        if path and os.path.isfile(path):
            self.input_player.load(path)
            self.output_path.set_path(catalog.default_output_path(path))
        else:
            self.input_player.clear()

    def _unload(self) -> None:
        self.run("rectified_unload", {}, busy_text=_("Unloading models…"), buttons=[self.unload_button])

    def _convert(self) -> None:
        args = self.inference_args()
        audio = self.input_path.path()
        if not self.require(**{
            _("An input file"): audio,
            _("A flow model"): args["flow_path"],
            _("A vocoder model"): args["vocoder_path"],
        }):
            return
        if not os.path.isfile(audio):
            self.notify.emit("error", _("The input file does not exist."))
            return

        # As the Gradio tab: a folder takes the default name, and an audio
        # extension picks the export format; the pipeline writes .wav first.
        output = self.output_path.path().strip()
        if not output:
            output = catalog.default_output_path(audio)
        elif os.path.isdir(output):
            output = os.path.join(output, os.path.basename(catalog.default_output_path(audio)))
        else:
            stem, ext = os.path.splitext(output)
            if ext.lower().lstrip(".") in _EXPORT_EXTENSIONS:
                args["export_format"] = ext.lstrip(".").upper()
            output = stem + ".wav"
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

        written = catalog.conversion_outputs(output, args["export_format"])
        self.output_player.release(*written)
        self.input_player.release(*written)
        _remember_input(audio)
        self.input_path.set_suggestions(_input_suggestions())

        self.run(
            "rectified_infer",
            {**args, "input_path": audio, "output_path": output},
            busy_text=_("Converting…"),
            buttons=[self.convert_button],
            on_result=self._on_converted,
        )

    def _on_converted(self, data: dict) -> None:
        preview = data.get("preview")
        if preview and os.path.isfile(preview):
            self.output_player.load(preview)

    # -- training ----------------------------------------------------------

    def _build_training(self) -> QWidget:
        page = TabPage()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(18)
        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_preprocess_card())
        layout.addWidget(self._build_extract_card())
        layout.addWidget(self._build_flow_card())
        layout.addWidget(self._build_index_card())
        layout.addWidget(self._build_vocoder_card())
        layout.addStretch(1)
        return page

    def _build_model_card(self) -> Card:
        card = Card(_("1 · Model"), _("Its data lives in logs/<name>, like an RVC model."), icon="chip")
        self.model_name = SearchableCombo()
        self.model_name.refreshRequested.connect(self._refresh_training)
        self.model_name.currentTextChanged.connect(self._on_model_changed)
        self.gpu = SearchableCombo()
        self.gpu.refresh_button.hide()
        self.gpu.set_items(["0"])
        self.precision = SearchableCombo(editable=False)
        self.precision.refresh_button.hide()
        self.precision.set_pairs([("FP32", "fp32"), ("FP16", "fp16"), ("BF16", "bf16")])
        self.precision.setEnabled(False)
        card.add_row(
            Field(_("Model name"), self.model_name, _("Reuse a name to continue a run.")),
            Field(_("GPU"), self.gpu, _("Separate several with '-', e.g. 0-1. Batch size is per GPU.")),
            Field(_("Precision"), self.precision, _("Training precision of both models.")),
            stretch=[2, 1, 1],
        )
        rate = QLabel(_("Sample rate: {rate} Hz. The rectified recipe has one configuration.")
                      .format(rate=catalog.RECTIFIED_SAMPLE_RATE))
        rate.setObjectName("FieldHint")
        card.add(rate)
        return card

    def _build_preprocess_card(self) -> Card:
        card = Card(_("2 · Preprocess"), _("Slice and normalise the dataset."), icon="waveform")
        self.dataset = PathPicker(mode="dir", placeholder=_("Folder with your training audio"))
        card.add(Field(_("Dataset folder"), self.dataset, _("Every audio file below this folder is used.")))

        self.cut_preprocess = _combo(catalog.CUT_PREPROCESS, "New Automatic")
        self.chunk_len = SliderSpin(0.5, 30.0, 0.1, decimals=1, value=3.0)
        self.overlap_len = SliderSpin(0.0, 0.42, 0.01, decimals=2, value=0.40)
        card.add_row(
            Field(_("Slicing"), self.cut_preprocess, ""),
            Field(_("Chunk length (s)"), self.chunk_len, _("For Simple cutting.")),
            Field(_("Overlap (s)"), self.overlap_len, _("For Simple cutting.")),
        )

        advanced = Collapsible(_("Advanced settings"), _("normalisation, cleanup, format, threads"))
        card.add(advanced)
        self.normalization = _combo(catalog.NORMALIZATION_MODES, "pre_peak_rvc")
        self.rms_db = SliderSpin(-24, -3, 1, decimals=0, value=-16)
        self.resampling = _combo(catalog.LOADING_RESAMPLING, "ffmpeg")
        advanced.add_row(
            Field(_("Normalisation"), self.normalization, ""),
            Field(_("Target level (LUFS)"), self.rms_db, _("Used by pre_loudness only.")),
            Field(_("Resampler"), self.resampling, ""),
        )
        self.process_effects = Toggle(_("DC / high-pass filtering"), "", checked=True)
        self.noise_reduction = Toggle(_("Noise reduction"), "")
        self.clean_strength = SliderSpin(0, 1, 0.01, decimals=2, value=0.5)
        advanced.add(self.process_effects, self.noise_reduction,
                     Field(_("Noise reduction strength"), self.clean_strength, ""))
        self.dataset_format = _combo(["WAV", "FLAC"])
        self.cpu_threads = SliderSpin(1, _MAX_CPU_THREADS, 1, decimals=0, value=_DEFAULT_CPU_THREADS)
        advanced.add_row(
            Field(_("Dataset format"), self.dataset_format, ""),
            Field(_("CPU threads"), self.cpu_threads, _("Also used by extraction.")),
        )
        self.noise_reduction.toggled.connect(self.clean_strength.setEnabled)
        self.clean_strength.setEnabled(False)

        self.preprocess_button = progress_button(_("Run preprocessing"))
        self.preprocess_button.clicked.connect(self._preprocess)
        card.add(self.preprocess_button)
        return card

    def _build_extract_card(self) -> Card:
        card = Card(_("3 · Extract features"), _("Pitch curves and content embeddings."), icon="search")
        self.extract_f0 = _combo(catalog.F0_METHODS, "rmvpe")
        self.extract_embedder = _combo(catalog.TRAINING_EMBEDDER_MODELS)
        self.include_mutes = SliderSpin(0, 10, 1, decimals=0, value=5)
        self.feature_precision = _combo(catalog.FEATURE_PRECISIONS)
        card.add_row(
            Field(_("Pitch algorithm"), self.extract_f0, ""),
            Field(_("Embedder"), self.extract_embedder, _("Content features the flow model is conditioned on.")),
        )
        card.add_row(
            Field(_("Mute samples"), self.include_mutes, _("Silent examples so the model can reproduce silence.")),
            Field(_("Feature precision"), self.feature_precision, _("fp16 halves the cache on disk.")),
        )
        self.extract_button = progress_button(_("Run extraction"))
        self.extract_button.clicked.connect(self._extract)
        card.add(self.extract_button)
        return card

    def _run_controls(self, card: Card, epochs: int = 250) -> tuple[SliderSpin, SliderSpin, SliderSpin]:
        total = SliderSpin(1, 10000, 1, decimals=0, value=epochs)
        batch = SliderSpin(1, 128, 1, decimals=0, value=16)
        save = SliderSpin(1, 100, 1, decimals=0, value=10)
        card.add_row(
            Field(_("Total epochs"), total, ""),
            Field(_("Batch size"), batch, _("Lower it if training runs out of VRAM.")),
            Field(_("Save every N epochs"), save, ""),
        )
        return total, batch, save

    def _start_stop(self, card: Card, start) -> tuple:
        row = QHBoxLayout()
        row.setSpacing(12)
        start_button = primary_button(_("Start training"))
        start_button.clicked.connect(start)
        stop_button = danger_button(_("Stop training"))
        stop_button.clicked.connect(self._stop_training)
        stop_button.setVisible(False)
        row.addWidget(start_button, 1)
        row.addWidget(stop_button, 1)
        card.body.addLayout(row)
        return start_button, stop_button

    def _build_flow_card(self) -> Card:
        card = Card(_("4 · Voice model"),
                    _("The rectified flow: content, pitch, loudness and speaker to a mel."), icon="trend")
        self.flow_epochs, self.flow_batch, self.flow_save = self._run_controls(card)

        self.flow_pretrained = Toggle(
            _("Start from a pretrained flow"),
            _("The default pretrain for the embedder the experiment was extracted with "
              "(pretrain_flow_contentvec.pth or pretrain_flow_spin_v2.pth). Off trains from scratch."),
            checked=True,
        )
        self.flow_custom = Toggle(_("Use my own pretrained flow"), "")
        self.custom_flow = PathPicker(filters="Flow (*.pth)")
        self.custom_flow_field = Field(
            _("Pretrained flow"), self.custom_flow,
            _("Its speakers are replaced by this dataset's."),
        )
        self.flow_vocoder = PathPicker(filters="Vocoder (*.pth *.ckpt)")
        card.add(
            self.flow_pretrained, self.flow_custom, self.custom_flow_field,
            Field(_("Vocoder"), self.flow_vocoder,
                  _("Renders the previews and is paired with the exports. Empty uses the newest pretrained one.")),
        )

        advanced = Collapsible(_("Advanced settings"), _("learning rate, compilation, mean flow, checkpoints"))
        card.add(advanced)
        self.flow_learning_rate = SliderSpin(0, 0.01, 0.00001, decimals=5, value=0)
        advanced.add(Field(_("Learning rate"), self.flow_learning_rate, _("0 uses the config's rate.")))
        self.flow_compile = Toggle(
            _("torch.compile the backbone"),
            _("The first steps take longer while the graph is built. Needs CUDA and Triton."),
        )
        self.flow_compile_mode = _combo(catalog.TORCH_COMPILE_MODES)
        self.flow_compile_mode_field = Field(_("Compile mode"), self.flow_compile_mode, "")
        self.flow_checkpoints, flow_checkpoints_field = _checkpoint_field()
        self.flow_fresh = Toggle(_("Fresh training"), _("Ignore this run's checkpoints and start over."))
        self.flow_mean = Toggle(
            _("Mean flow"),
            _("Also train the mean velocity (MeanFlow), which the mean sampler takes in one or two "
              "steps. The other samplers work as before. A resumed run keeps what it started with."),
        )
        advanced.add(self.flow_compile, self.flow_compile_mode_field, self.flow_mean,
                     flow_checkpoints_field, self.flow_fresh)

        self.flow_compile_mode_field.setVisible(False)
        self.flow_compile.toggled.connect(self.flow_compile_mode_field.setVisible)
        self.flow_pretrained.toggled.connect(self._sync_flow_pretrained)
        self.flow_custom.toggled.connect(self._sync_flow_pretrained)
        self._sync_flow_pretrained()

        self.flow_start_button, self.flow_stop_button = self._start_stop(card, self._start_flow)
        return card

    def _sync_flow_pretrained(self) -> None:
        pretrained = self.flow_pretrained.isChecked()
        self.flow_custom.setVisible(pretrained)
        self.custom_flow_field.setVisible(pretrained and self.flow_custom.isChecked())

    def _build_index_card(self) -> Card:
        card = Card(_("5 · Index"),
                    _("The same retrieval index as RVC; one built in the Training page for this experiment works as is."),
                    icon="folder")
        self.index_algorithm = _combo(catalog.INDEX_ALGORITHMS)
        self.index_metric = _combo(catalog.INDEX_METRICS)
        card.add_row(
            Field(_("Algorithm"), self.index_algorithm, _("Auto picks by dataset size.")),
            Field(_("Similarity"), self.index_metric, _("l2 is what upstream RVC builds; cosine compares direction only.")),
        )
        self.index_single_speaker = Toggle(_("Index one speaker only"), "")
        self.index_speaker = _combo([])
        self.index_speaker_field = Field(_("Speaker"), self.index_speaker, _("Read from the extracted features."))
        self.index_speaker_field.setVisible(False)
        self.index_single_speaker.toggled.connect(self._on_index_speaker_toggled)
        card.add(self.index_single_speaker, self.index_speaker_field)
        self.index_button = primary_button(_("Generate index"))
        self.index_button.clicked.connect(self._build_index)
        card.add(self.index_button)
        return card

    def _build_vocoder_card(self) -> Card:
        card = Card(
            _("Vocoder Pretrain"),
            _("Builds the NSF-BigVGAN vocoder pretrain, on a large multi-speaker dataset. Not needed "
              "to fine-tune a voice: the vocoder has no speaker input, so one pretrain renders every voice."),
            icon="volume",
        )
        self.voc_epochs, self.voc_batch, self.voc_save = self._run_controls(card)
        self.pretrained_g = PathPicker(filters="Generator (*.pth)")
        self.pretrained_d = PathPicker(filters="Discriminator (*.pth)")
        card.add_row(
            Field(_("Pretrained generator"), self.pretrained_g, _("Optional checkpoint or export to fine-tune from.")),
            Field(_("Pretrained discriminator"), self.pretrained_d, _("Optional. The matching D checkpoint.")),
        )
        self.voc_checkpoints, voc_checkpoints_field = _checkpoint_field()
        self.voc_fresh = Toggle(_("Fresh training"), _("Ignore this run's checkpoints and start over."))
        card.add(voc_checkpoints_field, self.voc_fresh)
        self.voc_start_button, self.voc_stop_button = self._start_stop(card, self._start_vocoder)
        return card

    def _refresh_training(self) -> None:
        self.model_name.set_items(catalog.list_training_models())
        self.dataset.set_suggestions(catalog.list_dataset_folders())
        self.custom_flow.set_suggestions(catalog.list_rectified_pretrained("flow"))
        self.flow_vocoder.set_suggestions(catalog.list_rectified_exports("vocoder"))
        if not self.flow_vocoder.path():
            self.flow_vocoder.set_path(catalog.default_rectified_pretrained("vocoder"))
        self.pretrained_g.set_suggestions(catalog.list_rectified_pretrained("vocoder_g"))
        self.pretrained_d.set_suggestions(catalog.list_rectified_pretrained("vocoder_d"))

    def _on_model_changed(self, _name: str) -> None:
        if self.index_single_speaker.isChecked():
            self._refresh_index_speakers()

    def _on_index_speaker_toggled(self, enabled: bool) -> None:
        self.index_speaker_field.setVisible(enabled)
        if enabled:
            self._refresh_index_speakers()

    def _refresh_index_speakers(self) -> None:
        speakers = catalog.list_experiment_speakers(self.model_name.text().strip())
        self.index_speaker.set_items([str(sid) for sid in speakers])

    def _model(self) -> str | None:
        name = self.model_name.text().strip()
        return name if self.require(**{_("A model name"): name}) else None

    def _preprocess(self) -> None:
        name = self._model()
        if name is None or not self.require(**{_("A dataset folder"): self.dataset.path()}):
            return
        if not os.path.isdir(self.dataset.path()):
            self.notify.emit("error", _("The dataset folder does not exist."))
            return
        self.run(
            "preprocess",
            {
                "model_name": name,
                "dataset_path": self.dataset.path(),
                "sample_rate": catalog.RECTIFIED_SAMPLE_RATE,
                "cpu_threads": int(self.cpu_threads.value()),
                "cut_preprocess": self.cut_preprocess.text(),
                "process_effects": self.process_effects.isChecked(),
                "noise_reduction": self.noise_reduction.isChecked(),
                "clean_strength": self.clean_strength.value(),
                "chunk_len": self.chunk_len.value(),
                "overlap_len": self.overlap_len.value(),
                "normalization_mode": self.normalization.text(),
                "loading_resampling": self.resampling.text(),
                "dataset_format": self.dataset_format.text(),
                "rms_norm_db": self.rms_db.value(),
            },
            busy_text=_("Preprocessing dataset…"),
            buttons=[self.preprocess_button],
            progress=self.preprocess_button,
        )

    def _extract(self) -> None:
        name = self._model()
        if name is None:
            return
        self.run(
            "extract",
            {
                "model_name": name,
                "f0_method": self.extract_f0.text(),
                "cpu_threads": int(self.cpu_threads.value()),
                "gpu": self.gpu.text(),
                "sample_rate": catalog.RECTIFIED_SAMPLE_RATE,
                "vocoder_arch": catalog.RECTIFIED_EXTRACTION,
                "embedder_model": self.extract_embedder.text(),
                "include_mutes": int(self.include_mutes.value()),
                "feature_precision": self.feature_precision.text(),
            },
            busy_text=_("Extracting features…"),
            buttons=[self.extract_button],
            progress=self.extract_button,
        )

    def _build_index(self) -> None:
        name = self._model()
        if name is None:
            return
        speaker = "all"
        if self.index_single_speaker.isChecked():
            speaker = self.index_speaker.text().strip()
            if not self.require(**{_("A speaker to index"): speaker}):
                return
        self.run(
            "index",
            {
                "model_name": name,
                "index_algorithm": self.index_algorithm.text(),
                "index_metric": self.index_metric.text(),
                "index_speaker": speaker,
            },
            busy_text=_("Building index…"),
            buttons=[self.index_button],
        )

    def flow_train_args(self) -> dict | None:
        """``core.run_rectified_flow_train_script``'s arguments, or None after
        reporting what is missing."""
        name = self._model()
        if name is None:
            return None
        flow = ""
        if self.flow_pretrained.isChecked():
            if self.flow_custom.isChecked():
                flow = self.custom_flow.path()
                if not self.require(**{_("A custom pretrained flow"): flow}):
                    return None
            else:
                # The pretrain has to match the features the experiment was extracted with.
                flow, embedder, file = catalog.default_flow_pretrain(name)
                if not flow:
                    self.notify.emit("error", _(
                        "No pretrained flow for {embedder} ({file}) in "
                        "rvc/models/pretraineds/rectified. Add it, pick your own, or turn "
                        "off the pretrained start.").format(embedder=embedder, file=file or "?"))
                    return None
        return {
            "part": "flow",
            "model_name": name,
            "total_epochs": int(self.flow_epochs.value()),
            "save_every": int(self.flow_save.value()),
            "batch_size": int(self.flow_batch.value()),
            "gpu": self.gpu.text(),
            "pretrained_flow": flow,
            "vocoder": self.flow_vocoder.path(),
            "learning_rate": self.flow_learning_rate.value(),
            "checkpoints": self.flow_checkpoints.value(),
            "fresh": self.flow_fresh.isChecked(),
            "precision": self.precision.value(),
            "compile": self.flow_compile.isChecked(),
            "torch_compile_mode": self.flow_compile_mode.text(),
            "mean_flow": self.flow_mean.isChecked(),
        }

    def vocoder_train_args(self) -> dict | None:
        name = self._model()
        if name is None:
            return None
        return {
            "part": "vocoder",
            "model_name": name,
            "total_epochs": int(self.voc_epochs.value()),
            "save_every": int(self.voc_save.value()),
            "batch_size": int(self.voc_batch.value()),
            "gpu": self.gpu.text(),
            "pretrained_g": self.pretrained_g.path(),
            "pretrained_d": self.pretrained_d.path(),
            "checkpoints": self.voc_checkpoints.value(),
            "fresh": self.voc_fresh.isChecked(),
            "precision": self.precision.value(),
        }

    def _start_flow(self) -> None:
        args = self.flow_train_args()
        if args is not None:
            self._start_training(args)

    def _start_vocoder(self) -> None:
        args = self.vocoder_train_args()
        if args is not None:
            self._start_training(args)

    def _start_training(self, args: dict) -> None:
        self._training_part = args["part"]
        self._set_training_controls()
        self.busy.emit(True, _("Training…"))
        self._dismiss_timer.stop()
        self.progress.title.setText(
            _("Voice model") if args["part"] == "flow" else _("Vocoder Pretrain")
        )
        self.progress.begin(args["total_epochs"])
        self._progress_used = True
        self.progress.reveal()
        self.progressActive.emit(True)

        def finish(level: str, message: str, card_text: str) -> None:
            self._training_part = None
            self._set_training_controls()
            self.busy.emit(False, "")
            self.progress.finish(card_text)
            self._dismiss_timer.start(self.FINISHED_HOLD_MS)
            self.notify.emit(level, message)

        def on_result(data: dict) -> None:
            message = str(data.get("message") or _("Training finished."))
            finish("success", message, message)

        self.engine.call(
            "rectified_train",
            args,
            on_result=on_result,
            on_error=lambda error: finish(
                "error", error, _("Stopped: {error}").format(error=error)
            ),
        )

    def _on_engine_log(self, line: str) -> None:
        if self._training_part is not None:
            self.progress.consume(line)

    @property
    def has_progress(self) -> bool:
        return self._progress_used

    def reclaim_progress(self) -> None:
        """Take the progress card back from the window's floating dock."""
        if self.progress.parentWidget() is not self._progress_slot.parentWidget():
            self._progress_slot.insertWidget(0, self.progress)
        self.progress.setVisible(self._progress_used)

    def _dismiss_progress(self) -> None:
        if self._training_part is not None:  # a new run beat the timer to it
            return
        self._progress_used = False
        self.progress.dismiss(on_done=lambda: self.progressActive.emit(False))

    def _set_training_controls(self) -> None:
        """Only one rectified run at a time: Stop replaces the running part's
        Start, and the other part's Start is disabled."""
        running = self._training_part
        for part, start, stop in (("flow", self.flow_start_button, self.flow_stop_button),
                                  ("vocoder", self.voc_start_button, self.voc_stop_button)):
            start.setVisible(running != part)
            start.setEnabled(running is None)
            stop.setVisible(running == part)
            stop.setEnabled(running == part)
            stop.setText(_("Stop training"))
        self.progress.set_stoppable(running is not None)

    def _stop_training(self) -> None:
        for stop in (self.flow_stop_button, self.voc_stop_button):
            stop.setEnabled(False)
            stop.setText(_("Stopping…"))
        self.log.emit(_("Stop requested; waiting for the current checkpoint to finish writing."))
        self.engine.call(
            "stop_rectified_train",
            {},
            on_result=lambda data: self.notify.emit("info", str(data.get("message") or _("Stop requested."))),
            on_error=lambda error: self.notify.emit("error", error),
        )

    # -- lifecycle ---------------------------------------------------------

    def on_shown(self) -> None:
        self._refresh_inference()
        self._refresh_training()

    def populate_gpus(self, devices: list[dict]) -> None:
        apply_precision_support(self.precision, devices)
        if not devices:
            return
        self.gpu.set_items([str(device["index"]) for device in devices])
        for position, device in enumerate(devices):
            self.gpu.combo.setItemData(
                position,
                f"{device['name']} · {device['total_vram'] / 2**30:.0f} GB",
                Qt.ToolTipRole,
            )
