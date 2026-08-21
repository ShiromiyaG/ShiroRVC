"""Text to speech, then conversion to a trained voice."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QWidget

from ..services import catalog, paths
from ..widgets.audio import AudioPlayer
from ..widgets.forms import (
    Card,
    Field,
    PathPicker,
    SearchableCombo,
    SliderSpin,
    primary_button,
)
from .base import Page
from .inference import ConversionSettings, ModelSelector

from ..i18n import _, N_, ngettext


class TtsPage(Page):
    title = N_("Text to speech")
    subtitle = N_("Synthesise a line with Edge TTS, then convert it to your voice.")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        text_card = Card(_("Text"), _("Type a line, or load a .txt file to read from."))
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(_("What should the voice say?"))
        self.text.setMinimumHeight(110)
        self.text.textChanged.connect(self._update_counter)
        text_card.add(self.text)

        self.counter = QLabel(ngettext("{count} character", "{count} characters", 0).format(count=0))
        self.counter.setObjectName("FieldHint")
        text_card.add(self.counter)

        self.text_file = PathPicker(
            filters="Text files (*.txt);;All files (*.*)",
            placeholder=_("Optional: read the text from a file instead"),
        )
        text_card.add(Field(_("Text file"), self.text_file, _("When set, this wins over the box above.")))
        self.content.addWidget(text_card)

        voice_card = Card(_("Synthesis voice"), _("The Edge TTS voice used before conversion."))
        self.voice = SearchableCombo()
        self.voice.refresh_button.hide()
        self.voice.set_pairs(catalog.tts_voices())
        self.rate = SliderSpin(-100, 100, 1, decimals=0, value=0)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(Field(_("Voice"), self.voice, _("Pick one close to your target accent; conversion handles timbre.")), 3)
        row.addWidget(Field(_("Speaking rate"), self.rate, _("Percent faster or slower than default.")), 1)
        voice_card.body.addLayout(row)
        self.content.addWidget(voice_card)

        model_card = Card(_("Target voice"))
        self.selector = ModelSelector(self)
        model_card.add(self.selector)
        self.content.addWidget(model_card)

        settings_card = Card(_("Conversion settings"))
        self.settings = ConversionSettings("tts")
        # These have no meaning for a synthesised source: there is no formant
        # mismatch to correct and no external pitch curve to apply.
        self.settings.formant.setChecked(False)
        self.settings.formant.hide()
        self.settings.f0_file.hide()
        settings_card.add(self.settings)
        self.content.addWidget(settings_card)

        output_card = Card(_("Output"))
        self.tts_output = PathPicker(mode="save", filters="Audio (*.wav)")
        self.tts_output.set_path(str(paths.AUDIO_DIR / catalog.TTS_RAW_NAME))
        self.rvc_output = PathPicker(mode="save", filters="Audio (*.wav)")
        self.rvc_output.set_path(str(paths.AUDIO_DIR / catalog.TTS_CONVERTED_NAME))
        output_card.add(
            Field(_("Synthesised speech"), self.tts_output, _("Intermediate file, before conversion.")),
            Field(_("Converted speech"), self.rvc_output, _("The final result.")),
        )

        self.generate_button = primary_button(_("Synthesise and convert"))
        self.generate_button.clicked.connect(self._generate)
        output_card.add(self.generate_button)

        self.player = AudioPlayer()
        output_card.add(self.player)
        self.content.addWidget(output_card)
        self.content.addStretch(1)

    def _update_counter(self) -> None:
        count = len(self.text.toPlainText())
        self.counter.setText(
            ngettext("{count} character", "{count} characters", count).format(count=count)
        )

    def _generate(self) -> None:
        values = self.selector.values()
        text = self.text.toPlainText().strip()
        text_file = self.text_file.path()

        if not text and not text_file:
            self.notify.emit("error", _("Enter some text, or pick a text file."))
            return
        if not self.require(**{
            "A voice model": values["pth_path"],
            "An output path": self.rvc_output.path(),
        }):
            return

        settings = self.settings.values()
        # run_tts_script does not take these; the synthesised source has no
        # formants to shift and no latent to temperature-scale.
        for key in ("formant_shifting", "formant_qfrency", "formant_timbre",
                    "deterministic", "latent_temperature"):
            settings.pop(key, None)
        values.pop("bundle_submodel", None)

        args = {
            **values,
            **settings,
            "tts_file": text_file or "",
            "tts_text": text,
            "tts_voice": self.voice.value(),
            "tts_rate": int(self.rate.value()),
            "output_tts_path": self.tts_output.path(),
            "output_rvc_path": self.rvc_output.path(),
        }
        for path in (args["output_tts_path"], args["output_rvc_path"]):
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        self.run(
            "tts",
            args,
            busy_text=_("Synthesising…"),
            buttons=[self.generate_button],
            on_result=self._on_done,
        )

    def _on_done(self, data: dict) -> None:
        output = data.get("output") or self.rvc_output.path()
        if output and os.path.isfile(output):
            self.player.load(output)

    def on_shown(self) -> None:
        self.selector.refresh()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        super().apply_theme(tokens)
