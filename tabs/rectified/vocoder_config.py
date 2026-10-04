"""Gradio tab editing the main vocoder settings an experiment trains by: the
``vocoder`` section of its rectified config for PCPH-BigVGAN, its recipe file
for the SingingVocoders architectures."""

import os

import gradio as gr

from rvc.lib.i18n import _
from rvc.lib.paths import LOGS_DIR
from rvc.rectified import singing_vocoders as sv


#: What the checkboxes turn on, as the shipped configs have them.
DEFAULT_FINAL_RATIO = 0.1
DEFAULT_EMA_DECAY = 0.999
HIFIGANS = ("pc-nsf-hifigan", "nsf-hifigan")
CLASSIC_HIFIGAN = "nsf-hifigan"


def vocoder_config_tab(model_name, architectures, tab):
    """The Config step of the Vocoder tab, for the experiment in ``model_name``;
    its values are read again whenever ``tab`` is opened."""
    gr.Markdown(
        _("Settings the experiment's vocoder trains by, per architecture. They are read when training "
          "starts; the other settings are in the experiment's JSON files and are kept as they are. "
          "Saving for a new model name creates its folder.")
    )
    architecture = gr.Radio(
        label=_("Vocoder Architecture"),
        info=_("Whose settings to edit."),
        choices=architectures,
        value=sv.PCPH,
        interactive=True,
    )
    d_choice = gr.Radio(
        label=_("Discriminator"),
        info=_("Original is the recipe's own (MSD + MPD for the NSF-HiFiGANs, MRD + MPD for NSF-UnivNet and "
               "Wavehax) with its losses. v3 is this repo's, with its adversarial and feature-matching losses; "
               "the spectral losses stay the recipe's. Changing it on a run that has checkpoints needs "
               "Fresh Training."),
        choices=[(_("Original"), "original"), ("v3", "v3")],
        value="original",
        interactive=True,
        # "hidden", not False: unmounted, it does not show on the first update.
        visible="hidden",
    )
    with gr.Row():
        learning_rate = gr.Number(
            label=_("Learning Rate"), info=_("Of both optimizers, from scratch."), minimum=0, interactive=True
        )
        lr_final_ratio = gr.Checkbox(
            label=_("Final LR Ratio"),
            info=_("Decay the learning rate to a tenth by the last epoch. Off keeps the scheduler's own decay."),
            interactive=True,
        )
        ema_decay = gr.Checkbox(
            label=_("EMA"),
            info=_("Previews and exports use an average of the generator's weights. Off uses them as trained."),
            interactive=True,
        )
    aux_steps = gr.Number(
        label=_("Spectral-only Steps"),
        info=_("Steps trained on the spectral loss alone before the discriminator joins; the progress line "
               "shows D as off until then. SingingVocoders' recipe uses 400000. 0 starts the discriminator "
               "at once."),
        minimum=0,
        precision=0,
        interactive=True,
        visible="hidden",
    )
    with gr.Row():
        # "hidden", not False: unmounted, they do not show on the first update.
        snakebeta = gr.Checkbox(
            label=_("SnakeBeta"),
            info=_("BigVGAN's periodic activation in the residual blocks, in place of the leaky ReLU. "
                   "A leaky-ReLU pretrain still loads; a run that has checkpoints needs Fresh Training."),
            interactive=True,
            visible="hidden",
        )
        oversampling = gr.Checkbox(
            label=_("Oversampling"),
            info=_("Runs each SnakeBeta at twice the rate behind low-pass filters, against aliasing. Slower."),
            interactive=True,
            visible="hidden",
        )
        mini_nsf = gr.Checkbox(
            label=_("Mini NSF"),
            info=_("The source is a single sine at the fundamental, made at the second stage's rate and "
                   "added there, as in PC-NSF-HiFiGAN, instead of sine plus harmonics strided down to every "
                   "stage. A run that has checkpoints needs Fresh Training."),
            interactive=True,
            visible="hidden",
        )
        filtered_upsampling = gr.Checkbox(
            label=_("Filtered Upsampling"),
            info=_("Fixed low-pass upsamplers in place of the transposed convolutions, against spectral "
                   "images. An OpenVPI pretrain no longer fits; a run that has checkpoints needs Fresh Training."),
            interactive=True,
            visible="hidden",
        )
    with gr.Row():
        save_button = gr.Button(_("Save"), variant="primary")
        reset_button = gr.Button(_("Reset to defaults"), variant="stop")
    output = gr.Textbox(label=_("Output Information"), info=_("Config status."), value="", interactive=False)

    fields = [d_choice, learning_rate, lr_final_ratio, ema_decay, snakebeta, oversampling, filtered_upsampling,
              mini_nsf, aux_steps]

    def hifigan(arch):
        """Visibility of the NSF-HiFiGANs' own options."""
        return True if arch in HIFIGANS else "hidden"

    def missing(name):
        return not name or not os.path.isdir(os.path.join(LOGS_DIR, name))

    def load(name, arch, message=""):
        # An experiment without a folder yet shows what it will start from.
        settings = sv.shipped_vocoder_settings(arch) if missing(name) else sv.read_vocoder_settings(name, arch)
        return [
            gr.update(
                value=settings.get("discriminator_choice", "original"),
                visible=True if arch != sv.PCPH else "hidden",
            ),
            settings["learning_rate"],
            float(settings.get("lr_final_ratio") or 0) > 0,
            float(settings["ema_decay"]) > 0,
            gr.update(value=settings["model"].get("activation") == "snakebeta", visible=hifigan(arch)),
            gr.update(value=bool(settings["model"].get("antialias", False)), visible=hifigan(arch)),
            gr.update(value=settings["model"].get("upsampling") == "filtered", visible=hifigan(arch)),
            gr.update(
                value=bool(settings["model"].get("mini_nsf", False)),
                visible=True if arch == CLASSIC_HIFIGAN else "hidden",
            ),
            gr.update(
                value=int(settings.get("aux_steps", 0)), visible=True if arch == sv.UNIVNET else "hidden"
            ),
            message,
        ]

    def save(name, arch, choice, lr, final_ratio, ema, snake, oversample, filtered, mini, aux):
        if not name:
            return _("Pick a model name first.")
        settings = sv.read_vocoder_settings(name, arch)
        if arch != sv.PCPH:
            settings["discriminator_choice"] = choice
        settings["learning_rate"] = float(lr)
        # Turned on, a value already in the file is kept.
        if not final_ratio:
            settings.pop("lr_final_ratio", None)
        elif not float(settings.get("lr_final_ratio") or 0) > 0:
            settings["lr_final_ratio"] = DEFAULT_FINAL_RATIO
        if not ema:
            settings["ema_decay"] = 0.0
        elif not float(settings["ema_decay"]) > 0:
            settings["ema_decay"] = DEFAULT_EMA_DECAY
        if arch in HIFIGANS:
            settings["model"]["activation"] = "snakebeta" if snake else "leaky_relu"
            settings["model"]["antialias"] = bool(snake and oversample)
            settings["model"]["upsampling"] = "filtered" if filtered else "transposed"
        if arch == CLASSIC_HIFIGAN:
            settings["model"]["mini_nsf"] = bool(mini)
        if arch == sv.UNIVNET:
            settings["aux_steps"] = int(aux or 0)
        sv.write_vocoder_settings(name, arch, settings)
        return _("Saved the {architecture} settings of {name}.").format(architecture=arch, name=name)

    def reset(name, arch):
        if missing(name):
            return load(name, arch)
        sv.reset_vocoder_settings(name, arch)
        return load(name, arch, _("Back to the shipped {architecture} settings.").format(architecture=arch))

    outputs = [*fields, output]
    for trigger in (architecture.change, model_name.change, tab.select):
        trigger(fn=load, inputs=[model_name, architecture], outputs=outputs, show_progress="hidden")
    reset_button.click(fn=reset, inputs=[model_name, architecture], outputs=outputs, show_progress="hidden")
    save_button.click(fn=save, inputs=[model_name, architecture, *fields], outputs=[output])
