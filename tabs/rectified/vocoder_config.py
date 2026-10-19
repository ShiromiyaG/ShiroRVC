"""Gradio tab editing the main settings an experiment's vocoder trains by:
its copy of the ChouwaGAN recipe for its sample rate."""

import os

import gradio as gr

from rvc.lib.i18n import _
from rvc.rectified.common import DEFAULT_SAMPLE_RATE, load_run_config, run_config_path
from rvc.rectified.vocoder import (
    read_vocoder_settings,
    reset_vocoder_settings,
    shipped_vocoder_settings,
    write_vocoder_settings,
)


#: What the checkboxes turn on, as the shipped recipes have them.
DEFAULT_FINAL_RATIO = 0.1
DEFAULT_EMA_DECAY = 0.999


def vocoder_config_tab(model_name, tab):
    """The Config step of the Vocoder tab, for the experiment in ``model_name``;
    its values are read again whenever ``tab`` is opened."""
    gr.Markdown(
        _("Settings the experiment's vocoder trains by. They are read when training starts; the other "
          "settings are in the experiment's rectified_vocoder_chouwagan.json and are kept as they are. "
          "The recipe follows the sample rate the dataset was preprocessed at, so preprocess first.")
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
    aux_epochs = gr.Number(
        label=_("Spectral-only Epochs"),
        info=_("Epochs trained on the spectral and harmonic losses alone before the discriminator joins, at most "
               "a quarter of the run; a run from a pretrained generator skips them. The progress line shows D as "
               "off until then. 0 starts the discriminator at once; empty keeps an older recipe's aux_steps."),
        minimum=0,
        interactive=True,
    )
    with gr.Row():
        save_button = gr.Button(_("Save"), variant="primary")
        reset_button = gr.Button(_("Reset to defaults"), variant="stop")
    output = gr.Textbox(label=_("Output Information"), info=_("Config status."), value="", interactive=False)

    fields = [learning_rate, lr_final_ratio, ema_decay, aux_epochs]

    def missing(name):
        return not name or not os.path.isfile(run_config_path(name))

    def load(name, message=""):
        # An experiment not preprocessed yet shows what the default rate starts from.
        if missing(name):
            settings = shipped_vocoder_settings(DEFAULT_SAMPLE_RATE)
        else:
            try:
                settings = read_vocoder_settings(name)
            except ValueError as problem:
                # A recipe left from another rate: what Reset would give.
                settings = shipped_vocoder_settings(load_run_config(name)["data"]["sample_rate"])
                message = str(problem)
        return [
            settings["learning_rate"],
            float(settings.get("lr_final_ratio") or 0) > 0,
            float(settings["ema_decay"]) > 0,
            settings.get("aux_epochs"),
            message,
        ]

    def save(name, lr, final_ratio, ema, aux):
        if missing(name):
            return _("Preprocess the dataset of {name} first: the recipe follows its sample rate.").format(
                name=name or "?"
            )
        try:
            settings = read_vocoder_settings(name)
        except ValueError as problem:
            return str(problem)
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
        if aux is not None:
            settings["aux_epochs"] = float(aux)
            settings.pop("aux_steps", None)
        write_vocoder_settings(name, settings)
        return _("Saved the vocoder settings of {name}.").format(name=name)

    def reset(name):
        if missing(name):
            return load(name)
        reset_vocoder_settings(name)
        return load(name, _("Back to the shipped settings for the experiment's sample rate."))

    outputs = [*fields, output]
    for trigger in (model_name.change, tab.select):
        trigger(fn=load, inputs=[model_name], outputs=outputs, show_progress="hidden")
    reset_button.click(fn=reset, inputs=[model_name], outputs=outputs, show_progress="hidden")
    save_button.click(fn=save, inputs=[model_name, *fields], outputs=[output])
