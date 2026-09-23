import json

import gradio as gr

from rvc.configs.config import Config, bf16_is_supported, get_training_precision

from rvc.lib.i18n import _
from rvc.lib.paths import CONFIG_PATH
from rvc.lib.terminal import success

config = Config()

PRECISION_CHOICES = ["FP32", "FP16", "BF16"]


def set_training_precision(choice: str) -> str:
    """Persist the training precision (``FP32``, ``FP16`` or ``BF16``).

    It lives in ``assets/config.json`` rather than on the training tab because
    it is a property of the machine more than of the run.  The training tab
    reads it at launch and hands it to the run spec, so it is still recorded in
    ``run_spec.json``.
    """
    precision = str(choice).lower()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    config_json["precision"] = precision
    config_json.pop("use_fp16", None)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=4)

    success(f"Training precision set to {choice}.", tag="[SETTINGS]")
    messages = {
        "fp32": _("Runs started from now on train in FP32 with TF32 tensor cores."),
        "fp16": _("Runs started from now on use FP16 autocast with a GradScaler."),
        "bf16": _(
            "Runs started from now on use BF16 autocast without a GradScaler; "
            "precision-sensitive paths stay in FP32."
        ),
    }
    message = messages[precision]
    if precision == "bf16" and not bf16_is_supported():
        message += "\n" + _(
            "This GPU has no native BF16 support, so it would be emulated and slow."
        )
    return message


def precision_tab():
    with gr.Row():
        with gr.Column():

            precision = gr.Radio(
                label=_("Training precision"),
                info=_(
                    "Applies to runs started after this is saved. FP16 and BF16 need a CUDA GPU."
                ),
                choices=PRECISION_CHOICES,
                value=get_training_precision().upper(),
                interactive=True,
            )

            precision_output = gr.Textbox(
                label=_("Output Information"),
                info=_("The output information will be displayed here."),
                value="",
                max_lines=8,
                interactive=False,
            )

            precision.change(
                fn=set_training_precision,
                inputs=[precision],
                outputs=[precision_output],
            )

            check_button = gr.Button(_("Check precision"))
            check_button.click(
                fn=config.check_precision,
                inputs=[precision],
                outputs=[precision_output],
            )
