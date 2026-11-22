import json

import gradio as gr

from rvc.configs.config import (
    Config,
    get_precision_preference,
    resolve_training_precision,
)

from rvc.lib.i18n import _
from rvc.lib.paths import CONFIG_PATH
from rvc.lib.terminal import success

config = Config()

PRECISION_CHOICES = ["AUTO", "FP32", "FP16", "BF16"]


def set_training_precision(choice: str) -> str:
    """Persist the training precision (``AUTO``, ``FP32``, ``FP16`` or ``BF16``).

    It lives in ``assets/config.json`` rather than on the training tab because
    it is a property of the machine more than of the run.  The training tab
    reads it at launch and hands it to the run spec, so it is still recorded in
    ``run_spec.json``.  ``AUTO`` removes the key, so the machine is asked at
    every launch.
    """
    precision = str(choice).lower()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    if precision == "auto":
        config_json.pop("precision", None)
    else:
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
    resolved = resolve_training_precision(precision)
    if precision == "auto":
        return (
            _("Auto: runs use the best precision this machine supports, here {p}.").format(
                p=resolved.upper()
            )
            + "\n"
            + messages[resolved]
        )
    message = messages[precision]
    if resolved != precision:
        message += "\n" + _(
            "This machine does not support it natively, so runs fall back to {p}."
        ).format(p=resolved.upper())
    return message


def precision_tab():
    with gr.Row():
        with gr.Column():

            precision = gr.Radio(
                label=_("Training precision"),
                info=_(
                    "Applies to runs started after this is saved. AUTO picks BF16 on "
                    "Ampere or newer, FP16 on Volta/Turing, else FP32. A choice the "
                    "GPU cannot run falls back to the best supported one."
                ),
                choices=PRECISION_CHOICES,
                value=get_precision_preference().upper(),
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
