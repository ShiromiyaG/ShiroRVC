import os
import sys
import json

import gradio as gr

from rvc.configs.config import Config, get_use_fp16

from rvc.lib.i18n import _
from rvc.lib.terminal import success

now_dir = os.getcwd()
sys.path.append(now_dir)

config = Config()

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def set_use_fp16(enabled: bool) -> str:
    """Persist the FP16 preference.

    It lives in ``assets/config.json`` rather than on the training tab because
    it is a property of the machine more than of the run: a card either has the
    tensor cores to make FP16 pay off or it does not.  The training tab reads it
    at launch and hands it to the run spec, so it still travels with the run and
    is recorded in ``run_spec.json``.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    config_json["use_fp16"] = bool(enabled)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=4)

    state = "enabled" if enabled else "disabled"
    success(f"FP16 training {state}.", tag="[SETTINGS]")
    return (
        _("FP16 training enabled. Runs started from now on use autocast with a GradScaler.")
        if enabled
        else _("FP16 training disabled. Runs start from now on in FP32 with TF32 tensor cores.")
    )


def precision_tab():
    with gr.Row():
        with gr.Column():

            gr.Markdown(
                _("### Precision\n"
                  "The default is **FP32** master weights with **TF32** tensor "
                  "cores - no autocast, no loss scaler. TF32 is toggled per run "
                  "in the Training tab.\n\n"
                  "**FP16** adds `torch.autocast` plus a `GradScaler` on top. It "
                  "keeps the same 11-bit mantissa as TF32 (gradient cosine 0.99 "
                  "against a true-FP32 reference, where BF16's 8-bit mantissa "
                  "measured only 0.77-0.82), but its narrow exponent range can "
                  "overflow, which is what the scaler is for. Distribution math "
                  "and the NSF source stay in FP32; the convolutional backbone - "
                  "the dominant cost - runs in FP16.\n\n"
                  "Turn it on for the memory and speed, turn it off if a run "
                  "shows the scaler backing off repeatedly or losses going "
                  "non-finite. BF16 is not offered: it loses too much mantissa "
                  "for this model, whose oscillatory NSF source and L1-on-log-mel "
                  "loss produce heavily cancelling sums.")
            )

            use_fp16 = gr.Checkbox(
                label=_("FP16 training (autocast + GradScaler)"),
                info=_("Applies to runs started after this is saved. Needs a CUDA GPU."),
                value=get_use_fp16(),
                interactive=True,
            )

            precision_output = gr.Textbox(
                label=_("Output Information"),
                info=_("The output information will be displayed here."),
                value="",
                max_lines=8,
                interactive=False,
            )

            use_fp16.change(
                fn=set_use_fp16,
                inputs=[use_fp16],
                outputs=[precision_output],
            )

            check_button = gr.Button(_("Check precision"))
            check_button.click(
                fn=config.check_precision,
                inputs=[use_fp16],
                outputs=[precision_output],
            )
