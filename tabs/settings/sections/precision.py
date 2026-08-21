import gradio as gr

from rvc.configs.config import Config

from rvc.lib.i18n import _

config = Config()

def precision_tab():
    with gr.Row():
        with gr.Column():

            gr.Markdown(
                _("### Precision\n"
                "Training runs in **FP32** end to end - there is no autocast and "
                "no GradScaler.\n\n"
                "FP16 was removed because it needs a loss scaler and was the "
                "source of the AMP instability. BF16 was removed because its "
                "8-bit mantissa measured a gradient cosine of only **0.77-0.82** "
                "against a true-FP32 reference, while TF32 and FP16 (11-bit "
                "mantissa) sit at 0.9997 and 0.99. This model is unusually "
                "sensitive: the oscillatory NSF source and the L1-on-log-mel "
                "loss produce heavily cancelling sums.\n\n"
                "The precision knob that remains is **TF32**, which gives tensor "
                "cores at 11-bit mantissa with no autocast and no scaler. Toggle "
                "it per run with `use 'TF32' precision` in the Training tab.")
            )

            precision_output = gr.Textbox(
                label=_("Output Information"),
                info=_("The output information will be displayed here."),
                value="",
                max_lines=8,
                interactive=False,
            )

            check_button = gr.Button(_("Check precision"))
            check_button.click(
                fn=config.check_precision,
                outputs=[precision_output],
            )
