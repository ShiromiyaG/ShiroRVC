import gradio as gr
from core import run_model_information_script

from rvc.lib.i18n import _

def model_information_tab():
    with gr.Column():
        model_name = gr.Textbox(
            label=_("Path to Model"),
            info=_("Introduce the model pth path"),
            placeholder=_("Introduce the model pth path"),
            interactive=True,
        )
        model_information_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Model information."),
            value="",
            max_lines=12,
            interactive=False,
        )
        model_information_button = gr.Button(_("See Model Information"))
        model_information_button.click(
            fn=run_model_information_script,
            inputs=[model_name],
            outputs=[model_information_output_info],
        )
