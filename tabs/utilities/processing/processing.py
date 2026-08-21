import os
import sys
import gradio as gr

from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_model_information_script

def processing_tab():
    model_view_model_path = gr.Textbox(
        label=_("Path to Model"),
        info=_("Introduce the model pth path"),
        value="",
        interactive=True,
        placeholder=_("Enter path to model"),
    )

    model_view_output_info = gr.Textbox(
        label=_("Output Information"),
        info=_("The output information will be displayed here."),
        value="",
        max_lines=11,
    )
    model_view_button = gr.Button(_("View"))
    model_view_button.click(
        fn=run_model_information_script,
        inputs=[model_view_model_path],
        outputs=[model_view_output_info],
    )
