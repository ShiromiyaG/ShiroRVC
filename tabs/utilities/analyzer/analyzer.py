import os, sys
import gradio as gr

from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_audio_analyzer_script


def analyzer_tab():
    with gr.Column():
        audio_input = gr.Audio(type="filepath")
        output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        get_info_button = gr.Button(value="Get information about the audio")
        image_output = gr.Image(type="filepath", interactive=False)

    get_info_button.click(
        fn=run_audio_analyzer_script,
        inputs=[audio_input],
        outputs=[output_info, image_output],
    )
