import gradio as gr

from rvc.lib.i18n import _

from tabs.utilities.processing import processing_tab
from tabs.utilities.analyzer import analyzer_tab
from tabs.utilities.f0_extractor import f0_extractor_tab
from tabs.utilities.model_bundle import model_bundle_tab
from tabs.utilities.model_processing import extract_small_model_tab
from tabs.utilities.experiment_config import experiment_config_tab

def utilities_tab():
    gr.Markdown(
        value=_(
            "This section contains some extra utilities. You might find some of 'em helpful."
        )
    )
    with gr.TabItem(_("Model information")):
        processing_tab()

    with gr.TabItem(_("F0 Curve")):
        f0_extractor_tab()

    with gr.TabItem(_("Audio Analyzer")):
        analyzer_tab()
    
    with gr.TabItem(_("Model Bundles")):
        model_bundle_tab()

    with gr.TabItem(_("Model Processing")):
        extract_small_model_tab()

    with gr.TabItem(_("Experiment Config")):
        experiment_config_tab()
