import gradio as gr

import os
import sys

from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.utilities.processing.processing import processing_tab
from tabs.utilities.analyzer.analyzer import analyzer_tab
from tabs.utilities.f0_extractor.f0_extractor import f0_extractor_tab
from tabs.utilities.model_bundle.model_bundle import model_bundle_tab
from tabs.utilities.model_processing.model_processing import extract_small_model_tab

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
