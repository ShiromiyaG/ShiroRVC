import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.settings.sections.language import language_tab
from tabs.settings.sections.precision import precision_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.model_author import model_author_tab
from tabs.settings.sections.rmvpe_high_register import rmvpe_high_register_tab


def settings_tab():
    language_tab()
    model_author_tab()
    precision_tab()
    rmvpe_high_register_tab()
    restart_tab()
