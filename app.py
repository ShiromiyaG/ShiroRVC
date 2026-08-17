import gradio as gr
import sys
import os
import logging
import asyncio

from rvc.lib.terminal import install_rich_print

install_rich_print()

# Suppress noisy Windows ProactorEventLoop connection-reset errors.
# These fire whenever a browser tab closes/resets an SSE or long-poll
# connection mid-flight (which Gradio does constantly). Completely harmless.
#
# set_exception_handler only catches coroutine/future exceptions — this error
# comes from a plain *callback*, so we have to patch the offending method directly.
try:
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
    def _patched_call_connection_lost(self, exc):
        try:
            _original_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass
    _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
except Exception:
    pass  # Non-Windows or future Python version where this doesn't exist ( hopefully )

# Constants
DEFAULT_PORT = 7897
MAX_PORT_ATTEMPTS = 10
APP_NAME = "ShiroRVC"
APP_VERSION = "1.0.0"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"
APP_DESCRIPTION = "RVC voice conversion and training by Shiromiya."

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

GUI_CSS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "themes",
    "gui.css",
)

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.utilities.utilities import utilities_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.settings.settings import settings_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
    smartcutter=True,
)

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

APP_THEME = loadThemes.load_theme() or "ShiromiyaBlue"

# Define Gradio interface
with gr.Blocks(title=APP_TITLE) as interface:
    with gr.Column(elem_classes=["rvc-header"]):
        with gr.Row(elem_classes=["rvc-header-grid"]):
            gr.Markdown(f"# {APP_TITLE}\n\n{APP_DESCRIPTION}")

    with gr.Column(elem_classes=["rvc-workspace"]):
        with gr.Tabs(elem_id="rvc-main-tabs"):
            with gr.Tab("Inference"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    inference_tab()

            with gr.Tab("Training"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    train_tab()

            with gr.Tab("TTS"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    tts_tab()

            with gr.Tab("Voice Blender"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    voice_blender_tab()

            with gr.Tab("Download"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    download_tab()

            with gr.Tab("Utilities"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    utilities_tab()

            with gr.Tab("Settings"):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    settings_tab()


def launch_gradio(port):
    interface.launch(
        favicon_path="assets/logo.png",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
        theme=APP_THEME,
        css_paths=GUI_CSS_PATH,
        footer_links=[],
    )


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
