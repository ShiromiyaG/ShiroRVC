import sys
import os
import logging
import asyncio
import threading
import traceback

# Gradio's analytics does a version-check HTTP request on launch; this app is
# local-only, so skip it. Set before importing gradio -- it is read at import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Notebook hosts (Colab) export MPLBACKEND=module://matplotlib_inline.backend_inline
# for their own kernel. We run in our own venv where that module usually is not
# installed, and Gradio imports matplotlib on every API call -- which then dies on
# the unresolvable backend. Fall back to the headless one when the pointed-at
# module is missing; a backend that does resolve is left alone.
_mpl_backend = os.environ.get("MPLBACKEND", "")
if _mpl_backend.startswith("module://"):
    import importlib.util

    try:
        _resolved = importlib.util.find_spec(_mpl_backend[len("module://") :])
    except (ImportError, ValueError):
        _resolved = None
    if _resolved is None:
        os.environ["MPLBACKEND"] = "Agg"

# Torch and Gradio are both ~2s to import and have no import relationship, so
# pulling torch in on a worker while the main thread imports Gradio overlaps
# whatever each spends outside the GIL (mostly loading native extensions).
# The tabs import torch transitively further down and will simply find it in
# sys.modules -- or block on the per-module import lock until this finishes.
def _warm_torch():
    try:
        import torch  # noqa: F401
    except Exception:
        # Nothing to do here: the real import below raises where it can be
        # reported properly. This thread only exists to prefetch.
        pass


_torch_warmup = threading.Thread(target=_warm_torch, name="torch-warmup", daemon=True)
_torch_warmup.start()

import gradio as gr

from rvc.lib.terminal import (
    info,
    install_rich_print,
    print_error_panel,
    success,
    warning,
)

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

# Identity comes from version.py so that this interface, the native one and the
# release workflow cannot disagree about which build they are.
from version import (  # noqa: E402 - after the environment setup above
    APP_DESCRIPTION,
    APP_NAME,
    APP_TITLE,
    __version__ as APP_VERSION,
)

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

# Language, before anything builds a widget.
#
# Gradio resolves ``label=`` and ``info=`` when the component is constructed,
# and the tab modules construct theirs at import time -- so the catalog has to
# be installed above these imports, not in ``__main__``.  Read straight off
# argv for the same reason: there is no argparse here, and adding one below
# would be too late.
#
# ``--language`` overrides everything; without it the stored preference wins,
# and failing that the operating system's display language.  See
# rvc/lib/i18n.py for the full precedence chain.
from rvc.lib import i18n  # noqa: E402

from rvc.lib.i18n import _  # noqa: E402
from tabs.settings.sections.language import get_language  # noqa: E402

APP_LANGUAGE = i18n.install_resolved(
    explicit=i18n.language_from_argv(),
    stored=get_language(),
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
)

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

APP_THEME = loadThemes.load_theme() or "ShiromiyaBlue"

def _runtime_badges():
    """Factual chips for the header.

    A probe that fails is dropped rather than reported as "unknown", so the
    header never shows anything that was not actually measured.
    """
    import html

    chips = []
    try:
        import torch

        chips.append(f"torch {torch.__version__.split('+')[0]}")
        chips.append(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
    except Exception:
        pass
    return "".join(f'<span class="rvc-chip">{html.escape(c)}</span>' for c in chips)


# Define Gradio interface
with gr.Blocks(title=APP_TITLE) as interface:
    with gr.Column(elem_classes=["rvc-header"]):
        with gr.Row(elem_classes=["rvc-header-grid"]):
            gr.Markdown(f"# {APP_TITLE}\n\n{APP_DESCRIPTION}")
            gr.HTML(
                f'<div class="rvc-chips">{_runtime_badges()}</div>',
                elem_classes=["rvc-header-side"],
            )

    with gr.Column(elem_classes=["rvc-workspace"]):
        with gr.Tabs(elem_id="rvc-main-tabs"):
            with gr.Tab(_("Inference")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    inference_tab()

            with gr.Tab(_("Training")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    train_tab()

            with gr.Tab(_("TTS")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    tts_tab()

            with gr.Tab(_("Voice Blender")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    voice_blender_tab()

            with gr.Tab(_("Download")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    download_tab()

            with gr.Tab(_("Utilities")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    utilities_tab()

            with gr.Tab(_("Settings")):
                with gr.Column(elem_classes=["rvc-card", "rvc-form-card"]):
                    settings_tab()


def launch_gradio(port):
    """Start the interface and report it the way every other stage reports.

    ``quiet`` silences Gradio's own ``* Running on local URL:`` banner, which is
    the one line in a session that does not follow the ``glyph [TAG] message``
    shape used by preprocessing, extraction and training.  ``prevent_thread_lock``
    hands the URLs back so they can be printed in that shape instead; blocking
    then happens explicitly below.
    """

    _app, local_url, share_url = interface.launch(
        favicon_path="assets/logo.png",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
        theme=APP_THEME,
        css_paths=GUI_CSS_PATH,
        footer_links=[],
        quiet=True,
        prevent_thread_lock=True,
    )
    success(f"Interface ready at {local_url}", tag="[APP]")
    if share_url:
        success(f"Public link: {share_url}", tag="[APP]")
    elif "--share" in sys.argv:
        warning("The public link could not be created.", tag="[APP]")
    info("Press Ctrl+C to stop.", tag="[APP]")
    interface.block_thread()


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    # Not ``for _ in`` -- ``_`` is the translation function at module scope.
    for _attempt in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(port)
            break
        except OSError:
            warning(
                f"Port {port} is taken; trying {port - 1}.",
                tag="[APP]",
            )
            port -= 1
        except Exception as error:
            print_error_panel(
                error,
                title="Could not launch the interface",
                details=traceback.format_exc(),
            )
            break
