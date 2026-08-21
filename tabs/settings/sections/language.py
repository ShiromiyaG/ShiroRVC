import json
import os
import sys

import gradio as gr

from rvc.lib import i18n
from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def get_language() -> str | None:
    """The stored choice, or ``None`` when the user has never made one.

    ``None`` is not the same as ``"en"``: it means "follow the system", so
    somebody who has never opened this section keeps tracking their operating
    system, while somebody who deliberately picked English stays on English.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return i18n.normalize(json.load(handle).get("language"))
    except (OSError, ValueError):
        return None


def set_language(language_name: str) -> str:
    tag = next(
        (code for code, name in i18n.LANGUAGES.items() if name == language_name),
        i18n.DEFAULT_LANGUAGE,
    )
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        config = {}

    config["language"] = tag
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    # Deliberately not translated into the *new* language: the interface around
    # this message is still in the old one, and mixing the two in a single
    # screen reads like a bug.
    return _(
        "Language set to {name}. Restart the interface to apply it."
    ).format(name=language_name)


def language_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown(_("### Language"))
            language_dropdown = gr.Dropdown(
                label=_("Interface language"),
                info=_(
                    "Applied on the next start. Takes effect immediately when "
                    "launched with --language, which overrides this setting."
                ),
                choices=list(i18n.LANGUAGES.values()),
                value=i18n.language_name(i18n.current_language()),
                interactive=True,
            )
            language_output_info = gr.Textbox(
                label=_("Output Information"),
                value="",
                max_lines=1,
                interactive=False,
            )
            language_dropdown.change(
                fn=set_language,
                inputs=[language_dropdown],
                outputs=[language_output_info],
            )
