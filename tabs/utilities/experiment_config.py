"""Gradio tab for rewriting an extracted experiment's ``config.json``.

The logic lives in ``rvc.configs.experiments``, shared with the Qt interface.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from rvc.configs import experiments
from rvc.configs.vocoders import get_vocoder_choices
from rvc.lib.i18n import _

now_dir = os.getcwd()


def _logs_dir() -> Path:
    return Path(now_dir) / "logs"


def get_experiments_list() -> list[str]:
    return experiments.list_experiments(_logs_dir())


def refresh_experiments():
    return {"choices": get_experiments_list(), "__type__": "update"}


def describe_experiment(experiment: str | None):
    """Status text plus a target-vocoder preselection for the chosen folder."""
    if not experiment:
        return (
            _("Select an experiment to see what its config was written for."),
            gr.update(),
        )
    text, current_vocoder = experiments.describe(_logs_dir() / experiment)
    # An unrecognised architecture has no vocoder to preselect, and the way to
    # say "leave this dropdown alone" is a bare ``gr.update()`` for the
    # component -- not one nested in ``value=``, which hands the dropdown the
    # update dict itself as its value and warns that it is not among the
    # choices.
    return text, (gr.update(value=current_vocoder) if current_vocoder else gr.update())


def rebuild_config(
    experiment: str | None,
    vocoder: str | None,
    keep_backup: bool,
    move_checkpoints: bool,
):
    if not experiment:
        return _("Select an experiment first.")
    try:
        return experiments.rebuild(
            _logs_dir() / experiment, vocoder, keep_backup, move_checkpoints
        )
    except experiments.RebuildRefused as refusal:
        return str(refusal)


def experiment_config_tab():
    gr.Markdown(
        value=_(
            "Rewrite an extracted experiment's `config.json` for a different "
            "vocoder, without re-running preprocessing or feature extraction. "
            "The new config is the shipped one for that vocoder at this "
            "experiment's sample rate, so anything hand-tuned in the old file "
            "is replaced rather than merged."
        )
    )

    with gr.Row():
        experiment = gr.Dropdown(
            label=_("Experiment"),
            info=_("A folder under logs/ that already has a config.json."),
            choices=get_experiments_list(),
            interactive=True,
            allow_custom_value=False,
        )
        target_vocoder = gr.Dropdown(
            label=_("Vocoder / Architecture"),
            info=_("The architecture the rebuilt config will be written for."),
            choices=get_vocoder_choices(),
            value=experiments.default_target_vocoder(),
            interactive=True,
        )
    refresh_button = gr.Button(_("Refresh"))

    details = gr.Markdown(
        value=_("Select an experiment to see what its config was written for.")
    )

    with gr.Row():
        keep_backup = gr.Checkbox(
            label=_("Keep a backup of the current config"),
            info=_("Saved beside it as config.json.<old architecture>.bak."),
            value=True,
            interactive=True,
        )
        move_checkpoints = gr.Checkbox(
            label=_("Move existing checkpoints aside"),
            info=_(
                "G_*.pth / D_*.pth from the old architecture are moved into a "
                "subfolder instead of being deleted."
            ),
            value=False,
            interactive=True,
        )

    rebuild_button = gr.Button(_("Rebuild config"), variant="primary")
    output = gr.Markdown()

    refresh_button.click(fn=refresh_experiments, inputs=[], outputs=[experiment])
    experiment.change(
        fn=describe_experiment,
        inputs=[experiment],
        outputs=[details, target_vocoder],
    )
    rebuild_button.click(
        fn=rebuild_config,
        inputs=[experiment, target_vocoder, keep_backup, move_checkpoints],
        outputs=[output],
    )
