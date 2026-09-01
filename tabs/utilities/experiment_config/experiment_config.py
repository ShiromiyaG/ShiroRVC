"""Rewrite an already-extracted experiment's ``config.json`` for another vocoder.

``generate_config`` writes the shipped config once, during feature extraction,
and from then on it only *replaces* one whose ``architecture_id`` disagrees with
the build.  There is no path for the deliberate case: the dataset is extracted,
the folder is intact, and the architecture is the thing being changed.  Doing it
by hand means knowing which of the three shipped configs to copy, at which
sample rate, and that ``model_info.json`` carries the architecture a second time.

Nothing here touches the extracted features, the filelist, or the audio, because
none of those depend on the vocoder -- only the sample rate they were written at
does, and that is read from the experiment rather than offered as a choice.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import gradio as gr

from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_choices,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.i18n import _

now_dir = os.getcwd()

#: Same exclusions the training tab applies: these are shared assets, not runs.
EXCLUDED_FOLDERS = ("zips", "mute", "reference")


def _logs_dir() -> Path:
    return Path(now_dir) / "logs"


def _shipped_config(vocoder_id: str, sample_rate: int) -> Path:
    return (
        Path(now_dir)
        / "rvc"
        / "configs"
        / get_vocoder_spec(vocoder_id)["config_dir"]
        / f"{int(sample_rate)}.json"
    )


def get_experiments_list() -> list[str]:
    """Experiment folders that have been through feature extraction.

    Keyed on ``config.json`` rather than on the folder existing: before
    extraction there is nothing to regenerate, and offering such a folder would
    produce a config for a dataset that has no filelist to go with it.
    """
    logs_dir = _logs_dir()
    if not logs_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in logs_dir.iterdir()
        if entry.is_dir()
        and (entry / "config.json").is_file()
        and all(excluded not in entry.name for excluded in EXCLUDED_FOLDERS)
    )


def refresh_experiments():
    return {"choices": get_experiments_list(), "__type__": "update"}


def _read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {} if default is None else default


def _vocoder_from_architecture(architecture_id: str | None) -> str | None:
    """The registry entry that stamps this ``architecture_id``, if any.

    The id is the only thing in the config that names the architecture, and it
    is deliberately not the vocoder id -- it identifies the latent *and* the
    decoder, so the mapping is looked up rather than guessed from the string.
    """
    if not architecture_id:
        return None
    for _label, vocoder_id in get_vocoder_choices():
        if get_architecture_id(vocoder_id) == architecture_id:
            return vocoder_id
    return None


def _checkpoints(experiment_dir: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("G_*.pth", "D_*.pth")
        for path in experiment_dir.glob(pattern)
    )


def describe_experiment(experiment: str | None):
    """Status text plus a target-vocoder preselection for the chosen folder."""
    if not experiment:
        return (
            _("Select an experiment to see what its config was written for."),
            gr.update(),
        )

    experiment_dir = _logs_dir() / experiment
    config = _read_json(experiment_dir / "config.json")
    if not config:
        return (
            _("`{}` has no readable config.json.").format(experiment),
            gr.update(),
        )

    architecture_id = config.get("model", {}).get("architecture_id")
    sample_rate = config.get("data", {}).get("sample_rate")
    current_vocoder = _vocoder_from_architecture(architecture_id)
    model_info = _read_json(experiment_dir / "model_info.json")
    recorded = model_info.get("vocoder_architecture")

    lines = [
        _("**Sample rate:** {} Hz -- fixed by the extracted audio.").format(sample_rate),
        _("**Architecture id in config.json:** `{}`").format(architecture_id or _("absent")),
        _("**Vocoder that stamps it:** {}").format(
            get_vocoder_spec(current_vocoder)["label"]
            if current_vocoder
            else _("unrecognised -- this config predates the current registry")
        ),
        _("**model_info.json says:** `{}`").format(recorded or _("absent")),
    ]

    if recorded and current_vocoder and normalize_vocoder(recorded) != current_vocoder:
        lines.append(
            _(
                "The two disagree. Rebuilding writes both, which is how that "
                "gets resolved."
            )
        )

    existing_checkpoints = _checkpoints(experiment_dir)
    if existing_checkpoints:
        lines.append(
            _(
                "**{} checkpoint(s) in this folder.** They belong to the old "
                "architecture and cannot be resumed under a different one."
            ).format(len(existing_checkpoints))
        )

    return "\n\n".join(lines), gr.update(
        value=current_vocoder if current_vocoder else gr.update()
    )


def rebuild_config(
    experiment: str | None,
    vocoder: str | None,
    keep_backup: bool,
    move_checkpoints: bool,
):
    if not experiment:
        return _("Select an experiment first.")

    experiment_dir = _logs_dir() / experiment
    config_path = experiment_dir / "config.json"
    if not config_path.is_file():
        return _("`{}` has no config.json to rebuild.").format(experiment)

    current = _read_json(config_path)
    sample_rate = current.get("data", {}).get("sample_rate")
    if not sample_rate:
        return _(
            "`{}` has a config.json with no data.sample_rate, so there is "
            "nothing to match a new one against."
        ).format(experiment)

    vocoder_id = normalize_vocoder(vocoder)
    supported = get_vocoder_sample_rates(vocoder_id)
    if int(sample_rate) not in supported:
        # The sample rate is not offered as a choice on purpose: the sliced
        # audio, the f0 curves and the extracted features were all written at
        # one rate, and a config that disagreed with them would fail thousands
        # of steps in rather than here.
        return _(
            "{} has no configuration for {} Hz (it supports {}). The extracted "
            "audio fixes the rate, so this experiment cannot use that vocoder."
        ).format(
            get_vocoder_spec(vocoder_id)["label"],
            sample_rate,
            ", ".join(f"{rate} Hz" for rate in supported),
        )

    shipped = _shipped_config(vocoder_id, sample_rate)
    if not shipped.is_file():
        return _("The shipped config is missing: {}").format(shipped)

    report = []
    if keep_backup:
        previous = current.get("model", {}).get("architecture_id") or "unknown"
        backup = config_path.with_suffix(f".json.{previous}.bak")
        if backup.exists():
            backup = config_path.with_suffix(
                f".json.{previous}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            )
        shutil.copyfile(config_path, backup)
        report.append(_("Kept the previous config at `{}`.").format(backup.name))

    shutil.copyfile(shipped, config_path)
    report.append(
        _("Wrote `{}` from `{}`.").format(
            config_path.relative_to(Path(now_dir)),
            shipped.relative_to(Path(now_dir)),
        )
    )

    # ``model_info.json`` carries the architecture a second time, and
    # ``extract_model`` stamps the exported checkpoint from *that* copy.  A
    # rebuild that updated only config.json would export models labelled with
    # the architecture they are not.
    model_info_path = experiment_dir / "model_info.json"
    model_info = _read_json(model_info_path)
    model_info["vocoder_architecture"] = vocoder_id
    with open(model_info_path, "w", encoding="utf-8") as handle:
        json.dump(model_info, handle, indent=4)
    report.append(_("Set `vocoder_architecture` to `{}` in model_info.json.").format(vocoder_id))

    existing_checkpoints = _checkpoints(experiment_dir)
    if existing_checkpoints and move_checkpoints:
        destination = experiment_dir / f"old_{current.get('model', {}).get('architecture_id', 'checkpoints')}"
        destination.mkdir(exist_ok=True)
        for checkpoint in existing_checkpoints:
            shutil.move(str(checkpoint), str(destination / checkpoint.name))
        report.append(
            _("Moved {} checkpoint(s) into `{}/`.").format(
                len(existing_checkpoints), destination.name
            )
        )
    elif existing_checkpoints:
        report.append(
            _(
                "{} checkpoint(s) were left in place. Training will try to "
                "resume from them and they are not loadable under the new "
                "architecture -- move or delete them before starting a run."
            ).format(len(existing_checkpoints))
        )

    report.append(
        _(
            "The dataset, filelist and extracted features are untouched: none "
            "of them depend on the vocoder."
        )
    )
    return "\n\n".join(f"- {line}" for line in report)


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
            value="refinegan",
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
