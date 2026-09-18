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

No UI imports: the Gradio tab and the Qt interface both call into this.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from rvc.configs.vocoders import (
    get_architecture_id,
    get_default_vocoder,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    is_vocoder_enabled,
    load_vocoder_registry,
    normalize_vocoder,
)
from rvc.lib.i18n import _

#: The application root, for the shipped configs and for paths in reports.
ROOT = Path(__file__).resolve().parents[2]

#: Same exclusions the training tab applies: these are shared assets, not runs.
EXCLUDED_FOLDERS = ("zips", "mute", "reference")


class RebuildRefused(Exception):
    """The experiment cannot take the requested config; nothing was written."""


def _shipped_config(vocoder_id: str, sample_rate: int) -> Path:
    return (
        ROOT
        / "rvc"
        / "configs"
        / get_vocoder_spec(vocoder_id)["config_dir"]
        / f"{int(sample_rate)}.json"
    )


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def default_target_vocoder() -> str:
    """What the target dropdown starts on."""
    return "refinegan2" if is_vocoder_enabled("refinegan2") else get_default_vocoder()


def list_experiments(logs_dir: Path) -> list[str]:
    """Experiment folders that have been through feature extraction.

    Keyed on ``config.json`` rather than on the folder existing: before
    extraction there is nothing to regenerate, and offering such a folder would
    produce a config for a dataset that has no filelist to go with it.
    """
    if not logs_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in logs_dir.iterdir()
        if entry.is_dir()
        and (entry / "config.json").is_file()
        and all(excluded not in entry.name for excluded in EXCLUDED_FOLDERS)
    )


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
    # Every vocoder, disabled ones included: an existing config still has to be
    # identified.
    for vocoder_id in load_vocoder_registry():
        if get_architecture_id(vocoder_id) == architecture_id:
            return vocoder_id
    return None


def _checkpoints(experiment_dir: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("G_*.pth", "D_*.pth")
        for path in experiment_dir.glob(pattern)
    )


def describe(experiment_dir: Path) -> tuple[str, str | None]:
    """Markdown status for an experiment, and the vocoder its config is for.

    The vocoder is ``None`` when the config's architecture is not one this
    build recognises.
    """
    experiment = experiment_dir.name
    config = _read_json(experiment_dir / "config.json")
    if not config:
        return _("`{}` has no readable config.json.").format(experiment), None

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

    if recorded:
        try:
            recorded_id = normalize_vocoder(recorded)
        except ValueError:
            # ``model_info.json`` can name a vocoder this build no longer ships.
            # This panel exists to *report* that kind of drift, so it must not
            # raise on it -- say so and let Rebuild write a value that resolves.
            recorded_id = None
            lines.append(
                _(
                    "`{}` is not a vocoder this build knows -- it was likely "
                    "removed from the registry. Rebuilding writes a current one."
                ).format(recorded)
            )
        if recorded_id and current_vocoder and recorded_id != current_vocoder:
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

    return "\n\n".join(lines), current_vocoder


def rebuild(
    experiment_dir: Path,
    vocoder: str | None,
    keep_backup: bool,
    move_checkpoints: bool,
) -> str:
    """Write the shipped config for ``vocoder``; returns a Markdown report.

    Raises :class:`RebuildRefused` before writing anything when the experiment
    cannot take it.
    """
    experiment = experiment_dir.name
    config_path = experiment_dir / "config.json"
    if not config_path.is_file():
        raise RebuildRefused(_("`{}` has no config.json to rebuild.").format(experiment))

    current = _read_json(config_path)
    sample_rate = current.get("data", {}).get("sample_rate")
    if not sample_rate:
        raise RebuildRefused(_(
            "`{}` has a config.json with no data.sample_rate, so there is "
            "nothing to match a new one against."
        ).format(experiment))

    vocoder_id = normalize_vocoder(vocoder)
    supported = get_vocoder_sample_rates(vocoder_id)
    if int(sample_rate) not in supported:
        # The sample rate is not offered as a choice on purpose: the sliced
        # audio, the f0 curves and the extracted features were all written at
        # one rate, and a config that disagreed with them would fail thousands
        # of steps in rather than here.
        raise RebuildRefused(_(
            "{} has no configuration for {} Hz (it supports {}). The extracted "
            "audio fixes the rate, so this experiment cannot use that vocoder."
        ).format(
            get_vocoder_spec(vocoder_id)["label"],
            sample_rate,
            ", ".join(f"{rate} Hz" for rate in supported),
        ))

    shipped = _shipped_config(vocoder_id, sample_rate)
    if not shipped.is_file():
        raise RebuildRefused(_("The shipped config is missing: {}").format(shipped))

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
        _("Wrote `{}` from `{}`.").format(_display(config_path), _display(shipped))
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
