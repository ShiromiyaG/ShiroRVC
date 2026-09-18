"""Inference presets: the settings file both interfaces save and load.

One definition for the Gradio tab and the Qt window, so a preset saved in one
loads in the other and a setting added to one cannot quietly go missing from
the other's files.  Standard library only: the Qt window reads this from its
UI thread, and ``tests/test_gui_isolation.py`` holds it to not importing torch.

A preset is a flat JSON object in ``assets/inference_presets/<name>.json``.
Every key is optional on read -- a file saved before a setting existed loads
what it has and leaves the rest of the form alone -- and a value of the wrong
kind is dropped rather than handed to a control that would refuse it at
convert time.  Left out on purpose: the model, index and speaker, so a preset
works across voices; the input and output paths, which belong to the job; and
``filter_radius``, which ``Pipeline.get_f0`` ignores.
"""

from __future__ import annotations

import json
import math
import os

F0_METHODS = ("crepe", "crepe-tiny", "rmvpe", "fcpe")
EMBEDDER_MODELS = ("contentvec", "spin_v1", "spin_v2")
EXPORT_FORMATS = ("WAV", "MP3", "FLAC", "OGG", "M4A")

#: Keys whose value is one of a fixed set.
CHOICE_KEYS = {
    "f0_method": F0_METHODS,
    "embedder_model": EMBEDDER_MODELS,
    "export_format": EXPORT_FORMATS,
}
BOOL_KEYS = frozenset({"split_audio", "autotune", "clean_audio", "formant_shifting"})
NUMBER_KEYS = frozenset({
    "seed",
    "autotune_strength",
    "clean_strength",
    "formant_qfrency",
    "formant_timbre",
    "pitch",
    "index_rate",
    "index_k",
    "index_power",
    "index_continuity",
    "rms_mix_rate",
    "protect",
    "silence_gate_db",
})

#: Every setting a preset carries.  The first five names are the ones presets
#: held before the rest were added, which is why the envelope blend is still
#: ``rms_mix_rate`` here whatever a form calls it.
PRESET_KEYS = frozenset(CHOICE_KEYS) | BOOL_KEYS | NUMBER_KEYS

#: Characters a preset name cannot hold, because it becomes a file name.
_FORBIDDEN = set('\\/:*?"<>|')


def _valid(key: str, value) -> bool:
    if key in CHOICE_KEYS:
        return value in CHOICE_KEYS[key]
    if key in BOOL_KEYS:
        return isinstance(value, bool)
    # ``bool`` is an ``int`` to Python, and NaN is legal JSON to ``json``.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_valid_name(name: str) -> bool:
    """Whether ``name`` can be a preset: non-empty, and a plain file name."""
    name = name.strip()
    return bool(name) and not (set(name) & _FORBIDDEN) and name not in (".", "..")


def list_presets(directory: str | os.PathLike[str]) -> list[str]:
    """Preset names in ``directory``, sorted, without the ``.json``."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        (name[: -len(".json")] for name in names if name.lower().endswith(".json")),
        key=str.lower,
    )


def parse_preset(path: str | os.PathLike[str]) -> dict:
    """The valid settings in the preset file at ``path``.

    Raises ``OSError`` or ``ValueError`` for a file that cannot be read or is
    not a JSON object.  Unknown keys and invalid values are dropped silently:
    they are what a newer or hand-edited file looks like, not a broken one.
    """
    with open(path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if not isinstance(stored, dict):
        raise ValueError("a preset must be a JSON object")
    return {
        key: value
        for key, value in stored.items()
        if key in PRESET_KEYS and _valid(key, value)
    }


def read_preset(directory: str | os.PathLike[str], name: str) -> dict:
    """``parse_preset`` on the preset called ``name`` in ``directory``."""
    return parse_preset(os.path.join(directory, f"{name}.json"))


def write_preset(directory: str | os.PathLike[str], name: str, values: dict) -> str:
    """Save ``values`` as the preset ``name``; returns the file written.

    Only preset keys are written, so a form can hand over everything it has.
    """
    name = name.strip()
    if not is_valid_name(name):
        raise ValueError(f"{name!r} is not a usable preset name")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in values.items() if key in PRESET_KEYS},
            handle,
            ensure_ascii=False,
            indent=4,
        )
    return path
