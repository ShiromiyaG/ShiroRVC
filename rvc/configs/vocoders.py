import json
import os
from functools import lru_cache


_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "vocoders.json")


@lru_cache(maxsize=1)
def load_vocoder_registry():
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as registry_file:
        return json.load(registry_file)


def normalize_vocoder(vocoder):
    value = str(vocoder or "hifi").strip()
    registry = load_vocoder_registry()
    if value in registry:
        return value

    value_lower = value.lower()
    for vocoder_id, spec in registry.items():
        if value_lower in {
            str(spec.get("label", "")).lower(),
            str(spec.get("architecture", "")).lower(),
        }:
            return vocoder_id
    return "hifi"


def get_vocoder_spec(vocoder):
    return load_vocoder_registry()[normalize_vocoder(vocoder)]


def get_vocoder_choices():
    return [
        (spec["label"], vocoder_id)
        for vocoder_id, spec in load_vocoder_registry().items()
    ]


def get_vocoder_ids():
    return list(load_vocoder_registry())


def get_all_vocoder_sample_rates():
    return sorted(
        {
            int(rate)
            for spec in load_vocoder_registry().values()
            for rate in spec["sample_rates"]
        }
    )


def get_vocoder_cli_choices():
    choices = list(get_vocoder_ids())
    choices.extend(
        spec["label"]
        for spec in load_vocoder_registry().values()
        if spec["label"] not in choices
    )
    return choices


def get_vocoder_sample_rates(vocoder):
    return [int(rate) for rate in get_vocoder_spec(vocoder)["sample_rates"]]


def get_vocoder_config_paths():
    paths = {}
    for vocoder_id, spec in load_vocoder_registry().items():
        paths[vocoder_id] = [
            os.path.join(spec["config_dir"], f"{rate}.json")
            for rate in spec["sample_rates"]
        ]
    return paths
