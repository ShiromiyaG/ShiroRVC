import json
import os
from functools import lru_cache


_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "vocoders.json")


@lru_cache(maxsize=1)
def load_vocoder_registry():
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as registry_file:
        return json.load(registry_file)


def normalize_vocoder(vocoder):
    """Resolve a vocoder id, label or alias to a registry key.

    Unknown values raise rather than defaulting.  ``refinegan`` was renamed to
    ``refinegan2`` and kept as an alias; without one, an older run spec would
    have quietly built HiFi-GAN, which is the failure this whole registry
    exists to make impossible.
    """

    value = str(vocoder or "hifi").strip()
    registry = load_vocoder_registry()
    if value in registry:
        return value

    value_lower = value.lower()
    for vocoder_id, spec in registry.items():
        names = {
            str(spec.get("label", "")).lower(),
            str(spec.get("architecture", "")).lower(),
        }
        names.update(str(alias).lower() for alias in spec.get("aliases", ()))
        if value_lower in names:
            return vocoder_id
    raise ValueError(
        f"Unknown vocoder {vocoder!r}; known ids are {sorted(registry)}."
    )


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


def get_discriminator_id(vocoder):
    return get_vocoder_spec(vocoder).get("discriminator", "mpd_msd")


def get_architecture_id(vocoder, options=None):
    """The id a checkpoint carries so an incompatible one fails to load.

    ``net_g`` loads non-strictly, so without an id a checkpoint from one decoder
    would load into another's synthesizer with every decoder module silently
    left at its initialisation.

    ``vits_gaussian_v1`` is the *opt-out*: every guard that reads this value
    treats it as "nothing to check".  Both VITS-latent vocoders return it
    deliberately, because their state dict is Applio's -- identical keys and
    identical shapes, verified against Applio's own RefineGAN -- and the guard
    would otherwise reject an Applio checkpoint purely for not carrying an id.
    """
    spec = get_vocoder_spec(vocoder)
    base = spec.get("gaussian_architecture_id", spec["architecture_id"])
    # ``refinegan2_source`` deliberately does *not* appear here.  It swaps the
    # excitation generator, and the state dicts differ -- the sine owns
    # ``dec.m_source.merge.0.weight``, the bank owns ``dec.m_source.phase_offset``
    # -- so it has every reason to.  But this id is also what the whole RVC
    # ecosystem reads off a checkpoint, and ``vits_gaussian_v1`` is the value
    # that keeps an Applio checkpoint loadable, so a suffix here costs external
    # compatibility for an internal guard.  The guard lives in
    # ``excitation_source`` instead (``rvc/train/utils.py``), which is an
    # additive checkpoint key: a checkpoint written before it reports ``None``
    # and is read as ``"sine"``, which is what every one of them is.  It also
    # names the mismatch instead of encoding it in an opaque string.
    return base


def get_vocoder_config_paths():
    paths = {}
    for vocoder_id, spec in load_vocoder_registry().items():
        paths[vocoder_id] = [
            os.path.join(spec["config_dir"], f"{rate}.json")
            for rate in spec["sample_rates"]
        ]
    return paths
