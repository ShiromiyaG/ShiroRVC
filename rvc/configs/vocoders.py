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


#: The two latent frontends a vocoder can be driven by.
#:
#: ``gaussian_flow`` is the stock VITS skeleton -- ``PosteriorEncoder`` over the
#: linear spectrogram, ``ResidualCouplingBlock``, and ``c_kl`` tying the training
#: distribution to the one inference draws from.  It is what Applio's RefineGAN
#: runs on: the vocoder replaces only ``net_g.dec`` and consumes ``z``.
def uses_chouwagan_stack(vocoder):
    """Whether the run gets the SAN/R1/branchwise discriminator and its losses.

    This asked about the ``latent`` registry field until 2026-08-31, back when
    a second latent frontend existed for that field to select.  It never
    decided anything about the latent, though -- every vocoder here runs the
    same posterior-and-flow skeleton.  What it actually turns on is the
    ChouwaGAN discriminator (SAN, the R1 strength controller, per-branch
    driving and the retain_graph backward it forces) plus the waveform
    reconstruction terms and band-weighted mel that go with it.

    So it reads the discriminator, which is the thing it selects.  Applio's
    RefineGAN is registered with plain MPD+MSD and leaves this False.
    """
    return get_discriminator_id(vocoder) == "chouwagan"


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
    The cost is that a ChouwaGAN and a RefineGAN checkpoint no longer refuse
    each other *by name*.  What catches it instead is the shape: the two
    decoders share ``dec.cond`` and ``dec.conv_post`` at different widths, and
    ``load_state_dict`` raises on a shape mismatch even when it is non-strict.
    That is a weaker guarantee than the id -- it holds only while the
    overlapping layers disagree -- so a decoder added later must not lean on it.
    The iSTFT hop below is exactly such a decoder, which is why it is spelled
    into the id rather than left to the shape check.
    """
    spec = get_vocoder_spec(vocoder)
    base = spec.get("gaussian_architecture_id", spec["architecture_id"])
    # ``chouwagan_istft_hop`` replaces the tail of the decoder: the time-domain
    # stages past the hop are gone and so is ``dec.conv_post``.  That matters
    # more than a usual option would, because the id is the opt-out above and
    # the shape of ``dec.conv_post`` is the only thing left separating two
    # decoders -- and this is precisely a decoder that no longer has one to
    # disagree about.  Two ChouwaGANs with different hops would otherwise share
    # an id, load non-strictly, and leave every replaced module at its
    # initialisation.  So the hop travels in the id.
    hop = int((options or {}).get("chouwagan_istft_hop", 0) or 0)
    if hop > 1:
        base = f"{base}_istft{hop}"
    return base


def get_vocoder_config_paths():
    paths = {}
    for vocoder_id, spec in load_vocoder_registry().items():
        paths[vocoder_id] = [
            os.path.join(spec["config_dir"], f"{rate}.json")
            for rate in spec["sample_rates"]
        ]
    return paths
