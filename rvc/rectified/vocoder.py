from types import SimpleNamespace

import torch
from torch import nn

from rvc.lib.algorithm.generators.nsf_bigvgan import NSFBigVGANGenerator

#: Mel settings a vocoder must share with the flow to render its output.
MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax")


#: Upsampler filters for the 44.1 kHz layout's five stages, ``[4, 4, 4, 4, 2]``.
#: RefineGAN2's four-stage widths extended on their own reasoning: the first
#: stage short, where a long kernel reaches furthest into the padded edge, the
#: later ones long; the last repeats the one before it. Rolloffs as
#: ``nsf_bigvgan.UPSAMPLE_ROLLOFF``, with no image above Nyquist. The config's
#: ``filter_width``, ``rolloff`` and ``filter_beta`` override them, and another
#: stage count keeps the generator's own (the older ``[5, 4, 4, 4]`` exports).
FILTER_WIDTH = (12, 24, 32, 48, 48)
ROLLOFF = (0.84, 0.92, 0.94, 0.94, 0.94)
FILTER_BETA = (6.0, 6.0, 6.0, 9.0, 9.0)


def build_vocoder(config: dict) -> NSFBigVGANGenerator:
    """NSF-BigVGAN from f0 and the *normalised* log mel (``normalize_mel``),
    with no speaker input. The raw log mel sits around -5 with a spread of
    ~3, several times the scale the trunk's init expects."""
    model = config["vocoder"]["model"]
    data = config["data"]
    stages = len(model["upsample_rates"])
    filters = {
        key: model[key] if key in model else default
        for key, default in (("filter_width", FILTER_WIDTH), ("rolloff", ROLLOFF),
                             ("filter_beta", FILTER_BETA))
        if key in model or len(default) == stages
    }
    return NSFBigVGANGenerator(
        sample_rate=data["sample_rate"],
        upsample_rates=tuple(model["upsample_rates"]),
        upsample_initial_channel=model["upsample_initial_channel"],
        resblock_kernel_sizes=model["resblock_kernel_sizes"],
        resblock_dilation_sizes=model["resblock_dilation_sizes"],
        resblock=model["resblock"],
        antialias=model["antialias"],
        num_mels=data["n_mels"],
        gin_channels=0,
        source_noise_std=model["source_noise_std"],
        source_harmonics=model["source_harmonics"],
        source_branch=model["source_branch"],
        source_type=model.get("source_type", "sine"),
        source_random_start_phase=model.get("source_random_start_phase", True),
        output_gain=model["output_gain"],
        stage_channels=model.get("stage_channels"),
        prenet_blocks=model.get("prenet_blocks", 0),
        deep_source_stages=model.get("deep_source_stages", 0),
        **filters,
    )


def build_discriminator(config: dict, use_checkpointing: bool = False):
    """The NSF-BigVGAN recipe's discriminator (v4 + UnivHD + SAN).

    ``d_mrd_mel_cond`` conditions its spectrogram branches on the normalised
    mel the vocoder is given; the discriminator then takes it as ``cond``."""
    from rvc.train.setup import get_d_model

    settings = dict(config["vocoder"]["discriminator"])
    data = config["data"]
    if settings.pop("d_mrd_mel_cond", False):
        settings["mrd_mel_cond"] = dict(
            sample_rate=data["sample_rate"],
            n_mels=data["n_mels"],
            fmin=data["mel_fmin"],
            fmax=data["mel_fmax"],
        )
    hparams = SimpleNamespace(
        model=SimpleNamespace(**settings),
        data=SimpleNamespace(sample_rate=data["sample_rate"]),
    )
    return get_d_model(hparams, "nsf-bigvgan", use_checkpointing)


class RawMelVocoder(nn.Module):
    """A vocoder trained on the raw log mel, fed the normalised one."""

    def __init__(self, generator: nn.Module, data: dict):
        super().__init__()
        self.generator = generator
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])

    def forward(self, mel, f0):
        return self.generator(mel * self.mel_std + self.mel_mean, f0)


def mel_mismatch(data: dict, vocoder_data: dict):
    """The first mel setting the vocoder does not share with ``data``, or None.

    ``mel_mean``/``mel_std`` count only for vocoders that carry them."""
    keys = MEL_KEYS + tuple(k for k in ("mel_mean", "mel_std") if k in vocoder_data)
    for key in keys:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            return f"{key} ({vocoder_data[key]} vs {data[key]})"
    return None


def load_vocoder(path: str, data: dict):
    """A rectified vocoder export (NSF-BigVGAN or converted OpenVPI) or an
    OpenVPI NSF-HiFiGAN checkpoint as a module taking the normalised mel of ``data`` and f0, and its mel settings.

    Raises ``ValueError`` when ``path`` is neither or renders another mel."""
    from rvc.rectified.openvpi import ARCHITECTURE, NSFHiFiGAN, generator_state, openvpi_spec

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") == "rectified_vocoder" and checkpoint.get("architecture") == ARCHITECTURE:
        generator = NSFHiFiGAN(**checkpoint["config"]["vocoder"]["model"])
        generator.load_state_dict(checkpoint["model"])
        model = RawMelVocoder(generator, data)
        vocoder_data = checkpoint["config"]["data"]
    elif checkpoint.get("kind") == "rectified_vocoder":
        model = build_vocoder(checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        model.remove_weight_norm()
        vocoder_data = checkpoint["config"]["data"]
    else:
        state = generator_state(checkpoint)
        if state is None:
            raise ValueError(f"{path} is neither a rectified vocoder export nor an OpenVPI vocoder.")
        hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
        model = RawMelVocoder(generator, data)
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    return model.eval(), vocoder_data
