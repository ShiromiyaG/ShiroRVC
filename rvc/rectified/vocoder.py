"""The rectified pipeline's vocoder, ChouwaGAN or SiFi-GAN: its recipe, shipped
per architecture and sample rate in ``rvc/configs/rectified/<architecture>``
and copied into each experiment on first use (ChouwaGAN's unless the
experiment is reset to another), and the loader of its exports, which also
renders OpenVPI's NSF-HiFiGANs."""

import json
import math
import os
import shutil

import torch
from torch import nn

from rvc.lib.paths import LOGS_DIR, ROOT
from rvc.rectified import openvpi, sifigan
from rvc.rectified.chouwagan import ARCHITECTURE, SOURCE_F0_FLOOR, ChouwaGAN, known_contrast
from rvc.rectified.common import load_run_config
from rvc.rectified.mel import denormalize_mel

#: Mel settings a vocoder must share with the flow to render its output.
MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax")
#: The generator of each ``architecture`` a recipe or an export names.
GENERATORS = {ARCHITECTURE: ChouwaGAN, sifigan.ARCHITECTURE: sifigan.SiFiGAN}
#: Exports rendered but not trained here, OpenVPI's NSF-HiFiGANs, by ``architecture``.
INFERENCE_ONLY = {openvpi.ARCHITECTURE: openvpi.NSFHiFiGAN, openvpi.LEGACY_ARCHITECTURE: openvpi.NSFHiFiGAN}
#: Every ``architecture`` an export may name.
EXPORT_ARCHITECTURES = (*GENERATORS, *INFERENCE_ONLY)
#: The shipped recipes, one ``<architecture>/<sample rate>.json`` each.
RECIPE_DIR = os.path.join(ROOT, "rvc", "configs", "rectified")
#: The rate of an experiment's recipe that names none: those predate the others.
LEGACY_RECIPE_RATE = 44100
#: ``Diffuser``'s bins turn from here up, faded in over the kHz below.
DIFFUSE_HZ = 4000.0
DIFFUSE_SEED = 0


def recipe_path(model_name: str) -> str:
    """The experiment's copy of the recipe, whatever its architecture (the
    name predates SiFi-GAN)."""
    return os.path.join(LOGS_DIR, model_name, f"rectified_vocoder_{ARCHITECTURE}.json")


def recipe_architecture(settings: dict) -> str:
    """The ``architecture`` a recipe trains; ChouwaGAN when it names none."""
    architecture = settings.get("architecture", ARCHITECTURE)
    if architecture not in GENERATORS:
        raise ValueError(f"The vocoder architecture must be one of {list(GENERATORS)}, not {architecture!r}.")
    return architecture


def generator_architecture(generator: nn.Module) -> str:
    return next(name for name, cls in GENERATORS.items() if isinstance(generator, cls))


def experiment_architecture(model_name: str) -> str:
    """The architecture of the experiment's recipe, ChouwaGAN before it has one."""
    path = recipe_path(model_name)
    if not os.path.exists(path):
        return ARCHITECTURE
    with open(path, encoding="utf-8") as handle:
        return recipe_architecture(json.load(handle))


def shipped_recipe(sample_rate: int, architecture: str = ARCHITECTURE) -> str:
    """Path of the shipped ``architecture`` recipe for ``sample_rate``."""
    path = os.path.join(RECIPE_DIR, architecture, f"{int(sample_rate)}.json")
    if not os.path.exists(path):
        raise ValueError(f"There is no {architecture} vocoder recipe for {sample_rate} Hz.")
    return path


def shipped_vocoder_settings(sample_rate: int, architecture: str = ARCHITECTURE) -> dict:
    """The recipe a new experiment at ``sample_rate`` starts from."""
    with open(shipped_recipe(sample_rate, architecture), encoding="utf-8") as handle:
        return json.load(handle)


def load_recipe_config(model_name: str) -> dict:
    """The experiment's rectified config with its recipe as the ``vocoder``
    section. The recipe is copied from the shipped one for the experiment's
    sample rate on first use, as ``load_run_config`` does."""
    config = load_run_config(model_name)
    rate = int(config["data"]["sample_rate"])
    path = recipe_path(model_name)
    if not os.path.exists(path):
        shutil.copyfile(shipped_recipe(rate), path)
    with open(path, encoding="utf-8") as handle:
        config["vocoder"] = json.load(handle)
    recipe_rate = int(config["vocoder"].get("sample_rate", LEGACY_RECIPE_RATE))
    if recipe_rate != rate:
        raise ValueError(
            f"{path} is the {recipe_rate} Hz recipe, and {model_name} is at {rate} Hz. "
            f"Reset the vocoder settings in the Config tab, or delete the file."
        )
    return config


def resolve_discriminator(settings: dict) -> dict:
    """``settings`` whose ``discriminator`` names one of its ``discriminators``
    with that section in its place; the section's ``gan_loss`` and
    ``san_loss`` become the recipe's. A section given as such is kept."""
    choice = settings.get("discriminator")
    if not isinstance(choice, str):
        return settings
    presets = settings.get("discriminators") or {}
    if choice not in presets:
        raise ValueError(f"discriminator {choice!r} is not one of the recipe's discriminators {list(presets)}.")
    section = dict(presets[choice])
    resolved = {key: value for key, value in settings.items() if key != "discriminators"}
    for key in ("gan_loss", "san_loss"):
        if key in section:
            resolved[key] = section.pop(key)
    resolved["discriminator"] = section
    return resolved


def read_vocoder_settings(model_name: str) -> dict:
    """The recipe the experiment's vocoder trains by, as stored."""
    return load_recipe_config(model_name)["vocoder"]


def write_vocoder_settings(model_name: str, settings: dict) -> None:
    with open(recipe_path(model_name), "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=4)


def reset_vocoder_settings(model_name: str, architecture: str | None = None) -> None:
    """Back to the shipped recipe: ``architecture``'s for the experiment's
    sample rate, or ChouwaGAN's, copied again on next use."""
    if os.path.exists(recipe_path(model_name)):
        os.remove(recipe_path(model_name))
    if architecture is not None and architecture != ARCHITECTURE:
        rate = load_run_config(model_name)["data"]["sample_rate"]
        shutil.copyfile(shipped_recipe(rate, architecture), recipe_path(model_name))


class Diffuser(nn.Module):
    """[B, 1, T] and the f0 [B, frames] it was rendered from -> [B, 1, T].
    Above ``DIFFUSE_HZ``, each bin of a ``n_fft`` STFT turns by a fixed random
    phase of the harmonic of f0 nearest to it, so each harmonic turns whole
    and keeps its phase as it moves. The pulse train has every harmonic in
    phase, which up there sounds like static; scattered harmonic phases spread
    each pulse's top over the cycle and leave the harmonics sharp, where a
    phase per bin blurred them. A generator trained through either learned
    steady lines, so it runs at inference only."""

    def __init__(self, sample_rate, n_fft, hop, frame_hop):
        super().__init__()
        self.n_fft, self.hop, self.frame_hop = int(n_fft), int(hop), int(frame_hop)
        bins = torch.linspace(0, sample_rate / 2, self.n_fft // 2 + 1).view(1, -1, 1)
        generator = torch.Generator().manual_seed(DIFFUSE_SEED)
        phases = math.pi * (2 * torch.rand(int(sample_rate / 2 / SOURCE_F0_FLOOR) + 2, generator=generator) - 1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("ramp", ((bins - DIFFUSE_HZ) / 1000 + 1).clamp(0, 1), persistent=False)
        self.register_buffer("phases", phases, persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio, f0):
        x, window = audio.squeeze(1).float(), self.window.float()
        spectrum = torch.stft(x, self.n_fft, self.hop, window=window, return_complex=True)
        frame = torch.arange(spectrum.shape[-1], device=x.device) * self.hop // self.frame_hop
        f0 = f0.float()[:, frame.clamp(max=f0.shape[-1] - 1)].unsqueeze(1)
        harmonic = torch.round(self.bins / f0.clamp(min=SOURCE_F0_FLOOR)).long().clamp(max=len(self.phases) - 1)
        phase = self.phases[harmonic] * self.ramp * (f0 > 0)
        spectrum = spectrum * torch.polar(torch.ones_like(phase), phase)
        x = torch.istft(spectrum, self.n_fft, self.hop, window=window, length=x.shape[-1])
        return x.unsqueeze(1).to(audio.dtype)


class FlowFeatures(nn.Module):
    """A vocoder fed what the flow produces: the normalised mel and, from a
    flow with ``contrast_band_edges``, the band contrast after its bins. That
    goes to a vocoder that reads the same bands, and is dropped otherwise.
    With ``diffuse``, the output goes through a ``Diffuser``, unless the
    vocoder already scatters its harmonics' phases (``source_phase_hz``) or
    learns them (``post_filter``, the Wavehax ``synthesis``, SiFi-GAN)."""

    def __init__(self, vocoder: nn.Module, data: dict, diffuse: bool = True):
        super().__init__()
        self.vocoder = vocoder
        self.diffuser = None
        own_phase = (not isinstance(vocoder, ChouwaGAN)
                     or vocoder.source_phase_hz is not None
                     or vocoder.post_filter is not None
                     or vocoder.synthesis is not None)
        if diffuse and not own_phase:
            self.diffuser = Diffuser(vocoder.sample_rate, vocoder.noise_fft, vocoder.synthesis_hop, vocoder.hop_length)
        self.n_mels = int(data["n_mels"])
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])
        contrast = getattr(vocoder, "contrast", None)
        self.contrast_bands = contrast.bands if contrast is not None else 0

    def forward(self, mel, f0):
        audio = self._render(mel, f0)
        return audio if self.diffuser is None else self.diffuser(audio, f0)

    def _render(self, mel, f0):
        if mel.shape[1] == self.n_mels:
            return self.vocoder(mel, f0)
        mel, contrast = mel[:, :self.n_mels], mel[:, self.n_mels:]
        if contrast.shape[1] != self.contrast_bands:
            return self.vocoder(mel, f0)
        # The contrast rows went through the mel's normalisation with the rest.
        return self.vocoder(mel, f0, known_contrast(contrast * self.mel_std + self.mel_mean))


class RawMelVocoder(nn.Module):
    """An OpenVPI vocoder fed the flow's normalised mel: the log mel it takes,
    the band contrast rows dropped."""

    def __init__(self, vocoder: nn.Module, data: dict):
        super().__init__()
        self.vocoder = vocoder
        self.data = {key: data[key] for key in ("n_mels", "mel_mean", "mel_std")}

    def forward(self, mel, f0):
        return self.vocoder(denormalize_mel(mel[:, :self.data["n_mels"]], self.data), f0)


def mel_mismatch(data: dict, vocoder_data: dict):
    """The first mel setting the vocoder does not share with ``data``, or None."""
    for key in MEL_KEYS + ("mel_mean", "mel_std"):
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            return f"{key} ({vocoder_data[key]} vs {data[key]})"
    return None


def load_vocoder(path: str, data: dict, diffuse: bool = True):
    """A ChouwaGAN, SiFi-GAN or OpenVPI NSF-HiFiGAN export as a module taking
    the flow's output for ``data`` and f0 (a ChouwaGAN's through a ``Diffuser``
    with ``diffuse``), and its mel settings.

    Raises ``ValueError`` when ``path`` is not one or renders another mel."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if checkpoint.get("kind") != "rectified_vocoder" or architecture not in EXPORT_ARCHITECTURES:
        raise ValueError(f"{path} is not a rectified vocoder export ({', '.join(EXPORT_ARCHITECTURES)}).")
    vocoder_data = checkpoint["config"]["data"]
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    if architecture in INFERENCE_ONLY:
        generator = INFERENCE_ONLY[architecture](**checkpoint["config"]["vocoder"]["model"])
        generator.load_state_dict(checkpoint["model"])
        return RawMelVocoder(generator, data).eval(), vocoder_data
    generator = GENERATORS[architecture](**checkpoint["config"]["vocoder"]["model"])
    generator.load_state_dict(checkpoint["model"])
    return FlowFeatures(generator, data, diffuse).eval(), vocoder_data
