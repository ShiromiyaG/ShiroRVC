"""The vocoders the rectified pipeline trains by their own recipes: OpenVPI
SingingVocoders' PC-NSF-HiFiGAN, NSF-HiFiGAN and NSF-UnivNet, and Wavehax
(``wavehax.py``). Trained by ``train_vocoder.py``.

The NSF-UnivNet generator, the discriminators and the losses are ported from
SingingVocoders (MIT, see THIRD_PARTY_NOTICES); the NSF-HiFiGAN generator is
``openvpi.NSFHiFiGAN``. Module names follow SingingVocoders', so its
checkpoints load as starting points through ``training_state``.
"""

import json
import math
import os
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.paths import LOGS_DIR, ROOT
from rvc.rectified.common import DEFAULT_CONFIG, load_run_config
from rvc.rectified import wavehax
from rvc.rectified.openvpi import ARCHITECTURE, NSFHiFiGAN, SourceModuleHnNSF

LRELU_SLOPE = 0.1
#: The trainer's own vocoder, and the ones trained here.
PCPH = "pcph-bigvgan"
#: Discriminators a SingingVocoders run trains against: the recipe's own, or
#: the repo's v3 with its GAN losses.
DISCRIMINATORS = ("original", "v3")
ARCHITECTURES = ("pc-nsf-hifigan", "nsf-hifigan", "nsf-univnet", "wavehax", "wavehax-v2")
UNIVNET = "nsf-univnet"
WAVEHAX = wavehax.ARCHITECTURE
#: The runs that train a Wavehax generator, each by its own recipe.
WAVEHAX_RUNS = (WAVEHAX, wavehax.V2)
#: The ones trained on the raw log mel, as SingingVocoders does; the others
#: take the pipeline's normalised one.
RAW_MEL = ("pc-nsf-hifigan", "nsf-hifigan", "nsf-univnet")
#: ``architecture`` of an NSF-UnivNet rectified vocoder export.
UNIVNET_ARCHITECTURE = "nsf_univnet"
#: Training architecture -> ``architecture`` of its export.
EXPORT_ARCHITECTURE = {
    "pc-nsf-hifigan": ARCHITECTURE, "nsf-hifigan": ARCHITECTURE, UNIVNET: UNIVNET_ARCHITECTURE,
    WAVEHAX: WAVEHAX, wavehax.V2: WAVEHAX,
}
RECIPE_DIR = os.path.join(ROOT, "rvc", "configs", "rectified", "vocoders")


def recipe_path(model_name: str, architecture: str) -> str:
    return os.path.join(LOGS_DIR, model_name, f"rectified_vocoder_{architecture}.json")


def shipped_vocoder_settings(architecture: str) -> dict:
    """The ``vocoder`` section a new experiment starts from."""
    path = DEFAULT_CONFIG if architecture == PCPH else os.path.join(RECIPE_DIR, f"{architecture}.json")
    with open(path, encoding="utf-8") as handle:
        settings = json.load(handle)
    return settings["vocoder"] if architecture == PCPH else settings


def read_vocoder_settings(model_name: str, architecture: str) -> dict:
    """The ``vocoder`` section a run of ``architecture`` trains by, as stored."""
    if architecture == PCPH:
        return load_run_config(model_name)["vocoder"]
    # Copies the shipped recipe on first use.
    load_recipe_config(model_name, architecture)
    with open(recipe_path(model_name, architecture), encoding="utf-8") as handle:
        return json.load(handle)


def write_vocoder_settings(model_name: str, architecture: str, settings: dict) -> None:
    """Store ``settings`` as that section: in the experiment's rectified config
    for PCPH-BigVGAN, in its recipe file for the others."""
    path, content = recipe_path(model_name, architecture), settings
    if architecture == PCPH:
        path = os.path.join(LOGS_DIR, model_name, "rectified_config.json")
        content = {**load_run_config(model_name), "vocoder": settings}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=4)


def reset_vocoder_settings(model_name: str, architecture: str) -> None:
    """Back to the shipped section."""
    if architecture == PCPH:
        write_vocoder_settings(model_name, architecture, shipped_vocoder_settings(architecture))
    elif os.path.exists(recipe_path(model_name, architecture)):
        os.remove(recipe_path(model_name, architecture))


def load_recipe_config(model_name: str, architecture: str) -> dict:
    """The experiment's rectified config with ``architecture``'s recipe as its
    ``vocoder`` section. The recipe is copied from the shipped one on first
    use, as ``load_run_config`` does."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown vocoder architecture {architecture!r}.")
    config = load_run_config(model_name)
    repo = config["vocoder"]
    path = recipe_path(model_name, architecture)
    if not os.path.exists(path):
        shutil.copyfile(os.path.join(RECIPE_DIR, f"{architecture}.json"), path)
    with open(path, encoding="utf-8") as handle:
        config["vocoder"] = json.load(handle)
    # The repo's discriminator, for runs that pick it over the recipe's: the
    # experiment's own settings at v3, unless the recipe file sets them. Its
    # mel conditioning is off: it cannot score pitch-shifted renders.
    config["vocoder"].setdefault(
        "repo_discriminator", {**repo["discriminator"], "d_version": "v3", "d_mrd_mel_cond": False}
    )
    config["vocoder"].setdefault("c_fm", repo["c_fm"])
    return config


class GLU(nn.Module):
    def forward(self, x):
        out, gate = x.chunk(2, dim=1)
        return out * gate.sigmoid()


class MelUpsampler(nn.Module):
    """Doubles the mel's frame rate: [B, n_mels, T] -> [B, n_mels, 2T]."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, kernel_size=1)
        self.UP = nn.ConvTranspose2d(4, 8, [3, 32], stride=[1, 2], padding=[1, 15])
        self.c2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(4, 2, kernel_size=1)
        self.glu = GLU()

    def forward(self, x):
        x = self.glu(self.UP(self.glu(self.c1(x.unsqueeze(1)))))
        x = self.glu(self.c2(x)) + x
        return self.glu(self.c3(x)).squeeze(1)


class DownBlock(nn.Module):
    def __init__(self, down, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv1d(in_channels, out_channels * 2, kernel_size=down * 2, stride=down, padding=down // 2)
        self.out = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.glu = GLU()

    def forward(self, x):
        return F.gelu(self.out(self.glu(self.c(x))))


class SourceDown(nn.Module):
    """The excitation at each LVC block's rate, coarsest first."""

    def __init__(self, channels, upsample_rates):
        super().__init__()
        downs = list(reversed(upsample_rates))[:-1]
        self.fistpp = nn.Conv1d(1, channels, kernel_size=1)
        self.downs = nn.ModuleList(
            DownBlock(rate, channels * i if i else 1, channels * (i + 1)) for i, rate in enumerate(downs)
        )
        self.ppls = nn.ModuleList(nn.Conv1d(channels * (i + 1), channels, kernel_size=1) for i in range(len(downs)))

    def forward(self, x):
        out = [self.fistpp(x)]
        for down, project in zip(self.downs, self.ppls):
            x = down(x)
            out.append(project(x))
        return out[::-1]


class KernelPredictor(nn.Module):
    """Per-frame kernels and biases of a block's location-variable convolutions."""

    def __init__(self, cond_channels, in_channels, out_channels, layers, kernel_size, hidden, conv_size, dropout):
        super().__init__()
        self.shape = (layers, in_channels, out_channels, kernel_size)
        padding = (conv_size - 1) // 2

        def conv():
            return [nn.Conv1d(hidden, hidden, conv_size, padding=padding), nn.LeakyReLU(0.1)]

        self.input_conv = nn.Sequential(nn.Conv1d(cond_channels, hidden, 5, padding=2), nn.LeakyReLU(0.1))
        self.residual_conv = nn.Sequential(
            nn.Dropout(dropout), *conv(), *conv(), nn.Dropout(dropout), *conv(), *conv(),
            nn.Dropout(dropout), *conv(), *conv(),
        )
        self.kernel_conv = nn.Conv1d(hidden, in_channels * out_channels * kernel_size * layers, conv_size, padding=padding)
        self.bias_conv = nn.Conv1d(hidden, out_channels * layers, conv_size, padding=padding)

    def forward(self, c):
        layers, in_channels, out_channels, kernel_size = self.shape
        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        kernels = self.kernel_conv(c).view(c.shape[0], layers, in_channels, out_channels, kernel_size, c.shape[-1])
        bias = self.bias_conv(c).view(c.shape[0], layers, out_channels, c.shape[-1])
        return kernels, bias


class LVCBlock(nn.Module):
    """Upsamples, takes the excitation in, then runs the location-variable convolutions."""

    def __init__(self, channels, cond_channels, upsample_ratio, layers, kernel_size, cond_hop_length,
                 kpnet_hidden_channels, kpnet_conv_size, dropout):
        super().__init__()
        self.cond_hop_length = cond_hop_length
        self.nsfppj = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.upsample = nn.ConvTranspose1d(
            channels, channels, kernel_size=upsample_ratio * 2, stride=upsample_ratio,
            padding=upsample_ratio // 2 + upsample_ratio % 2, output_padding=upsample_ratio % 2,
        )
        self.kernel_predictor = KernelPredictor(
            cond_channels, channels, 2 * channels, layers, kernel_size,
            kpnet_hidden_channels, kpnet_conv_size, dropout,
        )
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, padding=3**i * ((kernel_size - 1) // 2), dilation=3**i)
            for i in range(layers)
        )

    def forward(self, x, c, source):
        channels = x.shape[1]
        kernels, bias = self.kernel_predictor(c)
        x = self.upsample(F.leaky_relu(x, 0.2))
        x = self.nsfppj(torch.cat([x, source], dim=1))
        for i, conv in enumerate(self.convs):
            y = F.leaky_relu(conv(F.leaky_relu(x, 0.2)), 0.2)
            y = self._lvc(y, kernels[:, i], bias[:, i])
            x = x + torch.sigmoid(y[:, :channels]) * torch.tanh(y[:, channels:])
        return x

    def _lvc(self, x, kernel, bias):
        """``x`` [B, C, frames * hop] convolved with each frame's own ``kernel``
        [B, C, out, K, frames], plus ``bias`` [B, out, frames]."""
        batch, _, out_channels, kernel_size, _ = kernel.shape
        hop = self.cond_hop_length
        padding = (kernel_size - 1) // 2
        x = F.pad(x, (padding, padding)).unfold(2, hop + 2 * padding, hop).unfold(3, kernel_size, 1)
        out = torch.einsum("bilsk,biokl->bols", x, kernel) + bias.unsqueeze(-1)
        return out.reshape(batch, out_channels, -1)


class NSFUnivNet(nn.Module):
    """SingingVocoders' ``nsfUnivNet``: log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop].

    UnivNet's LVC stack over noise drawn here, with the NSF excitation taken
    in at every block."""

    def __init__(self, sample_rate, num_mels, upsample_rates, channels, noise_channels, lvc_layers,
                 lvc_kernel_size, kpnet_hidden_channels, kpnet_conv_size, upmel=2, dropout=0.0,
                 harmonic_num=8):
        super().__init__()
        doublings = int(math.log2(upmel))
        if 2**doublings != upmel:
            raise ValueError(f"upmel must be a power of two, not {upmel}.")
        self.noise_channels = noise_channels
        self.upmel = upmel
        self.upp = int(np.prod(upsample_rates)) * upmel
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
        self.ddspd = SourceDown(channels, upsample_rates)
        self.upblocke = nn.Sequential(*[MelUpsampler() for _ in range(doublings)])
        self.first_conv = nn.Conv1d(noise_channels, channels, kernel_size=7, padding=3)
        self.lvc_blocks = nn.ModuleList()
        cond_hop_length = 1
        for rate in upsample_rates:
            cond_hop_length *= rate
            self.lvc_blocks.append(
                LVCBlock(channels, num_mels, rate, lvc_layers, lvc_kernel_size, cond_hop_length,
                         kpnet_hidden_channels, kpnet_conv_size, dropout)
            )
        self.last_conv_layers = nn.ModuleList([nn.Conv1d(channels, 1, kernel_size=7, padding=3)])

    def forward(self, mel, f0):
        sources = self.ddspd(self.m_source(f0.float(), self.upp).transpose(1, 2))
        noise = torch.randn(
            mel.shape[0], self.noise_channels, mel.shape[-1] * self.upmel, device=mel.device, dtype=mel.dtype
        )
        x = self.first_conv(noise)
        c = self.upblocke(mel)
        for block, source in zip(self.lvc_blocks, sources):
            x = block(x, c, source)
        return torch.tanh(self.last_conv_layers[0](F.leaky_relu(x, LRELU_SLOPE)))


#: Export ``architecture`` -> generator class, for ``vocoder.load_vocoder``.
GENERATORS = {
    ARCHITECTURE: NSFHiFiGAN, UNIVNET_ARCHITECTURE: NSFUnivNet, WAVEHAX: wavehax.WavehaxGenerator,
}
#: The exports whose generator takes the raw log mel.
RAW_MEL_EXPORTS = (ARCHITECTURE, UNIVNET_ARCHITECTURE)


def generator_hparams(architecture: str, config: dict) -> dict:
    """The generator's constructor arguments, from the recipe and the mel."""
    data, model = config["data"], config["vocoder"]["model"]
    if architecture in WAVEHAX_RUNS:
        return dict(
            sample_rate=int(data["sample_rate"]), num_mels=int(data["n_mels"]),
            hop_length=int(data["hop_length"]), **model,
        )
    hop = int(np.prod(model["upsample_rates"])) * int(model.get("upmel", 1) if architecture == UNIVNET else 1)
    if hop != int(data["hop_length"]):
        raise ValueError(f"The {architecture} recipe upsamples by {hop}, but the mel hop is {data['hop_length']}.")
    hparams = dict(sample_rate=int(data["sample_rate"]), num_mels=int(data["n_mels"]), **model)
    if hparams.get("upsampling") == "filtered" and not hparams.get("upsample_filters"):
        # The filters are not saved with the weights: the export has to name them.
        hparams["upsample_filters"] = upsample_filters(model["upsample_rates"])
    return hparams


def upsample_filters(rates) -> list:
    """Each stage's (filter width, rolloff, beta) for filtered upsampling.

    The first stage is short: its input is a few dozen frames, and a long
    kernel reads mostly padding. A later stage of x4 or more leaves several
    images inside the band the next stages work in, so it takes the deeper
    stopband (beta 9, ~90 dB); the x2 stages have the input length for the
    longest filter. Each rolloff puts the stopband's start on the input's
    Nyquist, by Kaiser's formulas."""
    filters = []
    for stage, rate in enumerate(rates):
        width, beta = (12, 6.0) if stage == 0 else (32, 9.0) if rate >= 4 else (48, 9.0)
        attenuation = beta / 0.1102 + 8.7
        filters.append([width, round(1.0 - (attenuation - 8.0) / (28.72 * width), 2), beta])
    return filters


def build_generator(architecture: str, hparams: dict) -> nn.Module:
    """The generator as its recipe trains it: SingingVocoders' with their
    weight norm and init."""
    if architecture in WAVEHAX_RUNS:
        return wavehax.WavehaxGenerator(**hparams)
    if architecture == UNIVNET:
        generator = NSFUnivNet(**hparams)
        normed = [m for m in generator.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
    else:
        generator = NSFHiFiGAN(**hparams)
        blocks = [
            conv for block in generator.resblocks
            for name in ("convs1", "convs2", "convs") for conv in getattr(block, name, ())
        ]
        # A filtered upsampler's weights are its conv's.
        ups = [getattr(up, "conv", up) for up in generator.ups]
        normed = [generator.conv_pre, generator.conv_post, *ups, *blocks]
        # HiFi-GAN's normal init runs after weight norm there, where it only
        # reaches the biases; the source conv has no weight norm and takes it whole.
        for module in (generator.conv_post, *ups, *blocks):
            module.bias.data.normal_(0.0, 0.01)
        if generator.mini_nsf:
            generator.source_conv.weight.data.normal_(0.0, 0.01)
            generator.source_conv.bias.data.normal_(0.0, 0.01)
    for module in normed:
        weight_norm(module)
    return generator


_PARAMETRIZED = ".parametrizations.weight."


def _row_norm(weight):
    return weight.flatten(1).norm(dim=1).view(-1, *([1] * (weight.dim() - 1)))


def training_state(state: dict, reference: dict) -> dict:
    """``state`` under the names of ``reference``, a training model's state. A
    plain ``weight`` it holds under weight norm is split, and the names of the
    older weight and spectral norm (``weight_g``, ``weight_v``, ``weight_orig``,
    ``weight_u``), which SingingVocoders' checkpoints carry, become the
    parametrizations'."""
    renamed = {}
    for key, value in state.items():
        base, _, leaf = key.rpartition(".")
        prefix = base + _PARAMETRIZED
        if leaf == "weight" and prefix + "original1" in reference:
            renamed[prefix + "original0"] = _row_norm(value)
            renamed[prefix + "original1"] = value
            continue
        # ``weight_v`` is the direction under weight norm, a power-iteration
        # vector under spectral norm.
        target = {
            "weight_g": "original0",
            "weight_v": "0._v" if prefix + "original" in reference else "original1",
            "weight_orig": "original",
            "weight_u": "0._u",
        }.get(leaf)
        renamed[prefix + target if target and prefix + target in reference else key] = value
    return renamed


def load_pretrained_generator(generator: nn.Module, weights: dict) -> None:
    """Load a starting point into the training generator. The only weights it
    may lack are the activations', which a leaky-ReLU pretrain does not have."""
    missing, unexpected = generator.load_state_dict(
        training_state(weights, generator.state_dict()), strict=False
    )
    missing = [key for key in missing if ".acts" not in key and not key.startswith("act_post.")]
    if missing or unexpected:
        raise ValueError(f"The pretrained generator does not fit: missing {missing}, unexpected {unexpected}.")


def pretrained_states(path: str):
    """``(weights, discriminator weights or None)`` of a starting point: a
    SingingVocoders checkpoint (both networks) or export, or one of this
    trainer's checkpoints or exports."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        def part(prefix):
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

        return part("generator."), part("discriminator.") or None
    if isinstance(checkpoint.get("generator"), dict):
        return checkpoint["generator"], None
    return checkpoint["model"], None


def export_state(state: dict) -> dict:
    """A generator's training state with weight norm folded, as ``load_vocoder`` reads it."""
    folded = {}
    for key, value in state.items():
        if key.endswith(_PARAMETRIZED + "original0"):
            continue
        if key.endswith(_PARAMETRIZED + "original1"):
            base = key[: -len(_PARAMETRIZED + "original1")]
            value = value * (state[base + _PARAMETRIZED + "original0"] / _row_norm(value))
            key = base + ".weight"
        folded[key] = value.detach().float().cpu().contiguous()
    return folded


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 512, 1024)
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(a, b, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0)))
             for a, b in zip(channels, channels[1:])]
            + [weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class SpecDiscriminator(nn.Module):
    def __init__(self, fft_size, shift_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.discriminators = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        self.out = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        y = torch.stft(
            y.squeeze(1).float(), self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True
        ).abs().unsqueeze(1)
        for conv in self.discriminators:
            y = F.leaky_relu(conv(y), LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap


class _Heads(nn.Module):
    """A family of discriminators: ``(outputs, feature maps)``, one per head."""

    def _inputs(self, y):
        return [y] * len(self.discriminators)

    def forward(self, y):
        outputs, fmaps = [], []
        for discriminator, x in zip(self.discriminators, self._inputs(y)):
            output, fmap = discriminator(x)
            outputs.append(output)
            fmaps.append(fmap)
        return outputs, fmaps


class MultiPeriodDiscriminator(_Heads):
    def __init__(self, periods):
        super().__init__()
        self.discriminators = nn.ModuleList(DiscriminatorP(period) for period in periods)


class MultiScaleDiscriminator(_Heads):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()]
        )
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def _inputs(self, y):
        inputs = [y]
        for pool in self.meanpools:
            inputs.append(pool(inputs[-1]))
        return inputs


class MultiResSpecDiscriminator(_Heads):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.discriminators = nn.ModuleList(
            SpecDiscriminator(*sizes) for sizes in zip(fft_sizes, hop_sizes, win_lengths)
        )


class Discriminators(nn.ModuleDict):
    """SingingVocoders' discriminator dict; ``forward`` runs every family and
    returns ``{family: (outputs, feature maps)}``."""

    @property
    def use_spectral_norm(self) -> bool:
        """The MSD's first head is under spectral norm."""
        return "msd" in self

    def forward(self, y):
        return {name: family(y) for name, family in self.items()}


def build_discriminators(architecture: str, settings: dict) -> Discriminators:
    """MSD + MPD for the NSF-HiFiGANs, MRD + MPD for NSF-UnivNet, Wavehax's
    own MPD + MRD for it."""
    if architecture in WAVEHAX_RUNS:
        return Discriminators(wavehax.build_families(settings))
    if architecture == UNIVNET:
        spectral = {"mrd": MultiResSpecDiscriminator(
            settings["mrd_fft_sizes"], settings["mrd_hop_sizes"], settings["mrd_win_lengths"]
        )}
    else:
        spectral = {"msd": MultiScaleDiscriminator()}
    return Discriminators({**spectral, "mpd": MultiPeriodDiscriminator(settings["periods"])})


def discriminator_loss(real: dict, fake: dict, hinge: bool = False):
    """LSGAN (or hinge) loss over every head of every family, and its real and
    fake halves."""
    if hinge:
        loss_real = sum(torch.mean(F.relu(1 - d.float())) for outputs, _ in real.values() for d in outputs)
        loss_fake = sum(torch.mean(F.relu(1 + d.float())) for outputs, _ in fake.values() for d in outputs)
    else:
        loss_real = sum(torch.mean((1 - d.float()) ** 2) for outputs, _ in real.values() for d in outputs)
        loss_fake = sum(torch.mean(d.float() ** 2) for outputs, _ in fake.values() for d in outputs)
    return loss_real + loss_fake, loss_real, loss_fake


def adversarial_loss(outputs, hinge: bool = False) -> torch.Tensor:
    if hinge:
        return sum(-torch.mean(d.float()) for d in outputs)
    return sum(torch.mean((1 - d.float()) ** 2) for d in outputs)


def feature_loss(fmaps_real, fmaps_fake) -> torch.Tensor:
    """Feature matching over the clips both batches hold; the fake one is
    longer under pitch-controllable augmentation."""
    loss = 0.0
    for head_real, head_fake in zip(fmaps_real, fmaps_fake):
        for real, fake in zip(head_real, head_fake):
            count = min(real.shape[0], fake.shape[0])
            loss = loss + torch.mean(torch.abs(real[:count].float() - fake[:count].float()))
    return loss * 2


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude losses, averaged over the resolutions."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, x, index):
        fft_size, hop_size, win_length = self.resolutions[index]
        window = getattr(self, f"window_{index}")
        spec = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        return torch.clamp(spec.abs(), min=10**-3.5)

    def forward(self, x, y):
        """Predicted ``x`` and target ``y``, [B, T] each."""
        x, y = x.float(), y.float()
        sc_loss = mag_loss = 0.0
        for index in range(len(self.resolutions)):
            x_mag, y_mag = self._magnitude(x, index), self._magnitude(y, index)
            sc_loss = sc_loss + torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            mag_loss = mag_loss + F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss / len(self.resolutions), mag_loss / len(self.resolutions)


def volume_augment(mel, audio, probability: float):
    """SingingVocoders' volume augmentation: with ``probability`` per clip, a
    gain of e^-3 up to e^3 that keeps the peak under 1, on the audio and, as a
    shift, on its log mel."""
    peak = audio.abs().amax(dim=(1, 2)) + 1e-5
    high = torch.clamp(torch.log(1 / peak), max=3.0)
    shift = -3.0 + (high + 3.0) * torch.rand_like(high)
    shift = torch.where(torch.rand_like(high) < probability, shift, torch.zeros_like(shift))
    mel = torch.clamp(mel + shift.view(-1, 1, 1), min=math.log(1e-5))
    return mel, audio * shift.exp().view(-1, 1, 1)
