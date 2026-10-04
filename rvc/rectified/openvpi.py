"""OpenVPI NSF-HiFiGAN vocoders (SingingVocoders, DiffSinger), for inference.

Reads their training checkpoints (``state_dict`` with ``generator.*`` keys) and
their exports (``{"generator": ...}``), plain and pitch-controllable
(``mini_nsf``) alike, and converts them to rectified vocoder exports. The
architecture comes from the weights; what they cannot tell (rates, dilations,
the mel) from a ``config.json`` beside the file, else the 44.1 kHz, hop 512,
128-bin release defaults.

The generator is ported from SingingVocoders (MIT, see THIRD_PARTY_NOTICES).

Usage: python rvc/rectified/openvpi.py <checkpoint> <output.pth>
"""

import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1
#: ``architecture`` of a rectified vocoder export holding one of these.
ARCHITECTURE = "openvpi_nsf_hifigan"

#: The mel of the 44.1 kHz releases, in this repo's key names.
DEFAULT_MEL = {
    "sample_rate": 44100,
    "hop_length": 512,
    "n_fft": 2048,
    "win_length": 2048,
    "n_mels": 128,
    "mel_fmin": 40.0,
    "mel_fmax": 16000.0,
}
#: ``config.json`` (as ``export_ckpt.py`` writes it) -> this repo's key names.
CONFIG_MEL_KEYS = {
    "sampling_rate": "sample_rate",
    "hop_size": "hop_length",
    "n_fft": "n_fft",
    "win_size": "win_length",
    "num_mels": "n_mels",
    "fmin": "mel_fmin",
    "fmax": "mel_fmax",
}


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def activation_factory(activation: str, antialias: bool):
    """``channels -> module`` for the generator's activations: HiFi-GAN's leaky
    ReLU, or BigVGAN's SnakeBeta, oversampled 2x with ``antialias``."""
    if activation == "leaky_relu":
        return lambda channels: nn.LeakyReLU(LRELU_SLOPE)
    if activation == "snakebeta":
        from rvc.lib.algorithm.generators.pcph_bigvgan import snake_activation

        return lambda channels: snake_activation(channels, antialias)
    raise ValueError(f"activation must be 'leaky_relu' or 'snakebeta', not {activation!r}.")


class FilteredUpsample(nn.Module):
    """A conv at the input rate, then a fixed windowed-sinc upsampler, in
    place of the transposed conv: that one has to learn to filter the images
    of its own zero-stuffing, with a kernel only two input samples long."""

    def __init__(self, in_channels, out_channels, rate, filter_width, rolloff, filter_beta):
        super().__init__()
        from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

        self.conv = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.up = AntiAliasedUpsample1d(rate, filter_width=filter_width, rolloff=rolloff, filter_beta=filter_beta)

    def forward(self, x):
        x = self.conv(x)
        # The filter in FP32, as PCPH-BigVGAN runs its own under autocast.
        with torch.autocast(x.device.type, enabled=False):
            return self.up(x.float())


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilation, activation):
        super().__init__()
        self.acts1 = nn.ModuleList(activation(channels) for _ in dilation)
        self.acts2 = nn.ModuleList(activation(channels) for _ in dilation)
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
             for _ in dilation]
        )

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size, dilation, activation):
        super().__init__()
        self.acts = nn.ModuleList(activation(channels) for _ in dilation)
        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )

    def forward(self, x):
        for c, a in zip(self.convs, self.acts):
            x = c(a(x)) + x
        return x


class SourceModuleHnNSF(nn.Module):
    """Sine plus harmonics merged to one excitation, as in the plain NSF-HiFiGAN."""

    def __init__(self, sample_rate, harmonic_num, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.l_linear = nn.Linear(self.dim, 1)

    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sample_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1) * torch.arange(1, self.dim + 1, device=f0.device)
        start = torch.rand(1, 1, self.dim, device=f0.device)
        start[..., 0] = 0
        sines = torch.sin(2 * np.pi * (rad + start)) * self.sine_amp
        uv = F.interpolate((f0 > 0).float().transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sines)
        return torch.tanh(self.l_linear(sines * uv + noise))


class NSFHiFiGAN(nn.Module):
    """OpenVPI's ``Generator``: log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop].

    ``activation`` "snakebeta" puts SnakeBeta in the residual blocks and before
    the output conv, oversampled 2x with ``antialias``; the leaky ReLU before
    each upsampler stays. ``upsampling`` "filtered" replaces the transposed
    convs with ``FilteredUpsample``, by ``upsample_filters``: each stage's
    (filter width, rolloff, beta). Neither is OpenVPI's, whose checkpoints hold
    no such weights."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, mini_nsf, harmonic_num=8, noise_sigma=0.0,
                 activation="leaky_relu", antialias=False, upsampling="transposed",
                 upsample_filters=None):
        super().__init__()
        if upsampling not in ("transposed", "filtered"):
            raise ValueError(f"upsampling must be 'transposed' or 'filtered', not {upsampling!r}.")
        make_activation = activation_factory(activation, antialias)
        self.num_kernels = len(resblock_kernel_sizes)
        self.mini_nsf = mini_nsf
        self.noise_sigma = noise_sigma
        if mini_nsf:
            # The sine is made at the rate after the second stage and joins there.
            self.source_sr = sample_rate / int(np.prod(upsample_rates[2:]))
            self.upp = int(np.prod(upsample_rates[:2]))
        else:
            self.upp = int(np.prod(upsample_rates))
            self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
            self.noise_convs = nn.ModuleList()

        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        block = ResBlock1 if resblock == "1" else ResBlock2
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            if upsampling == "filtered":
                self.ups.append(FilteredUpsample(2 * ch, ch, u, *upsample_filters[i]))
            else:
                self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block(ch, kernel, dilation, make_activation))
            if not mini_nsf:
                if i + 1 < len(upsample_rates):
                    stride = int(np.prod(upsample_rates[i + 1:]))
                    self.noise_convs.append(
                        nn.Conv1d(1, ch, kernel_size=stride * 2, stride=stride, padding=stride // 2)
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = nn.Conv1d(1, ch, 1)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        # HiFi-GAN's last leaky ReLU is at PyTorch's default slope.
        self.act_post = nn.LeakyReLU() if activation == "leaky_relu" else make_activation(ch)

    def _fast_sine(self, f0):
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        return torch.sin(2 * np.pi * rad.reshape(f0.shape[0], 1, -1))

    def forward(self, mel, f0):
        f0 = f0.float()
        source = self._fast_sine(f0) if self.mini_nsf else self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(mel)
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            if not self.mini_nsf:
                x = x + self.noise_convs[i](source)
            elif i == 1:
                x = x + self.source_conv(source)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(self.act_post(x)))


def generator_state(checkpoint: dict):
    """The generator weights of an OpenVPI checkpoint, or None when it is not one."""
    if isinstance(checkpoint.get("generator"), dict):
        return checkpoint["generator"]
    state = checkpoint.get("state_dict")
    if isinstance(state, dict) and "generator.conv_pre.bias" in state:
        return {k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")}
    return None


def fold_weight_norm(state: dict) -> dict:
    """``weight_g``/``weight_v`` pairs -> plain ``weight`` (weight norm over dim 0)."""
    folded = {}
    for key, value in state.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            g = state[base + "_g"]
            norm = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
            folded[base] = value * (g / norm)
        else:
            folded[key] = value
    return folded


def _read_config(path: str) -> dict:
    config_path = os.path.join(os.path.dirname(path), "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def openvpi_spec(path: str, state: dict):
    """``(hparams, mel, weights)`` for ``state`` (from ``generator_state``):
    ``NSFHiFiGAN`` arguments, the mel in this repo's key names, and the
    weights with weight norm folded."""
    config = _read_config(path)
    state = fold_weight_norm(state)
    mel = dict(DEFAULT_MEL)
    for source, target in CONFIG_MEL_KEYS.items():
        if source in config:
            mel[target] = config[source]

    count = sum(1 for key in state if key.startswith("ups.") and key.endswith(".weight"))
    kernels = [state[f"ups.{i}.weight"].shape[-1] for i in range(count)]
    # OpenVPI's upsamplers are all ``kernel = 2 * rate``.
    rates = config.get("upsample_rates") or [k // 2 for k in kernels]
    resblock = "1" if "resblocks.0.convs2.0.weight" in state else "2"
    blocks = {int(key.split(".")[1]) for key in state if key.startswith("resblocks.")}
    per_stage = len(blocks) // count
    first_conv = "convs1.0.weight" if resblock == "1" else "convs.0.weight"
    resblock_kernels = [state[f"resblocks.{j}.{first_conv}"].shape[-1] for j in range(per_stage)]
    default_dilation = [1, 3, 5] if resblock == "1" else [1, 3]
    dilations = config.get("resblock_dilation_sizes") or [default_dilation] * per_stage
    mini_nsf = "source_conv.weight" in state
    harmonics = 8 if mini_nsf else state["m_source.l_linear.weight"].shape[1] - 1

    hop = int(np.prod(rates))
    if hop != int(mel["hop_length"]):
        raise ValueError(f"{path} upsamples by {hop}, but its mel hop is {mel['hop_length']}.")
    if state["conv_pre.weight"].shape[1] != int(mel["n_mels"]):
        raise ValueError(f"{path} takes {state['conv_pre.weight'].shape[1]} mel bins, not {mel['n_mels']}.")

    hparams = dict(
        sample_rate=int(mel["sample_rate"]),
        num_mels=int(mel["n_mels"]),
        upsample_initial_channel=state["conv_pre.weight"].shape[0],
        upsample_rates=rates,
        upsample_kernel_sizes=kernels,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernels,
        resblock_dilation_sizes=dilations,
        mini_nsf=mini_nsf,
        harmonic_num=harmonics,
        noise_sigma=float(config.get("noise_sigma") or 0.0),
    )
    return hparams, mel, state


def export_openvpi(path: str, output: str) -> None:
    """Write the generator of the OpenVPI checkpoint at ``path`` as a rectified
    vocoder export, without the discriminator and optimiser state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = generator_state(checkpoint)
    if state is None:
        raise ValueError(f"{path} is not an OpenVPI vocoder checkpoint.")
    hparams, mel, weights = openvpi_spec(path, state)
    NSFHiFiGAN(**hparams).load_state_dict(weights)
    torch.save(
        {
            "kind": "rectified_vocoder",
            "architecture": ARCHITECTURE,
            "config": {"data": mel, "vocoder": {"model": hparams}},
            "model": {k: v.float().contiguous() for k, v in weights.items()},
            "source": os.path.basename(path),
        },
        output,
    )


if __name__ == "__main__":
    export_openvpi(sys.argv[1], sys.argv[2])
