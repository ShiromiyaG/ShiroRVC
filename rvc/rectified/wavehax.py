"""Wavehax (Yoneyama et al., 2024): a vocoder that estimates the complex
spectrogram with 2D convs over the STFT of a harmonic prior, then inverts it.
It has no upsampling layers and no activation at the audio rate, so nothing
of its own aliases.

Generator and discriminators ported from https://github.com/chomeyama/wavehax
(MIT, see THIRD_PARTY_NOTICES); the prior is this repo's ``PCPHSource``.
Trained by ``train_vocoder.py`` with the recipe in
``rvc/configs/rectified/vocoders/wavehax.json``.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.commons import expand_f0
from rvc.lib.algorithm.generators.pcph_bigvgan import PCPHSource

try:
    from rvc.rectified.depthwise_triton import depthwise_conv2d
except ImportError:
    depthwise_conv2d = None

#: ``architecture`` of a Wavehax rectified vocoder export, and of its runs.
ARCHITECTURE = "wavehax"


class STFT(nn.Module):
    """Frames every ``hop_length`` with no centring: zero padding either side
    gives ``samples // hop_length`` frames, one per mel frame."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, x):
        """(batch, samples) -> real and imaginary parts, (batch, bins, frames)."""
        pad = self.n_fft - self.hop_length
        x = F.pad(x.float(), (pad // 2, pad - pad // 2))
        spec = torch.stft(
            x, self.n_fft, self.hop_length, window=self.window, center=False, return_complex=True
        )
        return spec.real, spec.imag

    def inverse(self, real, imag):
        """(batch, bins, frames) each -> (batch, 1, frames * hop_length), by
        overlap-add normalised by the window's own."""
        frames = real.shape[-1]
        length = (frames - 1) * self.hop_length + self.n_fft

        def overlap_add(windows):
            return F.fold(windows, (1, length), (1, self.n_fft), stride=(1, self.hop_length)).squeeze(2)

        x = torch.fft.irfft(torch.complex(real.float(), imag.float()), n=self.n_fft, dim=1)
        x = overlap_add(x * self.window[:, None])
        envelope = overlap_add(self.window.square()[None, :, None].expand(1, -1, frames))
        # Cropped before the division: the envelope is 0 at the very edges,
        # and dividing there puts NaN in every gradient.
        keep = slice((self.n_fft - self.hop_length) // 2, (self.n_fft - self.hop_length) // 2 + frames * self.hop_length)
        return x[..., keep] / envelope[..., keep]


class LayerNorm2d(nn.Module):
    """Normalises over channels, bins and frames together, with a per-channel
    affine. In FP32: the statistics run over the whole spectrogram."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.float()
        var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, correction=0)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return torch.addcmul(self.beta.view(1, -1, 1, 1), x, self.gamma.view(1, -1, 1, 1))


class ConvNeXtBlock2d(nn.Module):
    def __init__(self, channels, mult_channels, kernel_size, layer_scale):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.dwconv = nn.Conv2d(
            channels, channels, tuple(kernel_size), padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=channels, bias=False, padding_mode="reflect",
        )
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * mult_channels, 1)
        self.pwconv2 = nn.Conv2d(channels * mult_channels, channels, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(1, channels, 1, 1))

    def _depthwise(self, x):
        """``self.dwconv(x)``, through the Triton kernels on CUDA."""
        if depthwise_conv2d is None or not x.is_cuda:
            return self.dwconv(x)
        pad_h, pad_w = self.dwconv.padding
        return depthwise_conv2d(F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect"), self.dwconv.weight)

    def forward(self, x):
        return x + self.gamma * self.pwconv2(F.gelu(self.pwconv1(self.norm(self._depthwise(x)))))


class WavehaxGenerator(nn.Module):
    """Normalised mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop_length].

    The prior is the PCPH pulse train plus white noise at ``prior_noise``; its
    spectrogram, a projection of it and the projected mel are the input
    channels of a ConvNeXt stack over (bins, frames), which outputs the real
    and imaginary parts of the waveform's spectrogram. The first frames have
    to be at least ``kernel_size[1] // 2 + 1`` for the reflect padding."""

    def __init__(self, sample_rate, num_mels, hop_length, n_fft, channels, mult_channels, kernel_size,
                 num_blocks, prior_noise=0.01, prior_max_frequency=None):
        super().__init__()
        self.hop_length = int(hop_length)
        self.prior_noise = float(prior_noise)
        self.prior = PCPHSource(sample_rate, random_start_phase=True, max_frequency=prior_max_frequency)
        self.stft = STFT(n_fft, hop_length)
        n_bins = n_fft // 2 + 1
        self.prior_proj = nn.Conv1d(n_bins, n_bins, 7, padding=3, padding_mode="reflect")
        self.cond_proj = nn.Conv1d(num_mels, n_bins, 7, padding=3, padding_mode="reflect")
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock2d(channels, mult_channels, kernel_size, 1 / num_blocks) for _ in range(num_blocks)
        )
        self.output_norm = LayerNorm2d(channels)
        self.output_proj = nn.Conv2d(channels, 2, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def compile_blocks(self) -> None:
        """torch.compile the ConvNeXt blocks in place, which keeps their keys:
        about a quarter off a training step."""
        for block in self.blocks:
            block.compile()

    def forward(self, mel, f0):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        # The prior and both transforms in FP32, whatever the autocast.
        with torch.no_grad(), torch.autocast(mel.device.type, enabled=False):
            f0 = expand_f0(f0.float(), mel.shape[-1] * self.hop_length)
            pulses, _ = self.prior.components(f0[:, 0])
            real, imag = self.stft(pulses + self.prior_noise * torch.randn_like(pulses))
        x = torch.stack(
            [real, imag, self.prior_proj(real), self.prior_proj(imag), self.cond_proj(mel)], dim=1
        )
        x = self.input_norm(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(self.output_norm(x))
        with torch.autocast(mel.device.type, enabled=False):
            return self.stft.inverse(x[:, 0], x[:, 1])


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN's period discriminator, as Wavehax sizes it."""

    def __init__(self, period, channels=32, kernel_sizes=(5, 3), downsample_scales=(3, 3, 3, 3, 1),
                 max_channels=1024):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, channels
        for scale in downsample_scales:
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_sizes[0], 1), (scale, 1),
                padding=((kernel_sizes[0] - 1) // 2, 0),
            )))
            in_channels, out_channels = out_channels, min(out_channels * 4, max_channels)
        self.output_conv = weight_norm(nn.Conv2d(
            out_channels, 1, (kernel_sizes[1], 1), padding=((kernel_sizes[1] - 1) // 2, 0)
        ))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return torch.flatten(self.output_conv(x), 1, -1), fmap


class SpectralDiscriminator(nn.Module):
    """UnivNet's spectrogram discriminator, as Wavehax sizes it."""

    KERNEL_SIZES = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3), (3, 3))
    STRIDES = ((2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1))

    def __init__(self, fft_size, hop_size, win_length, channels=32):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.input_conv = weight_norm(nn.Conv2d(1, channels, 1))
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(channels, channels, kernel, padding=(kernel[0] // 2, kernel[1] // 2), stride=stride))
            for kernel, stride in zip(self.KERNEL_SIZES, self.STRIDES)
        )
        self.output_conv = weight_norm(nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        x = torch.stft(
            x.squeeze(1).float(), self.fft_size, self.hop_size, self.win_length, self.window,
            center=True, return_complex=True,
        ).abs().unsqueeze(1)
        x = F.leaky_relu(self.input_conv(x), 0.1)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return self.output_conv(x), fmap


class Heads(nn.Module):
    """A family of discriminators: ``(outputs, feature maps)``, one per head."""

    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y):
        outputs, fmaps = [], []
        for discriminator in self.discriminators:
            output, fmap = discriminator(y)
            outputs.append(output)
            fmaps.append(fmap)
        return outputs, fmaps


def build_families(settings: dict) -> dict:
    """Wavehax's discriminator: the period and the multi-resolution families."""
    return {
        "mpd": Heads(PeriodDiscriminator(period) for period in settings["periods"]),
        "mrd": Heads(
            SpectralDiscriminator(*sizes)
            for sizes in zip(settings["mrd_fft_sizes"], settings["mrd_hop_sizes"], settings["mrd_win_lengths"])
        ),
    }
