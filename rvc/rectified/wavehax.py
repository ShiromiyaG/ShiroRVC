"""Wavehax (Yoneyama et al., 2024): a vocoder that estimates the complex
spectrogram with 2D convs over the STFT of a harmonic prior, then inverts it.
It has no upsampling layers and no activation at the audio rate, so nothing
of its own aliases.

Generator, discriminators and the closed-form prior ported from
https://github.com/chomeyama/wavehax (MIT, see THIRD_PARTY_NOTICES); the
default prior is this repo's ``PCPHSource``. Trained by ``train_vocoder.py``
with the recipes in ``rvc/configs/rectified/vocoders``: ``wavehax.json`` is
the original's fullband layout, ``wavehax-v2.json`` its lighter v2 and
``ms-wavehax.json`` Multi-Scale Wavehax (Yoneyama et al., 2025) at 44.1 kHz.
"""

import math

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
#: The run that trains the v2 layout; its export is a ``wavehax`` one.
V2 = "wavehax-v2"
#: ``architecture`` of a Multi-Scale Wavehax export, and of its run.
MS = "ms-wavehax"
#: The "light" discriminator, in place of the recipe's own: three compressed,
#: frequency-aware spectrograms sized for 44.1 kHz, and the official fullband
#: periods at full width with their two widest convs made separable.
#: A recipe's ``light_discriminator`` overrides any of it.
LIGHT_DISCRIMINATOR = {
    "periods": [2, 3, 5, 7, 11, 13, 17],
    "mpd_max_channels": 1024,
    "mpd_separable": True,
    "mrd_fft_sizes": [4096, 2048, 1024],
    "mrd_hop_sizes": [512, 256, 128],
    "mrd_win_lengths": [4096, 2048, 1024],
    "mrd_compression": 0.3,
    "mrd_position": True,
}
#: Floor under a compressed magnitude, as a share of a full-scale sine's:
#: bounds the slope at silence.
COMPRESSION_FLOOR = 1e-4
#: Width from which a separable period head splits its convs.
SEPARABLE_FROM = 512
#: The harmonic priors: this repo's ``PCPHSource``, or the original's.
PRIORS = ("pcph", "pcph_closed_form")
#: Power of the closed-form prior's harmonics.
CLOSED_FORM_POWER = 0.1
#: Multi-Scale Wavehax's norms: the paper's batch norm, or a ``LayerNorm2d``
#: over each frame (the official config) or the whole spectrogram.
NORMS = ("batch", "frame", "layer")
#: Cap on the magnitude a log-magnitude output can ask for, as Vocos has it.
MAX_MAGNITUDE = 1e2


def pcph_closed_form(f0, hop_length, sample_rate, noise_std=0.01, max_frequency=None):
    """Wavehax's ``generate_pcph_closed_form``: f0 [B, 1, T] in Hz -> the
    prior, [B, T * hop_length]. Every harmonic under ``max_frequency`` (Nyquist
    when None), summed by the Dirichlet kernel's closed form at constant
    power, plus white noise. As there, f0 is interpolated linearly and the
    starting phase is random on every call."""
    f0 = F.interpolate(f0.float(), scale_factor=hop_length, mode="linear", align_corners=False)
    increment = (f0 / sample_rate).double()
    increment[:, :, :1] += torch.rand(f0.shape[0], 1, 1, device=f0.device, dtype=torch.float64)
    phase = torch.fmod(torch.cumsum(increment, dim=2) * 2.0 * math.pi, 2.0 * math.pi).float()
    count = torch.floor((max_frequency or sample_rate / 2.0) / f0.clamp_min(1e-5))
    # sum_k sin(k x) = (cos(x / 2) - cos((N + 1 / 2) x)) / (2 sin(x / 2)), and 0 where that is 0 / 0.
    denominator = 2.0 * torch.sin(phase / 2.0)
    regular = denominator.abs() > 1e-6
    harmonics = (torch.cos(phase / 2.0) - torch.cos((count + 0.5) * phase)) / torch.where(regular, denominator, 1.0)
    amplitude = CLOSED_FORM_POWER * torch.sqrt(2.0 / count.clamp_min(1.0))
    harmonics = harmonics * amplitude * regular * (f0 > 0)
    return (harmonics + noise_std * torch.randn_like(harmonics))[:, 0]


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
    """Normalises over channels, bins and frames together, or each frame on
    its own when ``framewise``, with a per-channel affine. In FP32: the
    statistics run over the whole spectrogram."""

    def __init__(self, channels: int, eps: float = 1e-6, framewise: bool = False):
        super().__init__()
        self.eps = eps
        self.dims = (1, 2) if framewise else (1, 2, 3)
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.float()
        var, mean = torch.var_mean(x, dim=self.dims, keepdim=True, correction=0)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return torch.addcmul(self.beta.view(1, -1, 1, 1), x, self.gamma.view(1, -1, 1, 1))


class BatchNorm2d(nn.BatchNorm2d):
    """In FP32, as ``LayerNorm2d``."""

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())


def build_norm(channels: int, norm: str) -> nn.Module:
    """One of ``NORMS`` over ``channels``."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, not {norm!r}.")
    if norm == "batch":
        return BatchNorm2d(channels, eps=1e-6)
    return LayerNorm2d(channels, framewise=norm == "frame")


class ConvNeXtBlock2d(nn.Module):
    def __init__(self, channels, mult_channels, kernel_size, layer_scale, norm="layer"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.dwconv = nn.Conv2d(
            channels, channels, tuple(kernel_size), padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=channels, bias=False, padding_mode="reflect",
        )
        self.norm = build_norm(channels, norm)
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


class MultiStream(nn.Module):
    """Multi-stream HiFi-GAN's learned decomposition, as Multi-Scale Wavehax
    uses it: ``splits`` sub-signals at 1 / ``splits`` of the rate, and back."""

    def __init__(self, splits: int, taps: int = 62):
        super().__init__()
        self.splits = int(splits)
        self.conv_analysis = nn.Conv1d(
            1, self.splits, taps + 1, padding=taps // 2, padding_mode="reflect", bias=False
        )
        self.conv_synthesis = nn.Conv1d(
            self.splits, 1, taps + 1, padding=taps // 2, padding_mode="reflect", bias=False
        )

    def analysis(self, x):
        """(batch, samples) -> (batch, splits, samples // splits)."""
        return self.conv_analysis(x.unsqueeze(1))[..., :: self.splits]

    def synthesis(self, x):
        """(batch, splits, samples // splits) -> (batch, 1, samples)."""
        stuffed = x.new_zeros(*x.shape[:-1], x.shape[-1] * self.splits)
        stuffed[..., :: self.splits] = x * self.splits
        return self.conv_synthesis(stuffed)


class _Generator(nn.Module):
    """What the Wavehax generators share: the prior, the init and the
    compiled blocks."""

    def __init__(self, sample_rate, hop_length, prior_noise, prior_max_frequency, prior_type):
        super().__init__()
        if prior_type not in PRIORS:
            raise ValueError(f"prior_type must be one of {PRIORS}, not {prior_type!r}.")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.prior_noise = float(prior_noise)
        self.prior_max_frequency = prior_max_frequency
        self.prior = None
        if prior_type == "pcph":
            self.prior = PCPHSource(sample_rate, random_start_phase=True, max_frequency=prior_max_frequency)
        #: Set by the trainer from the recipe's ``generator_fp32``: the whole
        #: generator out of autocast, which the discriminator stays under.
        self.fp32 = False

    def forward(self, mel, f0):
        if not self.fp32:
            return self._forward(mel, f0)
        with torch.autocast(mel.device.type, enabled=False):
            return self._forward(mel.float(), f0)

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

    def _prior(self, f0):
        """The prior waveform, [B, T * hop_length], from f0 [B, 1, T]."""
        if self.prior is None:
            return pcph_closed_form(
                f0, self.hop_length, self.sample_rate, self.prior_noise, self.prior_max_frequency
            )
        pulses, _ = self.prior.components(expand_f0(f0.float(), f0.shape[-1] * self.hop_length)[:, 0])
        return pulses + self.prior_noise * torch.randn_like(pulses)


class WavehaxGenerator(_Generator):
    """Normalised mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop_length].

    The prior is the PCPH pulse train plus white noise at ``prior_noise``,
    made by ``prior_type``, one of ``PRIORS``; its
    spectrogram, a projection of it and the projected mel are the input
    channels of a ConvNeXt stack over (bins, frames), which outputs the real
    and imaginary parts of the waveform's spectrogram; with
    ``use_logmag_phase``, the prior's go in and the waveform's come out as log
    magnitude and phase instead. The first frames have
    to be at least ``kernel_size[1] // 2 + 1`` for the reflect padding."""

    def __init__(self, sample_rate, num_mels, hop_length, n_fft, channels, mult_channels, kernel_size,
                 num_blocks, prior_noise=0.01, prior_max_frequency=None, prior_type="pcph",
                 use_logmag_phase=False):
        super().__init__(sample_rate, hop_length, prior_noise, prior_max_frequency, prior_type)
        self.stft = STFT(n_fft, hop_length)
        self.use_logmag_phase = bool(use_logmag_phase)
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

    def _forward(self, mel, f0):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        # The prior and both transforms in FP32, whatever the autocast.
        with torch.no_grad(), torch.autocast(mel.device.type, enabled=False):
            real, imag = self.stft(self._prior(f0))
            if self.use_logmag_phase:
                magnitude = torch.sqrt((real.square() + imag.square()).clamp_min(1e-10))
                real, imag = torch.log(magnitude), torch.atan2(imag, real)
        # Every projection in FP32: the signal passes through them in series,
        # and the few weights of the 1x1 ones take the whole spectrogram's
        # gradient, which overflows FP16.
        with torch.autocast(mel.device.type, enabled=False):
            x = torch.stack(
                [real, imag, self.prior_proj(real), self.prior_proj(imag), self.cond_proj(mel.float())], dim=1
            )
            x = self.input_proj(x)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        with torch.autocast(mel.device.type, enabled=False):
            x = self.output_proj(x.float())
            real, imag = x[:, 0], x[:, 1]
            if self.use_logmag_phase:
                magnitude = torch.exp(real).clamp(max=MAX_MAGNITUDE)
                real, imag = magnitude * torch.cos(imag), magnitude * torch.sin(imag)
            return self.stft.inverse(real, imag)


class MultiScaleWavehaxGenerator(_Generator):
    """Multi-Scale Wavehax, with ``WavehaxGenerator``'s inputs and output.

    The prior is split into ``num_splits`` sub-signals at 1 / ``num_splits``
    of the rate, and the ConvNeXt stack estimates the spectrogram of each from
    theirs and the projected mel: that many times fewer bins per frame, which
    pays for a wider stack. ``n_fft`` is the sub-signals', whose hop is
    ``hop_length // num_splits``; ``norm`` is one of ``NORMS``."""

    def __init__(self, sample_rate, num_mels, hop_length, n_fft, channels, mult_channels, kernel_size,
                 num_blocks, num_splits=4, norm="batch", prior_noise=0.01,
                 prior_max_frequency=None, prior_type="pcph"):
        super().__init__(sample_rate, hop_length, prior_noise, prior_max_frequency, prior_type)
        if hop_length % num_splits or n_fft % (hop_length // num_splits):
            raise ValueError(
                f"hop_length ({hop_length}) must be a multiple of num_splits ({num_splits}), and n_fft "
                f"({n_fft}) of the sub-signals' hop."
            )
        self.num_splits = int(num_splits)
        self.n_bins = n_fft // 2 + 1
        self.decomposer = MultiStream(num_splits)
        self.stft = STFT(n_fft, hop_length // num_splits)
        self.cond_proj = nn.Conv1d(num_mels, num_splits * self.n_bins, 7, padding=3, padding_mode="reflect")
        self.input_proj = nn.Conv2d(3 * num_splits, channels, 1, bias=False)
        self.input_norm = build_norm(channels, norm)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock2d(channels, mult_channels, kernel_size, 1 / num_blocks, norm)
            for _ in range(num_blocks)
        )
        self.output_norm = build_norm(channels, norm)
        self.output_proj = nn.Conv2d(channels, 2 * num_splits, 1)
        self.apply(self._init_weights)

    def _forward(self, mel, f0):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        batch, splits = mel.shape[0], self.num_splits
        # The prior, its decomposition and the transforms in FP32, whatever the autocast.
        with torch.autocast(mel.device.type, enabled=False):
            with torch.no_grad():
                prior = self._prior(f0)
            real, imag = self.stft(self.decomposer.analysis(prior).flatten(0, 1))
        frames = real.shape[-1]
        # Each sub-signal's real then imaginary part, as the output has them.
        specs = torch.stack([real, imag], dim=1).view(batch, 2 * splits, self.n_bins, frames)
        # The projections in FP32, as in ``WavehaxGenerator``.
        with torch.autocast(mel.device.type, enabled=False):
            cond = self.cond_proj(mel.float()).view(batch, splits, self.n_bins, frames)
            x = self.input_proj(torch.cat([cond, specs], dim=1))
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        with torch.autocast(mel.device.type, enabled=False):
            x = self.output_proj(x.float()).reshape(batch * splits, 2, self.n_bins, frames)
            streams = self.stft.inverse(x[:, 0], x[:, 1]).view(batch, splits, -1)
            return self.decomposer.synthesis(streams)


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN's period discriminator, as Wavehax sizes it.

    ``separable`` splits each conv that reads ``SEPARABLE_FROM`` channels or
    more into a depthwise one along time and a pointwise one, which keep the
    widths at about a fifth of those convs' cost."""

    def __init__(self, period, channels=32, kernel_sizes=(5, 3), downsample_scales=(3, 3, 3, 3, 1),
                 max_channels=1024, separable=False):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, channels
        for scale in downsample_scales:
            taps = dict(kernel_size=(kernel_sizes[0], 1), stride=(scale, 1), padding=((kernel_sizes[0] - 1) // 2, 0))
            if separable and in_channels >= SEPARABLE_FROM:
                conv = nn.Sequential(
                    weight_norm(nn.Conv2d(in_channels, in_channels, groups=in_channels, **taps)),
                    weight_norm(nn.Conv2d(in_channels, out_channels, 1)),
                )
            else:
                conv = weight_norm(nn.Conv2d(in_channels, out_channels, **taps))
            self.convs.append(conv)
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
        for index, conv in enumerate(self.convs):
            if index:
                x = conv(x)
            else:
                # On the waveform, out of autocast: BF16 would keep 8 bits of each sample.
                with torch.autocast(x.device.type, enabled=False):
                    x = conv(x.float())
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        return torch.flatten(self.output_conv(x), 1, -1), fmap


class SpectralDiscriminator(nn.Module):
    """UnivNet's spectrogram discriminator, as Wavehax sizes it.

    ``compression`` under 1 raises the magnitude to that power, which brings
    the quiet high band up to the scale of the low one: the kernels are shared
    along frequency and cannot weigh it up themselves. ``position`` adds each
    bin's place in the spectrum as a second channel, for the same reason."""

    KERNEL_SIZES = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3), (3, 3))
    STRIDES = ((2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1))

    def __init__(self, fft_size, hop_size, win_length, channels=32, compression=1.0, position=False):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compression = float(compression)
        self.position = bool(position)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.input_conv = weight_norm(nn.Conv2d(2 if position else 1, channels, 1))
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
        if self.compression != 1.0:
            x = (x / (0.5 * self.win_length) + COMPRESSION_FLOOR) ** self.compression
        if self.position:
            place = torch.linspace(-1.0, 1.0, x.shape[2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
            x = torch.cat((x, place.expand(x.shape[0], -1, -1, x.shape[3])), dim=1)
        # On the magnitude, out of autocast: unnormalised, FP16's range is
        # short for its gradient.
        with torch.autocast(x.device.type, enabled=False):
            x = F.leaky_relu(self.input_conv(x.float()), 0.1)
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
    """Wavehax's discriminator: the period and the multi-resolution families.
    The keys ``LIGHT_DISCRIMINATOR`` adds are optional and default to the
    original's."""
    period = dict(
        max_channels=int(settings.get("mpd_max_channels", 1024)), separable=settings.get("mpd_separable", False)
    )
    spectral = dict(
        compression=settings.get("mrd_compression", 1.0), position=settings.get("mrd_position", False)
    )
    return {
        "mpd": Heads(PeriodDiscriminator(p, **period) for p in settings["periods"]),
        "mrd": Heads(
            SpectralDiscriminator(*sizes, **spectral)
            for sizes in zip(settings["mrd_fft_sizes"], settings["mrd_hop_sizes"], settings["mrd_win_lengths"])
        ),
    }
