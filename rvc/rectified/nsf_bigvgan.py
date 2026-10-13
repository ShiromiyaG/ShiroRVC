"""NSF-BigVGAN: BigVGAN's anti-aliased SnakeBeta blocks over HiFi-GAN's
transposed upsamplers, with the NSF sine excitation added at every stage.

Generator, source, AMP block and anti-aliasing filters ported from
https://github.com/PlayVoice/NSF-BigVGAN (MIT, see THIRD_PARTY_NOTICES); the
filtered SnakeBeta runs as one Triton op on CUDA (``snake_alias_triton.py``).
Trained by ``train_vocoder.py`` with the recipe in
``rvc/configs/rectified/vocoders/nsf-bigvgan.json``.
"""

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rvc.lib.algorithm.generators.pcph_bigvgan import NoiseBranch, SnakeBeta

try:
    from rvc.lib.algorithm.upsample_triton import PolyphaseDecimate
    from rvc.rectified.snake_alias_triton import FusedResidualSnakeAlias, FusedSnakeAlias
except ImportError:
    PolyphaseDecimate = FusedSnakeAlias = FusedResidualSnakeAlias = None

#: ``architecture`` of an NSF-BigVGAN rectified vocoder export.
ARCHITECTURE = "nsf_bigvgan"
#: The fixed merge of the sine and its ten overtones into one excitation.
MERGE_WEIGHT = (0.2942, -0.2243, 0.0033, -0.0056, -0.0020, -0.0046, 0.0221, -0.0083, -0.0241, -0.0036, -0.0581)
MERGE_BIAS = 0.0008
#: Taps of the 2x filters around each SnakeBeta, and their paddings: the
#: upsampler's on each side of its input, the lowpass's on the left.
ALIAS_KERNEL = 48
#: Their cutoff and half width: flat to 0.8 and -49 dB from 1.1 times the stage's Nyquist,
#: where the original's 12 taps (0.25, 0.3) fold the overtones back at -9 dB.
ALIAS_CUTOFF = 0.23
ALIAS_HALF_WIDTH = 0.05
PHASE_PAD = (ALIAS_KERNEL // 4, ALIAS_KERNEL // 4)
DOWN_PAD = ALIAS_KERNEL // 2 - 1
#: The shorter one of the stages under ``LOW_RATE`` Hz, where the long filter
#: would ring for tens of milliseconds.
LOW_ALIAS = (24, 0.22, 0.12)
LOW_RATE = 4000
#: Zero crossings of the resampling filters on each side, as julius' default.
SINC_ZEROS = 8
#: The lowpass after the upsamplers of the stages from ``LOW_RATE`` up: taps
#: per side and cutoff for a 1x stride, and the Kaiser beta. The stopband
#: starts at the input's Nyquist, where a filter centred there lets the images
#: just above it through at -6 dB.
IMAGE_HALF_TAPS = 20
IMAGE_CUTOFF = 0.46
IMAGE_BETA = 5.0
#: Cutoffs in Hz of the output's high-pass, on voiced and on unvoiced steps:
#: the stream carries the mel's frame jitter and the unvoiced source noise to
#: the output as a 30-80 Hz rumble.
RUMBLE_CUTOFFS = (60.0, 90.0)
#: Seconds the excitation takes to cross a voicing edge with ``source_smooth``.
EDGE_FADE = 0.002


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def kaiser_sinc_filter(cutoff, half_width, kernel_size):
    """alias-free-torch's Kaiser-windowed sinc lowpass, [kernel_size] (even);
    ``cutoff`` and ``half_width`` as fractions of the sample rate."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return kernel / kernel.sum()


def sinc_lowpass(cutoff: float) -> torch.Tensor:
    """julius' Hann-windowed sinc lowpass, [taps]; ``cutoff`` as a fraction of
    the sample rate."""
    half_size = int(SINC_ZEROS / cutoff / 2)
    time = torch.arange(-half_size, half_size + 1, dtype=torch.float64)
    kernel = 2 * cutoff * torch.hann_window(2 * half_size + 1, periodic=False, dtype=torch.float64)
    kernel = kernel * torch.sinc(2 * cutoff * time)
    return (kernel / kernel.sum()).float()


def image_lowpass(stride: int) -> torch.Tensor:
    """Kaiser-windowed sinc lowpass after a ``stride`` times upsampler, [taps]:
    flat to 0.85 and -52 dB from 1.0 times the Nyquist of the upsampler's input."""
    half_size = IMAGE_HALF_TAPS * stride
    cutoff = IMAGE_CUTOFF / stride
    time = torch.arange(-half_size, half_size + 1, dtype=torch.float64)
    window = torch.kaiser_window(2 * half_size + 1, periodic=False, beta=IMAGE_BETA, dtype=torch.float64)
    kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (kernel / kernel.sum()).float()


def rumble_kernels(sample_rate) -> torch.Tensor:
    """The lowpasses at ``RUMBLE_CUTOFFS``, centred, [2, 1, taps]."""
    kernels = [sinc_lowpass(cutoff / sample_rate) for cutoff in RUMBLE_CUTOFFS]
    taps = max(kernel.shape[0] for kernel in kernels)
    return torch.stack([F.pad(kernel, ((taps - kernel.shape[0]) // 2,) * 2) for kernel in kernels])[:, None]


def lowpass(x: torch.Tensor, kernel: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """``x`` [B, C, T] through ``kernel`` behind replicate padding, per channel."""
    batch, channels, length = x.shape
    half_size = kernel.shape[-1] // 2
    if PolyphaseDecimate is not None and x.is_cuda:
        padded = F.pad(x, (half_size, half_size), mode="replicate")
        # The outputs past the last whole window read zeros beyond the padding.
        return PolyphaseDecimate.apply(padded, kernel, stride, 0)[..., : -(-length // stride)]
    x = F.pad(x.reshape(batch * channels, 1, -1), (half_size, half_size), mode="replicate")
    return F.conv1d(x, kernel[None, None], stride=stride).reshape(batch, channels, -1)


def lowpass_fp32(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``lowpass`` in FP32 whatever the autocast, back in ``x``'s dtype."""
    with torch.autocast(x.device.type, enabled=False):
        return lowpass(x.float(), kernel).to(x.dtype)


class SnakeAlias(nn.Module):
    """SnakeBeta at twice the rate, between 2x Kaiser filters;
    one Triton op on CUDA."""

    def __init__(self, channels, kernel_size=ALIAS_KERNEL, cutoff=ALIAS_CUTOFF, half_width=ALIAS_HALF_WIDTH):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.kernel_size = kernel_size
        # The upsampler's padding on each side of its input, the lowpass's on the left.
        self.phase_pad = (kernel_size // 4, kernel_size // 4)
        self.down_pad = kernel_size // 2 - 1
        kernel = kaiser_sinc_filter(cutoff, half_width, kernel_size)
        self.register_buffer("filter", kernel, persistent=False)
        # The upsampler as the fused op takes it: per output phase, the weight
        # of each input sample from phase_pad[0] before to phase_pad[1] after.
        reach = self.phase_pad[0]
        phase_weight = torch.zeros(2, 2 * reach + 1)
        for phase in range(2):
            for offset in range(-reach, reach + 1):
                tap = phase + (kernel_size - 2) // 2 - 2 * offset
                if 0 <= tap < kernel_size:
                    phase_weight[phase, offset + reach] = 2 * kernel[tap]
        self.register_buffer("phase_weight", phase_weight, persistent=False)

    def forward(self, x):
        if FusedSnakeAlias is not None and x.is_cuda:
            return FusedSnakeAlias.apply(x, *self._fused_args(x))
        channels = x.shape[1]
        kernel = self.filter.to(x.dtype)[None, None].expand(channels, -1, -1)
        pad = self.kernel_size // 2 - 1
        crop = 2 * pad + (self.kernel_size - 2) // 2
        x = F.pad(x, (pad, pad), mode="replicate")
        x = 2 * F.conv_transpose1d(x, kernel, stride=2, groups=channels)[..., crop:-crop]
        x = F.pad(self.act(x), (self.down_pad, self.kernel_size // 2), mode="replicate")
        return F.conv1d(x, kernel, stride=2, groups=channels)

    def forward_residual(self, x, r):
        """``(self(x + r), x + r)`` with the add inside the kernel; CUDA only."""
        return FusedResidualSnakeAlias.apply(x, r, *self._fused_args(x))

    def _fused_args(self, x):
        # Written in the dtype the next conv runs in rather than cast to it.
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        else:
            dtype = x.dtype
        return self.act.alpha, self.act.beta, self.phase_weight, self.filter, self.phase_pad, self.down_pad, dtype


class SineGen(nn.Module):
    """The sine at f0 and its overtones, with noise in place of the unvoiced
    steps: f0 [B, T, 1] at the sample rate -> [B, T, harmonic_num + 1].

    ``noise_std`` None leaves the noise out; ``random_phase`` off starts every
    overtone at phase 0."""

    def __init__(self, sample_rate, harmonic_num, sine_amp=0.1, noise_std=0.003, random_phase=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.random_phase = random_phase

    @torch.no_grad()
    def forward(self, f0, voiced=None):
        """``voiced`` [B, T, 1] in 0..1 weighs the sines against the unvoiced
        noise; ``f0 > 0`` without it."""
        harmonics = f0 * torch.arange(1, self.dim + 1, device=f0.device)
        rad = (harmonics / self.sample_rate) % 1
        if self.random_phase:
            # Random initial phase, but for the fundamental.
            start = torch.rand(f0.shape[0], self.dim, device=f0.device)
            start[:, 0] = 0
            rad[:, 0, :] = rad[:, 0, :] + start
        # -1 at every step where the running phase wraps, against cumsum overflow.
        wrapped = torch.cumsum(rad, 1) % 1
        shift = torch.zeros_like(rad)
        shift[:, 1:, :] = ((wrapped[:, 1:, :] - wrapped[:, :-1, :]) < 0) * -1.0
        sines = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * np.pi) * self.sine_amp
        uv = (f0 > 0).to(sines.dtype) if voiced is None else voiced
        # An overtone past Nyquist would fold back below it.
        sines = sines * uv * (harmonics < self.sample_rate / 2)
        if self.noise_std is None:
            return sines
        return sines + (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sines)


class SourceModuleHnNSF(nn.Module):
    """``SineGen``'s harmonics merged to one excitation by fixed weights,
    [B, T, 1]. With ``noise_channel`` the sines carry no noise and a second
    channel is noise alone, at the levels it would have had in the sines: at
    the unvoiced level on voiced steps the stages build the band above the
    overtones out of it."""

    def __init__(self, sample_rate, sine_amp=0.1, noise_std=0.003, random_phase=True, noise_channel=False):
        super().__init__()
        self.noise_amp = sine_amp / 3 if noise_channel else None
        self.noise_std = noise_std
        self.l_sin_gen = SineGen(
            sample_rate, len(MERGE_WEIGHT) - 1, sine_amp, None if noise_channel else noise_std, random_phase
        )
        self.register_buffer("merge_w", torch.tensor([MERGE_WEIGHT]))
        self.register_buffer("merge_b", torch.tensor([MERGE_BIAS]))

    def forward(self, f0, voiced=None):
        source = torch.tanh(F.linear(self.l_sin_gen(f0, voiced), self.merge_w) + self.merge_b)
        if self.noise_amp is None:
            return source
        uv = (f0 > 0).to(source.dtype) if voiced is None else voiced
        level = uv * self.noise_std + (1 - uv) * self.noise_amp
        return torch.cat((source, level * torch.randn_like(source)), dim=-1)


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), alias=()):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
             for _ in dilation]
        )
        self.activations = nn.ModuleList(SnakeAlias(channels, *alias) for _ in range(2 * len(dilation)))

    def forward(self, x):
        units = list(zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]))
        if FusedResidualSnakeAlias is None or not x.is_cuda or x.dtype != torch.float32:
            for c1, c2, a1, a2 in units:
                x = c2(a2(c1(a1(x)))) + x
            return x
        # Each residual add is done inside the next activation's kernel.
        branch = None
        for c1, c2, a1, a2 in units:
            if branch is None:
                hidden = a1(x)
            else:
                hidden, x = a1.forward_residual(x, branch)
            branch = c2(a2(c1(hidden)))
        return branch + x


class NSFBigVGAN(nn.Module):
    """Log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop].

    ``source_noise_std`` is the noise added to the excitation's voiced steps,
    against sines of amplitude 0.1. ``source_noise_channel`` gives the stages
    the noise as a channel of its own beside noiseless sines.
    ``noise_branch_bands`` adds PCPH-BigVGAN's ``NoiseBranch``: noise shaped by
    that many band gains read from the mel and f0, added at the output, its
    excitation gains shaping the noise channel. ``source_smooth`` glides the
    pitch between frames and fades the excitation across a voicing edge;
    ``source_random_phase`` off starts the overtones at phase 0."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilation_sizes,
                 source_noise_std=0.003, source_noise_channel=False, source_smooth=False,
                 source_random_phase=True, noise_branch_bands=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = int(np.prod(upsample_rates))
        self.source_noise_channel = bool(source_noise_channel)
        self.source_smooth = bool(source_smooth)
        self.edge_samples = min(self.upp, max(1, round(EDGE_FADE * sample_rate)))
        self.m_source = SourceModuleHnNSF(
            sample_rate, noise_std=source_noise_std, random_phase=bool(source_random_phase),
            noise_channel=self.source_noise_channel,
        )
        source_channels = 2 if self.source_noise_channel else 1
        self.noise_branch = None
        if int(noise_branch_bands) > 0:
            self.noise_branch = NoiseBranch(num_mels, noise_branch_bands, sample_rate, self.upp, f0_input=True)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            # The transposed conv leaves images of the stream around the
            # multiples of its input rate: cut at the Nyquist it came from.
            rate = sample_rate / np.prod(upsample_rates[i + 1:])
            image_kernel = sinc_lowpass(0.5 / u) if rate < LOW_RATE else image_lowpass(u)
            self.register_buffer(f"image_kernel_{i}", image_kernel, persistent=False)
            if i + 1 < len(upsample_rates):
                stride = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(
                    nn.Conv1d(source_channels, ch, kernel_size=stride * 2, stride=stride, padding=stride // 2)
                )
                # The strided conv would fold the excitation past the stage's
                # Nyquist: low-passed first, the stopband starting there.
                self.register_buffer(f"source_kernel_{i}", sinc_lowpass(0.4 / stride), persistent=False)
            else:
                self.noise_convs.append(nn.Conv1d(source_channels, ch, kernel_size=1))
            alias = LOW_ALIAS if rate < LOW_RATE else ()
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, kernel, dilation, alias))
        self.activation_post = SnakeAlias(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.register_buffer("rumble_kernels", rumble_kernels(sample_rate), persistent=False)
        # Set by ``apply_precision_policy`` under AMP: the excitation, the mel
        # conv, the upsamplers, the residual stream and the output layer stay
        # in FP32.
        self.fp32_residuals = False

    def _fp32_region(self, x):
        """A context with autocast off, when ``fp32_residuals`` is set."""
        if not self.fp32_residuals:
            return nullcontext()
        return torch.autocast(x.device.type, enabled=False)

    def _fp32(self, x):
        return x.float() if self.fp32_residuals else x

    def _highpass(self, y, f0):
        """``y`` [B, 1, T] without its band under ``RUMBLE_CUTOFFS``, by the
        voicing of ``f0`` [B, T, 1]."""
        taps = self.rumble_kernels.shape[-1]
        with torch.autocast(y.device.type, enabled=False):
            wave = y.float()
            # As a product of spectra: cuDNN stalls on a kernel this long.
            padded = F.pad(wave, (taps // 2, taps // 2), mode="replicate")
            size = padded.shape[-1]
            spectrum = torch.fft.rfft(padded, n=size) * torch.fft.rfft(self.rumble_kernels[:, 0], n=size)
            low = torch.fft.irfft(spectrum, n=size)[..., taps - 1:]
            # The cutoff slides over a frame on each side of a voicing edge.
            unvoiced = F.pad((f0.transpose(1, 2) == 0).float(), (self.upp, self.upp), mode="replicate")
            unvoiced = F.avg_pool1d(unvoiced, 2 * self.upp + 1, 1)
            return (wave - torch.lerp(low[:, :1], low[:, 1:], unvoiced)).to(y.dtype)

    def _glide(self, frames):
        """f0 [B, 1, F] -> Hz and voicing at the sample rate, [B, T, 1]: the
        pitch linear between voiced frames and held across a voicing edge,
        which the voicing crosses in ``EDGE_FADE`` seconds."""
        size = frames.shape[-1] * self.upp
        weight = F.interpolate((frames > 0).float(), size=size, mode="linear")
        # The voiced frames' mean: an unvoiced neighbour's 0 Hz takes no part.
        pitch = F.interpolate(frames, size=size, mode="linear") / weight.clamp_min(0.25)
        voiced = ((weight - 0.5) * (self.upp / self.edge_samples) + 0.5).clamp(0.0, 1.0)
        return pitch.transpose(1, 2), voiced.transpose(1, 2)

    def forward(self, mel, f0):
        frames = f0.float()[:, None]
        f0 = F.interpolate(frames, scale_factor=self.upp).transpose(1, 2)
        pitch, voiced = self._glide(frames) if self.source_smooth else (f0, None)
        with self._fp32_region(mel):
            noise_gains = excitation_gains = None
            if self.noise_branch is not None:
                noise_gains, excitation_gains = self.noise_branch.gains(self._fp32(mel), frames)
            source = self.m_source(pitch, voiced).transpose(1, 2)
            if excitation_gains is not None and self.source_noise_channel:
                noise = self.noise_branch.shape(source[:, 1], excitation_gains).to(source.dtype)
                source = torch.stack((source[:, 0], noise), dim=1)
            sources = [
                lowpass_fp32(source, getattr(self, f"source_kernel_{i}")) if i + 1 < len(self.ups) else source
                for i in range(len(self.ups))
            ]
            x = F.mish(self.conv_pre(self._fp32(mel)))
        for i, up in enumerate(self.ups):
            # The upsampler and the excitation in FP32 keep the stage's
            # residual stream there; the convs in the blocks stay under autocast.
            with self._fp32_region(x):
                x = lowpass_fp32(up(self._fp32(x)), getattr(self, f"image_kernel_{i}"))
                x = x + self.noise_convs[i](sources[i])
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        # conv_post's few weights take the whole waveform's gradient, which
        # overflows FP16.
        with self._fp32_region(x):
            y = self.conv_post(self.activation_post(self._fp32(x)))
            if noise_gains is not None:
                noise = torch.randn(y.shape[0], y.shape[-1], device=y.device)
                y = y + self.noise_branch.shape(noise, noise_gains).unsqueeze(1).to(y.dtype)
            y = torch.tanh(y)
        return self._highpass(y, f0)
