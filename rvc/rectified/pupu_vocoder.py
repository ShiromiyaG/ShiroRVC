"""Pupu-Vocoder (Aliasing-Free Neural Audio Synthesis, arXiv 2512.20211):
BigVGAN with the transposed upsamplers replaced by zero-stuffing behind
windowed-sinc filters, and SnakeBeta by its anti-derivative anti-aliased form.

Generator, sine generator, multi-scale mel loss and the multi-band STFT and
sub-band CQT discriminators ported from
https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis (MIT, see
THIRD_PARTY_NOTICES). The original takes the mel alone and leaves its sine
generator unused: here the sine at f0 joins every stage, low-passed down to
the stage's rate so that it brings no aliasing of its own. On CUDA the
activations and the resampling filters run as Triton kernels. Trained by
``train_vocoder.py`` with the recipe in
``rvc/configs/rectified/vocoders/pupu-vocoder.json``.
"""

import math

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.rectified.nsf_bigvgan import SINC_ZEROS, kaiser_sinc_filter, lowpass, sinc_lowpass
from rvc.rectified.wavehax import Heads

try:
    from rvc.lib.algorithm.upsample_triton import PolyphaseUpsample
    from rvc.rectified.adaa_snake_triton import FusedADAASnake, FusedResidualADAASnake
except ImportError:
    PolyphaseUpsample = FusedADAASnake = FusedResidualADAASnake = None

#: The filters around each activation, as the original has them and not
#: NSF-BigVGAN's longer ones: the released weights render noise through those.
ALIAS_KERNEL = 12
ALIAS_CUTOFF = 0.25
ALIAS_HALF_WIDTH = 0.3
PHASE_PAD = (ALIAS_KERNEL // 4, ALIAS_KERNEL // 4)
DOWN_PAD = ALIAS_KERNEL // 2 - 1
#: ``architecture`` of a Pupu-Vocoder rectified vocoder export.
ARCHITECTURE = "pupu_vocoder"
LRELU_SLOPE = 0.1


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def zero_stuff(x: torch.Tensor, rate: int) -> torch.Tensor:
    stuffed = x.new_zeros(x.shape[0], x.shape[1], x.shape[2] * rate)
    stuffed[:, :, ::rate] = x
    return stuffed


class ADAASnakeBeta(nn.Module):
    """SnakeBeta by first-order anti-derivative anti-aliasing: the mean of
    ``x + sin^2(alpha x) / beta`` between each sample and the one before."""

    def __init__(self, channels: int):
        super().__init__()
        # Logs of alpha and beta: zeros start both at 1.
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        previous = F.pad(x[..., :-1], (1, 0))
        x_sum, x_delta = x + previous, x - previous
        return x_sum / 2.0 + (1.0 / (2.0 * (beta + 1e-9))) * (
            1.0 - torch.cos(alpha * x_sum) * torch.sinc(alpha * x_delta / math.pi)
        )


class Activation1d(nn.Module):
    """``ADAASnakeBeta`` at twice the rate, between 2x Kaiser filters. In
    FP32 whatever the autocast: the activation reads sample differences."""

    def __init__(self, channels: int):
        super().__init__()
        self.act = ADAASnakeBeta(channels)
        kernel = kaiser_sinc_filter(ALIAS_CUTOFF, ALIAS_HALF_WIDTH, ALIAS_KERNEL)
        self.register_buffer("filter", kernel, persistent=False)
        # The upsampler as the fused op takes it: per output phase, the weight
        # of each input sample from PHASE_PAD[0] before to PHASE_PAD[1] after.
        reach = PHASE_PAD[0]
        phase_weight = torch.zeros(2, 2 * reach + 1)
        for phase in range(2):
            for offset in range(-reach, reach + 1):
                tap = phase + (ALIAS_KERNEL - 2) // 2 - 2 * offset
                if 0 <= tap < ALIAS_KERNEL:
                    phase_weight[phase, offset + reach] = 2 * kernel[tap]
        self.register_buffer("phase_weight", phase_weight, persistent=False)

    def _fused_args(self, x):
        # Written in the dtype the next conv runs in rather than cast to it.
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        else:
            dtype = x.dtype
        return self.act.alpha, self.act.beta, self.phase_weight, self.filter, PHASE_PAD, DOWN_PAD, dtype

    def forward_residual(self, x, r):
        """``(self(x + r), x + r)`` with the add inside the kernel; CUDA only."""
        return FusedResidualADAASnake.apply(x, r, *self._fused_args(x))

    def forward(self, x):
        if FusedADAASnake is not None and x.is_cuda:
            return FusedADAASnake.apply(x, *self._fused_args(x))
        channels = x.shape[1]
        with torch.autocast(x.device.type, enabled=False):
            kernel = self.filter[None, None].expand(channels, -1, -1)
            pad = ALIAS_KERNEL // 2 - 1
            crop = 2 * pad + (ALIAS_KERNEL - 2) // 2
            y = F.pad(x.float(), (pad, pad), mode="replicate")
            y = 2 * F.conv_transpose1d(y, kernel, stride=2, groups=channels)[..., crop:-crop]
            y = F.pad(self.act(y), (DOWN_PAD, ALIAS_KERNEL // 2), mode="replicate")
            y = F.conv1d(y, kernel, stride=2, groups=channels)
        return y.to(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=_padding(kernel_size, d))
             for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=_padding(kernel_size))
             for _ in dilation]
        )
        self.activations = nn.ModuleList(Activation1d(channels) for _ in range(2 * len(dilation)))

    def forward(self, x):
        units = list(zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]))
        if FusedResidualADAASnake is None or not x.is_cuda or x.dtype != torch.float32:
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


class ResampleUpsampler(nn.Module):
    """A stage's upsampler: the stream zero-stuffed and low-passed, plus the
    mel features zero-stuffed from the frame rate, convolved and high-passed,
    which fill the band the lowpass leaves empty."""

    def __init__(self, initial_channels, rate, in_channels, out_channels):
        super().__init__()
        self.rate = rate
        self.convolution_after = nn.Conv1d(in_channels, out_channels, 1)
        self.convolution_noise = nn.Conv1d(initial_channels, in_channels, 7, padding=3)
        kernel = sinc_lowpass(0.5 / rate)
        self.register_buffer("kernel", kernel, persistent=False)
        # The lowpass of the zero-stuffed stream by phase: output a * rate + p
        # reads input a + t - SINC_ZEROS through tap (t - SINC_ZEROS) * rate + half - p.
        half_size = kernel.shape[0] // 2
        phase_weight = torch.zeros(rate, 2 * SINC_ZEROS + 1)
        for phase in range(rate):
            for t in range(2 * SINC_ZEROS + 1):
                tap = (t - SINC_ZEROS) * rate + half_size - phase
                if 0 <= tap < kernel.shape[0]:
                    phase_weight[phase, t] = kernel[tap]
        self.register_buffer("phase_weight", phase_weight, persistent=False)
        # The replicate padding's share: the first sample fills every padded
        # position, stuffed or not, and each early output reads some of them.
        edge = torch.cat((kernel.new_zeros(1), kernel.cumsum(0)))[:half_size + 1].flip(0)[:half_size]
        self.register_buffer("edge", edge, persistent=False)

    def _upsample(self, x):
        """``lowpass(zero_stuff(x, rate), kernel)``, without the stuffed tensor on CUDA."""
        if PolyphaseUpsample is None or not x.is_cuda:
            return lowpass(zero_stuff(x, self.rate), self.kernel)
        padded = F.pad(x, (SINC_ZEROS, SINC_ZEROS))
        y = PolyphaseUpsample.apply(padded, self.phase_weight, x.shape[-1])
        reach = min(self.edge.shape[0], y.shape[-1])
        y[..., :reach] += x[..., :1] * self.edge[:reach]
        return y

    def forward(self, x, x0, total_rate):
        # ``convolution_noise`` over ``x0`` zero-stuffed by ``total_rate``, as
        # the transposed conv it equals: the stuffed tensor is never built.
        conv = self.convolution_noise
        y0 = F.conv_transpose1d(
            x0, conv.weight.transpose(0, 1).flip(-1), conv.bias,
            stride=total_rate, padding=3, output_padding=total_rate - 1,
        )
        with torch.autocast(x.device.type, enabled=False):
            y0 = y0.float()
            y0 = y0 - lowpass(y0, self.kernel)
            y = self._upsample(x.float())
        return self.convolution_after((y + y0).to(x.dtype))


def sample_f0(f0: torch.Tensor, upp: int):
    """f0 [B, T] -> per sample [B, T * upp], and where it is voiced. A frame's
    value is its centre's, at sample ``i * upp`` as the mel's is: held over
    the frame instead, the sine would trail the audio by half a frame, which a
    vibrato turns into the wrong pitch. Linear between voiced frames, held up
    to an unvoiced one."""
    after = torch.cat((f0[:, 1:], f0[:, -1:]), dim=1)
    left = torch.where(f0 > 0, f0, after)[..., None]
    right = torch.where(after > 0, after, f0)[..., None]
    along = torch.arange(upp, device=f0.device, dtype=f0.dtype) / upp
    voiced = torch.where(along < 0.5, (f0 > 0)[..., None], (after > 0)[..., None]).flatten(1)
    return (left + (right - left) * along).flatten(1) * voiced, voiced


class SineSource(nn.Module):
    """The repository's ``SineGen`` (the sine at f0 and ``harmonic_num``
    overtones, noise where unvoiced) merged to one excitation: f0 [B, T]
    -> [B, 1, T * upp]."""

    def __init__(self, sample_rate, harmonic_num=0, amp=0.1, noise_std=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.amp = amp
        self.noise_std = noise_std
        self.l_linear = nn.Linear(self.dim, 1)

    @torch.no_grad()
    def _sines(self, f0, upp):
        rate, voiced = sample_f0(f0, upp)
        harmonics = rate.unsqueeze(-1) * torch.arange(1, self.dim + 1, device=f0.device)
        # Random initial phase, but for the fundamental.
        start = torch.rand(f0.shape[0], 1, self.dim, device=f0.device)
        start[..., 0] = 0
        phase = torch.cumsum(harmonics.double() / self.sample_rate, dim=1) % 1
        sines = torch.sin((phase.float() + start) * 2 * np.pi) * self.amp
        uv = voiced.unsqueeze(-1).float()
        noise = (uv * self.noise_std + (1 - uv) * self.amp / 3) * torch.randn_like(sines)
        # An overtone past Nyquist would fold back below it.
        return sines * uv * (harmonics < self.sample_rate / 2) + noise

    def forward(self, f0, upp):
        return torch.tanh(self.l_linear(self._sines(f0, upp))).transpose(1, 2)


class PupuVocoder(nn.Module):
    """Log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop]."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 resblock_kernel_sizes, resblock_dilation_sizes, harmonic_num=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = int(np.prod(upsample_rates))
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.m_source = SineSource(sample_rate, harmonic_num)
        self.ups = nn.ModuleList()
        self.source_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        # Each stage's rate over the frame rate, and the output's over the stage's.
        self.total_rates = [int(np.prod(upsample_rates[: i + 1])) for i in range(len(upsample_rates))]
        self.source_rates = [self.upp // rate for rate in self.total_rates]
        ch = upsample_initial_channel
        for i, rate in enumerate(upsample_rates):
            self.ups.append(ResampleUpsampler(upsample_initial_channel, rate, ch, ch // 2))
            ch //= 2
            self.source_convs.append(nn.Conv1d(1, ch, 7, padding=3))
            if self.source_rates[i] > 1:
                self.register_buffer(
                    f"source_kernel_{i}", sinc_lowpass(0.4 / self.source_rates[i]), persistent=False
                )
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, kernel, dilation))
        self.activation_post = Activation1d(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, mel, f0):
        with torch.autocast(mel.device.type, enabled=False):
            source = self.m_source(f0.float(), self.upp)
            sources = [
                lowpass(source, getattr(self, f"source_kernel_{i}"), stride=rate) if rate > 1 else source
                for i, rate in enumerate(self.source_rates)
            ]
        x = x0 = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(x, x0, self.total_rates[i])
            x = x + self.source_convs[i](sources[i].to(x.dtype))
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.clamp(self.conv_post(self.activation_post(x)), min=-1.0, max=1.0)


def weight_normed(generator: PupuVocoder) -> list:
    """The convs the original trains under weight norm."""
    ups = [conv for up in generator.ups for conv in (up.convolution_after, up.convolution_noise)]
    blocks = [conv for block in generator.resblocks for conv in (*block.convs1, *block.convs2)]
    return [generator.conv_pre, generator.conv_post, *ups, *blocks]


class PhaseLockLoss(nn.Module):
    """How far the render's harmonics drift from the f0 sine's phase: 0 when
    each keeps its phase against the sine from one ``window`` to the next, up
    to 2. The mel carries no phase, so a generator that follows it alone slips
    along a sustained note; one that follows the sine cannot.

    ``forward`` takes audio [B, 1, T * hop] and f0 [B, T], and reads the frames
    voiced on both sides of the step. With ``fmax`` a harmonic counts only
    where it is under that frequency, so ``harmonics`` is how many at most:
    the same band is held at any pitch. The first ``low`` harmonics, when
    there are more, are also a term of their own: one among dozens, the
    fundamental weighs too little to stay held."""

    def __init__(self, sample_rate: int, hop: int, harmonics: int = 4, window: int = 2048, fmax: float = 0.0,
                 low: int = 0):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop = hop
        self.fmax = float(fmax)
        self.low = int(low) if 0 < low < harmonics else 0
        self.stride = window // 4
        self.register_buffer("harmonics", torch.arange(1, harmonics + 1, dtype=torch.float32), persistent=False)
        self.register_buffer("window", torch.hann_window(window).view(1, 1, -1) / window, persistent=False)

    def forward(self, audio: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        audio = audio.float().squeeze(1)
        samples = min(audio.shape[-1], f0.shape[-1] * self.hop)
        # The source's own phase.
        rate = sample_f0(f0.float(), self.hop)[0][:, :samples]
        phase = (torch.cumsum(rate.double() / self.sample_rate, dim=-1) % 1).float() * (2 * math.pi)
        angle = phase[:, None, :] * self.harmonics[None, :, None]
        audio = audio[:, None, :samples]
        # Each harmonic brought down to 0 Hz, where the window is its low-pass.
        mixed = torch.stack((audio * angle.cos(), -audio * angle.sin()), dim=2).flatten(0, 2)[:, None, :]
        real, imag = F.conv1d(mixed, self.window, stride=self.stride).view(
            audio.shape[0], len(self.harmonics), 2, -1
        ).unbind(2)
        voiced = -F.max_pool1d(
            -(rate > 0).float()[:, None, :], self.window.shape[-1], self.stride
        )
        weight = (voiced[..., 4:] * voiced[..., :-4]).expand(-1, len(self.harmonics), -1)
        if self.fmax > 0:
            pitch = F.avg_pool1d(rate[:, None, :], self.window.shape[-1], self.stride)
            pitch = torch.maximum(pitch[..., 4:], pitch[..., :-4])
            weight = weight * (pitch * self.harmonics[None, :, None] <= self.fmax)
        aligned = (real[..., 4:] * real[..., :-4] + imag[..., 4:] * imag[..., :-4]) * weight
        size = torch.sqrt(
            (real[..., 4:] ** 2 + imag[..., 4:] ** 2) * (real[..., :-4] ** 2 + imag[..., :-4] ** 2) + 1e-20
        ) * weight
        # Over the harmonics the batch has under ``fmax``.
        present = weight.sum((0, 2)) > 0
        drift = (1 - aligned.sum((0, 2)) / (size.sum((0, 2)) + 1e-8)) * present
        loss = drift.sum() / present.sum().clamp(min=1)
        if self.low:
            loss = loss + drift[: self.low].sum() / present[: self.low].sum().clamp(min=1)
        return loss


class MultiScaleMelLoss(nn.Module):
    """DAC's multi-scale mel loss as the original sets it: L1 between log10
    mels over seven windows, each with its own number of bands."""

    def __init__(self, sample_rate, n_mels=(5, 10, 20, 40, 80, 160, 320),
                 window_lengths=(32, 64, 128, 256, 512, 1024, 2048), clamp_eps=1e-5):
        super().__init__()
        self.window_lengths = tuple(window_lengths)
        self.clamp_eps = clamp_eps
        for bands, window in zip(n_mels, window_lengths):
            basis = librosa_mel_fn(sr=sample_rate, n_fft=window, n_mels=bands, fmin=0, fmax=None)
            self.register_buffer(f"basis_{window}", torch.from_numpy(basis).float(), persistent=False)
            self.register_buffer(f"window_{window}", torch.hann_window(window), persistent=False)

    def _log_mel(self, audio, window):
        spec = torch.stft(
            audio, window, hop_length=window // 4, window=getattr(self, f"window_{window}"),
            center=True, return_complex=True,
        ).abs()
        return torch.log10(torch.clamp(getattr(self, f"basis_{window}") @ spec, min=self.clamp_eps))

    def forward(self, x, y):
        """``x`` and ``y`` [B, 1, T]."""
        x, y = x.float().squeeze(1), y.float().squeeze(1)
        return sum(F.l1_loss(self._log_mel(x, window), self._log_mel(y, window)) for window in self.window_lengths)


class DiscriminatorB(nn.Module):
    """DAC's multi-band discriminator: the complex STFT split into bands, one
    conv stack per band."""

    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

    def __init__(self, window_length, channels=32):
        super().__init__()
        self.window_length = window_length
        self.register_buffer("window", torch.hann_window(window_length), persistent=False)
        bins = window_length // 2 + 1
        self.bands = [(int(low * bins), int(high * bins)) for low, high in self.BANDS]

        def stack():
            return nn.ModuleList(
                [weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4)))]
                + [weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))) for _ in range(3)]
                + [weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)))]
            )

        self.band_convs = nn.ModuleList(stack() for _ in self.bands)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def forward(self, x):
        fmap, outputs = [], []
        # The STFT and each band's first conv out of autocast: FP16's range is
        # short for the gradient of an unnormalised spectrum.
        with torch.autocast(x.device.type, enabled=False):
            spec = torch.stft(
                x.squeeze(1).float(), self.window_length, hop_length=self.window_length // 4,
                window=self.window, return_complex=True,
            )
            spec = torch.view_as_real(spec).permute(0, 3, 2, 1)
            bands = [stack[0](spec[..., low:high]) for (low, high), stack in zip(self.bands, self.band_convs)]
        for band, stack in zip(bands, self.band_convs):
            band = F.leaky_relu(band, LRELU_SLOPE)
            for layer in stack[1:]:
                band = F.leaky_relu(layer(band), LRELU_SLOPE)
                fmap.append(band)
            outputs.append(band)
        x = self.conv_post(torch.cat(outputs, dim=-1))
        fmap.append(x)
        return x, fmap


class DiscriminatorCQT(nn.Module):
    """A sub-band CQT discriminator (arXiv 2311.14957): the complex CQT of the
    audio at twice its rate, a conv per octave, then a dilated stack."""

    def __init__(self, sample_rate, hop_length, n_octaves, bins_per_octave, filters=32, dilations=(1, 2, 4)):
        super().__init__()
        from nnAudio import features
        from torchaudio.transforms import Resample

        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.resample = Resample(orig_freq=sample_rate, new_freq=sample_rate * 2)
        self.cqt_transform = features.cqt.CQT2010v2(
            sr=sample_rate * 2, hop_length=hop_length, n_bins=bins_per_octave * n_octaves,
            bins_per_octave=bins_per_octave, output_format="Complex", pad_mode="constant", verbose=False,
        )
        self.conv_pres = nn.ModuleList(nn.Conv2d(2, 2, (3, 9), padding=(1, 4)) for _ in range(n_octaves))
        self.convs = nn.ModuleList(
            [nn.Conv2d(2, filters, (3, 9), padding=(1, 4))]
            + [weight_norm(nn.Conv2d(filters, filters, (3, 9), stride=(1, 2), dilation=(d, 1), padding=(d, 4)))
               for d in dilations]
            + [weight_norm(nn.Conv2d(filters, filters, (3, 3), padding=(1, 1)))]
        )
        self.conv_post = weight_norm(nn.Conv2d(filters, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            z = self.cqt_transform(self.resample(x.float()))
            # Real and imaginary parts as channels, [B, 2, frames, bins].
            z = z.permute(0, 3, 2, 1)
            width = self.bins_per_octave
            z = torch.cat(
                [conv(z[..., i * width:(i + 1) * width]) for i, conv in enumerate(self.conv_pres)], dim=-1
            )
        fmap = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), LRELU_SLOPE)
            fmap.append(z)
        return self.conv_post(z), fmap


def build_spectral_families(settings: dict, sample_rate: int, names=("mbd", "cqt")) -> dict:
    """The original's multi-band STFT (``mbd``) and sub-band CQT (``cqt``)
    discriminators, those of ``names``."""
    builders = {
        "mbd": lambda: Heads(
            DiscriminatorB(window, int(settings.get("mbd_channels", 32))) for window in settings["mbd_fft_sizes"]
        ),
        "cqt": lambda: Heads(
            DiscriminatorCQT(sample_rate, hop, octaves, bins, int(settings.get("cqt_filters", 32)))
            for hop, octaves, bins in zip(
                settings["cqt_hop_lengths"], settings["cqt_n_octaves"], settings["cqt_bins_per_octave"]
            )
        ),
    }
    return {name: builders[name]() for name in names}
