"""SiFi-GAN (Yoneyama et al., 2023): a source network makes an excitation from
a sine at f0 through pitch-dependent dilated convolutions, and a filter
network, HiFi-GAN's, shapes it into the waveform from the mel, reading the
excitation at every stage. Ported from https://github.com/chomeyama/SiFiGAN
(MIT, see THIRD_PARTY_NOTICES), with the CheapTrick source loss and its
discriminator (UnivNet spectral and HiFi-GAN period heads), which a recipe's
``discriminator.architecture`` "sifigan" picks.

With ``upsampler`` "sinc" (the default), each upsampling is a learned conv at
the lower rate followed by a fixed Kaiser-windowed sinc interpolation, whose
stopband removes the spectral images a transposed conv leaves (a mirror of
the harmonics around each stage's Nyquist); "transpose" is the original's.
``downsampler`` and ``antialias_stages`` take out the folds left: the
excitation's decimations and the nonlinearities (see ``SiFiGAN``).
The generator also reads log f0, voicing and, with ``contrast_band_edges``,
the band harmonic contrast ChouwaGAN reads.
"""

import math
from functools import partial

import librosa
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.rectified.chouwagan import (
    CONTRAST_SCALE,
    SOURCE_F0_FLOOR,
    AntiAliasedLeakyReLU,
    BandContrast,
    continuous_f0,
    known_contrast,
)

#: ``architecture`` of a SiFi-GAN rectified vocoder export, and of its runs.
ARCHITECTURE = "sifigan"
UPSAMPLERS = ("sinc", "transpose")
DOWNSAMPLERS = ("strided", "sinc")
#: The original's layout at 24 kHz, hop 120: each stage's rate and dense
#: factor. Ours are set so that a dilation spans the same time at its stage.
RELEASE_RATE = 24000
RELEASE_STAGE_RATES = (1000, 4000, 12000, 24000)
RELEASE_DENSE_FACTORS = (0.5, 1, 4, 8)


def release_dense_factors(hop_length, upsample_rates):
    """The dense factors whose dilations span, at each stage, the time the
    original's span at its own stage (one per stage, as many as it has)."""
    if len(upsample_rates) != len(RELEASE_DENSE_FACTORS):
        raise ValueError(f"Give dense_factors for {len(upsample_rates)} stages; the original has 4.")
    factors, product = [], 1
    for rate, release_rate, factor in zip(upsample_rates, RELEASE_STAGE_RATES, RELEASE_DENSE_FACTORS):
        product *= rate
        factors.append(factor * release_rate * hop_length / (RELEASE_RATE * product))
    return factors


class Conv1d(nn.Conv1d):
    """The original's ``Conv1d``, Kaiming-initialised."""

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


def sinc_kernel(ratio, taps, rolloff, beta):
    """A Kaiser-windowed sinc of ``taps`` zero crossings per side at the
    higher rate, cut at ``rolloff`` of the lower rate's Nyquist, unit DC gain."""
    positions = torch.arange(-taps * ratio, taps * ratio + 1, dtype=torch.float64)
    kernel = torch.sinc(rolloff * positions / ratio)
    kernel = kernel * torch.kaiser_window(len(positions), periodic=False, beta=beta, dtype=torch.float64)
    return (kernel / kernel.sum()).float().view(1, 1, -1)


class SincUpsample(nn.Module):
    """[B, C, T] -> [B, C, T * ratio] by zero stuffing and ``sinc_kernel``;
    the input is extended by replicating its ends."""

    def __init__(self, ratio, taps=8, rolloff=0.9, beta=8.0):
        super().__init__()
        self.ratio, self.taps = int(ratio), int(taps)
        self.register_buffer("kernel", sinc_kernel(self.ratio, self.taps, rolloff, beta) * self.ratio,
                             persistent=False)

    def forward(self, x):
        batch, channels, length = x.shape
        # Every channel as a batch item of one: a grouped conv this long is twice as slow.
        x = F.pad(x, (self.taps, self.taps), mode="replicate").reshape(batch * channels, 1, -1)
        up = F.conv_transpose1d(x, self.kernel.to(x.dtype), stride=self.ratio)
        # Input sample n lands on (n + 2 * taps) * ratio.
        start = 2 * self.taps * self.ratio
        return up[..., start:start + length * self.ratio].view(batch, channels, -1)


class SincDownsample(nn.Module):
    """[B, C, T] -> [B, C, T // ratio] through ``sinc_kernel`` before keeping
    every ``ratio``-th sample, which a strided conv's short kernel cannot
    filter enough: what it leaves above the lower Nyquist folds back."""

    def __init__(self, ratio, taps=8, rolloff=0.9, beta=8.0):
        super().__init__()
        self.ratio, self.taps = int(ratio), int(taps)
        self.register_buffer("kernel", sinc_kernel(self.ratio, self.taps, rolloff, beta), persistent=False)

    def forward(self, x):
        batch, channels, pad = x.shape[0], x.shape[1], self.taps * self.ratio
        x = F.pad(x, (pad, pad), mode="replicate").reshape(batch * channels, 1, -1)
        return F.conv1d(x, self.kernel.to(x.dtype), stride=self.ratio).view(batch, channels, -1)


def leaky_relu(slope, antialias, width):
    """LeakyReLU, or with ``antialias`` ChouwaGAN's at twice the rate."""
    return AntiAliasedLeakyReLU(slope, width) if antialias else nn.LeakyReLU(slope)


def upsample_layer(in_channels, out_channels, ratio, upsampler, sinc, activation):
    """``activation``, then ``ratio`` times the rate and ``out_channels``."""
    if upsampler == "transpose":
        layer = weight_norm(nn.ConvTranspose1d(
            in_channels, out_channels, 2 * ratio, ratio, padding=ratio // 2 + ratio % 2, output_padding=ratio % 2,
        ))
        return nn.Sequential(activation, layer)
    # Linear, so it may run before the interpolation, at the lower rate.
    conv = weight_norm(nn.Conv1d(in_channels, out_channels, 3, padding=1))
    return nn.Sequential(activation, conv, SincUpsample(ratio, **sinc))


def downsample_layer(in_channels, out_channels, ratio, downsampler, sinc, activation):
    """``ratio`` times lower rate and ``out_channels``, then ``activation``."""
    if downsampler == "strided":
        return nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, out_channels, 2 * ratio, ratio, padding=ratio - 1)), activation,
        )
    return nn.Sequential(
        SincDownsample(ratio, **sinc), weight_norm(nn.Conv1d(in_channels, out_channels, 3, padding=1)), activation,
    )


def pitch_indexing(x, d, dilation):
    """The samples ``d * dilation`` before and after each of ``x`` [B, C, T],
    reflected at the ends as the original's; ``d`` [B, 1, T]."""
    length = x.shape[-1]
    shift = (d * dilation).long().clamp(min=1)
    base = torch.arange(length, device=x.device).view(1, 1, -1)
    past = (base - shift).abs() % length
    future = base + shift
    future = torch.where(future >= length, (length - future % length) % length, future)
    expand = (-1, x.shape[1], -1)
    return x.gather(2, past.expand(*expand)), x.gather(2, future.expand(*expand))


class ResidualBlock(nn.Module):
    """HiFi-GAN's residual block; ``activation()`` makes each nonlinearity."""

    def __init__(self, channels, kernel_size, dilations, activation):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(activation(), weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=d, padding=(kernel_size - 1) // 2 * d,
            )))
            for d in dilations
        )

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class AdaptiveResidualBlock(nn.Module):
    """The source network's block: each layer reads every sample and those a
    pitch-dependent ``d * dilation`` before and after it, by 1x1 convs."""

    def __init__(self, channels, dilations, activation):
        super().__init__()
        self.dilations = tuple(dilations)

        def pointwise():
            return nn.ModuleList(weight_norm(Conv1d(channels, channels, 1)) for _ in self.dilations)

        self.convs_c, self.convs_p, self.convs_f = pointwise(), pointwise(), pointwise()
        self.activations = nn.ModuleList(activation() for _ in self.dilations)
        self.convs_a = nn.ModuleList(
            nn.Sequential(activation(), weight_norm(nn.Conv1d(channels, channels, 3, padding=1)))
            for _ in self.dilations
        )

    def forward(self, x, d):
        for i, dilation in enumerate(self.dilations):
            xt = self.activations[i](x)
            past, future = pitch_indexing(xt, d, dilation)
            xt = self.convs_c[i](xt) + self.convs_p[i](past) + self.convs_f[i](future)
            x = x + self.convs_a[i](xt)
        return x


class SiFiGAN(nn.Module):
    """Normalised log mel [B, n_mels, T], f0 [B, T] (0 unvoiced) and, with
    ``contrast_band_edges``, optionally ``measure_contrast``'s [B, bands + 1, T]
    -> [B, 1, T * hop_length]. The source network's excitation of the last
    call is kept in ``excitation`` for the source loss.

    ``dense_factors`` default to the original's spans (``release_dense_factors``).
    ``downsampler`` "sinc" filters the excitation's pyramids before each
    decimation, which "strided" (the original's) folds around each stage's
    Nyquist. The nonlinearities at the output rate of the stages in
    ``antialias_stages`` (their blocks', and the next upsampling's and the
    downsampling's into them) run at twice the rate, through filters of
    ``antialias_width`` taps per side. ``sinc_taps``, ``sinc_rolloff`` and
    ``sinc_beta`` design both resamplers."""

    def __init__(self, sample_rate, num_mels, hop_length, channels=512, kernel_size=7,
                 upsample_rates=(8, 8, 4, 2), dense_factors=None,
                 source_dilations=((1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)),
                 filter_kernel_sizes=(3, 5, 7), filter_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 upsampler="sinc", downsampler="strided", sinc_taps=8, sinc_rolloff=0.9, sinc_beta=8.0,
                 antialias_stages=(), antialias_width=8, sine_amp=0.1, noise_amp=0.003, contrast_band_edges=None):
        super().__init__()
        if upsampler not in UPSAMPLERS:
            raise ValueError(f"upsampler must be one of {UPSAMPLERS}, not {upsampler!r}.")
        if downsampler not in DOWNSAMPLERS:
            raise ValueError(f"downsampler must be one of {DOWNSAMPLERS}, not {downsampler!r}.")
        if math.prod(upsample_rates) != hop_length:
            raise ValueError(f"upsample_rates {list(upsample_rates)} must multiply to the hop {hop_length}.")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.upsample_rates = [int(r) for r in upsample_rates]
        self.dense_factors = [float(f) for f in (dense_factors or release_dense_factors(hop_length, upsample_rates))]
        self.sine_amp, self.noise_amp = float(sine_amp), float(noise_amp)
        self.contrast = BandContrast(sample_rate, hop_length, contrast_band_edges) if contrast_band_edges else None
        contrast_inputs = self.contrast.bands + 1 if self.contrast is not None else 0
        stages = len(self.upsample_rates)
        widths = [channels // 2 ** i for i in range(stages + 1)]
        sinc = dict(taps=int(sinc_taps), rolloff=float(sinc_rolloff), beta=float(sinc_beta))
        filtered = {int(stage) for stage in antialias_stages}

        def activation(stage, slope=0.1):
            """A factory of the nonlinearity at ``stage``'s output rate (-1: the frames')."""
            return partial(leaky_relu, slope, stage in filtered, int(antialias_width))

        self.input_conv = weight_norm(Conv1d(num_mels + 2 + contrast_inputs, channels, kernel_size,
                                             padding=kernel_size // 2))

        def upsamples():
            return nn.ModuleList(
                upsample_layer(widths[i], widths[i + 1], r, upsampler, sinc, activation(i - 1)())
                for i, r in enumerate(self.upsample_rates)
            )

        def downsamples():
            # From the audio rate down: layer i lands at the output rate of stage i - 1.
            return nn.ModuleList(
                downsample_layer(widths[i + 1], widths[i], self.upsample_rates[i], downsampler, sinc,
                                 activation(i - 1)())
                for i in reversed(range(1, stages))
            )

        self.sn_upsamples, self.fn_upsamples = upsamples(), upsamples()
        self.sn_blocks = nn.ModuleList(
            AdaptiveResidualBlock(widths[i + 1], source_dilations[i], activation(i)) for i in range(stages)
        )
        self.fn_blocks = nn.ModuleList(
            ResidualBlock(widths[i + 1], k, d, activation(i))
            for i in range(stages) for k, d in zip(filter_kernel_sizes, filter_dilations)
        )
        self.filter_kernels = len(filter_kernel_sizes)
        # The source's pyramid for the sine and the filter's for the excitation.
        self.sn_emb = weight_norm(Conv1d(1, widths[-1], kernel_size, padding=kernel_size // 2))
        self.sn_downsamples, self.fn_downsamples = downsamples(), downsamples()
        last = activation(stages - 1, 0.01)
        self.sn_output = nn.Sequential(last(), weight_norm(nn.Conv1d(widths[-1], 1, kernel_size,
                                                                       padding=kernel_size // 2)))
        self.fn_output = nn.Sequential(
            last(), weight_norm(nn.Conv1d(widths[-1], 1, kernel_size, padding=kernel_size // 2)), nn.Tanh(),
        )
        self.excitation = None

    def compile_blocks(self) -> None:
        for block in (*self.sn_blocks, *self.fn_blocks):
            block.compile()

    def measure_contrast(self, audio, f0):
        """The ``contrast`` input from the audio itself ([B, 1, T] or [B, T])."""
        return known_contrast(self.contrast(audio, f0) / CONTRAST_SCALE)

    def _sine(self, f0, samples):
        """The original's input: a sine at f0 where voiced plus a little noise."""
        f0 = F.interpolate(f0.unsqueeze(1).double(), size=samples, mode="nearest")
        voiced = (f0 > 0).float()
        phase = torch.cumsum(f0 / self.sample_rate % 1, dim=-1) % 1
        sine = voiced * torch.sin(2 * math.pi * phase).float() * self.sine_amp
        noise_amp = voiced * self.noise_amp + (1 - voiced) * self.noise_amp / 3
        return sine + torch.randn_like(sine) * noise_amp

    def _dilation_factors(self, cf0):
        """Per stage, ``sample_rate / (dense factor * f0)`` at its rate, as the
        original's (which reads the full rate at every stage)."""
        cf0 = cf0.clamp(min=SOURCE_F0_FLOOR).unsqueeze(1)
        factors, length = [], cf0.shape[-1]
        for rate, dense in zip(self.upsample_rates, self.dense_factors):
            length *= rate
            stage = F.interpolate(cf0, size=length, mode="linear", align_corners=False)
            factors.append(self.sample_rate / (dense * stage))
        return factors

    def forward(self, mel, f0, contrast=None):
        f0 = f0.float()
        voiced = f0 > 0
        cf0 = continuous_f0(f0)
        inputs = [mel, torch.log(cf0.clamp(min=SOURCE_F0_FLOOR) / 200.0).unsqueeze(1),
                  voiced.unsqueeze(1).to(mel.dtype)]
        if self.contrast is not None:
            if contrast is None:
                contrast = mel.new_zeros(mel.shape[0], self.contrast.bands + 1, mel.shape[-1])
            inputs.append(contrast.to(mel.dtype))
        with torch.no_grad():
            sine = self._sine(f0, f0.shape[-1] * self.hop_length)
            factors = self._dilation_factors(cf0)

        c = self.input_conv(torch.cat(inputs, dim=1))
        e = c
        x = self.sn_emb(sine)
        embs = [x]
        for down in self.sn_downsamples:
            x = down(x)
            embs.append(x)
        for i, (up, block) in enumerate(zip(self.sn_upsamples, self.sn_blocks)):
            e = block(up(e) + embs[-i - 1], factors[i])
        self.excitation = self.sn_output(e)

        embs = [e]
        for down in self.fn_downsamples:
            e = down(e)
            embs.append(e)
        for i, up in enumerate(self.fn_upsamples):
            c = up(c) + embs[-i - 1]
            blocks = self.fn_blocks[i * self.filter_kernels:(i + 1) * self.filter_kernels]
            c = sum(block(c) for block in blocks) / self.filter_kernels
        return self.fn_output(c)


class CheapTrick(nn.Module):
    """WORLD's CheapTrick spectral envelope, as the original computes it:
    waveform [B, T] and f0 [B, frames] -> log envelope [B, frames, bins],
    a frame every ``hop_size`` samples."""

    def __init__(self, sample_rate, hop_size, fft_size, f0_floor, f0_ceil, q1=-0.15):
        super().__init__()
        if fft_size <= 3.0 * sample_rate / f0_floor:
            raise ValueError(f"CheapTrick needs fft_size over {3.0 * sample_rate / f0_floor:.0f} for f0_floor {f0_floor}.")
        self.hop_size, self.fft_size = int(hop_size), int(fft_size)
        self.f0_floor, self.f0_ceil = int(f0_floor), int(f0_ceil)
        bins = fft_size // 2 + 1
        windows = torch.zeros(f0_ceil + 1, fft_size)
        smoothing = torch.zeros(f0_ceil + 1, bins)
        compensation = torch.zeros(f0_ceil + 1, bins)
        q0 = 1.0 - 2.0 * q1
        quefrency = torch.arange(1, bins) / sample_rate
        for f0 in range(f0_floor, f0_ceil + 1):
            half = round(1.5 * sample_rate / f0)
            position = torch.arange(-half, half + 1) / 1.5 / sample_rate
            window = torch.zeros(fft_size)
            window[fft_size // 2 - half:fft_size // 2 + half + 1] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            windows[f0] = window / window.square().sum().sqrt()
            smoothing[f0, 0] = 1.0
            smoothing[f0, 1:] = torch.sin(math.pi * f0 * quefrency) / (math.pi * f0 * quefrency)
            compensation[f0, 0] = q0 + 2.0 * q1
            compensation[f0, 1:] = q0 + 2.0 * q1 * torch.cos(2.0 * math.pi * f0 * quefrency)
        self.register_buffer("windows", windows, persistent=False)
        self.register_buffer("smoothing", smoothing, persistent=False)
        self.register_buffer("compensation", compensation, persistent=False)

    def forward(self, x, f0, elim_0th=True):
        f0 = torch.where(f0 > 0, f0, torch.full_like(f0, self.f0_ceil))
        f0 = f0.clamp(self.f0_floor, self.f0_ceil).round().long()
        frames = F.pad(x, (self.fft_size // 2, self.fft_size // 2)).unfold(1, self.fft_size, self.hop_size)
        frames = frames[:, :f0.shape[1]]
        spectrum = torch.fft.rfft(frames * self.windows[f0]).abs()
        mirrored = torch.cat((spectrum, spectrum[..., 1:-1].flip(-1)), dim=-1)
        cepstrum = torch.fft.rfft(torch.log(mirrored.clamp(min=1e-7))).real
        if elim_0th:
            cepstrum[..., 0] = 0
        liftered = cepstrum * self.smoothing[f0] * self.compensation[f0]
        return torch.fft.irfft(liftered)[..., :spectrum.shape[-1]]


class SourceLoss(nn.Module):
    """hn-uSFGAN's regularisation, the original's ``ResidualLoss``: the
    excitation's log mel against the target's spectrum over its CheapTrick
    envelope (the target with its envelope flattened), at the target's level."""

    def __init__(self, sample_rate, hop_size, fft_size=4096, f0_floor=50, f0_ceil=1100, n_mels=80):
        super().__init__()
        self.hop_size, self.fft_size = int(hop_size), int(fft_size)
        self.cheaptrick = CheapTrick(sample_rate, hop_size, fft_size, f0_floor, f0_ceil)
        self.register_buffer("window", torch.hann_window(self.fft_size), persistent=False)
        melmat = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mels, fmin=0, fmax=sample_rate / 2).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float(), persistent=False)

    def _magnitude(self, x):
        spectrum = torch.stft(x, self.fft_size, self.hop_size, self.fft_size, self.window, return_complex=True)
        return spectrum.abs().clamp(min=1e-7).transpose(1, 2)

    def forward(self, excitation, y, f0):
        """``excitation`` and ``y`` [B, 1, T], ``f0`` [B, T // hop] (0 unvoiced)."""
        with torch.autocast(y.device.type, enabled=False):
            excitation, y, f0 = excitation.squeeze(1).float(), y.squeeze(1).float(), f0.float()
            with torch.no_grad():
                envelope = self.cheaptrick(y, f0)
                target = self._magnitude(y)
                frames = min(envelope.shape[1], target.shape[1])
                envelope, target = envelope[:, :frames], target[:, :frames]
                level = target.mean(-1, keepdim=True)
                flat = (torch.log(target) - envelope).exp()
                flat = level / flat.mean(-1, keepdim=True) * flat
                flat = torch.log((flat @ self.melmat).clamp(min=1e-7))
            source = self._magnitude(excitation)[:, :frames]
            source = torch.log((source @ self.melmat).clamp(min=1e-7))
            return F.l1_loss(source, flat)


class PeriodDiscriminator(nn.Module):
    """The original's HiFi-GAN period discriminator, output conv included
    (kernel ``kernel_sizes[1] - 1``)."""

    use_san = False

    def __init__(self, period, channels=32, kernel_sizes=(5, 3), downsample_scales=(3, 3, 3, 3, 1),
                 max_channels=1024):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, channels
        for scale in downsample_scales:
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_sizes[0], 1), (scale, 1), ((kernel_sizes[0] - 1) // 2, 0)
            )))
            in_channels, out_channels = out_channels, min(out_channels * 4, max_channels)
        self.output_conv = weight_norm(nn.Conv2d(
            out_channels, 1, (kernel_sizes[1] - 1, 1), padding=((kernel_sizes[1] - 1) // 2, 0)
        ))

    def forward(self, x, san_training=False):
        b, c, t = x.shape
        if t % self.period:
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


class UnivNetSpectralDiscriminator(nn.Module):
    """The original's UnivNet spectral discriminator: the linear magnitude
    [frames, bins] through unpadded convs, strided along frequency; every
    layer's output, the last included, is a feature map."""

    use_san = False
    KERNEL_SIZES = ((3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3))
    STRIDES = ((1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1))

    def __init__(self, fft_size, hop_size, win_length, channels=32):
        super().__init__()
        self.fft_size, self.hop_size, self.win_length = int(fft_size), int(hop_size), int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(1 if index == 0 else channels, channels, kernel, stride=stride))
            for index, (kernel, stride) in enumerate(zip(self.KERNEL_SIZES[:-1], self.STRIDES[:-1]))
        )
        self.output_conv = weight_norm(nn.Conv2d(channels, 1, self.KERNEL_SIZES[-1], stride=self.STRIDES[-1]))

    def forward(self, x, san_training=False):
        # As torchaudio's spectrogram with ``pad``: zeros, then the centred STFT.
        with torch.autocast(x.device.type, enabled=False):
            x = F.pad(x.squeeze(1).float(), (self.win_length // 2, self.win_length // 2))
            x = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, center=True,
                           return_complex=True).abs()
            x = x.transpose(1, 2).unsqueeze(1)
            x = F.leaky_relu(self.convs[0](x), 0.2)
        fmap = [x]
        for conv in self.convs[1:]:
            x = F.leaky_relu(conv(x), 0.2)
            fmap.append(x)
        x = self.output_conv(x)
        fmap.append(x)
        return x, fmap


def build_families(settings: dict, sample_rate: int) -> dict:
    """The original's discriminator: UnivNet spectral heads at ``fft_sizes``,
    ``hop_sizes`` and ``win_lengths``, and HiFi-GAN period heads at
    ``periods``, ``mpd_max_channels`` wide."""
    from rvc.rectified.chouwagan import Heads

    return {
        "univnet": Heads(
            UnivNetSpectralDiscriminator(*sizes, channels=int(settings.get("channels", 32)))
            for sizes in zip(settings["fft_sizes"], settings["hop_sizes"], settings["win_lengths"])
        ),
        "mpd": Heads(
            PeriodDiscriminator(p, max_channels=int(settings.get("mpd_max_channels", 1024)))
            for p in settings["periods"]
        ),
    }
