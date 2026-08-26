import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding


def _norm_layer(layer):
    return weight_norm(layer)


def _normalize_weight(weight):
    shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    return weight / weight.flatten(1).norm(p=2, dim=1).clamp_min(1e-12).view(shape)


class SANConv2d(nn.Conv2d):
    """Discriminative normalized convolution used by Slicing Adversarial Network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale = self.weight.detach().flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
        self.weight = nn.Parameter(self.weight.detach() / scale.view(-1, 1, 1, 1))
        self.scale = nn.Parameter(scale)
        if self.bias is not None:
            self.bias = nn.Parameter(
                torch.zeros(self.in_channels, device=self.weight.device, dtype=self.weight.dtype)
            )

    def forward(self, input, san_training=False):
        # The SAN head is small, so keeping this operation in FP32 avoids
        # overflow in the normalized projection when the rest of training is
        # running under autocast/FP16.
        with torch.autocast(device_type=input.device.type, enabled=False):
            input = input.float()
            normalized_weight = _normalize_weight(self.weight.float())
            scale = self.scale.float().clamp_min(1e-4).clamp_max(4.0)
            scale = scale.view(1, self.out_channels, 1, 1)
            if self.bias is not None:
                input = input + self.bias.float().view(1, self.in_channels, 1, 1)
            if san_training:
                function_output = F.conv2d(
                    input, normalized_weight.detach(), None, self.stride,
                    self.padding, self.dilation, self.groups
                ) * scale
                direction_output = F.conv2d(
                    input.detach(), normalized_weight, None, self.stride,
                    self.padding, self.dilation, self.groups
                ) * scale.detach()
                return function_output, direction_output
            return F.conv2d(
                input, normalized_weight, None, self.stride,
                self.padding, self.dilation, self.groups
            ) * scale

    @torch.no_grad()
    def normalize_weight(self):
        """Reproject the SAN direction after an optimizer update."""
        self.weight.copy_(_normalize_weight(self.weight))


class SANConv1d(nn.Conv1d):
    """1-D counterpart of :class:`SANConv2d` used by the sub-band branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scale = self.weight.detach().flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
        self.weight = nn.Parameter(self.weight.detach() / scale.view(-1, 1, 1))
        self.scale = nn.Parameter(scale)
        if self.bias is not None:
            self.bias = nn.Parameter(
                torch.zeros(self.in_channels, device=self.weight.device, dtype=self.weight.dtype)
            )

    def forward(self, input, san_training=False):
        with torch.autocast(device_type=input.device.type, enabled=False):
            input = input.float()
            normalized_weight = _normalize_weight(self.weight.float())
            scale = self.scale.float().clamp_min(1e-4).clamp_max(4.0)
            scale = scale.view(1, self.out_channels, 1)
            if self.bias is not None:
                input = input + self.bias.float().view(1, self.in_channels, 1)
            if san_training:
                function_output = F.conv1d(
                    input, normalized_weight.detach(), None, self.stride,
                    self.padding, self.dilation, self.groups
                ) * scale
                direction_output = F.conv1d(
                    input.detach(), normalized_weight, None, self.stride,
                    self.padding, self.dilation, self.groups
                ) * scale.detach()
                return function_output, direction_output
            return F.conv1d(
                input, normalized_weight, None, self.stride,
                self.padding, self.dilation, self.groups
            ) * scale

    @torch.no_grad()
    def normalize_weight(self):
        """Reproject the SAN direction after an optimizer update."""
        self.weight.copy_(_normalize_weight(self.weight))


# The period branch is expressed with 2-D convolutions on purpose.  Folding the
# period axis into the batch and using conv1d computes exactly the same thing
# and is faster on CPU, but on GPU the permute it needs costs more than the
# narrow conv2d saves -- measured at +12% end-to-end on an RTX 5060.
class ChouwaPeriodDiscriminator(nn.Module):
    def __init__(self, period: int, use_spectral_norm: bool = False, use_san: bool = True):
        super().__init__()
        norm = _norm_layer
        self.use_san = bool(use_san)
        in_channels = [1, 16, 64, 128, 192]
        out_channels = [16, 64, 128, 192, 256]
        strides = [3, 3, 3, 3, 1]
        self.period = int(period)
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        (5, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                )
                for in_ch, out_ch, stride in zip(
                    in_channels, out_channels, strides, strict=True
                )
            ]
        )
        self.conv_post = SANConv2d(256, 1, (3, 1), padding=(1, 0)) if self.use_san else norm(
            nn.Conv2d(256, 1, (3, 1), padding=(1, 0))
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor, san_training=False):
        fmap = []
        batch, channels, length = x.shape
        if length % self.period:
            x = F.pad(x, (0, self.period - length % self.period), mode="reflect")
        x = x.view(batch, channels, -1, self.period)
        for conv in self.convs:
            x = self.activation(conv(x))
            fmap.append(x)
        x = self.conv_post(x, san_training=san_training) if self.use_san else self.conv_post(x)
        if san_training and self.use_san:
            function_output, direction_output = x
            fmap.append(function_output)
            return [torch.flatten(function_output, 1, -1), torch.flatten(direction_output, 1, -1)], fmap
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap



#: Lower bound on the STFT magnitude before power-law compression, in linear
#: amplitude (1e-5 is -100 dBFS).  See `ChouwaSpectrogramDiscriminator`.
MAGNITUDE_FLOOR = 1e-5

#: The periods the multi-period block folds the waveform by.
#:
#: Three, not the five this shipped with. The five agreed with each other to a
#: Spearman rho of 0.86-0.98 *after* partialling out clip RMS -- the redundancy
#: is real and not an artefact of loud clips being easy to judge -- while all
#: three STFT branches scored -0.62 to -0.04 against the period block, i.e.
#: independent. Five near-copies of one opinion and three of another means the
#: mean the losses take is 5/9 one opinion; the branch count *is* a weight.
#:
#: So the cut is not a saving, it is a reweighting: with `normalize=True` on
#: every ChouwaGAN loss, dropping two period branches leaves `loss_adv` and
#: `loss_fm` at the same scale but moves the independent branches from 44% to
#: 57% of the adversarial mean, and from 35% to 47% of the feature-matching
#: mean. That the freed 834 k parameters and 2 of 9 kernel launches come along
#: is incidental -- see [[chouwagan-step-is-launch-bound]] for why the launches
#: are the part that could show up in wall clock.
#:
#: 2, 5 and 11 keep the range and stay pairwise coprime, so no branch's folding
#: grid is a sub-multiple of another's; 3 and 7 go because 2 and 5 already
#: bracket the low end and 11 holds the top.
PERIODS = (2, 5, 11)

#: Default: a single full-band stack.
FULL_BAND = ((0.0, 1.0),)

#: Optional split of the frequency axis into sub-bands with independent weights.
#: Sound in principle -- the sparse high bins stop competing with the loud low
#: ones -- but it turns 3 convolutions per depth into 15, and the branch is
#: launch-bound rather than FLOP-bound: measured ~2x slower end to end on an
#: RTX 5060 even at batch 8.  Opt in only with GPU headroom to spare.
SPECTROGRAM_BAND_EDGES = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

#: The STFT branches, as ``n_fft``/``hop_length`` plus whatever each one needs to
#: override.  Anything omitted falls back to the discriminator-wide default.
#:
#: The three branches are *not* three resolutions of one design.
#:
#: `stft_512` and `stft_1024` cannot resolve a harmonic comb at all -- at 86 Hz
#: per bin, harmonics of a 120 Hz f0 sit 1.4 bins apart -- so their response
#: collapses onto the low band and they are best read as *transient* branches:
#: hop 128 is 2.9 ms, the finest time resolution in the set, and `stft_512`
#: carries the most independent signal anywhere in the discriminator.  Do not
#: widen them in frequency; there is nothing there for a wider kernel to see.
#:
#: `stft_2048` is the only branch that can resolve the comb (5.57 bins between
#: harmonics at f0=120) and the only one carrying real high-frequency signal, so
#: it is the one that gets the resolution budget:
#:
#: * A ``(9, 5)`` first kernel spans 194 Hz = 1.6 harmonic periods, against the
#:   0.54 of a period a ``(3, 5)`` kernel saw.  Below one period a layer cannot
#:   see a comb at all -- only local slope -- which is what "cannot resolve"
#:   meant here.  It is also the cheapest layer to widen (2 -> 64 channels).
#: * The third stage keeps its frequency axis (stride ``(1, 2)``) so the output
#:   grid stops sampling the comb below its own Nyquist and aliasing away the
#:   structure the first kernel just made visible.
#: * ``(64, 128, 192)`` channels take it from ~200 parameters per logit position
#:   to ~450, in line with the ~1900 the period and sub-band branches have.  At
#:   (32, 64, 96) there was almost nothing with which to judge spectral texture.
SPECTROGRAM_SPECS = (
    {"n_fft": 512, "hop_length": 128},
    {"n_fft": 1024, "hop_length": 256},
    {
        "n_fft": 2048,
        "hop_length": 512,
        "channels": (64, 128, 192),
        "kernels": ((9, 5), (3, 5), (3, 5)),
        "strides": ((2, 2), (2, 2), (1, 2)),
    },
)


class ChouwaSpectrogramDiscriminator(nn.Module):
    """Discriminator over the compressed complex STFT.

    Two things separate this from a plain magnitude MRD, both of which cost
    almost nothing (+4% end to end, measured on GPU):

    * It consumes the real and imaginary parts, so the adversarial signal
      carries phase.  The mel/spectral reconstruction losses are phase-blind,
      which otherwise leaves the waveform loss and the period branches as the
      only sources of phase supervision.
    * Magnitudes are power-law compressed.  Linear magnitudes span some five
      orders of magnitude across the spectrum, so the stack would otherwise be
      driven almost entirely by the low bins, with the sparse 10-20 kHz region
      contributing close to nothing to the logit.

    Splitting the frequency axis is supported through `band_edges` but is off by
    default; see `SPECTROGRAM_BAND_EDGES` for why.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        use_spectral_norm: bool = False,
        use_san: bool = True,
        channels: tuple[int, ...] = (32, 64, 96),
        compression: float = 0.3,
        band_edges: tuple[tuple[float, float], ...] = FULL_BAND,
        kernels: tuple[tuple[int, int], ...] | None = None,
        strides: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__()
        norm = _norm_layer
        self.use_san = bool(use_san)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.compression = float(compression)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        bins = self.n_fft // 2 + 1
        bands = []
        for start, end in band_edges:
            lo = int(round(float(start) * bins))
            hi = int(round(float(end) * bins))
            # A band has to survive the stride schedule, so never let rounding
            # collapse one to fewer bins than there are downsampling stages.
            if hi - lo < len(channels):
                hi = min(bins, lo + len(channels))
            if hi > lo:
                bands.append((lo, hi))
        self.bands = tuple(bands)

        # Real and imaginary parts enter as two channels.
        stack_channels = (2,) + tuple(channels)
        self.depth = len(stack_channels) - 1
        # A per-stage schedule, so one branch can be given the resolution it can
        # actually use without changing the two that cannot.  ``None`` keeps the
        # uniform ``(3, 5)`` kernel and ``(2, 2)`` stride every branch shipped
        # with; see `SPECTROGRAM_SPECS` for which branch deviates and why.
        kernels = self._per_stage(kernels, (3, 5))
        strides = self._per_stage(strides, (2, 2))
        self.band_convs = nn.ModuleList(
            nn.ModuleList(
                ChouwaSpectrogramConv2d(in_ch, out_ch, stride, norm, kernel=kernel)
                for in_ch, out_ch, kernel, stride in zip(
                    stack_channels[:-1],
                    stack_channels[1:],
                    kernels,
                    strides,
                    strict=True,
                )
            )
            for _ in self.bands
        )
        self.conv_post = SANConv2d(channels[-1], 1, (3, 3), padding=(1, 1)) if self.use_san else norm(
            nn.Conv2d(channels[-1], 1, (3, 3), padding=(1, 1))
        )

    def _per_stage(self, values, default: tuple[int, int]):
        """Broadcast one ``(freq, time)`` pair over the stack, or check a list.

        A wrong-length schedule is a configuration error that would otherwise
        surface as a shape mismatch several layers down, or -- worse -- as a
        silently shortened stack.
        """
        if values is None:
            return (tuple(default),) * self.depth
        values = tuple(tuple(int(axis) for axis in value) for value in values)
        if len(values) != self.depth:
            raise ValueError(
                f"Expected {self.depth} entries for a {self.depth}-stage "
                f"spectrogram stack, received {len(values)}."
            )
        return values

    def forward(self, x: Tensor, san_training=False):
        return self.forward_spectrogram(self.spectrogram(x), san_training=san_training)

    def spectrogram(self, x: Tensor) -> Tensor:
        waveform = x.squeeze(1).float()
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        # Power-law compression rescales the magnitude to |X| ** compression
        # while leaving the phase untouched, because the gain is applied to
        # both parts of the complex value.
        #
        # The floor is a gradient guard, not a numerical one.  d|X|**0.3/d|X|
        # diverges as the magnitude goes to zero, so near-silent bins hand the
        # backward pass an unbounded gain.  1e-5 is -100 dBFS, below the noise
        # floor of any real recording.
        #
        # The value is inherited from an earlier constraint -- containing the R1
        # penalty, back when it differentiated this transform -- and no floor was
        # ever enough for that; see ``r1_penalty``.  It stays because the
        # ordinary discriminative backward runs through here too, but do not
        # read 1e-5 as tuned for that.
        magnitude = spec.abs().clamp_min(MAGNITUDE_FLOOR)
        gain = magnitude.pow(self.compression - 1.0)
        return torch.stack((spec.real * gain, spec.imag * gain), dim=1)

    def forward_spectrogram(self, x: Tensor, san_training=False):
        fmap = []
        bands = [x[:, :, start:end] for start, end in self.bands]
        # Every band runs the same stride schedule, so the time axis stays
        # aligned and the per-depth maps can be stitched back into a single
        # feature map.  Keeping one map per depth rather than one per band
        # leaves the feature-matching loss weighted as it was before the split.
        for depth in range(self.depth):
            bands = [convs[depth](band) for convs, band in zip(self.band_convs, bands)]
            fmap.append(torch.cat(bands, dim=2))
        x = fmap[-1]
        x = self.conv_post(x, san_training=san_training) if self.use_san else self.conv_post(x)
        if san_training and self.use_san:
            function_output, direction_output = x
            fmap.append(function_output)
            return [torch.flatten(function_output, 1, -1), torch.flatten(direction_output, 1, -1)], fmap
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class ChouwaSpectrogramConv2d(nn.Module):
    """One dense convolution per stage of the spectrogram stack.

    Dense rather than the depthwise-separable pair it looks like it should be.
    Depthwise convolution is memory-bound and gets a poor kernel, so the FLOPs
    it saves here do not become time -- measured, the dense form is faster while
    computing *more* arithmetic.  Do not "optimise" it back on a FLOP count.

    Capacity comes along for free, which matters for a second reason: these
    branches are the only place the discriminator can judge spectral texture,
    which is the one thing the adversarial term is uniquely able to teach.
    """

    def __init__(self, in_channels: int, out_channels: int, stride, norm, kernel=(3, 5)):
        super().__init__()
        kernel = tuple(int(axis) for axis in kernel)
        # 'same' padding for the odd kernels this stack uses, so widening a
        # kernel changes what a layer *sees* and not the shape it produces.
        padding = tuple(axis // 2 for axis in kernel)
        self.conv = norm(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding)
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv(x))


def _pqmf_filters(num_bands: int, taps: int = 62) -> Tensor:
    """Cosine-modulated analysis bank used as a fixed front-end.

    Only band separation matters here -- the bank is never inverted -- so the
    prototype is a plain Kaiser-windowed sinc rather than an optimised
    perfect-reconstruction filter.
    """
    positions = torch.arange(taps + 1, dtype=torch.float32) - taps / 2
    cutoff = 0.5 / num_bands
    prototype = cutoff * torch.sinc(cutoff * positions)
    prototype = prototype * torch.kaiser_window(taps + 1, periodic=False, beta=9.0)
    filters = [
        2
        * prototype
        * torch.cos(
            (2 * band + 1) * math.pi / (2 * num_bands) * positions
            + (-1) ** band * math.pi / 4
        )
        for band in range(num_bands)
    ]
    return torch.stack(filters).unsqueeze(1)


class ChouwaCQTDiscriminator(nn.Module):
    """Discriminator on a log-frequency (constant-Q) projection of the STFT.

    What this buys that no linear-frequency branch can: **a harmonic stack is
    translation-invariant in pitch.**  On a log axis, changing f0 slides the
    whole pattern rigidly without changing its shape, so one kernel learns "this
    is a harmonic series" once and detects it at every pitch.  On the linear
    axis the comb spacing *is* f0, so a fixed kernel is tuned to one pitch and
    mis-sized at every other -- `stft_2048`'s ``(9, 5)`` spans 1.6 harmonic
    periods at f0=120 and 0.6 at f0=320.  That is the gap this fills, and it is
    a different gap from the one `SPECTROGRAM_BAND_EDGES` fills.

    It is a *pseudo*-CQT: a log-spaced triangular filterbank applied to the
    linear STFT rather than a bank of per-bin kernels at their own hop sizes.
    That is the cheap construction and its cost is the bottom octaves, where a
    log bin is narrower than a linear one and the projection degrades to
    nearest-bin interpolation.  Measured, bins that read a single STFT bin:

        n_fft 2048   44 of 128, everything under 386 Hz
        n_fft 4096   28 of 128, under 193 Hz
        n_fft 8192   12 of 128, under  97 Hz

    ``n_fft`` is close to free here -- the conv stack always sees ``n_bins`` rows
    whatever the transform, so only the matmul and the STFT itself grow, and the
    branch measured 4.94 / 4.76 / 4.78 ms across those three.  4096 rather than
    8192 is a *time*-resolution choice, not a cost one: 8192 is a 186 ms window,
    longer than a vibrato period, so the harmonic stack it reports is averaged
    across the modulation the discriminator ought to be judging.  4096 is 93 ms.

    The degenerate rows are not wasted so much as oversampled, and they sit
    below f0 for most voices anyway: the branch exists for the *shape* of the
    stack, which lives in the octaves above the fundamental.

    **A true CQT does not escape this, it only hides it.**  A constant-Q bin at
    ``f`` needs ``Q * sr / f`` samples of support, and at 16 bins per octave Q is
    22.6, so a 55 Hz bin needs 413 ms -- longer than the 400 ms training segment.
    nnAudio's ``CQT1992v2`` reports that bin anyway by zero-padding its kernel to
    32768 samples (743 ms), which fabricates resolution the segment cannot carry,
    and charges **6x** this front end to do it: 2.76 ms against 0.46 ms at batch
    8, with the "efficient" ``CQT2010v2`` still 4x at 1.88 ms.  The honest
    response is to put ``f_min`` where the data supports it rather than to buy a
    dearer transform, which is why the default is 80 Hz and not 55: 20 of 128
    rows interpolated instead of 28, and the bank reaches 20.5 kHz instead of
    14.1.

    The filterbank is applied to the real and imaginary parts separately, which
    is linear and therefore keeps phase, in the same spirit as
    `ChouwaSpectrogramDiscriminator`.  Compression is applied after projection,
    so the gain law sees the log-band magnitude rather than the raw bin.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 512,
        f_min: float = 80.0,
        bins_per_octave: int = 16,
        n_bins: int = 128,
        use_san: bool = True,
        channels: tuple[int, ...] = (64, 128, 192),
        compression: float = 0.3,
        kernels: tuple[tuple[int, int], ...] = ((9, 5), (3, 5), (3, 5)),
        strides: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (1, 2)),
    ):
        super().__init__()
        norm = _norm_layer
        self.use_san = bool(use_san)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.bins_per_octave = int(bins_per_octave)
        self.n_bins = int(n_bins)
        self.compression = float(compression)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer(
            "filterbank",
            _log_filterbank(
                sample_rate, self.n_fft, float(f_min), self.bins_per_octave, self.n_bins
            ),
            persistent=True,
        )
        stack_channels = (2,) + tuple(channels)
        self.depth = len(stack_channels) - 1
        self.convs = nn.ModuleList(
            ChouwaSpectrogramConv2d(in_ch, out_ch, stride, norm, kernel=kernel)
            for in_ch, out_ch, kernel, stride in zip(
                stack_channels[:-1], stack_channels[1:], kernels, strides, strict=True
            )
        )
        self.conv_post = SANConv2d(channels[-1], 1, (3, 3), padding=(1, 1)) if self.use_san else norm(
            nn.Conv2d(channels[-1], 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: Tensor, san_training=False):
        return self.forward_spectrogram(self.spectrogram(x), san_training=san_training)

    def spectrogram(self, x: Tensor) -> Tensor:
        waveform = x.squeeze(1).float()
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        bank = self.filterbank.to(dtype=spec.real.dtype, device=spec.real.device)
        # Projected before compression, so the power law sees the energy of a
        # log band rather than of one linear bin.  Real and imaginary go through
        # the same real-valued bank, which is what keeps this linear in the
        # complex value and therefore phase-preserving.
        real = torch.matmul(bank, spec.real)
        imag = torch.matmul(bank, spec.imag)
        magnitude = torch.sqrt(real * real + imag * imag).clamp_min(MAGNITUDE_FLOOR)
        gain = magnitude.pow(self.compression - 1.0)
        return torch.stack((real * gain, imag * gain), dim=1)

    def forward_spectrogram(self, x: Tensor, san_training=False):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        x = self.conv_post(x, san_training=san_training) if self.use_san else self.conv_post(x)
        if san_training and self.use_san:
            function_output, direction_output = x
            fmap.append(function_output)
            return [torch.flatten(function_output, 1, -1), torch.flatten(direction_output, 1, -1)], fmap
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


def _log_filterbank(
    sample_rate: int, n_fft: int, f_min: float, bins_per_octave: int, n_bins: int
) -> Tensor:
    """Triangular filters on log-spaced centres, each normalised to unit sum.

    Unit sum rather than unit energy: the bank is applied to a *complex* value,
    so the natural reading of each output is a weighted average of neighbouring
    bins rather than a power accumulation, and unit sum keeps a flat-magnitude
    input flat across the axis instead of tilting it by bandwidth.
    """
    bins = n_fft // 2 + 1
    frequencies = torch.linspace(0.0, sample_rate / 2.0, bins)
    centres = f_min * (2.0 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))
    bank = torch.zeros(n_bins, bins)
    ratio = 2.0 ** (1.0 / bins_per_octave)
    for index, centre in enumerate(centres.tolist()):
        low, high = centre / ratio, centre * ratio
        if low >= sample_rate / 2.0:
            break
        rising = (frequencies - low) / max(centre - low, 1e-6)
        falling = (high - frequencies) / max(high - centre, 1e-6)
        weights = torch.minimum(rising, falling).clamp_min(0.0)
        total = weights.sum()
        if total <= 0:
            # Narrower than one STFT bin -- the bottom-octave case the docstring
            # names.  Fall back to the nearest bin so the row is never all zero,
            # which would hand the stack a dead input channel.
            nearest = int(torch.argmin((frequencies - centre).abs()))
            weights = torch.zeros(bins)
            weights[nearest] = 1.0
            total = 1.0
        bank[index] = weights / total
    return bank


class ChouwaSubBandDiscriminator(nn.Module):
    """Sub-band waveform discriminator in the spirit of Avocodo's SBD.

    A fixed PQMF bank decimates the waveform into ``num_bands`` critically
    sampled channels, and the convolutions then run *across* those channels.
    That makes inter-band consistency the thing being judged, which is what
    exposes the aliasing an upsampling generator folds back into neighbouring
    bands -- a failure mode neither the period branches (single broadband
    trace) nor the STFT branches (bands treated independently) are shaped to
    catch.  Because the bank is critically sampled the whole stack runs at
    1/num_bands of the sample rate, so the branch is nearly free.
    """

    def __init__(
        self,
        num_bands: int = 8,
        taps: int = 62,
        channels: tuple[int, ...] = (64, 128, 192),
        kernel_size: int = 5,
        stride: int = 3,
        use_san: bool = True,
    ):
        super().__init__()
        norm = _norm_layer
        self.use_san = bool(use_san)
        self.num_bands = int(num_bands)
        self.register_buffer("pqmf", _pqmf_filters(self.num_bands, taps), persistent=True)
        stack_channels = (self.num_bands,) + tuple(channels)
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for in_ch, out_ch in zip(
                    stack_channels[:-1], stack_channels[1:], strict=True
                )
            ]
        )
        self.conv_post = SANConv1d(channels[-1], 1, 3, padding=1) if self.use_san else norm(
            nn.Conv1d(channels[-1], 1, 3, padding=1)
        )
        self.activation = nn.LeakyReLU(0.1)

    def analysis(self, x: Tensor) -> Tensor:
        # Striding by num_bands decimates inside the convolution instead of
        # computing every sample and throwing most of them away.
        return F.conv1d(
            x.float(), self.pqmf, stride=self.num_bands,
            padding=self.pqmf.shape[-1] // 2,
        )

    def forward(self, x: Tensor, san_training=False):
        fmap = []
        x = self.analysis(x)
        for conv in self.convs:
            x = self.activation(conv(x))
            fmap.append(x)
        x = self.conv_post(x, san_training=san_training) if self.use_san else self.conv_post(x)
        if san_training and self.use_san:
            function_output, direction_output = x
            fmap.append(function_output)
            return [torch.flatten(function_output, 1, -1), torch.flatten(direction_output, 1, -1)], fmap
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class ChouwaGANDiscriminator(nn.Module):
    uses_branchwise_r1 = True

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 44100,
        use_san: bool = True,
        periods: tuple[int, ...] = PERIODS,
        spectrogram_channels: tuple[int, ...] = (32, 64, 96),
        spectrogram_compression: float = 0.3,
        spectrogram_specs: tuple[dict, ...] = SPECTROGRAM_SPECS,
        use_subband: bool = False,
        subband_bands: int = 8,
        subband_channels: tuple[int, ...] = (64, 128, 192),
        use_cqt: bool = False,
        cqt_bins_per_octave: int = 16,
        cqt_bins: int = 128,
        cqt_f_min: float = 55.0,
        cqt_channels: tuple[int, ...] = (64, 128, 192),
        **_: object,
    ):
        super().__init__()
        if int(sample_rate) != 44100:
            raise ValueError("ChouwaGAN only supports 44.1 kHz configurations.")
        self.use_checkpointing = bool(use_checkpointing)
        self.use_san = bool(use_san)
        self.force_fp32 = self.use_san
        period_branches = [
            ChouwaPeriodDiscriminator(
                period,
                use_spectral_norm=False,
                use_san=self.use_san,
            )
            for period in periods
        ]
        spectrogram_branches = [
            ChouwaSpectrogramDiscriminator(
                int(spec["n_fft"]),
                int(spec["hop_length"]),
                int(spec["n_fft"]),
                use_spectral_norm=False,
                use_san=self.use_san,
                channels=tuple(spec.get("channels") or spectrogram_channels),
                compression=float(spec.get("compression", spectrogram_compression)),
                band_edges=tuple(spec.get("band_edges") or FULL_BAND),
                kernels=spec.get("kernels"),
                strides=spec.get("strides"),
            )
            for spec in spectrogram_specs
        ]
        # Appended to the spectrogram family rather than kept apart, because it
        # is one: it owns a fixed unlearned transform, so it wants the same
        # precomputed-input sharing and the same R1 treatment (differentiate the
        # branch, hold the transform outside the graph).  ``_spectrogram_index``
        # covers a contiguous range, which is why this sits here and not after
        # the sub-band branch.
        if use_cqt:
            spectrogram_branches.append(
                ChouwaCQTDiscriminator(
                    sample_rate=int(sample_rate),
                    f_min=float(cqt_f_min),
                    bins_per_octave=int(cqt_bins_per_octave),
                    n_bins=int(cqt_bins),
                    use_san=self.use_san,
                    channels=tuple(cqt_channels),
                    compression=float(spectrogram_compression),
                )
            )
        subband_branches = (
            [
                ChouwaSubBandDiscriminator(
                    num_bands=subband_bands,
                    channels=tuple(subband_channels),
                    use_san=self.use_san,
                )
            ]
            if use_subband
            else []
        )
        self.period_count = len(period_branches)
        self.spectrogram_count = len(spectrogram_branches)
        self.discriminators = nn.ModuleList(
            period_branches + spectrogram_branches + subband_branches
        )
        # Built from what was actually constructed, not from a literal list, so
        # a changed `periods` or a disabled sub-band branch cannot silently
        # mislabel a series in TensorBoard.  Index order matches
        # `self.discriminators`, which is what every per-branch caller uses.
        self.branch_names = tuple(
            [f"period_{branch.period}" for branch in period_branches]
            + [
                f"cqt_{branch.n_bins}"
                if isinstance(branch, ChouwaCQTDiscriminator)
                else f"stft_{branch.n_fft}"
                for branch in spectrogram_branches
            ]
            + [f"subband_{branch.num_bands}" for branch in subband_branches]
        )

    @property
    def num_branches(self) -> int:
        return len(self.discriminators)

    def _spectrogram_index(self, index: int) -> int | None:
        """Map a branch index onto its slot in a precomputed spectrogram list."""
        offset = index - self.period_count
        return offset if 0 <= offset < self.spectrogram_count else None

    def _forward_one(
        self,
        discriminator: nn.Module,
        audio: Tensor,
        spectrogram: Tensor | None = None,
        san_training: bool = False,
    ):
        forward = (
            discriminator.forward_spectrogram
            if spectrogram is not None
            else discriminator
        )
        value = spectrogram if spectrogram is not None else audio
        if self.force_fp32:
            value = value.float()
        if self.training and self.use_checkpointing:
            with torch.autocast(device_type=value.device.type, enabled=False):
                return checkpoint(forward, value, san_training=san_training, use_reentrant=False)
        if self.force_fp32:
            with torch.autocast(device_type=value.device.type, enabled=False):
                return forward(value, san_training=san_training)
        return forward(value, san_training=san_training)

    def prepare_spectrograms(self, audio: Tensor):
        return [
            discriminator.spectrogram(audio)
            for discriminator in self.discriminators[
                self.period_count : self.period_count + self.spectrogram_count
            ]
        ]

    def forward_branch(self, audio: Tensor, branch_index: int):
        discriminator = self.discriminators[int(branch_index)]
        return self._forward_one(discriminator, audio)

    def _forward_audio(self, audio: Tensor, spectrograms=None, san_training=False):
        logits, feature_maps = [], []
        for index, discriminator in enumerate(self.discriminators):
            spectrogram = None
            if spectrograms is not None:
                slot = self._spectrogram_index(index)
                if slot is not None:
                    spectrogram = spectrograms[slot]
            logit, fmap = self._forward_one(discriminator, audio, spectrogram, san_training=san_training)
            logits.append(logit)
            feature_maps.append(fmap)
        return logits, feature_maps

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor | None = None,
        real_spectrograms=None,
        fake_spectrograms=None,
        branch_index: int | None = None,
        pair_batches: bool = False,
        san_training: bool = False,
    ):
        if branch_index is not None:
            return self.forward_branch(y, branch_index)

        if y_hat is None:
            return self._forward_audio(y, real_spectrograms, san_training=san_training)

        if not pair_batches:
            real_logits, real_features = self._forward_audio(y, real_spectrograms, san_training=san_training)
            fake_logits, fake_features = self._forward_audio(
                y_hat,
                fake_spectrograms,
                san_training=san_training,
            )
            return real_logits, fake_logits, real_features, fake_features

        if real_spectrograms is None:
            real_spectrograms = self.prepare_spectrograms(y)
        if fake_spectrograms is None:
            fake_spectrograms = self.prepare_spectrograms(y_hat)

        real_batch_size = y.shape[0]
        paired_audio = torch.cat((y, y_hat), dim=0)
        paired_spectrograms = [
            torch.cat((real, fake), dim=0)
            for real, fake in zip(
                real_spectrograms,
                fake_spectrograms,
                strict=True,
            )
        ]
        paired_logits, paired_features = self._forward_audio(
            paired_audio,
            paired_spectrograms,
            san_training=san_training,
        )
        real_logits = [
            [value[0][:real_batch_size], value[1][:real_batch_size]]
            if isinstance(value, (list, tuple))
            else value[:real_batch_size]
            for value in paired_logits
        ]
        fake_logits = [
            [value[0][real_batch_size:], value[1][real_batch_size:]]
            if isinstance(value, (list, tuple))
            else value[real_batch_size:]
            for value in paired_logits
        ]
        real_features = [
            [value[:real_batch_size] for value in branch]
            for branch in paired_features
        ]
        fake_features = [
            [value[real_batch_size:] for value in branch]
            for branch in paired_features
        ]
        return real_logits, fake_logits, real_features, fake_features

    def enable_compile(self, mode: str = "default") -> bool:
        """Compile the paired real/fake path without wrapping the module.

        Only :meth:`forward` is compiled, and that is deliberate rather than
        incidental.  The R1 penalty reaches its branch through
        :meth:`forward_branch` and differentiates it twice with
        ``create_graph=True``; double-backward through a compiled region is the
        part of this loop most likely to break, and it runs on one branch every
        ``r1_interval`` steps, so there is nothing to win by including it.
        Leaving it out of the compiled callable keeps it eager for free.

        Shapes here are static -- the discriminator only ever sees
        ``segment_size`` samples -- which is why ``dynamic=False`` holds and why
        this succeeds where compiling the variable-length frontend does not.

        Measured on an RTX 5060 at batch 8, forward and backward: 53.65 ms
        eager against 41.99 ms compiled, a 1.28x speedup, with no CUDA graphs
        involved.  That distinction is the point -- ``reduce-overhead`` records
        CUDA graphs and needs a static memory pool this 8 GiB card cannot
        spare, while plain fusion cuts kernel launches without it.
        """
        if getattr(self, "_compile_enabled", False):
            return True

        eager_forward = self.forward
        try:
            compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
        except Exception as error:
            from rvc.train.messages import DISCRIMINATOR_COMPILE_ENABLE_FAILED

            print(DISCRIMINATOR_COMPILE_ENABLE_FAILED, str(error))
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            # ``branch_index`` is the R1 path; keep it on the eager callable.
            if not self.training or compile_failed or kwargs.get("branch_index") is not None:
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                from rvc.train.messages import DISCRIMINATOR_COMPILE_RUNTIME_FAILED

                print(DISCRIMINATOR_COMPILE_RUNTIME_FAILED, str(error))
                return eager_forward(*args, **kwargs)

        self.forward = training_forward
        self._compile_enabled = True
        self._compile_mode = mode
        return True

    def r1_penalty(self, real_audio: Tensor, branch_index: int) -> Tensor:
        """Gradient penalty on the input the branch actually judges.

        For every branch but the spectrogram ones that is the waveform. For a
        spectrogram branch it is the **compressed spectrogram**, and the
        difference is not cosmetic: taking the gradient with respect to the
        waveform instead puts the STFT branches eight orders of magnitude above
        the period branches, while their *discriminative* gradients sit within a
        factor of 1.4 of each other.

        The cause is `d|X|**0.3/d|X|`, which diverges as the magnitude goes to
        zero, and real audio has near-silent bins by the thousand.
        `MAGNITUDE_FLOOR` cannot contain it at any usable value.  But that term
        belongs to a **fixed, unlearned transform**: R1
        exists to keep the *discriminator* from getting too sharp, and there is
        nothing the branch can do about the conditioning of its own front end
        except shrink its weights toward zero -- which is precisely what the
        branches that "read as near-constant" had done.

        Differentiating the branch proper, with the transform held outside the
        graph, restores what R1 is for. The units change -- per spectrogram bin
        rather than per sample -- and that is harmless because the per-branch
        controller normalises against each branch's own discriminative
        gradient, so an absolute scale is never compared across branches. It is
        also cheaper: the double backward no longer runs through `torch.stft`.
        """
        slot = self._spectrogram_index(int(branch_index))
        discriminator = self.discriminators[int(branch_index)]
        if slot is not None:
            source = discriminator.spectrogram(real_audio)
            source = source.detach().requires_grad_(True)
            real_logits, _ = self._forward_one(
                discriminator, real_audio, spectrogram=source
            )
        else:
            source = real_audio.detach().requires_grad_(True)
            real_logits, _ = self._forward_one(discriminator, source)
        score = real_logits.float().mean()
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=source,
            create_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().flatten(1).sum(dim=1).mean()

    def remove_weight_norm(self):
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
