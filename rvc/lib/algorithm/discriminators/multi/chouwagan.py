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

#: Default: a single full-band stack.
FULL_BAND = ((0.0, 1.0),)

#: Optional split of the frequency axis into sub-bands with independent weights.
#: Sound in principle -- the sparse high bins stop competing with the loud low
#: ones -- but it turns 3 convolutions per depth into 15, and the branch is
#: launch-bound rather than FLOP-bound: measured ~2x slower end to end on an
#: RTX 5060 even at batch 8.  Opt in only with GPU headroom to spare.
SPECTROGRAM_BAND_EDGES = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


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
        self.band_convs = nn.ModuleList(
            nn.ModuleList(
                ChouwaSeparableConv2d(in_ch, out_ch, (2, 2), norm)
                for in_ch, out_ch in zip(
                    stack_channels[:-1], stack_channels[1:], strict=True
                )
            )
            for _ in self.bands
        )
        self.depth = len(stack_channels) - 1
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
        # diverges as the magnitude goes to zero, and the R1 penalty
        # differentiates the branch with respect to the *waveform*, straight
        # through this transform -- with a 1e-7 floor the near-silent bins drove
        # the R1 gradient norm to 20-5000 against an adversarial norm of ~2.
        # 1e-5 is -100 dBFS, below the noise floor of any real recording, and
        # brings R1 back in line with the period branches.
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


class ChouwaSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, norm):
        super().__init__()
        self.depthwise = norm(
            nn.Conv2d(
                in_channels,
                in_channels,
                (3, 5),
                stride,
                padding=(1, 2),
                groups=in_channels,
            )
        )
        self.pointwise = norm(nn.Conv2d(in_channels, out_channels, 1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.pointwise(self.activation(self.depthwise(x))))


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
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        spectrogram_channels: tuple[int, ...] = (32, 64, 96),
        spectrogram_compression: float = 0.3,
        use_subband: bool = False,
        subband_bands: int = 8,
        subband_channels: tuple[int, ...] = (64, 128, 192),
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
                n_fft,
                hop_length,
                n_fft,
                use_spectral_norm=False,
                use_san=self.use_san,
                channels=tuple(spectrogram_channels),
                compression=spectrogram_compression,
            )
            for n_fft, hop_length in (
                (512, 128),
                (1024, 256),
                (2048, 512),
            )
        ]
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

    def r1_penalty(self, real_audio: Tensor, branch_index: int) -> Tensor:
        real_audio = real_audio.detach().requires_grad_(True)
        real_logits, _ = self.forward_branch(real_audio, branch_index)
        score = real_logits.float().mean()
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=real_audio,
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
