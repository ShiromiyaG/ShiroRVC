"""Wavehax (Yoneyama et al., 2024): a 2D ConvNeXt over (bins, frames) makes the
complex spectrogram from the mel and the STFT of a harmonic prior, and an iSTFT
renders it; no layer runs at the audio rate. Ported from
https://github.com/chomeyama/wavehax (MIT, see THIRD_PARTY_NOTICES) in its v2
layout (16 channels, x3, kernel (13, 7)) with the closed-form PCPH prior.
With ``contrast_band_edges``, the network also reads ChouwaGAN's band harmonic
contrast spread over each band's bins, as ChouwaGAN's Wavehax synthesis does:
the mel cannot show the valleys between harmonics above ~6 kHz, which Wavehax
alone filled. It trains against ChouwaGAN's discriminator.

With ``ddsp``, a small frame-rate head predicts a log gain per band for the
PCPH harmonics and for white noise, their shaped spectra enter as two more
planes, and ``forward(..., return_aux=True)`` returns the rendered audio for
its own mel loss. It is an input only: the Wavehax output stays as it was.
"""

import math

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

from rvc.rectified.chouwagan import (
    CONTRAST_SCALE,
    SOURCE_F0_FLOOR,
    STFT,
    BandContrast,
    ConvNeXtBlock2d,
    LayerNorm2d,
    band_centers,
    band_matrix,
    continuous_f0,
    known_contrast,
)

#: ``architecture`` of a Wavehax rectified vocoder export, and of its runs.
ARCHITECTURE = "wavehax"
#: Under this, the closed-form sum's denominator is taken as zero.
PRIOR_EPSILON = 1e-6
#: Range of the DDSP head's per-band log gains.
DDSP_GAIN_RANGE = (-12.0, 6.0)


def pcph_sources(f0, hop_length, sample_rate, noise=0.01, power=0.1, max_hz=None):
    """The harmonic and white-noise sources of Wavehax's closed-form
    pseudo-constant-power prior: f0 [B, frames] (0 unvoiced) -> [B, samples]
    each, whose sum is ``pcph_prior``."""
    f0 = F.interpolate(f0.unsqueeze(1), scale_factor=hop_length, mode="linear", align_corners=False).squeeze(1)
    white = torch.randn_like(f0) * noise
    if not (f0 > 0).any():
        return torch.zeros_like(f0), white
    increment = f0.double() / sample_rate
    increment[:, :1] += torch.rand(f0.shape[0], 1, device=f0.device, dtype=torch.float64)
    phase = torch.fmod(torch.cumsum(increment, dim=1) * 2 * math.pi, 2 * math.pi).float()
    count = torch.floor((max_hz or sample_rate / 2) / f0.clamp(min=1e-5))
    # sum_k sin(k * phase), whose limit where sin(phase / 2) vanishes is 0.
    denominator = 2 * torch.sin(phase / 2)
    singular = denominator.abs() <= PRIOR_EPSILON
    harmonics = (torch.cos(phase / 2) - torch.cos((count + 0.5) * phase)) / torch.where(singular, 1.0, denominator)
    harmonics = harmonics.masked_fill(singular, 0.0)
    amplitude = power * torch.sqrt(2 / count.clamp(min=1)) * (f0 > 0)
    return harmonics * amplitude, white


def pcph_prior(f0, hop_length, sample_rate, noise=0.01, power=0.1, max_hz=None):
    """Wavehax's closed-form pseudo-constant-power harmonics: f0 [B, frames]
    (0 unvoiced) -> [B, frames * hop_length], every harmonic of f0 up to
    ``max_hz`` (Nyquist) at ``power * sqrt(2 / count)`` from a random initial
    phase, plus white noise of ``noise`` RMS."""
    harmonics, white = pcph_sources(f0, hop_length, sample_rate, noise, power, max_hz)
    return harmonics + white


class DDSPGains(nn.Module):
    """Per-frame log gains for a harmonic and a noise source, one per band of
    ``band_centers`` and interpolated to the STFT's bins by ``band_bins``.
    The gain and input layers start at zero, so the render starts as the plain
    prior and leaves the trunk unchanged while it learns."""

    def __init__(self, num_mels, sample_rate, n_fft, contrast_bands=0, channels=32,
                 band_min_hz=100.0, band_ratio=0.15, model_channels=16):
        super().__init__()
        self.contrast_bands = int(contrast_bands)
        centers = band_centers(sample_rate, float(band_min_hz), float(band_ratio))
        self.bands = len(centers)
        self.register_buffer("band_bins", band_matrix(centers, n_fft, sample_rate).float(), persistent=False)
        self.embed = nn.Sequential(
            nn.Conv1d(num_mels + 2 + self.contrast_bands, int(channels), 7, padding=3, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(int(channels), int(channels), 5, padding=2, padding_mode="replicate"),
            nn.GELU(),
        )
        self.head = nn.Conv1d(int(channels), 2 * self.bands, 1)
        # The rendered planes enter the trunk through their own linear map, so
        # they never change the main input projection's shape.
        self.out = nn.Conv2d(2, int(model_channels), 1, bias=False)

    def forward(self, mel, f0, contrast=None):
        """``mel`` [B, n_mels, T], ``f0`` [B, T] and the vocoder's contrast, if
        any -> the harmonic and noise log gains [B, bands, T] each."""
        cf0 = continuous_f0(f0.float())
        inputs = [
            mel.float(),
            torch.log(cf0.clamp(min=SOURCE_F0_FLOOR) / 200.0).unsqueeze(1),
            (f0 > 0).float().unsqueeze(1),
        ]
        if self.contrast_bands:
            if contrast is None:
                contrast = mel.new_zeros(mel.shape[0], self.contrast_bands, mel.shape[-1])
            inputs.append(contrast.float())
        gains = self.head(self.embed(torch.cat(inputs, dim=1))).clamp(*DDSP_GAIN_RANGE)
        return gains.chunk(2, dim=1)

    def project(self, first, second):
        """The two rendered planes -> [B, channels, bins, T]."""
        return self.out(torch.stack((first, second), dim=1))


class Wavehax(nn.Module):
    """Normalised log mel [B, n_mels, T], f0 [B, T] (0 unvoiced) and, with
    ``contrast_band_edges``, optionally ``measure_contrast``'s [B, bands + 1, T]
    -> [B, 1, T * hop_length].

    ``n_fft`` defaults to twice the hop, as every official layout. With
    ``use_logmag_phase``, the prior enters and the output leaves as log
    magnitude and phase instead of real and imaginary parts.

    With ``mel_prior`` (FreeV's idea), the prior also enters shaped by the
    mel: each frame of its STFT at unit RMS, times the magnitude the mel's
    pseudo-inverse gives per bin. It is an input only, so it speeds the start
    without binding the output. It needs the mel's ``mel_n_fft``,
    ``mel_win_length``, ``mel_fmin``, ``mel_fmax`` and normalisation
    (``mel_mean``, ``mel_std``), which the trainer fills in.

    With ``ddsp``, ``DDSPGains`` shapes the harmonic and noise sources per
    band, their spectra enter as two more planes, and ``return_aux`` gives
    the rendered audio for the trainer's own mel loss."""

    #: The input projection reads the prior's two spectra, their own
    #: projections and the mel: five planes, then the mel prior's two, then
    #: the contrast's two.
    BASE_PLANES = 5
    #: The mel prior enters as its real and imaginary parts.
    MEL_PRIOR_PLANES = 2

    def __init__(self, sample_rate, num_mels, hop_length, n_fft=None, channels=16, mult_channels=3,
                 kernel_size=(13, 7), num_blocks=8, prior_noise=0.01, prior_power=0.1, prior_max_hz=None,
                 use_logmag_phase=False, contrast_band_edges=None, mel_prior=False, mel_n_fft=None,
                 mel_win_length=None, mel_fmin=0.0, mel_fmax=None, mel_mean=0.0, mel_std=1.0,
                 ddsp=False, ddsp_channels=32, ddsp_band_min_hz=100.0, ddsp_band_ratio=0.15):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft or 2 * hop_length)
        self.bins = self.n_fft // 2 + 1
        self.prior_noise, self.prior_power = float(prior_noise), float(prior_power)
        self.prior_max_hz = None if prior_max_hz is None else float(prior_max_hz)
        self.use_logmag_phase = bool(use_logmag_phase)
        self.stft = STFT(self.n_fft, self.hop_length)
        self.prior_proj = nn.Conv1d(self.bins, self.bins, 7, padding=3, padding_mode="reflect")
        self.cond_proj = nn.Conv1d(num_mels, self.bins, 7, padding=3, padding_mode="reflect")
        planes = 5
        self.contrast = None
        if contrast_band_edges:
            self.contrast = BandContrast(sample_rate, hop_length, contrast_band_edges)
            freqs = torch.linspace(0, sample_rate / 2, self.bins)
            edges = torch.tensor([float(hz) for hz in contrast_band_edges])
            members = F.one_hot(torch.bucketize(freqs, edges), len(edges) + 1).T.float()
            self.register_buffer("contrast_members", members, persistent=False)
            # The contrast over its bands' bins and whether it is known.
            planes += 2
        self.mel_prior = bool(mel_prior)
        if self.mel_prior:
            self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
            mel_n_fft = int(mel_n_fft or self.n_fft)
            basis = librosa_mel_fn(sr=sample_rate, n_fft=self.n_fft, n_mels=num_mels, fmin=mel_fmin,
                                   fmax=mel_fmax or sample_rate / 2)
            inverse = torch.linalg.pinv(torch.from_numpy(basis).double())
            # The mel's bins are denser by mel_n_fft / n_fft and its window
            # sums mel_win_length / n_fft of this STFT's: back to this STFT's magnitudes.
            inverse = inverse * (self.n_fft / mel_n_fft) * (self.n_fft / int(mel_win_length or mel_n_fft))
            self.register_buffer("mel_inverse", inverse.float(), persistent=False)
            planes += 2
        self.input_proj = nn.Conv2d(planes, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock2d(channels, mult_channels, kernel_size, 1 / num_blocks) for _ in range(num_blocks)
        )
        self.output_norm = LayerNorm2d(channels)
        self.output_proj = nn.Conv2d(channels, 2, 1)
        # After every base module, so toggling it on/off only adds or drops
        # parameters at the end and a run's optimiser state stays aligned.
        self.ddsp = None
        if ddsp:
            self.ddsp = DDSPGains(
                num_mels, sample_rate, self.n_fft,
                contrast_bands=self.contrast.bands + 1 if self.contrast is not None else 0,
                channels=ddsp_channels, band_min_hz=ddsp_band_min_hz, band_ratio=ddsp_band_ratio,
                model_channels=channels,
            )
        self.apply(self._init_weights)
        if self.ddsp is not None:
            # Gains at zero and no input contribution: the DDSP branch starts
            # as the plain prior and leaves the output unchanged while it learns.
            nn.init.zeros_(self.ddsp.head.weight)
            nn.init.zeros_(self.ddsp.head.bias)
            nn.init.zeros_(self.ddsp.out.weight)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The DDSP branch is optional: an export from before it loads it fresh,
        # and one with it loads into a model without it.
        if self.ddsp is not None:
            for key, value in self.ddsp.state_dict().items():
                state_dict.setdefault(f"{prefix}ddsp.{key}", value.detach().clone())
        else:
            for key in [key for key in state_dict if key.startswith(prefix + "ddsp.")]:
                del state_dict[key]
        # The mel prior enters as two more planes, so it changes the input
        # projection's width. A run that turned it off keeps the weights of
        # the other planes, and one that turned it on starts its own fresh.
        key = f"{prefix}input_proj.weight"
        weight = state_dict.get(key)
        if weight is not None:
            state_dict[key] = self.adapt_plane_weight(weight)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def plane_layout(self):
        """The input projection's plane groups in this model: the base five,
        the mel prior's two when it has one, and the contrast's two when it
        has one."""
        return (
            self.BASE_PLANES,
            self.MEL_PRIOR_PLANES if self.mel_prior else 0,
            2 if self.contrast is not None else 0,
        )

    def _saved_plane_layout(self, saved_width):
        """The saved width's plane groups: the mel prior's two and the
        contrast's two, read as the contrast being the same as this model's.
        A width that fits neither is refused rather than guessed at."""
        base, _, contrast = self.plane_layout()
        saved_contrast = contrast if saved_width >= base + contrast else 0
        rest = saved_width - base - saved_contrast
        if rest == 0:
            saved_mel = 0
        elif rest == self.MEL_PRIOR_PLANES:
            saved_mel = self.MEL_PRIOR_PLANES
        else:
            raise RuntimeError(
                f"A checkpoint's input projection has {saved_width} planes, which this model cannot read: it "
                f"reads {base} base, {self.MEL_PRIOR_PLANES} mel prior and {contrast} contrast."
            )
        return saved_mel, saved_contrast

    def adapt_plane_weight(self, weight, fill=None):
        """``input_proj``'s weight from another run's plane layout, in this
        model's: the mel prior's two planes are dropped when this model has
        none, and take their place when it has them. ``fill`` is what a new
        plane starts from: the model's own weights by default, and zeros for
        Adam's moments."""
        base, mel, contrast = self.plane_layout()
        if weight.shape[1] == base + mel + contrast:
            return weight
        saved_mel, saved_contrast = self._saved_plane_layout(weight.shape[1])
        fresh = self.input_proj.weight if fill is None else fill
        parts = [weight[:, :base]]
        if mel:
            if saved_mel:
                parts.append(weight[:, base:base + self.MEL_PRIOR_PLANES])
            else:
                parts.append(fresh[:, base:base + self.MEL_PRIOR_PLANES])
        if contrast:
            if saved_contrast:
                start = base + saved_mel
                parts.append(weight[:, start:start + contrast])
            else:
                start = base + mel
                parts.append(fresh[:, start:start + contrast])
        return torch.cat(parts, dim=1)

    def compile_blocks(self) -> None:
        for block in self.blocks:
            block.compile()

    def measure_contrast(self, audio, f0):
        """The ``contrast`` input from the audio itself ([B, 1, T] or [B, T])."""
        return known_contrast(self.contrast(audio, f0) / CONTRAST_SCALE)

    def _contrast_planes(self, contrast, batch, frames, like):
        bands = self.contrast.bands
        if contrast is None:
            contrast = like.new_zeros(batch, bands + 1, frames)
        contrast = contrast.float()
        spread = torch.einsum("bnt,nf->bft", contrast[:, :bands], self.contrast_members)
        known = contrast[:, bands:].unsqueeze(2).expand(-1, -1, self.bins, -1)
        return [spread.unsqueeze(1), known]

    def forward(self, mel, f0, contrast=None, return_aux=False):
        device = mel.device.type
        with torch.no_grad():
            harmonics, white = pcph_sources(f0.float(), self.hop_length, self.sample_rate,
                                            self.prior_noise, self.prior_power, self.prior_max_hz)
            prior = harmonics + white
        # The transforms and projections in FP32: the few weights of the 1x1
        # ones take the whole spectrogram's gradient.
        with torch.autocast(device, enabled=False):
            with torch.no_grad():
                spec = self.stft(prior)
                if self.ddsp is not None:
                    harmonic_spec = self.stft(harmonics)
                    noise_spec = spec - harmonic_spec
                if self.use_logmag_phase:
                    first = torch.log(spec.abs().clamp(min=1e-5))
                    second = spec.angle()
                else:
                    first, second = spec.real, spec.imag
                if self.mel_prior:
                    magnitude = (self.mel_inverse @ torch.exp(mel.float() * self.mel_std + self.mel_mean)).clamp(
                        min=1e-5
                    )
                    unit = spec * spec.abs().square().mean(1, keepdim=True).clamp(min=1e-12).rsqrt()
                    shaped = unit * magnitude
                    if self.use_logmag_phase:
                        shaped = [torch.log(shaped.abs().clamp(min=1e-5)), shaped.angle()]
                    else:
                        shaped = [shaped.real, shaped.imag]
            ddsp_audio = None
            if self.ddsp is not None:
                harmonic_gain, noise_gain = self.ddsp(mel, f0, contrast)
                harmonic_gain = torch.exp(torch.einsum("bct,cf->bft", harmonic_gain, self.ddsp.band_bins))
                noise_gain = torch.exp(torch.einsum("bct,cf->bft", noise_gain, self.ddsp.band_bins))
                render = harmonic_spec * harmonic_gain + noise_spec * noise_gain
                if return_aux:
                    ddsp_audio = self.stft.inverse(render.real, render.imag)
                if self.use_logmag_phase:
                    ddsp_first = torch.log(render.abs().clamp(min=1e-5))
                    ddsp_second = render.angle()
                else:
                    ddsp_first, ddsp_second = render.real, render.imag
            planes = [first, second, self.prior_proj(first), self.prior_proj(second), self.cond_proj(mel.float())]
            if self.mel_prior:
                planes += shaped
            planes = [plane.unsqueeze(1) for plane in planes]
            if self.contrast is not None:
                planes += self._contrast_planes(contrast, mel.shape[0], spec.shape[-1], first)
            x = self.input_proj(torch.cat(planes, dim=1))
            if self.ddsp is not None:
                x = x + self.ddsp.project(ddsp_first, ddsp_second)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        with torch.autocast(device, enabled=False):
            x = self.output_proj(x.float())
            if self.use_logmag_phase:
                magnitude = x[:, 0].exp().clamp(max=1e2)
                audio = self.stft.inverse(magnitude * torch.cos(x[:, 1]), magnitude * torch.sin(x[:, 1]))
            else:
                audio = self.stft.inverse(x[:, 0], x[:, 1])
        if return_aux:
            return audio, ({"ddsp": ddsp_audio} if ddsp_audio is not None else {})
        return audio
