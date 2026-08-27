"""Wavehax's discriminator: UnivNet's multi-resolution spectral plus HiFi-GAN's MPD.

Port of ``wavehax/discriminators/univnet.py`` and ``config/discriminator/
univnet.yaml`` from https://github.com/chomeyama/wavehax -- MIT licensed,
Copyright 2024 Reo Yoneyama (Nagoya University).

It shares its *families* with ``RefineGANDiscriminator`` -- a multi-period
branch over the waveform and a multi-resolution branch over STFT magnitudes --
and almost nothing else.  The differences are the whole reason this file exists
rather than a shared config:

============================  ============================  ============================
                              RefineGAN (this fork)         Wavehax (UnivNet)
============================  ============================  ============================
MPD channel schedule          fixed 64/128/256/512/1024     32, then x4 capped at 1024
MPD downsampling              stride 3 on all five layers   ``[3, 3, 3, 3, 1]``
Leaky ReLU slope              0.2                           0.1
Feature maps                  include the output conv       exclude it
STFT window                   rectangular                   Hann
STFT framing                  ``center=False``, manual pad  ``center=True``
Spectral kernels              ``(3, 9)`` and ``(3, 3)``     ``(7,5) (5,3) (5,3) (3,3)x3``
Spectral strides              ``(1, 2)`` -- time only       ``(2,2) (2,1) ...`` -- both
Spectral channels             constant 32, 5 convs          constant 32, 1x1 in + 6 convs
Resolutions (n_fft/hop/win)   1024/120/600 and friends      1024/256/1024 and friends
============================  ============================  ============================

The two that matter most are the last four rows.  RefineGAN's spectral branch
downsamples *time only*, so it keeps full frequency resolution all the way to
its output and reads a harmonic comb bin by bin.  UnivNet's halves frequency at
five of its six layers, so it ends up judging a coarse time-frequency envelope
instead -- a different question about the same signal, which is the point of
pairing it with a decoder that predicts complex spectrograms directly.

What this port adds
-------------------
Two things upstream has no need for, both required by this fork's training loop
rather than by the architecture:

* The per-branch contract -- R1 on one branch at a time, per-branch loss series,
  the paired real/fake forward, the precomputed spectrograms.  It lives in
  :class:`~rvc.lib.algorithm.discriminators.multi.branchwise.BranchwiseDiscriminator`
  and is shared with RefineGAN, so the spectral branch is split into
  ``spectrogram`` and ``forward_spectrogram``.
* Mixed-precision guards.  Every branch drops non-finite activations between
  layers: a single overflow in FP16 otherwise reaches the *generator* as a NaN
  gradient through feature matching, and a non-finite activation carries no
  adversarial signal worth preserving.  The STFT and its magnitude stay in FP32
  because cuFFT's FP16 path underflows on quiet frames.  On FP32 both are
  identities, so neither changes the published model.

The spectral head's output is flattened, where upstream returns the raw
``(batch, 1, bins, frames)`` map.  Every consumer reduces it with ``mean`` over
all elements, so the value is identical; the flat shape just matches the period
branch and the rest of the loop.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.discriminators.multi.branchwise import BranchwiseDiscriminator

#: ``config/discriminator/univnet.yaml``.
PERIODS = (2, 3, 5, 7, 11)
PERIOD_CHANNELS = 32
PERIOD_KERNEL_SIZES = (5, 3)
PERIOD_DOWNSAMPLE_SCALES = (3, 3, 3, 3, 1)
PERIOD_MAX_CHANNELS = 1024

#: ``(n_fft, hop_length, win_length)`` per resolution.
RESOLUTIONS = (
    (1024, 256, 1024),
    (2048, 512, 2048),
    (512, 128, 512),
)
SPECTRAL_CHANNELS = 32
SPECTRAL_KERNEL_SIZES = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3), (3, 3))
SPECTRAL_STRIDES = ((2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1))

#: UnivNet's, not HiFi-GAN's 0.2.
LRELU_SLOPE = 0.1


class PeriodDiscriminator(nn.Module):
    """Reshape the waveform to ``(time/period, period)`` and judge it in 2D.

    The kernels are ``(k, 1)``: every convolution runs along the *within-period*
    axis and none of them mixes neighbouring periods, so the branch measures how
    well one period predicts the next at exactly this period and nothing else.
    """

    def __init__(
        self,
        period: int,
        channels: int = PERIOD_CHANNELS,
        kernel_sizes: Sequence[int] = PERIOD_KERNEL_SIZES,
        downsample_scales: Sequence[int] = PERIOD_DOWNSAMPLE_SCALES,
        max_channels: int = PERIOD_MAX_CHANNELS,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if len(kernel_sizes) != 2:
            raise ValueError("Wavehax MPD takes a first and a last kernel size.")
        if any(value % 2 == 0 for value in kernel_sizes):
            raise ValueError("Wavehax MPD kernel sizes must be odd.")

        self.period = int(period)
        norm = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, int(channels)
        for scale in downsample_scales:
            self.convs.append(
                norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_sizes[0], 1),
                        (int(scale), 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    )
                )
            )
            in_channels = out_channels
            # x4 per stage until the cap, where UnivNet's MPD is much narrower
            # early and just as wide late as HiFi-GAN's.
            out_channels = min(out_channels * 4, int(max_channels))
        self.conv_post = norm(
            nn.Conv2d(
                in_channels,
                1,
                (kernel_sizes[1], 1),
                padding=((kernel_sizes[1] - 1) // 2, 0),
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, list[Tensor]]:
        batch, channels, length = x.shape
        if length % self.period:
            padding = self.period - (length % self.period)
            x = F.pad(x, (0, padding), "reflect")
            length = length + padding
        x = x.view(batch, channels, length // self.period, self.period)

        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            x = torch.nan_to_num(x)
            feature_maps.append(x)
        # Upstream does not put the output convolution in the feature maps: it
        # is the logit, and matching it would make the feature-matching loss a
        # second, unscaled copy of the adversarial one.
        x = torch.nan_to_num(self.conv_post(x))
        return torch.flatten(x, 1, -1), feature_maps


class SpectralDiscriminator(nn.Module):
    """Judge one STFT magnitude resolution, downsampling both of its axes.

    ``spectrogram`` is deliberately separable from the convolutional head: the
    trainer computes each real resolution once and reuses it across the paired
    forward, and the R1 penalty differentiates the head with the transform held
    outside the graph.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        channels: int = SPECTRAL_CHANNELS,
        kernel_sizes: Sequence[Sequence[int]] = SPECTRAL_KERNEL_SIZES,
        strides: Sequence[Sequence[int]] = SPECTRAL_STRIDES,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        kernel_sizes = tuple(tuple(int(v) for v in k) for k in kernel_sizes)
        strides = tuple(tuple(int(v) for v in s) for s in strides)
        if len(kernel_sizes) != len(strides):
            raise ValueError(
                f"{len(kernel_sizes)} spectral kernels for {len(strides)} strides"
            )

        norm = spectral_norm if use_spectral_norm else weight_norm
        # Hann, where RefineGAN's branch uses a rectangular window.  UnivNet
        # reads a time-frequency envelope rather than individual harmonics, and
        # a taper is what keeps a loud frame from smearing across every bin.
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )

        channels = int(channels)
        self.input_conv = norm(nn.Conv2d(1, channels, 1))
        self.convs = nn.ModuleList(
            norm(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                )
            )
            for kernel_size, stride in zip(kernel_sizes, strides)
        )
        self.conv_post = norm(nn.Conv2d(channels, 1, 1))

    def spectrogram(self, x: Tensor) -> Tensor:
        """``(batch, 1, samples)`` -> ``(batch, bins, frames)`` magnitude."""
        device_type = x.device.type
        autocast_enabled = torch.is_autocast_enabled(device_type=device_type)
        output_dtype = (
            torch.get_autocast_dtype(device_type) if autocast_enabled else x.dtype
        )
        with torch.autocast(device_type=device_type, enabled=False):
            value = torch.stft(
                x.float().squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            magnitude = value.abs()
        if autocast_enabled and output_dtype in (torch.float16, torch.bfloat16):
            if output_dtype == torch.float16:
                # The magnitude is already exact in FP32; saturating only the
                # values FP16 cannot hold keeps the cast itself from creating
                # infinities out of loud frames.
                magnitude = magnitude.clamp(max=torch.finfo(output_dtype).max)
            return magnitude.to(output_dtype)
        return magnitude

    def forward_spectrogram(self, x: Tensor) -> Tuple[Tensor, list[Tensor]]:
        x = F.leaky_relu(self.input_conv(x.unsqueeze(1)), LRELU_SLOPE)
        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            x = torch.nan_to_num(x)
            feature_maps.append(x)
        x = torch.nan_to_num(self.conv_post(x))
        return torch.flatten(x, 1, -1), feature_maps

    def forward(self, x: Tensor) -> Tuple[Tensor, list[Tensor]]:
        return self.forward_spectrogram(self.spectrogram(x))


class WavehaxDiscriminator(BranchwiseDiscriminator):
    """UnivNet's MPD + MRD, over the per-branch contract in ``branchwise``."""

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 44100,
        periods: Sequence[int] = PERIODS,
        period_channels: int = PERIOD_CHANNELS,
        period_kernel_sizes: Sequence[int] = PERIOD_KERNEL_SIZES,
        period_downsample_scales: Sequence[int] = PERIOD_DOWNSAMPLE_SCALES,
        period_max_channels: int = PERIOD_MAX_CHANNELS,
        resolutions: Sequence[Sequence[int]] = RESOLUTIONS,
        spectral_channels: int = SPECTRAL_CHANNELS,
        spectral_kernel_sizes: Sequence[Sequence[int]] = SPECTRAL_KERNEL_SIZES,
        spectral_strides: Sequence[Sequence[int]] = SPECTRAL_STRIDES,
        **_: object,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)

        period_branches = [
            PeriodDiscriminator(
                period,
                channels=period_channels,
                kernel_sizes=period_kernel_sizes,
                downsample_scales=period_downsample_scales,
                max_channels=period_max_channels,
                use_spectral_norm=use_spectral_norm,
            )
            for period in periods
        ]
        spectral_branches = [
            SpectralDiscriminator(
                int(n_fft),
                int(hop_length),
                int(win_length),
                channels=spectral_channels,
                kernel_sizes=spectral_kernel_sizes,
                strides=spectral_strides,
                use_spectral_norm=use_spectral_norm,
            )
            for n_fft, hop_length, win_length in resolutions
        ]
        self.register_branches(
            period_branches,
            spectral_branches,
            [f"period_{branch.period}" for branch in period_branches]
            + [f"stft_{branch.n_fft}" for branch in spectral_branches],
            use_checkpointing=use_checkpointing,
        )
