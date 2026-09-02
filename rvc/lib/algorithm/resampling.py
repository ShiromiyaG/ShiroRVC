"""Fixed windowed-sinc resamplers shared by the anti-aliased vocoders.

These used to live in ``chouwagan.py``.  RefineGAN upsampled its feature maps
with ``nn.Upsample(mode="linear")`` instead, whose triangular kernel rejects the
first spectral image by only 1.7-9.6 dB at the top of the band; both decoders
therefore stamped the frame grid into the waveform as a mirrored image pair
around every harmonic.  One implementation, used by both, is what stops that
from being two separate tuning problems.

**Design note.**  ``rolloff`` is the fraction of the stage's Nyquist the filter
tries to keep, so ``1 - rolloff`` is the whole transition band.  At 0.95 that
band is 5% wide, which no kernel of the lengths used here can realise: the
measured rejection of the first image at 95% of Nyquist is 9 dB, and the images
of a full-band input pass essentially untouched.  Width, rolloff and beta are
one design, not three independent knobs -- ``width=4, rolloff=0.95, beta=1.5``
is the best of a bad set at 17-25 taps, while ``width=12, rolloff=0.88,
beta=6.0`` buys 50 dB at the same point for 73 taps and 15% of the band.  Pick
them together, per stage.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rvc.lib.algorithm.commons import cache_scope


def _safe_pad(x: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return x
    mode = "reflect" if x.shape[-1] > padding else "replicate"
    return F.pad(x, (padding, padding), mode=mode)


#: Kaiser window shape for the resampling filters.  14.0 -- what this was pinned
#: to until 2026-08-31 -- targets a ~130 dB stopband, which is a design for a
#: *long* filter: it spends every tap on stopband depth and leaves a transition
#: band far too wide for the 17 taps ``filter_width = 4`` actually builds.
#:
#: Measured by driving a tone through the snake at the output rate, that pinned
#: value was the worse choice on every axis at this length -- 2.23 dB of
#: passband ripple, -3 dB already at 17.3 kHz, and as little as 11.9 dB of alias
#: rejection at 19 kHz.  That last one matters because ``sin()**2`` generates
#: harmonics of every order, not just the second, so a k-th order product folds
#: to ``|k*f - m*sr|`` -- 19 kHz lands at 6.1 kHz, plainly audible, and shows up
#: as a stationary horizontal line low in the spectrogram.  Around 1.5 the same
#: 17 taps give 34 dB of rejection, 0.83 dB of ripple and 19.7 kHz of bandwidth,
#: for exactly the same arithmetic.
DEFAULT_FILTER_BETA = 14.0


def lowpass_kernel(
    factor: int,
    width: int,
    rolloff: float,
    filter_beta: float = DEFAULT_FILTER_BETA,
) -> Tensor:
    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=float(filter_beta), dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


class FixedLowPass1d(nn.Module):
    def __init__(
        self,
        factor: int,
        width: int = 4,
        rolloff: float = 0.95,
        stride: int = 1,
        filter_beta: float = DEFAULT_FILTER_BETA,
    ):
        super().__init__()
        self.stride = int(stride)
        self.register_buffer(
            "kernel",
            lowpass_kernel(factor, width, rolloff, filter_beta),
            persistent=False,
        )

    def _grouped_kernel(self, x: Tensor) -> Tensor:
        """The per-channel kernel, cached across calls.

        ``.to()`` plus ``.expand()`` ran on every forward of every instance.
        Each is cheap on its own and neither shows up in the GPU trace, but at
        26 instances called several times per step they are pure dispatch on a
        step that is already CPU-bound.  The cache is keyed by the only three
        things that can change it.
        """
        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_kernel_key", None) != key:
            with cache_scope():
                self._kernel_cache = (
                    self.kernel.to(device=x.device, dtype=x.dtype)
                    .expand(channels, -1, -1)
                    .contiguous()
                )
            self._kernel_key = key
        return self._kernel_cache

    def forward(self, x: Tensor) -> Tensor:
        kernel = self._grouped_kernel(x)
        padding = (kernel.shape[-1] - 1) // 2
        return F.conv1d(
            _safe_pad(x, padding),
            kernel,
            stride=self.stride,
            groups=x.shape[1],
        )


class AntiAliasedUpsample1d(nn.Module):
    def __init__(
        self,
        factor: int,
        filter_width: int,
        rolloff: float,
        filter_beta: float = DEFAULT_FILTER_BETA,
    ):
        super().__init__()
        self.factor = int(factor)
        # The ``factor`` gain that compensates the zero-stuffing is folded into
        # the kernel here instead of multiplying the convolution's output.  That
        # output lives at the *upsampled* rate, so ``factor * conv(...)`` was
        # allocating a second full-rate tensor on every call -- 36 MiB at the
        # output stage alone, at batch 8 over 0.4 s.  Exact, and free.
        kernel = lowpass_kernel(self.factor, filter_width, rolloff, filter_beta)
        self.register_buffer("kernel", kernel * self.factor, persistent=False)

        kernel_size = int(kernel.shape[-1])
        self.pad = kernel_size // self.factor - 1
        self.pad_left = self.pad * self.factor + (kernel_size - self.factor) // 2
        self.pad_right = self.pad * self.factor + (kernel_size - self.factor + 1) // 2

    def _grouped_kernel(self, x: Tensor) -> Tensor:
        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_kernel_key", None) != key:
            with cache_scope():
                self._kernel_cache = (
                    self.kernel.to(device=x.device, dtype=x.dtype)
                    .expand(channels, -1, -1)
                    .contiguous()
                )
            self._kernel_key = key
        return self._kernel_cache

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        channels = x.shape[1]
        kernel = self._grouped_kernel(x)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        # ``padding`` crops the transposed convolution's own output instead of
        # computing 43 samples per call and then throwing them away.  It is
        # symmetric, and ``pad_left``/``pad_right`` differ by one whenever
        # ``kernel_size - factor`` is odd -- which it always is here, since the
        # kernel length is odd and the factor even.  So the symmetric part goes
        # to the convolution and only the remaining single sample is trimmed;
        # using ``output_padding`` to absorb it instead would shift the whole
        # signal by one sample at 2x rate, which is a phase change in the
        # anti-aliasing path, not an optimisation.
        x = F.conv_transpose1d(
            x,
            kernel,
            stride=self.factor,
            padding=self.pad_left,
            groups=channels,
        )
        trim = self.pad_right - self.pad_left
        return x if trim == 0 else x[..., :-trim]


def filter_schedule(
    value: "float | Sequence[float]",
    stages: int,
    name: str,
    minimum: float | None = None,
) -> tuple[float, ...]:
    """Normalise a scalar-or-per-stage filter setting into one value per stage.

    Width, rolloff and beta all take either form and all have to agree on
    length, so the check lives in one place rather than three.
    """

    if isinstance(value, (int, float)):
        schedule = (float(value),) * stages
    else:
        schedule = tuple(float(item) for item in value)
    if len(schedule) != stages:
        raise ValueError(
            f"{name} has {len(schedule)} entries for {stages} stages; "
            f"give one per stage or a single value."
        )
    if minimum is not None and any(item < minimum for item in schedule):
        raise ValueError(f"{name} values must be >= {minimum}.")
    return schedule
