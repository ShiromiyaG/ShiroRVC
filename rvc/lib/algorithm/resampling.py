"""Fixed windowed-sinc resamplers for the decoder's up and down paths.

``rolloff`` is the fraction of the stage's Nyquist the filter keeps, so
``1 - rolloff`` is the whole transition band -- and width, rolloff and beta are
one design, not three knobs.  At 0.95 the band is 5% wide, which no kernel of
these lengths can realise: measured rejection of the first image at 95% of
Nyquist is 9 dB.  ``width=12, rolloff=0.88, beta=6.0`` buys 50 dB at the same
point for 73 taps and 15% of the band.
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


# ``filter_beta`` has no default on purpose.  It was ``14.0`` -- a ~130 dB
# stopband, which is a design for a *long* filter: it spends every tap on
# stopband depth and leaves a transition band far too wide for the 17 taps
# ``filter_width = 4`` actually builds.  Measured at that length it was the
# worse choice on every axis: 2.23 dB of passband ripple, -3 dB already at
# 17.3 kHz, 11.9 dB of alias rejection at 19 kHz.  Every live call site passes
# its own value, so the default was dead code that documented a bad number --
# and width, rolloff and beta are one design, so a caller that supplies two of
# the three and inherits the third has not chosen a design at all.


def lowpass_kernel(
    factor: int,
    width: int,
    rolloff: float,
    filter_beta: float,
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
        width: int,
        rolloff: float,
        filter_beta: float,
        stride: int = 1,
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
        filter_beta: float,
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
        # The kernel is a symmetric windowed sinc of odd length, so its own
        # group delay is ``(kernel_size - 1) / 2`` samples at the *output*
        # rate, and that is what the transposed convolution's ``padding`` has
        # to crop for input sample ``n`` to land on output ``n * factor``.
        #
        # This was ``(kernel_size - factor) // 2`` until 2026-09-03, which is
        # short by ``factor // 2``, so every instance delayed its output by
        # that much: measured with an impulse, +1 at x2 and x3, +2 at x4 and
        # x5, +3 at x7, +4 at x8, at every filter width.  The decimator on the
        # other side is delay-free (torchaudio's resample compensates its
        # own), so the trunk arrived late against the ``downs[]`` skips it is
        # concatenated with -- 5.31 ms, 170 samples at 32 kHz, summed over the
        # four ``[5, 4, 4, 4]`` stages.
        #
        # Fixing it changes what the decoder computes, so a checkpoint trained
        # against the old offset renders *worse* under the new one: the
        # weights encoded the misalignment, and on ``G_35332`` the correction
        # cost 0.9 dB of envelope ripple and added 6.4 dB of ``(k + 1/2) * f0``
        # energy -- period doubling, from residual adds that reinforce and
        # cancel on alternate pitch periods.  So this is a pretrain-from-zero
        # change, not a patch.
        #
        # It is *only* an alignment fix, and the impulse test above is the
        # whole case for it.  Paired across 8 seeds on an untrained decoder at
        # constant f0, old against new measures +0.09 +- 0.20 dB of envelope
        # ripple and -0.09 +- 0.13 dB of ``(k + 1/2) * f0`` -- both zero, and
        # 4/8 seeds each way.  A single-seed comparison suggests otherwise and
        # is wrong: seed spread on that metric is 3.0-5.6 dB.
        self.pad_left = self.pad * self.factor + (kernel_size - 1) // 2

        # Everything the polyphase forward needs about padding is a constant,
        # so it is computed here rather than from ``x.shape[-1]``.  It was
        # computed there, and under ``torch.compile`` that makes the pad widths
        # symbolic and ``if left or right:`` a data-dependent branch -- the
        # class of guard that fails as "vr must not be None for symbol ...".
        # The length cancels out of the right-hand pad:
        #
        #     right = max(0, max(shift) + L - (L + 2*pad + 1))
        #           = max(0, max(shift) - 2*pad - 1)
        taps = -(-kernel_size // self.factor)
        whole, offset = divmod(self.pad_left, self.factor)
        shifts = tuple(
            whole + (1 if phase + offset >= self.factor else 0)
            for phase in range(self.factor)
        )
        self.taps = taps
        self.shifts = shifts
        self.phase_offset = offset
        self.extra_left = max(0, taps - 1 - min(shifts))
        self.extra_right = max(0, max(shifts) - 2 * self.pad - 1)
        self.starts = tuple(
            self.extra_left + shift - taps + 1 for shift in shifts
        )

    def _polyphase(self, x: Tensor):
        """The kernel split into ``factor`` phases, cached per device/dtype.

        A transposed convolution of stride ``F`` with a ``K``-tap kernel *is*
        ``F`` convolutions of ``K/F`` taps whose outputs interleave -- the same
        multiplies, none of them at the output rate.  Grouped
        ``conv_transpose1d`` is the slow path in cuDNN and grouped ``conv1d``
        is not, so this is 5-14x faster for a bit-comparable result (relative
        error 4e-7 against the transposed form at every shape in the decoder).

        With ``pad_left = a*F + b``, the tap index ``qF + p - nF + pad_left``
        splits into phase ``(p + b) mod F`` and tap ``q - n + a`` (plus one
        when ``p + b >= F``), which is where ``shifts`` comes from.
        """

        channels = int(x.shape[1])
        key = (channels, x.dtype, x.device)
        if getattr(self, "_poly_key", None) != key:
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)[0, 0]
            taps = self.taps
            weight = kernel.new_zeros(self.factor, 1, taps)
            for phase in range(self.factor):
                index = (phase + self.phase_offset) % self.factor
                part = kernel[index :: self.factor]
                # ``w[taps-1-j] = phase[j]``, so a phase shorter than ``taps``
                # -- which happens whenever ``K`` is not a multiple of
                # ``factor`` -- is right-aligned.  Left-aligning it shifts that
                # phase alone by one sample, which barely moves a spectrum and
                # is plainly wrong in an A/B.
                weight[phase, 0, taps - part.numel() :] = part.flip(-1)
            with cache_scope():
                self._poly_cache = weight.repeat(channels, 1, 1).contiguous()
            self._poly_key = key
        return self._poly_cache

    def _transposed(self, x: Tensor) -> Tensor:
        """The original formulation, kept for ``torch.compile``.

        Identical output -- 4e-7 relative against the polyphase form at every
        shape in the decoder, pinned in
        ``test_the_polyphase_upsample_matches_the_transposed_form`` -- and
        about 4x slower in eager, which is why it is not the default.
        """

        channels, length = x.shape[1], x.shape[-1]
        key = (channels, x.dtype, x.device)
        if getattr(self, "_dense_key", None) != key:
            with cache_scope():
                self._dense_cache = (
                    self.kernel.to(device=x.device, dtype=x.dtype)
                    .expand(channels, -1, -1)
                    .contiguous()
                )
            self._dense_key = key
        padded = F.pad(x, (self.pad, self.pad + 1), mode="replicate")
        out = F.conv_transpose1d(
            padded,
            self._dense_cache,
            stride=self.factor,
            padding=self.pad_left,
            groups=channels,
        )
        return out[..., : length * self.factor]

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        # Interleaving ``factor`` phases needs a modular index -- ``i // F`` and
        # ``i % F`` -- however it is written: a strided store, a ``stack`` plus
        # ``reshape``, a transpose, all of them.  Inductor's
        # ``sizevars.visit_modular_indexing`` crashes on the one this builds,
        # with ``AssertionError: vr must not be None for symbol q1`` from
        # ``_maybe_evaluate_static_worker`` -- an internal assertion about a
        # symbol with no value range, not something callers can shape around.
        # It took the whole decoder's compile down with it.
        #
        # So: the fast form in eager, the transposed form under compile.  The
        # branch is on ``torch.compiler.is_compiling()``, which Dynamo folds at
        # trace time, so the compiled graph contains no branch and eager pays
        # no test.  The two agree to 4e-7.
        if torch.compiler.is_compiling():
            return self._transposed(x)

        batch, channels, length = x.shape[0], x.shape[1], x.shape[-1]
        weight = self._polyphase(x)

        # The replicate pad is the same one the transposed form uses: the extra
        # input sample on the right is what makes the output reach
        # ``length * factor``, and replicating is what keeps the filter from
        # inventing an edge.  The widths are constants -- see ``extra_left``.
        padded = F.pad(
            x,
            (self.pad + self.extra_left, self.pad + 1 + self.extra_right),
            mode="replicate",
        )
        phases = F.conv1d(padded, weight, groups=channels)
        phases = phases.view(batch, channels, self.factor, -1)
        out = x.new_empty(batch, channels, length * self.factor)
        for phase, start in enumerate(self.starts):
            out[..., phase :: self.factor] = phases[
                :, :, phase, start : start + length
            ]
        return out


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


class AntiAliasedActivation(nn.Module):
    """A pointwise nonlinearity evaluated at twice the rate, then filtered back.

    The upsamplers fix the *images* a stage stamps into its output; they cannot
    touch the other half.  A pointwise nonlinearity at the stage rate creates
    harmonics above that stage's Nyquist and those fold back inharmonically --
    visible in a render as partials whose vibrato runs *backwards*, since a
    component folded about ``F`` sits at ``2F - k*f0``.  Measured on a real
    render: inter-harmonic peaks tracking f0 at -10.2 Hz/Hz against +1.8 in the
    reference.

    BigVGAN's ``Activation1d``: 2x up, activate, 2x down.  Nothing subtler
    works -- a smooth activation looks like a free win on a single tone
    (``SiLU`` 68 dB better) and is not one on a realistic multi-partial input
    (-20..-26 dB against ``leaky_relu``'s -23.5 and this module's -40.7).  It
    is also not homogeneous, so its edge vanishes as the trunk's amplitude
    grows.  The oversampling is the mechanism, not the curve.

    ``rolloff`` is the trap.  The round trip's filter cuts at ``rolloff`` of the
    *stage's own* Nyquist, so this module is a lowpass as well as an
    anti-aliaser, and at the 0.90 it was pinned to until 2026-09-03 that
    lowpass cost more than the aliasing it removed.  Measured against a
    reference built by running ``leaky_relu`` at 32x and band-limiting back
    down -- a comparison that needs no trained weights -- on a 1/k harmonic
    series filling 90% of the band, error against that reference as a fraction
    of its total energy:

        design            alias    passband    total
        plain leaky_relu  -38.0      -65.0     -38.0
        f2 w6  r0.90      -56.6      -31.9     -31.9   <- worse than no filter
        f2 w12 r0.99      -54.3      -49.5     -48.3
        f2 w16 r0.99      -55.3      -52.3     -50.5   <- the defaults here
        f2 w16 r1.00      -54.7      -54.0     -51.3
        f4 w16 r0.99      -62.2      -52.3     -51.8

    So the old design bought 18.6 dB of alias rejection and paid 33 dB of band
    edge for it; the two error kinds are not equivalent -- a fixed lowpass is
    something the trunk can learn around and an inharmonic fold is not -- but
    at ``r0.90`` on the 8 kHz down-loop site it was discarding 3.6-4 kHz of the
    excitation outright, and raising the rolloff is nearly free.

    ``factor`` is where the alias floor comes from, not the filter: at 2x it
    sits at about -55 dB whatever the width, because ``leaky_relu`` generates
    products of every order and only the second-order ones fit under 2x.  4x
    buys 7 dB more and costs 40% of the decoder step, so it stays at 2.

    Cost is mostly the 2x tensor, not the taps: at batch 8 / 0.4 s / 32 kHz,
    forward+backward over the whole decoder with every loop rate protected is
    110.7 ms at ``w6 r0.90`` and 129.9 ms at ``w16 r0.99``, against 98.3 ms
    with the activations raw -- and the width difference between 49 and 65 taps
    is inside the +-15 ms noise of that measurement.

    Both resamplers' kernels are non-persistent, so wrapping an activation adds
    no state-dict key -- which is why ``rvc.train.utils.decoder_layout`` writes
    the layout into the checkpoint.
    """

    def __init__(
        self,
        activation: nn.Module | None = None,
        *,
        leaky_relu_slope: float = 0.2,
        factor: int = 2,
        filter_width: int = 16,
        rolloff: float = 0.99,
        filter_beta: float = 6.0,
    ):
        super().__init__()
        self.factor = int(factor)
        self.activation = (
            nn.LeakyReLU(leaky_relu_slope) if activation is None else activation
        )
        # Neither resampler's kernel is a state-dict key, so the design leaves
        # no trace in the weights -- and it decides what the decoder computes.
        # ``rvc.train.utils.decoder_layout`` reads this off the built module so
        # the layout guard cannot drift from what was actually constructed.
        self.design = (
            self.factor,
            int(filter_width),
            float(rolloff),
            float(filter_beta),
        )
        self.up = AntiAliasedUpsample1d(
            self.factor,
            filter_width=filter_width,
            rolloff=rolloff,
            filter_beta=filter_beta,
        )
        self.down = FixedLowPass1d(
            self.factor,
            width=filter_width,
            rolloff=rolloff,
            stride=self.factor,
            filter_beta=filter_beta,
        )

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        # In-place on the upsampler's own output.  That tensor is freshly
        # allocated here and nothing else refers to it, so this is safe for
        # autograd, and it is the largest intermediate in the module: at the
        # 8 kHz res-block shape it is the difference between holding two
        # ``factor``-sized tensors and one.  Only for the stock ``LeakyReLU``,
        # which has an ``inplace`` flag and whose backward needs the *output*;
        # anything else takes the ordinary path.
        upsampled = self.up(x)
        if type(self.activation) is nn.LeakyReLU:
            upsampled = F.leaky_relu(
                upsampled, self.activation.negative_slope, inplace=True
            )
        else:
            upsampled = self.activation(upsampled)
        x = self.down(upsampled)
        # The two resamplers round their padding independently, so the round
        # trip can land a sample or two either side.  Cropping is right and
        # padding is a fallback: an output shorter than the input would break
        # the residual add in the block that owns this.
        if x.shape[-1] > length:
            return x[..., :length]
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1]), mode="replicate")
        return x
