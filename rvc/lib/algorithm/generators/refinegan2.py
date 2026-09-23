import math
from contextlib import nullcontext
from typing import Sequence

import numpy as np
import torch
import rvc.lib.torchaudio_guard  # noqa: F401 -- must precede torchaudio
from torchaudio.functional.functional import (
    _apply_sinc_resample_kernel,
    _get_sinc_resample_kernel,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import (
    register_parametrization,
    remove_parametrizations,
)
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import (
    cache_scope,
    expand_f0,
    get_padding,
    init_weights,
)
from rvc.lib.algorithm.resampling import (
    AntiAliasedUpsample1d,
    filter_schedule,
)


#: Interpolation filter for the trunk's upsamplers, one entry per stage.
#:
#: Zero-stuffing by ``factor`` copies the input spectrum to every multiple of
#: the input rate; what the filter leaves of those copies is an *image* at
#: ``|k*R_in +- f|``.  That is imaging rather than aliasing, and only this
#: filter touches it -- but an image at ``k*R_in - j*f0`` moves against f0
#: exactly like a fold does, so the two are indistinguishable in a spectrogram.
#:
#: These were a flat ``12 / 0.90 / 6.0``, which left the worst image at
#: **-37.1 dB** while attenuating the partial that made it by 13.0 dB.
#: Measured on stage 3 with a partial at 0.95 of the input Nyquist,
#: image / passband:
#:
#:     w12 r0.90 b6.0   -37.1 / -13.0    97 taps
#:     w24 r0.95 b6.0   -68.4 /  -6.0   193
#:     w32 r0.97 b6.0   -68.0 /  -2.0   257
#:     w48 r0.99 b9.0   -94.4 /  -0.1   385
#:
#: The last stage's rolloff came down from 0.99 to 0.97 on 2026-09-10.  The row
#: above measures the image of a partial at 0.95 of the input Nyquist, which is
#: deep in the stopband; the image that matters is the one from content just
#: *under* the mirror, which lands just over it, inside the transition band
#: where the design has barely started attenuating.  Read off the kernel's own
#: response, as a fraction of the mirror -- the numbers are the same at any
#: factor, since the kernel is defined in normalised frequency, so at
#: ``[5,4,4,4]``'s last stage the mirror is 4000 Hz and these land at 4050 and
#: 4150:
#:
#:     rolloff   mirror+1.25%   +3.75%   |   -6.25%   -1.25%
#:     0.99         -17.8       -47.1    |   -0.01     -5.22
#:     0.97         -38.8      -127.0    |   -0.42    -14.44
#:     0.95         -90.7       -92.9    |   -2.70    -32.26
#:
#: 21 dB on the near image for 0.4 dB at 6% under the mirror is the trade 0.97
#: takes.  0.95 buys another 52 dB and starts costing real passband, so it is
#: where to go next if the near image is still what a measurement finds.  At
#: ``[5,4,4,4]`` this also lowers the ceiling the last block synthesises from,
#: 3960 -> 3880 Hz, which the overfit probe above says is not binding.
#:
#: Injecting a known sinusoid after each ``upsample_blocks[k]`` and subtracting
#: a clean render gives the path gain from that stage to the output directly:
#:
#:     stage 1 image -37.1 dB, path -59.3 dB  ->  -96.4 dB at the output
#:     stage 2 image -37.1     path -49.5     ->  -86.6
#:     stage 3 image -37.1     path -19.9     ->  -57.0
#:
#: which is why stage 3 was lengthened first.  Stages 1 and 2 follow anyway,
#: because that measurement is of the *linear* path and the images do not only
#: travel it: an image at ``2000 - f`` leaving stage 2 enters stage 3 and meets
#: dozens of ``leaky_relu``, and its second-order product with a strong partial
#: lands at ``2000 - (j+1)*f0`` with an amplitude set by harmonic energy rather
#: than by -49.5 dB of path gain.
#:
#: Stage 0 is left short.  What leaves it still has three stages of filtering
#: ahead of it, and it is the one stage where the input is short enough that a
#: long kernel costs edge: ``AntiAliasedUpsample1d`` pads, so a longer kernel
#: reaches further into an invented continuation.  Fraction of a 0.4 s training
#: segment whose samples that padding corrupts by more than 1%:
#:
#:     stage 0   40 in, 121 taps   21.0%    <- already the worst, left alone
#:     stage 1  200 in, 193 taps    9.2%
#:     stage 2  800 in, 257 taps    3.9%
#:     stage 3 3200 in, 385 taps    0.3%
DEFAULT_UPSAMPLE_WIDTH = (12, 24, 32, 48)
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.95, 0.97, 0.97)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)

#: Frames the excitation gain reads from ``z``.  More than one, so the prior's
#: frame-independent draw is averaged before it modulates the source.
SOURCE_GAIN_KERNEL = 5


#: What ``AdaIN`` does with its noise.  ``always`` is RefineGAN's -- the
#: paper's Figure 1 draws this block as gaussian noise, a learnable weight, the
#: sum and a LeakyReLU, with no mention of dropping it at inference, and the
#: reference implementation adds it unconditionally.
#:
#: ``train`` was this fork's, as an inference saving, and it is the one setting
#: that is not self-consistent: the decoder builds its high-frequency floor out
#: of a noise the render then does not have -- 3.1 dB of 8-13 kHz on a 220k-step
#: checkpoint -- and the discriminator, seeing the same noisy output, never asks
#: for the floor either.
ADAIN_NOISE_MODES = ("always", "train", "off")


def adain_noise_mode(value):
    """``ADAIN_NOISE_MODES`` entry for a config value; bools are the old spelling."""
    if isinstance(value, bool):
        return "train" if value else "off"
    mode = str(value).lower()
    if mode not in ADAIN_NOISE_MODES:
        raise ValueError(
            f"adain_noise must be one of {ADAIN_NOISE_MODES}; received {value!r}."
        )
    return mode


class UnitNorm(nn.Module):
    """Weight norm without the gain: every output channel's filter at unit norm.

    This parametrizes ``conv_post``, and the reason is the gain it removes.
    Under ``weight_norm`` that layer's ``g`` is one scalar standing between the
    whole trunk and the ``tanh``, and every nonlinearity before it is a
    ``leaky_relu``, so the loss sees only the *product* of ``g`` and the
    trunk's scale.  Nothing in the objective decides the split; the optimizer
    decides it.  Each stage's ``input_conv`` is a plain conv, and under Adam a
    plain weight's norm grows by random walk, about ``lr * sqrt(fan_in *
    steps)`` per row -- predicted against measured row norms:

        run                          stage 0         stage 3
        this decoder, 2e-4, 98k     4.2 / 3.27      1.5 / 2.11
        RefineGAN,    1e-4, 53k     1.5 / 1.42      0.54 / 0.47

    Four stages in series multiply that, and ``g`` shrinks to keep the output
    at its level.  The first 32 kHz pretrain ended at ``g = 2.9e-4`` from
    0.574 at init.  ``dL/dg`` grows with the trunk's features, so that one
    scalar carried 99.9% of the generator's gradient norm (~14k, from the
    optimizer's second moment).  BF16 carried it; the first FP16 finetune of
    that pretrain went to NaN.

    With the norm fixed there is no such direction: the trunk's last features
    *are* the output's amplitude, so their scale is something the loss pins
    rather than something the optimizer drifts along.  Only the direction is
    learned.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight / torch.linalg.vector_norm(
            weight, dim=tuple(range(1, weight.dim())), keepdim=True
        )


#: What a ``conv_post`` with a learned gain leaves in a state dict: the new
#: ``weight_norm`` API's gain, then the old one's.
LEGACY_OUTPUT_GAIN_KEYS = (
    "conv_post.parametrizations.weight.original0",
    "conv_post.weight_g",
)


def _refuse_learned_output_gain(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    """Load hook: refuse a decoder trained with a gain on ``conv_post``.

    Such a checkpoint cannot be converted.  Its trunk was trained against that
    gain, so without it the output is ``1 / g`` too loud -- ~3400x on the
    pretrain that motivated ``UnitNorm``.  And inference loads non-strictly, so
    without this the gain key would be dropped as unexpected, ``conv_post``
    left at its random init, and the old trunk's inflated features would drive
    the ``tanh`` into clipping without a single error.
    """

    found = [
        prefix + key
        for key in LEGACY_OUTPUT_GAIN_KEYS
        if prefix + key in state_dict
    ]
    if found:
        error_msgs.append(
            f"'{found[0]}' is a learned output gain, and RefineGAN2's conv_post "
            "no longer has one (unit-norm since 2026-09-18; see UnitNorm). This "
            "decoder was trained against that gain and cannot be converted -- "
            "train a fresh pretrain."
        )


class ResBlock(nn.Module):
    """Residual block of dilated convolutions at multiple dilation rates."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

        # Set by ``apply_precision_policy`` under BF16: an FP32 stream keeps an
        # update smaller than BF16's rounding step from being lost in the sum.
        self.fp32_residuals = False

    def forward(self, x: torch.Tensor):
        if self.fp32_residuals:
            x = x.float()
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            # In place on the conv's own output: autograd keeps one tensor, not
            # two.  The residual add stays out of place (``fp32_residuals``).
            xt = F.leaky_relu(c1(xt), self.leaky_relu_slope, inplace=True)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """RefineGAN's noise-injecting activation.

    There are two of these wrapped around every ``ResBlock`` -- six per
    ``ParallelResBlock``.  ``noise`` is one of ``ADAIN_NOISE_MODES``.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
        noise: str = "always",
    ):
        super().__init__()

        self.noise = adain_noise_mode(noise)
        if self.noise != "off":
            self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        if self.noise == "off" or (self.noise == "train" and not self.training):
            return self.activation(x)

        # In place only on this branch: ``x`` is shared by the parallel blocks,
        # the noisy sum is a fresh tensor.
        noisy = torch.addcmul(x, torch.randn_like(x), self.weight[:, None])
        return F.leaky_relu(noisy, self.activation.negative_slope, inplace=True)


class ParallelResBlock(nn.Module):
    """Runs several ResBlocks (different kernel sizes) in parallel and averages them."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
        adain_noise: str = "always",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.input_conv.apply(init_weights)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                        noise=adain_noise,
                    ),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                        noise=adain_noise,
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        # Summed as they come: ``stack`` held every output plus a copy of all
        # of them before ``mean`` read it back.
        out = self.blocks[0](x)
        for block in self.blocks[1:]:
            out = out + block(x)
        return out / len(self.blocks)


class SineGenerator(nn.Module):
    """Sine + additive-noise harmonic excitation source.

    On a fixed trunk over 3 seeds (held-out multi-scale mel, lower better):
    sine 1.9714 -> **1.7418** with ``source_gain`` on, bank 1.7357 -> 1.8057,
    comb 1.9649 -> 1.8363.

    ``comb`` and ``bank`` were removed on 2026-09-03: neither had beaten the
    sine once the excitation gain was on.  ``excitation_source`` in
    ``rvc/train/utils.py`` names the mismatch if a checkpoint from another
    source is loaded here.

    ``harmonic_num`` (2026-09-09)
    ----------------------------
    At 0 -- the default this shipped with, and Applio's -- the excitation is
    one partial, so *every* harmonic in the output is manufactured by the
    trunk.  This knob was added while chasing renders whose harmonics stopped
    around 6 kHz, on the theory that one partial under the trunk's 3960 Hz
    ceiling left nothing to build on above it.

    It ships at 0 because that theory did not survive measurement: overfitting
    a single clip, this decoder at ``harmonic_num=0`` reproduces a target's
    harmonic contrast to within 0.7 dB all the way to 13 kHz, under either
    stage layout.  See ``RefineGAN2Generator`` for the table.  A source short
    of partials is not what costs a *trained* model its high harmonics, so
    this is an escape hatch rather than a fix -- raising it puts scaffolding
    into the source, where it is alias-free and in tune by construction.

    ``harmonic_tilt`` gives partial ``j`` an amplitude of ``j ** -tilt``, so
    the source arrives with a slope instead of flat.  1.0 is a sawtooth's
    -6 dB/octave, and it lands close to a real voice (levels relative to the
    300-600 Hz band, f0=200):

        band          real voice    tilt 1.0
        1-2 kHz          -5.5         -9.1
        12-15.5 kHz     -33.8        -28.9

    Within ~5 dB across the whole band.  A single power law cannot match both ends -- a voice's slope steepens with
    frequency and this one does not -- which is what the knob is for; 1.17
    matches the top and costs 5 dB at 1-2 kHz.

    And a partial above Nyquist is *removed*, not folded: ``_f02sine`` takes
    ``f / sr`` mod 1, so ``j * f0`` past ``sr / 2`` aliases down onto the band
    as an inharmonic line that walks against f0.  The fade below zeroes it
    over the top ``NYQUIST_TAPER`` of the band rather than switching it off,
    because a hard gate clicks every time f0 drifts across the boundary.

    ``dim`` sizes ``merge.0.weight``, so this is a state-dict shape and a
    fresh pretrain -- and ``decoder_layout`` carries ``harmonic_tilt``, which
    is *not* in the state dict at any count.
    """

    #: Fraction of Nyquist over which a partial fades out instead of being
    #: switched off.  f0 moves between frames, so a hard mask makes the top
    #: partial blink on and off around the boundary -- a click at the frame
    #: rate on exactly the harmonics that are hardest to hear it in.
    NYQUIST_TAPER = 0.1

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        harmonic_tilt=1.0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.harmonic_tilt = float(harmonic_tilt)

        # Non-persistent: ``m_source`` owns exactly one state-dict key and
        # that is pinned by a test.  A buffer that changes the signal and
        # leaves no key is a ``decoder_layout`` field, like the upsamplers'
        # interpolation design -- see ``rvc/train/utils.py``.
        orders = torch.arange(1, self.dim + 1, dtype=torch.float32)
        self.register_buffer("harmonic_order", orders, persistent=False)
        self.register_buffer(
            "harmonic_gain",
            orders ** (-self.harmonic_tilt),
            persistent=False,
        )

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )
        # One scalar decides the excitation's amplitude, and it used to be
        # drawn from ``U(-1, 1)`` -- ``nn.Linear``'s default init at
        # ``fan_in = 1``.  With ``harmonic_num = 0`` this layer has nothing to
        # merge: it is that draw multiplying the source, and nothing downstream
        # normalises it.  Over 200 seeds it landed anywhere in +-0.99, negative
        # in 48% of them and under 0.1 in 14%, a 719x spread between the
        # loudest and the quietest; measured on the excitation itself, five
        # seeds gave RMS 0.0005 / 0.0364 / 0.0162 / 0.0700 / 0.0084 -- 132x,
        # with the low end a source that is effectively muted while the trunk
        # tries to learn from it.
        #
        # ``sine_amp`` is what states the intended amplitude, so 1.0 is what
        # lets it: RMS 0.0707 on every seed instead of one seed in seven
        # starting an order of magnitude down.  It also pins the ``Tanh``
        # above at -61.6 dB against its best linear fit, which is the whole
        # reason that ``Tanh`` needs no attention -- at 3x the gain it is
        # -42.9, and what it makes there is harmonic anyway.
        #
        # This is not the fresh-pretrain kind of change: ``merge.0.weight`` is
        # a state-dict key, so a resumed run loads what it learned and only new
        # runs start anywhere different.
        #
        # At ``dim > 1`` the harmonics sum rather than replace each other and
        # the ``Tanh`` is what bounds the total -- which is what it is for in
        # the source module this comes from.  Ones is still the right init
        # there: ``harmonic_gain`` has already set each partial's level, so
        # this layer only has to add them up, and the sum stays small enough
        # for the ``Tanh`` to leave alone.  At tilt 1.0 the partials are
        # phase-randomised, so the total is ``sine_amp * sqrt(sum 1/j^2)``,
        # which converges: measured 0.0707 RMS at 1 partial, 0.0890 at 16 and
        # 0.0898 at 32 -- 2.1 dB, and bounded above by 0.0907 at any count.
        # The ``Tanh``'s departure from its best linear fit follows: -61.6 dB
        # at 1, -47.1 at 16, -53.8 at 32 (it tracks the peak factor of the
        # phase draw, not the count).  What it makes is harmonic and lands on
        # partials the source already has.
        nn.init.ones_(self.merge[0].weight)

        # Octave band of each partial (1 | 2-3 | 4-7 | ...), so ``source_gain``
        # can shape the tilt with a handful of channels at any harmonic count.
        band = torch.floor(torch.log2(orders)).long()
        self.band_count = int(band.max()) + 1
        #: (dim, band_count) one-hot: sums the partials into their bands.
        self.register_buffer(
            "band_matrix", F.one_hot(band, self.band_count).float(), persistent=False
        )
        #: Noise gain channels: a first-order lowpass of the draw, the draw
        #: itself, and a first-order highpass of it, each with its own gain.
        #: One gain on white noise can only move the floor as a block, and what
        #: the floor is short of is *tilt* -- measured against a reference, too
        #: little above 8 kHz and too much below 1.5.  Mixing the three is a
        #: slope the projection can choose per frame.
        self.noise_gain_channels = 3
        #: Gain channels ``forward`` takes: one per octave band, then the noise.
        self.gain_channels = self.band_count + self.noise_gain_channels

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _nyquist_fade(self, f0_buf):
        """Per-partial gain that reaches 0 at Nyquist. 1.0 everywhere at dim 1.

        ``f0_buf`` holds each partial's own frequency, so this is read off the
        partial rather than off the fundamental: which harmonics survive
        depends on the note, and at ``harmonic_num = 0`` nothing is ever near
        the boundary and this is exactly ``ones``.
        """

        nyquist = self.sampling_rate / 2.0
        return (
            (nyquist - f0_buf).div_(nyquist * self.NYQUIST_TAPER).clamp_(0.0, 1.0)
        )

    # Inductor cannot compile the cumsum over the sample axis: it lowers it to
    # a ``SplitScan`` whose codegen raises ``TypeError: list indices must be
    # integers or slices, not NoneType`` -- reproduced on torch 2.10 + cu130,
    # RTX 5060 -- and a failure inside the compiled region takes the *whole*
    # decoder down with it.  Only the phase stays out: it is one channel,
    # while everything after it is a chain over (batch, length, dim) that the
    # graph fuses.
    @torch.compiler.disable
    def _phase(self, f0):
        """The fundamental's phase in cycles, (batch, length, 1), and each
        partial's random initial phase, (batch, dim)."""
        # rad_values is F0 in rad mod 1 (the integer cycle count doesn't affect phase)
        rad_values = (f0 / self.sampling_rate) % 1

        # random initial phase per harmonic, none for the fundamental
        rand_ini = torch.rand(f0.shape[0], self.dim, device=f0.device)
        rand_ini[:, 0] = 0

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        return torch.cumsum(rad_values + cumsum_shift, dim=1), rand_ini

    def _f02sine(self, f0):
        """f0: (batchsize, length, 1).  Returns (batchsize, length, dim) sines."""
        phase, rand_ini = self._phase(f0)

        # Partial j's phase is j times the fundamental's (mod 1), so the two
        # cumsums run on one channel instead of ``dim``.
        phase = torch.addcmul(rand_ini.unsqueeze(1), phase, self.harmonic_order)
        # In place from here: at ``dim`` partials every temporary is a full
        # (batch, length, dim) tensor.
        return phase.mul_(2 * np.pi).sin_()

    def forward(self, f0, gain=None):
        """f0: (batch, length, 1).  gain: (batch, length, gain_channels) or None."""
        with torch.no_grad():
            # Only the phase and the fade vary per sample *and* per partial.
            # Every other factor is applied after the sum over partials, where
            # it costs one channel instead of ``dim``.
            sine_waves = self._f02sine(f0)
            sine_waves.mul_(self._nyquist_fade(f0 * self.harmonic_order))

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(uv)

        # merge with grad
        if gain is not None:
            # Unit variance each, so the three start out interchangeable and a
            # gain reads as a level rather than as a filter's own scale.
            previous = F.pad(noise[:, :-1], (0, 0, 1, 0))
            lowpass = (noise + previous) * 0.7071067811865476
            highpass = (noise - previous) * 0.7071067811865476
            # No Tanh under a gain: a learned gain can push the sum into
            # saturation, and saturating a harmonic sum at the output rate
            # folds.  The gain used to multiply after the Tanh, so the output
            # is no less bounded than it was.
            noise = (
                lowpass * gain[..., -3:-2]
                + noise * gain[..., -2:-1]
                + highpass * gain[..., -1:]
            )

        if self.dim == 1:
            # The fade is exactly one here and ``harmonic_gain`` is ``[1.0]``,
            # so this is the single-partial body as it always was, bit for bit.
            sine_waves.mul_(self.sine_amp).mul_(uv)
            if gain is not None:
                sine_waves = sine_waves * gain[..., :1]
            merged = self.merge[0](sine_waves + noise)
        else:
            weight = self.merge[0].weight[0] * self.harmonic_gain * self.sine_amp
            if gain is None:
                sine = sine_waves @ weight[:, None]
            else:
                # The gain is constant within an octave band, so each band is
                # summed first: no (batch, length, dim) gain tensor, and no
                # scatter-add of every partial into its band in the backward.
                bands = sine_waves @ (self.band_matrix * weight[:, None])
                sine = (bands * gain[..., : self.band_count]).sum(-1, keepdim=True)
            # ``merge`` over ``dim`` iid draws of std ``s / sqrt(dim)`` is one
            # draw of std ``s * ||w|| / sqrt(dim)``, so a single channel stands
            # in for all of them and the merged level stays ``noise_std`` at
            # init (``w`` is ones) whatever the harmonic count.
            noise_scale = self.merge[0].weight.norm() / self.dim**0.5
            merged = sine * uv + noise * noise_scale

        return merged if gain is not None else self.merge[1](merged)


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2: RefineGAN with its signal-path defects fixed.

    Downsamples/upchannels the excitation, fuses it with the latent, and
    upsamples through parallel residual blocks.  Against the original: a
    tilted harmonic sine instead of the truncated-sinc comb, a windowed-sinc
    interpolation filter that crops its own group delay, an excitation gain
    projected from the conditioning, f0 interpolated in log with a hard
    voiced/unvoiced gate, and an output projection with no learned gain
    (``UnitNorm``).

    Args:
        source_gain (bool, optional): Scale the excitation by envelopes
            projected from the conditioning and the speaker -- one per octave
            band of partials and one for the noise -- as RefineGAN's paper
            does with the mel. Defaults to False.
        source_harmonics (int, optional): Partials *above* the fundamental in
            the excitation. 0 -- one sine, everything else manufactured by the
            trunk -- is what every checkpoint before 2026-09-09 was trained on
            and is kept as the default so those still build. It sizes
            ``m_source.merge.0.weight``, so it cannot change on a resume.
        source_tilt (float, optional): Partial ``j`` gets amplitude
            ``j ** -tilt``. 1.0 is a sawtooth's -6 dB/octave and is within
            ~5 dB of a real voice across the band. Leaves no state-dict key,
            so ``decoder_layout`` carries it. Defaults to 1.0.

    Every pointwise nonlinearity here is a plain ``leaky_relu`` at its own
    rate.  The anti-aliased activations this decoder used to wrap them in are
    gone -- what remains from ``resampling`` is the interpolation filter on the
    upsamplers and the lowpass inside ``_decimate``, which are imaging and
    decimation rather than activation fold, and are not optional.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        gin_channels: int = 256,
        checkpointing: bool = False,
        upsample_initial_channel=512,
        filter_width: "int | Sequence[int]" = DEFAULT_UPSAMPLE_WIDTH,
        rolloff: "float | Sequence[float]" = DEFAULT_UPSAMPLE_ROLLOFF,
        filter_beta: "float | Sequence[float]" = DEFAULT_UPSAMPLE_BETA,
        source_gain: bool = False,
        source_noise_std: float = 0.003,
        adain_noise: str = "always",
        source_harmonics: int = 0,
        source_tilt: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        # The down path doubles ``start_channels`` once per stage and the up
        # path concatenates ``downs[]`` into ``channels + channels // 4``, so
        # the two only meet at one value.  It was a config knob that produced a
        # shape error deep in ``forward`` for every other value; this says so
        # at construction instead.
        required = upsample_initial_channel // (4 * 2 ** (len(upsample_rates) - 1))
        if int(start_channels) != required:
            raise ValueError(
                f"start_channels must be {required} for "
                f"upsample_initial_channel={upsample_initial_channel} over "
                f"{len(upsample_rates)} stages, not {start_channels}: the down "
                f"path doubles it per stage and the up path expects the skip to "
                f"be a quarter of the trunk."
            )

        # Scalar or one-per-stage, normalised in one place -- see
        # ``DEFAULT_UPSAMPLE_WIDTH`` for why these are a schedule and not a
        # single number.  ``filter_schedule`` is what refuses a list of the
        # wrong length, which is the only way to get this silently wrong.
        count = len(upsample_rates)
        self.filter_width = filter_schedule(filter_width, count, "filter_width", 1)
        self.rolloff = filter_schedule(rolloff, count, "rolloff", 0.0)
        self.filter_beta = filter_schedule(filter_beta, count, "filter_beta", 0.0)
        if any(value > 1.0 for value in self.rolloff):
            raise ValueError(
                f"rolloff is a fraction of the stage's Nyquist and cannot "
                f"exceed 1.0, received {self.rolloff}."
            )

        # A stage's anti-image filter keeps ``rolloff`` of the rate it *reads*,
        # so the last residual block synthesises everything above
        # ``rolloff[-1] * (sr / rate[-1]) / 2`` from scratch.  That ceiling is
        # set by the *last* factor, not by the ordering, and descending order
        # alone does nothing for it -- ``[5, 4, 4, 4]`` and ``[4, 4, 4, 5]``
        # are both descending-or-not arrangements of the same multiset and put
        # it at 3960 and 3168 Hz.  Factorising 320 differently is what moves
        # it.  Measured per stage on a 0.4 s segment ("edge" is the fraction of
        # the stage's output that ``AntiAliasedUpsample1d``'s padding changes
        # by more than 1%, against the same input embedded in a real
        # continuation):
        #
        #   [5,4,4,4]  stage 0 x5   in    100 Hz    40 smp  121 taps  edge 29.6%
        #              stage 1 x4   in    500       200     193       10.8%
        #              stage 2 x4   in   2000       800     257        3.1%
        #              stage 3 x4   in   8000      3200     385        1.0%   passes 3960 Hz
        #
        #   [10,8,2,2] stage 0 x10  in    100 Hz    40 smp  241 taps  edge 31.7%
        #              stage 1 x8   in   1000       400     385        5.1%
        #              stage 2 x2   in   8000      3200     129        0.8%   passes 3880 Hz
        #              stage 3 x2   in  16000      6400     193        0.6%   passes 7920 Hz
        #
        # ``[10,8,2,2]`` shipped for one day (2026-09-09) on the theory that
        # the ceiling was why renders lost their harmonics above ~6 kHz: at
        # ``[5,4,4,4]`` exactly one residual block runs above 4 kHz and it is
        # handed a signal cut at 3960 Hz, while ``[10,8,2,2]`` puts blocks at
        # 16 kHz *and* 32 kHz and feeds the last one content to 7920 Hz.
        #
        # **That theory is wrong, and the ceiling is not the constraint.**
        # Measured by overfitting a single 3 s clip (f0 297, 83% voiced) with
        # ``z`` as a free parameter -- one target, so an L1 has no averaging to
        # hide behind and its minimiser is the target itself, harmonic peaks
        # included.  6000 steps, ``source_harmonics=0``, harmonic contrast in
        # dB (partials above the floor between them), overfit vs target:
        #
        #     band        [10,8,2,2]   [5,4,4,4]   target
        #     1-2 kHz        14.8        14.9       15.5
        #     2-4 kHz        11.7        11.6       11.9
        #     4-6 kHz         9.4         9.4        9.4
        #     6-8 kHz         6.2         6.1        6.2
        #     8-10 kHz        5.2         5.2        5.1
        #     10-13 kHz       3.8          --        3.6
        #
        # Both layouts reproduce the target to within 0.7 dB everywhere, to
        # 13 kHz, from a one-partial excitation.  The final residual block
        # regenerates harmonics well past the ceiling without help, so the
        # 3960 Hz figure is real and simply not binding.  The same probe run
        # with ``build_ms_mel_loss`` -- the actual training reconstruction loss
        # -- lands within 0.2 dB in every band, so that loss is not the
        # constraint either.  Whatever costs a trained model its high
        # harmonics is upstream of this decoder, and no arrangement of these
        # stages addresses it.
        #
        # So the shipped layout stays ``[5,4,4,4]`` -- and it stayed after a
        # second attempt at ``[10,8,2,2]`` on 2026-09-10, for a reason the
        # table above does not cover.  That attempt was not the ceiling
        # argument returning: it was about *inharmonic* content, which
        # harmonic contrast is blind to because contrast measures the partials
        # and not what sits between them.  Renders at ``[5,4,4,4]`` showed
        # three lines between 3 and 5 kHz, straddling the 3960 Hz mirror the
        # last factor of 4 puts there, and two mechanisms live at a stage
        # boundary: the stage's own ``leaky_relu`` mirrors ``j*f0`` above the
        # boundary back down, with nothing after it to filter, and the
        # upsampler images content just under the mirror to just over it.
        #
        # It was reverted on cost against coverage.  Cost: the decoder measured
        # ~2x for fwd+bwd, because the residual blocks then run at 1000 / 8000
        # / 16000 / 32000 Hz instead of 500 / 2000 / 8000 / 32000.  Coverage:
        # a block folds around half the rate it runs at, and *a block runs at
        # 8000 Hz in both* -- so the 4 kHz fold is created either way and only
        # moves one stage earlier.  What the refactorisation actually buys is
        # the last upsampler's imaging boundary, and the rolloff below buys the
        # same thing for nothing.
        #
        # The table is kept because the numbers are real and the next person to
        # reach for a refactorisation should see that it was tried, measured,
        # and what each attempt did and did not address.
        #
        # A reorder or a refactorisation is invisible in the state dict -- all
        # tensors keep their keys and shapes, since channel counts follow the
        # stage index and not the rate -- so
        # ``rvc.train.utils.decoder_layout`` writes it into the checkpoint.
        #
        # ``int``, not the ``np.int64`` ``np.prod`` returns.  Dynamo wraps a
        # numpy scalar used inside a traced function as a *CPU* tensor, and one
        # CPU node is enough to make Inductor emit a C++ kernel -- which on
        # Windows needs ``cl.exe`` and fails the whole compile with
        # ``InvalidCxxCompiler``.  This is one of three independent CPU/codegen
        # sources in this decoder -- the other two are marked
        # ``torch.compiler.disable`` -- and removing any one of them alone
        # still fails the compile.
        self.upp = int(np.prod(upsample_rates))

        # The excitation.  See ``SineGenerator`` for the probe that ranked the
        # sources.
        #
        # ``source_harmonics`` can put tilted partials into the source, but
        # ships at 0.  The overfit above says why: at ``harmonics=0`` this
        # decoder already reproduces a target's harmonic contrast to 13 kHz, so
        # a source short of partials is not what a trained model's missing
        # harmonics are made of.
        #
        # What the tilt does *not* answer is the objection in ``source_gain``
        # below: a source that hands the trunk its harmonics for free lets the
        # trunk stop consulting ``z``, and the KL falls because the decoder
        # needs less.  ``excitation_source`` in ``rvc/train/utils.py`` names
        # the mismatch if a checkpoint from another source is loaded.
        self.source_type = "sine"
        # The dither the excitation carries in *voiced* frames, and the only
        # stochastic material the decoder is given there.  0.003 against a
        # harmonic RMS of ``wave_amp / sqrt(2)`` is -27.4 dB, while the band
        # above 10 kHz in real voiced speech is barely harmonic at all: its
        # floor between the partials sits 2.6 dB under them.  A decoder short
        # of that floor -- measured at 11.1 dB down on the valleys against 7.1
        # on the peaks -- is short of it partly because it was never handed
        # any, and no reconstruction loss can ask for it: the minimiser of an
        # L1 against an unpredictable component is less of that component.
        #
        # Swept on a 4-epoch pretrain at 32 kHz, rendered through ``infer``
        # (deficits in dB, MS mel on the same pairs):
        #
        #   noise_std   whole set 10k   worst twelfth 10k   MS mel (all / worst)
        #     0.003         -2.00            -9.96          0.723 / 0.762
        #     0.01          -1.21            -8.28          0.713 / 0.723
        #     0.03          +0.61            -4.85          0.781 / 0.782
        #     0.06          +3.17            -0.54          0.944 / 0.948
        #
        # 0.01 is better on both axes at once, which is why it is worth having
        # as a knob.  What does *not* work, measured on the same 80 excerpts,
        # is shaping the dither instead of raising it: a first-order difference
        # tilts it +11.8 dB towards the top and does flatten the bands (10 kHz
        # -0.17, 2 kHz +0.06 against white 0.01's -1.15 and +0.60), but the MS
        # mel goes the wrong way, 0.739 against 0.713, and the band's own
        # peak-to-valley contrast does not improve -- 15.33 against 15.38, with
        # the reference at 15.90.  White at the right level beats shaped at any
        # level tried.  0.03 buys half the remaining tail and pays for it in the
        # bands below 6 kHz, where it overshoots.  Note the sweep is a *level*
        # the trunk downstream was not trained against, so it measures whether
        # the material propagates, not what training at that level converges
        # to.
        #
        # Ships at 0.003: at 0.01 this dither is most of what sits between the
        # harmonics above 4 kHz (+4.7 dB at 8-10 kHz over 0.003 on a stage-1
        # render), which the sweep above cannot see -- it measures only whether
        # the band gets filled.
        self.source_noise_std = float(source_noise_std)
        self.adain_noise = adain_noise_mode(adain_noise)
        # How many partials the excitation carries, and how steeply they fall.
        # 0 is what this shipped with and what every checkpoint before
        # 2026-09-09 was trained on; ``source_harmonics`` sizes
        # ``m_source.merge.0.weight``, so it cannot be changed on a resume.
        # ``source_tilt`` leaves no key at all and rides in
        # ``decoder_layout``.  See ``SineGenerator`` for what the source looks
        # like at each, and the stage table below for why 0 stopped being
        # survivable once the trunk's ceiling was the only thing above it.
        self.source_harmonics = int(source_harmonics)
        self.source_tilt = float(source_tilt)
        self.m_source = SineGenerator(
            sample_rate,
            harmonic_num=self.source_harmonics,
            noise_std=source_noise_std,
            harmonic_tilt=self.source_tilt,
        )

        # ``start_channels``, not a literal 16.  It was hardcoded here while
        # the down path below was built from ``start_channels``, so any value
        # but 16 produced a channel mismatch at ``downsample_blocks[0]`` -- a
        # config knob that could only take one value, which is worse than no
        # knob.
        self.pre_conv = weight_norm(
            nn.Conv1d(1, start_channels, 7, 1, padding=3)
        )

        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []
        for i, u in enumerate(upsample_rates):

            new_size = int(size / upsample_rates[-i - 1])
            # T dimension factors for torchaudio.functional.resample
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        channels = upsample_initial_channel

        self.mel_conv = weight_norm(
            nn.Conv1d(
                num_mels,
                channels // 2,
                7,
                1,
                padding=3,
            )
        )

        self.mel_conv.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, channels // 2, 1)

        # The paper scales its template by "intensity-like values calculated
        # from the Mel-spectrogram"; this decoder is handed ``z``, and a
        # least-squares fit recovers the log intensity from it at r = 0.996.
        #
        # Held-out multi-scale mel on a fixed trunk, 3 seeds: sine
        # 1.9714 -> 1.7418, comb 1.9649 -> 1.8363.  A large win for a source
        # that carries no envelope of its own.
        #
        # With a flat, f0-driven source the trunk gets its harmonics without
        # consulting ``z`` at all, and the KL falls because the decoder needs
        # less rather than because the prior caught up.  Projecting the
        # source's *envelope* from the conditioning puts ``z`` back on the
        # critical path for harmonic content: the trunk cannot get the envelope
        # from f0, so the posterior has to carry it and the KL has to pay.
        #
        # What stays f0-driven is the scaffolding -- which partials exist, at
        # what frequencies, under the Nyquist mask -- so the source is still
        # alias-free and in tune whatever the projection predicts.  A bad ``z``
        # can dull or brighten it; it cannot detune it.
        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
            gain_channels = self.m_source.gain_channels
            self.source_gain = nn.Conv1d(
                num_mels,
                gain_channels,
                SOURCE_GAIN_KERNEL,
                padding=SOURCE_GAIN_KERNEL // 2,
            )
            # Identity at initialisation: ``softplus(0.5413) = 1.0`` with zero
            # weights, so a run that switches this on starts from exactly the
            # excitation it had before and the projection has to earn every
            # departure from unity.  It also means the module cannot silently
            # rescale a fine-tune's source on step zero.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)
            # The two tilt channels start muted (``softplus(-6) = 0.0025``), so
            # the excitation at initialisation is the white draw it always was
            # and a slope is something the projection has to ask for.
            with torch.no_grad():
                self.source_gain.bias[-3] = -6.0
                self.source_gain.bias[-1] = -6.0
            if gin_channels != 0:
                # Zero as well, so the speaker term starts out adding nothing.
                self.source_gain_cond = nn.Conv1d(gin_channels, gain_channels, 1)
                nn.init.zeros_(self.source_gain_cond.weight)
                nn.init.zeros_(self.source_gain_cond.bias)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic -- ``f0 +- (R_stage - f)``,
            # which walks against f0 and reads as a fold.  A smooth envelope
            # makes ``F.interpolate`` good enough (-81.9 dB against -82.5), but
            # this gain is learned: on a frame-rate-white one the same
            # comparison is -50.3 against -82.6.
            #
            # The trunk's schedule rather than its longest kernel at every
            # stage: at width 48, stage 0 reaches 48 frames each way, more than
            # a 40-frame training segment, so the padding decided the gain.
            self.source_gain_ups = nn.ModuleList(
                [
                    AntiAliasedUpsample1d(
                        rate,
                        filter_width=self.filter_width[stage],
                        rolloff=self.rolloff[stage],
                        filter_beta=self.filter_beta[stage],
                    )
                    for stage, rate in enumerate(upsample_rates)
                ]
            )

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for stage, rate in enumerate(upsample_rates):
            new_channels = channels // 2

            # Was ``nn.Upsample(mode="linear")``, whose triangular kernel
            # rejects the first image by only 1.7-9.6 dB, stamping the frame
            # grid into the waveform as a mirrored partial either side of every
            # harmonic.  Measured at x2 from 16 kHz, a partial at 7500 Hz comes
            # back with a mirror at 8500 Hz only 2.6 dB down -- a line that
            # walks *against* f0 as pitch moves.  The down path already uses a
            # windowed sinc; this makes the up path agree with it.
            #
            # A trained ``ConvTranspose1d`` here -- HiFi-GAN's arrangement, and
            # ``hifigan_nsf``'s -- was tried on 2026-09-09 and removed: nothing
            # measured says this filter is what limits the decoder, and the
            # overfit probe above says it is not.
            self.upsample_blocks.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=self.filter_width[stage],
                    rolloff=self.rolloff[stage],
                    filter_beta=self.filter_beta[stage],
                )
            )

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                    adain_noise=self.adain_noise,
                )
            )

            channels = new_channels

        # Unit-norm rather than ``weight_norm``: see ``UnitNorm`` for the gain
        # this leaves out and what it did.  No ``init_weights`` -- on a
        # parametrized conv it writes into a temporary and never did anything,
        # and with only a direction to learn any isotropic draw is as good.
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        register_parametrization(self.conv_post, "weight", UnitNorm())
        self.register_load_state_dict_pre_hook(_refuse_learned_output_gain)

        self.out_tanh = nn.Tanh()

        # Set by ``apply_precision_policy`` under BF16: the excitation, the
        # upsampling filters and the output layer then run in FP32.
        self.fp32_residuals = False

    def _fp32_region(self, x: torch.Tensor):
        if not self.fp32_residuals:
            return nullcontext()
        return torch.autocast(x.device.type, enabled=False)

    def _fp32(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() if self.fp32_residuals else x

    def _upsample(self, ups: nn.Module, x: torch.Tensor) -> torch.Tensor:
        with self._fp32_region(x):
            return ups(self._fp32(x))

    # The other half of the compile story.  ``torchaudio.functional.resample``
    # builds its sinc kernel from Python ints on every call, and Inductor
    # compiles that construction to a *CPU* kernel, which on Windows needs
    # ``cl.exe``; without MSVC the whole decoder fails with
    # ``InvalidCxxCompiler: Compiler: cl is not found`` and drops to eager.
    # Kept out of the graph rather than replaced: torchaudio's filter is
    # 385/953 taps against the 73/169 a ``FixedLowPass1d`` of the same shape as
    # the upsamplers would build, and its stopband is 135-156 dB against 68-78,
    # so swapping it would be a numerical change smuggled in under a build fix.
    #
    # The kernel is what ``torchaudio.functional.resample`` builds, cached per
    # reduced ratio: it rebuilt it on every call, with a host-to-device copy
    # each time, and it only depends on ``orig / gcd``, ``new / gcd``.
    @torch.compiler.disable
    def _decimate(self, x: torch.Tensor, orig_freq: int, new_freq: int):
        gcd = math.gcd(orig_freq, new_freq)
        key = (orig_freq // gcd, new_freq // gcd, x.dtype, x.device)
        cache = self.__dict__.setdefault("_decimate_kernels", {})
        if key not in cache:
            with cache_scope():
                cache[key] = _get_sinc_resample_kernel(
                    orig_freq,
                    new_freq,
                    gcd,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                    device=x.device,
                    dtype=x.dtype,
                )
        kernel, width = cache[key]
        return _apply_sinc_resample_kernel(
            x.contiguous(), orig_freq, new_freq, gcd, kernel, width
        )

    @staticmethod
    def _expand_f0(f0: torch.Tensor, length: int) -> torch.Tensor:
        """f0 at the frame rate -> f0 at the output rate, (batch, 1, length).

        Two things the plain ``F.interpolate(f0, mode="linear")`` got wrong.

        Linear interpolation in Hz makes the frame-rate ripple a *constant*
        absolute wobble, so its effect in cents grows as f0 falls and its
        sidebands at ``j*f0 +- m*R_frame`` grow with ``j``: a fan of lines
        around the high harmonics that is not a fold but reads as one.  In log
        the wobble is constant in cents instead, which is both smaller and
        pitch-independent.

        And interpolating across a voiced/unvoiced boundary ramps f0 linearly
        toward zero over a whole frame while ``uv`` stays 1, which is a
        descending chirp of every harmonic at once at every boundary.  Gating
        with a ``nearest`` ``uv`` keeps the boundary where the f0 estimator put
        it.

        The first version of that gate left the chirp half in place: it
        interpolated ``log f0`` with the ``log 1 = 0`` of the unvoiced frames
        still in it, so the last half-frame before every voiced/unvoiced
        boundary glided toward 1 Hz while the gate read voiced -- 200 Hz
        reached 14.3 Hz at the frame edge, 157 of those 160 samples under
        190 Hz.  Fixed 2026-09-11 by weighting the interpolation with the
        voiced mask; see ``commons.expand_f0``.

        Not a ``decoder_layout`` field.  It moves the excitation only within
        half a frame of each boundary, and a checkpoint trained on the glide
        should resume into the corrected source rather than be refused.
        """

        return expand_f0(f0, length)

    def _source_gain(self, mel: torch.Tensor, g: torch.Tensor = None):
        """The excitation gains, (batch, frames * upp, gain_channels), or None.

        ``mel`` is this decoder's conditioning -- ``z``, despite the name -- at
        the frame rate; ``g`` is the speaker embedding.
        """

        if not self.has_source_gain:
            return None
        gain = self.source_gain(mel)
        if g is not None and hasattr(self, "source_gain_cond"):
            gain = gain + self.source_gain_cond(g)
        for ups in self.source_gain_ups:
            gain = ups(gain)
        # After the interpolation, not before: the sinc overshoots at onsets
        # and would take a small positive gain below zero.
        return F.softplus(gain).transpose(1, 2)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = self._expand_f0(f0, f0_size * self.upp)
        # ``SineGenerator`` works in (batch, time, dim) -- ``dim`` is the
        # harmonic axis, 1 here -- while the trunk is channel-first throughout.
        # The sine is the one signal whose low-order bits carry phase, so it is
        # kept out of BF16 until the first conv has turned it into features.
        with self._fp32_region(mel):
            gain = self._source_gain(
                self._fp32(mel), None if g is None else self._fp32(g)
            )
            har_source = self.m_source(f0.transpose(1, 2), gain).transpose(1, 2)
            x = self.pre_conv(har_source)
        downs = []
        for index, (block, (old_size, new_size)) in enumerate(
            zip(self.downsample_blocks, self.df0)
        ):
            # Only the first site is an activation.  Every pointwise
            # nonlinearity folds about the Nyquist of *its own* rate, and the
            # down path runs at 32000 / 8000 / 2000 / 500 Hz, so the sites past
            # the first are three fold generators sitting on the excitation --
            # the one signal in this decoder that arrives alias-free and in
            # tune by construction.  Bisected on a 35k checkpoint with constant
            # ``z`` and constant f0, excess over a matched control at
            # ``8000 - k*f0``:
            #
            #     har_source / pre_conv / downs[0] / decimate / block   ~ -2 dB
            #     act(x) at 8000 Hz                                    +38.4 dB
            #
            # The first one earns its place: it runs at ``sample_rate``, where
            # there is no band to fold into, and rectifying the sine is what
            # *creates* the harmonics the pyramid then band-limits.  Applying
            # ``leaky_relu`` again at 8 kHz to a signal that is already
            # rectified and already cut at 4 kHz adds nothing a linear conv
            # cannot do, and the products it makes land straight above that
            # stage's Nyquist.  Removing the sites costs negative time, where a
            # filter could only ever attenuate what they make.
            if index == 0:
                x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            # The activation runs on the upsampler's *output*, not its input.
            # A pointwise nonlinearity folds about the Nyquist of its own rate,
            # and what decides how much it makes is how full the band already
            # is: on the input it sees a band ``upsample_conv_blocks`` has just
            # filled -- occupancy 1/1, where second-order products land
            # straight above Nyquist -- while on the output the same content
            # occupies 1/factor of a band ``factor`` times wider, so the fold
            # is deferred to a much higher order.  Under ``[5, 4, 4, 4]`` this
            # moves the four sites from 100 / 500 / 2000 / 8000 Hz to 500 /
            # 2000 / 8000 / 32000, and the one that mattered -- 8000 Hz, whose
            # mirror is the 4 kHz this decoder's inharmonic lines sat around --
            # goes from reading a full band to reading a quarter of one.
            #
            # Measured at no cost: the elementwise op covers 425 samples per
            # frame instead of 106, which is invisible against the convolutions
            # (medians 688.5 against 693.7 ms over ten randomised repetitions,
            # ranges overlapping).
            #
            # What this does *not* touch: the ``leaky_relu`` pairs inside every
            # ``ResBlock``, which run after the band is filled by construction.
            # Those are the generic case anti-aliasing exists for; these four
            # were the ones that could simply be moved.
            if self.training and self.checkpointing:
                x = checkpoint(self._upsample, ups, x, use_reentrant=False)
                x = F.leaky_relu(x, self.leaky_relu_slope)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = self._upsample(ups, x)
                x = F.leaky_relu(x, self.leaky_relu_slope)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        with self._fp32_region(x):
            x = F.leaky_relu(self._fp32(x), self.leaky_relu_slope)
            x = self.conv_post(x)
            x = self.out_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """Fold every weight parametrization into its weight -- the weight norms
        and ``conv_post``'s ``UnitNorm`` alike -- by walking the modules.

        Walking rather than listing layers by name: a hand-written list goes
        stale the moment one is added, and nothing catches it because
        ``Synthesizer`` walks the decoder itself and never calls this.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)