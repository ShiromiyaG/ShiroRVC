from typing import Sequence

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights, get_padding
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
#: than by -49.5 dB of path gain.  A full-band BLIT excitation makes that term
#: larger than it ever was against the sine.
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
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.95, 0.97, 0.99)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)

#: The excitation gain is one channel, so its upsample chain is free whatever
#: the kernel length -- and it is the one path where an image is not attenuated
#: but *multiplied* onto every harmonic as a sideband.  It gets the last
#: stage's design at every stage rather than the trunk's schedule.
SOURCE_GAIN_WIDTH = 48
SOURCE_GAIN_ROLLOFF = 0.99
SOURCE_GAIN_BETA = 9.0


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

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """Noise-regularised activation.

    There are two of these wrapped around every ``ResBlock`` -- six per
    ``ParallelResBlock``.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        # The noise is a training-time regulariser, and at inference it is pure
        # cost: this module runs six times per decoder stage, so at the output
        # rate it draws 13.5 M samples per forward.  Measured on CPU at batch 4
        # over 0.4 s, dropping it in eval takes the whole generator from 418 ms
        # to 313 ms -- 25% of the forward.  Inference stays stochastic anyway:
        # the excitation draws its own noise, and that one is not optional --
        # it *is* the unvoiced content.
        if not self.training:
            return self.activation(x)

        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


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
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class BlitGenerator(nn.Module):
    """Band-limited impulse train excitation.

    Replaces the sine source.  The sine carries one partial and the trunk has
    to *manufacture* every harmonic above it, which means asking a stack of
    ``leaky_relu`` for products of order j to reach ``j*f0`` -- the aliasing is
    not a side effect of that arrangement, it is the mechanism being used.  A
    BLIT hands the trunk every harmonic under Nyquist already, so the trunk
    only has to shape an envelope.

    The closed form is the Dirichlet kernel

        blit[n] = sin(pi * M * phi[n]) / (M * sin(pi * phi[n])),  M = 2N+1

    which is exactly ``sum_{k=-N..N} exp(2*pi*i*k*phi) / M`` -- a sum of N
    cosines and nothing else.  Band-limited by construction, with no window and
    no truncation, unlike ``sinc(sr * x / f0)`` truncated at the period edge:
    that one is cut where the sinc has not yet decayed, and the step it leaves
    once per period is a broadband floor.

    The harmonic count is recomputed per sample from f0, so it follows the
    pitch and no partial is ever placed above Nyquist.  There is no fold to
    clean up afterwards, which is the whole point of using the closed form
    rather than summing cosines and hoping.

    It is deliberately *fractional*: an integer count taken per sample steps
    whenever f0 crosses ``limit / N``, and each step is a discontinuity whose
    rate tracks the pitch.  The top pair is weighted by the fraction and the
    limit carries a one-partial margin, which also keeps the highest partial
    off Nyquist where its conjugate image would double it.  See ``forward``.

    Args:
        samp_rate: output sample rate in Hz.
        wave_amp: excitation level.  Under ``normalize=True`` it reads as the
            amplitude of the equivalent sine and the pulse is scaled to
            ``wave_amp / sqrt(2)`` RMS at every pitch; under ``False`` it is
            the pulse's peak, and the level is then f0-dependent.
        normalize: hold the excitation's *energy* constant across the range
            rather than its peak.  See ``forward``.  ``False`` restores the
            unit-peak kernel exactly -- both the 20 dB level tilt across a
            singer's range and the inverted voiced/unvoiced balance, which are
            what every run before 2026-09-04 was fitted to.  The two are not
            separable through this flag: if what is wanted is the noisier
            excitation without the tilt, raise ``noise_std`` and the unvoiced
            amplitude with ``normalize`` left on.
        noise_std: Gaussian noise std in voiced regions.
        voiced_threshold: f0 above which a frame counts as voiced.
        bandwidth: fraction of Nyquist to fill, in ``(0, 1]``.  1.0 is the
            true BLIT.  Lower values cap ``M`` and hand the trunk a source
            occupying less of the band: less high-frequency detail delivered,
            but less intermodulation at every downstream nonlinearity.

            The shipped configs leave this at 1.0 and it is worth knowing what
            that costs: a BLIT at 1.0 presents an occupancy of 1/1 to the first
            activation it meets, the worst operating point a pointwise
            nonlinearity has.  Now that the anti-aliased activations are gone
            this is the *only* control the decoder has over activation fold,
            and it works on the cause rather than on each site.

            It is left at 1.0 because it is a *trade*, not a fix, and the trade
            has not been measured on this decoder.  What it costs is that
            everything above the ceiling goes back to being manufactured out of
            activation products, which is the sine source's mechanism, at the
            output rate.

            Because the cap is a fixed *frequency*, occupancy is constant
            across the range -- measured 25.0% at ``bandwidth=0.25`` for every
            f0 from 80 to 800 Hz.  A fixed harmonic *count* instead makes
            occupancy track f0 (4% of Nyquist at 80 Hz, 40% at 800), so the
            intermodulation a stage sees would depend on the note.  Since
            occupancy is what sets that intermodulation, holding it fixed is
            what makes the downstream sites predictable.

            Do not go much below 0.4: the ceiling decides where the invented
            band starts.  At 0.5 that is 8 kHz, above the region where
            inharmonic lines are audible and where the mel loss puts its
            weight; at 0.25 it is 4 kHz, which is exactly where the folded
            lines used to sit.

            One number that does *not* argue for lowering it, since the
            energy normalisation landed: a narrower source does leave more
            level on each partial it keeps, but only +3.0 dB from 1.0 to 0.5
            (measured at f0=200).  It was +12 against the unit-peak kernel,
            and that was an artefact of the peak normalisation rather than a
            property of the bandwidth.
        learn_gain: a single learned scalar on the excitation, so the source
            level is not frozen at ``wave_amp``.  There is deliberately no
            ``tanh`` here -- the sine source could afford one because a single
            narrowband partial folds harmlessly, but a full-band BLIT through a
            saturating nonlinearity aliases immediately, and that would put an
            unfixable artefact upstream of every anti-aliased site in the
            decoder.
    """

    def __init__(
        self,
        samp_rate: int,
        wave_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        bandwidth: float = 1.0,
        learn_gain: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        if not 0.0 < float(bandwidth) <= 1.0:
            raise ValueError(
                f"bandwidth is a fraction of Nyquist and must be in (0, 1], "
                f"not {bandwidth!r}."
            )

        self.sampling_rate = int(samp_rate)
        self.wave_amp = float(wave_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)
        self.bandwidth = float(bandwidth)
        self.normalize = bool(normalize)

        # One scalar, with grad, so the excitation level is learned while the
        # waveform itself stays a pure function of f0.
        self.gain = (
            nn.Parameter(torch.ones(1)) if learn_gain else None
        )

    # Inductor cannot compile this body.  The phase is a cumsum over the sample
    # axis, and Inductor lowers it to a ``SplitScan`` whose codegen raises
    # ``TypeError: list indices must be integers or slices, not NoneType`` --
    # reproduced on torch 2.10 + cu130, RTX 5060.  A failure inside the
    # compiled region takes the *whole* decoder down with it, so
    # ``enable_decoder_compile`` fell back to eager for every step.
    #
    # Everything here runs under ``no_grad`` and is a pure function of f0, so
    # keeping it out of the graph costs no fusion.
    @torch.compiler.disable
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (batch, 1, samples) at the output rate.  Returns the same shape."""

        with torch.no_grad():
            uv = (f0 > self.voiced_threshold).to(f0.dtype)
            f0_safe = f0.clamp_min(1.0)

            # float64 for the phase accumulator.  In float32 the running sum
            # reaches ~1e4 cycles within a second of audio, where the ulp is
            # about 1e-3 of a cycle -- audible phase jitter on every harmonic
            # at once.  The alternative is the wrap-and-subtract trick the sine
            # generator used; float64 is the same fix with none of the indexing.
            phase = torch.cumsum(f0_safe.double() / self.sampling_rate, dim=-1)
            phase = phase - torch.floor(phase)

            # Fractional harmonic count.  ``floor(limit / f0)`` is evaluated per
            # *sample*, so a moving f0 steps the count integer by integer, and
            # ``Dirichlet(M, phi) != Dirichlet(M-2, phi)`` at any phi but the
            # peak of the pulse: every step is a discontinuity in the waveform.
            # Measured in the 20 Hz .. f0/2 band -- which a clean BLIT cannot
            # occupy at all, so anything there is artefact -- at f0 = 200 Hz:
            #
            #     constant pitch                    -125.9 dB
            #     vibrato    5 cents ( 4 steps)      -67.1
            #     vibrato   50 cents (20 steps)      -62.5
            #     portamento one octave (67 steps)   -60.4
            #
            # Freezing M through the same 50-cent vibrato gives -83.9, so it is
            # the steps and not the modulation.  They happen exactly where f0
            # crosses ``limit / N``, so their rate is a function of f0 and the
            # lines they leave walk with the pitch -- which is precisely the
            # signature that reads as a fold in a spectrogram.  ``_expand_f0``
            # is not involved: sample-wise f0 with no frame grid measures -61.7
            # against -62.5 for the log-linear interpolation.
            #
            # ``D_{2N+1} * (2N+1)`` is the sum over ``k = -N..N``; adding the
            # next conjugate pair with weight ``w = kmax - N`` makes the count
            # continuous in f0, for one extra cosine.  Same test:
            #
            #     vibrato    5 cents   -67.1 -> -106.7 dB
            #     vibrato   20 cents   -67.7 ->  -94.7
            #     vibrato   50 cents   -62.5 ->  -86.9
            #     vibrato  100 cents   -59.9 ->  -83.8
            #     portamento one octave -60.4 -> -82.0
            #
            # The ``- f0_safe`` is a one-partial margin.  Whenever f0 divides
            # ``sr/2`` -- 100, 200, 320, 400, 500, 640, 800 Hz at 32 kHz --
            # ``floor(limit / f0) * f0`` lands *on* Nyquist, where the two
            # conjugate images coincide and sum: that partial comes out 6 dB
            # above every other one, at the exact frequency every nonlinearity
            # downstream folds about.  Under wide FM the top partial wants more
            # room than one harmonic; ``1.5 * f0_safe`` is the next step.
            limit = self.bandwidth * self.sampling_rate / 2.0 - f0_safe.double()
            kmax = (limit / f0_safe.double()).clamp_min(1.0)
            n_har = torch.floor(kmax)
            w = kmax - n_har
            m = 2.0 * n_har + 1.0

            denominator = torch.sin(np.pi * phase)
            # phi -> 0 is the removable singularity where every cosine is in
            # phase and the unnormalised kernel equals M.
            singular = denominator.abs() < 1e-12
            core = torch.where(
                singular,
                m,
                torch.sin(np.pi * m * phase)
                / torch.where(singular, torch.ones_like(denominator), denominator),
            )
            core = core + 2.0 * w * torch.cos(2.0 * np.pi * (n_har + 1.0) * phase)
            # The value at phi = 0, so the kernel keeps its unit peak.  The
            # normalisation below has to divide by this one too: ``sqrt(m / 2)``
            # would put the integer step straight back into the level.
            weight = m + 2.0 * w

            blit = core / weight
            # ``1/sqrt(M)`` is the RMS of the Dirichlet kernel: it is
            # normalised to a *unit peak*, so each of its M harmonics carries
            # ``1/M`` and the whole excitation gets quieter the lower the note.
            # Measured at 32 kHz, level per partial: -46.1 dB at f0=80 against
            # -26.5 at f0=800 -- a 20 dB tilt across a singer's range that is a
            # pure function of f0 and that nothing downstream knows about.  The
            # only thing placed to undo it is ``source_gain``, 193 parameters
            # that would spend their capacity on an analytic factor and come
            # out coupled to pitch.
            #
            # It also inverted the voiced/unvoiced balance.  The noise below is
            # specified as an RMS and the pulse was a peak, so unvoiced frames
            # ran about 12 dB *louder* than voiced ones at 200 Hz.
            #
            # ``sqrt(M/2)`` puts the RMS at ``wave_amp / sqrt(2)`` for every
            # pitch, which is exactly a sine of amplitude ``wave_amp`` -- so the
            # number keeps the meaning it had under the sine source and the
            # voiced/unvoiced ratio stays where it was.  Full compensation
            # (``M``) would hold each *harmonic* constant instead and let the
            # peak grow with M; constant energy is the compromise, and it keeps
            # the pulse from dominating ``pre_conv`` at low f0.
            #
            # ``normalize=False`` is the unit-peak kernel, with both of those
            # back: it is what every run before 2026-09-04 was fitted to, and
            # the trunk downstream is calibrated to whatever level it was
            # trained against.  Comparing the two on shared weights measures
            # that calibration, not the excitation.
            if self.normalize:
                blit = blit * torch.sqrt(weight / 2.0)
            blit = blit.to(f0.dtype) * self.wave_amp

            # Unvoiced regions are noise; voiced ones get a small dither.
            # Same schedule the sine source used, and now the same balance:
            # both sides are RMS since the normalisation above, so voiced runs
            # ``wave_amp / sqrt(2)`` against unvoiced ``wave_amp / 3`` -- a
            # ratio of 2.12, exactly what a sine of ``wave_amp`` gave.  Against
            # the unit-peak kernel it was inverted, unvoiced sitting 12 dB
            # above voiced at f0=200.
            noise_amp = uv * self.noise_std + (1.0 - uv) * self.wave_amp / 3.0
            excitation = blit * uv + noise_amp * torch.randn_like(blit)

        if self.gain is not None:
            excitation = excitation * self.gain

        return excitation


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2: RefineGAN with its signal-path defects fixed, on a BLIT source.

    Downsamples/upchannels the excitation, fuses it with the latent, and
    upsamples through parallel residual blocks.  Against the original:
    a band-limited impulse train instead of the sine or the truncated-sinc
    comb, descending stage rates, a windowed-sinc interpolation filter that
    crops its own group delay, an excitation gain projected from the
    conditioning, and f0 interpolated in log with a hard voiced/unvoiced gate.

    Args:
        source_gain (bool, optional): Scale the excitation by an intensity
            envelope projected from the conditioning, as RefineGAN's paper
            does with the mel. Defaults to False.
        source_bandwidth (float, optional): Fraction of Nyquist the BLIT
            fills, and the only control this decoder has over aliasing that
            works on the cause rather than on each site. Defaults to 1.0 -- the
            true BLIT, and the worst case for every nonlinearity downstream.
            The shipped configs do not set it; see :class:`BlitGenerator` for
            what lowering it buys, what it costs, and why 0.4 is the floor.
        source_normalize (bool, optional): Hold the excitation's energy
            constant across the pitch range instead of its peak. See
            :class:`BlitGenerator`. Defaults to True; False is the pre
            2026-09-04 excitation exactly.

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
        source_bandwidth: float = 1.0,
        source_normalize: bool = True,
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

        # Descending order, as every HiFi-GAN variant uses.  A stage's
        # anti-image filter keeps ``rolloff`` of the rate it reads, so the last
        # residual block synthesises everything above ``rolloff * rate[-2] / 2``
        # from scratch: ``320 = 4*4*4*5`` as ``[4,4,4,5]`` puts that ceiling at
        # 2880 Hz, as ``[5,4,4,4]`` at 3600.
        #
        # A reorder is invisible in the state dict -- all tensors keep their
        # keys and shapes -- so ``rvc.train.utils.decoder_layout`` writes it
        # into the checkpoint.
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

        # The excitation.  ``comb`` and ``sine`` are both gone: the sine made
        # the trunk manufacture every harmonic out of activation products, and
        # the comb's truncated sinc left a step at every period edge.  The BLIT
        # is what the comb was reaching for, in closed form.
        # ``excitation_source`` in ``rvc/train/utils.py`` names the mismatch if
        # an old checkpoint is loaded, since the state dicts differ.
        self.source_type = "blit"
        # Invisible to ``load_state_dict``: ``BlitGenerator`` owns one scalar
        # parameter whatever the bandwidth, so a checkpoint trained against a
        # full-band source loads into a band-limited one without a murmur.
        # ``rvc.train.utils.decoder_layout`` reports this for that reason.
        self.source_bandwidth = float(source_bandwidth)
        # Invisible in the same way, and a larger change than the bandwidth:
        # it moves the excitation's level by 13-23 dB depending on the note.
        self.source_normalize = bool(source_normalize)
        self.m_source = BlitGenerator(
            sample_rate,
            bandwidth=source_bandwidth,
            normalize=source_normalize,
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
        # that carries no envelope of its own.  The BLIT is in that category
        # too -- it is flat by construction -- so this is worth keeping on.
        # 193 parameters.
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
            self.source_gain = nn.Conv1d(num_mels, 1, 1)
            # Identity at initialisation: ``softplus(0.5413) = 1.0`` with zero
            # weights, so a run that switches this on starts from exactly the
            # excitation it had before and the projection has to earn every
            # departure from unity.  It also means the module cannot silently
            # rescale a fine-tune's source on step zero.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic -- ``f0 +- (R_stage - f)``,
            # which walks against f0 and reads as a fold.  No anti-aliasing
            # reaches it, because a multiplication is not a pointwise
            # nonlinearity.  A smooth envelope makes ``F.interpolate`` good
            # enough (-81.9 dB against -82.5), but this gain is learned: on a
            # frame-rate-white one the same comparison is -50.3 against -82.6.
            #
            # So this chain does *not* follow the trunk's schedule.  It runs on
            # (B, 1, T) and the taps are free at any length, so every stage
            # gets the last stage's design.
            self.source_gain_ups = nn.ModuleList(
                [
                    AntiAliasedUpsample1d(
                        rate,
                        filter_width=SOURCE_GAIN_WIDTH,
                        rolloff=SOURCE_GAIN_ROLLOFF,
                        filter_beta=SOURCE_GAIN_BETA,
                    )
                    for rate in upsample_rates
                ]
            )

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for stage, rate in enumerate(upsample_rates):
            new_channels = channels // 2

            # Was ``nn.Upsample(mode="linear")``, whose triangular kernel
            # rejects the first image by only 1.7-9.6 dB, stamping the frame
            # grid into the waveform as a mirrored partial either side of every
            # harmonic.  The down path already uses a windowed sinc; this makes
            # the up path agree with it.
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
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

        self.out_tanh = nn.Tanh()

    # The other half of the compile story.  ``torchaudio.functional.resample``
    # builds its sinc kernel from Python ints on every call, and Inductor
    # compiles that construction to a *CPU* kernel, which on Windows needs
    # ``cl.exe``; without MSVC the whole decoder fails with
    # ``InvalidCxxCompiler: Compiler: cl is not found`` and drops to eager.
    # Kept out of the graph rather than replaced: torchaudio's filter is
    # 385/953 taps against the 73/169 a ``FixedLowPass1d`` of the same shape as
    # the upsamplers would build, and its stopband is 135-156 dB against 68-78,
    # so swapping it would be a numerical change smuggled in under a build fix.
    # On a full-band BLIT that margin stops being academic: this filter is what
    # keeps each decimation from folding the harmonics it is discarding.
    @torch.compiler.disable
    def _decimate(self, x: torch.Tensor, orig_freq: int, new_freq: int):
        return torchaudio.functional.resample(
            x.contiguous(),
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
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
        """

        voiced = (f0 > 0).to(f0.dtype)
        # Interpolate the *pitch*, in log Hz, and the gate separately.
        log_f0 = torch.log(f0.clamp_min(1.0))
        log_f0 = F.interpolate(log_f0, size=length, mode="linear", align_corners=False)
        voiced = F.interpolate(voiced, size=length, mode="nearest")
        return torch.exp(log_f0) * voiced

    def _apply_source_gain(self, har_source: torch.Tensor, mel: torch.Tensor):
        """Scale the excitation by an intensity envelope read off ``mel``.

        ``mel`` is this decoder's conditioning -- ``z``, despite the name -- at
        the frame rate; ``har_source`` is (batch, 1, frames * upp).
        """

        if not self.has_source_gain:
            return har_source
        gain = F.softplus(self.source_gain(mel))
        for ups in self.source_gain_ups:
            gain = ups(gain)
        length = har_source.shape[-1]
        if gain.shape[-1] > length:
            gain = gain[..., :length]
        elif gain.shape[-1] < length:
            gain = F.pad(gain, (0, length - gain.shape[-1]), mode="replicate")
        return har_source * gain

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = self._expand_f0(f0, f0_size * self.upp)
        har_source = self.m_source(f0)
        har_source = self._apply_source_gain(har_source, mel)
        x = self.pre_conv(har_source)
        downs = []
        for block, (old_size, new_size) in zip(self.downsample_blocks, self.df0):
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
            x = F.leaky_relu(x, self.leaky_relu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = self.out_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """Fold every weight norm back into its weight, by walking the modules.

        Walking rather than listing layers by name: a hand-written list goes
        stale the moment one is added, and nothing catches it because
        ``Synthesizer`` walks the decoder itself and never calls this.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)