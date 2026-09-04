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
    AntiAliasedActivation,
    AntiAliasedUpsample1d,
    filter_schedule,
)


#: How many of a ``ResBlock`` conv pair's two activations are anti-aliased.
#: ``"half"`` takes the first of each pair and leaves the second plain -- the
#: second reads a signal the first has already band-limited.  Full step at
#: 32 kHz, batch 8, eager: 458.7 ms / 3314 MiB none, 529.2 / 3906 half,
#: 576.8 / 4503 full.
#:
#: The modes are ordered by coverage, and the flag reaches the ``AdaIN``
#: activations wrapped around each ``ResBlock`` as well as the block's own conv
#: pairs:
#:
#:     "none"    nothing
#:     "adain"   the 6 AdaIN per stage, and nothing else   <- ships
#:     "half"    those, plus the first activation of each of the 9 conv pairs
#:     "full"    those, plus both activations of each pair (18)
#:
#: ``"adain"`` exists because the A/B renders off ``G_8833`` put the whole
#: artefact there.  Same weights, same input, only the anti-aliasing changed,
#: the lines read off a spectrogram:
#:
#:     loops only                              lines plainly there
#:     loops + the 18 conv activations         lines plainly there
#:     loops + conv + the 6 AdaIN              clean
#:     the 6 AdaIN alone, nothing else         clean      <- ships
#:
#: So the 18 conv activations and the two loop activations are both doing
#: nothing here that the 6 AdaIN are not already doing, and they are 20 of the
#: 24 sites a stage would otherwise pay for.
ANTIALIAS_MODES = ("none", "adain", "half", "full")


#: Interpolation filter for the trunk's upsamplers, one entry per stage.
#:
#: Zero-stuffing by ``factor`` copies the input spectrum to every multiple of
#: the input rate; what the filter leaves of those copies is an *image* at
#: ``|k*R_in +- f|``.  That is not aliasing and no ``antialias_*`` option
#: touches it -- but an image at ``k*R_in - j*f0`` moves against f0 exactly
#: like a fold does, so the two are indistinguishable in a spectrogram.
#:
#: These were a flat ``12 / 0.90 / 6.0`` until 2026-09-03, which left the worst
#: image at **-37.1 dB** while attenuating the partial that made it by 13.0 dB
#: -- the same rolloff trap as ``AntiAliasedActivation``.  Measured on stage 3
#: with a partial at 0.95 of the input Nyquist, image / passband:
#:
#:     w12 r0.90 b6.0   -37.1 / -13.0    97 taps   <- was
#:     w24 r0.95 b6.0   -68.4 /  -6.0   193
#:     w32 r0.97 b6.0   -68.0 /  -2.0   257
#:     w48 r0.99 b9.0   -94.4 /  -0.1   385
#:
#: Until 2026-09-04 only the last stage got it, on a path-gain argument:
#: inject a known sinusoid after each ``upsample_blocks[k]``, subtract a clean
#: render, and read how much of it reaches the output.
#:
#:     stage 1 image -37.1 dB, path -59.3 dB  ->  -96.4 dB at the output
#:     stage 2 image -37.1     path -49.5     ->  -86.6
#:     stage 3 image -37.1     path -19.9     ->  -57.0
#:
#: Those path gains were measured on **untrained weights**, and that is the
#: flaw: at init every layer is a small random conv and the trunk is a
#: cascade of attenuators, so the earlier a stage the more it appears to be
#: buried.  A trained trunk is nothing of the kind -- the residual blocks of
#: the stages that follow amplify whatever sits in their input, image or not,
#: and there is no reason for them to treat a mirrored partial differently
#: from a real one.  So the -37.1 dB the earlier stages still produced was
#: arriving at the output at whatever gain the trunk had learned, and on a
#: checkpoint that is a mirrored harmonic series about that stage's input
#: Nyquist (1 kHz for stage 2 under ``[5,4,4,4]``, 3.2 kHz under
#: ``(8,8,2,2)``) that no ``antialias_*`` option can touch, because it is not
#: aliasing.  The same schedule feeds ``source_gain_ups``, so the gain
#: envelope carried the same image and stamped it on every harmonic as a
#: sideband.
#:
#: What does limit the earlier stages is edge, not budget:
#: ``AntiAliasedUpsample1d`` pads with ``replicate``, so a longer kernel
#: reaches further into an invented continuation.  Fraction of a 0.4 s
#: training segment whose samples that padding corrupts by more than 1%:
#:
#:     stage 0   40 in, 121 taps   21.0%    <- already the worst, left alone
#:     stage 1  200 in,  97 taps    4.2%    (w24 / 193 taps:  ~14%)
#:     stage 2  800 in,  97 taps    1.1%    (w32 / 257 taps:   3.9%)
#:     stage 3 3200 in, 385 taps    0.3%
#:
#: Stage 2 takes ``w32 r0.97`` (-68 dB image for 3.9% of edge) and stage 1
#: ``w24 r0.95`` (-68 dB for about 14%).  Stage 0 has 40 input samples and
#: its mirror sits about 250 Hz, under the lowest partial of most voices; a
#: longer kernel there would corrupt most of the segment for an image that
#: lands where nothing is.  Prefer widening stage 1 further to touching 0.
DEFAULT_UPSAMPLE_WIDTH = (12, 24, 32, 48)
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.95, 0.97, 0.99)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)


def loop_rates(
    sample_rate: int,
    upsample_rates: "Sequence[int]",
    linear_down_path: bool = False,
    up_activation_after_upsample: bool = False,
):
    """The rate each of the two loops' activations actually runs at, in Hz.

    Every pointwise nonlinearity folds about the Nyquist of *its own* rate, so
    the rate is what decides whether its aliasing lands in the audible band --
    not which block it happens to live in.  ``antialias_stages`` indexes
    ``upsample_conv_blocks`` and therefore cannot express this: it never had a
    way to reach the ``downs[]`` activations at all.

    Bisected on a 35k checkpoint with constant ``z`` and constant f0, excess
    over a matched control at ``8000 - k*f0``:

        har_source / pre_conv / downs[0] / decimate / block    ~ -2 dB
        downs[1] = act(x) @ 8000 Hz                          +38.4 dB

    ``downs[1]`` runs on the excitation already band-limited to 4 kHz *and*
    band-filled, so the activation's second-order products land straight above
    Nyquist; the result is concatenated into ``upsample_conv_blocks[2]`` and the
    fold is in the trunk before the trunk does anything.

    The two structural flags move the *sites* rather than filter them, so they
    change what this returns -- which is why it is returned rather than derived
    at each call site.  ``linear_down_path`` deletes every down activation
    after the first, leaving one site at ``sample_rate`` and none at 8000, 2000
    or 500.  ``up_activation_after_upsample`` runs each up activation on the
    upsampler's *output*, moving every up site one stage up: under
    ``[5, 4, 4, 4]`` the (100, 500, 2000, 8000) sites become
    (500, 2000, 8000, 32000), which is also the set of rates the skip
    connections arrive at.
    """

    total = 1
    for rate in upsample_rates:
        total *= int(rate)
    frame_rate = int(sample_rate) // total

    down, rate = [], int(sample_rate)
    for factor in reversed([int(r) for r in upsample_rates]):
        down.append(rate)
        rate //= factor

    up, rate = [], frame_rate
    for factor in [int(r) for r in upsample_rates]:
        # The rate the site actually runs at: the stage's input rate when the
        # activation precedes the upsampler, its output rate when it follows.
        rate_out = rate * factor
        up.append(rate_out if up_activation_after_upsample else rate)
        rate = rate_out

    if linear_down_path:
        down = down[:1]

    return tuple(down), tuple(up)


class ResBlock(nn.Module):
    """Residual block of dilated convolutions at multiple dilation rates.

    ``antialias`` wraps the activations in :class:`AntiAliasedActivation` --
    see that class for why the oversampling is the mechanism and a smoother
    curve is not a substitute.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
        antialias: str = "none",
        antialias_factor: int = 2,
        antialias_width: int = 16,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope
        if antialias not in ANTIALIAS_MODES:
            raise ValueError(
                f"antialias must be one of {ANTIALIAS_MODES}, not {antialias!r}."
            )
        self.antialias = antialias

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

        # One instance each rather than one shared: they are stateless, but
        # each caches its expanded per-channel kernel, and sharing would thrash
        # that cache on a dispatch-bound step.
        # ``"adain"`` covers the ``AdaIN`` activations that wrap this block and
        # leaves the conv pairs alone -- see ``ANTIALIAS_MODES``.  Inside a
        # ``ResBlock`` it is therefore the same as ``"none"``.
        count = {
            "none": 0,
            "adain": 0,
            "half": len(self.convs1),
            "full": 2 * len(self.convs1),
        }[antialias]
        self.activations = nn.ModuleList(
            [
                AntiAliasedActivation(
                    leaky_relu_slope=leaky_relu_slope,
                    factor=antialias_factor,
                    filter_width=antialias_width,
                )
                for _ in range(count)
            ]
        )

    def forward(self, x: torch.Tensor):
        index = 0
        # ``"adain"`` wraps the activations *around* this block, not the ones
        # inside it, so here it is ``"none"``.
        wraps_pairs = self.antialias in ("half", "full")
        for c1, c2 in zip(self.convs1, self.convs2):
            if not wraps_pairs:
                xt = F.leaky_relu(x, self.leaky_relu_slope)
            else:
                xt = self.activations[index](x)
                index += 1
            xt = c1(xt)
            if self.antialias == "full":
                xt = self.activations[index](xt)
                index += 1
            else:
                xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """Noise-regularised activation.  ``antialias`` is not optional in practice.

    There are two of these wrapped around every ``ResBlock`` -- six per
    ``ParallelResBlock`` -- and until 2026-09-03 nothing could reach them:
    ``antialias`` was passed to the ``ResBlock`` only, so a stage with
    ``antialias_stages`` switched on still ran six raw nonlinearities at its
    own rate, right next to eighteen anti-aliased ones.

    That is not a rounding error.  A/B renders off ``G_8833`` on the reference
    input, same weights, only the anti-aliasing changed: covering the loops and
    the res blocks at 2 and 8 kHz leaves the inharmonic lines plainly visible,
    and adding these six per stage removes them.  Dropping them again from the
    same configuration brings them straight back.  So this follows
    ``antialias`` rather than getting a flag of its own -- a setting that
    leaves the dominant site raw is a trap, and it was one.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
        antialias: bool = False,
        antialias_factor: int = 2,
        antialias_width: int = 16,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.antialias = bool(antialias)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = (
            AntiAliasedActivation(
                leaky_relu_slope=leaky_relu_slope,
                factor=antialias_factor,
                filter_width=antialias_width,
            )
            if self.antialias
            else nn.LeakyReLU(leaky_relu_slope)
        )

    def forward(self, x: torch.Tensor):
        # The noise is a training-time regulariser, and at inference it is pure
        # cost: this module runs six times per decoder stage, so at the output
        # rate it draws 13.5 M samples per forward.  Measured on CPU at batch 4
        # over 0.4 s, dropping it in eval takes the whole generator from 418 ms
        # to 313 ms -- 25% of the forward.  Inference stays stochastic anyway:
        # the NSF source draws its own noise, and that one is not optional --
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
        antialias: str = "none",
        antialias_factor: int = 2,
        antialias_width: int = 16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.antialias = antialias

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
                        antialias=antialias != "none",
                        antialias_factor=antialias_factor,
                        antialias_width=antialias_width,
                    ),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                        antialias=antialias,
                        antialias_factor=antialias_factor,
                        antialias_width=antialias_width,
                    ),
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                        antialias=antialias != "none",
                        antialias_factor=antialias_factor,
                        antialias_width=antialias_width,
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class SineGenerator(nn.Module):
    """Sine + additive-noise harmonic excitation source.

    The only one left.  ``comb`` and ``bank`` were removed on 2026-09-03: the
    inharmonic lines they were being traded against turned out to be the
    ``AdaIN`` activations, and on a fixed trunk the bank had been *negative*
    against the sine once ``source_gain`` was on (multi-scale mel 1.7357 ->
    1.8057 for the bank, 1.9714 -> 1.7418 for the sine).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

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
        # the source module this comes from.  That path is unused here.
        nn.init.ones_(self.merge[0].weight)

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim), dim = fundamental + overtones."""
        # rad_values is F0 in rad mod 1 (the integer cycle count doesn't affect phase)
        rad_values = (f0_values / self.sampling_rate) % 1

        # random initial phase per harmonic, none for the fundamental
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    # Inductor cannot compile this body.  ``_f02sine`` is a cumsum over the
    # sample axis, and Inductor lowers it to a ``SplitScan`` whose codegen
    # raises ``TypeError: list indices must be integers or slices, not
    # NoneType`` -- reproduced on torch 2.10 + cu130, RTX 5060.  A failure
    # inside the compiled region takes the *whole* decoder down with it, so
    # ``enable_decoder_compile`` fell back to eager for every step.
    #
    # Everything up to ``merge`` runs under ``no_grad`` and is a pure function
    # of f0, so keeping it out of the graph costs no fusion.
    @torch.compiler.disable
    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # merge with grad
        return self.merge(sine_waves)


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2: RefineGAN with its signal-path defects fixed.

    Downsamples/upchannels the excitation source, fuses it with the latent, and
    upsamples through parallel residual blocks.  Against the original:
    descending stage rates, a windowed-sinc interpolation filter that crops its
    own group delay, anti-aliased ``AdaIN`` activations at the stages running
    at 2 and 8 kHz, and an excitation gain projected from the conditioning.

    Args:
        source_gain (bool, optional): Scale the excitation by an intensity
            envelope projected from the conditioning, as RefineGAN's paper
            does with the mel. The ``softplus`` that makes it positive runs
            after the interpolators, at the output rate -- see
            ``_apply_source_gain``, where the other order is measured putting
            10.8% of the envelope's samples below zero. Defaults to False.
        antialias_rates (Sequence[int], optional): Which of the two loops'
            activation rates, in Hz, get anti-aliased activations -- see
            ``loop_rates``. This is the knob that matters: it selects by the
            rate a nonlinearity actually runs at, so it can reach the
            ``downs[]`` activation at 8 kHz where the fold at ``8000 - k*f0``
            is created, and naming ``sample_rate`` also covers the activation
            before ``conv_post``. Protecting every rate the decoder has costs
            12% of the step against raw activations; the shipped config does
            that. Defaults to none.
        antialias_stages (Sequence[int], optional): Which upsampling stages get
            anti-aliased activations in their residual blocks. Defaults to none,
            and the shipped config leaves it there: it addresses the residual
            blocks' *internal* activations, never the loops', it did nothing
            measurable against the mirroring, and at ``"full"`` on one stage it
            cost 47 ms and 795 MiB at batch 8 -- more than protecting all five
            loop rates with a filter three times as long.
        antialias_factor (int, optional): oversampling factor for the stages'
            anti-aliased activations.  This is what sets the alias floor, not
            the filter.  2 by default; 3 buys about 7 dB for a quarter of the
            step and 4 about 10.5 for a third.
        antialias_width (int or Sequence[int], optional): the anti-aliased
            activations' filter width, scalar or one per stage.  16 by default;
            8 is the knee -- it gives back about a quarter of the round trip
            for around 1 dB, and 4 saves little more and costs three.
        antialias_rate_width (int, optional): the same for the loops' and the
            output activation.  ``None`` takes the widest stage value.
        antialias_rate_factor (int, optional): the same for the two loops' and
            the output activation -- nine sites rather than the stages' dozens,
            so a higher factor is affordable there.  ``None`` follows
            ``antialias_factor``.
        antialias (str, optional): ``"none"``, ``"half"`` or ``"full"`` -- how
            many of each conv pair's two activations are anti-aliased in those
            stages. Defaults to ``"half"``, and is forced to ``"none"`` when
            ``antialias_stages`` is empty.
        linear_down_path (bool, optional): drop every activation in the down
            loop after the first. Anti-aliasing is the generic answer for a
            nonlinearity that has to exist; three of these do not. ``downs[]``
            exists to hand the trunk band-limited copies of the excitation, and
            the only activation there doing work the trunk needs is the first,
            at ``sample_rate``, which rectifies the sine and *creates* the
            harmonics. Applying ``leaky_relu`` again at 8 kHz to a signal that
            is already rectified and already band-limited to 4 kHz adds nothing
            the trunk cannot get from a linear conv, and it is the site that
            bisected at **+38.4 dB** of excess at ``8000 - k*f0``. This removes
            three fold sites (8000, 2000, 500 under ``[5, 4, 4, 4]``) outright,
            at *negative* cost -- no activation and no round trip -- where a
            filter can only ever attenuate what they make. Defaults to False.
        up_activation_after_upsample (bool, optional): run each up loop's
            activation on the upsampler's output instead of its input. Same
            nonlinearity, same parameters, different rate: stage 3's runs at
            32 kHz rather than 8 kHz, so it folds about 16 kHz instead of about
            4 kHz. That is what an anti-aliased activation does -- oversample,
            activate, come back -- except at the stage's own factor and with no
            trip back, because the signal is going to that rate anyway. It
            costs one activation on a tensor ``rate`` times longer, which is
            less than the round trip a factor-``rate`` ``AntiAliasedActivation``
            would pay, and it moves the site's rate, so ``antialias_rates`` has
            to name the new one. Defaults to False.

    ``linear_down_path`` and ``up_activation_after_upsample`` change what the
    decoder computes at fixed weights, so neither is a patch on a checkpoint:
    both are carried in ``decoder_layout`` and refuse to load across.
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
        antialias_stages: "Sequence[int] | None" = None,
        antialias: str = "half",
        source_gain: bool = False,
        antialias_rates: "Sequence[int] | None" = None,
        antialias_factor: int = 2,
        antialias_rate_factor: "int | None" = None,
        antialias_width: "int | Sequence[int]" = 16,
        antialias_rate_width: "int | None" = None,
        linear_down_path: bool = False,
        up_activation_after_upsample: bool = False,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing
        # Structural: these delete or move fold sites rather than filtering
        # them, so they decide which rates ``loop_rates`` reports below.
        self.linear_down_path = bool(linear_down_path)
        self.up_activation_after_upsample = bool(up_activation_after_upsample)

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
        # A reorder is invisible in the state dict -- all 271 tensors keep their
        # keys and shapes -- so ``rvc.train.utils.decoder_layout`` writes it into
        # the checkpoint.
        stages = () if antialias_stages is None else tuple(int(s) for s in antialias_stages)
        if any(s < 0 or s >= len(upsample_rates) for s in stages):
            raise ValueError(
                f"antialias_stages must index the {len(upsample_rates)} "
                f"upsampling stages, received {stages}."
            )
        self.antialias_stages = tuple(sorted(set(stages)))
        self.antialias = antialias if self.antialias_stages else "none"

        # The oversampling factor, per site *group*.  It is what sets the alias
        # floor -- the filter cannot, since ``leaky_relu`` makes products of
        # every order and only the ones that fit under ``factor`` are removed
        # rather than folded -- and the two groups have very different prices,
        # which is why they are separate knobs and not one.
        #
        # Measured on an isolated activation against a 32x reference, alias
        # against harmonic energy at the occupancies a stage actually presents:
        #
        #     factor      1/2      1/4      1/5      1/8
        #     2         -42.7    -47.5    -49.2    -56.1
        #     3         -49.7    -54.4    -55.8    -63.1
        #     4         -53.2    -57.5    -59.5    -66.8
        #
        # 3 had never been tried: the choice was read as 2-or-4 and 4 was
        # rejected for costing 40% of the step, which left 7 of the available
        # 10.5 dB on the table.  On the nine loop sites, fwd+bwd at batch 8 over
        # 0.4 s, 2 -> 3 is 162 -> 199 ms and 3 -> 4 is 199 -> 224: most of the
        # quality is in the first step and most of the cost in the second.
        #
        # ``None`` means "whatever the stages use", so a config that names only
        # one of them cannot end up with the two silently disagreeing.
        for name, value in (
            ("antialias_factor", antialias_factor),
            ("antialias_rate_factor", antialias_rate_factor),
        ):
            if value is None:
                continue
            if int(value) < 2:
                raise ValueError(
                    f"{name} is an oversampling factor and must be at least 2, "
                    f"not {value!r}: at 1 the round trip does no oversampling "
                    f"and is a bare lowpass."
                )
        self.antialias_factor = int(antialias_factor)
        self.antialias_rate_factor = (
            self.antialias_factor
            if antialias_rate_factor is None
            else int(antialias_rate_factor)
        )

        # Filter width, per stage for the res-block sites and one value for the
        # loops.  Per stage because the price is not uniform: at the small
        # shapes the round trip is launch-bound and the taps are free, while at
        # the output-rate ones they are about 45% of it.  So ``[16, 16, 8]``
        # spends nothing to keep the wider filter where it costs nothing.
        #
        # Measured on one activation at the stage-3 shape, interleaved and
        # repeated because a single-shot timing here swings 13%:
        #
        #     f2 w16   2.75 ms      f2 w8   2.02 ms      f2 w4   1.84 ms
        #
        # and the quality that buys, on a pure sine at 752 / 1502 / 3002 Hz:
        # w16 -67.3 / -59.4 / -50.6 against w8 -65.6 / -59.2 / -49.9.  So 8 is
        # the knee: 4 saves another 9% of the time and costs 3.3 dB more.
        #
        # Width, rolloff and beta are one design, and only the width moves
        # here.  That is safe in this direction -- a *narrower* filter at the
        # same 0.99 rolloff trades stopband for taps and nothing else -- but it
        # is why the minimum is enforced rather than left to the caller.
        self.antialias_width = tuple(
            int(value)
            for value in filter_schedule(
                antialias_width, count, "antialias_width", 1
            )
        )
        self.antialias_rate_width = (
            max(self.antialias_width)
            if antialias_rate_width is None
            else int(antialias_rate_width)
        )

        # Anti-aliasing selected by rate rather than by block, so it can reach
        # the ``downs[]`` activations -- which is where the fold at
        # ``8000 - k*f0`` is created.  See ``loop_rates``.
        self.down_rates, self.up_rates = loop_rates(
            sample_rate,
            upsample_rates,
            linear_down_path=self.linear_down_path,
            up_activation_after_upsample=self.up_activation_after_upsample,
        )
        protected = {int(r) for r in (antialias_rates or ())}
        unknown = protected - set(self.down_rates) - set(self.up_rates)
        if unknown:
            raise ValueError(
                f"antialias_rates {sorted(unknown)} match no activation rate; "
                f"this decoder runs its down loop at {self.down_rates} and its "
                f"up loop at {self.up_rates} Hz. Both structural flags move "
                f"these: linear_down_path={self.linear_down_path} deletes "
                f"every down site after the first and "
                f"up_activation_after_upsample="
                f"{self.up_activation_after_upsample} moves each up site to "
                f"its stage's output rate."
            )
        self.antialias_rates = tuple(sorted(protected))
        # ``None`` where a rate is not protected, so the forward stays a plain
        # index and the module list keeps one instance per site -- each caches
        # its expanded kernel, and sharing would thrash that cache.
        #
        # One entry per *site*, which is what makes ``linear_down_path`` a
        # single truncation: with it on ``down_rates`` is one long, this list
        # is one long, and the forward's ``index < len(...)`` leaves the
        # remaining stages with no nonlinearity at all -- not a raw one.
        self.down_activations = nn.ModuleList(
            [
                AntiAliasedActivation(
                    leaky_relu_slope=leaky_relu_slope,
                    factor=self.antialias_rate_factor,
                    filter_width=self.antialias_rate_width,
                )
                if rate in protected
                else nn.Identity()
                for rate in self.down_rates
            ]
        )
        self.up_activations = nn.ModuleList(
            [
                AntiAliasedActivation(
                    leaky_relu_slope=leaky_relu_slope,
                    factor=self.antialias_rate_factor,
                    filter_width=self.antialias_rate_width,
                )
                if rate in protected
                else nn.Identity()
                for rate in self.up_rates
            ]
        )

        # The last nonlinearity before ``conv_post``, which runs at the output
        # rate and folds about the output Nyquist -- so whatever it creates
        # lands in the render with one convolution left to traverse, the
        # shortest path any site in this decoder has.  Nothing reached it:
        # ``antialias_stages`` indexes the residual blocks and
        # ``antialias_rates`` indexed the two loops' lists, and this activation
        # is in neither.  It follows ``sample_rate in antialias_rates``, which
        # is ``down_rates[0]`` -- the same rate as ``downs[0]``, and the two are
        # the only sites at the output rate outside the last stage's blocks --
        # rather than getting a flag of its own: a decoder that protects the
        # output rate everywhere except the one place it reaches the output
        # would be the same trap ``AdaIN`` was.
        self.out_activation = (
            AntiAliasedActivation(
                leaky_relu_slope=leaky_relu_slope,
                factor=self.antialias_rate_factor,
                filter_width=self.antialias_rate_width,
            )
            if int(sample_rate) in protected
            else nn.Identity()
        )

        # ``int``, not the ``np.int64`` ``np.prod`` returns.  Dynamo wraps a
        # numpy scalar used inside a traced function as a *CPU* tensor, and one
        # CPU node is enough to make Inductor emit a C++ kernel -- which on
        # Windows needs ``cl.exe`` and fails the whole compile with
        # ``InvalidCxxCompiler``.  This is
        # one of three independent CPU/codegen sources in this decoder -- the
        # other two are marked ``torch.compiler.disable`` below -- and removing
        # any one of them alone still fails the compile.
        self.upp = int(np.prod(upsample_rates))

        # ``comb`` and ``bank`` were removed on 2026-09-03: the artefact they
        # were being traded against turned out to be the ``AdaIN``
        # activations, and neither had ever beaten the sine once the excitation
        # gain was on.  ``excitation_source`` in ``rvc/train/utils.py`` still
        # names the mismatch if an old checkpoint is loaded, since the state
        # dicts differ.
        self.source_type = "sine"
        self.m_source = SineGenerator(sample_rate)

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
        # 1.9714 -> 1.7418, comb 1.9649 -> 1.8363, bank 1.7357 -> 1.8057.  A
        # large win for the weak sources and *negative* on the bank, which
        # already carries what the gain buys -- they are alternatives, not a
        # stack.  193 parameters.
        # With a flat, f0-driven source the trunk gets its harmonics without
        # consulting ``z`` at all, and the KL falls because the decoder needs
        # less rather than because the prior caught up.  Projecting the source's
        # *envelope* from the conditioning puts ``z`` back on the critical path
        # for harmonic content: the trunk cannot get the envelope from f0, so
        # the posterior has to carry it and the KL has to pay for it.
        #
        # What stays f0-driven is the scaffolding -- which partials exist, at
        # what frequencies, under the Nyquist mask -- so the source is still
        # alias-free and in tune whatever the projection predicts.  A bad ``z``
        # can dull or brighten it; it cannot detune it.
        self.has_source_gain = bool(source_gain)
        #: Where the ``softplus`` runs, for ``decoder_layout``.  Not an option
        #: -- there is one order and this names it -- but it decides what the
        #: decoder computes and leaves no trace in the weights, so a checkpoint
        #: from before the move has to be refused rather than loaded and
        #: rendered through the wrong path.  ``"pre"`` is every run before
        #: 2026-09-04.
        self.source_gain_order = "post" if self.has_source_gain else None
        if self.has_source_gain:
            self.source_gain = nn.Conv1d(num_mels, 1, 1)
            # Identity at initialisation: ``softplus(0.5413) = 1.0`` with zero
            # weights, so a run that switches this on starts from exactly the
            # excitation it had before and the projection has to earn every
            # departure from unity.  It also means the module cannot silently
            # rescale a fine-tune's source on step zero.  The interpolators
            # below preserve DC, so this holds with the ``softplus`` after them
            # as it did with it before -- more tightly, in fact: the largest
            # per-sample departure from 1 over a segment is 7.3e-5 this way
            # round against 2.1e-4 the other, both of them the filters' edge.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic.  A smooth envelope makes
            # ``F.interpolate`` good enough (-81.9 dB against -82.5 here), but
            # this gain is learned: on a frame-rate-white one the same
            # comparison is -50.3 against -82.6.
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
                    antialias=(
                        self.antialias
                        if stage in self.antialias_stages
                        else "none"
                    ),
                    antialias_factor=self.antialias_factor,
                    antialias_width=self.antialias_width[stage],
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

        # The output ``tanh``.  It is a pointwise nonlinearity at the output
        # rate with nothing after it, so it folds like any other -- and until
        # now nothing reached it.
        #
        # It is not the decoder's problem, and the measurement says so: against
        # a 32x reference of itself it contributes -61.4 dB at ``|max| = 0.9``
        # and saturates around -57 even at 3.0, which is 10-25 dB below the
        # activation floor.  ``tanh`` is smooth, so its harmonic series decays
        # fast; it is the *corner* of ``leaky_relu`` that makes products of
        # every order.  Wrapping it closes a real hole, not the one that
        # matters.
        #
        # Nearly free, which is why it is here anyway: ``conv_post`` emits one
        # channel, so this runs on ``(B, 1, T)`` -- a thirty-second of the cost
        # of any activation inside the last stage.  It follows
        # ``sample_rate in antialias_rates``, like ``out_activation``, rather
        # than taking a flag of its own.
        self.out_tanh = (
            AntiAliasedActivation(
                nn.Tanh(),
                factor=self.antialias_rate_factor,
                filter_width=self.antialias_rate_width,
            )
            if int(sample_rate) in protected
            else nn.Tanh()
        )

    # The other half of the compile story.  ``torchaudio.functional.resample``
    # builds its sinc kernel from Python ints on every call, and Inductor
    # compiles that construction to a *CPU* kernel, which on Windows needs
    # ``cl.exe``; without MSVC the whole decoder fails with
    # ``InvalidCxxCompiler: Compiler: cl is not found`` and drops to eager.
    # Kept out of the graph rather than replaced: torchaudio's filter is
    # 385/953 taps against the 73/169 a ``FixedLowPass1d`` of the same shape as
    # the upsamplers would build, and its stopband is 135-156 dB against 68-78,
    # so swapping it would be a numerical change smuggled in under a build fix.
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

    def _apply_source_gain(self, har_source: torch.Tensor, mel: torch.Tensor):
        """Scale the excitation by an intensity envelope read off ``mel``.

        ``mel`` is this decoder's conditioning -- ``z``, despite the name -- at
        the frame rate; ``har_source`` is (batch, 1, frames * upp).

        The ``softplus`` runs at the output rate, after the interpolators, and
        that is not a stylistic choice.  Run at the frame rate it guarantees a
        positive envelope to a *sinc* that then rings through zero: the
        upsampled gain goes negative, and a negative gain is the excitation
        changing sign inside a frame -- every harmonic phase-flipped, which no
        filter downstream can undo because nothing was aliased.  Measured with
        this decoder's own filter schedule at ``[5, 4, 4, 4]``, minimum of the
        upsampled gain and the fraction of samples below zero:

            envelope at 100 Hz    softplus first        softplus last
            smooth                +0.127   0.00%        +0.127
            frame-rate-white      -1.658  10.81%        +0.004
            step (an onset)       -0.330   1.50%        +0.023

        The white row is not a hypothetical: it is the same premise that put
        ``AntiAliasedUpsample1d`` here rather than ``F.interpolate`` -- the gain
        is learned, and nothing holds it smooth.  Order last, the envelope is
        positive by construction at every sample.

        It is also the cheaper end to run it at, which is unusual: the tensor
        is one channel, so a pointwise call on it at 32 kHz costs about as much
        as the crop below.
        """

        if not self.has_source_gain:
            return har_source
        gain = self.source_gain(mel)
        for ups in self.source_gain_ups:
            gain = ups(gain)
        # After the interpolators, not before -- see above.
        gain = F.softplus(gain)
        length = har_source.shape[-1]
        if gain.shape[-1] > length:
            gain = gain[..., :length]
        elif gain.shape[-1] < length:
            gain = F.pad(gain, (0, length - gain.shape[-1]), mode="replicate")
        return har_source * gain

    def _activate(self, x: torch.Tensor, activation: nn.Module):
        """``activation`` if it is one, a plain ``leaky_relu`` if it is ``Identity``.

        ``Identity`` here means "this site is not anti-aliased", never "this
        site has no nonlinearity" -- a deleted site is deleted by shortening
        the module list, which is what ``linear_down_path`` does.
        """

        if isinstance(activation, nn.Identity):
            return F.leaky_relu(x, self.leaky_relu_slope)
        return activation(x)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        f0 = F.interpolate(f0.unsqueeze(1), size=f0_size * self.upp, mode="linear")
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        har_source = self._apply_source_gain(har_source, mel)
        x = self.pre_conv(har_source)
        downs = []
        for index, (block, (old_size, new_size)) in enumerate(
            zip(self.downsample_blocks, self.df0)
        ):
            # Past the first stage ``linear_down_path`` leaves the path
            # linear: conv, decimate, conv.  ``downs[]`` is a band-limited copy
            # of an excitation the first activation has already rectified, and
            # the nonlinearity here only folded it.
            if index < len(self.down_activations):
                x = self._activate(x, self.down_activations[index])
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for index, (ups, res, down) in enumerate(
            zip(
                self.upsample_blocks,
                self.upsample_conv_blocks,
                reversed(downs),
            )
        ):
            activation = self.up_activations[index]
            checkpointing = self.training and self.checkpointing

            if not self.up_activation_after_upsample:
                x = self._activate(x, activation)

            x = (
                checkpoint(ups, x, use_reentrant=False)
                if checkpointing
                else ups(x)
            )

            # Same nonlinearity at ``rate`` times the input rate, so it folds
            # about the stage's *output* Nyquist rather than its input one.  It
            # runs outside the checkpointed region on purpose: the upsampler's
            # output is already that region's boundary and is kept either way,
            # so wrapping it would recompute the interpolation to save a tensor
            # that is being saved anyway.
            if self.up_activation_after_upsample:
                x = self._activate(x, activation)

            x = torch.cat([x, down], dim=1)
            x = (
                checkpoint(res, x, use_reentrant=False)
                if checkpointing
                else res(x)
            )

        x = self._activate(x, self.out_activation)
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
