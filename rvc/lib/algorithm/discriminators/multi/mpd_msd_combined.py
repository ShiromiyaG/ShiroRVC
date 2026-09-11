import math
import traceback
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.discriminators.san import SANConv1d, SANConv2d, san_tail
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from rvc.train.messages import (
    DISCRIMINATOR_COMPILE_ENABLE_FAILED,
    DISCRIMINATOR_COMPILE_RUNTIME_FAILED,
)
from rvc.lib.terminal import warning

#: Applio's branch layouts, under their names so a diff is a diff.  ``v2`` is
#: HiFi-GAN's; ``v3`` trades its three widest period branches for three
#: multi-resolution spectrogram branches.  ``v4`` is this fork's: ``v3`` minus
#: its longest period.
#:
#: ``DiscriminatorR``'s frequency axis can be decimated -- ``(1, 2, 2)`` in the
#: last two layers measured no worse than Applio's ``(1, 1, 1)`` on both probe
#: defects and 24% cheaper (81.1 vs 107.0 ms, 342 vs 396 MiB, batch 8 / 0.4 s).
#: Reach it through ``d_frequency_strides``; it is not a version.
#:
#: What must not be touched is the 512-point branch's 50-sample hop.  The
#: branch catches frame-rate mirroring as a *temporal* modulation, so 100 Hz
#: needs a frame shorter than its 10 ms period; a 4096-point branch (10.9 ms)
#: drops held-out accuracy to chance, and so does ChouwaGAN's 128/256/512.
REFERENCE_SAMPLE_RATE = 22050


def rate_scaled_periods(periods, sample_rate, reference_rate=REFERENCE_SAMPLE_RATE):
    """The period set that keeps HiFi-GAN's *time scales* at another rate.

    A period-``p`` branch folds onto a grid at ``sr / p`` Hz and its receptive
    field spans ``647 * p / sr`` seconds, so both meanings of a period hold only
    if ``p`` scales with the rate.  ``[2, 3, 5, 7, 11]`` was chosen at 22.05 kHz
    and carried everywhere unchanged, which empties the *slow* end: at 32 kHz
    the longest branch drops from 323 ms to 222, and pitch structure lives
    there.

    Targets are rounded to the nearest unused prime in log space -- prime
    because two periods sharing a factor fold onto overlapping samples and
    become one branch at two branches' cost, log because the quantity preserved
    is a ratio.  ``reference_rate`` returns the input unchanged, which is what
    makes this a derivation rather than a new design.
    """

    def is_prime(value):
        return value > 1 and all(value % f for f in range(2, int(value**0.5) + 1))

    candidates = [value for value in range(2, 512) if is_prime(value)]
    used, scaled = set(), []
    for period in periods:
        target = int(period) * float(sample_rate) / float(reference_rate)
        best = min(
            (value for value in candidates if value not in used),
            key=lambda value: abs(math.log(value / target)),
        )
        used.add(best)
        scaled.append(best)
    return tuple(sorted(scaled))


#: The reviewed, frozen result of ``rate_scaled_periods`` for the versions and
#: rates that ship.  A *pin*, not a source: the constructor derives, and a test
#: asserts the two agree -- that is what keeps the rule and the numbers together.
#:
#: Membership marks a version as rate-scaled.  ``v1``/``v2`` are absent on
#: purpose: v2's eight periods are Applio's and every RVC v2 pretrained D is
#: trained against them, so they stay verbatim at every rate.
#:
#: Deriving instead of writing the list into each config costs one thing: an
#: unkeyed checkpoint from before the scaling no longer resumes into a
#: ``d_version``-only config.  It fails in ``assert_periods_match``, naming the
#: ``d_periods`` to set.  A period is invisible in every weight
#: (``DiscriminatorP``'s kernels are ``(k, 1)`` whatever ``p`` is), so that
#: guard is the only thing that can tell.
PERIODS_BY_RATE = {
    "v3": {32000: (3, 5, 7, 11, 17)},
    "v4": {32000: (3, 5, 7, 11)},
}


#: The three multi-resolution spectrogram branches, shared by ``v3`` and
#: ``v4``.  The 512-point branch's 50-sample hop is the one that must not be
#: touched -- see the note above ``REFERENCE_SAMPLE_RATE``.
V3_RESOLUTIONS = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]

DISCRIMINATOR_VERSIONS = {
    "v1": ([2, 3, 5, 7, 11, 17], [], (1, 1, 1)),
    "v2": ([2, 3, 5, 7, 11, 17, 23, 37], [], (1, 1, 1)),
    "v3": ([2, 3, 5, 7, 11], V3_RESOLUTIONS, (1, 1, 1)),
    # ``v3`` minus its longest period (17 at 32 kHz), and nothing else.  The
    # spectrogram branches keep full frequency resolution: they are the only
    # part of this D with any resolution above 10 kHz, which is the band being
    # chased.  A longer period folds at a lower rate, so it is the branch least
    # able to say anything up there -- the cheapest one to lose.
    "v4": ([2, 3, 5, 7], V3_RESOLUTIONS, (1, 1, 1)),
}


#: How much of the objective UnivHD is allowed to be, per version.
#:
#: ``1.0`` -- the paper's additive setting, and what every version outside this
#: table gets -- is wrong for the branch set ``v4`` actually runs, and the
#: pretrain that showed it is the argument for the number here.  ``disc_sep``
#: is ``mean(real logit) - mean(fake logit)`` per branch, i.e. how decisively a
#: head is separating.  Measured on a 32 kHz pretrain over these nine branches,
#: SAN on, rolling mean over 50 steps, between steps 2k and 8.5k:
#:
#:     branch            separation
#:     univhd               5.1 - 7.7
#:     msd                  0.7 - 2.1
#:     period_3             0.8 - 1.0
#:     period_5             0.4 - 0.6
#:     period_7             0.3 - 0.7
#:     period_11            0.5 - 0.8
#:     resolution_512       0.3
#:     resolution_1024      0.2 - 0.3
#:     resolution_2048      0.2 - 0.3
#:
#: An order of magnitude clear of the other eight, and it stayed there for the
#: whole window rather than converging toward them.  Because the generator's
#: term is ``(1 - dg)^2``, a head separating by ~6 contributes a term ~10x an
#: average branch's, so one of nine heads was most of ``loss_adv`` -- and the
#: gradient it put into the decoder is most of what drove that run's decoder
#: grad norm to 8-15 x 10^3.
#:
#: ``0.15`` is chosen to leave UnivHD the loudest single head without leaving
#: it the only one: it is roughly the ratio between that separation and the
#: rest, so the branch stops being most of the term while still outweighing
#: any one of the other eight.  It is a starting point, not a fixed point --
#: ``d_univhd_weight`` overrides it on any version, and ``disc_sep_50/univhd``
#: against the other branches is the series that says whether it landed.
#:
#: Only ``v4`` is listed.  ``v1``-``v3`` predate UnivHD here and nothing has
#: been trained with it on them, so there is no measurement behind a number for
#: them and they keep the paper's 1.0.
UNIVHD_WEIGHT_BY_VERSION = {
    "v4": 0.15,
}

#: What a branch with no entry in a weight table is worth: the plain sum every
#: HiFi-GAN-lineage discriminator has always used.
DEFAULT_BRANCH_WEIGHT = 1.0


def univhd_weight_for(version: str) -> float:
    """The default UnivHD weight for a version; 1.0 where none is pinned."""

    return float(UNIVHD_WEIGHT_BY_VERSION.get(str(version), DEFAULT_BRANCH_WEIGHT))


def periods_for(version: str, sample_rate: int):
    """The frozen set for a version at a rate; raises rather than guessing."""

    try:
        return PERIODS_BY_RATE[version][int(sample_rate)]
    except KeyError as error:
        raise KeyError(
            f"No frozen period set for version {version!r} at {sample_rate} Hz; "
            f"known: {{v: sorted(r) for v, r in PERIODS_BY_RATE.items()}}."
        ) from error


class MPD_MSD_Combined(torch.nn.Module):
    """Multi-period, multi-scale (and optionally multi-resolution / UnivHD) discriminators combined."""

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        version: str = "v2",
        periods=None,
        resolutions=None,
        frequency_strides=None,
        use_msd: bool = True,
        use_fast_mpd: bool = False,
        sample_rate: int = 32000,
        use_univhd: bool = False,
        use_san: bool = False,
        univhd_n_fft: int = 2048,
        univhd_hop_length: int = 256,
        univhd_harmonics: int = 10,
        univhd_bins_per_octave: int = 24,
        univhd_f_min: float = 80.0,
        univhd_channels: int = 32,
        univhd_half_harmonic: bool = True,
        univhd_weight: Optional[float] = None,
        mrd_fp32_input: bool = True,
    ):
        """``version`` picks a preset; the overrides edit it branch by branch.

        ``None`` means "whatever the version says"; an empty list means "none of
        this family" -- a distinction a falsy check would lose.

        Branch costs, fwd+bwd at batch 8 over 0.4 s: scale 16 ms / 168 MiB,
        each period ~9 ms / ~250 MiB, each spectrogram ~30 ms / ~430 MiB.
        Dropping a period is a bigger lever than anything inside a branch.

        ``use_fast_mpd`` swaps every period branch for
        :class:`FastDiscriminatorP` -- 15x fewer parameters at indistinguishable
        probe accuracy; see that class for the numbers and for what the probe
        cannot see.  It changes every period branch's shapes, so a strict load
        of a checkpoint trained the other way fails on its own.

        ``sample_rate`` selects branch *frequencies*, not just UnivHD's
        filterbank: for the versions in ``PERIODS_BY_RATE`` the periods are
        derived from it.  ``use_univhd`` appends the harmonic branch (arXiv
        2512.03486) for +9% time and memory and +0.33 M parameters -- additive,
        as the paper runs it.

        ``mrd_fp32_input`` runs each spectrogram branch's STFT and first conv
        outside FP16 autocast; see :meth:`DiscriminatorR.forward`.  It adds no
        parameters, so a checkpoint loads either way, and without autocast it
        changes nothing.
        """

        super().__init__()
        if version not in DISCRIMINATOR_VERSIONS:
            raise ValueError(
                f"Unknown discriminator version {version!r}: "
                f"expected one of {sorted(DISCRIMINATOR_VERSIONS)}."
            )
        preset_periods, preset_resolutions, preset_strides = DISCRIMINATOR_VERSIONS[
            version
        ]
        if periods is None:
            # ``version in PERIODS_BY_RATE`` is the rate-scaled marker; v1 and
            # v2 keep Applio's set at every rate.  Derived rather than looked
            # up so an unfrozen rate still gets a correct set instead of an
            # exception -- the frozen table is the reviewed pin, not the only
            # legal answer, and ``rate_scaled_periods`` is the rule it pins.
            periods = (
                list(rate_scaled_periods(preset_periods, sample_rate))
                if version in PERIODS_BY_RATE
                else preset_periods
            )
        else:
            periods = list(periods)
        resolutions = (
            preset_resolutions if resolutions is None else [list(r) for r in resolutions]
        )
        frequency_strides = (
            preset_strides if frequency_strides is None else tuple(frequency_strides)
        )
        if any(len(r) != 3 for r in resolutions):
            raise ValueError(
                "Each resolution is [n_fft, hop_length, win_length]; "
                f"received {resolutions}."
            )
        self.version = version
        self.periods = tuple(int(p) for p in periods)
        self.resolutions = tuple(tuple(int(v) for v in r) for r in resolutions)
        self.frequency_strides = tuple(int(s) for s in frequency_strides)
        self.use_msd = bool(use_msd)
        self.use_fast_mpd = bool(use_fast_mpd)
        self.mrd_fp32_input = bool(mrd_fp32_input)
        self.use_univhd = bool(use_univhd)
        # ``None`` means "whatever the version pins", the same convention the
        # branch overrides above use; an explicit number wins on any version.
        self.univhd_weight = (
            univhd_weight_for(version)
            if univhd_weight is None
            else float(univhd_weight)
        )
        if self.univhd_weight < 0.0:
            raise ValueError(
                f"univhd_weight is a loss weight and cannot be negative; "
                f"received {self.univhd_weight}."
            )
        # ``train.py`` reads this to decide whether the losses take their SAN
        # form; it is an attribute rather than a lookup on the branches so a
        # discriminator that is *asked* for SAN and could not build it cannot
        # report otherwise.
        self.use_san = bool(use_san)
        self.supports_san = self.use_san
        self.sample_rate = int(sample_rate)
        # Read by ``forward``: spectral norm's power iteration advances once
        # per weight access, so a batched pass would run it once where two
        # separate passes run it twice.  That is a change to the training
        # dynamics, not an optimisation, so the batched path is off there.
        self.use_spectral_norm = bool(use_spectral_norm)
        self.use_checkpointing = use_checkpointing
        branches = []
        if self.use_msd:
            branches.append(
                DiscriminatorS(use_spectral_norm=use_spectral_norm, use_san=self.use_san)
            )
        period_branch = FastDiscriminatorP if self.use_fast_mpd else DiscriminatorP
        branches += [
            period_branch(p, use_spectral_norm=use_spectral_norm, use_san=self.use_san)
            for p in self.periods
        ]
        branches += [
            DiscriminatorR(
                list(r),
                use_spectral_norm=use_spectral_norm,
                frequency_strides=self.frequency_strides,
                use_san=self.use_san,
                fp32_input=self.mrd_fp32_input,
            )
            for r in self.resolutions
        ]
        if self.use_univhd:
            from rvc.lib.algorithm.discriminators.single import UnivHDDiscriminator

            branches.append(
                UnivHDDiscriminator(
                    sample_rate=self.sample_rate,
                    n_fft=int(univhd_n_fft),
                    hop_length=int(univhd_hop_length),
                    harmonics=int(univhd_harmonics),
                    bins_per_octave=int(univhd_bins_per_octave),
                    f_min=float(univhd_f_min),
                    channels=int(univhd_channels),
                    half_harmonic=bool(univhd_half_harmonic),
                    use_spectral_norm=use_spectral_norm,
                    use_san=self.use_san,
                )
            )
        if not branches:
            raise ValueError(
                "A discriminator needs at least one branch; the scale branch, "
                "the periods and the resolutions were all turned off."
            )
        self.discriminators = torch.nn.ModuleList(branches)

    @property
    def branch_labels(self) -> tuple:
        """One name per entry of ``discriminators``, in the same order.

        Built from what ``__init__`` actually assembled rather than from a
        preset, so a config that turns a family off or replaces a period set
        still gets labels that line up with the branches -- which is the whole
        point of having them, since they exist to name a per-branch diagnostic.
        """

        labels = ["msd"] if self.use_msd else []
        labels += [f"period_{p}" for p in self.periods]
        labels += [f"resolution_{n_fft}" for n_fft, _, _ in self.resolutions]
        if self.use_univhd:
            labels.append("univhd")
        return tuple(labels)

    @property
    def branch_weights(self) -> tuple:
        """One loss weight per entry of ``discriminators``, in the same order.

        ``None`` is not an option here even when every weight is 1.0: the
        losses take this as a positional list and a tuple that is one entry
        short would silently weight the wrong branches.  Built from the same
        assembly ``branch_labels`` walks, so the two cannot drift apart.

        The generator's adversarial term, the feature-matching term and the
        discriminator's own loss all take these.  See
        ``UNIVHD_WEIGHT_BY_VERSION`` for the one branch that is not 1.0 and the
        measurements behind it.
        """

        count = len(self.discriminators)
        weights = [DEFAULT_BRANCH_WEIGHT] * count
        if self.use_univhd:
            # UnivHD appends itself last -- see ``__init__`` -- which is also
            # what ``branch_labels`` records.
            weights[-1] = self.univhd_weight
        return tuple(weights)

    @property
    def uses_branch_weights(self) -> bool:
        """Whether any branch is weighted away from 1.0.

        Lets a caller skip passing the weights entirely on the common case, so
        an unweighted run builds exactly the graph it built before this
        existed.
        """

        return any(w != DEFAULT_BRANCH_WEIGHT for w in self.branch_weights)

    def enable_compile(self, mode: str = "default") -> bool:
        """Compile the paired real/fake forward, replacing ``forward`` in place.

        ``train.py`` looks this method up with ``getattr(model, "enable_compile",
        None)`` and silently reports "not supported" when it is missing, which
        is what ``compile_discriminator`` did for every run after the ChouwaGAN
        discriminator -- the only class that ever defined it -- was removed.

        Worth compiling: at batch 1 over 0.4 s this forward dispatches 558 ATen
        ops, and the training step runs it twice (once for the discriminator
        update, once with ``no_grad_real`` for the generator's), so 1116 of the
        ~2283 the step still spends outside the compiled decoder are here.  Of
        those 558, 132 are ``_weight_norm_interface`` -- the parametrisation
        recomputing ``g * v / ||v||`` for every convolution on every forward,
        which is exactly what a fused graph stops paying separately.

        ``no_grad_real`` and ``combine_inputs`` are Python ``bool``s, so Dynamo
        guards on them and keeps one graph per combination rather than
        branching inside any of them.  Checkpointing
        is the case that is *not* compiled: it is a fallback for a card that
        cannot hold the activations, and pairing it with compilation trades a
        known-good path for an untested one.
        """

        if getattr(self, "_compile_enabled", False):
            return getattr(self, "_compile_mode", mode) == mode

        eager_forward = self.forward
        try:
            compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
        except Exception as error:
            warning(
                f"{DISCRIMINATOR_COMPILE_ENABLE_FAILED} {error}\n"
                f"{traceback.format_exc()}",
                tag="[INIT]",
            )
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            if not self.training or compile_failed or self.use_checkpointing:
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                # The traceback, for the same reason the decoder's fallback
                # keeps one: this fires once and then stays eager, so without
                # it the run reports a failure nobody can act on.
                warning(
                    f"{DISCRIMINATOR_COMPILE_RUNTIME_FAILED} {error}\n"
                    f"{traceback.format_exc()}",
                    tag="[TRAIN]",
                )
                return eager_forward(*args, **kwargs)

        self.forward = training_forward
        self._compile_enabled = True
        self._compile_mode = mode
        return True

    def forward(
        self,
        y,
        y_hat,
        no_grad_real: bool = False,
        san_training: bool = False,
        combine_inputs: bool = False,
    ):
        """``no_grad_real`` runs the real branch under ``no_grad``.

        The generator update needs the real side only as a *target*: its logits
        are thrown away and its feature maps are the constant the feature
        matching loss measures against.  Left differentiable it still builds a
        full activation graph, and the feature loss then backwards through it
        into discriminator weights whose gradients are zeroed before they are
        ever stepped -- the generator update runs after the discriminator's,
        and ``optim_d.zero_grad`` brackets it on both sides.  So the whole real
        backward is work with no consumer.

        Off by default because the discriminator update *does* need it: that is
        the pass whose gradient trains ``net_d``.

        ``combine_inputs`` runs the real and the fake side as one batch of
        ``2B`` instead of two batches of ``B``.  Nothing in any branch mixes
        samples -- there is no batch normalisation here, and ``weight_norm``,
        the STFTs, the harmonic bank and ``san_tail`` are all per-sample -- so
        the outputs are the same numbers, reached in half the kernel launches
        and with one ``weight_norm`` recompute per convolution instead of two.
        Measured on an RTX 5060 at batch 8 over 0.4 s, discriminator update
        only, fwd+bwd: v2 124 -> 113 ms, v3 161 -> 150, v4 149 -> 136, v4 with
        SAN 151 -> 140, all at peak VRAM within 1% of the paired path
        (``torch.split`` returns views, and the real side's activations were
        already being held alive across the fake pass).  Checkpointing is the
        one case that trades memory for the time: 194 -> 178 ms at +360 MiB,
        because one checkpoint boundary now holds a ``2B`` input instead of two
        boundaries holding ``B`` each.

        Upstream pairs this with ``parametrize.cached()``.  That was measured
        here too and is worth nothing once the passes are batched (-0.0%,
        because batching already halves the ``weight_norm`` recomputes) while
        holding every materialised weight alive for the whole forward
        (+250 MiB on v2), so it is not used.

        It is mutually exclusive with ``no_grad_real`` -- half a batch cannot
        be under ``no_grad`` -- which is why only the discriminator update
        asks for it, and it is off under spectral norm for the reason given at
        ``self.use_spectral_norm``.
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        checkpointing = self.training and self.use_checkpointing
        # Only the discriminator update asks for the direction output.  In the
        # generator's pass the direction is not something the generator may
        # move, so requesting it would build a graph with no consumer -- the
        # same waste ``no_grad_real`` exists to avoid.
        san = bool(san_training) and self.use_san
        combined = combine_inputs and not no_grad_real and not self.use_spectral_norm

        if combined:
            paired = torch.cat((y, y_hat), dim=0)
            sizes = (y.shape[0], y_hat.shape[0])
            for d in self.discriminators:
                if checkpointing:
                    y_d, fmap = checkpoint(
                        d, paired, san_training=san, use_reentrant=False
                    )
                else:
                    y_d, fmap = d(paired, san_training=san)
                # Under SAN a branch returns ``[function, direction]`` rather
                # than one tensor, and both halves have to be split.
                if isinstance(y_d, (list, tuple)):
                    split = [torch.split(part, sizes, dim=0) for part in y_d]
                    y_d_r = [part[0] for part in split]
                    y_d_g = [part[1] for part in split]
                else:
                    y_d_r, y_d_g = torch.split(y_d, sizes, dim=0)
                fmap_r, fmap_g = [], []
                for feature in fmap:
                    feature_r, feature_g = torch.split(feature, sizes, dim=0)
                    fmap_r.append(feature_r)
                    fmap_g.append(feature_g)

                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

            return y_d_rs, y_d_gs, fmap_rs, fmap_gs

        for d in self.discriminators:
            # The other two arms add no context manager at all, and not
            # ``enable_grad``: an outer ``no_grad`` (the validation path) must
            # stay in force.
            if no_grad_real:
                with torch.no_grad():
                    y_d_r, fmap_r = d(y, san_training=san)
            elif checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, san_training=san, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y, san_training=san)

            if checkpointing:
                y_d_g, fmap_g = checkpoint(
                    d, y_hat, san_training=san, use_reentrant=False
                )
            else:
                y_d_g, fmap_g = d(y_hat, san_training=san)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Multi-scale discriminator branch, operating directly on the waveform."""

    def __init__(self, use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.use_san = bool(use_san)
        # No ``norm_f`` on a SAN head: it normalises its own weight, and the two
        # reparametrisations would fight over the same tensor.
        self.conv_post = (
            SANConv1d(1024, 1, 3, 1, padding=1)
            if self.use_san
            else norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        )
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x, san_training: bool = False):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class DiscriminatorP(torch.nn.Module):
    """Multi-period discriminator branch: reshapes the waveform onto a period-`p` grid."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        use_san: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(1024, 1, (3, 1), 1, padding=(1, 0))
            if self.use_san
            else norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        )
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x, san_training: bool = False):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class FastDiscriminatorP(torch.nn.Module):
    """A period branch at a fraction of ``DiscriminatorP``'s width.

    Same reshape and the same six feature maps; the channel schedule is
    ``32, 64, 128, 256`` capped rather than ``32, 128, 512, 1024, 1024``, and
    there are four strided layers instead of five plus a stride-1 layer at the
    end.  Ported from KazeFlow's ChouwaGAN discriminator.

    Why it is worth having: the period family is 32.88 M of the 38.81 M
    parameters in a ``v4`` discriminator, so this is where the memory is.  At
    32 kHz over the shipped period set, 300 steps / 3 seeds, held-out accuracy
    at separating real audio from one defect:

        defect             stock (32.88 M)   this at 256 (2.18 M)
        >9 kHz shelf loss   66.2 +- 8.2       62.1 +- 13.1  (at 128)
        f0 jitter           74.2 +- 6.2       71.7 +-  5.2  (at 128)
        slow dynamics       72.1 +- 3.9       70.4 +-  3.3  (at 128)

    Every gap is smaller than at least one side's seed spread.  A capacity
    sweep on the same probe was flat from 128 to 512 channels (78.5 / 80.0 /
    80.0 / 80.6 on frame-rate AM against the stock schedule's 83.1), which is
    why 256 is the shipped width: it is the knee, not a compromise.

    What the probe cannot see, stated because it is the reason this is opt-in:
    it measures detection of a fixed defect by a freshly trained branch, not
    whether a 15x smaller adversary keeps providing gradient against a
    generator adapting to it for 100k steps.  The failure mode there is
    saturation, not blindness.  It also changes ``loss_fm``'s scale -- six maps
    at <=256 channels instead of <=1024 -- which the feature-matching governor,
    the adversarial ceiling and the per-branch R1 all read.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channels: int = 32,
        max_channels: int = 256,
        n_layers: int = 4,
        use_spectral_norm: bool = False,
        use_san: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList()
        in_channels = 1
        for layer in range(n_layers):
            out_channels = min(channels * (2 ** layer), max_channels)
            self.convs.append(
                norm_f(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
            )
            in_channels = out_channels

        self.conv_final = norm_f(
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                (kernel_size, 1),
                1,
                padding=(get_padding(kernel_size, 1), 0),
            )
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(in_channels, 1, (3, 1), 1, padding=(1, 0))
            if self.use_san
            else norm_f(torch.nn.Conv2d(in_channels, 1, (3, 1), 1, padding=(1, 0)))
        )
        # ``LRELU_SLOPE`` and not KazeFlow's 0.1: this is a capacity swap, and
        # an activation that differs from the branch it replaces would make it
        # two changes wearing one name.
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x, san_training: bool = False):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.lrelu(self.conv_final(x))
        fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class DiscriminatorR(torch.nn.Module):
    """Multi-resolution spectrogram discriminator (Applio's, verbatim).

    A period branch reshapes the waveform and looks at it in the time domain, so
    a defect that is narrow in frequency and stationary in time -- an image, a
    tonal artefact, a missing band -- is spread across its receptive field and
    barely visible.  This branch takes the STFT magnitude at three resolutions
    instead, which is where such a defect is a single bright or missing line.

    The window is deliberately rectangular (``torch.ones``): the branch is a
    discriminator input, not an analysis, and the leakage a boxcar produces is
    part of what it learns to read.
    """

    def __init__(
        self,
        resolution,
        use_spectral_norm: bool = False,
        frequency_strides=(1, 1, 1),
        use_san: bool = False,
        fp32_input: bool = True,
    ):
        super().__init__()

        self.resolution = resolution
        self.fp32_input = bool(fp32_input)
        self.lrelu_slope = 0.1
        self.frequency_strides = tuple(int(s) for s in frequency_strides)
        if len(self.frequency_strides) != 3:
            raise ValueError(
                "DiscriminatorR has three strided layers; "
                f"received {len(self.frequency_strides)} frequency strides."
            )
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [norm_f(torch.nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))]
            + [
                norm_f(
                    torch.nn.Conv2d(32, 32, (3, 9), stride=(s, 2), padding=(1, 4))
                )
                for s in self.frequency_strides
            ]
            + [norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))]
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(32, 1, (3, 3), padding=(1, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        )

        # ``win_length`` is fixed at construction, so the boxcar is a constant
        # and was being rebuilt on every call -- three resolution branches,
        # both the real and the fake pass, and both the discriminator and the
        # generator update, i.e. twelve allocations of the same vector per
        # training step.  Non-persistent so no checkpoint gains a key.
        self.register_buffer(
            "window", torch.ones(int(self.resolution[2])), persistent=False
        )

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return torch.norm(torch.view_as_real(x), p=2, dim=-1)

    def forward(self, x, san_training: bool = False):
        fmap = []
        if self.fp32_input:
            # The boxcar magnitude is linear and unnormalised -- a full-scale
            # sine reads ``win_length / 2``, i.e. 120 to 600 here, against the
            # [-1, 1] every period branch sees -- so under FP16 autocast this
            # stage is where the activations and the weight gradient of the
            # first conv run out of range first, and at a raised learning rate
            # the branch diverges.  Only the STFT and ``convs[0]`` leave
            # autocast: after one conv and a LeakyReLU the scale is the other
            # branches', and the rest goes back to FP16.  Measured on an RTX
            # 5060, v4 + UnivHD + SAN at batch 8 over 0.4 s, D update plus the
            # generator's pass through D: 140.1 -> 148.5 ms, +0.8 GiB peak;
            # the whole branch in FP32 was 200.6 ms, +1.2 GiB.
            with torch.autocast(x.device.type, enabled=False):
                x = self.spectrogram(x.float()).unsqueeze(1)
                x = F.leaky_relu(self.convs[0](x), self.lrelu_slope)
        else:
            x = self.spectrogram(x).unsqueeze(1)
            x = F.leaky_relu(self.convs[0](x), self.lrelu_slope)
        fmap.append(x)
        for layer in self.convs[1:]:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)
