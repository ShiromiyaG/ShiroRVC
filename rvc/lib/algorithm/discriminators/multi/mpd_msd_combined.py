import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE

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
        univhd_n_fft: int = 2048,
        univhd_hop_length: int = 256,
        univhd_harmonics: int = 10,
        univhd_bins_per_octave: int = 24,
        univhd_f_min: float = 80.0,
        univhd_channels: int = 32,
        univhd_half_harmonic: bool = True,
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
        self.use_univhd = bool(use_univhd)
        self.sample_rate = int(sample_rate)
        self.use_checkpointing = use_checkpointing
        branches = []
        if self.use_msd:
            branches.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        period_branch = FastDiscriminatorP if self.use_fast_mpd else DiscriminatorP
        branches += [
            period_branch(p, use_spectral_norm=use_spectral_norm)
            for p in self.periods
        ]
        branches += [
            DiscriminatorR(
                list(r),
                use_spectral_norm=use_spectral_norm,
                frequency_strides=self.frequency_strides,
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
                )
            )
        if not branches:
            raise ValueError(
                "A discriminator needs at least one branch; the scale branch, "
                "the periods and the resolutions were all turned off."
            )
        self.discriminators = torch.nn.ModuleList(branches)

    def forward(self, y, y_hat, no_grad_real: bool = False):
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
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        checkpointing = self.training and self.use_checkpointing
        for d in self.discriminators:
            # The other two arms add no context manager at all, and not
            # ``enable_grad``: an outer ``no_grad`` (the validation path) must
            # stay in force.
            if no_grad_real:
                with torch.no_grad():
                    y_d_r, fmap_r = d(y)
            elif checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)

            if checkpointing:
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Multi-scale discriminator branch, operating directly on the waveform."""

    def __init__(self, use_spectral_norm: bool = False):
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
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """Multi-period discriminator branch: reshapes the waveform onto a period-`p` grid."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
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

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


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
        self.conv_post = norm_f(
            torch.nn.Conv2d(in_channels, 1, (3, 1), 1, padding=(1, 0))
        )
        # ``LRELU_SLOPE`` and not KazeFlow's 0.1: this is a capacity swap, and
        # an activation that differs from the branch it replaces would make it
        # two changes wearing one name.
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
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
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


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
    ):
        super().__init__()

        self.resolution = resolution
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
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

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

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap
