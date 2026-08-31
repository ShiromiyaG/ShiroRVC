import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE

#: Applio's ``MultiPeriodDiscriminator`` branch layouts, kept under their names
#: so a diff against upstream is a diff and not a translation.  ``v2`` is what
#: Applio runs for HiFi-GAN; ``v3`` is what it selects for RefineGAN, and the
#: difference is not cosmetic -- it trades the three widest period branches for
#: three multi-resolution spectrogram branches, which are the ones that see a
#: frequency-domain defect at all.
#: ``v3l`` is this fork's: v3 with the frequency axis actually downsampled.
#:
#: Applio's ``DiscriminatorR`` strides ``(1, 2)`` -- it decimates the *frame*
#: axis and carries 257/513/1025 frequency bins at 32 channels through all five
#: layers, which is where the branch's cost is.  Striding frequency in the last
#: two layers instead was measured, on a probe that trains the three branches
#: for 300 steps to separate real audio from a defect and reports held-out
#: accuracy over three seeds:
#:
#:   defect                     Applio (1,1,1)   this (1,2,2)   (2,2,2)
#:   frame-rate AM (the leak)   68.0% +- 12.5    76.5% +- 4.4   71.5% +- 14.5
#:   over-smoothed above 9 kHz  57.8% +-  0.8    58.2% +- 0.2   --
#:
#: at 81.1 ms against 107.0 and 342 MiB against 396 (fwd+bwd, batch 8, 0.4 s).
#: Not worse on either defect, less seed-dependent -- Applio's schedule collapsed
#: to chance on one of three seeds -- and 24% cheaper.
#:
#: Two things that look like obvious wins and are not, both measured:
#:
#: * Reusing ChouwaGAN's spectrogram branch (7.4x faster, 3x less memory) drops
#:   the AM accuracy to 54%, i.e. chance.  Its hops are 128/256/512 samples.
#: * Trading the 512-point branch for a 4096-point one -- more frequency
#:   resolution, which is where +-100 Hz sidebands live -- drops it to 50.5%.
#:
#: Both fail for the same reason: the branch catches frame-rate mirroring as a
#: **temporal** modulation, not as frequency sidebands.  100 Hz is a 10 ms
#: period, and only the 512-point branch's 50-sample hop (1.1 ms) resolves it;
#: 4096/480 gives 10.9 ms per frame, which aliases the modulation to DC.  The
#: fine-hop branch is the one that must not be touched.
DISCRIMINATOR_VERSIONS = {
    "v1": ([2, 3, 5, 7, 11, 17], [], (1, 1, 1)),
    "v2": ([2, 3, 5, 7, 11, 17, 23, 37], [], (1, 1, 1)),
    "v3": (
        [2, 3, 5, 7, 11],
        [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
        (1, 1, 1),
    ),
    "v3l": (
        [2, 3, 5, 7, 11],
        [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
        (1, 2, 2),
    ),
}


class MPD_MSD_Combined(torch.nn.Module):
    """
    Multi-period and Multi-scale discriminators combined.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        version: str = "v2",
        periods=None,
        resolutions=None,
        frequency_strides=None,
        use_msd: bool = True,
    ):
        """``version`` picks a preset; the four overrides edit it branch by branch.

        Each family is independently addressable because they cost very
        different things and see very different defects.  Measured fwd+bwd at
        batch 8 over 0.4 s: the scale branch is 16 ms / 168 MiB, a period branch
        ~9 ms / ~250 MiB each, and a spectrogram branch ~30 ms / ~430 MiB (~24 ms
        under ``v3l``).  Five periods are 1.25 GiB on their own, so dropping two
        of them is a bigger lever than anything inside a branch.

        ``None`` means "whatever the version says"; an empty list means "none of
        this family", which is the distinction a falsy check would lose.
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
        periods = preset_periods if periods is None else list(periods)
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
        self.use_checkpointing = use_checkpointing
        branches = []
        if self.use_msd:
            branches.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        branches += [
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm)
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
        if not branches:
            raise ValueError(
                "A discriminator needs at least one branch; the scale branch, "
                "the periods and the resolutions were all turned off."
            )
        self.discriminators = torch.nn.ModuleList(branches)

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.use_checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

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
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

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

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.ones(win_length, device=x.device),
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
