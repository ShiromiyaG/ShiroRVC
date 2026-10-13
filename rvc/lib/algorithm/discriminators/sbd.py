"""Sub-band discriminator branch, after Avocodo's SBD (arXiv 2206.13404): the
waveform split by a PQMF analysis bank, and a group of neighbouring sub-bands
judged as decimated waveforms.

A group of high sub-bands holds none of the low band's energy, which is what
hides the top of the spectrum from the period and linear-magnitude branches,
and stays a waveform, so a modulation at the frame rate is a pattern in time.
"""

import math

import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.discriminators.discriminator import waveform_conv
from rvc.lib.algorithm.discriminators.san import SANConv1d, san_tail
from rvc.lib.algorithm.residuals import LRELU_SLOPE

#: The prototype lowpass, as Avocodo's 16-band bank: its length and Kaiser
#: beta, and the cutoff as a fraction of Nyquist times the number of bands.
PQMF_TAPS = 256
PQMF_BETA = 10.0
PQMF_CUTOFF = 0.48


def pqmf_analysis(subbands: int) -> torch.Tensor:
    """The cosine-modulated analysis filters, [subbands, 1, PQMF_TAPS + 1];
    band k covers k to k + 1 times Nyquist / subbands."""
    centre = PQMF_TAPS / 2
    time = torch.arange(PQMF_TAPS + 1, dtype=torch.float64) - centre
    cutoff = PQMF_CUTOFF / subbands
    prototype = cutoff * torch.sinc(cutoff * time)
    prototype = prototype * torch.kaiser_window(
        PQMF_TAPS + 1, periodic=False, beta=PQMF_BETA, dtype=torch.float64
    )
    band = torch.arange(subbands, dtype=torch.float64)[:, None]
    phase = (2 * band + 1) * math.pi / (2 * subbands) * time + (1 - 2 * (band % 2)) * math.pi / 4
    return (2 * prototype * torch.cos(phase)).float()[:, None]


class MDC(torch.nn.Module):
    """Avocodo's multi-scale dilated block: parallel dilated convs, summed,
    then a strided conv."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilations, norm_f):
        super().__init__()
        self.dilated = torch.nn.ModuleList(
            norm_f(torch.nn.Conv1d(
                in_channels, out_channels, kernel_size, dilation=d, padding=d * (kernel_size - 1) // 2
            ))
            for d in dilations
        )
        self.down = norm_f(torch.nn.Conv1d(out_channels, out_channels, 3, stride, padding=1))

    def forward(self, x, first=None):
        """``first`` runs each dilated conv instead of calling it."""
        run = first or (lambda conv, value: conv(value))
        x = sum(run(conv, x) for conv in self.dilated)
        return self.down(F.leaky_relu(x, LRELU_SLOPE))


class SubBandDiscriminator(torch.nn.Module):
    """One branch over the sub-bands ``bands[0]`` up to ``bands[1]`` (exclusive)
    of a ``subbands``-band analysis; waveform [B, 1, T] -> logits and features.

    ``gain_db`` is a fixed gain on the sub-bands. A voice's high bands sit
    tens of dB under its low ones, and a conv stack's logits scale with its
    input: unraised, real and fake come out the same number."""

    #: Set by ``MPD_MSD_Combined``; see ``waveform_conv``.
    wave_fp32_input = False

    def __init__(self, bands, subbands: int = 16, channels: int = 64, gain_db: float = 0.0,
                 use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()
        low, high = int(bands[0]), int(bands[1])
        if not 0 <= low < high <= subbands:
            raise ValueError(f"Sub-bands {list(bands)} are not a range within 0..{subbands}.")
        self.subbands = int(subbands)
        analysis = pqmf_analysis(self.subbands)[low:high] * 10.0 ** (float(gain_db) / 20.0)
        self.register_buffer("analysis", analysis, persistent=False)
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        widths = [high - low, channels, 2 * channels, 4 * channels, 4 * channels]
        strides = [1, 2, 2, 1]
        self.blocks = torch.nn.ModuleList(
            MDC(a, b, 5, stride, (1, 2, 3), norm_f) for a, b, stride in zip(widths, widths[1:], strides)
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv1d(widths[-1], 1, 3, 1, padding=1)
            if self.use_san
            else norm_f(torch.nn.Conv1d(widths[-1], 1, 3, 1, padding=1))
        )

    def forward(self, x, san_training: bool = False):
        # The fixed filters in FP32: BF16 would keep 8 bits of each sample.
        with torch.autocast(x.device.type, enabled=False):
            x = F.pad(x.float(), (PQMF_TAPS // 2, PQMF_TAPS // 2))
            x = F.conv1d(x, self.analysis, stride=self.subbands)
        fmap = []
        for index, block in enumerate(self.blocks):
            first = None if index else (lambda conv, value: waveform_conv(conv, value, self.wave_fp32_input))
            x = F.leaky_relu(block(x, first), LRELU_SLOPE)
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)
