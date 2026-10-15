"""BigVGAN v2 (https://github.com/NVIDIA/BigVGAN, MIT, see
THIRD_PARTY_NOTICES) as its released checkpoints have it, with Pupu-Vocoder's
f0 source added at every stage.

The generator is the release's: transposed upsamplers, SnakeBeta between
12-tap 2x filters, a clamped output. ``tools/pretrain/convert_bigvgan_v2.py``
writes the released weights with the source convs as zeros, so the export
renders as the original until ``adapt_pupu_f0.py --architecture bigvgan-v2``
trains them. The release's MPD and sub-band CQT discriminator are this repo's
``mpd`` and ``cqt`` families.
"""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from rvc.rectified.nsf_bigvgan import AMPBlock, SnakeAlias, lowpass, sinc_lowpass
from rvc.rectified.pupu_vocoder import SineSource

#: ``architecture`` of a BigVGAN v2 rectified vocoder export.
ARCHITECTURE = "bigvgan_v2"
#: The release's filter around each SnakeBeta: taps, cutoff and half width.
#: Its weights render noise through NSF-BigVGAN's longer ones.
ALIAS = (12, 0.25, 0.3)


class BigVGANv2(nn.Module):
    """Log mel [B, n_mels, T] and f0 [B, T] -> [B, 1, T * hop]."""

    def __init__(self, sample_rate, num_mels, upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilation_sizes, harmonic_num=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = int(np.prod(upsample_rates))
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.m_source = SineSource(sample_rate, harmonic_num)
        self.ups = nn.ModuleList()
        self.source_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        # The output's rate over each stage's.
        self.source_rates = [int(np.prod(upsample_rates[i + 1:])) for i in range(len(upsample_rates))]
        ch = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, kernel, rate, padding=(kernel - rate) // 2))
            ch //= 2
            self.source_convs.append(nn.Conv1d(1, ch, 7, padding=3))
            if self.source_rates[i] > 1:
                self.register_buffer(
                    f"source_kernel_{i}", sinc_lowpass(0.4 / self.source_rates[i]), persistent=False
                )
            for size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, size, dilation, ALIAS))
        self.activation_post = SnakeAlias(ch, *ALIAS)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # Set by ``apply_precision_policy`` under AMP: the mel conv, the
        # upsamplers, the residual stream and the output layer stay in FP32,
        # as the release was trained.
        self.fp32_residuals = False

    def _fp32_region(self, x):
        return torch.autocast(x.device.type, enabled=False) if self.fp32_residuals else nullcontext()

    def _fp32(self, x):
        return x.float() if self.fp32_residuals else x

    def forward(self, mel, f0):
        with torch.autocast(mel.device.type, enabled=False):
            source = self.m_source(f0.float(), self.upp)
            sources = [
                lowpass(source, getattr(self, f"source_kernel_{i}"), stride=rate) if rate > 1 else source
                for i, rate in enumerate(self.source_rates)
            ]
        with self._fp32_region(mel):
            x = self.conv_pre(self._fp32(mel))
        for i, up in enumerate(self.ups):
            with self._fp32_region(x):
                x = up(self._fp32(x))
                x = x + self.source_convs[i](sources[i].to(x.dtype))
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        with self._fp32_region(x):
            x = self.conv_post(self.activation_post(self._fp32(x)))
        return torch.clamp(x, min=-1.0, max=1.0)


def weight_normed(generator: BigVGANv2) -> list:
    """The convs the release trains under weight norm. Not the source convs:
    they start at zero, which weight norm cannot hold."""
    blocks = [conv for block in generator.resblocks for conv in (*block.convs1, *block.convs2)]
    return [generator.conv_pre, *generator.ups, *blocks, generator.conv_post]
