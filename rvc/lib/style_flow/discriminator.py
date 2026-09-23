"""Multi-period discriminator on the vibrato band of the residual, against
flow matching's pull toward smooth, averaged vibrato.  It is conditioned on
the descriptors: without them a damped vibrato and a naturally small one look
alike, and the discriminator sat at chance.  Training only; it is not part of
the exported model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels=(16, 32, 64, 128), kernel: int = 5, stride: int = 3, cond_dim: int = 0):
        super().__init__()
        self.period = period
        layers, prev = [], 1
        for ch in channels:
            layers.append(weight_norm(nn.Conv2d(prev, ch, (kernel, 1), (stride, 1), padding=(kernel // 2, 0))))
            prev = ch
        self.convs = nn.ModuleList(layers)
        self.post = weight_norm(nn.Conv2d(prev, 1, (3, 1), padding=(1, 0)))
        # Projection discriminator: the logit gains <features, W cond>.  Zero at
        # init, so it starts as the unconditional one.
        self.embed = nn.Linear(cond_dim, prev) if cond_dim else None
        if self.embed is not None:
            nn.init.zeros_(self.embed.weight)
            nn.init.zeros_(self.embed.bias)

    def forward(self, x, cond=None):
        """``x`` (B, 1, T), ``cond`` (B, cond_dim) -> ``(logits, feature maps)``."""
        B, C, T = x.shape
        if T % self.period:
            x = F.pad(x, (0, self.period - T % self.period))
        x = x.view(B, C, -1, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        logits = self.post(x)
        if self.embed is not None:
            logits = logits + (x * self.embed(cond)[:, :, None, None]).sum(dim=1, keepdim=True)
        fmap.append(logits)
        return logits.flatten(1), fmap


class VibratoDiscriminator(nn.Module):
    def __init__(self, band_hz, frame_rate: float, periods=(2, 3, 5, 7, 11), taps: int = 101, cond_dim: int = 0):
        super().__init__()
        # Only the vibrato band: note centres, scoops and drops are left to the flow loss.
        fir = signal.firwin(taps, list(band_hz), pass_zero=False, fs=frame_rate)
        self.register_buffer("fir", torch.tensor(fir, dtype=torch.float32).view(1, 1, -1), persistent=False)
        self.discs = nn.ModuleList(PeriodDiscriminator(p, cond_dim=cond_dim) for p in periods)

    def forward(self, residual, voiced, cond=None):
        """``residual`` (B, T) normalized, as the target's channel 0;
        ``voiced`` (B, T) bool; ``cond`` (B, cond_dim), the model's descriptor
        input.  One ``(logits, feature maps)`` per period."""
        m = voiced[:, None].to(residual.dtype)
        x = F.conv1d(residual[:, None] * m, self.fir.to(residual.dtype), padding=self.fir.shape[-1] // 2) * m
        return [d(x, cond) for d in self.discs]


def _mean(x, weight):
    """Mean of ``x`` (B, ...) over samples weighted by ``weight`` (B,)."""
    per_sample = x.float().flatten(1).mean(dim=1)
    return (per_sample * weight).sum() / weight.sum().clamp(min=1.0)


def discriminator_loss(real_out, fake_out, weight):
    """LSGAN, averaged over periods."""
    total = sum(_mean((1.0 - r) ** 2, weight) + _mean(f**2, weight) for (r, _), (f, _) in zip(real_out, fake_out))
    return total / len(real_out)


def generator_loss(real_out, fake_out, weight):
    """``(adversarial, feature matching)``, averaged over periods; the real
    side is a constant."""
    adv = sum(_mean((1.0 - f) ** 2, weight) for f, _ in fake_out) / len(fake_out)
    fm = sum(
        _mean(torch.abs(a.detach() - b), weight)
        for (_, real_maps), (_, fake_maps) in zip(real_out, fake_out)
        for a, b in zip(real_maps, fake_maps)
    ) / len(fake_out)
    return adv, fm
