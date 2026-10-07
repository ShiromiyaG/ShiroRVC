"""Blocks the flow's encoder, backbone and aux decoder are built of."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    """``layer_scale`` > 0 starts the branch that small, as DiffSinger's aux
    decoder does. With ``speaker_channels`` the speaker's embedding shifts and
    scales the norm, from zero."""

    def __init__(self, channels: int, layer_scale: float = 0.0, dropout: float = 0.0,
                 speaker_channels: int = 0):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale)) if layer_scale > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.speaker = None
        if speaker_channels > 0:
            self.speaker = nn.Linear(speaker_channels, channels * 2)
            nn.init.zeros_(self.speaker.weight)
            nn.init.zeros_(self.speaker.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, voice: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``voice`` [B, speaker_channels] is the speaker's embedding, for a
        block that takes one."""
        y = self.norm(self.depthwise(x * mask).transpose(1, 2))
        if self.speaker is not None:
            shift, scale = self.speaker(voice)[:, None, :].chunk(2, dim=-1)
            # Not ``1 + scale``, as in ``LYNXNet2Block``.
            y = y + y * scale + shift
        y = self.down(F.gelu(self.up(y)))
        if self.gamma is not None:
            y = y * self.gamma
        return (x + self.dropout(y.transpose(1, 2))) * mask


def timestep_embedding(t: torch.Tensor, channels: int, scale: float = 1000.0) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = (t.float() * scale)[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _ATanGLU(torch.autograd.Function):
    """``out * atan(gate)``, keeping two tensors for backward instead of three.
    Adapted from DiffSinger (Apache 2.0, see THIRD_PARTY_NOTICES)."""

    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        ctx.save_for_backward(out / gate.square().add(1.0), atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad):
        decay_out, atan_gate = ctx.saved_tensors
        return grad * atan_gate, grad * decay_out


def atan_glu(x: torch.Tensor) -> torch.Tensor:
    out, gate = x.chunk(2, dim=-1)
    if torch.is_grad_enabled():
        return _ATanGLU.apply(out, gate)
    return out * torch.atan(gate)
