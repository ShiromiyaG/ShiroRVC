"""Low-rank bottleneck on the content features.

Timbre is fine detail in the content features, and it is the first thing a
narrow code drops.  The prior plays this role on the VITS path; the direct path
has nothing else in its place.
"""

from typing import Optional

import torch
from torch import nn


class ContentBottleneck(nn.Module):
    """``features -> k-dim code -> features``, with optional training noise.

    The code is layer-normalised so ``noise`` is a fraction of its scale.  The
    last code is kept on ``code`` for the speaker adversary.
    """

    def __init__(self, channels: int, width: int, noise: float = 0.0):
        super().__init__()
        self.down = nn.Linear(channels, width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, channels)
        self.noise = float(noise)
        self.code: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features`` is [batch, frames, channels], as the text encoder takes it."""
        code = self.norm(self.down(features))
        if self.training and self.noise > 0:
            code = code + torch.randn_like(code) * self.noise
        self.code = code
        return self.up(code)
