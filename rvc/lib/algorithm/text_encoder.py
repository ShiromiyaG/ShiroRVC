import torch
import numpy as np
import math

from torch import nn
from torch.nn import functional as F

from typing import Optional

from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.transformer_encoder import TransformerEncoder


class TextEncoder(nn.Module):
    """Args: ``embedding_dim`` is the phone embedding size (v1 = 256, v2 = 768)."""
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None

        self.encoder = TransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        lengths: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
    ):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)

        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)

        if skip_head is not None:
            assert isinstance(skip_head, torch.Tensor)
            head = int(skip_head.item())
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, x_mask


class ProjectionEncoder(nn.Module):
    """Per-frame content encoder for ``latent_mode: "direct"``, after
    fish-diffusion's HiFiSinger: projection and an MLP, no attention.

    The content features already carry context and the decoder has a wide
    receptive field; a transformer here sits in the adversarial loop, where its
    gradient spikes.  Returns ``(m, logs, x_mask)`` like ``TextEncoder``, with
    ``logs`` zero since nothing is sampled.

    ``f0=False`` leaves pitch to the decoder's excitation, as fish-diffusion
    does: the coarse pitch embedding jumps at every bin change, and per frame
    those jumps reach the decoder as frame-rate jitter in the high bands.

    ``smoothing`` > 1 adds a depthwise conv over that many frames on the
    output, started as a moving average: the content features also vary from
    frame to frame, and nothing else in the direct path smooths them.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        f0: bool = True,
        smoothing: int = 1,
    ):
        super().__init__()
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels) if f0 else None
        self.fuser = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.proj = nn.Linear(hidden_channels, out_channels)
        smoothing = int(smoothing)
        if smoothing < 1 or smoothing % 2 == 0:
            raise ValueError(f"smoothing must be odd and >= 1, not {smoothing}.")
        self.smooth = None
        if smoothing > 1:
            self.smooth = nn.Conv1d(
                out_channels,
                out_channels,
                smoothing,
                padding=smoothing // 2,
                groups=out_channels,
                bias=False,
            )
            nn.init.constant_(self.smooth.weight, 1.0 / smoothing)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor):
        x = self.emb_phone(phone)
        if self.emb_pitch is not None and pitch is not None:
            x = x + self.emb_pitch(pitch)
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(1)), 1).to(x.dtype)
        m = self.proj(self.fuser(x)).transpose(1, 2) * x_mask
        if self.smooth is not None:
            m = self.smooth(m) * x_mask
        return m, torch.zeros_like(m), x_mask
