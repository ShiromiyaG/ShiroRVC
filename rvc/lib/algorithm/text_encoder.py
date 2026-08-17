import torch
import numpy as np
import math

from torch import nn
from torch.nn import functional as F

from typing import Optional

from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.transformer_encoder import TransformerEncoder


class TextEncoder(nn.Module):
    """
    Text Encoder.

    Args:
        out_channels (int): Output channels of the encoder.
        hidden_channels (int): Hidden channels of the encoder.
        filter_channels (int): Filter channels of the encoder.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        embedding_dim (int): Embedding dimension for phone embeddings (v1 = 256, v2 = 768).
        f0 (bool, optional): Whether to use F0 embedding. Defaults to True.
    """
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

        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
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
