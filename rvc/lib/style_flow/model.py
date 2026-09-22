"""DiT velocity field for the style flow: AdaLN-Zero blocks with RoPE.

Per frame it sees the noisy target, the content units and the coarse melody;
the timestep and the descriptor vector condition every block through AdaLN.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 320
    depth: int = 8
    heads: int = 5
    mlp_ratio: float = 4.0
    conv_kernel: int = 7
    rope_base: float = 10000.0
    n_units: int = 256
    n_descriptors: int = 16
    target_channels: int = 2
    #: RMSNorm on queries and keys, against attention-logit blow-ups at high
    #: learning rates.  Off for checkpoints trained before it existed.
    qk_norm: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "ModelConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def _rope(length: int, head_dim: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    rotated = torch.cat([-b, a], dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = 1000.0 * t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim, hidden = cfg.dim, int(cfg.dim * cfg.mlp_ratio)
        self.heads = cfg.heads
        self.q_norm = nn.RMSNorm(dim // cfg.heads, eps=1e-6) if cfg.qk_norm else None
        self.k_norm = nn.RMSNorm(dim // cfg.heads, eps=1e-6) if cfg.qk_norm else None
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, cos, sin, attn_mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        B, T, D = x.shape
        h = _modulate(self.norm1(x), shift1, scale1)
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + gate1 * self.proj(h.transpose(1, 2).reshape(B, T, D))
        x = x + gate2 * self.mlp(_modulate(self.norm2(x), shift2, scale2))
        return x


class StyleDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Set by the trainer; not part of the checkpoint.
        self.grad_checkpointing = False
        dim = cfg.dim
        self.in_proj = nn.Linear(cfg.target_channels + 1, dim)
        # Index ``n_units`` is the dropped-units token.
        self.unit_emb = nn.Embedding(cfg.n_units + 1, dim)
        # Depthwise conv before attention: vibrato and scoops are local shapes.
        self.local = nn.Conv1d(dim, dim, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=dim)
        self.time_emb = TimestepEmbedding(dim)
        # Values and a presence mask, so a missing descriptor is not read as the mean.
        self.desc_emb = nn.Sequential(nn.Linear(2 * cfg.n_descriptors, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.desc_null = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, cfg.target_channels)
        for layer in (self.ada_out[1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x_t, t, units, coarse, desc, desc_mask, mask, drop_units=None, drop_desc=None):
        """``x_t`` (B, C, T), ``t`` (B,), ``units`` (B, T) long, ``coarse``
        (B, T), ``desc``/``desc_mask`` (B, D), ``mask`` (B, T) bool for valid
        frames; ``drop_*`` (B,) bool replace that condition with its null.
        Returns the velocity (B, C, T)."""
        B, _, T = x_t.shape
        if drop_units is not None:
            units = torch.where(drop_units[:, None], torch.full_like(units, self.cfg.n_units), units)
        h = self.in_proj(torch.cat([x_t, coarse[:, None]], dim=1).transpose(1, 2)) + self.unit_emb(units)
        h = h * mask[..., None]
        h = h + F.gelu(self.local(h.transpose(1, 2)).transpose(1, 2))

        d = self.desc_emb(torch.cat([desc * desc_mask, desc_mask], dim=-1))
        if drop_desc is not None:
            d = torch.where(drop_desc[:, None], self.desc_null.to(d.dtype).expand_as(d), d)
        c = self.time_emb(t).to(d.dtype) + d

        cos, sin = _rope(T, self.cfg.dim // self.cfg.heads, self.cfg.rope_base, x_t.device)
        attn_mask = mask[:, None, None, :]
        for block in self.blocks:
            if self.grad_checkpointing and torch.is_grad_enabled():
                h = checkpoint(block, h, c, cos, sin, attn_mask, use_reentrant=False)
            else:
                h = block(h, c, cos, sin, attn_mask)
        shift, scale = self.ada_out(c).unsqueeze(1).chunk(2, dim=-1)
        v = self.out(_modulate(self.norm_out(h), shift, scale))
        return v.transpose(1, 2) * mask[:, None]
