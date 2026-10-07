"""The velocity network and the aux decoder sampling starts from."""

import torch
from torch import nn
from torch.nn import functional as F

from rvc.rectified.flow.layers import ConvNeXtBlock, atan_glu, timestep_embedding


class LYNXNet2Block(nn.Module):
    """Depthwise conv, then two ATanGLU projections, pre-norm and residual.

    With ``adaln``, the global embedding (time and speaker) shifts and scales
    the norm and gates the residual (DiT's adaLN-Zero); zero-initialised, so
    the block starts as the plain one.
    """

    def __init__(self, channels, expansion, kernel_size, adaln=False):
        super().__init__()
        inner = int(channels * expansion)
        self.norm = nn.LayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, mask, embedding=None):
        """``x`` [B, T, C], ``mask`` [B, T, 1], ``embedding`` [B, 1 or T, C]."""
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            # Not ``1 + scale``: in BF16 that rounds modulations under ~0.004 to nothing.
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y)))))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    """DiffSinger's current LYNXNet2: condition and time added once at the
    input, then depthwise-separable gated blocks. ``adaln`` also modulates
    every block by time and speaker, as RIFT-SVC's DiT does.

    ``time_scale`` multiplies the flow time before its sinusoids: 1000 is
    DiffSinger's; 1, as SiT has it, keeps the network smooth in time.

    ``step_levels`` makes it a shortcut model's: level k is a jump of
    1 / 2**k of the trained time range, and has its own embedding, from zero.
    The finest level, ``step_levels``, has none and is the plain flow, so the
    weights serve with or without the embeddings."""

    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 adaln=False, time_scale=1000.0, step_levels=0):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.layers = nn.ModuleList(
            [LYNXNet2Block(channels, expansion, kernel_size, adaln) for _ in range(layers)]
        )
        self.voice = nn.Linear(cond_channels, channels) if adaln else None
        self.step = None
        if step_levels > 0:
            self.step = nn.Embedding(step_levels + 1, channels, padding_idx=step_levels)
            nn.init.zeros_(self.step.weight)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, cond, mask, voice=None, level=None):
        """``t`` is [B], or [B, T] for a time per frame. ``level`` [B] is the
        jump each item takes; None is the plain flow."""
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels, self.time_scale))
        time = time.view(t.shape[0], -1, self.channels)
        if level is not None:
            time = time + self.step(level)[:, None, :]
        frame_mask = mask.transpose(1, 2)
        # Full precision in, which also keeps the residual stream in FP32: at
        # late t the leftover noise is smaller than BF16's step on x_t.
        with torch.autocast(x.device.type, enabled=False):
            h = self.input(x.transpose(1, 2).to(self.input.weight.dtype))
        h = h + self.input_cond(cond).transpose(1, 2) + time
        h = h * frame_mask
        embedding = None
        if self.voice is not None:
            embedding = time + self.voice(voice)[:, None, :]
        for layer in self.layers:
            h = layer(h, frame_mask, embedding)
        h = self.norm(h)
        return (self.output(h) * frame_mask).transpose(1, 2)


class AuxDecoder(nn.Module):
    """A deterministic mel from the conditioning, where shallow sampling
    starts. DiffSinger's ConvNeXt aux decoder; with ``speaker`` every block is
    also modulated by the speaker's embedding, [B, cond_channels]."""

    def __init__(self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1, speaker=False):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(channels, layer_scale=1e-6, dropout=dropout,
                           speaker_channels=cond_channels if speaker else 0) for _ in range(layers)]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask, voice=None):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask, voice)
        return self.output(x) * mask
