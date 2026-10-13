"""The velocity network and the aux decoder sampling starts from."""

import torch
from torch import nn
from torch.nn import functional as F

from rvc.rectified.flow.layers import ConvNeXtBlock, atan_glu, timestep_embedding

#: Multiplies the guidance scale, less 1, before its sinusoids.
GUIDANCE_SCALE = 30.0


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

    def forward(self, x, mask, embedding=None, fused=True):
        """``x`` [B, T, C], ``mask`` [B, T, 1], ``embedding`` [B, 1 or T, C];
        ``fused`` is ``atan_glu``'s."""
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            # Not ``1 + scale``: in BF16 that rounds modulations under ~0.004 to nothing.
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y), fused)), fused))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    """DiffSinger's current LYNXNet2: condition and time added once at the
    input, then depthwise-separable gated blocks. ``adaln`` also modulates
    every block by time and speaker, as RIFT-SVC's DiT does. ``span`` adds
    MeanFlow's second time input, the length of the step whose mean velocity
    is predicted, and with it the guidance scale that velocity is under, as
    Improved Mean Flows has it.

    ``time_scale`` multiplies the flow time before its sinusoids: 1000 is
    DiffSinger's; 1, as SiT has it, keeps the network smooth in time, which
    mean flow needs because its target is the network's own time derivative."""

    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 adaln=False, span=False, time_scale=1000.0):
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
        self.span_mlp = self.guide_mlp = None
        if span:
            # No biases, on features that vanish at 0: a zero span adds exactly
            # nothing, so the instantaneous velocity is the network without it.
            # A guidance scale of 1 likewise.
            self.span_mlp, self.guide_mlp = (
                nn.Sequential(
                    nn.Linear(channels, channels * 4, bias=False), nn.GELU(),
                    nn.Linear(channels * 4, channels, bias=False),
                )
                for _ in range(2)
            )
            nn.init.zeros_(self.span_mlp[-1].weight)
            nn.init.zeros_(self.guide_mlp[-1].weight)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _vanishing(self, value, scale):
        """Sinusoids of ``value`` [B] that are all 0 at 0."""
        features = timestep_embedding(value.reshape(-1), self.channels, scale)
        half = self.channels // 2
        return torch.cat((features[:, :half], 1.0 - features[:, half:]), dim=-1)

    def forward(self, x, t, cond, mask, voice=None, span=None, guidance=None):
        """``t`` is [B], or [B, T] for a time per frame. ``span`` [B] is the
        length of the step from ``t`` whose mean velocity is wanted; None is
        the velocity at ``t``. ``guidance`` [B] is the speaker guidance scale
        the output is under; None is 1, unguided."""
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels, self.time_scale))
        time = time.view(t.shape[0], -1, self.channels)
        if span is not None:
            features = self._vanishing(span, self.time_scale)
            time = time + self.span_mlp(features).view(span.shape[0], -1, self.channels)
        if guidance is not None:
            features = self._vanishing(guidance - 1.0, GUIDANCE_SCALE)
            time = time + self.guide_mlp(features).view(guidance.shape[0], -1, self.channels)
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
            h = layer(h, frame_mask, embedding, fused=span is None)
        if span is None:
            h = self.norm(h)
        else:
            # The affine applied apart: Dynamo cannot trace the forward-mode
            # derivative of a layer norm with weights.
            h = F.layer_norm(h, (self.channels,), eps=self.norm.eps) * self.norm.weight + self.norm.bias
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
