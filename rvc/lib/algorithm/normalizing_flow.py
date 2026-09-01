import torch
from typing import Optional, Tuple
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

import torch.nn as nn
from rvc.lib.algorithm.attentions import FFN, MultiHeadAttention
from rvc.lib.algorithm.transformer_encoder import LayerNorm
from rvc.lib.algorithm.wavenet import WaveNet


class TransformerCouplingNet(nn.Module):
    """VITS2's transformer coupling net, in place of the WaveNet residual stack.

    Same contract as ``WaveNet`` -- ``(x, x_mask, g) -> x`` at ``hidden_channels``
    -- so ``ResidualCouplingLayer`` does not care which one it holds.

    **Why.**  The WaveNet coupling ran 4 layers of kernel 5 at dilation 1, which
    is a receptive field of 17 frames: **0.17 s**.  The flow's whole job is to
    make the pushed-forward posterior meet the prior, and it was doing it
    through a 0.17 s keyhole.  Measured on the v1 run at step 24393, the prior
    mean sat 0.82 posterior-sigma from the posterior mean and cost +24% mel L1
    at inference.  It is also the single heaviest module that ships: 8.69 M
    against a 3.93 M decoder.

    **Cost.**  At ``filter_channels=384`` with a 1-wide FFN this is ~1.0 M per
    coupling block against the WaveNet's 2.13 M, so the flow roughly halves
    while its receptive field goes from 0.17 s to the attention window.  VITS2's
    own settings (3 layers, ``filter_channels`` 768, FFN kernel 3) come out
    *heavier* than the WaveNet, at 3.1 M -- the width is where the parameters
    are, not the attention.

    **``block_length`` is not optional here.**  It bounds attention to a band,
    which is what keeps a frame's encoding independent of how much audio arrived
    with it: training reads 3.00 s slices and the inference pipeline feeds a
    whole silence-delimited span.  The same argument, and the same measurement,
    as the prior's attention window was.  ``window_size`` is a different thing --
    Glow-TTS relative position embeddings, which the E-Branchformer prior does
    not have at all.

    Speaker conditioning is a per-layer additive bias through the same low-rank
    bottleneck ``WaveNet`` uses; see ``WaveNet.cond_rank`` for why factoring an
    embedding lookup is exact rather than approximate.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_layers: int,
        n_heads: int = 2,
        filter_channels: int = 384,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        cond_rank: int = 0,
        window_size: int = 10,
        block_length: int = 150,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.hidden_channels = hidden_channels
        self.n_layers = int(n_layers)
        self.gin_channels = int(gin_channels)

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    int(n_heads),
                    p_dropout=p_dropout,
                    window_size=window_size,
                    block_length=(int(block_length) or None),
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    int(filter_channels),
                    int(kernel_size),
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
        self.drop = nn.Dropout(p_dropout)

        self.cond_layer = None
        if self.gin_channels:
            width = hidden_channels * self.n_layers
            rank = int(cond_rank)
            if 0 < rank < min(self.gin_channels, width):
                self.cond_layer = nn.Sequential(
                    nn.Conv1d(self.gin_channels, rank, 1),
                    nn.Conv1d(rank, width, 1),
                )
            else:
                self.cond_layer = nn.Conv1d(self.gin_channels, width, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        speaker = None if (g is None or self.cond_layer is None) else self.cond_layer(g)
        x = x * x_mask
        for i in range(self.n_layers):
            if speaker is not None:
                offset = i * self.hidden_channels
                x = x + speaker[:, offset : offset + self.hidden_channels, :]
                x = x * x_mask
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask

    def remove_weight_norm(self) -> None:
        """No parametrisations to strip; present so the coupling layer is agnostic."""
        return


class Flip(nn.Module):
    # A plain function wouldn't torch.jit.script(): compiled functions can't
    # take a variable number of arguments or keyword-only args with defaults.
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0.0,
        mean_only=False,
        coupling_net="wavenet",
        cond_rank=0,
        filter_channels=384,
        n_heads=2,
        attention_window=150,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super(ResidualCouplingLayer, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.coupling_net = str(coupling_net).lower()
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        if self.coupling_net == "transformer":
            self.enc = TransformerCouplingNet(
                hidden_channels,
                n_layers,
                n_heads=n_heads,
                filter_channels=filter_channels,
                kernel_size=1,
                p_dropout=p_dropout,
                gin_channels=gin_channels,
                cond_rank=cond_rank,
                block_length=attention_window,
            )
        elif self.coupling_net == "wavenet":
            self.enc = WaveNet(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
                p_dropout=p_dropout,
                cond_rank=cond_rank,
            )
        else:
            raise ValueError(
                f"Unknown coupling_net {coupling_net!r}: "
                "expected 'wavenet' or 'transformer'."
            )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x, torch.zeros([1])

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.enc._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.enc)
        return self


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        p_dropout=0.0,
        coupling_net="wavenet",
        cond_rank=0,
        filter_channels=384,
        n_heads=2,
        attention_window=150,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    mean_only=True,
                    coupling_net=coupling_net,
                    cond_rank=cond_rank,
                    filter_channels=filter_channels,
                    n_heads=n_heads,
                    attention_window=attention_window,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        for i in range(self.n_flows):
            if getattr(self.flows[i * 2], "coupling_net", "wavenet") != "wavenet":
                continue
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])

        return self
