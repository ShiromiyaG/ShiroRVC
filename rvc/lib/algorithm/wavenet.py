import torch
import torch.nn as nn
from torch.nn import Conv1d
from typing import Optional

from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: int = 0,
        cond_rank: int = 0,
    ):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for proper padding."

        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = float(p_dropout)

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(float(p_dropout))

        # ``g`` is ``emb_g(sid)``: one row of an ``(n_speakers, gin_channels)``
        # table, constant over time.  The set of vectors this layer can ever be
        # shown therefore has rank at most ``n_speakers``, so factoring the map
        # through a bottleneck of that width or more reproduces it *exactly* --
        # this is not an approximation as long as ``cond_rank >= n_speakers``.
        # At the shipped 109 speakers, 256 -> 128 -> 2*h*L costs 0.82 M against
        # 1.59 M for the full map in the posterior encoder.
        #
        # ``0`` keeps the single dense map, which is what the plain RVC path
        # and every config predating the key get.
        self.cond_rank = max(0, int(cond_rank))
        if gin_channels != 0:
            cond_width = 2 * hidden_channels * n_layers
            if 0 < self.cond_rank < min(gin_channels, cond_width):
                self.cond_layer = torch.nn.Sequential(
                    torch.nn.Conv1d(gin_channels, self.cond_rank, 1),
                    torch.nn.utils.parametrizations.weight_norm(
                        torch.nn.Conv1d(self.cond_rank, cond_width, 1), name="weight"
                    ),
                )
            else:
                self.cond_rank = 0
                cond_layer = torch.nn.Conv1d(gin_channels, cond_width, 1)
                self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                    cond_layer, name="weight"
                )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.parametrizations.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def _cond_weight_norm_module(self):
        """The submodule carrying the parametrisation, factored or not."""
        if self.gin_channels == 0:
            return None
        return self.cond_layer[1] if self.cond_rank else self.cond_layer

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self._cond_weight_norm_module())
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)

    def __prepare_scriptable__(self):
        if self.gin_channels != 0:
            cond = self._cond_weight_norm_module()
            for hook in cond._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(cond)
        for l in self.in_layers:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self
