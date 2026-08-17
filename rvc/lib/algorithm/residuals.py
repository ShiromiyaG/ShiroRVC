from itertools import chain
from typing import Tuple

import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.commons import get_padding


LRELU_SLOPE = 0.1


def create_conv1d_layer(channels: int, kernel_size: int, dilation: int):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


class ResBlock(torch.nn.Module):
    """HiFi-GAN residual block used by the NSF generator."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, (1,) * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int, ...]):
        return torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, dilation) for dilation in dilations]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                x = x * x_mask
            x = conv1(x)
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                x = x * x_mask
            x = conv2(x) + residual
            if x_mask is not None:
                x = x * x_mask
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)
