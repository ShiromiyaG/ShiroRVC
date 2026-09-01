import torch
from torch import nn

from typing import Optional

from rvc.lib.algorithm.wavenet import WaveNet

def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class PosteriorEncoder(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		hidden_channels: int,
		kernel_size: int,
		dilation_rate: int,
		n_layers: int,
		gin_channels: int = 0,
	):
		super(PosteriorEncoder, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.gin_channels = gin_channels

		self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
		self.enc = WaveNet(
			hidden_channels,
			kernel_size,
			dilation_rate,
			n_layers,
			gin_channels=gin_channels,
		)
		self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(
		self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
	):
		x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(
			x.dtype
		)
		x = self.pre(x) * x_mask
		x = self.enc(x, x_mask, g=g)
		stats = self.proj(x) * x_mask
		m, logs = torch.split(stats, self.out_channels, dim=1)
		z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
		return z, m, logs, x_mask

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
