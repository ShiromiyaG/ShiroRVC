"""Speaker classifier on the content bottleneck, behind gradient reversal.

The head learns to name the speaker from the bottleneck code; the reversed
gradient teaches the bottleneck to make that impossible, which is what keeps
the source's timbre out of a conversion.  Training only: the head is not part
of the generator, so pretrains, exports and strict loads are unaffected.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

#: Checkpoint key the head is stored under, beside the generator's ``model``.
KEY = "speaker_adv_head"


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class SpeakerAdversary:
    """The head, its optimizer, and the cross-entropy it contributes."""

    def __init__(self, width, speakers, weight, config, device, device_id, n_gpus):
        self.weight = float(weight)
        self.head = nn.Sequential(
            nn.Conv1d(width, 256, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, speakers, 1),
        )
        self.head = self.head.to(device_id if device.type == "cuda" else device)
        self.module = (
            DDP(self.head, device_ids=[device_id])
            if n_gpus > 1 and device.type == "cuda"
            else self.head
        )
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=config.train.learning_rate_g,
            betas=tuple(config.train.betas),
            eps=config.train.eps,
        )
        self.accuracy = None

    def restore(self, path) -> bool:
        """Load the head from a generator checkpoint that carries a matching one."""
        if not path or not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=True).get(KEY)
        if not state or any(
            key not in state or state[key].shape != value.shape
            for key, value in self.head.state_dict().items()
        ):
            return False
        self.head.load_state_dict(state)
        return True

    def state_dict(self):
        return {key: value.detach().cpu() for key, value in self.head.state_dict().items()}

    def loss(self, code, mask, speakers):
        """``code`` [batch, frames, width]; ``mask`` [batch, 1, frames]."""
        with autocast(device_type="cuda", enabled=False):
            x = _Reverse.apply(code.float()).transpose(1, 2)
            mask = mask.float()
            logits = self.module(x * mask)
            # Pooled over valid frames: identity is a property of the clip.
            logits = (logits * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            with torch.no_grad():
                self.accuracy = (logits.argmax(-1) == speakers).float().mean()
            return F.cross_entropy(logits, speakers) * self.weight
