"""LoRA for the fine-tune.  Merged back before export, so a LoRA-trained
style model is an ordinary checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += (self.up.weight @ self.down.weight) * self.scale
        return self.base


def apply_lora(model: nn.Module, rank: int, alpha: float, targets, train_extra=()):
    """Wrap every ``nn.Linear`` whose qualified name contains one of
    ``targets`` and freeze everything else except parameters whose name
    contains one of ``train_extra``."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha).to(child.weight.device))
    for name, p in model.named_parameters():
        if ".down." in name or ".up." in name or any(t in name for t in train_extra):
            p.requires_grad_(True)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """``model`` with every ``LoRALinear`` folded into its base, in place."""
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
    return model
