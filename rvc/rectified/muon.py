"""Muon for the matrices, AdamW for the rest, in one optimizer, as DiffSinger's
``Muon_AdamW`` splits them.

Muon runs Nesterov momentum and replaces each update with its nearest
orthogonal matrix (Newton-Schulz), https://kellerjordan.github.io/posts/muon/.
"""

import torch
from torch import nn

#: Weights with fewer inputs per output than this (scalar embeddings) stay on AdamW.
MIN_FAN_IN = 16
#: Step at which the Gram iteration is rebuilt from the matrix.
GRAM_RESTART = 2


def _iteration_dtype(device: torch.device) -> torch.dtype:
    """BF16 where the GPU has it; Turing and older emulate it slowly, and the
    normalized matrix fits FP16's range."""
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def orthogonalize(g: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """Quintic Newton-Schulz on each matrix of ``g`` [..., rows, cols];
    singular values land near 1. A rectangular matrix is iterated through its
    Gram matrix, the small side squared, as DiffSinger's Muon does. ``dtype``
    is the iteration's, the device's own when None."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / x.flatten(-2).norm(dim=-1).clamp_min(1e-7)[..., None, None]
    x = x.to(dtype or _iteration_dtype(g.device))
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    if x.shape[-2] == x.shape[-1]:
        for _ in range(steps):
            gram = x @ x.mT
            x = a * x + (b * gram + c * gram @ gram) @ x
        return (x.mT if transposed else x).to(g.dtype)
    # x_k = q @ x_0, with q and the Gram matrix advanced instead of x.
    gram, q = x @ x.mT, None
    for index in range(steps):
        if index == GRAM_RESTART:
            # From x again, so the rounding of the first steps does not compound.
            x = q @ x
            gram, q = x @ x.mT, None
        z = b * gram + c * gram @ gram
        if q is None:
            q = z.clone()
            q.diagonal(dim1=-2, dim2=-1).add_(a)
        else:
            q = a * q + q @ z
        if index + 1 < steps and index + 1 != GRAM_RESTART:
            rz = a * gram + gram @ z
            gram = a * rz + z @ rz
    x = q @ x
    return (x.mT if transposed else x).to(g.dtype)


def muon_parameters(model: nn.Module) -> set:
    """Ids of the parameters Muon takes: matrices and conv kernels, except
    embeddings, output projections (``use_adamw``) and tiny fan-ins."""
    chosen = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding) or getattr(module, "use_adamw", False):
            continue
        for param in module.parameters(recurse=False):
            if param.requires_grad and param.dim() >= 2 and param[0].numel() >= MIN_FAN_IN:
                chosen.add(id(param))
    return chosen


class MuonAdamW(torch.optim.Optimizer):
    """Groups flagged ``muon`` get Muon, scaled by ``sqrt(max(rows, cols))`` so
    its update RMS matches AdamW's at the same ``lr``; the others get AdamW."""

    def __init__(self, model, lr, muon_weight_decay=0.1, adamw_weight_decay=0.0,
                 momentum=0.95, betas=(0.9, 0.98), eps=1e-8):
        chosen = muon_parameters(model)
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [
            dict(params=[p for p in params if id(p) in chosen], muon=True,
                 weight_decay=muon_weight_decay),
            dict(params=[p for p in params if id(p) not in chosen], muon=False,
                 weight_decay=adamw_weight_decay),
        ]
        super().__init__(groups, dict(lr=lr, momentum=momentum, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        # Batched over parameters: a loop per parameter launches thousands of
        # kernels per step, which bounds a small-CPU machine.
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, decay = group["lr"], group["weight_decay"]
            if decay > 0:
                torch._foreach_mul_(params, 1.0 - lr * decay)
            if group["muon"]:
                self._muon(group, params, lr)
            else:
                self._adamw(group, params, lr)

    def _muon(self, group, params, lr):
        grads = [p.grad for p in params]
        for p in params:
            # Not ``setdefault``, which would build the zeros on every step.
            if "momentum_buffer" not in self.state[p]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
        buffers = [self.state[p]["momentum_buffer"] for p in params]
        torch._foreach_lerp_(buffers, grads, 1.0 - group["momentum"])
        updates = torch._foreach_lerp(grads, buffers, group["momentum"])
        # One Newton-Schulz per matrix shape, each matrix wide side last.
        shapes = {}
        for p, update in zip(params, updates):
            update = update.reshape(p.shape[0], -1)
            tall = update.shape[0] > update.shape[1]
            shapes.setdefault(tuple(sorted(update.shape)), []).append((p, update.mT if tall else update, tall))
        for shape, members in shapes.items():
            stacked = torch.stack([update for _, update, _ in members])
            orthogonal = orthogonalize(stacked)
            if not torch.isfinite(orthogonal).all():
                # The half-precision iteration overflowed.
                orthogonal = orthogonalize(stacked, dtype=torch.float32)
            orthogonal = orthogonal.unbind(0)
            torch._foreach_add_(
                [p for p, _, _ in members],
                [(u.mT if tall else u).reshape(p.shape) for (p, _, tall), u in zip(members, orthogonal)],
                alpha=-lr * max(shape) ** 0.5,
            )

    def _adamw(self, group, params, lr):
        beta1, beta2 = group["betas"]
        # By step count, which differs only for a parameter that has gone
        # without a gradient.
        by_step = {}
        for p in params:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            by_step.setdefault(state["step"], []).append(p)
        for step, members in by_step.items():
            grads = [p.grad for p in members]
            exp_avg = [self.state[p]["exp_avg"] for p in members]
            exp_avg_sq = [self.state[p]["exp_avg_sq"] for p in members]
            torch._foreach_lerp_(exp_avg, grads, 1.0 - beta1)
            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1.0 - beta2)
            denom = torch._foreach_div(exp_avg_sq, 1.0 - beta2 ** step)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, group["eps"])
            torch._foreach_addcdiv_(members, exp_avg, denom, value=-lr / (1.0 - beta1 ** step))
