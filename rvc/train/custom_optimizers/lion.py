"""Lion: EvoLved Sign Momentum (Chen et al., 2023).

The update is the *sign* of an interpolation between the gradient and the
single momentum buffer, so every parameter moves by exactly ``lr`` regardless
of gradient magnitude -- hence the smaller ``LR_SCALE`` and larger
``WEIGHT_DECAY_SCALE`` below. The two betas are deliberately not the AdamW
pair: ``beta1`` (0.9) forms the update, the slower ``beta2`` (0.99) tracks
the buffer.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


#: Lion's learning rate relative to a tuned AdamW one; authors report 3-10x
#: smaller, conservative end chosen since a too-large generator step is
#: expensive to undo against the discriminator.
LR_SCALE = 1.0 / 3.0

#: Weight decay relative to AdamW's, scaled up by the same factor LR_SCALE
#: scales down, so lr * weight_decay (the pull to zero) stays unchanged.
WEIGHT_DECAY_SCALE = 3.0


class Lion(Optimizer):
    """Sign-momentum optimizer with decoupled weight decay.

    Args:
        params: parameters or param groups.
        lr: already scaled by ``LR_SCALE`` by the caller -- this class does not
            rescale, so it stays a faithful Lion.
        betas: ``(beta1, beta2)``; ``beta1`` forms the update, ``beta2`` tracks
            the momentum buffer.
        weight_decay: decoupled, as in AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    parameter.mul_(1 - lr * weight_decay)

                update = exp_avg.lerp(grad, 1 - beta1).sign_()
                parameter.add_(update, alpha=-lr)

                # Buffer updates after the step so this gradient isn't counted
                # in the direction it just produced.
                exp_avg.lerp_(grad, 1 - beta2)

        return loss
