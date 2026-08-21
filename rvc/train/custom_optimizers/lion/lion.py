"""Lion: EvoLved Sign Momentum (Chen et al., 2023).

Lion keeps one momentum buffer where Adam keeps two, and its update is the
*sign* of an interpolation between the gradient and that buffer.  Two
consequences matter here:

* **Optimizer state halves.**  For a GAN this is the generator, the
  discriminator and (usually) a weight EMA sharing one card, so the saving is
  real VRAM rather than a benchmark number.
* **Every parameter moves by exactly ``lr``.**  The update is a sign, so its
  magnitude carries no information about the gradient's scale.  That is why
  Lion needs a learning rate several times smaller than AdamW's -- see
  ``LR_SCALE`` -- and a correspondingly larger weight decay to keep the
  decay-to-step ratio where AdamW had it.

The two betas do different jobs and are deliberately not the AdamW pair: the
update is formed with ``beta1`` (0.9, a short lookahead) while the buffer is
tracked with the slower ``beta2`` (0.99).
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


#: Lion's learning rate relative to a tuned AdamW one.  The authors report 3-10x
#: smaller; the conservative end of that range is the right default for a GAN,
#: where the discriminator makes a too-large generator step expensive to undo.
LR_SCALE = 1.0 / 3.0

#: Weight decay relative to AdamW's.  ``lr * weight_decay`` is the quantity that
#: sets how hard weights are pulled to zero, so raising decay by the same factor
#: the learning rate fell keeps that pull unchanged.
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

                # The update looks one step ahead of the buffer, which is what
                # separates Lion from plain signSGD with momentum.
                update = exp_avg.lerp(grad, 1 - beta1).sign_()
                parameter.add_(update, alpha=-lr)

                # The buffer itself moves on the slower beta, and is updated
                # after the step so this iteration's gradient is not counted in
                # the direction it just produced.
                exp_avg.lerp_(grad, 1 - beta2)

        return loss
