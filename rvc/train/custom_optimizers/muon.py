"""Muon: momentum orthogonalised by Newton-Schulz, with an AdamW fallback.

The update is RMS-matched to AdamW (``_RMS_TARGET``) rather than run on a
separate, much larger learning rate as in published Muon setups: an
orthogonal (A, B) matrix has every singular value 1, so its RMS is
``1/sqrt(max(A, B))``, typically 20-100x smaller than AdamW's ~0.2. Scaling
the update instead lets Muon share this repo's learning rate, warmup and
scheduler, so switching optimizer doesn't silently invalidate a tuned
``learning_rate_g``.

Anything not genuinely a matrix falls back to AdamW inside the same
optimizer: biases/norm gains (``ndim < 2``), speaker embeddings
(orthogonalising rows would erase the learned per-speaker scale), and the
``original0`` gain tensors of ``weight_norm`` parametrisations, which reshape
to ``(out, 1)`` -- orthogonalising a single-column matrix just renormalises
it.
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


#: RMS an AdamW step lands near once the second moment has settled; matching
#: it is what lets Muon share this repository's learning rates.
_RMS_TARGET = 0.2

#: Coefficients of Keller Jordan's quintic iteration. They deliberately do
#: not converge to exactly 1, which is fine because only the direction matters.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315

#: Reference implementations use 5 steps, which suffices for wide matrices
#: but not near-square ones (smallest singular value starts near zero, where
#: the quintic converges slowest). Measured worst-case smallest singular
#: value after the iteration (8 random Gaussians per shape) only clears 0.5
#: on all tested shapes at 9 steps; cost is 1.31ms (5 steps) vs 2.31ms (9
#: steps) on this model's largest tensor (512x3584), against a training step
#: of tens of ms.
DEFAULT_NS_STEPS = 9


@torch.no_grad()
def newton_schulz(
    matrix: torch.Tensor, steps: int = DEFAULT_NS_STEPS, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the orthogonal factor of ``matrix``'s polar decomposition.

    Runs in bfloat16 on CUDA: the iteration only needs the singular values
    driven towards 1, so the precision buys nothing and the halved bandwidth is
    most of the cost.  CPU stays in float32, where bfloat16 matmul support is
    uneven.
    """
    assert matrix.ndim == 2, "Newton-Schulz needs a matrix"

    work_dtype = torch.bfloat16 if matrix.is_cuda else torch.float32
    X = matrix.to(work_dtype)

    # The iteration costs O(min(A, B)^2 * max(A, B)); transposing so the short
    # side leads keeps the Gram matrix small.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    # Normalising first puts every singular value inside the iteration's basin
    # of convergence.
    X = X / X.norm().clamp_min(eps)

    for _ in range(steps):
        gram = X @ X.T
        polynomial = _NS_B * gram + _NS_C * (gram @ gram)
        X = _NS_A * X + polynomial @ X

    if transposed:
        X = X.T
    return X.to(matrix.dtype)


def _is_matrix(parameter: torch.Tensor) -> bool:
    """Whether orthogonalising this parameter is meaningful.

    Convolution kernels count: folding the spatial taps into the input axis
    gives ``(out_channels, in_channels * taps)``, which is the linear map the
    layer applies. A reshape with a singleton axis is a vector wearing a
    matrix's shape, and orthogonalising it would only renormalise it.
    """
    if parameter.ndim < 2:
        return False
    rows = parameter.shape[0]
    columns = parameter.numel() // rows
    return min(rows, columns) > 1


class Muon(Optimizer):
    """Muon for matrices, AdamW for everything else.

    Args:
        params: parameters or param groups, exactly as for any torch optimizer.
        lr: shared with the AdamW fallback -- see the RMS matching above.
        momentum: heavy-ball coefficient for the orthogonalised branch.
        nesterov: look ahead one momentum step before orthogonalising.
        ns_steps: Newton-Schulz iterations; see ``DEFAULT_NS_STEPS``.
        betas, eps: AdamW-branch settings, for the parameters Muon skips.
        weight_decay: decoupled, applied identically on both branches.
        adamw_params: parameters to force onto the AdamW branch regardless of
            shape. Embeddings belong here.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = DEFAULT_NS_STEPS,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adamw_params=None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Identity, not equality: two parameters can hold equal tensors.
        self._forced_adamw = {id(p) for p in (adamw_params or [])}

    def _use_muon(self, parameter: torch.Tensor) -> bool:
        return id(parameter) not in self._forced_adamw and _is_matrix(parameter)

    def parameter_split(self) -> tuple[int, int]:
        """``(muon_count, adamw_count)``, for reporting which branch got what."""
        muon = adamw = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if self._use_muon(parameter):
                    muon += 1
                else:
                    adamw += 1
        return muon, adamw

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[parameter]

                # Decoupled: the decay is a pull towards zero applied to the
                # weight, never routed through the adaptive or orthogonalised
                # update, so it stays comparable across both branches.
                if weight_decay != 0:
                    parameter.mul_(1 - lr * weight_decay)

                if self._use_muon(parameter):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer = state["momentum_buffer"]
                    buffer.lerp_(grad, 1 - momentum)
                    direction = grad.lerp(buffer, momentum) if nesterov else buffer

                    rows = parameter.shape[0]
                    flattened = direction.reshape(rows, -1)
                    orthogonal = newton_schulz(flattened, steps=ns_steps)

                    # RMS of an orthogonal (A, B) matrix is 1/sqrt(max(A, B)).
                    scale = _RMS_TARGET * math.sqrt(max(flattened.shape))
                    parameter.add_(orthogonal.view_as(parameter), alpha=-lr * scale)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    denominator = (
                        exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(eps)
                    parameter.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)

        return loss
