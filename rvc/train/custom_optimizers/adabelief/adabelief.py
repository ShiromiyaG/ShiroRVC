import math

import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer


# -------------------------- Triton kernel --------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
    restore_value=["param_ptr", "exp_avg_ptr", "exp_avg_var_ptr"],
)
@triton.jit
def _adabelief_kernel(
    # Tensor pointers
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_var_ptr,
    # Scalar hyperparameters (passed as kernel args, not constexpr)
    beta1,
    beta2,
    eps,
    step_size,  # Pre-computed: lr * correction / bias_correction1
    wd_factor,  # Pre-computed: (1 - lr*wd) for decoupled, or wd for coupled
    inv_bc2,  # Pre-computed: 1/(1 - beta2^t), only used when IS_RECTIFY=False
    # Compile-time mode flags -> dead-code elimination
    USE_ADAPTIVE: tl.constexpr,  # True = divide by denom, False = SGD-momentum
    WEIGHT_DECOUPLE: tl.constexpr,
    HAS_WEIGHT_DECAY: tl.constexpr,
    IS_RECTIFY: tl.constexpr,
    # Tensor length
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # -- Load & upcast to fp32 for numerical stability --
    p = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    m = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
    s = tl.load(exp_avg_var_ptr + offsets, mask=mask).to(tl.float32)

    # -- Weight decay --
    if HAS_WEIGHT_DECAY:
        if WEIGHT_DECOUPLE:
            p = p * wd_factor  # p *= (1 - lr * wd)
        else:
            g = g + wd_factor * p  # L2 regularisation added to grad

    # -- First moment (momentum) --
    m = beta1 * m + (1.0 - beta1) * g

    # -- Belief variance (core AdaBelief difference from Adam) --
    # Uses (g - m_t)^2 instead of g^2, so the second moment tracks
    # how much the gradient deviates from the predicted direction.
    residual = g - m
    s = beta2 * s + (1.0 - beta2) * residual * residual

    # -- Parameter update --
    if USE_ADAPTIVE:
        if IS_RECTIFY:
            # Rectified path: bias-correction2 is folded into the rect term
            # that was pre-computed into step_size on the host side.
            denom = tl.sqrt(s) + eps
        else:
            # Standard path: apply bias-correction2 inside the sqrt.
            # Equivalent to sqrt(s + eps) / sqrt(bc2) + eps from the
            # original code, but without the in-place eps accumulation
            # into the state tensor.
            denom = tl.sqrt((s + eps) * inv_bc2) + eps
        p = p - step_size * m / denom
    else:
        # SGD-with-momentum fallback (early rectified steps when rho_t < 5)
        p = p - step_size * m

    # -- Store --
    tl.store(param_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, m, mask=mask)
    tl.store(exp_avg_var_ptr + offsets, s, mask=mask)


# -------------------------- Optimizer class --------------------------


class AdaBelief(Optimizer):
    r"""AdaBelief optimizer with a fused Triton kernel.

    All per-element work (weight decay, moment updates, parameter step) is
    executed in a single GPU kernel launch per parameter tensor, with no
    intermediate allocations, no Python-level data-dependent branches on
    tensor values, and therefore fully compatible with ``torch.compile``.

    Algorithm (rectified variant, ``rectify=True``):

        m_t  = b1*m_{t-1}  + (1-b1)*g_t
        s_t  = b2*s_{t-1}  + (1-b2)*(g_t - m_t)^2
        rho_t  = rho_inf - 2*t*b2^t / (1 - b2^t),  rho_inf = 2/(1-b2) - 1

        if rho_t >= 5:                      (variance is tractable)
            r_t  = sqrt[(1-b2^t)(rho_t-4)(rho_t-2)*rho_inf
                        / ((rho_inf-4)(rho_inf-2)*rho_t)]
            p_t  = p_{t-1}  - lr*r_t / (1-b1^t) * m_t / (sqrt(s_t) + eps)
        else:                               (SGD with momentum)
            p_t  = p_{t-1}  - lr / (1-b1^t) * m_t

    Weight decay modes (weight_decouple=True):
        - default: wd_factor = 1 - lr * wd (AdamW behaviour, scales with lr)
        - fixed_decay: wd_factor = 1 - wd (constant, ignores lr)
        - adamc: wd_factor = 1 - (lr/lr_max)*lr*wd (Defazio 2025 correction, keeps gradient-to-weight ratio stable under cosine/linear schedules)
    Reference:
        Zhuang et al., "AdaBelief Optimizer: Adapting Stepsizes by the
        Belief in Observed Gradients", NeurIPS 2020.

        Defazio, "Why Gradients Rapidly Increase Near the End of Training",
        arXiv 2506.02285, 2025.  (AdamC weight-decay correction)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0,
        weight_decouple: bool = True,
        rectify: bool = True,
        fixed_decay: bool = False,
        degenerated_to_sgd: bool = True,
        adamc: bool = False,
        lr_max: float | None = None,
    ):
        """
        Args:
            params:           Iterable of parameters or param groups.
            lr:               Learning rate.
            betas:            (beta1, beta2) momentum coefficients.
            eps:              Denominator floor. Use 1e-16 for voice/sequence
                              tasks (Adam-favourable); 1e-8 for CV tasks where
                              SGD is competitive.
            weight_decay:     Weight decay coefficient.
            weight_decouple:  If True, apply weight decay as a decoupled
                              multiplicative term (AdamW-style). If False,
                              add L2 penalty to the gradient.
            rectify:          If True, use RAdam-style variance rectification.
                              Falls back to SGD-momentum for early steps while
                              the variance estimate is unreliable (rho_t < 5).
            fixed_decay:      Only used when weight_decouple=True and adamc=False.
                              If True, wd_factor = (1 - wd), ignoring lr.
                              If False, wd_factor = (1 - lr*wd), matching AdamW.
            degenerated_to_sgd: When rectify=True and rho_t < 5, fall back to
                              SGD-momentum if True, or skip the param update if
                              False (moments still accumulate either way).
            adamc:            If True, apply the AdamC weight-decay correction
                              from Defazio (2025): scale wd by (lr / lr_max) so
                              the gradient-to-weight-norm ratio stays constant
                              under a decaying LR schedule, preventing the
                              gradient-norm blow-up seen near end-of-training.
                              Requires weight_decouple=True; fixed_decay is
                              ignored when adamc=True.
            lr_max:           Peak learning rate for the AdamC correction.
                              If None (default), it is auto-tracked as the
                              maximum lr seen across all param groups. Pass
                              explicitly if you use a warmup that ramps lr up
                              from a value lower than the true peak.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if adamc and not weight_decouple:
            raise ValueError("adamc=True requires weight_decouple=True")
        if lr_max is not None and not 0.0 < lr_max:
            raise ValueError(f"Invalid lr_max: {lr_max}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        self.degenerated_to_sgd = degenerated_to_sgd
        self.adamc = adamc
        # lr_max=None means auto-track; will be updated on first step().
        self._lr_max: float | None = lr_max

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #

    @torch._dynamo.disable
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step.

        The actual tensor work is inside a fused Triton kernel (already
        compiled to PTX), so dynamo tracing the Python wrapper provides
        no benefit. ``@torch._dynamo.disable`` keeps the step opaque to
        dynamo, avoiding symbolic-variable issues with autotune while
        preserving full speed.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # AdamC: update peak-lr tracker once per step, across all groups
        # Done before the main loop so peak_lr is always current when we compute wd_factor below.
        if self.adamc and self._lr_max is None:
            for pg in self.param_groups:
                if self._lr_max is None:
                    self._lr_max = pg["lr"]
                else:
                    self._lr_max = max(self._lr_max, pg["lr"])

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            has_wd = wd != 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBelief does not support sparse gradients"
                    )

                # -- State initialisation (runs once per param) --
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]

                # -- Pre-compute scalars on the host (no tensor ops) --
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                use_adaptive = True

                if self.rectify:
                    beta2_t = beta2 ** step
                    rho_inf = 2.0 / (1.0 - beta2) - 1.0
                    rho_t = rho_inf - 2.0 * step * beta2_t / (1.0 - beta2_t)

                    if rho_t >= 5.0:
                        rect = math.sqrt(
                            (1.0 - beta2_t)
                            * (rho_t - 4.0)
                            / (rho_inf - 4.0)
                            * (rho_t - 2.0)
                            / rho_t
                            * rho_inf
                            / (rho_inf - 2.0)
                        )
                        step_size = lr * rect / bias_correction1
                        use_adaptive = True
                    elif self.degenerated_to_sgd:
                        step_size = lr / bias_correction1
                        use_adaptive = False
                    else:
                        # Moments still get updated, but p is unchanged.
                        step_size = 0.0
                        use_adaptive = False
                else:
                    step_size = lr / bias_correction1
                    use_adaptive = True

                # Weight-decay factor
                if self.weight_decouple:
                    if self.adamc:
                        # AdamC correction (Defazio 2025):
                        wd_factor = 1.0 - (lr / self._lr_max) * lr * wd
                    elif self.fixed_decay:
                        # Constant decay regardless of lr — wd is the fraction
                        # subtracted each step, not scaled by the schedule.
                        wd_factor = 1.0 - wd
                    else:
                        # Standard AdamW: decay scales with current lr.
                        wd_factor = 1.0 - lr * wd
                else:
                    wd_factor = wd

                inv_bc2 = 1.0 / bias_correction2  # only read in non-rectify path

                # -- Launch Triton kernel --
                n = p.numel()
                grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

                _adabelief_kernel[grid](
                    p.data,
                    grad,
                    exp_avg,
                    exp_avg_var,
                    beta1,
                    beta2,
                    eps,
                    step_size,
                    wd_factor,
                    inv_bc2,
                    USE_ADAPTIVE=use_adaptive,
                    WEIGHT_DECOUPLE=self.weight_decouple,
                    HAS_WEIGHT_DECAY=has_wd,
                    IS_RECTIFY=self.rectify,
                    n_elements=n,
                )

        return loss
