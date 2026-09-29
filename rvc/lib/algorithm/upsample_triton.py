"""Triton kernels for ``AntiAliasedUpsample1d``'s polyphase interpolation and
RefineGAN2's integer-ratio decimation.

The same arithmetic as the grouped ``conv1d`` plus interleave, written
straight into the output layout.  cuDNN runs the grouped conv's backward in a
slow direct kernel and wraps every call in layout conversions; here forward
and backward are one kernel each.
"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _up_fwd(xp_ptr, w_ptr, y_ptr, T, XP_LEN, FACTOR: tl.constexpr,
            PHASES: tl.constexpr, TAPS: tl.constexpr, BLOCK: tl.constexpr):
    """Every phase from one read of the input: y[a * FACTOR + p]."""
    row = tl.program_id(0)
    a = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = a < T
    # ``tl.arange`` needs a power of two; ``PHASES`` rounds ``FACTOR`` up.
    p = tl.arange(0, PHASES)
    live = p < FACTOR
    xp_row = xp_ptr + row.to(tl.int64) * XP_LEN
    acc = tl.zeros((BLOCK, PHASES), dtype=tl.float32)
    for t in tl.range(TAPS):
        w = tl.load(w_ptr + p * TAPS + t, mask=live, other=0.0)
        x = tl.load(xp_row + a + t, mask=mask, other=0.0).to(tl.float32)
        acc += x[:, None] * w[None, :]
    out = y_ptr + row.to(tl.int64) * T * FACTOR + a[:, None] * FACTOR + p[None, :]
    tl.store(out, acc, mask=mask[:, None] & live[None, :])


@triton.jit
def _up_bwd(gp_ptr, w_ptr, gxp_ptr, T, XP_LEN, FACTOR: tl.constexpr,
            TAPS: tl.constexpr, BLOCK: tl.constexpr):
    """``gp`` is the output gradient split into phases, (rows, FACTOR, T)."""
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i < XP_LEN
    gp_row = gp_ptr + row.to(tl.int64) * FACTOR * T
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in tl.range(TAPS):
        a = i - t
        am = mask & (a >= 0) & (a < T)
        for p in tl.static_range(FACTOR):
            g = tl.load(gp_row + p * T + a, mask=am, other=0.0).to(tl.float32)
            acc += tl.load(w_ptr + p * TAPS + t) * g
    tl.store(gxp_ptr + row.to(tl.int64) * XP_LEN + i, acc, mask=mask)


class PolyphaseUpsample(torch.autograd.Function):
    """``(xp, weight, T) -> (B, C, T * factor)``.

    ``xp`` is the input replicate-padded by ``phase_pad``; ``weight`` the
    polyphase kernel, (factor, taps), float32.
    """

    @staticmethod
    def forward(ctx, xp, weight, T):
        B, C, XP_LEN = xp.shape
        factor, taps = weight.shape
        xp = xp.contiguous()
        y = torch.empty(B, C, T * factor, device=xp.device, dtype=xp.dtype)
        grid = (B * C, triton.cdiv(T, BLOCK))
        _up_fwd[grid](xp, weight, y, T, XP_LEN, FACTOR=factor,
                      PHASES=triton.next_power_of_2(factor), TAPS=taps, BLOCK=BLOCK)
        ctx.save_for_backward(weight)
        ctx.shape = (B, C, XP_LEN, T)
        ctx.dtype = xp.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        (weight,) = ctx.saved_tensors
        B, C, XP_LEN, T = ctx.shape
        factor, taps = weight.shape
        phases = gy.reshape(B * C, T, factor).transpose(1, 2).contiguous()
        gxp = torch.empty(B, C, XP_LEN, device=gy.device, dtype=ctx.dtype)
        grid = (B * C, triton.cdiv(XP_LEN, BLOCK))
        _up_bwd[grid](phases, weight, gxp, T, XP_LEN, FACTOR=factor, TAPS=taps, BLOCK=BLOCK)
        return gxp, None, None


@triton.jit
def _down_fwd(x_ptr, h_ptr, y_ptr, L, T, WIDTH, FACTOR: tl.constexpr,
              TAPS: tl.constexpr, BLOCK: tl.constexpr):
    """``y[n] = sum_k h[k] * x[n * FACTOR + k - WIDTH]``, zero outside ``x``."""
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < T
    x_row = x_ptr + row.to(tl.int64) * L
    base = n * FACTOR - WIDTH
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.range(TAPS):
        i = base + k
        x = tl.load(x_row + i, mask=mask & (i >= 0) & (i < L), other=0.0).to(tl.float32)
        acc += tl.load(h_ptr + k) * x
    tl.store(y_ptr + row.to(tl.int64) * T + n, acc, mask=mask)


@triton.jit
def _down_bwd(g_ptr, h_ptr, gx_ptr, L, T, WIDTH, FACTOR: tl.constexpr,
              TAPS: tl.constexpr, JTAPS: tl.constexpr, BLOCK: tl.constexpr):
    """The adjoint: input ``i`` gathers the ``JTAPS`` outputs whose taps reach it."""
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i < L
    q = i + WIDTH
    n0 = q // FACTOR
    phase = q - n0 * FACTOR
    g_row = g_ptr + row.to(tl.int64) * T
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.range(JTAPS):
        n = n0 - j
        k = phase + j * FACTOR
        live = mask & (n >= 0) & (n < T) & (k < TAPS)
        g = tl.load(g_row + n, mask=live, other=0.0).to(tl.float32)
        acc += tl.load(h_ptr + k, mask=live, other=0.0) * g
    tl.store(gx_ptr + row.to(tl.int64) * L + i, acc, mask=mask)


class PolyphaseDecimate(torch.autograd.Function):
    """``(x, kernel, factor, width) -> (B, C, ceil(L / factor))``.

    torchaudio's ``_apply_sinc_resample_kernel`` at a ``factor:1`` ratio:
    ``kernel`` is its (taps,) float32 filter and ``width`` its left zero pad.
    """

    @staticmethod
    def forward(ctx, x, kernel, factor, width):
        B, C, L = x.shape
        x = x.contiguous()
        T = -(-L // factor)
        y = torch.empty(B, C, T, device=x.device, dtype=x.dtype)
        grid = (B * C, triton.cdiv(T, BLOCK))
        _down_fwd[grid](x, kernel, y, L, T, width, FACTOR=factor,
                        TAPS=kernel.numel(), BLOCK=BLOCK)
        ctx.save_for_backward(kernel)
        ctx.meta = (B, C, L, T, factor, width, x.dtype)
        return y

    @staticmethod
    def backward(ctx, gy):
        (kernel,) = ctx.saved_tensors
        B, C, L, T, factor, width, dtype = ctx.meta
        taps = kernel.numel()
        gx = torch.empty(B, C, L, device=gy.device, dtype=dtype)
        grid = (B * C, triton.cdiv(L, BLOCK))
        _down_bwd[grid](gy.contiguous(), kernel, gx, L, T, width, FACTOR=factor,
                        TAPS=taps, JTAPS=triton.cdiv(taps, factor), BLOCK=BLOCK)
        return gx, None, None, None
