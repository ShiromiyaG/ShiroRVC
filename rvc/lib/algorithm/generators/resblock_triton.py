"""Triton kernels for RefineGAN2's ``ParallelResBlock`` elementwise chain.

Every tensor is channels-last (B, C, 1, T), read as (rows, C) with channels
contiguous.  Each branch is ``AdaIN -> ResBlock -> AdaIN``; these fuse the
leaky_relus, the FP32 residual adds and the casts to the conv dtype between its
convs.  AdaIN's noise is drawn in-kernel from ``(seed, element index)`` and
redrawn in backward, so it is never stored.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit(do_not_specialize=["seed"])
def _enter_fwd(h_ptr, w_ptr, x0_ptr, t_ptr, N, C, seed, s_ada, s_res,
               NOISE: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < N)[:, None] & (c < C)[None, :]
    off = r[:, None] * C + c[None, :]
    v = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if NOISE:
        w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        v += tl.randn(seed, off) * w[None, :]
    x0 = tl.where(v > 0, v, v * s_ada)
    tl.store(x0_ptr + off, x0, mask=mask)
    t = tl.where(x0 > 0, x0, x0 * s_res)
    tl.store(t_ptr + off, t.to(t_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["seed"])
def _enter_bwd(h_ptr, w_ptr, gx0_ptr, gt_ptr, gh_ptr, gw_ptr, N, C, seed, s_ada, s_res,
               NOISE: tl.constexpr, W_GRAD: tl.constexpr,
               BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < N)[:, None] & (c < C)[None, :]
    off = r[:, None] * C + c[None, :]
    v = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    n = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    if NOISE:
        w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        n = tl.randn(seed, off)
        v += n * w[None, :]
    # x0 = lrelu(v) has v's sign, so one comparison serves both leaky_relus.
    pos = v > 0
    gx0 = tl.load(gx0_ptr + off, mask=mask, other=0.0).to(tl.float32)
    gt = tl.load(gt_ptr + off, mask=mask, other=0.0).to(tl.float32)
    gv = tl.where(pos, 1.0, s_ada) * (gx0 + tl.where(pos, 1.0, s_res) * gt)
    tl.store(gh_ptr + off, gv.to(gh_ptr.dtype.element_ty), mask=mask)
    if W_GRAD:
        tl.atomic_add(gw_ptr + c, tl.sum(gv * n, axis=0), mask=c < C)


@triton.jit
def _step_fwd(x_ptr, y_ptr, xn_ptr, t_ptr, NUMEL, slope, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < NUMEL
    xn = (tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
          + tl.load(y_ptr + i, mask=m, other=0.0).to(tl.float32))
    tl.store(xn_ptr + i, xn, mask=m)
    t = tl.where(xn > 0, xn, xn * slope)
    tl.store(t_ptr + i, t.to(t_ptr.dtype.element_ty), mask=m)


@triton.jit
def _step_bwd(t_ptr, gxn_ptr, gt_ptr, gx_ptr, gy_ptr, NUMEL, slope, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < NUMEL
    t = tl.load(t_ptr + i, mask=m, other=0.0).to(tl.float32)
    g = (tl.load(gxn_ptr + i, mask=m, other=0.0).to(tl.float32)
         + tl.where(t > 0, 1.0, slope) * tl.load(gt_ptr + i, mask=m, other=0.0).to(tl.float32))
    tl.store(gx_ptr + i, g, mask=m)
    tl.store(gy_ptr + i, g.to(gy_ptr.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["seed"])
def _leave_fwd(x_ptr, y_ptr, w_ptr, acc_ptr, out_ptr, pos_ptr, N, C, seed, slope, scale,
               NOISE: tl.constexpr, HAS_ACC: tl.constexpr,
               BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < N)[:, None] & (c < C)[None, :]
    off = r[:, None] * C + c[None, :]
    v = (tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
         + tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32))
    if NOISE:
        w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        v += tl.randn(seed, off) * w[None, :]
    out = tl.where(v > 0, v, v * slope) * scale
    if HAS_ACC:
        out += tl.load(acc_ptr + off, mask=mask, other=0.0)
    tl.store(out_ptr + off, out, mask=mask)
    tl.store(pos_ptr + off, (v > 0).to(tl.int8), mask=mask)


@triton.jit(do_not_specialize=["seed"])
def _leave_bwd(go_ptr, pos_ptr, w_ptr, gx_ptr, gy_ptr, gw_ptr, N, C, seed, slope, scale,
               NOISE: tl.constexpr, W_GRAD: tl.constexpr,
               BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < N)[:, None] & (c < C)[None, :]
    off = r[:, None] * C + c[None, :]
    pos = tl.load(pos_ptr + off, mask=mask, other=0) != 0
    gv = tl.load(go_ptr + off, mask=mask, other=0.0) * scale * tl.where(pos, 1.0, slope)
    tl.store(gx_ptr + off, gv, mask=mask)
    tl.store(gy_ptr + off, gv.to(gy_ptr.dtype.element_ty), mask=mask)
    if W_GRAD:
        n = tl.randn(seed, off)
        tl.atomic_add(gw_ptr + c, tl.sum(gv * n, axis=0), mask=c < C)


def _tiles(rows, channels):
    block_c = min(triton.next_power_of_2(channels), 128)
    block_r = max(16, 2048 // block_c)
    return (triton.cdiv(rows, block_r), triton.cdiv(channels, block_c)), block_r, block_c


def _cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def _seed():
    # CPU generator: ``torch.utils.checkpoint`` restores it before recomputing,
    # so a checkpointed forward redraws the same noise.
    return int(torch.randint(0, 2**31 - 1, (1,)).item())


class AdaINEnter(torch.autograd.Function):
    """``h -> (x0, t)``: ``x0 = lrelu(h + noise * weight)`` in FP32 (the
    ResBlock's residual stream) and ``t = lrelu(x0)`` in ``dtype`` (its first
    conv's input).  ``weight`` is unused when ``noise`` is False."""

    @staticmethod
    def forward(ctx, h, weight, noise, s_ada, s_res, dtype):
        h = _cl(h)
        B, C, _, T = h.shape
        x0 = torch.empty_like(h, dtype=torch.float32)
        t = torch.empty_like(h, dtype=dtype)
        seed = _seed() if noise else 0
        weight = weight if noise else h
        grid, br, bc = _tiles(B * T, C)
        _enter_fwd[grid](h, weight, x0, t, B * T, C, seed, s_ada, s_res,
                         NOISE=noise, BLOCK_R=br, BLOCK_C=bc)
        ctx.save_for_backward(h, weight)
        ctx.args = (noise, seed, s_ada, s_res)
        return x0, t

    @staticmethod
    def backward(ctx, gx0, gt):
        h, weight = ctx.saved_tensors
        noise, seed, s_ada, s_res = ctx.args
        B, C, _, T = h.shape
        w_grad = noise and ctx.needs_input_grad[1]
        gh = torch.empty_like(h)
        gw = torch.zeros(C, device=h.device, dtype=torch.float32) if w_grad else None
        grid, br, bc = _tiles(B * T, C)
        _enter_bwd[grid](h, weight, _cl(gx0), _cl(gt), gh, gh if gw is None else gw,
                         B * T, C, seed, s_ada, s_res, NOISE=noise, W_GRAD=w_grad,
                         BLOCK_R=br, BLOCK_C=bc)
        return gh, gw, None, None, None, None


class ResidualStep(torch.autograd.Function):
    """``(x, y) -> (x + y, lrelu(x + y))``: the next residual in FP32 and the
    next pair's conv input in ``dtype``."""

    @staticmethod
    def forward(ctx, x, y, slope, dtype):
        x, y = _cl(x), _cl(y)
        xn = torch.empty_like(x, dtype=torch.float32)
        t = torch.empty_like(x, dtype=dtype)
        numel = x.numel()
        _step_fwd[(triton.cdiv(numel, BLOCK),)](x, y, xn, t, numel, slope, BLOCK=BLOCK)
        ctx.save_for_backward(t)
        ctx.args = (slope, x.dtype, y.dtype)
        return xn, t

    @staticmethod
    def backward(ctx, gxn, gt):
        (t,) = ctx.saved_tensors
        slope, x_dtype, y_dtype = ctx.args
        gx = torch.empty_like(t, dtype=x_dtype)
        gy = torch.empty_like(t, dtype=y_dtype)
        numel = t.numel()
        _step_bwd[(triton.cdiv(numel, BLOCK),)](t, _cl(gxn), _cl(gt), gx, gy, numel, slope,
                                                BLOCK=BLOCK)
        return gx, gy, None, None


class AdaINLeave(torch.autograd.Function):
    """``(x, y) -> acc + scale * lrelu(x + y + noise * weight)`` in FP32: the
    last residual add, the closing AdaIN and the branch average.  ``acc`` may
    be None for the first branch."""

    @staticmethod
    def forward(ctx, x, y, weight, noise, slope, acc, scale):
        x, y = _cl(x), _cl(y)
        B, C, _, T = x.shape
        out = torch.empty_like(x, dtype=torch.float32)
        pos = torch.empty_like(x, dtype=torch.int8)
        seed = _seed() if noise else 0
        weight = weight if noise else x
        grid, br, bc = _tiles(B * T, C)
        _leave_fwd[grid](x, y, weight, x if acc is None else _cl(acc), out, pos, B * T, C,
                         seed, slope, scale, NOISE=noise, HAS_ACC=acc is not None,
                         BLOCK_R=br, BLOCK_C=bc)
        ctx.save_for_backward(pos, weight)
        ctx.args = (noise, seed, slope, scale, x.dtype, y.dtype, acc is not None)
        return out

    @staticmethod
    def backward(ctx, go):
        pos, weight = ctx.saved_tensors
        noise, seed, slope, scale, x_dtype, y_dtype, has_acc = ctx.args
        B, C, _, T = pos.shape
        go = _cl(go)
        w_grad = noise and ctx.needs_input_grad[2]
        gx = torch.empty_like(pos, dtype=x_dtype)
        gy = torch.empty_like(pos, dtype=y_dtype)
        gw = torch.zeros(C, device=go.device, dtype=torch.float32) if w_grad else None
        grid, br, bc = _tiles(B * T, C)
        _leave_bwd[grid](go, pos, weight, gx, gy, gx if gw is None else gw, B * T, C,
                         seed, slope, scale, NOISE=noise, W_GRAD=w_grad,
                         BLOCK_R=br, BLOCK_C=bc)
        return gx, gy, gw, None, None, go if has_acc else None, None
