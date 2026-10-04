"""Triton kernels for a stride-1 depthwise conv2d over an already padded
input, with both gradients: Wavehax's 15 x 7 convs, where PyTorch's own
depthwise kernels take most of a training step (their backward above all).

Every kernel sees ``B * C`` rows. Names, with ``g`` in front for a gradient:

    xp      padded input, (rows, HP, WP)
    w       kernel, (C, KH * KW), FP32
    y       output, (rows, H, W), in the input's dtype

Sums are accumulated in FP32 whatever the tensors' dtype.
"""

import torch
import triton
import triton.language as tl

BLOCK = 512


@triton.jit
def _fwd(xp_ptr, w_ptr, y_ptr, C, H, W, WP, HPWP, KH: tl.constexpr, KW: tl.constexpr,
         BLOCK: tl.constexpr):
    """``y[h, w] = sum_ij w[i, j] * xp[h + i, w + j]``; a program is one block of one row."""
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < H * W
    base = xp_ptr + row.to(tl.int64) * HPWP + (n // W) * WP + n % W
    w_row = w_ptr + (row % C) * (KH * KW)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(KH):
        for j in tl.static_range(KW):
            acc += tl.load(w_row + i * KW + j) * tl.load(base + i * WP + j, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row.to(tl.int64) * (H * W) + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_input(gy_ptr, w_ptr, gx_ptr, C, H, W, HP, WP, KH: tl.constexpr, KW: tl.constexpr,
               BLOCK: tl.constexpr):
    """``gxp[p, q] = sum_ij w[i, j] * gy[p - i, q - j]`` over the outputs that exist."""
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < HP * WP
    p = n // WP
    q = n % WP
    gy_row = gy_ptr + row.to(tl.int64) * (H * W)
    w_row = w_ptr + (row % C) * (KH * KW)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(KH):
        hh = p - i
        in_rows = mask & (hh >= 0) & (hh < H)
        for j in tl.static_range(KW):
            ww = q - j
            inside = in_rows & (ww >= 0) & (ww < W)
            acc += tl.load(w_row + i * KW + j) * tl.load(gy_row + hh * W + ww, mask=inside, other=0.0).to(tl.float32)
    tl.store(gx_ptr + row.to(tl.int64) * (HP * WP) + n, acc.to(gx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_weight(xp_ptr, gy_ptr, gw_ptr, B, C, H, W, WP, HPWP, KH: tl.constexpr, KW: tl.constexpr,
                KW_LANES: tl.constexpr, BLOCK: tl.constexpr):
    """``gw[c, i, j] = sum_bhw xp[h + i, w + j] * gy[h, w]``. A program is one
    kernel row of one channel: it walks every batch item and output block,
    keeps the row's taps apart per lane and reduces once, which costs a
    fraction of a reduction per block. ``KW_LANES`` is ``KW`` rounded up to a
    power of two, as Triton's shapes have to be."""
    ch = tl.program_id(0)
    i = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    j = tl.arange(0, KW_LANES)
    taps = j < KW
    acc = tl.zeros((KW_LANES, BLOCK), dtype=tl.float32)
    for b in range(B):
        row = (b * C + ch).to(tl.int64)
        for start in range(0, H * W, BLOCK):
            n = start + lane
            mask = n < H * W
            gy = tl.load(gy_ptr + row * (H * W) + n, mask=mask, other=0.0).to(tl.float32)
            base = xp_ptr + row * HPWP + (n // W + i) * WP + n % W
            xv = tl.load(base[None, :] + j[:, None], mask=mask[None, :] & taps[:, None], other=0.0)
            acc += xv.to(tl.float32) * gy[None, :]
    tl.store(gw_ptr + ch * (KH * KW) + i * KW + j, tl.sum(acc, 1), mask=taps)


class DepthwiseConv2d(torch.autograd.Function):
    """``F.conv2d(xp, weight, groups=C)`` at stride 1 with no padding, for
    ``xp`` (B, C, HP, WP) on CUDA and ``weight`` (C, 1, KH, KW)."""

    @staticmethod
    def forward(ctx, xp, weight):
        batch, channels, hp, wp = xp.shape
        kh, kw = weight.shape[-2:]
        h, w = hp - kh + 1, wp - kw + 1
        xp = xp.contiguous()
        kernel = weight.detach().float().reshape(channels, kh * kw).contiguous()
        y = torch.empty(batch, channels, h, w, device=xp.device, dtype=xp.dtype)
        grid = (batch * channels, triton.cdiv(h * w, BLOCK))
        _fwd[grid](xp, kernel, y, channels, h, w, wp, hp * wp, KH=kh, KW=kw, BLOCK=BLOCK)
        ctx.save_for_backward(xp, weight)
        return y

    @staticmethod
    def backward(ctx, gy):
        xp, weight = ctx.saved_tensors
        batch, channels, hp, wp = xp.shape
        kh, kw = weight.shape[-2:]
        h, w = hp - kh + 1, wp - kw + 1
        gy = gy.contiguous()
        gx = gw = None
        if ctx.needs_input_grad[0]:
            kernel = weight.detach().float().reshape(channels, kh * kw).contiguous()
            gx = torch.empty_like(xp)
            grid = (batch * channels, triton.cdiv(hp * wp, BLOCK))
            _bwd_input[grid](gy, kernel, gx, channels, h, w, hp, wp, KH=kh, KW=kw, BLOCK=BLOCK)
        if ctx.needs_input_grad[1]:
            gw = torch.empty(channels, kh * kw, device=xp.device, dtype=torch.float32)
            _bwd_weight[(channels, kh)](
                xp, gy, gw, batch, channels, h, w, wp, hp * wp,
                KH=kh, KW=kw, KW_LANES=triton.next_power_of_2(kw), BLOCK=BLOCK,
            )
            gw = gw.reshape(weight.shape).to(weight.dtype)
        return gx, gw


@torch.compiler.disable  # the kernels are launched from Python
def depthwise_conv2d(xp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return DepthwiseConv2d.apply(xp, weight)
