"""Triton kernels for Nagi's block FIR: every block of ``W`` output samples
is its sources' samples through that block's own kernels, summed over the
sources. It reads the sources in place, where the grouped convolution it
replaces first copies every block's samples out.

With ``x`` [B, C, T] the padded sources, ``k`` [B, L, C, K] the kernels and
``hop`` the blocks' spacing:

    y[b, l, m] = sum_c sum_q x[b, c, l * hop + m + q] * k[b, l, c, K - 1 - q]

Its backward is the kernels' gradient alone: the sources are not learned.
"""

import torch
import triton
import triton.language as tl

BLOCK = 128
BLOCK_K = 64
BLOCK_W = 64


@triton.jit
def _fir(x_ptr, k_ptr, y_ptr, T, L, HOP: tl.constexpr, C: tl.constexpr, K: tl.constexpr, W: tl.constexpr,
         BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    m = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    batch = (row // L).to(tl.int64)
    start = (row % L) * HOP
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(C):
        x_row = x_ptr + (batch * C + c) * T + start
        k_row = k_ptr + (row.to(tl.int64) * C + c) * K
        for first in tl.static_range(0, K, BLOCK_K):
            q = first + tl.arange(0, BLOCK_K)
            h = tl.load(k_row + (K - 1 - q), mask=q < K, other=0.0)
            mask = (m[:, None] < W) & (q[None, :] < K)
            xv = tl.load(x_row + m[:, None] + q[None, :], mask=mask, other=0.0)
            acc += tl.sum(xv * h[None, :], axis=1)
    tl.store(y_ptr + row.to(tl.int64) * W + m, acc, mask=m < W)


@triton.jit
def _fir_grad(x_ptr, g_ptr, gk_ptr, T, L, HOP: tl.constexpr, C: tl.constexpr, K: tl.constexpr, W: tl.constexpr,
              BLOCK: tl.constexpr, BLOCK_W: tl.constexpr):
    row = tl.program_id(0)
    u = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    block = row // C
    batch = (block // L).to(tl.int64)
    x_row = x_ptr + (batch * C + row % C) * T + (block % L) * HOP
    g_row = g_ptr + block.to(tl.int64) * W
    # Tap u of the kernel met the sample K - 1 - u past each output's.
    q = K - 1 - u
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for first in tl.static_range(0, W, BLOCK_W):
        m = first + tl.arange(0, BLOCK_W)
        g = tl.load(g_row + m, mask=m < W, other=0.0)
        mask = (u[:, None] < K) & (m[None, :] < W)
        xv = tl.load(x_row + q[:, None] + m[None, :], mask=mask, other=0.0)
        acc += tl.sum(xv * g[None, :], axis=1)
    tl.store(gk_ptr + row.to(tl.int64) * K + u, acc, mask=u < K)


class BlockFIR(torch.autograd.Function):
    """``(x, kernels, hop, window) -> y`` [B, L, window], all FP32; ``x``
    must hold ``(L - 1) * hop + window + K - 1`` samples and gets no gradient."""

    @staticmethod
    def forward(ctx, x, kernels, hop, window):
        x, kernels = x.contiguous(), kernels.contiguous()
        batch, blocks, channels, taps = kernels.shape
        y = torch.empty(batch, blocks, window, device=x.device, dtype=torch.float32)
        _fir[(batch * blocks, triton.cdiv(window, BLOCK))](
            x, kernels, y, x.shape[-1], blocks, HOP=hop, C=channels, K=taps, W=window, BLOCK=BLOCK, BLOCK_K=BLOCK_K,
        )
        ctx.save_for_backward(x)
        ctx.shape, ctx.hop = kernels.shape, hop
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        batch, blocks, channels, taps = ctx.shape
        gk = torch.empty(ctx.shape, device=x.device, dtype=torch.float32)
        _fir_grad[(batch * blocks * channels, triton.cdiv(taps, BLOCK))](
            x, gy.contiguous(), gk, x.shape[-1], blocks, HOP=ctx.hop, C=channels, K=taps, W=gy.shape[-1],
            BLOCK=BLOCK, BLOCK_W=BLOCK_W,
        )
        return None, gk, None, None
