"""Triton kernels for ChouwaGAN's ``AntiAliasedLeakyReLU``: the 2x polyphase
upsampler, the leaky ReLU and the stride-2 lowpass, with the activated 2x
signal in one FP32 scratch buffer and recomputed from the input in backward,
so autograd keeps only the input.

Every kernel sees the tensors as ``B * C`` rows of ``T`` samples. With
``W`` the filter's width and ``L = T + 2W``, the scratch holds both phases of
the upsampled signal at ``q = -W .. T + W - 1``, stored at ``q + W``:

    up_w   (2, 2W + 1): phase e reads x[clamp(q + t - W)] with tap t
    down_w (2, 2W + 1): y[n] reads phase e at q = n - W + i with tap i
    v, a   the upsampled signal before and after the activation
"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _upsample(x_row, up_ptr, q, T, mask, TAPS: tl.constexpr, W: tl.constexpr):
    """Both phases of the upsampled signal at ``q``, from one read per tap."""
    v0 = tl.zeros(q.shape, dtype=tl.float32)
    v1 = tl.zeros(q.shape, dtype=tl.float32)
    for t in tl.static_range(TAPS):
        idx = tl.minimum(tl.maximum(q + t - W, 0), T - 1)
        xv = tl.load(x_row + idx, mask=mask, other=0.0).to(tl.float32)
        v0 += tl.load(up_ptr + t) * xv
        v1 += tl.load(up_ptr + TAPS + t) * xv
    return v0, v1


@triton.jit
def _fwd_up(x_ptr, up_ptr, a_ptr, T, slope, TAPS: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    """``x`` -> the activated upsampled signal ``a``, (rows, 2, L)."""
    row = tl.program_id(0)
    L = T + 2 * W
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = p < L
    v0, v1 = _upsample(x_ptr + row.to(tl.int64) * T, up_ptr, p - W, T, mask, TAPS, W)
    a_row = a_ptr + row.to(tl.int64) * 2 * L
    tl.store(a_row + p, tl.where(v0 > 0, v0, v0 * slope), mask=mask)
    tl.store(a_row + L + p, tl.where(v1 > 0, v1, v1 * slope), mask=mask)


@triton.jit
def _fwd_down(a_ptr, down_ptr, y_ptr, T, TAPS: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    """``y[n] = sum_i down_w[e, i] * a[e, n + i]``."""
    row = tl.program_id(0)
    L = T + 2 * W
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < T
    a_row = a_ptr + row.to(tl.int64) * 2 * L
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(TAPS):
        acc += tl.load(down_ptr + i) * tl.load(a_row + n + i, mask=mask, other=0.0)
        acc += tl.load(down_ptr + TAPS + i) * tl.load(a_row + L + n + i, mask=mask, other=0.0)
    tl.store(y_ptr + row.to(tl.int64) * T + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_act(gy_ptr, x_ptr, up_ptr, down_ptr, gv_ptr, T, slope, TAPS: tl.constexpr, W: tl.constexpr,
             BLOCK: tl.constexpr):
    """Backward through the lowpass and the activation: ``gy`` -> ``gv``, the
    gradient at the upsampled signal before the activation, (rows, 2, L)."""
    row = tl.program_id(0)
    L = T + 2 * W
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = p < L
    # Recomputed rather than saved by the forward.
    v0, v1 = _upsample(x_ptr + row.to(tl.int64) * T, up_ptr, p - W, T, mask, TAPS, W)
    gy_row = gy_ptr + row.to(tl.int64) * T
    g0 = tl.zeros((BLOCK,), dtype=tl.float32)
    g1 = tl.zeros((BLOCK,), dtype=tl.float32)
    # The lowpass's adjoint: a[e, p] reached y[p - i] through tap i.
    for i in tl.static_range(TAPS):
        n = p - i
        g = tl.load(gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
        g0 += tl.load(down_ptr + i) * g
        g1 += tl.load(down_ptr + TAPS + i) * g
    gv_row = gv_ptr + row.to(tl.int64) * 2 * L
    tl.store(gv_row + p, tl.where(v0 > 0, g0, g0 * slope), mask=mask)
    tl.store(gv_row + L + p, tl.where(v1 > 0, g1, g1 * slope), mask=mask)


@triton.jit
def _bwd_up(gv_ptr, up_ptr, gx_ptr, T, TAPS: tl.constexpr, W: tl.constexpr, EDGE: tl.constexpr,
            BLOCK: tl.constexpr):
    """Backward through the upsampler: ``gv`` -> ``gx``."""
    row = tl.program_id(0)
    L = T + 2 * W
    start = tl.program_id(1) * BLOCK
    m = start + tl.arange(0, BLOCK)
    mask = m < T
    gv_row = gv_ptr + row.to(tl.int64) * 2 * L
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Stored position p read x[m] through tap t when p + t - 2W = m.
    for t in tl.static_range(TAPS):
        p = m - t + 2 * W
        acc += tl.load(up_ptr + t) * tl.load(gv_row + p, mask=mask, other=0.0)
        acc += tl.load(up_ptr + TAPS + t) * tl.load(gv_row + L + p, mask=mask, other=0.0)
    # The clamp's adjoint: reads left of the signal took x[0], right of it x[T - 1].
    e = tl.arange(0, EDGE)
    if start == 0:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for t in tl.static_range(TAPS):
            inside = e < 2 * W - t
            edge += tl.load(up_ptr + t) * tl.load(gv_row + e, mask=inside, other=0.0)
            edge += tl.load(up_ptr + TAPS + t) * tl.load(gv_row + L + e, mask=inside, other=0.0)
        acc += tl.where(m == 0, tl.sum(edge, 0), 0.0)
    if start + BLOCK >= T:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for t in tl.static_range(TAPS):
            right = T + 2 * W - t + e
            inside = e < t
            edge += tl.load(up_ptr + t) * tl.load(gv_row + right, mask=inside, other=0.0)
            edge += tl.load(up_ptr + TAPS + t) * tl.load(gv_row + L + right, mask=inside, other=0.0)
        acc += tl.where(m == T - 1, tl.sum(edge, 0), 0.0)
    tl.store(gx_ptr + row.to(tl.int64) * T + m, acc.to(gx_ptr.dtype.element_ty), mask=mask)


def _grid(rows, length):
    return (rows, triton.cdiv(length, BLOCK))


class FusedAntiAliasedLeakyReLU(torch.autograd.Function):
    """``(x, up_w, down_w, slope) -> y``, the same as ``AntiAliasedLeakyReLU``'s
    PyTorch path; ``up_w``/``down_w`` are its polyphase weights, FP32."""

    @staticmethod
    def forward(ctx, x, up_w, down_w, slope):
        x = x.contiguous()
        batch, channels, length = x.shape
        taps = up_w.shape[1]
        width = (taps - 1) // 2
        rows, padded = batch * channels, length + 2 * width
        a = torch.empty(rows, 2, padded, device=x.device, dtype=torch.float32)
        _fwd_up[_grid(rows, padded)](x, up_w, a, length, slope, TAPS=taps, W=width, BLOCK=BLOCK)
        y = torch.empty_like(x)
        _fwd_down[_grid(rows, length)](a, down_w, y, length, TAPS=taps, W=width, BLOCK=BLOCK)
        ctx.save_for_backward(x, up_w, down_w)
        ctx.slope = slope
        return y

    @staticmethod
    def backward(ctx, gy):
        x, up_w, down_w = ctx.saved_tensors
        batch, channels, length = x.shape
        taps = up_w.shape[1]
        width = (taps - 1) // 2
        rows, padded = batch * channels, length + 2 * width
        gv = torch.empty(rows, 2, padded, device=x.device, dtype=torch.float32)
        _bwd_act[_grid(rows, padded)](gy.contiguous(), x, up_w, down_w, gv, length, ctx.slope, TAPS=taps,
                                      W=width, BLOCK=BLOCK)
        gx = torch.empty_like(x)
        _bwd_up[_grid(rows, length)](gv, up_w, gx, length, TAPS=taps, W=width,
                                     EDGE=triton.next_power_of_2(2 * width), BLOCK=BLOCK)
        return gx, None, None, None
