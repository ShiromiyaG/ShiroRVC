"""Triton kernels for ChouwaGAN's ``AntiAliasedLeakyReLU``: the polyphase
upsampler by ``R``, the leaky ReLU and the stride-``R`` lowpass, with the
activated upsampled signal in one FP32 scratch buffer and recomputed from the
input in backward, so autograd keeps only the input.

Every kernel sees the tensors as ``B * C`` rows of ``T`` samples. With
``W`` the filter's width and ``L = T + 2W``, the scratch holds the ``R``
phases of the upsampled signal at ``q = -W .. T + W - 1``, stored at ``q + W``:

    up_w   (R, 2W + 1): phase e reads x[clamp(q + t - W)] with tap t
    down_w (R, 2W + 1): y[n] reads phase e at q = n - W + i with tap i
    v, a   the upsampled signal before and after the activation
"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _upsample(x_row, up_ptr, phase, q, T, mask, TAPS: tl.constexpr, W: tl.constexpr):
    """Phase ``phase`` of the upsampled signal at ``q``."""
    v = tl.zeros(q.shape, dtype=tl.float32)
    for t in tl.static_range(TAPS):
        idx = tl.minimum(tl.maximum(q + t - W, 0), T - 1)
        v += tl.load(up_ptr + phase * TAPS + t) * tl.load(x_row + idx, mask=mask, other=0.0).to(tl.float32)
    return v


@triton.jit
def _fwd_up(x_ptr, up_ptr, a_ptr, T, slope, R: tl.constexpr, TAPS: tl.constexpr, W: tl.constexpr,
            BLOCK: tl.constexpr):
    """``x`` -> the activated upsampled signal ``a``, (rows, R, L)."""
    row = tl.program_id(0)
    L = T + 2 * W
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = p < L
    x_row = x_ptr + row.to(tl.int64) * T
    a_row = a_ptr + row.to(tl.int64) * R * L
    for phase in tl.static_range(R):
        v = _upsample(x_row, up_ptr, phase, p - W, T, mask, TAPS, W)
        tl.store(a_row + phase * L + p, tl.where(v > 0, v, v * slope), mask=mask)


@triton.jit
def _fwd_down(a_ptr, down_ptr, y_ptr, T, R: tl.constexpr, TAPS: tl.constexpr, W: tl.constexpr,
              BLOCK: tl.constexpr):
    """``y[n] = sum_e sum_i down_w[e, i] * a[e, n + i]``."""
    row = tl.program_id(0)
    L = T + 2 * W
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n < T
    a_row = a_ptr + row.to(tl.int64) * R * L
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for phase in tl.static_range(R):
        for i in tl.static_range(TAPS):
            acc += tl.load(down_ptr + phase * TAPS + i) * tl.load(a_row + phase * L + n + i, mask=mask, other=0.0)
    tl.store(y_ptr + row.to(tl.int64) * T + n, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_act(gy_ptr, x_ptr, up_ptr, down_ptr, gv_ptr, T, slope, R: tl.constexpr, TAPS: tl.constexpr,
             W: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the lowpass and the activation: ``gy`` -> ``gv``, the
    gradient at the upsampled signal before the activation, (rows, R, L)."""
    row = tl.program_id(0)
    L = T + 2 * W
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = p < L
    x_row = x_ptr + row.to(tl.int64) * T
    gy_row = gy_ptr + row.to(tl.int64) * T
    gv_row = gv_ptr + row.to(tl.int64) * R * L
    for phase in tl.static_range(R):
        # Recomputed rather than saved by the forward.
        v = _upsample(x_row, up_ptr, phase, p - W, T, mask, TAPS, W)
        g = tl.zeros((BLOCK,), dtype=tl.float32)
        # The lowpass's adjoint: a[e, p] reached y[p - i] through tap i.
        for i in tl.static_range(TAPS):
            n = p - i
            g += tl.load(down_ptr + phase * TAPS + i) * tl.load(
                gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0
            ).to(tl.float32)
        tl.store(gv_row + phase * L + p, tl.where(v > 0, g, g * slope), mask=mask)


@triton.jit
def _bwd_up(gv_ptr, up_ptr, gx_ptr, T, R: tl.constexpr, TAPS: tl.constexpr, W: tl.constexpr,
            EDGE: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the upsampler: ``gv`` -> ``gx``."""
    row = tl.program_id(0)
    L = T + 2 * W
    start = tl.program_id(1) * BLOCK
    m = start + tl.arange(0, BLOCK)
    mask = m < T
    gv_row = gv_ptr + row.to(tl.int64) * R * L
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Stored position p read x[m] through tap t when p + t - 2W = m.
    for phase in tl.static_range(R):
        for t in tl.static_range(TAPS):
            p = m - t + 2 * W
            acc += tl.load(up_ptr + phase * TAPS + t) * tl.load(gv_row + phase * L + p, mask=mask, other=0.0)
    # The clamp's adjoint: reads left of the signal took x[0], right of it x[T - 1].
    e = tl.arange(0, EDGE)
    if start == 0:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for phase in tl.static_range(R):
            for t in tl.static_range(TAPS):
                inside = e < 2 * W - t
                edge += tl.load(up_ptr + phase * TAPS + t) * tl.load(gv_row + phase * L + e, mask=inside, other=0.0)
        acc += tl.where(m == 0, tl.sum(edge, 0), 0.0)
    if start + BLOCK >= T:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for phase in tl.static_range(R):
            for t in tl.static_range(TAPS):
                right = T + 2 * W - t + e
                inside = e < t
                edge += tl.load(up_ptr + phase * TAPS + t) * tl.load(gv_row + phase * L + right, mask=inside,
                                                                     other=0.0)
        acc += tl.where(m == T - 1, tl.sum(edge, 0), 0.0)
    tl.store(gx_ptr + row.to(tl.int64) * T + m, acc.to(gx_ptr.dtype.element_ty), mask=mask)


def _grid(rows, length):
    return (rows, triton.cdiv(length, BLOCK))


class FusedAntiAliasedLeakyReLU(torch.autograd.Function):
    """``(x, up_w, down_w, slope) -> y``, the same as ``AntiAliasedLeakyReLU``'s
    PyTorch path; ``up_w``/``down_w`` are its polyphase weights, FP32, one
    row per phase."""

    @staticmethod
    def forward(ctx, x, up_w, down_w, slope):
        x = x.contiguous()
        batch, channels, length = x.shape
        ratio, taps = up_w.shape
        width = (taps - 1) // 2
        rows, padded = batch * channels, length + 2 * width
        a = torch.empty(rows, ratio, padded, device=x.device, dtype=torch.float32)
        _fwd_up[_grid(rows, padded)](x, up_w, a, length, slope, R=ratio, TAPS=taps, W=width, BLOCK=BLOCK)
        y = torch.empty_like(x)
        _fwd_down[_grid(rows, length)](a, down_w, y, length, R=ratio, TAPS=taps, W=width, BLOCK=BLOCK)
        ctx.save_for_backward(x, up_w, down_w)
        ctx.slope = slope
        return y

    @staticmethod
    def backward(ctx, gy):
        x, up_w, down_w = ctx.saved_tensors
        batch, channels, length = x.shape
        ratio, taps = up_w.shape
        width = (taps - 1) // 2
        rows, padded = batch * channels, length + 2 * width
        gv = torch.empty(rows, ratio, padded, device=x.device, dtype=torch.float32)
        _bwd_act[_grid(rows, padded)](gy.contiguous(), x, up_w, down_w, gv, length, ctx.slope, R=ratio, TAPS=taps,
                                      W=width, BLOCK=BLOCK)
        gx = torch.empty_like(x)
        _bwd_up[_grid(rows, length)](gv, up_w, gx, length, R=ratio, TAPS=taps, W=width,
                                     EDGE=triton.next_power_of_2(2 * width), BLOCK=BLOCK)
        return gx, None, None, None
