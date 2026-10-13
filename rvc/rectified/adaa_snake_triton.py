"""Triton kernels for Pupu-Vocoder's ``Activation1d``: ``snake_alias_triton``'s
fused op with SnakeBeta in its anti-derivative anti-aliased form.

The upsampler, the lowpass and the names are ``snake_alias_triton``'s.  The
activation reads two neighbours of the upsampled signal ``w`` (``w[2k]`` is
``v0[k]``, ``w[2k + 1]`` is ``v1[k]``, and ``w[-1]`` is zero):

    u[m] = A(w[m], w[m - 1])
    A(a, b) = (a + b) / 2 + (1 - cos(alpha (a + b)) sinc(alpha (a - b))) / (2 beta)

so a program also computes ``v1[k - 1]``, and in backward ``v0[k + 1]`` with
the gradient that reaches it.  Everything is computed in FP32.
"""

import torch
import triton
import triton.language as tl

from rvc.lib.algorithm.generators.snake_triton import BLOCK, _bwd_up, _grid, _preact
from rvc.rectified.snake_alias_triton import _fwd_down


@triton.jit
def _sinc(z):
    """``sin(z) / z`` and its derivative; by their series near zero, where the
    derivative's closed form cancels."""
    small = tl.abs(z) < 0.05
    zs = tl.where(small, 1.0, z)
    z2 = z * z
    value = tl.where(small, 1.0 - z2 / 6.0 + z2 * z2 / 120.0, tl.sin(zs) / zs)
    slope = tl.where(small, -z / 3.0 + z * z2 / 30.0, (zs * tl.cos(zs) - tl.sin(zs)) / (zs * zs))
    return value, slope


@triton.jit
def _adaa(a, b, alpha, ib):
    s = a + b
    value, slope = _sinc(alpha * (a - b))
    return 0.5 * s + 0.5 * ib * (1.0 - tl.cos(alpha * s) * value)


@triton.jit
def _adaa_bwd(a, b, alpha, ib):
    """``dA/da``, ``dA/db`` and ``dA/d log alpha``, ``dA/d log beta``."""
    zs = alpha * (a + b)
    zd = alpha * (a - b)
    value, slope = _sinc(zd)
    cs = tl.cos(zs)
    sn = tl.sin(zs)
    half = 0.5 * ib
    # Along the sum and along the difference of the two samples.
    ds = 0.5 + half * alpha * sn * value
    dd = -half * alpha * cs * slope
    dla = half * (zs * sn * value - cs * zd * slope)
    dlb = -half * (1.0 - cs * value)
    return ds + dd, ds - dd, dla, dlb


@triton.jit
def _fwd_up(x_ptr, r_ptr, w_ptr, la_ptr, lb_ptr, u_ptr, xn_ptr, T, C, PAD_L,
            TAPS: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """Upsample and activate: ``x`` (plus ``r``) -> the scratch ``u``.
    With ``HAS_RES`` also writes ``x + r`` to ``xn``."""
    row = tl.program_id(0)
    ch = row % C
    alpha = tl.exp(tl.load(la_ptr + ch).to(tl.float32))
    ib = tl.exp(-tl.load(lb_ptr + ch).to(tl.float32))
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = k < T
    x_row = x_ptr + row.to(tl.int64) * T
    r_row = r_ptr + row.to(tl.int64) * T
    v0, v1 = _preact(x_row, r_row, w_ptr, k, T, PAD_L, mask, TAPS, HAS_RES)
    q0, p1 = _preact(x_row, r_row, w_ptr, k - 1, T, PAD_L, mask, TAPS, HAS_RES)
    p1 = tl.where(k >= 1, p1, 0.0)
    if HAS_RES:
        xn = (tl.load(x_row + k, mask=mask, other=0.0).to(tl.float32)
              + tl.load(r_row + k, mask=mask, other=0.0).to(tl.float32))
        tl.store(xn_ptr + row.to(tl.int64) * T + k, xn.to(xn_ptr.dtype.element_ty), mask=mask)
    u_row = u_ptr + row.to(tl.int64) * 2 * T
    tl.store(u_row + k, _adaa(v0, p1, alpha, ib), mask=mask)
    tl.store(u_row + T + k, _adaa(v1, v0, alpha, ib), mask=mask)


@triton.jit
def _lowpass_adjoint(gy_row, d_ptr, k, T, mask, DPAD: tl.constexpr, DTAPS: tl.constexpr):
    """The lowpass's adjoint at upsampled samples 2k and 2k + 1, without the
    replicate padding's share: tap j of output n read sample 2n + j - DPAD, so
    each tap reaches one phase. Offsets are shifted by DTAPS to divide a
    non-negative number."""
    g0 = tl.zeros(k.shape, dtype=tl.float32)
    g1 = tl.zeros(k.shape, dtype=tl.float32)
    for j in tl.static_range(DTAPS):
        if (DPAD + j) % 2 == 0:
            n = k + (DPAD - j + 2 * DTAPS) // 2 - DTAPS
            g0 += tl.load(d_ptr + j) * tl.load(
                gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
        else:
            n = k + (DPAD + 1 - j + 2 * DTAPS) // 2 - DTAPS
            g1 += tl.load(d_ptr + j) * tl.load(
                gy_row + n, mask=mask & (n >= 0) & (n < T), other=0.0).to(tl.float32)
    return g0, g1


@triton.jit
def _bwd_act(gy_ptr, x_ptr, w_ptr, d_ptr, la_ptr, lb_ptr, gv_ptr, dla_ptr, dlb_ptr,
             T, C, PAD_L, TAPS: tl.constexpr, DPAD: tl.constexpr, DTAPS: tl.constexpr,
             EDGE: tl.constexpr, BLOCK: tl.constexpr):
    """Backward through the lowpass and the activation: ``gy`` -> ``gv``, the
    gradient at the upsampled signal, (rows, 2, T), and ``dla``/``dlb``."""
    row = tl.program_id(0)
    ch = row % C
    alpha = tl.exp(tl.load(la_ptr + ch).to(tl.float32))
    ib = tl.exp(-tl.load(lb_ptr + ch).to(tl.float32))
    start = tl.program_id(1) * BLOCK
    k = (start + tl.arange(0, BLOCK))[None, :]
    mask = k < T
    x_row = x_ptr + row.to(tl.int64) * T
    # Recomputed rather than saved by the forward.
    v0, v1 = _preact(x_row, x_row, w_ptr, k, T, PAD_L, mask, TAPS, False)
    q0, p1 = _preact(x_row, x_row, w_ptr, k - 1, T, PAD_L, mask, TAPS, False)
    p1 = tl.where(k >= 1, p1, 0.0)
    n0, q1 = _preact(x_row, x_row, w_ptr, k + 1, T, PAD_L, mask, TAPS, False)
    gy_row = gy_ptr + row.to(tl.int64) * T
    g0, g1 = _lowpass_adjoint(gy_row, d_ptr, k, T, mask, DPAD, DTAPS)
    # Replicate padding: every read left of the signal landed on sample 0,
    # every one right of it on sample 2T - 1.
    p = tl.arange(0, EDGE)
    if start == 0:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for j in tl.static_range(DTAPS):
            edge += tl.load(d_ptr + j) * tl.load(
                gy_row + p, mask=(2 * p + j < DPAD) & (p < T), other=0.0).to(tl.float32)
        g0 += tl.where(k == 0, tl.sum(edge, 0), 0.0)
    if start + BLOCK >= T:
        edge = tl.zeros((EDGE,), dtype=tl.float32)
        for j in tl.static_range(DTAPS):
            edge += tl.load(d_ptr + j) * tl.load(
                gy_row + T - 1 - p, mask=(2 * p + DPAD + 1 < j) & (p < T), other=0.0).to(tl.float32)
        g1 += tl.where(k == T - 1, tl.sum(edge, 0), 0.0)
    # The gradient at sample 2k + 2, which also read sample 2k + 1; it is
    # never sample 0, so the padding has no share in it.
    has_next = mask & (k + 1 < T)
    gn, h1 = _lowpass_adjoint(gy_row, d_ptr, k + 1, T, has_next, DPAD, DTAPS)
    gn = tl.where(has_next, gn, 0.0)

    a0, b0, la0, lb0 = _adaa_bwd(v0, p1, alpha, ib)
    a1, b1, la1, lb1 = _adaa_bwd(v1, v0, alpha, ib)
    an, bn, lan, lbn = _adaa_bwd(n0, v1, alpha, ib)
    gv_row = gv_ptr + row.to(tl.int64) * 2 * T
    tl.store(gv_row + k, g0 * a0 + g1 * b1, mask=mask)
    tl.store(gv_row + T + k, g1 * a1 + gn * bn, mask=mask)
    dla = tl.where(mask, g0 * la0 + g1 * la1, 0.0)
    dlb = tl.where(mask, g0 * lb0 + g1 * lb1, 0.0)
    one_row = tl.full((1,), True, tl.int1)
    tl.atomic_add(dla_ptr + row + tl.arange(0, 1), tl.sum(dla, 1), mask=one_row)
    tl.atomic_add(dlb_ptr + row + tl.arange(0, 1), tl.sum(dlb, 1), mask=one_row)


def _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
    """``(y, x + r)``, or ``(y, None)`` without a residual ``r``."""
    B, C, T = x.shape
    u = torch.empty(B * C, 2, T, device=x.device, dtype=torch.float32)
    xn = None if r is None else torch.empty_like(x, dtype=torch.promote_types(x.dtype, r.dtype))
    # Pointers the kernel does not use without a residual still need a tensor.
    _fwd_up[_grid(B * C, T)](x, x if r is None else r, up_w, log_alpha, log_beta, u,
                             x if xn is None else xn, T, C, pad[0], TAPS=up_w.shape[1],
                             HAS_RES=r is not None, BLOCK=BLOCK)
    y = torch.empty(B, C, T, device=x.device, dtype=dtype)
    _fwd_down[_grid(B * C, T)](u, down_w, y, T, DPAD=down_pad, DTAPS=down_w.shape[0], BLOCK=BLOCK)
    return y, xn


def _backward(gy, x, gxn, log_alpha, log_beta, up_w, down_w, pad, down_pad, x_dtype, r_dtype):
    """``x`` is what the activation read: the input, or ``x + r`` when fused.
    Returns ``(gx, gr, dla, dlb)``; ``gr`` is None without a residual."""
    B, C, T = x.shape
    taps, down_taps = up_w.shape[1], down_w.shape[0]
    gv = torch.empty(B * C, 2, T, device=x.device, dtype=torch.float32)
    dla = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    dlb = torch.zeros(B * C, device=x.device, dtype=torch.float32)
    _bwd_act[_grid(B * C, T)](gy.contiguous(), x, up_w, down_w, log_alpha, log_beta, gv, dla, dlb,
                              T, C, pad[0], TAPS=taps, DPAD=down_pad, DTAPS=down_taps,
                              EDGE=triton.next_power_of_2(down_taps), BLOCK=BLOCK)
    # Upsampler, and the residual's gradient when there is one.
    res = r_dtype is not None
    gx = torch.empty(B, C, T, device=x.device, dtype=x_dtype)
    gr = torch.empty(B, C, T, device=x.device, dtype=r_dtype) if res else None
    _bwd_up[_grid(B * C, T)](gv, up_w, gxn.contiguous() if res else gv, gx,
                             gr if res else gx, T, pad[0], pad[1], TAPS=taps, HAS_RES=res,
                             EDGE=triton.next_power_of_2(max(pad[0], pad[1], 1)), BLOCK=BLOCK)
    # Per-row sums -> per-channel gradients.
    return gx, gr, dla.view(B, C).sum(0), dlb.view(B, C).sum(0)


class FusedADAASnake(torch.autograd.Function):
    """``(x, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype) -> y`` in ``dtype``.

    ``up_w`` is the upsampler's polyphase weights, (2, taps), and ``pad`` its
    replicate padding; ``down_w`` the lowpass and ``down_pad`` its left padding.
    """

    @staticmethod
    def forward(ctx, x, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
        x = x.contiguous()
        y, _ = _forward(x, None, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype)
        ctx.save_for_backward(x, log_alpha, log_beta, up_w, down_w)
        ctx.pads = (pad, down_pad)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, _, dla, dlb = _backward(gy, x, None, log_alpha, log_beta, up_w, down_w, *ctx.pads,
                                    x.dtype, None)
        return gx, dla, dlb, None, None, None, None, None


class FusedResidualADAASnake(torch.autograd.Function):
    """``(x, r, ...) -> (y, x + r)``: ``FusedADAASnake`` of ``x + r``, with the
    sum computed in the kernel and returned for the residual stream."""

    @staticmethod
    def forward(ctx, x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype):
        x, r = x.contiguous(), r.contiguous()
        y, xn = _forward(x, r, log_alpha, log_beta, up_w, down_w, pad, down_pad, dtype)
        ctx.save_for_backward(xn, log_alpha, log_beta, up_w, down_w)
        ctx.pads = (pad, down_pad)
        ctx.dtypes = (x.dtype, r.dtype)
        return y, xn

    @staticmethod
    def backward(ctx, gy, gxn):
        xn, log_alpha, log_beta, up_w, down_w = ctx.saved_tensors
        gx, gr, dla, dlb = _backward(gy, xn, gxn, log_alpha, log_beta, up_w, down_w, *ctx.pads,
                                     *ctx.dtypes)
        return gx, gr, dla, dlb, None, None, None, None, None
