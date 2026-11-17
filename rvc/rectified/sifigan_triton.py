"""Triton kernels for SiFi-GAN's sinc resamplers: a FIR filter that keeps
every ``ratio``-th output (decimation) and its adjoint, which makes ``ratio``
outputs per input sample (polyphase interpolation). Each is the other's
backward, so autograd keeps nothing but the shapes.

Every kernel sees the tensors as rows of samples, ``x`` zero outside its row
and the FP32 kernel ``h`` zero outside its ``K`` taps:

    decimate     y[n] = sum_k h[k] x[n * ratio + k + offset]
    interpolate  y[j] = sum_n h[j + offset - n * ratio] x[n]    (offset >= 0)

With the same ``h``, interpolating at ``-offset`` is decimating's adjoint.
"""

import torch
import triton
import triton.language as tl

BLOCK = 128
BLOCK_K = 64


@triton.jit
def _decimate(x_ptr, h_ptr, y_ptr, T_in, T_out, offset, RATIO: tl.constexpr, K: tl.constexpr,
              BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x_row = x_ptr + row.to(tl.int64) * T_in
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in tl.static_range(0, K, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        h = tl.load(h_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        index = n[:, None] * RATIO + k[None, :] + offset
        mask = (n[:, None] < T_out) & (k[None, :] < K) & (index >= 0) & (index < T_in)
        xv = tl.load(x_row + index, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xv * h[None, :], axis=1)
    tl.store(y_ptr + row.to(tl.int64) * T_out + n, acc.to(y_ptr.dtype.element_ty), mask=n < T_out)


@triton.jit
def _interpolate(x_ptr, h_ptr, y_ptr, T_in, T_out, offset, RATIO: tl.constexpr, K: tl.constexpr,
                 TAPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = j + offset
    i = tl.arange(0, TAPS)
    # Output p reads the inputs n = p // ratio - i through tap p % ratio + i * ratio.
    k = (p % RATIO)[:, None] + i[None, :] * RATIO
    n = (p // RATIO)[:, None] - i[None, :]
    mask = (j[:, None] < T_out) & (k < K) & (n >= 0) & (n < T_in)
    h = tl.load(h_ptr + k, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptr + row.to(tl.int64) * T_in + n, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(h * xv, axis=1)
    tl.store(y_ptr + row.to(tl.int64) * T_out + j, acc.to(y_ptr.dtype.element_ty), mask=j < T_out)


def decimate(x, h, ratio, offset, length):
    """``x`` [rows, T] -> [rows, length]; ``h`` [K] FP32."""
    x = x.contiguous()
    y = torch.empty(x.shape[0], length, device=x.device, dtype=x.dtype)
    _decimate[(x.shape[0], triton.cdiv(length, BLOCK))](
        x, h, y, x.shape[1], length, offset, RATIO=ratio, K=h.numel(), BLOCK=BLOCK, BLOCK_K=BLOCK_K,
    )
    return y


def interpolate(x, h, ratio, offset, length):
    """``x`` [rows, T] -> [rows, length]; ``h`` [K] FP32, ``offset`` >= 0."""
    x = x.contiguous()
    y = torch.empty(x.shape[0], length, device=x.device, dtype=x.dtype)
    taps = triton.next_power_of_2(triton.cdiv(h.numel(), ratio))
    _interpolate[(x.shape[0], triton.cdiv(length, BLOCK))](
        x, h, y, x.shape[1], length, offset, RATIO=ratio, K=h.numel(), TAPS=taps, BLOCK=BLOCK,
    )
    return y


class Decimate(torch.autograd.Function):
    """``(x, h, ratio, length) -> decimate(x, h, ratio, 0, length)``."""

    @staticmethod
    def forward(ctx, x, h, ratio, length):
        ctx.save_for_backward(h)
        ctx.ratio, ctx.in_length = ratio, x.shape[1]
        return decimate(x, h, ratio, 0, length)

    @staticmethod
    def backward(ctx, gy):
        (h,) = ctx.saved_tensors
        return interpolate(gy, h, ctx.ratio, 0, ctx.in_length), None, None, None


class Interpolate(torch.autograd.Function):
    """``(x, h, ratio, offset, length) -> interpolate(x, h, ratio, offset, length)``."""

    @staticmethod
    def forward(ctx, x, h, ratio, offset, length):
        ctx.save_for_backward(h)
        ctx.ratio, ctx.offset, ctx.in_length = ratio, offset, x.shape[1]
        return interpolate(x, h, ratio, offset, length)

    @staticmethod
    def backward(ctx, gy):
        (h,) = ctx.saved_tensors
        return decimate(gy, h, ctx.ratio, -ctx.offset, ctx.in_length), None, None, None, None
