import math
import torch
from typing import List, Optional

def strip_parametrizations(module: torch.nn.Module):
    """
    Fold every parametrization (weight norm, spectral norm, ...) into the raw
    weights. Training keeps `g` and `v` separate and recomputes `g * v / ||v||`
    on every forward; at fixed inference weights that recompute is pure
    overhead, so folding it once lets torch.compile see plain convolutions.
    Idempotent: already-stripped modules are skipped.
    """
    removed = 0
    for submodule in module.modules():
        parametrizations = getattr(submodule, "parametrizations", None)
        if not parametrizations:
            continue
        for name in list(parametrizations.keys()):
            torch.nn.utils.parametrize.remove_parametrizations(
                submodule, name, leave_parametrized=True
            )
            removed += 1
    return removed


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def kl_divergence(m_p, logs_p, m_q, logs_q):
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def slice_segments_old(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 2
):
    if dim == 2:
        ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        if dim == 2:
            ret[i] = x[i, idx_str:idx_end]
        else:
            ret[i] = x[i, :, idx_str:idx_end]

    return ret


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 2
):
    """Vectorized equivalent of slice_segments_old."""
    b = x.size(0)
    idx = ids_str[:, None] + torch.arange(segment_size, device=x.device)
    rows = torch.arange(b, device=x.device)
    if dim == 2:
        return x[rows[:, None], idx]
    return x[
        rows[:, None, None],
        torch.arange(x.size(1), device=x.device)[None, :, None],
        idx[:, None, :],
    ]


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size, dim=3), ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = torch.nn.functional.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels: int):
    """``n_channels`` is a plain ``int``, not a one-element tensor.

    VITS passes a ``torch.IntTensor([hidden_channels])`` here and reads
    ``n_channels[0]`` back out to slice with, which this fork inherited.  That
    is an allocation plus a ``select`` plus an ``aten._local_scalar_dense`` per
    WaveNet layer -- 28 scalar extractions per generator forward, measured, for
    a number that was a Python ``int`` at the call site all along.  Under
    ``torch.compile`` it is worse than wasteful: reading a scalar out of a
    tensor breaks the graph, once per WaveNet forward.
    """

    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape: List[List[int]]):
    return torch.tensor(pad_shape).flip(0).reshape(-1).int().tolist()


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False):
    """Norm over the norms of the individual tensors, as if concatenated into one vector."""
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    tensors = [t for t in tensors if t is not None]

    if len(tensors) == 0:
        return torch.tensor(0.0)

    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(t.abs().max() for t in tensors)
    else:
        tensor_norms = [t.norm(norm_type) for t in tensors]
        total_norm = torch.norm(torch.stack(tensor_norms), norm_type)

    if error_if_nonfinite and (torch.isnan(total_norm) or torch.isinf(total_norm)):
        raise RuntimeError("The total norm is non-finite (NaN or Inf).")

    return total_norm

def cache_scope():
    """Context for building a tensor that outlives the call that built it.

    The holdout evaluation runs under ``torch.inference_mode()``, and any tensor
    first materialised there is an *inference tensor* -- it carries no version
    counter, so autograd refuses to save it for backward.  A per-call value is
    fine, but a value stashed on the module (a filter kernel, an attention mask)
    survives into the next training step and poisons every step after it with
    ``RuntimeError: Inference tensors cannot be saved for backward``.  The fix
    has to be at the point the cache is *filled*, not where it is used, so the
    stored tensor is a normal one whichever mode happened to see the first call.
    """

    return torch.inference_mode(False)
