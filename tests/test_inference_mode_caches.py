"""Module-level tensor caches must survive a pass under ``inference_mode``.

The holdout evaluation runs the generator inside ``torch.inference_mode()``.
Anything a module stashes on ``self`` there -- a resampling kernel expanded to
the channel count, an attention block mask -- is an *inference tensor*, which
autograd refuses to save for backward.  The eval happens before the training
step, so the failure is not in the eval at all: every later step dies with
``Inference tensors cannot be saved for backward``, and the decoder's
``torch.compile`` fails with the same message first.  A unit test that only
ever calls these modules in training mode cannot see it, so the order below --
inference pass first, backward second -- is the whole point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.lib.algorithm.attentions import MultiHeadAttention  # noqa: E402
from rvc.lib.algorithm.resampling import (  # noqa: E402
    AntiAliasedUpsample1d,
    FixedLowPass1d,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AntiAliasedUpsample1d(2, 4, 0.95),
        lambda: FixedLowPass1d(2, width=4, rolloff=0.95, stride=2),
    ],
)
def test_resampler_kernel_cache_is_not_an_inference_tensor(factory):
    module = factory()
    warm = torch.randn(2, 3, 64)
    with torch.inference_mode():
        module(warm)

    assert not torch.is_inference(module._kernel_cache)

    x = torch.randn(2, 3, 64, requires_grad=True)
    module(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_attention_block_mask_cache_is_not_an_inference_tensor():
    module = MultiHeadAttention(16, 16, 4, block_length=3)
    warm = torch.randn(2, 16, 20)
    with torch.inference_mode():
        module(warm, warm)

    cache = getattr(module, "_block_mask_cache", None)
    if cache is not None:
        assert not torch.is_inference(cache)

    x = torch.randn(2, 16, 20, requires_grad=True)
    module(x, x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
