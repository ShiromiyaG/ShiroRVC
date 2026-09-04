"""Compile options that a config can ask for, and the code that has to answer.

``train.py`` reaches every one of these through ``getattr(model, name, None)``
and reports "not supported" when the lookup misses.  That is the right shape for
a vocoder that genuinely cannot be compiled, and it is also how
``compile_discriminator`` came to be a dead key: the only class that ever
defined ``enable_compile`` was the ChouwaGAN discriminator, and it left with the
rest of ChouwaGAN.  Setting the key logged "ignored" and the run carried on
eager, which reads exactly like the option working.

Same silence had already happened twice over -- SAN's ``supports_san`` and R1's
``uses_branchwise_r1`` are both still consulted and nothing defines either.  So
these tests are about the *wiring*, not the speedup: a shipped ``compile_*`` key
must reach a method that exists.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

CONFIG = ROOT / "rvc" / "configs" / "refinegan2" / "32000.json"

#: Each shipped ``compile_*`` key, and the method the run looks up when it is on.
COMPILE_KEYS = {
    "compile_discriminator": ("discriminator", "enable_compile"),
    "compile_frontend": ("generator", "enable_frontend_compile"),
}


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _discriminator():
    model = _config()["model"]
    return MPD_MSD_Combined(
        model["use_spectral_norm"],
        version=model["d_version"],
        sample_rate=32000,
        use_univhd=bool(model.get("d_use_univhd", False)),
        use_fast_mpd=bool(model.get("d_use_fast_mpd", False)),
    )


def _generator():
    config = _config()
    model, data = config["model"], config["data"]
    return Synthesizer(
        data["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // data["hop_length"],
        **model,
        use_f0=True,
        sr=data["sample_rate"],
        vocoder="refinegan2",
    )


def _generator_inputs(net, batch=1, frames=40):
    config = _config()
    model, data = config["model"], config["data"]
    return (
        torch.randn(batch, data["filter_length"] // 2 + 1, frames),
        torch.tensor([frames] * batch),
        torch.zeros(batch, dtype=torch.long),
        torch.randn(batch, frames, model["text_enc_hidden_dim"]),
        torch.tensor([frames] * batch),
        torch.rand(batch, frames) * 200 + 100,
        torch.randint(1, 200, (batch, frames)),
    )


# --------------------------------------------------------------------------
# every shipped key reaches a method
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(COMPILE_KEYS))
def test_a_shipped_compile_key_reaches_a_method_that_exists(key):
    """The regression this file exists for.

    ``getattr(..., None)`` plus an informational log is indistinguishable, from
    the outside, between "this architecture opts out" and "the implementation
    was deleted and nobody noticed".
    """

    train = _config()["train"]
    assert key in train, f"{key} is not in the shipped config"

    owner, method = COMPILE_KEYS[key]
    model = _discriminator() if owner == "discriminator" else _generator()
    assert callable(
        getattr(model, method, None)
    ), f"{key} is a config key with no {method} behind it"


def test_the_discriminator_compile_key_is_on_and_the_frontend_key_is_off():
    """The frontend's three graphs are unmeasured and see variable lengths, so
    the key ships off; the discriminator's forward is a plain conv stack run
    twice per step, so it ships on."""

    train = _config()["train"]
    assert train["compile_discriminator"] is True
    assert train["compile_frontend"] is False


# --------------------------------------------------------------------------
# what the wrappers must not change
# --------------------------------------------------------------------------


def test_enabling_discriminator_compile_leaves_the_outputs_alone():
    """Without CUDA the compiled call falls back, which is the path under test:
    a fallback that returned something different would be worse than no option.
    """

    torch.manual_seed(0)
    net_d = _discriminator()
    net_d.train()
    y, y_hat = torch.randn(1, 1, 12800), torch.randn(1, 1, 12800)
    before = [t.detach().clone() for t in net_d(y, y_hat)[1]]

    assert net_d.enable_compile() is True
    after = [t.detach().clone() for t in net_d(y, y_hat)[1]]
    assert all(torch.equal(a, b) for a, b in zip(before, after))

    # eval and checkpointing both take the eager path deliberately -- the
    # second because pairing checkpointing with compilation trades a
    # known-good fallback for an untested one.
    net_d.use_checkpointing = True
    assert all(
        torch.equal(a, b) for a, b in zip(before, [t.detach() for t in net_d(y, y_hat)[1]])
    )
    net_d.use_checkpointing = False
    net_d.eval()
    assert all(
        torch.equal(a, b) for a, b in zip(before, [t.detach() for t in net_d(y, y_hat)[1]])
    )


def test_enabling_discriminator_compile_twice_is_idempotent():
    net_d = _discriminator()
    assert net_d.enable_compile("default") is True
    assert net_d.enable_compile("default") is True
    # a different mode is not silently ignored as "already on"
    assert net_d.enable_compile("max-autotune") is False


def test_the_frontend_wrapper_covers_all_three_modules():
    """Wrapped independently, so one module that cannot be traced falls back on
    its own instead of taking the prior and the flow with it."""

    net_g = _generator()
    assert net_g.enable_frontend_compile() is True
    for module in (net_g.enc_p, net_g.enc_q, net_g.flow):
        assert module.forward.__name__ == "training_forward"
    # the loop binds each module by default argument; a closure over the loop
    # variable would point all three wrappers at the flow
    # they sit after ``*args``, so they are keyword-only and live in
    # ``__kwdefaults__`` rather than ``__defaults__``
    wrapped = {
        id(module.forward.__kwdefaults__["_module"])
        for module in (net_g.enc_p, net_g.enc_q, net_g.flow)
    }
    assert wrapped == {id(net_g.enc_p), id(net_g.enc_q), id(net_g.flow)}


# --------------------------------------------------------------------------
# the scalar extractions the frontend used to pay for
# --------------------------------------------------------------------------


def test_the_wavenet_does_not_read_a_scalar_out_of_a_tensor():
    """VITS passes ``hidden_channels`` as a one-element ``IntTensor`` and reads
    ``n_channels[0]`` back to slice with, once per WaveNet layer.

    Measured before the fix: 32 ``aten._local_scalar_dense`` per generator
    forward, 28 of them from that one line.  In eager it is an allocation and a
    scalar extraction per layer on a step that is dispatch-bound; under
    ``torch.compile`` it breaks the graph, which is what makes it a blocker for
    ``compile_frontend`` rather than a tidy-up.

    The four that remain are inside ``torchaudio.functional.resample``, called
    from the decoder's ``_decimate`` -- which is ``@torch.compiler.disable``
    already, and deliberately, for its filter length.
    """

    net_g = _generator()
    net_g.train()
    args = _generator_inputs(net_g)

    counts = collections.Counter()

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counts[str(func)] += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad():
        net_g(*args)  # warm every cached kernel, so the count is steady state
        net_g(*args)
        with Count():
            net_g(*args)

    scalars = counts["aten._local_scalar_dense.default"]
    assert scalars <= 4, (
        f"{scalars} scalar extractions per generator forward; the WaveNet "
        f"should contribute none"
    )
