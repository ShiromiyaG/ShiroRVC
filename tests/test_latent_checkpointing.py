"""Activation checkpointing for the VITS latent frontend, on its own switch.

The decoder's ``use_checkpointing`` and the latent's are separate decisions:
the decoder runs at the sample rate, the latent at 100 Hz over
``segment_size / hop_length`` frames, and the stack that dominates the latent's
activations is the posterior's spectrogram stem -- 1025 bins wide, deleted at
export, and therefore paid for on every training step and never at inference.

Measured on CUDA at 192 latent channels with a 1025-bin spectrogram, peak
allocation over one forward+backward of ``forward_train``:

    batch  frames    off       on     saved
        8      40   177.5    134.8      24%
        8     200   523.8    232.2      56%
       16      40   273.6    188.7      31%
       16     200   968.6    376.7      61%

What this file pins is the part that must not drift: recomputation is an
exchange of time for memory and *nothing else*, so the losses and every
gradient have to come out identical, and the flag has to default to off so a
run that predates it is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", exc_type=ImportError)

from rvc.lib.algorithm.chouwa_vits import RefineVitsLatent  # noqa: E402

FRAMES = 12
SPEC_BINS = 129


def _latent(checkpointing: bool) -> RefineVitsLatent:
    torch.manual_seed(0)
    return RefineVitsLatent(
        input_channels=32,
        spec_channels=SPEC_BINS,
        content_channels=16,
        detail_channels=8,
        gin_channels=16,
        posterior_channels=32,
        prior_hidden_channels=32,
        latent_channels=16,
        posterior_layers=2,
        flow_blocks=2,
        flow_layers=2,
        prior_blocks=2,
        prior_heads=2,
        prior_kernel_size=5,
        checkpointing=checkpointing,
    ).train()


def _step(model: RefineVitsLatent):
    torch.manual_seed(1)
    content_stats = torch.randn(2, 32, FRAMES)
    spec = torch.rand(2, SPEC_BINS, FRAMES)
    g = torch.randn(2, 16, 1)
    mask = torch.ones(2, 1, FRAMES)
    pitchf = torch.rand(2, FRAMES) * 300.0

    # The posterior sample and the replacement selector both draw from the
    # global generator, and checkpointing must not consume it differently.
    torch.manual_seed(7)
    parts = model.forward_train(content_stats, spec, g, mask, pitchf)
    # The prior reaches the decoder path only through the KL: the posterior
    # takes its conditioning *detached* on purpose, so a loss built from
    # ``content``/``detail`` alone leaves the whole prior stack ungraded and
    # would test half of what the flag covers.
    loss = (
        parts["content"].square().mean()
        + parts["detail"].square().mean()
        + parts["posterior_z_p"].square().mean()
        + sum(value.square().mean() for value in parts["prior_fast_distribution"])
    )
    loss.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients


def test_recomputation_changes_neither_the_loss_nor_any_gradient():
    off_loss, off_grads = _step(_latent(False))
    on_loss, on_grads = _step(_latent(True))

    assert on_loss.item() == pytest.approx(off_loss.item(), rel=1e-6)
    assert set(on_grads) == set(off_grads)
    for name, gradient in off_grads.items():
        assert torch.allclose(on_grads[name], gradient, rtol=1e-5, atol=1e-7), name


def test_the_prior_stack_and_the_posterior_are_both_covered():
    """A flag that only reached one of the two stacks would still recompute
    something and still look like it worked -- so name both."""
    _, gradients = _step(_latent(True))
    assert any(name.startswith("prior_blocks.") for name in gradients)
    assert any(name.startswith("posterior_enc.") for name in gradients)
    assert any(name.startswith("posterior_input.") for name in gradients)


def test_it_is_off_unless_asked_for():
    assert _latent(False).checkpointing is False
    assert RefineVitsLatent(input_channels=8, spec_channels=17).checkpointing is False


def test_inference_never_recomputes():
    """``infer`` stores no activations to begin with, so checkpointing there is
    a second forward bought for nothing."""
    model = _latent(True).eval()
    calls = []
    original = model.prior_blocks[0].forward
    model.prior_blocks[0].forward = lambda *args: calls.append(1) or original(*args)

    with torch.no_grad():
        model.infer(
            torch.randn(1, 32, FRAMES),
            torch.randn(1, 16, 1),
            torch.ones(1, 1, FRAMES),
            torch.rand(1, FRAMES) * 300.0,
        )
    assert calls == [1]


@pytest.mark.parametrize("vocoder", ("chouwagan", "wavehax"))
def test_both_vits_configs_expose_the_switch(vocoder):
    config = json.loads(
        (ROOT / "rvc" / "configs" / vocoder / "44100.json").read_text(encoding="utf-8")
    )
    assert config["model"]["latent_checkpointing"] is False


def test_the_synthesizer_reads_its_own_key_not_the_decoders():
    """``checkpointing`` is the decoder's.  Sharing it would make one flag
    govern two stacks whose trade-offs are not the same."""
    source = (ROOT / "rvc" / "lib" / "algorithm" / "synthesizers.py").read_text(
        encoding="utf-8"
    )
    assert 'vocoder_options.get("latent_checkpointing", False)' in source
