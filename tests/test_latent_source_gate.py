"""The excitation skips must be gated by the latent, not free.

Without the gate the excitation U-Net is a full-width path from f0 to the
sample that never touches ``z``: a decoder able to rebuild a stage from pitch
alone makes the posterior optional, and the VITS rate collapses for a
structural reason no free-bits floor can reach.
"""

import torch

from rvc.lib.algorithm.generators.chouwagan import ChouwaGANGenerator


def _build(gate: bool) -> ChouwaGANGenerator:
    torch.manual_seed(0)
    return ChouwaGANGenerator(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=(3, 3, 7, 7),
        upsample_initial_channel=320,
        chouwagan_channels=(256, 160, 80, 40),
        chouwagan_block_kernels=((3, 7), (3, 7), (7,), (7,)),
        chouwagan_excitation_unet=True,
        chouwagan_latent_source_gate=gate,
    ).eval()


def _inputs(batch: int = 2, frames: int = 30):
    torch.manual_seed(1)
    return (
        torch.randn(batch, 192, frames),
        torch.rand(batch, frames) * 200.0 + 100.0,
        torch.randn(batch, 256, 1),
    )


def _render(model, x, f0, g):
    # The NSF source is stochastic, so a fixed seed is what makes two forward
    # passes comparable at all.
    torch.manual_seed(7)
    with torch.no_grad():
        return model(x, f0, g)


def test_the_gate_starts_as_an_exact_identity():
    model = _build(True)
    x, f0, g = _inputs()

    gated = _render(model, x, f0, g)
    gates, model.latent_gates = model.latent_gates, None
    ungated = _render(model, x, f0, g)
    model.latent_gates = gates

    # Bit-exact, not approximate: ``2 * sigmoid(0)`` is 1.0, so an existing run
    # can adopt the gate without a discontinuity in its loss curves.
    assert torch.equal(gated, ungated)


def test_the_gate_is_load_bearing_once_it_moves():
    model = _build(True)
    x, f0, g = _inputs()
    before = _render(model, x, f0, g)

    with torch.no_grad():
        torch.nn.init.normal_(model.latent_gates[-1].weight, std=0.05)
    after = _render(model, x, f0, g)

    assert not torch.allclose(before, after)


def test_the_gate_sees_the_latent_and_receives_gradient():
    model = _build(True)
    x, f0, g = _inputs()

    model(x, f0, g).pow(2).mean().backward()

    assert len(model.latent_gates) == 4
    for gate in model.latent_gates:
        assert gate.weight.grad is not None
        assert torch.isfinite(gate.weight.grad).all()
        # A zero gradient at every stage would mean the skip's scale does not
        # depend on the latent, which is the failure this whole path exists to
        # prevent.
        assert float(gate.weight.grad.abs().max()) > 0.0


def test_the_gate_is_off_when_the_excitation_unet_is():
    torch.manual_seed(0)
    model = ChouwaGANGenerator(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=(3, 3, 7, 7),
        upsample_initial_channel=320,
        chouwagan_channels=(256, 160, 80, 40),
        chouwagan_block_kernels=((3, 7), (3, 7), (7,), (7,)),
        chouwagan_excitation_unet=False,
        chouwagan_latent_source_gate=True,
    )
    # The additive source path has no skips to gate; asking for the gate there
    # must not build one rather than build a dead module.
    assert model.latent_gates is None


def test_disabling_the_gate_keeps_the_decoder_runnable():
    model = _build(False)
    assert model.latent_gates is None
    assert _render(model, *_inputs()).shape == (2, 1, 30 * 441)
