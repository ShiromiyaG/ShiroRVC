"""SAN heads: the geometry, the detaches, and the door they leave open.

SAN (arXiv 2301.12811) was in this fork for months inside the ChouwaGAN
discriminator and left with it, while every call site that consumes it stayed --
``losses.discriminator_loss`` still handles the ``(function, direction)`` tuple,
``san_direction_weight`` is still read, ``train.py`` still asks for
``supports_san``.  These tests cover what the port has to get right for those
call sites to mean anything again, and they exist because none of it is visible
in a loss curve: a direction that has drifted off the sphere, or a detach in the
wrong place, trains quietly and wrongly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402
from rvc.lib.algorithm.discriminators.san import (  # noqa: E402
    SANConv1d,
    SANConv2d,
    normalize_weight,
)
from rvc.train.losses import discriminator_loss, generator_loss  # noqa: E402

CONFIG = ROOT / "rvc" / "configs" / "refinegan2" / "32000.json"


def _discriminator(use_san):
    model = json.loads(CONFIG.read_text(encoding="utf-8"))["model"]
    return MPD_MSD_Combined(
        model["use_spectral_norm"],
        version=model["d_version"],
        sample_rate=32000,
        use_univhd=bool(model.get("d_use_univhd", False)),
        use_san=use_san,
    )


# --------------------------------------------------------------------------
# the config key
# --------------------------------------------------------------------------


def test_san_ships_on():
    """It stays a key rather than a hardcoded default: it changes the
    discriminator's last projection in every branch, so a run that wants the
    stock head has to be able to ask for one -- and the loss floor moves with
    it, which is why ``d_use_san`` has to be readable from the config rather
    than inferred from the version."""

    model = json.loads(CONFIG.read_text(encoding="utf-8"))["model"]
    assert model["d_use_san"] is True


def test_the_key_reaches_the_discriminator():
    assert _discriminator(False).supports_san is False
    assert _discriminator(True).supports_san is True


# --------------------------------------------------------------------------
# the geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer", [SANConv1d(8, 4, 3, padding=1), SANConv2d(8, 4, (3, 1), padding=(1, 0))]
)
def test_the_weight_starts_on_the_unit_sphere_with_the_norm_kept_as_scale(layer):
    """The split is exact: direction times scale reconstructs the norm the
    layer was initialised with, so this is a reparametrisation and not a
    reinitialisation."""

    norms = layer.weight.detach().flatten(1).norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert layer.scale.shape == (4,)
    # the bias moved to the input side, in input channels
    assert layer.bias.shape == (8,)


def test_an_optimizer_step_moves_the_direction_off_the_sphere_and_normalize_puts_it_back():
    """The reason ``_normalize_san_weights`` has to run every step.  Without
    it nothing raises and nothing looks wrong; the projection simply stops
    being a projection."""

    layer = SANConv2d(4, 2, (3, 1), padding=(1, 0))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1)
    out = layer(torch.randn(2, 4, 16, 1), san_training=True)
    (out[0].sum() + out[1].sum()).backward()
    optimizer.step()

    drifted = layer.weight.detach().flatten(1).norm(p=2, dim=1)
    assert not torch.allclose(drifted, torch.ones_like(drifted), atol=1e-4)
    layer.normalize_weight()
    fixed = layer.weight.detach().flatten(1).norm(p=2, dim=1)
    assert torch.allclose(fixed, torch.ones_like(fixed), atol=1e-5)


def test_normalize_weight_is_idempotent():
    weight = torch.randn(5, 3, 4)
    once = normalize_weight(weight)
    assert torch.allclose(once, normalize_weight(once), atol=1e-6)


# --------------------------------------------------------------------------
# the detaches, which are the method
# --------------------------------------------------------------------------


def test_the_direction_output_does_not_train_the_trunk():
    """``direction`` reads a detached input, so its gradient reaches the
    projection and stops.  If it reached the trunk, the direction term would be
    a second adversarial loss rather than a reparametrisation."""

    layer = SANConv2d(4, 2, (3, 1), padding=(1, 0))
    x = torch.randn(2, 4, 16, 1, requires_grad=True)
    _, direction = layer(x, san_training=True)
    direction.sum().backward()
    assert x.grad is None or torch.count_nonzero(x.grad) == 0
    assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad)
    # the scale is the *function* term's parameter, not the direction's
    assert layer.scale.grad is None or torch.count_nonzero(layer.scale.grad) == 0


def test_the_function_output_does_not_move_the_direction():
    layer = SANConv2d(4, 2, (3, 1), padding=(1, 0))
    x = torch.randn(2, 4, 16, 1, requires_grad=True)
    function, _ = layer(x, san_training=True)
    function.sum().backward()
    assert torch.count_nonzero(x.grad)
    assert torch.count_nonzero(layer.scale.grad)
    assert layer.weight.grad is None or torch.count_nonzero(layer.weight.grad) == 0


# --------------------------------------------------------------------------
# what the training loop sees
# --------------------------------------------------------------------------


def test_only_the_discriminator_pass_asks_for_the_direction():
    """The generator update must get plain logits: the direction is not
    something it may move, and asking for it would build a graph with no
    consumer -- the same waste ``no_grad_real`` exists to avoid."""

    net_d = _discriminator(True)
    net_d.train()
    y, y_hat = torch.randn(2, 1, 8192), torch.randn(2, 1, 8192)

    real, fake, _, _ = net_d(y, y_hat, san_training=True)
    assert all(isinstance(item, (list, tuple)) and len(item) == 2 for item in real)
    assert all(isinstance(item, (list, tuple)) and len(item) == 2 for item in fake)

    real, fake, _, _ = net_d(y, y_hat)
    assert all(torch.is_tensor(item) for item in real + fake)


def test_a_san_discriminator_cannot_load_a_plain_one():
    """``conv_post`` carries ``weight``/``scale`` under SAN and
    ``parametrizations.weight.original0/1`` without it, and ``net_d`` loads
    strictly -- so the two cannot be crossed silently.  That is the whole
    safety story for a key that changes what the weights mean."""

    plain, san = _discriminator(False), _discriminator(True)
    assert set(plain.state_dict()) != set(san.state_dict())
    with pytest.raises(RuntimeError):
        san.load_state_dict(plain.state_dict(), strict=True)

    scales = [k for k in san.state_dict() if k.endswith("conv_post.scale")]
    assert len(scales) == len(san.discriminators)


def test_the_losses_take_their_san_form_end_to_end():
    """The tuple path through ``discriminator_loss`` and the softplus path
    through ``generator_loss``, on a real discriminator -- these were written
    for a class that no longer existed."""

    net_d = _discriminator(True)
    net_d.train()
    y, y_hat = torch.randn(2, 1, 8192), torch.randn(2, 1, 8192)

    real, fake, _, _ = net_d(y, y_hat, san_training=True)
    loss_disc, loss_real, loss_fake = discriminator_loss(
        real, fake, san_direction_weight=0.25, normalize=False
    )
    assert torch.isfinite(loss_disc)
    assert torch.allclose(loss_disc, loss_real + loss_fake, atol=1e-5)
    loss_disc.backward()

    _, fake, _, _ = net_d(y, y_hat)
    loss_adv = generator_loss(fake, normalize=False, use_softplus=True)
    assert torch.isfinite(loss_adv)


def test_the_loss_floor_moves_and_the_old_one_no_longer_applies():
    """A discriminator that emits one constant for real and fake scores 0.5 per
    branch under LSGAN and ``softplus(0.5)**2`` per side under SAN.  Every
    reading of ``loss_disc`` -- "3.75 against a 4.5 floor is healthy" -- is
    relative to this number, so switching the key resets it."""

    import math

    branches = len(_discriminator(True).discriminators)
    lsgan = 0.5 * branches
    per_side = math.log1p(math.exp(0.5)) ** 2
    san = (1.0 + 0.25) * 2.0 * per_side * branches

    constant = [torch.full((2, 8), 0.5) for _ in range(branches)]
    measured_lsgan, _, _ = discriminator_loss(constant, constant, normalize=False)
    assert measured_lsgan.item() == pytest.approx(lsgan, rel=1e-4)

    pairs = [[torch.full((2, 8), 0.5), torch.full((2, 8), 0.5)] for _ in range(branches)]
    measured_san, _, _ = discriminator_loss(
        pairs, pairs, san_direction_weight=0.25, normalize=False
    )
    assert measured_san.item() == pytest.approx(san, rel=1e-4)
    assert measured_san.item() > 2 * measured_lsgan.item()
