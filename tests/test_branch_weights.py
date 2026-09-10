"""Per-branch discriminator weights, and the series that says whether one is right.

Two things live here.  The weights themselves -- ``MPD_MSD_Combined`` pins
UnivHD below 1.0 on ``v4``, and the three losses have to honour that in
branch order or they weight the wrong head.  And ``adv_sep``, the per-head share
of ``loss_adv``, which exists because the diagnostic that was already there
cannot answer the question the weight is chosen against.

That last point is the reason for the arithmetic in
``test_separation_ratio_is_not_the_loss_ratio``.  ``disc_sep`` is a logit gap and
the generator's term is a saturating function of it, so reading a weight off the
separations overstates the imbalance by more than an order of magnitude -- which
is exactly the mistake this series removes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (  # noqa: E402
    DEFAULT_BRANCH_WEIGHT,
    UNIVHD_WEIGHT_BY_VERSION,
    MPD_MSD_Combined,
    univhd_weight_for,
)
from rvc.train.losses import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
)

#: The separations measured on the two pretrains that motivated the weight: one
#: head an order of magnitude clear of the other eight.  Logits, not losses.
UNIVHD_LOGIT = -2.0
ORDINARY_LOGIT = 0.05


def _outputs(count=9, univhd_index=8):
    return [
        torch.full((2, 1, 16), UNIVHD_LOGIT if i == univhd_index else ORDINARY_LOGIT)
        for i in range(count)
    ]


def _build(version, **kwargs):
    return MPD_MSD_Combined(
        version=version,
        sample_rate=32000,
        use_univhd=True,
        use_san=True,
        **kwargs,
    )


@pytest.mark.parametrize("version", sorted(UNIVHD_WEIGHT_BY_VERSION))
def test_the_pinned_versions_land_on_the_univhd_branch(version):
    """The weight has to reach the branch it names, and no other."""

    model = _build(version)
    weights = model.branch_weights
    labels = model.branch_labels

    assert len(weights) == len(labels) == len(model.discriminators)
    assert labels[-1] == "univhd"
    assert weights[-1] == UNIVHD_WEIGHT_BY_VERSION[version]
    assert set(weights[:-1]) == {DEFAULT_BRANCH_WEIGHT}
    assert model.uses_branch_weights is True


def test_an_unpinned_version_keeps_the_papers_additive_weight():
    """``v3`` predates UnivHD here, so there is no measurement behind a number."""

    assert univhd_weight_for("v3") == DEFAULT_BRANCH_WEIGHT
    model = _build("v3")
    assert model.branch_weights[-1] == DEFAULT_BRANCH_WEIGHT
    assert model.uses_branch_weights is False


def test_an_explicit_weight_wins_on_any_version():
    assert _build("v4", univhd_weight=1.0).branch_weights[-1] == 1.0
    assert _build("v3", univhd_weight=0.5).branch_weights[-1] == 0.5


def test_a_negative_weight_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        _build("v4", univhd_weight=-0.25)


def test_no_weights_is_the_unweighted_loss_exactly():
    """The default path must not move: every existing run took it."""

    outputs = _outputs()
    fmaps = [[torch.randn(2, 4, 8)] for _ in range(9)]
    other = [[torch.zeros(2, 4, 8)] for _ in range(9)]
    ones = (DEFAULT_BRANCH_WEIGHT,) * 9

    assert torch.allclose(
        generator_loss(outputs, use_softplus=True),
        generator_loss(outputs, use_softplus=True, branch_weights=ones),
    )
    assert torch.allclose(
        feature_loss(fmaps, other),
        feature_loss(fmaps, other, branch_weights=ones),
    )
    assert torch.allclose(
        discriminator_loss(outputs, outputs)[0],
        discriminator_loss(outputs, outputs, branch_weights=ones)[0],
    )


def test_the_weight_scales_only_its_own_branch():
    outputs = _outputs()
    weights = (DEFAULT_BRANCH_WEIGHT,) * 8 + (0.25,)

    _, unweighted = generator_loss(outputs, use_softplus=True, per_branch=True)
    _, weighted = generator_loss(
        outputs, use_softplus=True, branch_weights=weights, per_branch=True
    )

    assert torch.allclose(weighted[:8], unweighted[:8])
    assert torch.allclose(weighted[8], unweighted[8] * 0.25)


def test_adv_sep_is_the_decomposition_of_the_loss_it_reports():
    """The series is only readable if the parts are the whole."""

    outputs = _outputs()
    weights = (DEFAULT_BRANCH_WEIGHT,) * 8 + (0.25,)

    total, per_branch = generator_loss(
        outputs, use_softplus=True, branch_weights=weights, per_branch=True
    )
    bare = generator_loss(outputs, use_softplus=True, branch_weights=weights)

    assert per_branch.shape == (9,)
    assert torch.allclose(per_branch.sum(), total)
    # Asking for the diagnostic must not change the number being trained on.
    assert torch.allclose(bare, total)
    assert not per_branch.requires_grad


def test_separation_ratio_is_not_the_loss_ratio():
    """Why ``disc_sep`` cannot stand in for this series.

    At the logits above the gap between UnivHD and an ordinary head is ~41x --
    the ratio the refinegan2 run showed.  The share of ``loss_adv`` it buys is
    5.7x, so a weight read off the separations would overshoot by seven times
    over.  The bounds are wide because the point is the order of magnitude
    between the two ratios, not either one's exact value.
    """

    outputs = _outputs()
    separation_ratio = abs(ORDINARY_LOGIT - UNIVHD_LOGIT) / abs(ORDINARY_LOGIT)
    assert separation_ratio > 40.0

    _, per_branch = generator_loss(outputs, use_softplus=True, per_branch=True)
    loss_ratio = (per_branch[8] / per_branch[0]).item()

    assert 4.0 < loss_ratio < 8.0
    assert loss_ratio < separation_ratio / 5.0


def test_the_weight_moves_the_share_it_is_meant_to():
    """0.25 is chosen against this number, so this is the number to pin."""

    outputs = _outputs()

    _, unweighted = generator_loss(outputs, use_softplus=True, per_branch=True)
    _, weighted = generator_loss(
        outputs,
        use_softplus=True,
        branch_weights=(DEFAULT_BRANCH_WEIGHT,) * 8 + (0.25,),
        per_branch=True,
    )

    before = (unweighted[8] / unweighted.sum()).item()
    after = (weighted[8] / weighted.sum()).item()

    # Was the largest single share of the term; is now smaller than the eight
    # ordinary heads carry between them, without being switched off.
    assert before > 0.35
    assert 0.10 < after < 0.20


def test_the_discriminators_own_loss_is_weighted_but_its_diagnostic_is_not():
    """The two per-branch tensors answer different questions -- see the losses."""

    outputs = _outputs()
    weights = (DEFAULT_BRANCH_WEIGHT,) * 8 + (0.25,)

    plain = discriminator_loss(outputs, outputs, per_branch=True)
    scaled = discriminator_loss(
        outputs, outputs, per_branch=True, branch_weights=weights
    )

    # The objective moved...
    assert scaled[0] < plain[0]
    # ...and the per-head report did not: it says what the head is, not what it
    # was told it is worth.
    assert torch.allclose(scaled[3], plain[3])
