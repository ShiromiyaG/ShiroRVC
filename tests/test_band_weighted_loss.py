"""The mel distance's frequency weighting.

The bug these pin: a mean over mel bins gives every bin the same vote, so the
sub-1 kHz region held 46% of the error while drawing 32% of the gradient and
never converged.  What matters is that the weights raise the low bins *without*
moving the loss scale, since the mel term feeds the adaptive adversarial
balance.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rvc" / "train"))

# A bare ``import torch`` here fails *collection* rather than the test, and a
# collection error aborts the whole session with exit 2 -- so on a runner
# without torch this one module took the entire suite down with it.
torch = pytest.importorskip("torch", reason="the losses need torch", exc_type=ImportError)
pytest.importorskip("librosa", reason="rvc.train.losses imports it", exc_type=ImportError)

from rvc.train.losses import (  # noqa: E402
    BandWeightedSpectralLoss,
    mel_low_frequency_weights,
)

SR = 44100
BINS = 128


def _weights(emphasis=2.0, **kw):
    return mel_low_frequency_weights(
        num_mels=BINS, sample_rate=SR, mel_fmin=0.0, mel_fmax=None,
        emphasis=emphasis, **kw
    )


def test_weights_leave_the_loss_scale_alone():
    """Mean 1.0, or the reweighting silently retunes the GAN balance."""
    for emphasis in (1.0, 1.5, 2.0, 4.0):
        assert _weights(emphasis).mean().item() == pytest.approx(1.0, abs=1e-6)


def test_emphasis_of_one_is_a_no_op():
    assert torch.allclose(_weights(1.0), torch.ones(BINS))


def test_low_bins_are_raised_and_high_bins_are_not():
    w = _weights(2.0, cutoff_hz=1000.0)
    # Bin 31 centres at 992 Hz, bin 32 at 1024 Hz: the boundary of the region
    # that was losing the vote.
    assert w[0].item() > 1.0
    assert w[:32].min().item() > w[64:].max().item()
    # Above the taper the weight has to come back to a plain shared value.
    assert w[-1].item() < 1.0
    assert w[96:].std().item() == pytest.approx(0.0, abs=1e-6)


def test_the_taper_has_no_step_in_it():
    """No bin may sit on a cliff.  The bound is per-bin, not absolute: Slaney
    mel is linear below 1 kHz and logarithmic above, so the octave the taper
    spans is covered by ~20 bins and each one carries a slice of the ramp."""
    w = _weights(3.0)
    steps = (w[1:] - w[:-1]).abs()
    span = w.max() - w.min()
    assert steps.max().item() < 0.15 * span.item()


def test_weighted_loss_matches_the_unweighted_one_when_flat():
    base = torch.nn.SmoothL1Loss(beta=0.3, reduction="none")
    weighted = BandWeightedSpectralLoss(base, torch.ones(BINS))
    a = torch.randn(2, BINS, 40)
    b = torch.randn(2, BINS, 40)
    plain = torch.nn.SmoothL1Loss(beta=0.3)(a, b)
    assert weighted(a, b).item() == pytest.approx(plain.item(), rel=1e-6)


def test_weighted_loss_reacts_more_to_a_low_band_error():
    weighted = BandWeightedSpectralLoss(
        torch.nn.SmoothL1Loss(beta=0.3, reduction="none"), _weights(2.0)
    )
    target = torch.zeros(1, BINS, 40)
    low, high = target.clone(), target.clone()
    low[:, :16] = 1.0
    high[:, -16:] = 1.0
    assert weighted(target, low).item() > weighted(target, high).item()


def test_a_reduced_base_is_rejected():
    with pytest.raises(ValueError, match="unreduced"):
        BandWeightedSpectralLoss(torch.nn.L1Loss(), torch.ones(BINS))


def test_a_bin_count_mismatch_is_rejected():
    weighted = BandWeightedSpectralLoss(
        torch.nn.L1Loss(reduction="none"), torch.ones(BINS)
    )
    with pytest.raises(ValueError, match="mel bins"):
        weighted(torch.zeros(1, 80, 10), torch.zeros(1, 80, 10))


def test_gradient_share_reaches_parity_with_the_error_share():
    """The default of 2.0 is calibrated, not picked: it puts the gradient
    share of bins 0-32 on top of their measured 46% error share."""
    w = _weights(2.0)
    # Measured on the epoch-16 validation sample in the real loss domain,
    # log1p(mel * 1000): bins 0-32 drew 32.5% of the gradient while owning
    # 46.3% of the error.  Back out the per-bin gradient ratio that produces
    # that 32.5% and check the weights close the gap.
    low, high = 1.0, 0.325 * 96 / (0.675 * 32)
    per_bin_gradient = torch.cat([torch.full((32,), low), torch.full((96,), 1 / high)])
    weighted = per_bin_gradient * w
    assert weighted[:32].sum().item() / weighted.sum().item() == pytest.approx(
        0.463, abs=0.02
    )


def test_a_weight_factory_serves_other_resolutions():
    """Multi-scale mel evaluates the same distance at 5..320 bands.

    Without a factory this raised ``Band weights cover 128 mel bins but the
    loss was handed 5`` on the first step, so selecting "Multi-Scale Mel Loss"
    in the UI could not run at all with a low-emphasis config.  The weighting
    is defined by frequency, not by bin count, so it is rebuilt per resolution.
    """
    def factory(num_mels):
        return mel_low_frequency_weights(
            num_mels=num_mels,
            sample_rate=SR,
            mel_fmin=0.0,
            mel_fmax=None,
            emphasis=2.0,
            cutoff_hz=1000.0,
        )

    loss = BandWeightedSpectralLoss(
        torch.nn.L1Loss(reduction="none"), factory(BINS), weight_factory=factory
    )
    for bands in (5, 20, 80, 320):
        a = torch.randn(2, bands, 7)
        assert torch.isfinite(loss(a, torch.randn_like(a)))


def test_a_mismatch_without_a_factory_is_still_rejected():
    """The strict default stays: for a single-resolution mel a mismatch means
    the config and the weights disagree, and silently reweighting the wrong
    bands is worse than stopping."""
    loss = BandWeightedSpectralLoss(
        torch.nn.L1Loss(reduction="none"), torch.ones(BINS)
    )
    with pytest.raises(ValueError, match="mel bins"):
        loss(torch.randn(2, 5, 7), torch.randn(2, 5, 7))
