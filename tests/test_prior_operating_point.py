"""The prior path has to be trained at the point inference evaluates it.

Two separate failures, both measured on the 44.1 kHz pretrain and both invisible
in every series that existed at the time:

* Prior replacement fed the decoder prior *samples* (``scale=1``) while
  ``infer`` runs ``deterministic=True`` and hands it the prior *mean*.  Scoring
  the same held-out audio with the same weights at step 65k gave mel L1 0.7358
  at scale 0 against 0.6198 at scale 1 -- a 15.8% penalty for evaluating at an
  operating point the model was never trained at.  Over that same run, 34k
  steps of training bought 0.3%.

* The KL rate controller held one multiplier for the whole branch against the
  per-dim *mean*, and a plain sum applies identical marginal pressure to every
  dimension.  Concentration is therefore free: a mean of 0.1489 against a target
  of 0.15 while two of 64 dimensions carried 29% of the divergence.

Both fixes are structural and neither has a loss series that would catch a
regression, so they are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.chouwagan_vits import RefineVitsLatent


def _latent(**overrides):
    kwargs = dict(
        input_channels=16,
        content_channels=16,
        detail_channels=16,
        spec_channels=12,
        posterior_channels=16,
        prior_hidden_channels=16,
        latent_channels=8,
        posterior_layers=2,
        flow_blocks=2,
        flow_layers=2,
        prior_blocks=1,
        prior_heads=2,
        prior_kernel_size=7,
        kl_target=0.15,
    )
    kwargs.update(overrides)
    return RefineVitsLatent(**kwargs)


# --------------------------------------------------------------------------
# Operating point


def test_replacement_scales_split_across_both_operating_points():
    """Half the replaced items decode from the mean, half from a sample."""
    module = _latent(prior_replacement_mean_share=0.5)
    selector = torch.zeros(8, 1, 1)
    selector[:4] = 1.0

    scales = module._replacement_scales(selector)

    chosen = selector.reshape(-1).bool()
    assert float(scales[chosen].sum()) == pytest.approx(2.0)
    # Untouched items must stay untouched: a scale is only meaningful where the
    # selector actually replaces the latent.
    assert float(scales[~chosen].sum()) == 0.0


def test_replacement_mean_share_of_zero_keeps_every_item_sampled():
    """The opt-out has to be exact, not merely rare."""
    module = _latent(prior_replacement_mean_share=0.0)
    selector = torch.ones(6, 1, 1)

    assert torch.equal(module._replacement_scales(selector), selector)


def test_replacement_mean_share_defaults_to_training_both_points():
    assert _latent().prior_replacement_mean_share == pytest.approx(0.5)


def test_sample_accepts_a_per_item_scale_and_zero_returns_the_mean():
    mean = torch.randn(4, 8, 6)
    logs = torch.zeros(4, 8, 6)
    scale = torch.zeros(4, 1, 1)
    scale[2:] = 1.0

    drawn = RefineVitsLatent._sample(mean, logs, scale=scale)

    assert torch.equal(drawn[:2], mean[:2])
    assert not torch.equal(drawn[2:], mean[2:])


def test_forward_train_reports_the_mean_share_it_applied():
    module = _latent(
        prior_replacement_max=1.0,
        prior_replacement_start=0,
        prior_replacement_ramp=1,
        prior_replacement_mean_share=0.5,
    )
    module.train()
    module.set_training_step(1000)

    parts = module.forward_train(
        content_stats=torch.randn(4, 16, 20),
        spec=torch.rand(4, 12, 20),
        g=None,
        mask=torch.ones(4, 1, 20),
    )

    assert parts["prior_replacement"] == pytest.approx(1.0)
    assert parts["prior_replacement_mean"] == pytest.approx(0.5)
    assert "prior_replacement_mean" in module.diagnostics(parts, torch.ones(4, 1, 20))


# --------------------------------------------------------------------------
# Per-dimension rate control


def test_rate_multipliers_are_per_dimension():
    module = _latent()

    assert module.kl_log_beta_fast.shape == (8,)
    assert module.kl_rate_ema_fast.shape == (8,)


def test_a_hot_dimension_is_pressured_and_a_starved_one_is_released():
    """The failure the branch-wide multiplier could not see.

    A distribution whose *mean* sits exactly on target but whose mass is in one
    dimension: the old controller read zero error and stopped moving.
    """
    module = _latent()
    module.train()
    target = module.kl_target_fast
    width = module.fast_latent_channels
    rate = torch.full((width,), target / 10.0)
    rate[0] = target * width - rate[1:].sum()
    assert rate.mean() == pytest.approx(target, rel=1e-5)

    # The error is clamped to +/-1 and ``kl_beta_lr`` is 0.01, so log-beta moves
    # at most 0.01 per step: reaching either bound from 0 takes ~1000 steps.
    for _ in range(1500):
        beta = module._kl_beta(rate, target)

    assert beta[0] == pytest.approx(module.kl_beta_max, rel=1e-3)
    assert beta[1] == pytest.approx(module.kl_beta_min, rel=1e-3)


def test_prior_losses_weight_each_dimension_by_its_own_multiplier():
    module = _latent()
    module.eval()  # freeze the multipliers so the arithmetic is checkable
    with torch.no_grad():
        module.kl_log_beta_fast.copy_(torch.linspace(-2.0, 0.0, 8))

    frames = 6
    # The VITS estimator is ``logs_p - logs_q - 0.5 + 0.5 * (z_p - mu_p)^2 /
    # var_p``.  With unit variances and a zero prior mean, ``z_p = sqrt(2)``
    # makes it exactly 0.5 per dimension, so the weighted sum is checkable.
    parts = {
        "posterior_z_p": torch.full((1, 8, frames), 2.0 ** 0.5),
        "posterior_fast_distribution": (
            torch.ones(1, 8, frames), torch.zeros(1, 8, frames)
        ),
        "prior_fast_distribution": (
            torch.zeros(1, 8, frames), torch.zeros(1, 8, frames)
        ),
        # The scale anchor reads what the decoder is handed.
        "content": torch.ones(1, 16, frames),
        "detail": torch.ones(1, 16, frames),
    }
    _slow, loss_fast, _total = module.prior_losses(parts, mask=None)

    expected = 0.5 * module.kl_log_beta_fast.exp().sum()
    assert loss_fast == pytest.approx(float(expected), rel=1e-5)
    # The reported beta stays a scalar so the existing series keeps its meaning.
    assert parts["kl_beta_fast"].ndim == 0


def test_free_bits_floor_still_releases_dimensions_below_it():
    """The floor is a separate mechanism and per-dim beta does not replace it.

    The raw estimate is -0.5 here rather than 0: this is the Monte Carlo form,
    which is unbiased for the KL but goes negative for a single sample drawn
    closer to the prior mean than the prior's own spread.  The floor has to hold
    against that, not merely against small positive values.
    """
    module = _latent(kl_free_bits=0.5, kl_target=0.0)
    clamped, raw = module._kl(
        torch.zeros(1, 8, 4),
        (torch.zeros(1, 8, 4), torch.zeros(1, 8, 4)),
        (torch.zeros(1, 8, 4), torch.zeros(1, 8, 4)),
        free_bits=0.5,
    )

    assert torch.allclose(clamped, torch.full((8,), 0.5))
    assert torch.allclose(raw, torch.full((8,), -0.5))
