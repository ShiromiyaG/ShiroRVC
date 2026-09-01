"""The period set scaled to the sample rate, and the guard that makes it safe.

RefineGAN's ``[2, 3, 5, 7, 11]`` is HiFi-GAN's, chosen at 22.05 kHz and carried
to 32 and 44.1 kHz unchanged.  A period is a *time* -- ``p`` samples -- so the
same integers mean a different thing at a different rate, and the whole set
slides up an octave at 44.1 kHz, emptying the slow end where pitch structure
lives.  These tests pin the derivation, the values in the shipped configs, and
the checkpoint key without which the change would be silent.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402
from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (  # noqa: E402
    DISCRIMINATOR_VERSIONS,
    PERIODS_BY_RATE,
    REFERENCE_SAMPLE_RATE,
    rate_scaled_periods,
)
from rvc.train.utils import (  # noqa: E402
    assert_periods_match,
    discriminator_periods,
)

CONFIGS = {
    32000: ROOT / "rvc" / "configs" / "refinegan" / "32000.json",
    44100: ROOT / "rvc" / "configs" / "refinegan" / "44100.json",
}
STOCK = tuple(DISCRIMINATOR_VERSIONS["v3"][0])

#: ``DiscriminatorP``'s receptive field in rows: kernel 5 at strides
#: [3, 3, 3, 3, 1] under a final (3, 1).  It is what turns a period into a span
#: in seconds, so it is what makes "the same periods mean less time at a higher
#: rate" a measurable claim rather than an intuition.
RECEPTIVE_ROWS = 647


def _span_ms(period, sample_rate):
    return RECEPTIVE_ROWS * period / sample_rate * 1000.0


# --------------------------------------------------------------------------
# the derivation
# --------------------------------------------------------------------------


def test_the_reference_rate_is_a_fixed_point():
    """The property that makes this a derivation and not a new design: at the
    rate the set was chosen for, it returns the set."""

    assert rate_scaled_periods(STOCK, REFERENCE_SAMPLE_RATE) == STOCK


@pytest.mark.parametrize("sample_rate", sorted(PERIODS_BY_RATE))
def test_the_table_is_what_the_rule_produces(sample_rate):
    """The table is frozen so the values are greppable; this is what stops it
    from drifting away from the rule it claims to be."""

    assert rate_scaled_periods(STOCK, sample_rate) == PERIODS_BY_RATE[sample_rate]


@pytest.mark.parametrize("sample_rate", sorted(PERIODS_BY_RATE))
def test_the_periods_stay_pairwise_coprime(sample_rate):
    """The reason the targets are rounded to *primes* rather than used as-is:
    two periods sharing a factor fold onto overlapping sample subsets, and the
    pair becomes one branch at two branches' cost.  The naive scaling at 44.1
    kHz is [4, 6, 10, 14, 22], every one of which shares 2."""

    import math

    periods = PERIODS_BY_RATE[sample_rate]
    for i, a in enumerate(periods):
        for b in periods[i + 1 :]:
            assert math.gcd(a, b) == 1


@pytest.mark.parametrize("sample_rate", sorted(PERIODS_BY_RATE))
def test_the_scaled_set_restores_the_reference_time_scales(sample_rate):
    """The point of the whole change, stated as the measurement: every branch
    lands within 25% of the span its 22.05 kHz counterpart covered, where the
    un-scaled set is off by the full rate ratio (a factor of 2 at 44.1 kHz)."""

    reference = [_span_ms(p, REFERENCE_SAMPLE_RATE) for p in STOCK]
    scaled = [_span_ms(p, sample_rate) for p in PERIODS_BY_RATE[sample_rate]]
    stock = [_span_ms(p, sample_rate) for p in STOCK]
    for want, got, before in zip(reference, scaled, stock):
        # 25% is not a slack bound -- it is the worst residual the rounding
        # leaves, period 5 at 44.1 kHz against a target of 4, and it is hit
        # exactly.  Tightening it means finding a better rounding, not moving
        # the number.
        assert abs(got - want) / want <= 0.25 + 1e-9
        assert abs(got - want) < abs(before - want)


def test_the_slow_end_is_what_the_un_scaled_set_loses():
    """Not "worse at the top" -- emptier at the bottom.  At 44.1 kHz the stock
    set's longest branch spans 161 ms; the scaled one restores something past
    300 ms, which is the range pitch and its slow structure occupy."""

    stock_longest = max(_span_ms(p, 44100) for p in STOCK)
    scaled_longest = max(_span_ms(p, 44100) for p in PERIODS_BY_RATE[44100])
    assert stock_longest < 200.0 < 300.0 < scaled_longest


def test_the_resolutions_are_deliberately_not_scaled():
    """The frequency branches are pinned by a measurement and the period
    branches are not, which is the whole reason only one of the two families
    moves.  Scaling the resolutions would double the 50-sample hop, and that
    hop is the only thing in the discriminator that sees the frame-rate
    defect (68-76% against 50-54% for every alternative tried)."""

    for path in CONFIGS.values():
        assert json.loads(path.read_text())["model"]["d_resolutions"] is None
    hops = sorted(hop for _n_fft, hop, _win in DISCRIMINATOR_VERSIONS["v3"][1])
    assert hops[0] == 50


# --------------------------------------------------------------------------
# the configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_names_the_scaled_set(sample_rate):
    """The values reach a run through its config, not through code: an
    experiment keeps its own ``config.json``, so this changes new experiments
    and leaves runs in flight on the set their discriminator was trained with.
    Applying it from ``_build_d_model`` when ``d_periods`` is ``None`` would
    have re-periodised every existing run on its next resume."""

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    assert tuple(model["d_periods"]) == PERIODS_BY_RATE[sample_rate]


def test_the_two_rates_do_not_get_the_same_set():
    """A single hard-coded list for both rates is the bug this replaces, so it
    is worth one line to pin that they actually differ."""

    assert PERIODS_BY_RATE[32000] != PERIODS_BY_RATE[44100]


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_config_builds_the_branches_it_names(sample_rate):
    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    built = MPD_MSD_Combined(
        model["use_spectral_norm"],
        version=model["d_version"],
        periods=model["d_periods"],
    )
    assert built.periods == PERIODS_BY_RATE[sample_rate]
    assert [d.period for d in built.discriminators if hasattr(d, "period")] == list(
        PERIODS_BY_RATE[sample_rate]
    )


def test_hifigan_keeps_applios_periods():
    """Deliberately not scaled: v2's eight periods are Applio's, every HiFi-GAN
    pretrain in circulation is trained against them, and the parity is worth
    more than the alignment."""

    assert 40000 not in PERIODS_BY_RATE and 48000 not in PERIODS_BY_RATE
    assert MPD_MSD_Combined(False).periods == tuple(DISCRIMINATOR_VERSIONS["v2"][0])


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def test_a_period_change_is_invisible_in_the_weights():
    """The premise of the guard, pinned so it is not taken on faith.  If this
    ever fails, ``load_state_dict`` catches the mismatch on its own and the
    explicit key is redundant -- but it does not, because the period lives in a
    ``view`` and the kernels are ``(k, 1)`` regardless of it."""

    a = MPD_MSD_Combined(False, version="v3", periods=STOCK)
    b = MPD_MSD_Combined(False, version="v3", periods=PERIODS_BY_RATE[44100])
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not missing and not unexpected
    # ...and it stays wrong, silently: the branches now fold at frequencies the
    # weights never saw, and nothing raised.
    assert b.periods != a.periods


def test_the_checkpoint_carries_the_periods():
    model = MPD_MSD_Combined(False, version="v3", periods=PERIODS_BY_RATE[44100])
    assert discriminator_periods(model) == list(PERIODS_BY_RATE[44100])
    assert_periods_match(
        model, {"discriminator_periods": list(PERIODS_BY_RATE[44100])}
    )


def test_a_mismatch_is_refused_and_says_what_to_do():
    model = MPD_MSD_Combined(False, version="v3", periods=PERIODS_BY_RATE[44100])
    with pytest.raises(ValueError, match="period mismatch") as excinfo:
        assert_periods_match(model, {"discriminator_periods": list(STOCK)})
    assert "d_periods" in str(excinfo.value)
    assert str(list(STOCK)) in str(excinfo.value)


def test_a_checkpoint_without_the_key_is_read_as_the_un_scaled_set():
    """"Absent" is not "nothing to check": every checkpoint predating the key
    is one of the runs this change would re-periodise, which is exactly the
    case the guard exists for.  Waving them through would make it useless."""

    scaled = MPD_MSD_Combined(False, version="v3", periods=PERIODS_BY_RATE[44100])
    with pytest.raises(ValueError, match="period mismatch"):
        assert_periods_match(scaled, {})
    # A run that kept the stock set resumes from such a checkpoint untouched,
    # which is what keeps every in-flight run working.
    stock = MPD_MSD_Combined(False, version="v3", periods=STOCK)
    assert_periods_match(stock, {})


def test_a_model_with_no_periods_is_not_checked():
    """The ChouwaGAN stack and any period-less variant have nothing to compare,
    and the guard must not invent a mismatch for them."""

    periodless = MPD_MSD_Combined(False, version="v3", periods=[])
    assert discriminator_periods(periodless) == []
    assert_periods_match(periodless, {"discriminator_periods": []})
    assert_periods_match(types.SimpleNamespace(), {})


def test_the_trainer_guards_both_doors():
    """``load_checkpoint`` covers the resume; the pretrained-D path loads with
    a bare ``load_state_dict`` and needed the call spelled out.  Both matter:
    every pretrained D in circulation was trained on the un-scaled set."""

    import ast

    train = ast.parse((ROOT / "rvc" / "train" / "train.py").read_text())
    utils = ast.parse((ROOT / "rvc" / "train" / "utils.py").read_text())
    called = [
        node
        for tree in (train, utils)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "assert_periods_match"
    ]
    assert len(called) == 2
