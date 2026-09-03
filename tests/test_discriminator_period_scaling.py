"""The period set scaled to the sample rate, and the guard that makes it safe.

RefineGAN's ``[2, 3, 5, 7, 11]`` is HiFi-GAN's, chosen at 22.05 kHz and carried
to every other rate unchanged.  A period is a *time* -- ``p`` samples -- so
the same integers mean a different thing at a different rate, and the set
slides up as the rate rises, emptying the slow end where pitch structure lives.
These tests pin the derivation, the shipped values, and the checkpoint key
without which the change would be silent.
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
    periods_for,
    rate_scaled_periods,
)
from rvc.train.utils import (  # noqa: E402
    assert_periods_match,
    discriminator_periods,
)

CONFIGS = {
    32000: ROOT / "rvc" / "configs" / "refinegan2" / "32000.json",
}
STOCK = tuple(DISCRIMINATOR_VERSIONS["v3"][0])

#: The table is nested by version now, because ``v3`` and ``v4`` no longer
#: agree on the period count -- ``v4`` is ``v3`` minus its longest branch.
#: Every derivation below therefore runs against both, since a rule that only
#: holds for one of the two shipped sets is not the rule the table claims.
VERSIONED = [
    (version, rate)
    for version, table in sorted(PERIODS_BY_RATE.items())
    for rate in sorted(table)
]

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


@pytest.mark.parametrize("version,sample_rate", VERSIONED)
def test_the_table_is_what_the_rule_produces(version, sample_rate):
    """The table is frozen so the values are greppable; this is what stops it
    from drifting away from the rule it claims to be."""

    stock = tuple(DISCRIMINATOR_VERSIONS[version][0])
    assert rate_scaled_periods(stock, sample_rate) == periods_for(version, sample_rate)


def test_v4_is_v3_without_its_longest_branch():
    """The one relationship between the two tables, pinned so a future edit
    to either has to say what it means.  Dropping the *longest* period is
    what makes this the cheap cut that costs the least above 10 kHz: a
    longer period folds at a lower rate, so it is the branch least able to
    say anything about the top octave."""

    assert DISCRIMINATOR_VERSIONS["v4"][0] == DISCRIMINATOR_VERSIONS["v3"][0][:-1]
    for sample_rate in sorted(PERIODS_BY_RATE["v3"]):
        assert periods_for("v4", sample_rate) == periods_for("v3", sample_rate)[:-1]


def test_an_unknown_version_or_rate_raises_rather_than_guessing():
    """A wrong period set is invisible in every weight, so the lookup must
    not have a fallback that could quietly supply one."""

    with pytest.raises(KeyError):
        periods_for("v2", 32000)
    with pytest.raises(KeyError):
        periods_for("v3", 48000)
    # 44.1 kHz is gone from the table on purpose: RefineGAN ships at 32 kHz
    # only, and a frozen row nothing can select is a value nobody checks.
    with pytest.raises(KeyError):
        periods_for("v3", 44100)


@pytest.mark.parametrize("version,sample_rate", VERSIONED)
def test_the_periods_stay_pairwise_coprime(version, sample_rate):
    """The reason the targets are rounded to *primes* rather than used as-is:
    two periods sharing a factor fold onto overlapping sample subsets, and the
    pair becomes one branch at two branches' cost.  The naive scaling at 44.1
    kHz is [4, 6, 10, 14, 22], every one of which shares 2."""

    import math

    periods = periods_for(version, sample_rate)
    for i, a in enumerate(periods):
        for b in periods[i + 1 :]:
            assert math.gcd(a, b) == 1


@pytest.mark.parametrize("version,sample_rate", VERSIONED)
def test_the_scaled_set_restores_the_reference_time_scales(version, sample_rate):
    """The point of the whole change, stated as the measurement: every branch
    lands within 25% of the span its 22.05 kHz counterpart covered, where the
    un-scaled set is off by the full rate ratio."""

    unscaled = tuple(DISCRIMINATOR_VERSIONS[version][0])
    reference = [_span_ms(p, REFERENCE_SAMPLE_RATE) for p in unscaled]
    scaled = [_span_ms(p, sample_rate) for p in periods_for(version, sample_rate)]
    stock = [_span_ms(p, sample_rate) for p in unscaled]
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

    # Stated at 44.1 kHz, which is where the un-scaled set is worst, even
    # though RefineGAN no longer ships there -- this is a property of the
    # rule, and ``rate_scaled_periods`` is the rule.
    scaled = rate_scaled_periods(STOCK, 44100)
    stock_longest = max(_span_ms(p, 44100) for p in STOCK)
    scaled_longest = max(_span_ms(p, 44100) for p in scaled)
    assert stock_longest < 200.0 < 300.0 < scaled_longest


def test_the_resolutions_are_deliberately_not_scaled():
    """The frequency branches are pinned by a measurement and the period
    branches are not, which is the whole reason only one of the two families
    moves.  Scaling the resolutions would double the 50-sample hop, and that
    hop is the only thing in the discriminator that sees the frame-rate
    defect (68-76% against 50-54% for every alternative tried)."""

    for sample_rate, path in CONFIGS.items():
        model = json.loads(path.read_text())["model"]
        assert model.get("d_resolutions") is None
        built = MPD_MSD_Combined(
            model["use_spectral_norm"],
            version=model["d_version"],
            sample_rate=sample_rate,
        )
        # Same three resolutions at any rate -- unlike the periods.
        assert built.resolutions == tuple(
            tuple(r) for r in DISCRIMINATOR_VERSIONS[model["d_version"]][1]
        )
    hops = sorted(hop for _n_fft, hop, _win in DISCRIMINATOR_VERSIONS["v3"][1])
    assert hops[0] == 50
    # ``v4`` drops a period branch and nothing else, so it inherits the
    # same three resolutions -- the fine-hop one included.
    assert DISCRIMINATOR_VERSIONS["v4"][1] == DISCRIMINATOR_VERSIONS["v3"][1]


# --------------------------------------------------------------------------
# the configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_names_only_the_version(sample_rate):
    """The set is derived now, not written out.

    It used to be explicit in every config, because applying the scaling from
    ``_build_d_model`` would have re-periodised in-flight runs on their next
    resume, silently.  ``assert_periods_match`` closed that door, so the three
    derivable lines came back out of the config -- what is pinned here is that
    they are gone *and* that what gets built is still the reviewed set."""

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    for absent in ("d_periods", "d_resolutions", "d_frequency_strides"):
        assert absent not in model, f"{absent} is derivable from d_version"

    built = MPD_MSD_Combined(
        model["use_spectral_norm"],
        version=model["d_version"],
        sample_rate=sample_rate,
    )
    assert built.periods == periods_for(model["d_version"], sample_rate)


@pytest.mark.parametrize("version,sample_rate", VERSIONED)
def test_the_derivation_and_the_frozen_table_agree(version, sample_rate):
    """The table stopped being the source and became the pin.

    The constructor calls ``rate_scaled_periods``; the table is the reviewed
    artefact.  If these two ever disagree, either the rule drifted or the
    numbers were edited by hand, and both are worth failing over."""

    built = MPD_MSD_Combined(False, version=version, sample_rate=sample_rate)
    assert built.periods == periods_for(version, sample_rate)


def test_only_the_listed_versions_are_rate_scaled():
    """Membership in ``PERIODS_BY_RATE`` is what marks a version as scaled.

    ``v2``'s eight periods are Applio's and every RVC v2 pretrained
    discriminator is trained against them, so they must come out identical at
    every rate -- the parity is worth more than the alignment."""

    for version in ("v1", "v2"):
        assert version not in PERIODS_BY_RATE
        stock = tuple(DISCRIMINATOR_VERSIONS[version][0])
        for rate in (32000, 40000, 44100, 48000):
            assert MPD_MSD_Combined(False, version=version,
                                    sample_rate=rate).periods == stock

    # ...and the scaled ones really do move with the rate.
    assert (MPD_MSD_Combined(False, version="v4", sample_rate=32000).periods
            != MPD_MSD_Combined(False, version="v4", sample_rate=44100).periods)


def test_an_explicit_list_still_wins():
    """The knob did not become unreachable -- it became unnecessary.  An
    experiment resuming a pre-scaling run sets it and gets its old set back,
    which is exactly what ``assert_periods_match``'s message tells it to do."""

    built = MPD_MSD_Combined(False, version="v4", periods=[2, 3],
                             sample_rate=32000)
    assert built.periods == (2, 3)


def test_two_rates_do_not_get_the_same_set():
    """A single hard-coded list for every rate is the bug this replaces, so it
    is worth one line to pin that the rule actually separates them.

    Asked of ``rate_scaled_periods`` rather than the table, because the table
    now holds one rate: RefineGAN ships at 32 kHz only.  The property belongs
    to the rule regardless of what is currently selectable."""

    for version in sorted(PERIODS_BY_RATE):
        stock = tuple(DISCRIMINATOR_VERSIONS[version][0])
        assert rate_scaled_periods(stock, 32000) != rate_scaled_periods(stock, 44100)


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_config_builds_the_branches_it_names(sample_rate):
    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    built = MPD_MSD_Combined(
        model["use_spectral_norm"],
        version=model["d_version"],
        sample_rate=sample_rate,
    )
    expected = periods_for(model["d_version"], sample_rate)
    assert built.periods == expected
    assert [d.period for d in built.discriminators if hasattr(d, "period")] == list(
        expected
    )


def test_hifigan_keeps_applios_periods():
    """Deliberately not scaled: v2's eight periods are Applio's, every HiFi-GAN
    pretrain in circulation is trained against them, and the parity is worth
    more than the alignment."""

    assert "v2" not in PERIODS_BY_RATE
    for table in PERIODS_BY_RATE.values():
        assert 40000 not in table and 48000 not in table
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
    b = MPD_MSD_Combined(False, version="v3", periods=periods_for("v3", 32000))
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not missing and not unexpected
    # ...and it stays wrong, silently: the branches now fold at frequencies the
    # weights never saw, and nothing raised.
    assert b.periods != a.periods


def test_the_checkpoint_carries_the_periods():
    model = MPD_MSD_Combined(False, version="v3", periods=periods_for("v3", 32000))
    assert discriminator_periods(model) == list(periods_for("v3", 32000))
    assert_periods_match(
        model, {"discriminator_periods": list(periods_for("v3", 32000))}
    )


def test_a_mismatch_is_refused_and_says_what_to_do():
    model = MPD_MSD_Combined(False, version="v3", periods=periods_for("v3", 32000))
    with pytest.raises(ValueError, match="period mismatch") as excinfo:
        assert_periods_match(model, {"discriminator_periods": list(STOCK)})
    assert "d_periods" in str(excinfo.value)
    assert str(list(STOCK)) in str(excinfo.value)


def test_a_checkpoint_without_the_key_is_read_as_the_un_scaled_set():
    """"Absent" is not "nothing to check": every checkpoint predating the key
    is one of the runs this change would re-periodise, which is exactly the
    case the guard exists for.  Waving them through would make it useless."""

    scaled = MPD_MSD_Combined(False, version="v3", periods=periods_for("v3", 32000))
    with pytest.raises(ValueError, match="period mismatch"):
        assert_periods_match(scaled, {})
    # A run that kept the stock set resumes from such a checkpoint untouched,
    # which is what keeps every in-flight run working.
    stock = MPD_MSD_Combined(False, version="v3", periods=STOCK)
    assert_periods_match(stock, {})


def test_an_unkeyed_checkpoint_is_read_against_this_run_s_own_version():
    """The un-scaled set to compare against is the *version's*, not v3's.

    A stock HiFi-GAN run builds ``v2``, whose eight periods are Applio's, and
    the bundled RVC v2 pretrained discriminators are trained against exactly
    those and carry no ``discriminator_periods`` key.  Charging them with v3's
    five rejected every default fine-tune before it began.
    """

    # ``v2`` is never scaled, so a bundled RVC v2 pretrained discriminator --
    # which carries no key -- still resumes untouched.  This is the case the
    # fallback exists for and it must keep working.
    stock_v2 = DISCRIMINATOR_VERSIONS["v2"][0]
    model = MPD_MSD_Combined(False, version="v2")
    assert discriminator_periods(model) == list(stock_v2)
    assert_periods_match(model, {}, origin="pretrained discriminator")

    # ``v3`` is scaled, and since the periods are derived rather than written
    # into the config, a *default* v3 run now builds the scaled set where the
    # pre-scaling run built Applio's.  So the unkeyed checkpoint is refused --
    # correctly, because it really was trained on the other set -- and the
    # message names the key to write into the experiment's config to get it
    # back.  That is the price of deriving them, paid loudly.
    scaled = MPD_MSD_Combined(False, version="v3", sample_rate=44100)
    assert discriminator_periods(scaled) != list(DISCRIMINATOR_VERSIONS["v3"][0])
    with pytest.raises(ValueError, match="period mismatch") as excinfo:
        assert_periods_match(scaled, {}, origin="pretrained discriminator")
    assert "d_periods" in str(excinfo.value)


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
