"""Step-denominated schedules against the length of the run they schedule.

Two bugs are pinned here, and they are the same bug seen from two ends.

The first: the adversarial ceiling's ramp is a counter built fresh on every
process start, so resuming a pretrain reset it to zero and dropped the ceiling
back to its starting value -- the run re-earned a ramp it had already earned,
and spent thousands of steps under *less* adversarial pressure than the step
before the resume.

The second: the same literals that are a sane 2% of a pretrain are the whole of
a fine-tune.  At one holdout evaluation per 2000 steps, an 8k fine-tune gets
four of them against a patience of eight, so the overtrain detector cannot fire
even in principle -- and a fine-tune is where overtraining is most likely,
because the dataset is smallest.

``train.py`` reads a run spec from ``sys.argv[1]`` at import, so it is parsed
with ``ast`` and the pieces under test are exec'd in isolation.  That is the
same approach ``test_run_spec.py`` takes.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WANTED = {
    "planned_step_count",
    "fit_schedule",
    "fit_eval_interval",
    "_AdversarialCeilingGovernor",
    "HEALTHY_DISC_DRIFT_PER_10K",
    "_R1StrengthController",
}


@pytest.fixture(scope="module")
def train():
    """The handful of names under test, lifted out of ``train.py``."""
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
            kept.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANTED for t in node.targets
        ):
            kept.append(node)
    names = {
        node.name if hasattr(node, "name") else node.targets[0].id for node in kept
    }
    assert names == WANTED, f"train.py no longer defines {sorted(WANTED - names)}"
    namespace: dict = {"math": math}
    exec(compile(ast.Module(body=kept, type_ignores=[]), "train.py", "exec"), namespace)
    return namespace


class _Loader(list):
    """Stands in for a DataLoader; only ``len`` is ever consulted."""

    def __init__(self, batches: int):
        super().__init__(range(batches))


# The run this was measured on: 62 epochs over 7000 batches.
PRETRAIN_STEPS = 62 * 7000


def test_planned_steps_is_epochs_times_batches(train):
    assert train["planned_step_count"](62, _Loader(7000)) == PRETRAIN_STEPS


def test_max_steps_caps_the_budget(train):
    assert train["planned_step_count"](62, _Loader(7000), max_steps=8000) == 8000


def test_unknown_budget_reads_as_zero(train):
    assert train["planned_step_count"](0, _Loader(7000)) == 0


def test_pretrain_schedules_are_untouched(train):
    """The whole safety argument: a long run gets exactly what it configured.

    If this fails, fitting has started re-tuning a recipe that already works.
    """
    fit = train["fit_schedule"]
    assert fit(16000, PRETRAIN_STEPS, 0.2) == 16000
    assert fit(64000, PRETRAIN_STEPS, 0.5) == 64000


def test_a_schedule_longer_than_its_run_is_shrunk(train):
    fit = train["fit_schedule"]
    assert fit(2000, 4000, 0.25) == 1000
    assert fit(64000, 8000, 0.5) == 4000


def test_fitting_never_grows_a_schedule(train):
    assert train["fit_schedule"](500, PRETRAIN_STEPS, 0.5) == 500


def test_unknown_budget_leaves_schedules_alone(train):
    assert train["fit_schedule"](64000, 0, 0.5) == 64000


def test_detector_stays_inert_without_fitting():
    """The bug itself, stated as arithmetic rather than as a claim."""
    assert 8000 // 2000 < 8


def test_finetune_interval_fits_enough_evaluations(train):
    fitted = train["fit_eval_interval"](2000, 8000, patience=8)
    assert 8000 // fitted >= 3 * 8


def test_pretrain_interval_is_untouched(train):
    assert train["fit_eval_interval"](2000, PRETRAIN_STEPS, patience=8) == 2000


def test_eval_interval_never_grows(train):
    assert train["fit_eval_interval"](200, 8000, patience=8) == 200


def _governor(train, **kwargs):
    defaults = dict(
        start_step=16000,
        ramp_steps=64000,
        ceiling_start=1.0,
        ceiling_end=6.0,
        floor_loss=2.3695,
    )
    defaults.update(kwargs)
    return train["_AdversarialCeilingGovernor"](**defaults)


# Comfortably above the 0.08 headroom floor for a floor_loss of 2.3695: the
# measured value at the step this resume scenario is taken from.
HEALTHY_DISC_LOSS = 2.241


def test_resume_picks_the_ramp_up_where_it_left_off(train):
    governor = _governor(train)
    assert governor.ceiling == pytest.approx(1.0), "a governor starts at the floor"
    ceiling = governor.update(step=168000, loss_disc=HEALTHY_DISC_LOSS)
    assert ceiling == pytest.approx(6.0), (
        "resuming past the end of the ramp must restore the ceiling, not "
        "re-ramp from the start"
    )


def test_resume_midway_lands_midway(train):
    governor = _governor(train)
    governor.update(step=48000, loss_disc=HEALTHY_DISC_LOSS)
    # Half the ramp consumed, cosine-eased: 1.0 + 5.0 * 0.5.
    assert governor.ceiling == pytest.approx(3.5, abs=0.01)


def test_a_fresh_pretrain_still_waits_out_its_delay(train):
    governor = _governor(train)
    for step in range(0, 200):
        governor.update(step=step, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.ceiling == pytest.approx(1.0), (
        "seeding must not hand a from-scratch run a ramp it has not earned"
    )


def test_a_fresh_finetune_ramps_from_the_start(train):
    """Fine-tuning passes ``phase_step``, which is 0 on its first batch."""
    governor = _governor(train, start_step=0, ramp_steps=2000)
    governor.update(step=0, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.ceiling == pytest.approx(1.0)
    for step in range(1, 2001):
        governor.update(step=step, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.ceiling == pytest.approx(6.0)


def test_seeding_happens_once(train):
    governor = _governor(train)
    governor.update(step=168000, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.seeded
    governor.update(step=1, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.progress == pytest.approx(64000), (
        "a later, smaller step must not re-seed the ramp"
    )


def test_one_unlucky_first_batch_does_not_rewind_the_ramp(train):
    """The regression the 187013 resume produced.

    The slow EMA's time constant is 10k steps, so the first batch of a resumed
    run dominates ``headroom`` for a very long time.  A first batch reading high
    -- normal right after a resume -- put ``headroom`` under the 0.08 floor, and
    the backstop then rewound a ramp that nothing was wrong with: measured,
    ``ceiling_holding`` pinned at 1.0 and the ceiling walking 3.0 -> 2.82 while
    the true headroom was 0.1117.
    """
    governor = _governor(train, ceiling_end=3.0)
    governor.update(step=187013, loss_disc=2.318)  # an unlucky first batch
    for step in range(187014, 187014 + 2500):
        governor.update(step=step, loss_disc=HEALTHY_DISC_LOSS)
    assert not governor.holding, "a healthy discriminator must not read as dead"
    assert governor.ceiling == pytest.approx(3.0), "the ramp must not have rewound"


def test_warmup_is_counted_in_observations_not_global_step(train):
    governor = _governor(train)
    governor.update(step=187013, loss_disc=HEALTHY_DISC_LOSS)
    assert governor.observations == 1
    # A fresh governor at step 187013 is one batch old, not 187013 batches old,
    # so its EMAs are still inside their warmup.
    assert governor.observations < governor.warmup_steps


def test_the_warmup_estimate_is_the_mean_of_what_it_saw(train):
    governor = _governor(train)
    for value in (2.0, 2.2, 2.4, 2.6):
        governor.update(step=187013, loss_disc=value)
    assert governor.slow == pytest.approx(2.3)
    assert governor.fast == pytest.approx(2.3)


def test_a_dead_discriminator_still_retreats_after_the_warmup(train):
    """Seeding is optimistic; the backstop is what makes that safe.

    This used to assert the backstop fired on the *first* call after a resume,
    on the reasoning that seeding the EMAs from that batch made the headroom
    immediately true.  It does not -- one batch is one sample, and acting on it
    is what rewound a healthy ramp at the 187013 resume.  The contract is now
    that a discriminator which is *sustainedly* dead is caught once there are
    enough samples to tell it apart from an unlucky batch.
    """
    governor = _governor(train)
    dead = 2.3721  # exactly the constant-discriminator loss: headroom 0.
    governor.update(step=168000, loss_disc=dead)
    assert not governor.holding, "one sample is not evidence"
    for step in range(168001, 168001 + governor.warmup_steps):
        governor.update(step=step, loss_disc=dead)
    assert governor.holding, "a dead discriminator must not read as healthy"
    assert governor.progress < 64000, "the ramp must give ground once it is sure"


#: Per step, from the 2026-08-24 44.1 kHz pretrain: ``loss_disc`` rose 0.042 per
#: 10k while ``holdout/mel_l1`` was still setting records and headroom sat at
#: 0.289, 3.6x the collapse floor.  Above ``HEALTHY_DISC_DRIFT_PER_10K`` of 0.03,
#: so the trend gate fires on it -- which is the point of these two tests.
MEASURED_HEALTHY_DRIFT = 0.042 / 10_000


def _drift(governor, start_loss, steps, first_step=16001):
    for offset in range(steps):
        governor.update(
            step=first_step + offset,
            loss_disc=start_loss + MEASURED_HEALTHY_DRIFT * offset,
        )


def test_a_healthy_drift_far_from_the_floor_does_not_hold_the_ramp(train):
    """The regression that froze the 2026-08-24 pretrain's ceiling.

    Two EMAs following a linear drift settle at a fixed separation, so a drift
    faster than the one ``tolerance`` was derived from holds the ramp *forever*
    rather than briefly.  Measured: ``ceiling_holding`` pinned at 1.0 from step
    50k, the ceiling frozen at 1.803 where an unheld ramp would have reached
    2.857, while the balance rule asked for up to 2.52 and was clipped 28%.

    The drift here is deliberately above the configured healthy rate.  Pinning
    it any lower would make this test pass for the reason the old code passed --
    a well-chosen constant -- and this is the third constant to have been
    outgrown.  What must hold is that headroom this far from the floor makes the
    drift irrelevant.
    """
    governor = _governor(train, ceiling_end=3.0)
    _drift(governor, start_loss=2.0, steps=40_000)  # headroom 0.370 -> 0.202
    assert governor.headroom > governor.trend_gate_headroom
    assert not governor.holding, (
        "a discriminator at 2.5x the collapse floor is not on its way to the "
        "floor, so its drift carries nothing the backstop does not have"
    )
    assert governor.progress > 39_900, "the ramp must advance at the full rate"


def test_the_same_drift_still_holds_once_the_floor_is_in_reach(train):
    """The early warning has to survive where it can actually be right.

    Same drift, same tolerance -- only the distance to the floor differs.  This
    is the half of the contract that keeps the fix above from being a plain
    deletion of the trend gate.
    """
    governor = _governor(train, ceiling_end=3.0)
    _drift(governor, start_loss=2.24, steps=20_000)  # headroom 0.130 -> 0.098
    assert governor.headroom > governor.headroom_floor, "not dead, just close"
    assert governor.headroom < governor.trend_gate_headroom
    assert governor.holding, "the approach to the floor is what the gate is for"
    assert governor.progress < 20_000, "and holding must cost the ramp its steps"


# The measured 44.1 kHz pretrain, sampled at 30k/40k/50k/60k/69k: the
# discriminative gradient norm decays while the R1 penalty's own norm grows, so
# a fixed ``r1_gamma`` is a rising share of the discriminator's movement.
MEASURED_DISC_NORMS = (5.03, 3.88, 3.19, 2.81, 2.47)
MEASURED_R1_NORMS = (5.75, 7.24, 8.91, 15.59, 10.48)
R1_INTERVAL = 16


def _r1_share(disc_norm, r1_norm, interval=R1_INTERVAL):
    """R1's fraction of the gradient the discriminator moves on per cycle."""
    return r1_norm / (interval * disc_norm + r1_norm)


def test_the_measured_run_really_did_let_r1_take_over(train):
    """The observation the controller exists for, pinned as arithmetic.

    If this ever stops holding, the controller is solving a problem that is not
    there and the rest of these tests are measuring nothing.
    """
    shares = [
        _r1_share(disc, r1)
        for disc, r1 in zip(MEASURED_DISC_NORMS, MEASURED_R1_NORMS)
    ]
    assert shares[0] == pytest.approx(0.067, abs=0.005), "6.7% at step 30k"
    assert shares[-1] == pytest.approx(0.210, abs=0.005), "21% at step 69k"
    assert shares == sorted(shares[:-1]) + [shares[-1]], "and it climbed"


#: The nine branches, in ``ChouwaGANDiscriminator.branch_names`` order.
BRANCHES = (
    "period_2", "period_3", "period_5", "period_7", "period_11",
    "stft_512", "stft_1024", "stft_2048", "subband_8",
)

#: Measured 2026-08-24 by calling ``r1_penalty`` per branch on identical audio.
#: The R1 parameter-gradient norm spans 47x across branch types, which is what a
#: single global scale cannot serve.
MEASURED_BRANCH_R1 = {
    "period_2": 19.23, "period_3": 12.17, "period_5": 2.69,
    "period_7": 30.92, "period_11": 12.77,
    "stft_512": 45.93, "stft_1024": 37.81, "stft_2048": 54.71,
    "subband_8": 0.98,
}

BRANCH_PARAMS = {
    **{n: 417_137 for n in BRANCHES if n.startswith("period")},
    **{n: 125_185 for n in BRANCHES if n.startswith("stft")},
    "subband_8": 167_937,
}


def _controller(train, **kwargs):
    kwargs.setdefault("branch_names", BRANCHES)
    kwargs.setdefault("target_ratio", 1.0)
    return train["_R1StrengthController"](**kwargs)


def _drive(controller, disc_norms, r1_norms, cycles):
    """Run ``cycles`` full rotations of the real schedule.

    One R1 event lands on one branch per ``r1_interval`` steps, so a branch's own
    penalty fires every ``r1_interval * len(BRANCHES)`` steps -- while every
    branch's discriminative reference is sampled on each event.  ``r1_norms`` is
    what each penalty would produce at scale 1.0, so the actuator is applied to
    it and the feedback loop stays closed.
    """
    for _ in range(cycles):
        for branch in controller.names:
            for name in controller.names:
                controller.observe_discriminator(name, disc_norms[name])
            controller.observe_r1(
                branch, r1_norms[branch] * controller.scale_for(branch)
            )


def test_one_global_scale_cannot_serve_these_branches(train):
    """The observation the per-branch split exists for, stated as arithmetic.

    Per *parameter* -- the branches differ in size too -- the STFT stack takes
    9.9x the R1 update the period branches take and 63x the sub-band's.  If this
    stops holding, the split is solving a problem that is not there and the rest
    of these tests are measuring nothing.
    """
    per_param = {n: MEASURED_BRANCH_R1[n] / BRANCH_PARAMS[n] for n in BRANCHES}
    period = sum(per_param[n] for n in BRANCHES if n.startswith("period")) / 5
    stft = sum(per_param[n] for n in BRANCHES if n.startswith("stft")) / 3
    assert stft / period == pytest.approx(9.9, abs=0.5)
    assert stft / per_param["subband_8"] == pytest.approx(63.0, abs=3.0)


def test_each_branch_reaches_the_target_on_its_own(train):
    """The fix: nine independent shares, not one average of nine.

    Driven with the measured spread, which one scale can only satisfy on
    average -- leaving the STFT branches over-flattened and the sub-band barely
    regularised at the same instant.
    """
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 5.0), MEASURED_BRANCH_R1, cycles=600)
    for name in BRANCHES:
        assert controller.ratio_for(name) == pytest.approx(1.0, abs=0.15), (
            f"{name} settled at {controller.ratio_for(name)}"
        )
    scales = [controller.scale_for(n) for n in BRANCHES]
    assert max(scales) / min(scales) > 10.0, (
        "branches needing 47x different strengths must not end up together"
    )


def test_a_branch_is_measured_against_its_own_gradient(train):
    """Not against the ensemble total, which is what the global version used.

    Two branches with identical R1 penalties but different discriminative
    gradients are in different states and have to be corrected differently.
    """
    controller = _controller(train)
    disc = dict.fromkeys(BRANCHES, 5.0)
    disc["stft_2048"] = 0.5  # same penalty, a tenth of the discriminative signal
    _drive(controller, disc, dict.fromkeys(BRANCHES, 10.0), cycles=600)
    assert controller.scale_for("stft_2048") < 0.5 * controller.scale_for("period_2"), (
        "the branch with less gradient of its own must be flattened less"
    )
    for name in BRANCHES:
        assert controller.ratio_for(name) == pytest.approx(1.0, abs=0.15)


def test_the_actuator_moves_both_ways(train):
    """A stabiliser that can only weaken is a slow deletion of itself."""
    controller = _controller(train)
    disc = dict.fromkeys(BRANCHES, 3.0)
    _drive(controller, disc, dict.fromkeys(BRANCHES, 30.0), cycles=600)
    weakened = controller.scale_for("period_2")
    assert weakened < 1.0, "an over-strong penalty must be backed off"
    _drive(controller, disc, dict.fromkeys(BRANCHES, 0.05), cycles=600)
    assert controller.scale_for("period_2") > weakened, "and a vanishing one restored"


def test_the_strength_is_bounded_per_branch(train):
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 1e-3), dict.fromkeys(BRANCHES, 1e6), cycles=3000)
    assert controller.scale_for("period_2") == pytest.approx(controller.minimum, rel=1e-6)
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 1e6), dict.fromkeys(BRANCHES, 1e-6), cycles=3000)
    assert controller.scale_for("period_2") == pytest.approx(controller.maximum, rel=1e-6)


def test_a_zero_target_is_the_old_fixed_gamma(train):
    """The escape hatch, and what ``training_loop`` falls back to."""
    controller = _controller(train, target_ratio=0.0)
    assert not controller.active
    _drive(controller, dict.fromkeys(BRANCHES, 2.47), MEASURED_BRANCH_R1, cycles=600)
    for name in BRANCHES:
        assert controller.scale_for(name) == pytest.approx(1.0), (
            "a disabled controller must leave r1_gamma exactly where it was"
        )


def test_one_spiky_event_cannot_swing_a_branch(train):
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 2.47), dict.fromkeys(BRANCHES, 10.48), cycles=300)
    settled = controller.log_scale["period_2"]
    neighbour = controller.log_scale["period_3"]
    controller.observe_discriminator("period_2", 2.47)
    controller.observe_r1("period_2", 1e9)
    assert abs(controller.log_scale["period_2"] - settled) <= controller.lr + 1e-9
    assert controller.log_scale["period_3"] == pytest.approx(neighbour, abs=1e-12), (
        "and it must not reach any other branch"
    )


def test_the_strengths_survive_a_resume(train):
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 5.0), MEASURED_BRANCH_R1, cycles=600)
    restored = _controller(train)
    assert restored.load_state_dict(controller.state_dict)
    for name in BRANCHES:
        assert restored.scale_for(name) == pytest.approx(controller.scale_for(name))
        assert restored.ratio_for(name) == pytest.approx(controller.ratio_for(name))


def test_a_flat_checkpoint_seeds_every_branch(train):
    """The layout written before the split, which applied one scale to all nine.

    Seeding every branch from it restores the state the run was actually in.
    The scalar EMAs are dropped on purpose: they were norms over the *whole*
    discriminator, so carrying them would seed every per-branch ratio wrong.
    """
    controller = _controller(train)
    assert controller.load_state_dict(
        {"log_scale": math.log(0.25), "disc_ema": 2.47, "r1_ema": 10.48}
    )
    for name in BRANCHES:
        assert controller.scale_for(name) == pytest.approx(0.25)
        assert controller.ratio_for(name) == 0.0, "stale ensemble EMAs must not carry"


def test_a_branch_absent_from_the_checkpoint_starts_cold(train):
    """``periods`` changed, or the sub-band branch was switched on mid-project."""
    controller = _controller(train)
    _drive(controller, dict.fromkeys(BRANCHES, 5.0), MEASURED_BRANCH_R1, cycles=600)
    state = controller.state_dict
    state["log_scale"].pop("stft_2048")
    state["r1_ema"].pop("stft_2048")
    restored = _controller(train)
    assert restored.load_state_dict(state)
    assert restored.scale_for("stft_2048") == pytest.approx(1.0), (
        "an unknown branch must not inherit another branch's strength"
    )
    assert restored.scale_for("period_2") == pytest.approx(
        controller.scale_for("period_2")
    )


def test_no_checkpoint_starts_cold(train):
    controller = _controller(train)
    assert not controller.load_state_dict(None)
    assert not controller.load_state_dict({})
    assert controller.scale == pytest.approx(1.0)
