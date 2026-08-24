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
