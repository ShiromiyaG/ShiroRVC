"""Step-denominated schedules against the length of the run they schedule.

The bug pinned here: the same literals that are a sane 2% of a pretrain are the
whole of a fine-tune.  At one holdout evaluation per 2000 steps, an 8k
fine-tune gets four of them against a patience of eight, so the overtrain
detector cannot fire even in principle -- and a fine-tune is where
overtraining is most likely, because the dataset is smallest.

The fitting functions live in ``rvc/train/schedules.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="module")
def train():
    """The module under test."""
    return pytest.importorskip("rvc.train.schedules")


class _Loader(list):
    """Stands in for a DataLoader; only ``len`` is ever consulted."""

    def __init__(self, batches: int):
        super().__init__(range(batches))


# The run this was measured on: 62 epochs over 7000 batches.
PRETRAIN_STEPS = 62 * 7000


def test_planned_steps_is_epochs_times_batches(train):
    assert train.planned_step_count(62, _Loader(7000)) == PRETRAIN_STEPS


def test_max_steps_caps_the_budget(train):
    assert train.planned_step_count(62, _Loader(7000), max_steps=8000) == 8000


def test_unknown_budget_reads_as_zero(train):
    assert train.planned_step_count(0, _Loader(7000)) == 0


def test_pretrain_schedules_are_untouched(train):
    """The whole safety argument: a long run gets exactly what it configured.

    If this fails, fitting has started re-tuning a recipe that already works.
    """
    fit = train.fit_schedule
    assert fit(16000, PRETRAIN_STEPS, 0.2) == 16000
    assert fit(64000, PRETRAIN_STEPS, 0.5) == 64000


def test_a_schedule_longer_than_its_run_is_shrunk(train):
    fit = train.fit_schedule
    assert fit(2000, 4000, 0.25) == 1000
    assert fit(64000, 8000, 0.5) == 4000


def test_fitting_never_grows_a_schedule(train):
    assert train.fit_schedule(500, PRETRAIN_STEPS, 0.5) == 500


def test_unknown_budget_leaves_schedules_alone(train):
    assert train.fit_schedule(64000, 0, 0.5) == 64000


def test_detector_stays_inert_without_fitting():
    """The bug itself, stated as arithmetic rather than as a claim."""
    assert 8000 // 2000 < 8


def test_finetune_interval_fits_enough_evaluations(train):
    fitted = train.fit_eval_interval(2000, 8000, patience=8)
    assert 8000 // fitted >= 3 * 8


def test_pretrain_interval_is_untouched(train):
    assert train.fit_eval_interval(2000, PRETRAIN_STEPS, patience=8) == 2000


def test_eval_interval_never_grows(train):
    assert train.fit_eval_interval(200, 8000, patience=8) == 200

