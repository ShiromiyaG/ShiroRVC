"""Which weights the holdout monitor keeps, and when it calls a run finished.

Two bugs are pinned here, from opposite directions. First, a single
``min_delta`` used to gate both "is this a new best?" and "has the run
stopped improving?", even though those want opposite thresholds -- the
monitor now answers them separately. Second, once selection was ungated,
``best`` became a single-point argmin over a curve whose tail is flat and
noisy, so it could pick the luckiest evaluation instead of the best weights.
The monitor now median-filters the score and measures every decision against
a noise band estimated from recent history, so a tie inside that band goes to
the earlier step.

``train.py`` reads a run spec from ``sys.argv[1]`` at import, so the class is
lifted out with ``ast``, the same way ``test_run_spec.py`` reads that file.
"""

from __future__ import annotations

import ast
import math
import sys
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def monitor_class():
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_OvertrainMonitor"
        ),
        None,
    )
    assert node is not None, "train.py no longer defines _OvertrainMonitor"
    namespace: dict = {
        "math": math,
        "deque": deque,
        # The monitor's only dependency: it clones whatever it is handed.  What
        # matters for these tests is *which* object it cloned, not the tensors.
        "_cpu_state_dict": lambda source: {"scored": source},
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"),
        namespace,
    )
    return namespace["_OvertrainMonitor"]


# The tail of the 44.1 kHz pretrain, from the last delta-significant
# improvement to the end.  Flat and noisy: nothing here is evidence of
# anything.
REAL_TAIL = [
    (140000, 0.7025),
    (142000, 0.7056),
    (144000, 0.7046),
    (146000, 0.7053),
    (148000, 0.7093),
    (150000, 0.7063),
    (152000, 0.7036),
    (154000, 0.7049),
    (156000, 0.7044),
    (158000, 0.7056),
    (160000, 0.7124),
    (162000, 0.7049),
    (164000, 0.7039),
    (166000, 0.7021),
    (168000, 0.7048),
]

# ``andre-multi``, every evaluation of it: a clean descent, a minimum around
# step 6k, and a slow rise afterwards.  The run the smoothing was written
# against.
ANDRE_MULTI = [
    (362, 0.71902),
    (724, 0.70143),
    (1086, 0.69415),
    (1448, 0.68742),
    (1810, 0.68315),
    (2172, 0.68255),
    (2534, 0.68144),
    (2896, 0.67789),
    (3258, 0.67610),
    (3620, 0.67517),
    (3982, 0.67389),
    (4344, 0.67277),
    (4706, 0.67109),
    (5068, 0.67066),
    (5430, 0.66902),
    (5792, 0.66749),
    (6154, 0.66631),
    (6516, 0.66709),
    (6878, 0.66806),
    (7240, 0.66910),
    (7602, 0.66979),
    (7964, 0.67327),
    (8326, 0.67211),
    (8688, 0.67160),
    (9050, 0.67244),
]


def _run(monitor, values):
    for step, value in values:
        monitor.update(f"weights@{step}", value, step)
    return monitor


def test_a_lucky_evaluation_does_not_win_the_run(monitor_class):
    """A late step that beats the true best by a hair, on a flat noisy tail,
    must not win: inside the noise band they are the same measurement, and
    the earlier step is less overtrained."""
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.best_step == 140000
    assert monitor.sigma > (0.7025 - 0.7021)


def test_exported_weights_are_the_ones_that_scored_best(monitor_class):
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.state_dict == {"scored": "weights@140000"}


def test_a_flat_tail_is_called_overtrained(monitor_class):
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.since_progress == 14
    assert monitor.overtrained


def test_the_real_turn_is_found_and_called(monitor_class):
    """End-to-end regression on the run that motivated the rewrite: the
    smoothed minimum lands one evaluation earlier than the raw argmin, and the
    run is still called overtrained at the same point the unsmoothed monitor
    called it."""
    monitor = _run(monitor_class(patience=8, min_delta=0.001), ANDRE_MULTI)
    assert monitor.best_step == 5792
    assert monitor.overtrained
    assert monitor.history[-1][0] == 9050


def test_the_turn_is_not_called_early(monitor_class):
    """Nothing before the minimum may fire the detector."""
    for cut in range(4, 18):
        monitor = _run(monitor_class(patience=8, min_delta=0.001), ANDRE_MULTI[:cut])
        assert not monitor.overtrained, f"fired after {cut} evaluations"


def test_a_falling_curve_has_a_narrow_band(monitor_class):
    """The band is the spread about the *trend*, not about the mean.

    A run improving by 1% an evaluation has a huge spread and no noise; if the
    band could not tell those apart, a healthy descent would be read as noise
    and every improvement in it discarded.
    """
    falling = monitor_class(patience=8, min_delta=0.0)
    _run(falling, [(step, 0.9 - 0.01 * step) for step in range(8)])
    noisy = monitor_class(patience=8, min_delta=0.0)
    _run(noisy, [(step, 0.9 + (0.01 if step % 2 else -0.01)) for step in range(8)])
    assert falling.sigma < 1e-9
    assert noisy.sigma > 0.005


def test_a_real_improvement_resets_patience(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    _run(monitor, [(step, 0.7050) for step in range(1000, 7000, 1000)])
    assert monitor.since_progress > 0
    monitor.update("c", 0.6900, 9000)  # 2% better: unambiguous progress
    monitor.update("c", 0.6900, 10000)
    assert monitor.since_progress == 0


def test_patience_reference_never_walks_backwards(monitor_class):
    """A worse evaluation must not raise the bar for what counts as progress."""
    monitor = monitor_class(patience=8, min_delta=0.001)
    monitor.update("a", 0.70, 1000)
    monitor.update("b", 0.90, 2000)
    assert monitor.patience_reference == pytest.approx(0.70)


def test_a_short_run_is_judged_unsmoothed(monitor_class):
    """Before the window fills there is nothing to filter with.

    A fine-tune can be over in a handful of evaluations, and half a median is
    not a median -- on an even window it is whichever side the tie broke
    towards.
    """
    monitor = monitor_class(patience=8, min_delta=0.001, smoothing=3)
    assert monitor.update("a", 0.7025, 1000) is True
    assert monitor.best_step == 1000
    assert monitor.update("b", 0.6900, 2000) is True
    assert monitor.best_step == 2000
    assert monitor.smoothed == pytest.approx(0.6900)


def test_non_finite_scores_are_ignored(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    monitor.update("a", 0.70, 1000)
    assert monitor.update("b", float("nan"), 2000) is False
    assert monitor.since_progress == 0
    assert monitor.best_step == 1000


def test_nothing_is_exported_before_a_single_score(monitor_class):
    monitor = monitor_class(patience=1, min_delta=0.001)
    assert monitor.state_dict is None
    assert not monitor.overtrained


def test_only_candidate_weights_are_cloned(monitor_class):
    """A copy of the model per evaluation is the cost of a centred filter.

    Scores far enough above the best to have no path back under it are not
    cloned, which is what keeps the window from being three whole models.
    """
    monitor = monitor_class(patience=8, min_delta=0.001)
    _run(monitor, [(step, 0.70) for step in range(1000, 5000, 1000)])
    monitor.update("wild", 2.0, 5000)
    assert monitor._window[-1][2] is None
    assert monitor.state_dict == {"scored": "weights@1000"}


def test_backoff_only_ever_stretches_the_interval(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    assert monitor.interval_scale == 1
    assert monitor.backoff(4) == 4
    assert monitor.backoff(4) == 16
    assert monitor.backoff(0) == 16
