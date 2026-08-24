"""Which weights the holdout monitor keeps, and when it calls a run finished.

The bug pinned here cost the 44.1 kHz pretrain its best weights.  One
``min_delta`` gated both "is this a new best?" and "has the run stopped
improving?", and those want opposite thresholds.  The run's true minimum was
0.70210 at step 166k, but ``best`` was frozen at 0.70250 from step 140k because
the improvement was 0.06% against a required 0.1% -- and ``_deliverable_weights``
prefers the monitor's snapshot over the EMA and the live weights, so the run
would have exported a model it had itself measured as worse.

``train.py`` reads a run spec from ``sys.argv[1]`` at import, so the class is
lifted out with ``ast``, the same way ``test_run_spec.py`` reads that file.
"""

from __future__ import annotations

import ast
import math
import sys
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
        # The monitor's only dependency: it clones whatever it is handed.  What
        # matters for these tests is *which* object it cloned, not the tensors.
        "_cpu_state_dict": lambda source: {"scored": source},
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"),
        namespace,
    )
    return namespace["_OvertrainMonitor"]


# The tail of the real run, from the last delta-significant improvement to the
# end: a new true minimum at 166k that the old gate refused to record.
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


def _run(monitor, values):
    for step, value in values:
        monitor.update(f"weights@{step}", value, step)
    return monitor


def test_best_follows_the_true_minimum(monitor_class):
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.best == pytest.approx(0.7021)
    assert monitor.best_step == 166000


def test_exported_weights_are_the_ones_that_scored_best(monitor_class):
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.state_dict == {"scored": "weights@166000"}


def test_a_sub_delta_improvement_does_not_reset_patience(monitor_class):
    """The other half: the counter must still ignore improvements too small.

    0.7021 beats 0.7025 by 0.06%, under the 0.1% required, so it is a new best
    without being evidence that the run is still going anywhere.
    """
    monitor = _run(monitor_class(patience=8, min_delta=0.001), REAL_TAIL)
    assert monitor.since_best == 14
    assert monitor.overtrained


def test_a_real_improvement_resets_patience(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    monitor.update("a", 0.7025, 1000)
    for step in range(2000, 8000, 1000):  # 2000..7000: six evaluations
        monitor.update("b", 0.7050, step)
    assert monitor.since_best == 6
    monitor.update("c", 0.6900, 9000)  # 1.8% better: unambiguous progress
    assert monitor.since_best == 0
    assert monitor.best_step == 9000


def test_patience_reference_never_walks_backwards(monitor_class):
    """A worse evaluation must not raise the bar for what counts as progress."""
    monitor = monitor_class(patience=8, min_delta=0.001)
    monitor.update("a", 0.70, 1000)
    monitor.update("b", 0.90, 2000)
    assert monitor.patience_reference == pytest.approx(0.70)


def test_update_reports_a_new_best(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    assert monitor.update("a", 0.7025, 1000) is True
    assert monitor.update("b", 0.7056, 2000) is False
    # A new minimum, even a sub-delta one, is a new best and is marked as such.
    assert monitor.update("c", 0.7021, 3000) is True


def test_non_finite_scores_are_ignored(monitor_class):
    monitor = monitor_class(patience=8, min_delta=0.001)
    monitor.update("a", 0.70, 1000)
    assert monitor.update("b", float("nan"), 2000) is False
    assert monitor.since_best == 0
    assert monitor.best_step == 1000


def test_nothing_is_exported_before_a_single_score(monitor_class):
    monitor = monitor_class(patience=1, min_delta=0.001)
    assert monitor.state_dict is None
    assert not monitor.overtrained
