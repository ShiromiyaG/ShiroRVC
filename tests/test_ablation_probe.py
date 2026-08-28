"""The latent-usage probe is instrumentation, and its gate has to say so.

``ablation_delta`` is the only signal that separates a dead latent dimension
from one the prior simply predicts well.  Per-dimension KL cannot: it measures
``posterior || prior`` with both learned, so a low value means "the prior
predicts it" at least as often as it means "nothing is there" -- measured over
the 44.1 kHz pretrain the two rank-correlate at 0.24.

The probe used to be gated on ``ablation_loss_weight > 0``, so
switching off a loss term that provably does nothing also switched off the
measurement.  Those are separate decisions and this pins them apart.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
CONFIGS = sorted((ROOT / "rvc" / "configs" / "chouwagan").glob("*.json"))


@pytest.fixture(scope="module")
def source():
    return TRAIN_PY.read_text(encoding="utf-8")


def _probe_guard(source: str) -> ast.If:
    """The ``if`` that decides whether the ablation probe runs this step."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "ablation_interval" in names:
            return node
    raise AssertionError("could not find the ablation probe guard in train.py")


def test_the_probe_does_not_depend_on_the_loss_weight(source):
    guard = _probe_guard(source)
    names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    assert "ablation_weight" not in names, (
        "gating the measurement on the loss weight means the only way to stop "
        "paying for an inert loss term is to go blind to latent usage"
    )


def test_the_probe_still_respects_its_interval(source):
    guard = _probe_guard(source)
    names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    assert "ablation_interval" in names
    assert "chouwa_latent" in names


def test_the_loss_is_still_scaled_by_its_weight(source):
    """Decoupling the probe must not make the loss unconditional."""
    assert "loss_ablation = ablation_raw * ablation_weight" in source


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_measure_but_do_not_weight(config_path):
    train = json.loads(config_path.read_text(encoding="utf-8"))["train"]
    assert train["ablation_loss_weight"] == 0.0, (
        "the term has no gradient toward its stated purpose: ablated_error is "
        "detached, so it only adds generic reconstruction pressure"
    )
    interval = train["ablation_interval"]
    assert interval > 0, "a weight of zero must not turn the measurement off"
    # 96 latent dimensions drawn uniformly, one per interval.  Each dimension is
    # therefore sampled about once per ``96 * interval`` steps, and a histogram
    # needs enough samples to average over a run of a few hundred thousand.
    assert 96 * interval <= 15000, (
        f"at interval {interval} each dimension is probed once per "
        f"{96 * interval} steps, too rarely to build a usage histogram"
    )
