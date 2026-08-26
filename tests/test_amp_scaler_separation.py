"""The lazy R1 step needs its own GradScaler.

``GradScaler`` tracks state per optimizer per iteration: once ``step(opt)`` has
run, a further ``unscale_(opt)`` before ``update()`` raises ``unscale_() is
being called after step()``.  The discriminator update steps ``optim_d`` and the
lazy R1 block then steps it *again* inside the same training step, so sharing one
scaler between them crashed on the first R1 step of every FP16 run -- step 16 at
the default ``r1_interval``.

The first test pins PyTorch's actual behaviour, so the constraint is documented
rather than assumed.  The second reads ``train.py`` and pins that the R1 block
does not reach for the main scaler, which is the thing a future edit could undo.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the scaler needs torch", exc_type=ImportError)

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"


def _two_steps(first_scaler, second_scaler):
    """Two optimizer iterations on one optimizer, as the training step does."""
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    optimizer.zero_grad(set_to_none=True)
    first_scaler.scale(parameter.square().sum()).backward()
    first_scaler.unscale_(optimizer)
    first_scaler.step(optimizer)

    optimizer.zero_grad(set_to_none=True)
    second_scaler.scale(parameter.square().sum()).backward()
    second_scaler.unscale_(optimizer)
    second_scaler.step(optimizer)


def _scaler():
    # ``enabled=True`` on CPU exercises the same bookkeeping without CUDA.
    return torch.amp.GradScaler("cpu", enabled=True)


def test_one_scaler_cannot_step_the_same_optimizer_twice():
    scaler = _scaler()

    with pytest.raises(RuntimeError, match="after step"):
        _two_steps(scaler, scaler)


def test_two_scalers_can():
    _two_steps(_scaler(), _scaler())


def test_the_r1_block_never_touches_the_main_scaler():
    """Read the source: the crash was invisible to every test we had."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))

    r1_blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        source = ast.dump(node.test)
        if "r1_gamma" in source and "r1_interval" in source:
            r1_blocks.append(node)

    assert r1_blocks, "train.py no longer has an r1_gamma/r1_interval guarded block"
    for block in r1_blocks:
        used = {
            child.value.id
            for child in ast.walk(block)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id.endswith("grad_scaler")
        }
        assert used == {"r1_grad_scaler"}, (
            "the R1 step is a second optimizer iteration on optim_d and must use "
            f"its own scaler; found {sorted(used)}"
        )


def test_the_r1_scaler_is_checkpointed():
    """Restarting a scaler at ``init_scale`` replays the overflow search and
    hides a run that had settled much lower, which is why the main one is
    saved. The R1 scaler carries the same state for the same reason."""
    source = TRAIN_PY.read_text(encoding="utf-8")

    assert '"r1_grad_scaler"' in source
    assert 'resumed_extra_d.get("r1_grad_scaler")' in source
