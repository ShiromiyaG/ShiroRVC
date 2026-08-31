"""The balance rule's gradient probe runs outside the ``GradScaler``.

The 44.1 kHz FP16 pretrain never took a single generator step.  What the
TensorBoard run recorded is the whole shape of it:

    Grad_Norm_Diag/G_Skipped   1.0 on every one of 65 steps
    AMP/skip_rate_50           1.0
    AMP/grad_scaler_scale      2**10 -> ~2**-40, halved once per step
    loss_disc_50               2.3543   (the discriminator trained fine)
    loss_adv 1.19  loss_fm 0.134  loss_spectral 72.90  loss_waveform 4.71
    loss_gen_total_50          nan      <- only the total
    GAN/last_layer_rec_grad    nan
    GAN/adaptive_adv_weight    nan

Every loss component is finite and only the weights are NaN, which places the
fault in ``_adaptive_adversarial_weight`` rather than in the model.  That helper
measures gradient norms with ``torch.autograd.grad``, and the scaler covers only
what goes through ``scale(...).backward()`` -- so under autocast the probe
traverses the decoder in FP16 with no overflow protection at all.  A
reconstruction loss of ~78 overflows the half range on the way back, ``inf``
becomes ``NaN``, and the ratio of two NaN norms is NaN.  That NaN is then cached
for ``adaptive_adv_interval`` steps and multiplied into ``loss_gen_total``, so
the run cannot recover: the scaler lowering its scale does not help a
measurement the scaler never touched.

The fix rests on the rule being a *ratio*: ``||d(s*L)/dp|| / s`` is the same
number at every ``s``, so each term is measured through whatever scale keeps it
representable.  What this file pins is that property -- a probe that survives
both ends of the half range, and a rule that answers with a number rather than
a NaN when it does.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the probe differentiates", exc_type=ImportError)

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"

# 300 * 300 = 90000 overflows FP16 (max 65504), and 1e-4 * 1e-4 = 1e-8 falls
# under its smallest subnormal (~6e-8).  Both are reached by the chain rule, so
# the *loss* is representable and only the gradient is not -- which is exactly
# the run's signature.
OVERFLOWING = 300.0
VANISHING = 1e-4


@pytest.fixture(scope="module")
def lifted() -> dict:
    """Lift the helpers out of ``train.py``, whose module body reads argv."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    wanted = {
        "_PROBE_SCALE_ATTEMPTS",
        "_PROBE_SCALE_STEP",
        "_probe_gradient",
        "_adaptive_adversarial_weight",
        "_adaptive_feature_match_weight",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
            wanted.discard(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        ):
            body.append(node)
            wanted.difference_update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    if wanted:
        pytest.fail(f"train.py no longer defines {sorted(wanted)}")
    namespace: dict = {"torch": torch}
    exec(compile(ast.Module(body, []), str(TRAIN_PY), "exec"), namespace)
    return namespace


def _half_loss(factor: float):
    """An FP16 loss whose gradient at ``parameter`` is ``factor ** 2``."""
    parameter = torch.ones(1, dtype=torch.float16, requires_grad=True)
    return parameter, ((parameter * factor) * factor).sum()


def _grad(loss, parameter):
    return lambda scale: torch.autograd.grad(
        loss * scale, parameter, retain_graph=True, allow_unused=True
    )[0]


def test_the_unscaled_probe_really_does_overflow():
    """The premise.  Without this the rest of the file proves nothing."""
    parameter, loss = _half_loss(OVERFLOWING)
    gradient = torch.autograd.grad(loss, parameter)[0]
    assert not torch.isfinite(gradient).all(), (
        "FP16 no longer overflows here, so the regression this file pins is gone"
    )


def test_the_probe_measures_through_an_overflow(lifted):
    parameter, loss = _half_loss(OVERFLOWING)
    _, norm = lifted["_probe_gradient"](_grad(loss, parameter))

    assert torch.isfinite(norm).all()
    assert norm.item() == pytest.approx(OVERFLOWING**2, rel=1e-3)


def test_the_probe_measures_through_an_underflow(lifted):
    """The other end of the range, and the reason the scale moves both ways."""
    parameter, loss = _half_loss(VANISHING)
    assert torch.autograd.grad(loss, parameter, retain_graph=True)[0].item() == 0.0

    _, norm = lifted["_probe_gradient"](_grad(loss, parameter))
    assert norm.item() == pytest.approx(VANISHING**2, rel=1e-2)


def test_a_genuinely_zero_term_stays_zero(lifted):
    """A term that does not move the parameter must not be scaled into noise."""
    parameter = torch.ones(1, dtype=torch.float16, requires_grad=True)
    loss = (parameter * 0.0).sum()
    _, norm = lifted["_probe_gradient"](_grad(loss, parameter))
    assert norm.item() == 0.0


def test_the_scale_leaves_the_measurement_alone_in_fp32(lifted):
    """The rule is a ratio, so the scale is free -- but only if it is invisible
    where it was never needed.  In FP32 the first attempt succeeds and nothing
    about the reported number moves."""
    parameter = torch.ones(4, requires_grad=True)
    loss = (parameter * 3.0).sum()
    reference = torch.autograd.grad(loss, parameter, retain_graph=True)[0].norm()

    _, norm = lifted["_probe_gradient"](_grad(loss, parameter))
    assert norm.item() == reference.item()


def test_the_adversarial_weight_is_a_number_not_a_nan(lifted):
    """The failure as the run met it: both terms overflow, and the old code
    returned ``inf / inf`` -- a NaN that was cached and multiplied into
    ``loss_gen_total``."""
    parameter = torch.ones(1, dtype=torch.float16, requires_grad=True)
    reconstruction = ((parameter * OVERFLOWING) * OVERFLOWING).sum()
    adversarial = ((parameter * OVERFLOWING) * (OVERFLOWING * 0.5)).sum()

    weight, rec_norm, adv_norm, requested = lifted["_adaptive_adversarial_weight"](
        reconstruction, adversarial, parameter, balance_target=0.5
    )

    for value in (weight, rec_norm, adv_norm, requested):
        assert torch.isfinite(value).all(), "the probe still produces NaN weights"
    # Reconstruction is twice the adversarial gradient, so the rule asks for
    # twice the target and the ceiling refuses it.
    assert requested.item() == pytest.approx(1.0, rel=1e-2)
    assert weight.item() == pytest.approx(1.0, rel=1e-3)


def test_the_feature_match_weight_is_a_number_not_a_nan(lifted):
    parameter, loss = _half_loss(OVERFLOWING)
    weight, norm, requested = lifted["_adaptive_feature_match_weight"](
        loss,
        parameter,
        torch.tensor(float(OVERFLOWING**2)),
        balance_target=0.33,
    )

    for value in (weight, norm, requested):
        assert torch.isfinite(value).all()
    assert norm.item() == pytest.approx(OVERFLOWING**2, rel=1e-3)
    assert weight.item() == pytest.approx(0.33, rel=1e-2)


def test_a_failed_probe_never_reaches_the_cached_weight():
    """The guard, as source text.  A weight is reused for
    ``adaptive_adv_interval`` steps, so one non-finite value entering the cache
    is not one bad step -- it is a run that never takes another one."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    assert "probe_finite" in source, "the cache no longer checks the probe"
    assert "if probe_finite:\n                            cached_adaptive_adv" in source, (
        "the cache is being written without the finiteness guard"
    )
    assert '"GAN/adaptive_probe_failures"' in source, (
        "a failed probe has to stay visible; it was silent once already"
    )
