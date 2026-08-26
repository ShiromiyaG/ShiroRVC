"""Feature matching had no governor, and that was the loophole.

``_adaptive_adversarial_weight`` holds ``loss_adv`` at a fixed share of the
reconstruction gradient.  ``loss_fm`` is computed from the same discriminator's
feature maps and grows with it, and it carried a fixed weight -- so the balance
rule restrained one half of the GAN objective while the other half was free.

Measured on the 44.1 kHz pretrain over its first 28k steps, as gradient into the
output waveform:

    step        spectral    adv (weighted)    fm      fm/spectral
        0           61.3             266.4   161.4           2.63
     8000           56.2             245.0   379.7           6.76
    16000           54.9             238.0   443.6           8.09
    28000           55.6             236.1   500.2           9.00

The governed term is flat by construction -- its weight fell 0.33 -> 0.079 to
keep it there -- and reconstruction is flat on its own.  Feature matching
tripled, and ended as the largest single source of gradient into the waveform.
Over the same window ``disc_headroom`` went 0.40 -> 0.79 and the latent adapters'
gradient rose in near-exact proportion to it: ``grad_norm_latent / loss_adv``
held between 5.0 and 5.8 the whole way, so the "the latent heads are getting
loud" symptom was this, seen from the other end.

What this file pins is the shape of the correction, not the numbers above: the
weight has to *fall* when feature matching outgrows reconstruction, hold the
configured target when it does not, and never amplify past the configured
weight.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the rule differentiates", exc_type=ImportError)

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
CONFIG = ROOT / "rvc" / "configs" / "refinegan" / "44100.json"


@pytest.fixture(scope="module")
def rule():
    """Lift the helper out of ``train.py``, whose module body reads argv."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_adaptive_feature_match_weight":
            namespace: dict = {"torch": torch}
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
            return namespace["_adaptive_feature_match_weight"]
    pytest.fail("train.py no longer defines _adaptive_feature_match_weight")


def _probe(fm_gradient_scale: float):
    """A parameter whose fm gradient norm is ``fm_gradient_scale``."""
    parameter = torch.ones(1, requires_grad=True)
    return parameter, (parameter * fm_gradient_scale).sum()


def test_the_weight_falls_as_feature_matching_outgrows_reconstruction(rule):
    reconstruction = torch.tensor(100.0)
    weights = []
    for fm_norm in (33.0, 66.0, 132.0, 264.0):
        parameter, loss = _probe(fm_norm)
        weight, measured, _ = rule(loss, parameter, reconstruction, balance_target=0.33)
        assert measured.item() == pytest.approx(fm_norm, rel=1e-5)
        weights.append(weight.item())

    assert weights == sorted(weights, reverse=True), "the correction must be monotone"
    # At the target ratio the governor is a no-op, which is what makes the
    # configured ``refinegan_fm_weight`` still mean something.
    assert weights[0] == pytest.approx(1.0, rel=1e-3)
    # And at the run's measured drift -- fm/rec doubling -- it halves.
    assert weights[1] == pytest.approx(0.5, rel=1e-3)


def test_it_can_only_reduce_never_amplify(rule):
    """The configured weight is the ceiling.  A quiet fm must not be boosted."""
    parameter, loss = _probe(1e-3)
    weight, _, requested = rule(loss, parameter, torch.tensor(100.0), balance_target=0.33)

    assert requested.item() > 1.0, "the raw rule would have asked for more"
    assert weight.item() == pytest.approx(1.0), "and the clamp has to refuse it"


def test_the_floor_bounds_a_runaway(rule):
    parameter, loss = _probe(1e6)
    weight, _, _ = rule(loss, parameter, torch.tensor(1.0), balance_target=0.33, minimum=0.1)
    assert weight.item() == pytest.approx(0.1)


def test_a_detached_feature_match_loss_is_a_no_op(rule):
    """``loss_fm`` is a plain zero on the paths that do not compute it, and
    differentiating that raises rather than returning nothing."""
    weight, measured, requested = rule(
        torch.zeros(()), torch.ones(1, requires_grad=True), torch.tensor(1.0), balance_target=0.33
    )
    assert weight.item() == pytest.approx(1.0)
    assert measured.item() == 0.0
    assert requested.item() == 0.0


def test_the_weight_reaches_the_loss_the_optimizer_sees():
    """A governor computed and then not applied is the failure mode with no
    symptom -- the series would look corrected while the gradient was not."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    assert "adaptive_adv * loss_adv + adaptive_fm * loss_fm" in source, (
        "loss_gan no longer applies the governed feature-match weight"
    )


def _reported_ratio(tag: str) -> str:
    """The expression ``writer.add_scalar(tag, ...)`` reports, as source text."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        name = getattr(node.func, "attr", None)
        first = node.args[0]
        if name == "add_scalar" and isinstance(first, ast.Constant) and first.value == tag:
            return ast.get_source_segment(source, node.args[1]) or ""
    pytest.fail(f"train.py no longer reports {tag}")


def test_both_halves_report_the_ratio_the_optimizer_receives():
    """The two series sit side by side and are read as comparable.

    They were not: ``adv_to_rec_ratio`` was the raw gradient ratio while
    ``fm_to_rec_ratio`` was reported after its weight, so a balance rule sitting
    exactly on target read as tens against the other's fractions.  Measured on
    the pretrain at step 14.5k, ``adv_to_rec_ratio`` showed 29.1 with an
    adaptive weight of 0.0172 -- a product of 0.50, which *is*
    ``refinegan_adv_balance_target``.

    Each series has to carry its own weight, and neither may carry the other's.
    """
    adversarial = _reported_ratio("GAN/adv_to_rec_ratio")
    feature_match = _reported_ratio("GAN/fm_to_rec_ratio")

    assert "adaptive_adv" in adversarial and "last_layer_adv_grad" in adversarial
    assert "adaptive_fm" in feature_match and "last_layer_fm_grad" in feature_match
    assert "adaptive_fm" not in adversarial
    assert "adaptive_adv" not in feature_match
    # Both divide by the reconstruction gradient, which is what makes them
    # comparable to each other rather than only to their own targets.
    for expression in (adversarial, feature_match):
        assert "last_layer_rec_grad" in expression


def test_the_config_ships_the_knobs():
    import json

    train = json.loads(CONFIG.read_text(encoding="utf-8"))["train"]
    # 0.33 is where the run sat from 4k to 16k, the stretch over which the
    # holdout improved fastest -- not a round number.
    assert train["refinegan_fm_balance_target"] == 0.33
    assert train["refinegan_fm_balance_min"] == 0.1
    # Raised from 20.0, then 100.0, then 400.0 -- ``stft_2048`` pinned on the
    # first two.  Why the bound has to span a 700x spread, and how saturation is
    # detected now rather than found by hand: ``test_r1_saturation.py``.
    assert train["refinegan_r1_max_scale"] == 10000.0
