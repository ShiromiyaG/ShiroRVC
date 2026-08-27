"""``d_branchwise`` now buys memory on the generator step, not just R1 shape.

The flag used to change only *how* the discriminator was driven: R1 rotated one
branch at a time, one ``loss_disc`` series per branch.  The generator step still
ran a single joint forward over all six branches and held every feature map and
every branch's activation graph alive until one ``loss_gan`` backward at the far
end of the step -- six copies of the largest intermediates in the step, alive
across all of the reconstruction losses in between.

Branchwise, each branch is forwarded on a *detached* copy of the waveform,
differentiated immediately and dropped.  Measured on CUDA over the generator's
discriminator region with the real 44.1 kHz ChouwaGAN discriminator:

    batch     joint     branchwise    saved    time
        4   347.0 MiB    142.0 MiB     59%
        8   684.4 MiB    278.1 MiB     59%    43.9 -> 60.4 ms

The time is the honest half of the trade: each branch is differentiated twice,
once for ``loss_adv`` and once for ``loss_fm``, because their weights are
decided downstream from these very losses and the wave gradient is linear in
both -- keeping them apart is what lets the governors stay exact instead of
lagging a step.

What this file pins is that it is an exchange of time for memory and nothing
else: identical losses, identical gradient into the waveform.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", exc_type=ImportError)

from torch.amp import autocast  # noqa: E402

from rvc.lib.algorithm.discriminators.multi import chouwagan as chouwagan_d  # noqa: E402
from rvc.train.losses import feature_loss, generator_loss  # noqa: E402

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"


def _lift(*names):
    """Lift helpers out of ``train.py``, whose module body reads argv."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    namespace: dict = {
        "torch": torch,
        "autocast": autocast,
        "feature_loss": feature_loss,
        "generator_loss": generator_loss,
    }
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(found) == len(names), f"train.py no longer defines all of {names}"
    exec(compile(ast.Module(found, []), str(TRAIN_PY), "exec"), namespace)
    return [namespace[name] for name in names]


@pytest.fixture(scope="module")
def branch_terms():
    return _lift("_branchwise_generator_terms")[0]


@pytest.fixture(scope="module")
def discriminator():
    torch.manual_seed(0)
    model = chouwagan_d.ChouwaGANDiscriminator(
        use_checkpointing=False, sample_rate=44100, branchwise=True
    ).train()
    # The generator step freezes the discriminator before this runs; leaving it
    # trainable would have the branch loop accumulate into it.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def waveforms():
    torch.manual_seed(3)
    real = torch.randn(1, 1, 4410) * 0.2
    source = torch.randn(1, 1, 4410) * 0.2
    tail = torch.nn.Conv1d(1, 1, 1)
    return real, tail(source), tail


def _joint(discriminator, real, fake, spectrograms):
    with torch.no_grad():
        _, real_features = discriminator._forward_audio(real, spectrograms)
    logits, fake_features = discriminator._forward_audio(fake)
    return (
        generator_loss(logits, normalize=True),
        feature_loss(real_features, fake_features, normalize=True),
    )


def test_the_losses_are_the_ones_the_joint_pass_reports(
    branch_terms, discriminator, waveforms
):
    real, fake, _ = waveforms
    spectrograms = discriminator.prepare_spectrograms(real)
    joint_adv, joint_fm = _joint(discriminator, real, fake, spectrograms)

    terms = branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32,
    )

    # The denominators are the point: ``adv`` is a mean over branches and ``fm``
    # a mean over every layer term of every branch, and neither is known until
    # the loop has finished.  Getting one of them wrong rescales half the GAN
    # objective and still looks plausible.
    assert terms["loss_adv"].item() == pytest.approx(joint_adv.item(), rel=1e-6)
    assert terms["loss_fm"].item() == pytest.approx(joint_fm.item(), rel=1e-6)


def test_the_waveform_gradient_is_the_one_the_joint_pass_would_have_produced(
    branch_terms, discriminator, waveforms
):
    real, fake, _ = waveforms
    spectrograms = discriminator.prepare_spectrograms(real)
    joint_adv, joint_fm = _joint(discriminator, real, fake, spectrograms)
    joint_adv_grad = torch.autograd.grad(joint_adv, fake, retain_graph=True)[0]
    joint_fm_grad = torch.autograd.grad(joint_fm, fake, retain_graph=True)[0]

    terms = branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32,
    )

    for name, joint_gradient in (("adv", joint_adv_grad), ("fm", joint_fm_grad)):
        gradient = terms[f"{name}_wave_grad"]
        assert gradient.shape == joint_gradient.shape
        scale = joint_gradient.abs().max().clamp_min(1e-12)
        assert ((gradient - joint_gradient).abs().max() / scale).item() < 1e-3, name


def test_the_two_terms_stay_separable(branch_terms, discriminator, waveforms):
    """One combined gradient would force the branch loop to know
    ``adaptive_adv`` and ``adaptive_fm``, which are decided *from* these losses.
    Weighting afterwards has to give the same answer as weighting inside."""
    real, fake, _ = waveforms
    spectrograms = discriminator.prepare_spectrograms(real)
    joint_adv, joint_fm = _joint(discriminator, real, fake, spectrograms)
    combined = torch.autograd.grad(0.37 * joint_adv + 2.5 * joint_fm, fake)[0]

    terms = branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32,
    )
    weighted = 0.37 * terms["adv_wave_grad"] + 2.5 * terms["fm_wave_grad"]

    scale = combined.abs().max().clamp_min(1e-12)
    assert ((weighted - combined).abs().max() / scale).item() < 1e-3


def test_the_amp_scale_rides_on_the_gradient_and_not_on_the_losses(
    branch_terms, discriminator, waveforms
):
    """The gradients go to the optimizer, which unscales them.  The losses go to
    the controllers and the logs, which would read a scaled one as a real move."""
    real, fake, _ = waveforms
    spectrograms = discriminator.prepare_spectrograms(real)

    plain = branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32, loss_scale=1.0,
    )
    scaled = branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32, loss_scale=64.0,
    )

    assert scaled["loss_scale"] == 64.0
    assert scaled["loss_adv"].item() == pytest.approx(plain["loss_adv"].item(), rel=1e-6)
    assert scaled["loss_fm"].item() == pytest.approx(plain["loss_fm"].item(), rel=1e-6)
    for name in ("adv_wave_grad", "fm_wave_grad"):
        assert torch.allclose(
            scaled[name], plain[name] * 64.0, rtol=1e-4, atol=1e-9
        ), name


def test_it_never_touches_the_discriminators_own_gradients(
    branch_terms, discriminator, waveforms
):
    real, fake, _ = waveforms
    spectrograms = discriminator.prepare_spectrograms(real)
    branch_terms(
        discriminator, real, fake, spectrograms,
        use_amp=False, amp_dtype=torch.float32,
    )
    assert all(
        parameter.grad is None for parameter in discriminator.parameters()
    ), "the generator step must not write into the discriminator"


def test_the_flag_that_selects_it_is_the_one_the_discriminator_carries():
    """``branchwise`` is read off the discriminator rather than re-derived from
    the config, so the R1 layout and the generator step cannot disagree."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    assert 'getattr(discriminator_model, "branchwise", False)' in source
