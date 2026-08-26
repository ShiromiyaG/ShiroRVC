"""The MPD + MRD discriminator, and the contract ``train.py`` holds it to.

The trainer's per-branch machinery -- the R1 strength controller, the branch
loss series, the paired real/fake forward -- is written against a small surface
rather than against a specific discriminator.  Every method here is reached by
name from the training loop, so a rename surfaces as a crash mid-epoch, and a
``branch_names`` that does not match ``discriminators`` mislabels a metric
series forever without ever looking wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the discriminator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi.refinegan import (  # noqa: E402
    PERIODS,
    RESOLUTIONS,
    RefineGANDiscriminator,
)
from rvc.train.losses import discriminator_loss  # noqa: E402

SAMPLES = 17640


def _discriminator(**overrides) -> RefineGANDiscriminator:
    torch.manual_seed(0)
    return RefineGANDiscriminator(**overrides)


def _audio(batch=2):
    return torch.randn(batch, 1, SAMPLES) * 0.2


# --------------------------------------------------------------------------
# Branch layout


def test_the_branches_are_the_paper_s_mpd_and_mrd():
    model = _discriminator()

    assert model.period_count == len(PERIODS)
    assert model.resolution_count == len(RESOLUTIONS)
    assert model.num_branches == len(PERIODS) + len(RESOLUTIONS)


def test_branch_names_are_built_from_what_was_constructed():
    """A literal list would mislabel a series after a config change."""
    model = _discriminator(periods=(2, 3), resolutions=((256, 40, 160),))

    assert model.branch_names == ("period_2", "period_3", "stft_256")
    assert len(model.branch_names) == model.num_branches


def test_the_spectrogram_slots_line_up_with_the_branch_indices():
    """``prepare_spectrograms`` returns one entry per resolution branch, and
    ``_spectrogram_index`` has to map onto exactly those."""
    model = _discriminator()
    spectrograms = model.prepare_spectrograms(_audio())

    assert len(spectrograms) == model.resolution_count
    for index in range(model.period_count):
        assert model._spectrogram_index(index) is None
    for offset in range(model.resolution_count):
        assert model._spectrogram_index(model.period_count + offset) == offset


# --------------------------------------------------------------------------
# Forward contract


def test_the_paired_forward_matches_two_separate_passes():
    """Batching real and fake together must not change what any branch sees."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    paired = model(real, fake, pair_batches=True)
    separate = model(real, fake)

    for left, right in zip(paired[0], separate[0]):
        assert torch.allclose(left, right, atol=1e-5)
    for left, right in zip(paired[1], separate[1]):
        assert torch.allclose(left, right, atol=1e-5)


def test_precomputed_real_spectrograms_are_used_rather_than_recomputed():
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    with_cache = model(
        real, fake, real_spectrograms=model.prepare_spectrograms(real), pair_batches=True
    )
    without = model(real, fake, pair_batches=True)

    for left, right in zip(with_cache[0], without[0]):
        assert torch.allclose(left, right, atol=1e-5)


def test_branch_index_runs_one_branch_only():
    model = _discriminator().eval()
    audio = _audio()

    logits, fmap = model(audio, branch_index=3)
    full_logits, full_fmap = model._forward_audio(audio)

    assert torch.allclose(logits, full_logits[3], atol=1e-5)
    assert len(fmap) == len(full_fmap[3])


def test_every_head_returns_a_plain_logit_so_the_loss_takes_its_lsgan_path():
    """RefineGAN has no SAN projection; the paper's objective is LSGAN."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    real_logits, fake_logits, _, _ = model(real, fake, san_training=True)

    assert all(torch.is_tensor(value) for value in real_logits)
    loss, loss_real, loss_fake = discriminator_loss(real_logits, fake_logits)
    assert torch.isfinite(loss)
    assert loss.detach().item() == pytest.approx(
        (loss_real + loss_fake).detach().item(), rel=1e-5
    )


def test_a_constant_discriminator_scores_exactly_the_documented_floor():
    """``CONSTANT_DISCRIMINATOR_LOSS`` in train.py is the health baseline."""
    constant = [torch.full((2, 8), 0.5)]

    loss, _real, _fake = discriminator_loss(constant, constant, normalize=True)

    assert float(loss) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# R1


def test_r1_differentiates_the_input_each_branch_actually_judges():
    """Period branches see the waveform; resolution branches see magnitudes,
    with the FFT held outside the graph so the double backward skips it."""
    model = _discriminator()

    for index in range(model.num_branches):
        penalty = model.r1_penalty(_audio(), index).detach()
        assert torch.isfinite(penalty)
        assert penalty.item() >= 0.0


def test_r1_produces_a_gradient_for_the_branch_it_was_asked_about():
    model = _discriminator()
    branch = model.period_count  # the first resolution branch

    model.r1_penalty(_audio(), branch).backward()

    touched = [
        name
        for name, parameter in model.discriminators[branch].named_parameters()
        if parameter.grad is not None
    ]
    assert touched
    others = [
        parameter.grad
        for parameter in model.discriminators[0].parameters()
        if parameter.grad is not None
    ]
    assert not others


def test_the_trainer_can_see_that_r1_is_per_branch():
    assert RefineGANDiscriminator.uses_branchwise_r1 is True


# --------------------------------------------------------------------------
# Mixed precision


def test_the_stft_stays_in_fp32_under_autocast():
    """cuFFT's half path underflows on quiet frames, so the magnitude is
    computed in FP32 and only then cast to the autocast dtype."""
    model = _discriminator()
    branch = model.discriminators[model.period_count]
    audio = _audio()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        magnitude = branch.spectrogram(audio)

    assert magnitude.dtype == torch.bfloat16
    assert torch.allclose(
        magnitude.float(), branch.spectrogram(audio), atol=1e-1, rtol=1e-1
    )


def test_non_finite_activations_are_dropped_rather_than_propagated():
    """One overflow must not reach the generator as a NaN through feature
    matching."""
    model = _discriminator().eval()
    audio = _audio()
    audio[0, 0, 100] = float("inf")

    logits, fmap = model(audio, branch_index=0)

    assert bool(torch.isfinite(logits).all())
    assert all(bool(torch.isfinite(value).all()) for value in fmap)


def test_remove_weight_norm_preserves_the_logits():
    model = _discriminator().eval()
    audio = _audio()

    before, _ = model(audio, branch_index=0)
    model.remove_weight_norm()
    after, _ = model(audio, branch_index=0)

    assert torch.allclose(before, after, atol=1e-4)
