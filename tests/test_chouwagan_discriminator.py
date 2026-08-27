"""ChouwaGAN's discriminator, and the contract ``train.py`` holds it to.

The trainer's per-branch machinery -- the R1 strength controller, the branch
loss series, the paired real/fake forward -- is written against a small surface
rather than against a specific discriminator.  Every method here is reached by
name from the training loop, so a rename surfaces as a crash mid-epoch, and a
``branch_names`` that does not match ``discriminators`` mislabels a metric
series forever without ever looking wrong.

The second half is ``branchwise=False``, which collapses that whole surface onto
one branch.  It has to stay a *presentation* change: the same networks, the same
paired forward, the same R1 quantity summed instead of rotated.  The way it
could silently go wrong is by reporting a controller-sized truncation of the
real work -- one branch's R1 charged as if it covered eight -- so the tests
below pin the aggregate against the per-branch parts rather than merely checking
that it runs.

The third is SAN.  Its heads return two logits from one projection, which makes
several things that were previously tensors into two-element lists -- the batch
split in the paired forward, the feature maps, the loss input.  Every one of
those has a wrong version that runs, produces finite numbers, and trains
something.  What is asserted here is which half goes where, and that the
adversarial governor's "dead discriminator" floor moved with the loss form.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the discriminator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi.chouwagan import (  # noqa: E402
    PERIODS,
    SPECTROGRAM_SPECS,
    ChouwaCQTDiscriminator,
    ChouwaGANDiscriminator,
)
from rvc.train.losses import discriminator_loss, feature_loss  # noqa: E402

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
SAMPLES = 8820


def _discriminator(**overrides) -> ChouwaGANDiscriminator:
    torch.manual_seed(0)
    return ChouwaGANDiscriminator(**overrides)


def _audio(batch=2):
    return torch.randn(batch, 1, SAMPLES) * 0.2


# --------------------------------------------------------------------------
# Branch layout


def test_the_default_branches_are_the_periods_and_the_stft_stack():
    model = _discriminator()

    assert model.period_count == len(PERIODS)
    assert model.spectral_count == len(SPECTROGRAM_SPECS)
    assert model.num_branches == len(PERIODS) + len(SPECTROGRAM_SPECS)


def test_branch_names_are_built_from_what_was_constructed():
    """A literal list would mislabel a series after a config change."""
    model = _discriminator(periods=(2, 3), use_cqt=True, use_subband=True)

    assert model.branch_names == (
        "period_2",
        "period_3",
        "stft_512",
        "stft_1024",
        "stft_2048",
        "cqt_128",
        "subband_8",
    )
    assert len(model.branch_names) == model.num_branches


def test_the_cqt_joins_the_spectrogram_family_and_the_subband_does_not():
    """``_spectrogram_index`` maps a branch onto a slot by subtraction, so the
    branches owning a fixed transform have to be one contiguous run.  The
    sub-band branch judges the waveform and must fall outside it, or it would be
    handed some other branch's spectrogram."""
    model = _discriminator(use_cqt=True, use_subband=True)
    spectrograms = model.prepare_spectrograms(_audio())

    assert len(spectrograms) == model.spectral_count == len(SPECTROGRAM_SPECS) + 1
    assert isinstance(model.discriminators[model.spectral_count + model.period_count - 1], ChouwaCQTDiscriminator)
    for index in range(model.period_count):
        assert model._spectrogram_index(index) is None
    for offset in range(model.spectral_count):
        assert model._spectrogram_index(model.period_count + offset) == offset
    # The sub-band branch is last and owns no slot.
    assert model._spectrogram_index(model.num_branches - 1) is None


# --------------------------------------------------------------------------
# Forward contract


def test_the_paired_forward_matches_two_separate_passes():
    """Batching real and fake together must not change what any branch sees."""
    model = _discriminator(use_cqt=True, use_subband=True).eval()
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


def test_without_san_every_head_returns_a_plain_logit():
    """``use_san=False`` has to fall all the way back to LSGAN, including under
    ``san_training`` -- the flag is set unconditionally by the loop."""
    model = _discriminator(use_san=False).eval()
    real, fake = _audio(), _audio()

    real_logits, fake_logits, _, _ = model(real, fake, san_training=True)

    assert all(torch.is_tensor(value) for value in real_logits)
    assert model.supports_san is False
    loss, loss_real, loss_fake = discriminator_loss(real_logits, fake_logits)
    assert torch.isfinite(loss)
    assert loss.detach().item() == pytest.approx(
        (loss_real + loss_fake).detach().item(), rel=1e-5
    )


# --------------------------------------------------------------------------
# SAN


def test_the_heads_split_only_under_san_training():
    """``san_training`` is what the *discriminator* update sets.  The generator
    update runs the same branches without it and must get a single logit, or
    ``generator_loss`` would start charging it for the direction output it has
    no gradient path to."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    split, _, _, _ = model(real, fake, san_training=True)
    plain, _, _, _ = model(real, fake)

    assert all(isinstance(value, (list, tuple)) and len(value) == 2 for value in split)
    assert all(torch.is_tensor(value) for value in plain)
    # The function half is the same projection the generator sees.
    for pair, single in zip(split, plain):
        assert torch.allclose(pair[0], single, atol=1e-5)


def test_the_paired_forward_slices_both_halves_of_a_san_logit():
    """The batch split runs over a *list* here.  Slicing the list instead of its
    tensors would hand the loss the function output as "real" and the direction
    output as "fake" -- both finite, both moving, and completely wrong."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    paired = model(real, fake, pair_batches=True, san_training=True)
    separate = model(real, fake, san_training=True)

    for side in (0, 1):
        for pair, single in zip(paired[side], separate[side]):
            assert len(pair) == 2
            for left, right in zip(pair, single):
                assert left.shape == right.shape
                assert torch.allclose(left, right, atol=1e-5)


def test_the_direction_output_trains_the_projection_and_nothing_else():
    """That separation is the whole mechanism: the trunk must not be able to
    lower its loss by shrinking the projection, and the projection must be
    optimised on a metric that actually separates real from fake."""
    model = _discriminator(periods=(2,), spectrogram_specs=())
    branch = model.discriminators[0]
    audio = _audio()

    function_output, direction_output = branch(audio, san_training=True)[0]

    direction_output.mean().backward(retain_graph=True)
    assert branch.conv_post.weight.grad is not None, "direction must reach the projection"
    assert branch.conv_post.scale.grad is None, "direction must not train the scale"
    assert all(
        parameter.grad is None for parameter in branch.convs.parameters()
    ), "direction must not reach the trunk"

    function_output.mean().backward()
    assert branch.conv_post.scale.grad is not None, "function must train the scale"
    assert any(
        parameter.grad is not None for parameter in branch.convs.parameters()
    ), "function must reach the trunk"


def test_feature_matching_gets_the_function_output_not_the_pair():
    """``feature_loss`` subtracts feature maps elementwise; appending the pair
    would put a list where a tensor has to be."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    _, _, fmap_r, fmap_g = model(real, fake, san_training=True)

    assert all(torch.is_tensor(value) for branch in fmap_r for value in branch)
    assert torch.isfinite(feature_loss(fmap_r, fmap_g, normalize=True))


def test_the_reprojection_hook_is_reachable_by_the_name_train_py_looks_up():
    """``_normalize_san_weights`` walks ``modules()`` looking for a
    ``normalize_weight`` attribute.  Rename it and the factorisation silently
    degrades into an ordinary convolution with a redundant multiplier -- no
    error, no dead series, just a slowly drifting direction."""
    model = _discriminator(use_cqt=True, use_subband=True)
    heads = [
        module for module in model.modules() if hasattr(module, "normalize_weight")
    ]

    assert len(heads) == model.num_branches

    head = heads[0]
    with torch.no_grad():
        head.weight.mul_(3.0)
    assert float(head.weight.flatten(1).norm(dim=1).max()) > 1.5
    head.normalize_weight()
    assert torch.allclose(
        head.weight.flatten(1).norm(dim=1), torch.ones(head.out_channels), atol=1e-5
    )


def test_r1_never_takes_the_san_path():
    """The penalty needs one score to differentiate, and the direction output
    carries no gradient to the trunk R1 is meant to constrain."""
    model = _discriminator()

    logits, _ = model(_audio(), branch_index=0, san_training=True)

    assert torch.is_tensor(logits)
    assert torch.isfinite(model.r1_penalty(_audio(), 0).detach())


def test_the_governor_floor_matches_what_the_san_loss_actually_reports():
    """``_constant_discriminator_loss`` is the adversarial governor's definition
    of a dead discriminator.  Under SAN the floor is ~2.37, not LSGAN's 0.5;
    leaving it at 0.5 would tell the governor a healthy discriminator had
    already collapsed and pin the ceiling at its minimum for the whole run."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    namespace: dict = {"math": math, "CONSTANT_DISCRIMINATOR_LOSS": 0.5}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_constant_discriminator_loss":
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
    constant_loss = namespace.get("_constant_discriminator_loss")
    if constant_loss is None:
        pytest.fail("train.py no longer defines _constant_discriminator_loss")

    constant = [torch.full((2, 8), 0.5)]
    plain, _, _ = discriminator_loss(constant, constant, normalize=True)
    assert float(plain) == pytest.approx(constant_loss(0.25, False))

    paired = [[value, value] for value in constant]
    san, _, _ = discriminator_loss(
        paired, paired, san_direction_weight=0.25, normalize=True
    )
    assert float(san) == pytest.approx(constant_loss(0.25, True), rel=1e-6)
    assert float(san) > 2.0


# --------------------------------------------------------------------------
# R1


def test_r1_differentiates_the_input_each_branch_actually_judges():
    """Period and sub-band branches see the waveform; the spectral ones see
    their own compressed transform, held outside the graph so the double
    backward skips it and the divergent ``|X|**0.3`` Jacobian with it."""
    model = _discriminator(use_cqt=True, use_subband=True)

    for index in range(model.num_branches):
        penalty = model.r1_penalty(_audio(), index).detach()
        assert torch.isfinite(penalty)
        assert penalty.item() >= 0.0


def test_r1_produces_a_gradient_for_the_branch_it_was_asked_about():
    model = _discriminator()
    branch = model.period_count  # the first spectrogram branch

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
    assert _discriminator().uses_branchwise_r1 is True


# --------------------------------------------------------------------------
# The branchwise switch


def test_turning_it_off_reports_one_branch_and_no_series():
    """``branch_names`` is what the loop reads to decide whether to keep a loss
    series per branch, and ``uses_branchwise_r1`` what it reads to allow unused
    parameters under DDP.  Both have to move together with the switch."""
    model = _discriminator(branchwise=False, use_cqt=True, use_subband=True)

    assert model.num_branches == 1
    assert model.branch_names == ()
    assert model.uses_branchwise_r1 is False
    # The real names survive: they name modules, not just metric series.
    assert len(model.all_branch_names) == len(model.discriminators)


def test_turning_it_off_changes_nothing_about_the_adversarial_forward():
    """The switch is about how the trainer *drives* the branches, not about
    which networks exist -- a collapsed run that quietly dropped branches would
    still train and would be very hard to notice."""
    on = _discriminator(branchwise=True).eval()
    off = _discriminator(branchwise=False).eval()
    real, fake = _audio(), _audio()

    for left, right in zip(on(real, fake)[0], off(real, fake)[0]):
        assert torch.allclose(left, right, atol=1e-6)


def test_the_collapsed_r1_is_the_sum_of_the_branch_penalties():
    """Not a sample of one branch charged as if it covered all of them.

    ``r1_gamma`` is scaled by ``r1_interval`` on the assumption that the penalty
    it multiplies covers the whole discriminator; a truncation here would apply
    the full budget to a fraction of the network and read as a healthy series.
    """
    torch.manual_seed(0)
    audio = _audio()
    branched = _discriminator(branchwise=True)
    collapsed = _discriminator(branchwise=False)

    parts = sum(
        float(branched.r1_penalty(audio, index).detach())
        for index in range(branched.num_branches)
    )
    total = float(collapsed.r1_penalty(audio, 0).detach())

    assert total == pytest.approx(parts, rel=1e-4)


def test_the_collapsed_r1_reaches_every_branch():
    model = _discriminator(branchwise=False)

    model.r1_penalty(_audio(), 0).backward()

    for index, branch in enumerate(model.discriminators):
        assert any(
            parameter.grad is not None for parameter in branch.parameters()
        ), f"branch {index} received no R1 gradient"


def test_the_gradient_measurement_collapses_with_the_switch():
    """``_branch_discriminative_norms`` zips this against the controller's
    names; a per-branch grouping under a single controller would silently
    truncate to the first branch's norm."""
    assert len(_discriminator(branchwise=True).branch_parameter_groups()) == len(
        _discriminator().discriminators
    )
    groups = _discriminator(branchwise=False).branch_parameter_groups()
    assert len(groups) == 1
    assert len(groups[0]) == len(list(_discriminator().parameters()))


# --------------------------------------------------------------------------
# Mixed precision


def test_the_stft_stays_in_fp32_under_autocast():
    """cuFFT's half path underflows on quiet frames, and the compression gain
    multiplies that error rather than absorbing it."""
    model = _discriminator()
    branch = model.discriminators[model.period_count]
    audio = _audio()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        spectrogram = branch.spectrogram(audio)

    assert spectrogram.dtype == torch.float32
    assert torch.allclose(spectrogram, branch.spectrogram(audio))


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

    before = model(audio)[0]
    model.remove_weight_norm()
    after = model(audio)[0]

    for left, right in zip(before, after):
        assert torch.allclose(left, right, atol=1e-5)
