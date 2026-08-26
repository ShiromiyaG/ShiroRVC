"""The STFT branches of the ChouwaGAN discriminator.

Measured at step 8131 of the 44.1 kHz run: 94% of `stft_512`'s frequency rows
and 22% of `stft_2048`'s were near-inert, and the cause was resolution, not
compression or the magnitude floor. Every layer of the stack used a kernel 3
bins wide in frequency, which at n_fft 2048 spans 0.54 of the distance between
two harmonics of a 120 Hz f0 -- below one period a layer sees local slope and
cannot see a comb at all. `stft_512` and `stft_1024` cannot resolve the comb at
any kernel width and are transient branches; `stft_2048` can, and is the only
branch in the discriminator carrying high-frequency signal.

So the three branches stopped being three resolutions of one design. What is
tested here is that they can differ, that the difference is the intended one,
and that the two properties the R1 path and the feature-matching loss depend on
survived it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip(
    "torch", reason="the discriminator needs torch", exc_type=ImportError
)

from rvc.lib.algorithm.discriminators.multi.chouwagan import (  # noqa: E402
    PERIODS,
    SPECTROGRAM_SPECS,
    ChouwaCQTDiscriminator,
    ChouwaGANDiscriminator,
    ChouwaSpectrogramDiscriminator,
)

SEGMENT = 17640


def _spec(n_fft: int) -> dict:
    return next(spec for spec in SPECTROGRAM_SPECS if spec["n_fft"] == n_fft)


def _branch(discriminator, name):
    return discriminator.discriminators[discriminator.branch_names.index(name)]


@pytest.fixture(scope="module")
def discriminator():
    return ChouwaGANDiscriminator(sample_rate=44100, use_subband=True).eval()


def test_default_schedule_is_the_old_uniform_one():
    """A branch that overrides nothing is untouched by the change."""
    branch = ChouwaSpectrogramDiscriminator(1024, 256, 1024, channels=(32, 64, 96))
    for stage in branch.band_convs[0]:
        assert stage.conv.kernel_size == (3, 5)
        assert stage.conv.stride == (2, 2)


def test_only_the_branch_that_can_resolve_a_comb_is_widened():
    """`stft_512` and `stft_1024` keep the narrow kernel on purpose."""
    assert "kernels" not in _spec(512) and "kernels" not in _spec(1024)
    assert _spec(2048)["kernels"][0] == (9, 5)


def test_first_kernel_spans_more_than_one_harmonic_period():
    """The property the widening exists to buy, in Hz rather than in bins.

    At 44.1 kHz and n_fft 2048 a bin is 21.5 Hz, and harmonics of a 120 Hz f0
    are 120 Hz apart. Nine bins span 194 Hz: 1.6 periods, against the 0.54 a
    three-bin kernel saw.
    """
    hz_per_bin = 44100 / 2048
    bins = _spec(2048)["kernels"][0][0]
    assert bins * hz_per_bin / 120.0 > 1.0


def test_the_widened_branch_keeps_its_frequency_grid(discriminator):
    """Stride 1 in the last stage, so the output stops aliasing the comb.

    Resolving structure the output grid then samples below its own Nyquist
    would leave the widened kernel with nothing to show for itself.
    """
    stages = _branch(discriminator, "stft_2048").band_convs[0]
    assert stages[-1].conv.stride == (1, 2)
    index = discriminator.branch_names.index("stft_2048")
    with torch.no_grad():
        _, maps = discriminator.forward_branch(torch.randn(1, 1, SEGMENT), index)
    # maps is [stage1, stage2, stage3, conv_post]; the last stage strides in
    # time only, so it shares a frequency extent with the one before it.
    assert maps[2].shape[2] == maps[1].shape[2]
    assert maps[2].shape[3] < maps[1].shape[3]


def test_capacity_per_logit_position_is_no_longer_an_outlier(discriminator):
    """~200 parameters per position was the co-factor, ~1900 the reference."""
    with torch.no_grad():
        logits, _ = discriminator(torch.randn(1, 1, SEGMENT))
    index = discriminator.branch_names.index("stft_2048")
    branch = discriminator.discriminators[index]
    per_position = sum(p.numel() for p in branch.parameters()) / logits[index].shape[1]
    assert per_position > 350


def test_feature_map_count_is_unchanged(discriminator):
    """The feature-matching loss sums one term per map, so the count is a weight.

    Changing the shapes is free -- each term is a mean -- but adding or losing
    a map silently reweights `loss_fm` against everything else.
    """
    with torch.no_grad():
        _, fmaps = discriminator(torch.randn(1, 1, SEGMENT))
    counts = {
        name: len(fmap) for name, fmap in zip(discriminator.branch_names, fmaps)
    }
    assert counts["stft_2048"] == counts["stft_512"] == counts["stft_1024"] == 4


def test_r1_still_differentiates_the_widened_branch_twice(discriminator):
    """The R1 path double-backwards through the branch; a shape change breaks it."""
    index = discriminator.branch_names.index("stft_2048")
    penalty = discriminator.r1_penalty(torch.randn(2, 1, SEGMENT), index)
    penalty.backward()
    stage = discriminator.discriminators[index].band_convs[0][0].conv
    assert stage.parametrizations.weight.original1.grad is not None
    discriminator.zero_grad(set_to_none=True)


def test_period_branches_stay_pairwise_coprime():
    """No surviving branch folds on a sub-multiple of another's grid.

    The five original periods were pairwise coprime and the cut has to keep
    that: a period that divides another's makes the smaller branch see a strict
    subset of the larger one's alignments, which is redundancy by construction
    rather than the measured kind.
    """
    from math import gcd

    assert len(PERIODS) == 3
    for i, a in enumerate(PERIODS):
        for b in PERIODS[i + 1 :]:
            assert gcd(a, b) == 1


def test_the_cut_reweights_rather_than_rescales(discriminator):
    """Branch count is a weight, because every ChouwaGAN loss takes a mean.

    `discriminator_loss`, `generator_loss` and `feature_loss` all run with
    `normalize=True` here, so removing branches leaves the losses at the same
    scale and changes only whose opinion the mean reflects. That is the entire
    mechanism of the cut: the period block agreed with itself at rho 0.86-0.98
    partialling out clip RMS, so five of nine branches were near-copies of one
    opinion, and the independent branches were a minority of the mean.
    """
    with torch.no_grad():
        _, fmaps = discriminator(torch.randn(1, 1, SEGMENT))
    independent = [
        index
        for index, name in enumerate(discriminator.branch_names)
        if not name.startswith("period")
    ]
    maps = sum(len(fmap) for fmap in fmaps)
    # Was 4/9 and 16/46 with five period branches.
    assert len(independent) / discriminator.num_branches > 0.5
    assert sum(len(fmaps[i]) for i in independent) / maps > 0.45


def test_configured_periods_still_win(discriminator):
    """The cut is a default, not a hard-coded branch layout."""
    five = ChouwaGANDiscriminator(sample_rate=44100, periods=(2, 3, 5, 7, 11))
    assert five.branch_names[:5] == (
        "period_2",
        "period_3",
        "period_5",
        "period_7",
        "period_11",
    )
    assert discriminator.branch_names[: len(PERIODS)] == tuple(
        f"period_{period}" for period in PERIODS
    )


def _tone_with_near_silent_bins() -> torch.Tensor:
    """A 200 Hz sine: everything above the fundamental is near-silent.

    That is the condition that makes `d|X|**0.3/d|X|` blow up, and real speech
    has it by the thousand -- the measurement that motivated the change used
    dataset audio, but a tone reproduces it deterministically.
    """
    t = torch.arange(6144, dtype=torch.float32) / 44100.0
    return (0.5 * torch.sin(2 * torch.pi * 200.0 * t)).view(1, 1, -1).repeat(2, 1, 1)


def test_spectrogram_r1_does_not_differentiate_the_stft(discriminator):
    """R1 penalises the branch, not the conditioning of its own front end.

    The waveform-side penalty is computed here as its own reference rather
    than compared against a constant, so this pins the *choice of input* and
    not a number that drifts with initialisation.
    """
    index = discriminator.branch_names.index("stft_2048")
    branch = discriminator.discriminators[index]
    audio = _tone_with_near_silent_bins()

    waveform = audio.detach().requires_grad_(True)
    logits, _ = branch(waveform)
    through_stft = torch.autograd.grad(
        logits.float().mean(), waveform, create_graph=True
    )[0]
    waveform_side = through_stft.square().flatten(1).sum(dim=1).mean()

    spectrogram_side = discriminator.r1_penalty(audio, index)

    # The transform's Jacobian is the entire difference, and it is orders of
    # magnitude, not a factor of two.
    assert float(spectrogram_side.detach()) < float(waveform_side.detach()) / 1e3


def test_r1_input_choice_is_by_branch_type(discriminator):
    """A period branch keeps the waveform penalty; only STFT branches move."""
    audio = _tone_with_near_silent_bins()
    period = discriminator.branch_names.index(f"period_{PERIODS[0]}")
    branch = discriminator.discriminators[period]

    waveform = audio.detach().requires_grad_(True)
    logits, _ = branch(waveform)
    reference = (
        torch.autograd.grad(logits.float().mean(), waveform, create_graph=True)[0]
        .square()
        .flatten(1)
        .sum(dim=1)
        .mean()
    )
    assert torch.allclose(
        discriminator.r1_penalty(audio, period), reference, rtol=1e-4
    )


def test_r1_gradients_stay_within_a_few_orders_across_branches(discriminator):
    """The spread the per-branch controller has to close, bounded.

    Before the change this was ~1e8 on dataset audio and the controller, which
    moves 5% per event and sees a branch every `r1_interval * num_branches`
    steps, could not close it inside a run.
    """
    audio = _tone_with_near_silent_bins()
    norms = []
    for index in range(discriminator.num_branches):
        discriminator.zero_grad(set_to_none=True)
        discriminator.r1_penalty(audio, index).backward()
        norms.append(
            sum(
                float(p.grad.square().sum())
                for p in discriminator.discriminators[index].parameters()
                if p.grad is not None
            )
            ** 0.5
        )
    discriminator.zero_grad(set_to_none=True)
    assert min(norms) > 0.0
    assert max(norms) / min(norms) < 1e4


@pytest.fixture(scope="module")
def lazy_r1_penalty():
    """``train.py`` parses ``sys.argv`` at import, so lift the one function out.

    Same approach as `test_schedule_fitting.py`.
    """
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    kept = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_lazy_r1_penalty"
    ]
    assert kept, "train.py no longer defines _lazy_r1_penalty"
    namespace: dict = {"torch": torch}
    exec(compile(ast.Module(body=kept, type_ignores=[]), "train.py", "exec"), namespace)
    return namespace["_lazy_r1_penalty"]


def test_training_uses_the_model_r1_penalty(discriminator, lazy_r1_penalty):
    """The branch-aware penalty has to be on the path the training loop takes.

    It was not: `_lazy_r1_penalty` differentiated the waveform itself for every
    branch, which left `ChouwaGANDiscriminator.r1_penalty` as dead code that
    only tests exercised. A fix to the model that the loop never calls is not a
    fix, and nothing else in the suite would have noticed.
    """
    audio = _tone_with_near_silent_bins()
    index = discriminator.branch_names.index("stft_2048")
    # Pass exactly `segment_size` samples so no random window is drawn and the
    # two calls are comparable.
    routed = lazy_r1_penalty(discriminator, audio, index, audio.shape[-1])
    direct = discriminator.r1_penalty(audio, index)
    assert torch.allclose(routed.detach(), direct.detach(), rtol=1e-5)


def test_a_wrong_length_schedule_is_rejected():
    """Otherwise it surfaces as a shape mismatch several layers down."""
    with pytest.raises(ValueError, match="3-stage"):
        ChouwaSpectrogramDiscriminator(
            2048, 512, 2048, channels=(32, 64, 96), strides=((2, 2), (2, 2))
        )


# --------------------------------------------------------------------------
# The CQT branch exists on one claim: a harmonic stack keeps its *shape* under
# a change of f0 on a log-frequency axis, so one kernel detects it at every
# pitch, where a linear-axis kernel is tuned to one f0 and mis-sized at the
# rest.  That claim is the whole reason to pay for the branch, so it is what
# gets tested -- not merely that the module runs.


def _harmonic(f0: float, samples: int = SEGMENT, count: int = 24):
    t = torch.arange(samples, dtype=torch.float32) / 44100.0
    wave = sum(torch.sin(2 * torch.pi * f0 * k * t) / k for k in range(1, count + 1))
    return (wave / wave.abs().max() * 0.5).view(1, 1, -1)


def _profile(branch, f0: float):
    """Time-averaged magnitude down the frequency axis."""
    with torch.no_grad():
        spec = branch.spectrogram(_harmonic(f0))[0]
    return (spec[0] ** 2 + spec[1] ** 2).sqrt().mean(-1)


def _correlation(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


@pytest.mark.parametrize(("low", "high"), [(110.0, 220.0), (110.0, 165.0), (160.0, 320.0)])
def test_the_harmonic_pattern_survives_a_pitch_change(low, high):
    """Shift by the octave distance and the profile has to come back."""
    branch = ChouwaCQTDiscriminator()
    a, b = _profile(branch, low), _profile(branch, high)

    import math

    shift = round(math.log2(high / low) * branch.bins_per_octave)
    width = branch.n_bins - shift
    aligned = _correlation(a[:width], b[shift : shift + width])
    unshifted = _correlation(a, b)

    # The shift is predicted by the frequency ratio alone -- nothing is fitted.
    assert aligned > 0.85
    assert aligned > unshifted + 0.25


def test_the_linear_axis_cannot_do_the_same(  # noqa: D103
):
    """The comparison that justifies the branch, over the *best* linear shift."""
    import math

    cqt = ChouwaCQTDiscriminator()
    linear = ChouwaSpectrogramDiscriminator(
        2048, 512, 2048, channels=(64, 128, 192),
        kernels=((9, 5), (3, 5), (3, 5)), strides=((2, 2), (2, 2), (1, 2)),
    )
    low, high = 110.0, 220.0

    a, b = _profile(linear, low), _profile(linear, high)
    best_linear = max(
        _correlation(a[: len(a) - s], b[s:]) for s in range(0, 200)
    )
    ca, cb = _profile(cqt, low), _profile(cqt, high)
    shift = round(math.log2(high / low) * cqt.bins_per_octave)
    cqt_aligned = _correlation(ca[: cqt.n_bins - shift], cb[shift:])

    # Even allowed to search every rigid shift, the linear grid cannot match a
    # log grid handed the one shift the frequency ratio dictates.
    assert cqt_aligned > best_linear + 0.1


def test_the_filterbank_leaves_no_dead_row():
    """A row of zeros would hand the stack a permanently blank input channel."""
    branch = ChouwaCQTDiscriminator()
    sums = branch.filterbank.sum(dim=1)

    assert int((sums == 0).sum()) == 0
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_a_larger_transform_buys_bottom_octave_resolution():
    """Which is why the default is not the 2048 the STFT branch uses.

    Below a few hundred Hz a log bin is narrower than a linear one and the
    projection degrades to nearest-bin interpolation.  The conv stack sees
    `n_bins` rows whatever the transform, so this costs almost nothing.
    """
    def degenerate(n_fft):
        bank = ChouwaCQTDiscriminator(n_fft=n_fft).filterbank
        return int(((bank > 0).sum(dim=1) <= 1).sum())

    assert degenerate(2048) > degenerate(4096) > degenerate(8192)
    assert ChouwaCQTDiscriminator().n_fft == 4096


def test_the_cqt_branch_is_spectrogram_family():
    """It owns a fixed unlearned transform, so it wants the R1 and precompute
    treatment the STFT branches get -- differentiate the branch, hold the
    transform outside the graph.  ``_spectrogram_index`` covers a contiguous
    range, so this also pins the branch *order*."""
    model = ChouwaGANDiscriminator(sample_rate=44100, use_subband=False, use_cqt=True)
    index = model.branch_names.index("cqt_128")

    assert model._spectrogram_index(index) is not None
    audio = _harmonic(150.0).repeat(2, 1, 1)
    assert len(model.prepare_spectrograms(audio)) == model.spectrogram_count

    penalty = model.r1_penalty(audio, index)
    assert torch.isfinite(penalty)
    # Differentiating the transform instead would put this orders of magnitude
    # above the other branches; see `r1_penalty`.
    stft = model.r1_penalty(audio, model.branch_names.index("stft_2048"))
    assert abs(torch.log10(penalty / stft)) < 3


def test_the_cqt_branch_replaces_the_subband_rather_than_joining_it():
    model = ChouwaGANDiscriminator(sample_rate=44100, use_subband=False, use_cqt=True)

    assert model.branch_names == (
        "period_2", "period_5", "period_11",
        "stft_512", "stft_1024", "stft_2048", "cqt_128",
    )
    assert model.num_branches == len(PERIODS) + len(SPECTROGRAM_SPECS) + 1
