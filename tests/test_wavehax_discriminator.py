"""Wavehax's discriminator, pinned against upstream and against ChouwaGAN's.

The two share their families -- multi-period over the waveform, multi-resolution
over the STFT -- and it would be easy to conclude they are the same network with
different constants.  They are not, and the differences are the reason this
discriminator exists rather than a config override on the other one.  Every one
of them is asserted below, because a port that quietly drifted back toward the
other discriminator's shapes would still train and still produce audio.

The other half is the contract: this fork's loop drives R1 one branch at a time,
logs a loss series per branch and reuses precomputed spectrograms across a
paired real/fake forward.  That lives in ``BranchwiseDiscriminator`` and is
shared, so what is tested here is that Wavehax's branches actually satisfy it --
in particular that the spectral branch splits into ``spectrogram`` and
``forward_spectrogram``, which upstream has no reason to do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip(
    "torch", reason="the discriminator needs torch", exc_type=ImportError
)

from rvc.lib.algorithm.discriminators.multi import (  # noqa: E402
    BranchwiseDiscriminator,
    ChouwaGANDiscriminator,
    WavehaxDiscriminator,
)
from rvc.lib.algorithm.discriminators.multi.wavehax import (  # noqa: E402
    LRELU_SLOPE,
    PERIODS,
    RESOLUTIONS,
)

CONFIG = ROOT / "rvc" / "configs" / "wavehax" / "44100.json"


def _discriminator(**overrides) -> WavehaxDiscriminator:
    torch.manual_seed(0)
    return WavehaxDiscriminator(**overrides)


def _audio(batch=2, samples=8820):
    return torch.randn(batch, 1, samples)


# ---------------------------------------------------------------------------
# What makes it upstream's and not ChouwaGAN's
# ---------------------------------------------------------------------------


def test_the_period_branch_grows_by_four_and_stops_downsampling_last():
    """``channels: 32`` with ``min(x4, 1024)`` and ``[3,3,3,3,1]``.

    ChouwaGAN's period branch runs 16/64/128/192/256 and tops out at a quarter
    of this width.  UnivNet's starts at 32, quadruples, and reaches 1024 -- so it
    brings four times the capacity to the same folded grid, which is the bulk of
    the parameter gap pinned at the bottom of this file.
    """
    branch = _discriminator().discriminators[0]

    assert [conv.out_channels for conv in branch.convs] == [32, 128, 512, 1024, 1024]
    assert [conv.stride[0] for conv in branch.convs] == [3, 3, 3, 3, 1]
    # ``(k, 1)``: nothing ever mixes neighbouring periods.
    assert all(conv.kernel_size[1] == 1 for conv in branch.convs)
    assert branch.conv_post.kernel_size == (3, 1)


def test_the_spectral_branch_downsamples_frequency_as_well_as_time():
    """A six-layer stack that halves frequency at five of them.

    ChouwaGAN's spectral branches are three layers deep and its ``stft_2048``
    deliberately stops striding frequency at the last stage, so it can still
    read a harmonic comb where this one has long since reduced to a coarse
    time-frequency envelope.
    """
    branch = _discriminator().discriminators[len(PERIODS)]

    frequency_strides = [conv.stride[0] for conv in branch.convs]
    assert frequency_strides == [2, 2, 2, 2, 2, 1]
    assert [conv.stride[1] for conv in branch.convs] == [2, 1, 2, 1, 2, 1]
    assert [tuple(conv.kernel_size) for conv in branch.convs] == [
        (7, 5),
        (5, 3),
        (5, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]
    # A 1x1 lift to the working width before the stack, which ChouwaGAN folds
    # into its first strided convolution instead.
    assert branch.input_conv.kernel_size == (1, 1)
    assert branch.conv_post.kernel_size == (1, 1)


def test_the_spectral_window_is_hann_rather_than_rectangular():
    """UnivNet reads an envelope, where a taper is what keeps a loud frame from
    smearing across every bin.  Pinned because a rectangular window is the other
    defensible choice in this family and swapping it back would not fail
    anything else."""
    branch = _discriminator().discriminators[len(PERIODS)]

    assert not torch.allclose(branch.window, torch.ones_like(branch.window))
    assert torch.allclose(branch.window, torch.hann_window(branch.win_length))


def test_the_resolutions_are_upstreams():
    """Upstream's three transforms, with ``win_length == n_fft`` throughout."""
    assert RESOLUTIONS == ((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))
    for n_fft, _hop, win_length in RESOLUTIONS:
        assert win_length == n_fft


def test_the_leaky_relu_slope_is_univnets():
    assert LRELU_SLOPE == pytest.approx(0.1)


def test_the_output_convolution_is_not_a_feature_map():
    """Upstream excludes it, ChouwaGAN includes it.

    The output convolution *is* the logit; matching it under feature matching
    would make that loss a second, unscaled copy of the adversarial one.
    """
    model = _discriminator()
    _logits, feature_maps = model.forward_branch(_audio(), 0)

    assert len(feature_maps) == len(model.discriminators[0].convs)
    # ...and the same for a spectral branch, which has one more layer.
    spectral = len(PERIODS)
    _logits, feature_maps = model.forward_branch(_audio(), spectral)
    assert len(feature_maps) == len(model.discriminators[spectral].convs)


def test_it_is_a_materially_bigger_discriminator_than_chouwagans():
    """Worth pinning because it is the practical cost of picking this vocoder:
    the x4 growth in the period branch more than doubles the parameter count,
    against a generator that is far *smaller* than ChouwaGAN's."""
    wavehax = sum(p.numel() for p in _discriminator().parameters())
    chouwagan = sum(p.numel() for p in ChouwaGANDiscriminator().parameters())

    assert wavehax > 2 * chouwagan


# ---------------------------------------------------------------------------
# The contract the training loop drives
# ---------------------------------------------------------------------------


def test_it_implements_the_branchwise_contract():
    model = _discriminator()

    assert isinstance(model, BranchwiseDiscriminator)
    assert model.uses_branchwise_r1 is True
    assert model.period_count == len(PERIODS)
    assert model.spectral_count == len(RESOLUTIONS)
    assert model.num_branches == len(PERIODS) + len(RESOLUTIONS)
    assert len(model.branch_names) == model.num_branches


def test_branch_names_are_built_from_what_was_constructed():
    """A literal list would mislabel a metric series after a config change --
    invisibly, since the series would still be populated and still move."""
    model = _discriminator(periods=(2, 3), resolutions=((256, 64, 256),))

    assert model.branch_names == ("period_2", "period_3", "stft_256")


def test_a_mismatched_branch_name_list_is_rejected():
    model = WavehaxDiscriminator.__new__(WavehaxDiscriminator)
    torch.nn.Module.__init__(model)

    with pytest.raises(ValueError, match="branch names"):
        model.register_branches([torch.nn.Identity()], [], ["a", "b"])


def test_the_spectrogram_slots_line_up_with_the_branch_indices():
    model = _discriminator()
    spectrograms = model.prepare_spectrograms(_audio())

    assert len(spectrograms) == model.spectral_count
    for index in range(model.period_count):
        assert model._spectrogram_index(index) is None
    for offset in range(model.spectral_count):
        assert model._spectrogram_index(model.period_count + offset) == offset


def test_the_paired_forward_matches_two_separate_passes():
    """The loop runs ``cat(real, fake)`` through the branches once, which is
    only sound if every branch is batch-independent."""
    model = _discriminator().eval()
    real, fake = _audio(), _audio()

    with torch.no_grad():
        paired = model(real, fake, pair_batches=True)
        separate = model(real, fake, pair_batches=False)

    for lhs, rhs in zip(paired[0] + paired[1], separate[0] + separate[1]):
        assert torch.allclose(lhs, rhs, atol=1e-5)


def test_precomputed_real_spectrograms_are_used_rather_than_recomputed():
    """The trainer computes each real resolution once per step.  A branch that
    silently recomputed it would still be correct and would cost a second STFT
    on every branch of every step."""
    model = _discriminator().eval()
    audio = _audio()
    spectrograms = model.prepare_spectrograms(audio)
    poisoned = [value + 1.0 for value in spectrograms]

    with torch.no_grad():
        honest, _ = model(audio, real_spectrograms=spectrograms)
        altered, _ = model(audio, real_spectrograms=poisoned)

    spectral = len(PERIODS)
    assert torch.allclose(honest[0], altered[0])  # period branch, untouched
    assert not torch.allclose(honest[spectral], altered[spectral])


def test_r1_differentiates_the_input_each_branch_actually_judges():
    """A period branch is penalised on the waveform, a spectral branch on the
    magnitude -- with the STFT outside the graph, since a branch can do nothing
    about the conditioning of a fixed front end except shrink toward zero."""
    model = _discriminator()
    audio = _audio()

    for index in (0, len(PERIODS)):
        penalty = model.r1_penalty(audio, index)
        assert penalty.requires_grad
        assert torch.isfinite(penalty)
        gradients = torch.autograd.grad(
            penalty, [p for p in model.discriminators[index].parameters()],
            allow_unused=True,
        )
        assert any(g is not None and torch.isfinite(g).all() for g in gradients)


def test_the_stft_stays_in_fp32_under_autocast():
    """cuFFT's FP16 path underflows on quiet frames, and the magnitude is what
    the whole spectral branch reads."""
    branch = _discriminator().discriminators[len(PERIODS)]
    quiet = _audio() * 1e-4

    magnitude = branch.spectrogram(quiet)

    assert magnitude.dtype is torch.float32
    assert torch.isfinite(magnitude).all()
    assert magnitude.abs().sum() > 0.0


def test_non_finite_activations_are_dropped_rather_than_propagated():
    """One overflow in FP16 otherwise reaches the *generator* as a NaN gradient
    through feature matching, and carries no adversarial signal on the way."""
    model = _discriminator()
    audio = _audio()
    with torch.no_grad():
        model.discriminators[0].convs[0].bias.fill_(float("nan"))

    logits, feature_maps = model.forward_branch(audio, 0)

    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(value).all() for value in feature_maps)


def test_remove_weight_norm_preserves_the_logits():
    model = _discriminator().eval()
    audio = _audio()

    with torch.no_grad():
        before, _ = model.forward_branch(audio, 0)
        model.remove_weight_norm()
        after, _ = model.forward_branch(audio, 0)

    assert torch.allclose(before, after, atol=1e-5)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_shipped_config_names_every_knob_the_discriminator_takes():
    """The config is the only place these can be changed, and an unknown key is
    silently swallowed, so the names have to be pinned somewhere."""
    model = json.loads(CONFIG.read_text(encoding="utf-8"))["model"]

    assert model["d_periods"] == list(PERIODS)
    assert model["d_period_channels"] == 32
    assert model["d_period_downsample_scales"] == [3, 3, 3, 3, 1]
    assert model["d_resolutions"] == [list(r) for r in RESOLUTIONS]
    assert model["d_spectral_strides"] == [[2, 2], [2, 1], [2, 2], [2, 1], [2, 2], [1, 1]]
    # ChouwaGAN's key for the same idea, which would now be a silent no-op.
    assert "d_resolution_channels" not in model
