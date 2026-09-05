"""The excitation, and the guard that survives the two sources being removed.

RefineGAN's excitation is ``SineGenerator(sample_rate)`` with the default
``harmonic_num=0`` -- one sine at the fundamental, a ``Linear(1, 1)`` and a
tanh.  Every partial above f0 is manufactured by the trunk out of feature maps
on the frame grid.

``comb`` (fish-diffusion's band-limited impulse train) and ``bank`` (a
phase-randomised harmonic bank with a conditioning-driven envelope) were
removed on 2026-09-03.  They existed to be traded against the inharmonic lines
in the render, and those turned out to be the ``AdaIN`` activations -- see
``test_decoder_layout.py``, where the A/B renders are recorded.  On a
fixed trunk with ``source_gain`` on, the bank had measured *worse* than the
sine anyway (held-out multi-scale mel 1.7357 -> 1.8057 against 1.9714 ->
1.7418).

What is pinned here is what a removal must not break: the sine is unchanged in
the state dict and in ``architecture_id``, and the load guard still refuses a
checkpoint from either removed source instead of silently leaving
``m_source.merge`` at its random init.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.configs.vocoders import get_architecture_id  # noqa: E402
from rvc.lib.algorithm.generators.refinegan2 import (  # noqa: E402
    RefineGAN2Generator,
    SineGenerator,
)

SR = 44100


def _excitation(source, f0_hz, seconds=0.5):
    length = int(SR * seconds)
    f0 = torch.full((1, length, 1), float(f0_hz))
    with torch.no_grad():
        return source(f0)[0, :, 0]


def _harmonic_db(signal, f0_hz, harmonic, n_fft=8192, offset=4096):
    seg = signal[offset : offset + n_fft] * torch.hann_window(n_fft)
    spectrum = torch.fft.rfft(seg).abs()
    bin_index = int(round(harmonic * f0_hz / SR * n_fft))
    fundamental = spectrum[int(round(f0_hz / SR * n_fft))]
    return 20 * torch.log10(spectrum[bin_index] / fundamental).item()


def test_the_sine_is_one_partial_and_the_trunk_makes_the_rest():
    """The premise the whole decoder rests on, and the reason the ghost
    partials at +-``sr/hop`` were ever possible: nothing above f0 arrives in
    the excitation, so every harmonic in the output is synthesised."""

    sine = _excitation(SineGenerator(SR), 220.0)
    for harmonic in (2, 3, 5, 8):
        assert _harmonic_db(sine, 220.0, harmonic) < -30.0


def test_sine_is_the_only_source_and_keeps_its_identity():
    generator = RefineGAN2Generator(sample_rate=SR, upsample_rates=(3, 3, 7, 7))
    assert isinstance(generator.m_source, SineGenerator)
    assert {k for k in generator.state_dict() if "m_source" in k} == {
        "m_source.merge.0.weight"
    }
    assert generator.source_type == "sine"

    # The id is what the wider RVC ecosystem reads off a checkpoint; removing
    # the other two must not move it.
    for options in ({}, {"refinegan2_source": "sine"}):
        assert get_architecture_id("refinegan", options) == "vits_gaussian_v1"


def test_the_guard_still_refuses_a_removed_source():
    """The removal does not make the guard redundant -- it makes it the only
    thing standing between a bank checkpoint and a silent random-init
    ``merge``.  ``net_g`` resumes non-strictly, so the missing
    ``phase_offset`` raises nothing on its own.
    """

    sys.path.insert(0, str(ROOT / "rvc" / "train"))
    from utils import assert_excitation_matches, excitation_source

    holder = type("M", (), {"dec": RefineGAN2Generator(
        sample_rate=SR, upsample_rates=(3, 3, 7, 7))})()
    assert excitation_source(holder) == "sine"

    # No key is a stock upstream checkpoint, and every one of those is a sine.
    assert_excitation_matches(holder, {})
    assert_excitation_matches(holder, {"excitation_source": "sine"})
    for stale in ("comb", "bank32t0", "bank160t0"):
        with pytest.raises(ValueError, match="Excitation mismatch") as excinfo:
            assert_excitation_matches(holder, {"excitation_source": stale})
        assert "removed" in str(excinfo.value)


def test_a_model_with_no_decoder_is_not_checked():
    """The guard must not invent a mismatch for a stack it does not describe."""

    sys.path.insert(0, str(ROOT / "rvc" / "train"))
    from utils import assert_excitation_matches, excitation_source

    assert excitation_source(type("M", (), {})()) is None
    assert_excitation_matches(type("M", (), {})(), {"excitation_source": "bank"})


def test_the_excitation_amplitude_does_not_depend_on_the_seed():
    """``merge`` is a ``Linear(1, 1, bias=False)`` and, at ``harmonic_num=0``,
    has nothing to merge: it is one scalar multiplying the whole excitation.

    Its default init draws that scalar from ``U(-1, 1)``, which made the
    source's amplitude a property of the seed -- negative in about half of
    them, under a tenth of the intended level in one in seven, and 132x between
    the loudest and the quietest measured on the excitation itself.  Nothing
    downstream normalises it: ``sine_amp`` states the amplitude and this was
    free to ignore it.  The neighbouring ``source_gain`` is initialised to
    exactly 1.0 so that switching it on cannot rescale the source, which is the
    same intent this had to be brought in line with.
    """

    amplitudes = []
    for seed in range(8):
        torch.manual_seed(seed)
        source = SineGenerator(SR)
        assert source.merge[0].weight.item() == 1.0
        amplitudes.append(_excitation(source, 220.0).std().item())

    # A sine at ``sine_amp`` has RMS ``sine_amp / sqrt(2)``; the noise the
    # source adds on a voiced frame is ``noise_std``, three orders down.
    expected = 0.1 / 2**0.5
    for rms in amplitudes:
        assert abs(rms - expected) / expected < 0.02
    assert max(amplitudes) / min(amplitudes) < 1.01


def test_the_merge_weight_still_comes_from_the_checkpoint():
    """The init is not the value: a resumed run loads what it learned, which is
    what makes this the one change in this decoder that is not a fresh
    pretrain."""

    source = SineGenerator(SR)
    trained = {"merge.0.weight": torch.full_like(source.merge[0].weight, -0.42)}
    source.load_state_dict(trained, strict=True)
    assert source.merge[0].weight.item() == pytest.approx(-0.42)
