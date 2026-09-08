"""The held-out spectral deficit, reported per band.

The bug these pin: ``holdout/mel_l1`` is one number over a warped axis, so a
14 dB hole above 10 kHz and a 2 dB error at 1 kHz reach it as comparable
contributions.  A pretrain sat at a flat ``mel_l1`` for 87k steps while missing
14-18 dB above 10 kHz, and nothing logged said where the error was.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the metric needs torch", exc_type=ImportError)

from tests._band_deficit import band_deficit_db, deficit_band_edges  # noqa: E402

SR = 32000


def _shelved(wave, cutoff, db, sample_rate=SR):
    spectrum = torch.fft.rfft(wave, dim=-1)
    freqs = torch.fft.rfftfreq(wave.shape[-1], 1.0 / sample_rate)
    return torch.fft.irfft(
        spectrum * torch.where(freqs < cutoff, 1.0, 10.0 ** (db / 20.0)),
        n=wave.shape[-1],
        dim=-1,
    )


@pytest.fixture
def noise():
    torch.manual_seed(0)
    return torch.randn(4, 1, 32000)


def test_it_reads_back_the_shelf_it_was_given(noise):
    """-14 dB above 10 kHz has to come back as -14 dB in the bands above
    10 kHz and ~0 below, or the number cannot be read as a level."""
    deficits = band_deficit_db(_shelved(noise, 10000.0, -14.0), noise, SR)
    assert deficits["10k"] == pytest.approx(-14.0, abs=0.5)
    assert deficits["13k"] == pytest.approx(-14.0, abs=0.5)
    for label in ("1k", "2k", "4k", "6k", "8k"):
        assert deficits[label] == pytest.approx(0.0, abs=0.5)


def test_zero_deficit_for_an_identical_signal(noise):
    for value in band_deficit_db(noise, noise, SR).values():
        assert value == pytest.approx(0.0, abs=1e-4)


def test_the_sign_says_which_way_the_error_goes(noise):
    """Negative is the generator below the reference, which is the direction a
    vocoder fails in; excess has to read positive."""
    assert band_deficit_db(_shelved(noise, 10000.0, +6.0), noise, SR)["10k"] > 5.0


def test_a_silent_excerpt_does_not_poison_the_average():
    """Without the floor this is -inf, and one silent excerpt takes the whole
    evaluation's average with it."""
    silence = torch.zeros(2, 1, 32000)
    torch.manual_seed(1)
    reference = torch.randn(2, 1, 32000)
    for value in band_deficit_db(silence, reference, SR).values():
        assert np.isfinite(value)


@pytest.mark.parametrize("sample_rate", [16000, 24000, 32000, 40000, 48000])
def test_the_bands_stay_under_nyquist(sample_rate):
    """One list serves every shipped rate: a band starting above Nyquist is
    dropped and the last one is clipped to it."""
    edges = deficit_band_edges(sample_rate)
    assert edges, sample_rate
    assert all(high <= sample_rate / 2 for _, high, _ in edges)
    assert all(low < high for low, high, _ in edges)


def test_it_averages_in_db_not_over_pooled_energy():
    """One loud excerpt would otherwise decide the figure for the whole set."""
    torch.manual_seed(2)
    quiet = torch.randn(1, 1, 32000) * 1e-3
    loud = torch.randn(1, 1, 32000)
    reference = torch.cat([quiet, loud])
    # The quiet item is shelved, the loud one is untouched.
    generated = torch.cat([_shelved(quiet, 10000.0, -20.0), loud])
    # Averaged in dB the answer is the mean of -20 and 0; pooled energy would
    # be dominated by the loud item and report roughly nothing.
    assert band_deficit_db(generated, reference, SR)["10k"] == pytest.approx(
        -10.0, abs=0.5
    )
