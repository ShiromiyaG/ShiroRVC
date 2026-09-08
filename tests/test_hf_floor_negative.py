"""The synthetic negative that teaches the discriminator about the noise floor.

The bug this pins: what a RefineGAN2 generator is short of above 10 kHz is the
stochastic floor between the harmonics, not the harmonics -- measured at 11.1 dB
down on the valleys against 7.1 on the peaks -- and a discriminator trained only
against the generator's output scored the *quieter* floor as the more real of
the two (-0.13 on the 512-point spectrogram head, -0.82 on UnivHD).  The
negative has to reproduce that defect's shape, leave the rest of the band alone,
and not be recognisable by its own resynthesis.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="the transform needs torch", exc_type=ImportError)
pytest.importorskip("librosa", reason="rvc.train.losses imports it", exc_type=ImportError)

from rvc.train.losses import HighFrequencyFloorNegative  # noqa: E402
from tests._split_branch import split_branch_outputs  # noqa: E402

SR = 32000
CUT = 10000.0


@pytest.fixture
def speechlike():
    """A harmonic comb over a noise floor, which is what the transform is aimed at."""
    torch.manual_seed(0)
    t = torch.arange(SR) / SR
    comb = sum(torch.sin(2 * torch.pi * (180.0 * k) * t) / k for k in range(1, 80))
    return (comb + 0.05 * torch.randn(SR)).view(1, 1, -1).repeat(3, 1, 1)


def _band_stats(wave, low=11000.0, n_fft=1024):
    window = torch.hann_window(n_fft)
    mag = torch.stft(
        wave.squeeze(1), n_fft, 256, n_fft, window=window, center=True,
        return_complex=True,
    ).abs()
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / SR)
    band = mag[:, freqs >= low, :].clamp_min(1e-9)
    peaks = 20 * torch.log10(band.quantile(0.9, dim=1)).mean()
    valleys = 20 * torch.log10(band.quantile(0.25, dim=1)).mean()
    return float(peaks), float(valleys)


def test_the_valleys_fall_further_than_the_peaks(speechlike):
    """The defect's shape, and the reason a plain shelf will not do: a shelf
    moves both by the same amount and teaches the wrong thing."""
    negative = HighFrequencyFloorNegative(SR, cutoff_range=(CUT, CUT))(speechlike)
    peak_before, valley_before = _band_stats(speechlike)
    peak_after, valley_after = _band_stats(negative)
    assert valley_after - valley_before < peak_after - peak_before - 2.0
    assert peak_after < peak_before + 0.5


def test_it_leaves_the_band_below_the_cutoff_alone(speechlike):
    """Everything below the cutoff has to survive, or the discriminator learns
    a defect the generator does not have."""
    negative = HighFrequencyFloorNegative(SR, cutoff_range=(CUT, CUT))(speechlike)
    window = torch.hann_window(1024)
    freqs = torch.fft.rfftfreq(1024, 1.0 / SR)
    low = freqs < CUT - 1000.0

    def energy(wave):
        mag = torch.stft(
            wave.squeeze(1), 1024, 256, 1024, window=window, center=True,
            return_complex=True,
        ).abs()
        return float(mag[:, low, :].pow(2).sum())

    assert energy(negative) == pytest.approx(energy(speechlike), rel=1e-3)


def test_the_round_trip_alone_is_transparent(speechlike):
    """``gamma`` 1.0 is the identity on the magnitude, so what comes back is the
    resynthesis and nothing else.  It has to be inaudible: a discriminator that
    can hear the STFT round trip learns *that* instead of the floor, which is
    how this fails when it fails."""
    identity = HighFrequencyFloorNegative(SR, gamma_range=(1.0, 1.0))(speechlike)
    error = (identity - speechlike).norm() / speechlike.norm()
    assert 20 * torch.log10(error) < -60.0


def test_it_draws_a_new_defect_every_call(speechlike):
    """Fixed, a discriminator can learn one filter's signature instead of the
    thing the filter stands in for."""
    negative = HighFrequencyFloorNegative(SR)
    assert not torch.allclose(negative(speechlike), negative(speechlike))


def test_it_keeps_the_shape_and_builds_no_graph(speechlike):
    """The negative comes from the reference, not from the generator, so a
    graph here reaches nothing."""
    wave = speechlike.clone().requires_grad_(True)
    negative = HighFrequencyFloorNegative(SR)(wave)
    assert negative.shape == speechlike.shape
    assert not negative.requires_grad


def test_it_survives_the_autocast_the_discriminator_update_runs_under(speechlike):
    if not torch.cuda.is_available():
        pytest.skip("autocast(fp16) needs CUDA")
    transform = HighFrequencyFloorNegative(SR).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        negative = transform(speechlike.cuda())
    assert negative.dtype == torch.float32
    assert torch.isfinite(negative).all()


def test_it_rejects_a_cutoff_that_leaves_no_band():
    with pytest.raises(ValueError, match="Nyquist"):
        HighFrequencyFloorNegative(SR, cutoff_range=(6000.0, SR / 2))
    with pytest.raises(ValueError):
        HighFrequencyFloorNegative(SR, gamma_range=(2.0, 1.0))
    with pytest.raises(ValueError):
        HighFrequencyFloorNegative(SR, cutoff_range=(9000.0, 6000.0))


def test_the_two_fake_classes_come_back_out_of_one_batch():
    """They ride in one discriminator pass so the real side is computed once;
    the losses weight them differently, so they have to separate again."""
    branches = [torch.arange(12.0).reshape(6, 2) for _ in range(3)]
    generated, negative = split_branch_outputs(branches, (4, 2))
    assert len(generated) == len(negative) == 3
    assert generated[0].shape == (4, 2) and negative[0].shape == (2, 2)
    assert torch.equal(negative[0], branches[0][4:])


def test_a_san_branch_splits_both_of_its_halves():
    """Under SAN a branch returns ``(function, direction)`` rather than one
    tensor, and the direction is scored too."""
    branches = [[torch.arange(6.0).reshape(6, 1), torch.arange(6.0).reshape(6, 1) * 2]]
    generated, negative = split_branch_outputs(branches, (4, 2))
    assert len(generated[0]) == len(negative[0]) == 2
    assert torch.equal(negative[0][1], branches[0][1][4:])
