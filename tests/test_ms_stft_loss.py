"""What the multi-resolution STFT term spends its gradient on.

The reason to add MS-STFT beside a mel loss is the one thing mel cannot give:
*linear* frequency resolution, which is what resolves a harmonic comb at the top
of the band -- a 128-band mel bin at 8 kHz spans several harmonics, so a comb
and band-limited noise of equal energy score identically to mel.

Two things in the original implementation worked against that purpose, and both
are pinned here. ``log(mag.clamp(1e-5))`` pins near-silent bins to a large
negative constant, so a plausible near-silent output scored far worse than an
audible error elsewhere and carried a wildly outsized share of the gradient.
Spectral convergence -- a Frobenius norm over the whole spectrogram -- is
dominated by the loudest bins, duplicating the low end the mel term already
emphasises.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="the losses need torch", exc_type=ImportError)
pytest.importorskip("librosa", reason="rvc.train.losses imports it", exc_type=ImportError)

from rvc.train.losses import MultiScaleSTFTLoss  # noqa: E402


SR = 44100


def _tone(frequency: float, seconds: float = 0.25, amplitude: float = 0.5):
    t = torch.arange(int(SR * seconds)) / SR
    return (amplitude * torch.sin(2 * math.pi * frequency * t)).view(1, 1, -1)


def test_identical_audio_scores_zero():
    loss = MultiScaleSTFTLoss()
    x = _tone(440.0)
    assert float(loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_spectral_convergence_is_off_by_default():
    assert MultiScaleSTFTLoss().spectral_convergence is False


def test_silence_against_silence_is_free():
    """The bug, at its simplest.

    Two silent signals are identical, so any sane loss returns zero.  Under
    ``log(clamp)`` they did too -- both pin to the clamp -- but the clamp is
    also what made *near*-silence catastrophically expensive, which is the next
    test.
    """
    loss = MultiScaleSTFTLoss()
    silence = torch.zeros(1, 1, SR // 4)
    assert float(loss(silence, silence)) == pytest.approx(0.0, abs=1e-6)


def _broadband(seconds: float = 0.25, level: float = 0.1, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, int(SR * seconds), generator=generator) * level


@pytest.mark.parametrize("gain, decibels", [(1.26, 2), (2.0, 6)])
def test_an_audible_gain_error_scores_its_own_log(gain, decibels):
    """The compression must not distort the range the loss actually works in.

    Well above the ``1 / log_scale`` knee, ``log1p`` is a logarithm, so a
    broadband gain error of ``g`` has to score exactly ``ln(g)`` -- the same
    value ``log(clamp)`` gave.  Changing the compression bought the silence
    behaviour below for nothing here, which is the point.
    """
    target = _broadband()
    loss = MultiScaleSTFTLoss()
    assert float(loss(target * gain, target)) == pytest.approx(
        math.log(gain), rel=0.02
    )


def test_silence_is_not_scored_out_of_all_proportion():
    """The misallocation ``log(clamp)`` created, with a discriminating bound.

    An inaudible broadband floor at 1e-4 (~-80 dBFS) where the target is
    digital silence is a defect worth some cost -- hiss in silence is a real
    failure mode for this vocoder -- but not an order of magnitude more than an
    audible 2 dB error across the whole band.  The bound below is loose enough
    to pass the fixed loss and tight enough to fail the old ``log(clamp)``
    term, so it tests the change rather than merely describing it.
    """
    loss = MultiScaleSTFTLoss()
    target = _broadband()

    audible = float(loss(target * 1.26, target))  # 2 dB, every bin
    silence = torch.zeros(1, 1, int(SR * 0.25))
    inaudible = float(loss(_broadband(level=1e-4, seed=1), silence))

    assert inaudible / audible < 8.0, (
        f"an inaudible -80 dBFS floor scored {inaudible / audible:.1f}x an "
        "audible 2 dB error across the same bins"
    )


def test_it_hears_a_harmonic_comb_against_noise():
    """The whole reason the term exists.

    A mel bin at 8 kHz spans four harmonics and cannot tell these apart; a
    linear-resolution term must.
    """
    torch.manual_seed(0)
    seconds = 0.25
    t = torch.arange(int(SR * seconds)) / SR
    comb = sum(torch.sin(2 * math.pi * f * t) for f in range(8000, 9000, 120))
    comb = (comb / comb.abs().max() * 0.5).view(1, 1, -1)

    noise = torch.randn_like(comb)
    # Match the comb's energy so only the structure differs.
    noise = noise / noise.square().mean().sqrt() * comb.square().mean().sqrt()

    loss = MultiScaleSTFTLoss()
    assert float(loss(noise, comb)) > float(loss(comb, comb)) + 0.1


def test_spectral_convergence_can_be_switched_back_on():
    x = _tone(440.0, amplitude=0.5)
    y = _tone(440.0, amplitude=0.8)
    without = float(MultiScaleSTFTLoss()(y, x))
    with_sc = float(MultiScaleSTFTLoss(spectral_convergence=True)(y, x))
    assert with_sc > without


def test_the_log_scale_is_configurable():
    x = _tone(440.0, amplitude=0.5)
    y = _tone(440.0, amplitude=0.8)
    assert float(MultiScaleSTFTLoss(log_scale=10.0)(y, x)) != pytest.approx(
        float(MultiScaleSTFTLoss(log_scale=1000.0)(y, x))
    )
