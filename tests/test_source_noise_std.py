"""The excitation's voiced dither, as a knob.

The bug this pins: what a RefineGAN2 generator is short of above 10 kHz is the
stochastic floor between the harmonics -- 11.1 dB down on the valleys against
7.1 on the peaks -- and in voiced frames the only stochastic material it is
handed is ``noise_std``, which sat 27.4 dB under the harmonic RMS with no way
to change it.  A reconstruction loss cannot ask for that component either: the
minimiser of an L1 against an unpredictable signal is less of it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the generator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.generators.refinegan2 import RefineGAN2Generator  # noqa: E402

SR = 32000


def _generator(**kwargs):
    return RefineGAN2Generator(sample_rate=SR, **kwargs)


def test_absent_means_what_every_earlier_run_was_trained_against():
    """A checkpoint and its config have to agree on the excitation level, so a
    config naming nothing must build exactly the old source."""
    assert _generator().m_source.noise_std == pytest.approx(0.003)
    assert _generator().source_noise_std == pytest.approx(0.003)


def test_it_reaches_the_source():
    """It was a constructor default on ``BlitGenerator`` with nothing passing
    it, which is the same as not having the knob."""
    assert _generator(source_noise_std=0.01).m_source.noise_std == pytest.approx(0.01)


#: ``SineGenerator.forward`` wants ``(batch, samples, dim)`` at the *output*
#: rate, ``dim`` being the harmonic axis -- the phase is a cumsum over the
#: sample axis, so a transposed f0 gives back noise with no partials in it and
#: every assertion below still passes.  ``RefineGAN2Generator.forward`` does
#: this transpose too; the trunk itself is channel-first throughout.
def _excitation(noise_std, f0_hz, samples=32000):
    f0 = torch.full((1, samples, 1), float(f0_hz))
    with torch.no_grad():
        return _generator(source_noise_std=noise_std).m_source(f0)


def _floor_and_peak(excitation, n_fft=1024):
    window = torch.hann_window(n_fft)
    mag = torch.stft(
        excitation.reshape(1, -1), n_fft, 256, n_fft, window=window,
        center=True, return_complex=True,
    ).abs()
    return (
        float(mag.quantile(0.25, dim=1).mean()),
        float(mag.quantile(0.9, dim=1).mean()),
    )


def test_it_only_moves_the_voiced_frames():
    """Unvoiced frames are ``wave_amp / 3`` and must not follow this knob, or
    the voiced/unvoiced balance moves with it."""
    quiet = _excitation(0.003, 0.0)
    loud = _excitation(0.10, 0.0)
    assert loud.std().item() == pytest.approx(quiet.std().item(), rel=0.2)


def test_it_raises_the_floor_without_touching_the_partials():
    """The point of the knob: more energy between the partials, the partials
    themselves where they were.  A knob that raised both would be ``wave_amp``
    with extra steps."""
    quiet_floor, quiet_peak = _floor_and_peak(_excitation(0.003, 200.0))
    loud_floor, loud_peak = _floor_and_peak(_excitation(0.05, 200.0))
    assert loud_floor > 3.0 * quiet_floor
    assert loud_peak == pytest.approx(quiet_peak, rel=0.35)


def test_the_partials_are_actually_there_to_begin_with():
    """Guards the guard: with f0 on the wrong axis the source returns noise
    and every assertion above passes for the wrong reason."""
    floor, peak = _floor_and_peak(_excitation(0.003, 200.0))
    assert peak > 5.0 * floor
