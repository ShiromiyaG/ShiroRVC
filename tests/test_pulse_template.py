"""The pitch template RefineGAN's decoder refines.

This is section 2.2 of the paper and the single most consequential difference
from the Applio port, which substitutes NSF's band-limited sine bank.  The
template is what gives the vocoder its pitch and intensity response: the
decoder never estimates either, it only refines a signal that already has them
exactly right.  So the properties that matter are arithmetic, not perceptual,
and all of them are cheap to check.

The FP32 pinning is load-bearing rather than defensive.  The phase accumulator
is a ``cumsum`` over the whole waveform; in FP16 its increments (``f0 /
sample_rate``, order 1e-3) stop registering against the accumulated total within
a few hundred samples, and the pulse train drifts flat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the generator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.generators.refinegan import PulseTemplate  # noqa: E402

SR = 44100


def _template(**overrides):
    return PulseTemplate(sample_rate=SR, **overrides)


def _pulse_positions(signal: torch.Tensor, wave_amp: float = 0.1) -> torch.Tensor:
    """Sample indices carrying a pulse.

    Thresholded rather than ``nonzero``: unvoiced regions are filled with noise
    at ``wave_amp / 3``, so every unvoiced sample is nonzero too.
    """
    return (signal[0, 0].abs() > wave_amp * 0.9).nonzero().flatten()


# --------------------------------------------------------------------------
# Pitch response


@pytest.mark.parametrize("f0_hz", [80.0, 200.0, 523.25])
def test_pulses_land_exactly_one_pitch_period_apart(f0_hz):
    """The property the whole design exists for."""
    template = _template()
    f0 = torch.full((1, 1, SR), float(f0_hz))

    positions = _pulse_positions(template(f0))
    spacing = (positions[1:] - positions[:-1]).float()

    # One sample of jitter is inherent: a pulse lands on an integer sample and
    # the period generally is not one.
    assert spacing.mean() == pytest.approx(SR / f0_hz, rel=1e-3)
    assert spacing.max() - spacing.min() <= 1.0


def test_a_voiced_onset_starts_its_own_period():
    """A phrase must not open with a fraction inherited from an earlier one."""
    template = _template()
    f0 = torch.zeros(1, 1, 4410)
    onset = 1000
    f0[..., onset:] = 200.0

    positions = _pulse_positions(template(f0))

    assert int(positions[0]) == onset


def test_a_pitch_sweep_keeps_the_instantaneous_period():
    """Pulse spacing has to track f0 continuously, not an average of it."""
    template = _template()
    f0 = torch.linspace(100.0, 400.0, SR).view(1, 1, -1)

    positions = _pulse_positions(template(f0))
    spacing = (positions[1:] - positions[:-1]).float()

    # Monotone rising pitch means monotone falling period.
    assert bool((spacing[1:] <= spacing[:-1] + 1.0).all())
    assert spacing[0] > spacing[-1] * 3


# --------------------------------------------------------------------------
# Unvoiced regions


def test_unvoiced_regions_carry_uniform_noise_not_silence_and_not_gaussian():
    """The paper specifies uniform noise; consonants live here."""
    template = _template()
    unvoiced = template(torch.zeros(1, 1, 40000))

    assert float(unvoiced.abs().max()) > 0.0
    # Uniform on [-a, a] has kurtosis 1.8; a Gaussian has 3.
    normalized = unvoiced.flatten() / unvoiced.flatten().std()
    kurtosis = float((normalized**4).mean())
    assert kurtosis == pytest.approx(1.8, abs=0.15)


def test_the_unvoiced_floor_sits_below_the_voiced_pulse_amplitude():
    template = _template(wave_amp=0.1, noise_scale=1.0 / 3.0)
    unvoiced = template(torch.zeros(1, 1, 20000))

    assert float(unvoiced.abs().max()) <= 0.1 / 3.0 + 1e-6


# --------------------------------------------------------------------------
# Intensity response


@pytest.mark.parametrize("level", [0.2, 0.5, 0.8])
def test_the_amplitude_envelope_scales_the_pulses(level):
    """The paper's intensity response: loudness is set before refinement."""
    template = _template()
    f0 = torch.full((1, 1, 4410), 200.0)
    amplitude = torch.full((1, 1, 10), level)

    signal = template(f0, amplitude=amplitude)

    assert float(signal.abs().max()) == pytest.approx(level, rel=1e-3)


def test_the_envelope_is_interpolated_to_the_sample_rate():
    """A frame-rate envelope must not step; the ramp is what avoids clicks."""
    template = _template()
    f0 = torch.full((1, 1, 4410), 200.0)
    amplitude = torch.cat(
        (torch.full((1, 1, 5), 0.2), torch.full((1, 1, 5), 0.8)), dim=-1
    )

    signal = template(f0, amplitude=amplitude)
    peaks = signal[0, 0].abs()
    peaks = peaks[peaks > 0.1]

    assert float(peaks[0]) == pytest.approx(0.2, abs=0.05)
    assert float(peaks[-1]) == pytest.approx(0.8, abs=0.05)
    assert bool((peaks[1:] >= peaks[:-1]).all())
    # At least one pulse lands strictly between the two levels, so the frame
    # boundary is a ramp rather than a step.
    assert bool(((peaks > 0.25) & (peaks < 0.75)).any())


# --------------------------------------------------------------------------
# Mixed precision


def test_fp16_input_does_not_drift_the_pulse_train():
    """The reason the phase accumulator is pinned to FP32."""
    template = _template()
    f0 = torch.full((1, 1, SR), 200.0)

    reference = _pulse_positions(template(f0))
    half = _pulse_positions(template(f0.half()).float())

    assert half.numel() == reference.numel()
    assert int((half - reference).abs().max()) <= 1


def test_the_output_follows_the_input_dtype():
    template = _template()
    f0 = torch.full((1, 1, 4410), 200.0, dtype=torch.float16)

    assert template(f0).dtype == torch.float16
