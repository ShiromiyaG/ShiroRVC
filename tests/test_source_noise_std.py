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
def _excitation(noise_std, f0_hz, samples=32000, harmonics=0):
    f0 = torch.full((1, samples, 1), float(f0_hz))
    with torch.no_grad():
        source = _generator(
            source_noise_std=noise_std, source_harmonics=harmonics
        ).m_source
        return source(f0)


def _floor_and_peak(excitation, f0_hz, harmonics=0, n_fft=1024):
    """Mean magnitude on the source's own partials, and between them.

    This used to take the 0.25 and 0.9 quantiles over the frequency axis, and
    that stopped measuring anything the day the BLIT was replaced: the sine
    puts ``harmonics + 1`` partials into 513 bins, so at the shipped
    ``harmonics = 0`` a single bin in 513 cannot move a 0.9 quantile and
    "peak" was reading the noise floor.  Both assertions below then compared
    the floor against itself -- ``test_the_partials_are_actually_there``, the
    guard written to catch exactly this class of mistake, is what failed.

    So the partials are located from f0 rather than found by rank.  A bin
    counts as floor only if it is 2 bins clear of every partial, which keeps
    the analysis window's skirts out of it.
    """

    window = torch.hann_window(n_fft)
    mag = torch.stft(
        excitation.reshape(1, -1), n_fft, 256, n_fft, window=window,
        center=True, return_complex=True,
    ).abs()[0].mean(dim=-1)

    bins = torch.arange(mag.numel())
    partial = torch.zeros_like(bins, dtype=torch.bool)
    near = torch.zeros_like(partial)
    for order in range(1, harmonics + 2):
        freq = order * f0_hz
        if freq >= SR / 2:
            break
        index = int(round(freq / SR * n_fft))
        partial[index] = True
        near |= (bins - index).abs() <= 2
    # DC carries the noise's own mean and no partial; the window's skirt puts
    # it in neither camp.
    near[0] = True
    return float(mag[~near].mean()), float(mag[partial].mean())


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
    quiet_floor, quiet_peak = _floor_and_peak(_excitation(0.003, 200.0), 200.0)
    loud_floor, loud_peak = _floor_and_peak(_excitation(0.05, 200.0), 200.0)
    assert loud_floor > 3.0 * quiet_floor
    assert loud_peak == pytest.approx(quiet_peak, rel=0.35)


@pytest.mark.parametrize("harmonics", [0, 31])
def test_the_partials_are_actually_there_to_begin_with(harmonics):
    """Guards the guard: with f0 on the wrong axis the source returns noise
    and every assertion above passes for the wrong reason.  Run at both
    harmonic counts, since the probe has to find the partials at each."""
    floor, peak = _floor_and_peak(
        _excitation(0.003, 200.0, harmonics=harmonics), 200.0, harmonics
    )
    assert peak > 5.0 * floor


def test_the_dither_survives_the_harmonic_count():
    """``noise_std`` is a level the trunk is trained against, and ``merge``
    sums ``dim`` independent draws -- so without the ``1/sqrt(dim)`` in
    ``SineGenerator.forward`` the same config would mean 0.01 at one partial
    and 0.057 at 32, silently past where the sweep measured the low bands
    starting to suffer."""

    bare = _floor_and_peak(_excitation(0.01, 200.0), 200.0)[0]
    rich = _floor_and_peak(_excitation(0.01, 200.0, harmonics=31), 200.0, 31)[0]
    assert rich == pytest.approx(bare, rel=0.1)
