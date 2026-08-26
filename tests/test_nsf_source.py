"""The harmonic excitation ChouwaGAN's decoder is fed.

Two properties, both of which stopped holding as the harmonic count went up.

**Inference has to stay in the distribution training used.** Training draws a
random phase offset per harmonic on every batch, so the decoder only ever sees
excitations from that distribution and must be phase-agnostic. Deterministic
inference used to leave the offsets at zero, which is not a draw from that
distribution but its extreme -- aligning every harmonic maximises the crest
factor. Measured at f0=200 Hz over 200 draws, aligned was 1.31x peakier than a
training excitation at 16 harmonics and 1.57x at 32, and the gap widens with
every harmonic added. Those are ratios of *means*, and the training spread is
wide enough that the two distributions still overlap at these counts -- so this
is a defect of principle that grows into a real one as the harmonic count
climbs, not a catastrophe at 32. Inference should be a sample of what training
showed the decoder, not its deterministic extreme.

**Harmonics past Nyquist have to actually go away.** The mask is a sigmoid on
purpose, because a hard cutoff makes a harmonic blink on and off as vibrato
walks it across the threshold. But soft means leaky, and centred at 0.48 of the
sample rate the leak was 27% at Nyquist itself. Everything past Nyquist aliases
back into the top of the band.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the generator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.generators.chouwagan import (  # noqa: E402
    BandLimitedNSFSource,
)

SR = 44100
LENGTH = SR // 2
FRAMES = LENGTH // 441


def _source(harmonics: int) -> BandLimitedNSFSource:
    source = BandLimitedNSFSource(SR, harmonic_count=harmonics)
    source.deterministic = True
    return source


def _excite(source: BandLimitedNSFSource, f0_hz: float, training: bool):
    source.train(training)
    f0 = torch.full((1, FRAMES), float(f0_hz))
    return source(f0, LENGTH, float(SR))


def _crest(x: torch.Tensor) -> float:
    x = x.flatten()
    return float(x.abs().max() / x.pow(2).mean().sqrt())


def test_deterministic_inference_actually_draws_its_offsets(monkeypatch):
    """The mechanism, tested where it cannot be confused with noise.

    If the offsets were left at zero -- the old behaviour -- the seed would be
    irrelevant and both passes would be identical.  Changing it has to change
    the excitation.
    """
    import rvc.lib.algorithm.generators.chouwagan as generators

    source = _source(32)
    default = _excite(source, 200.0, training=False)
    monkeypatch.setattr(generators, "DETERMINISTIC_PHASE_SEED", 7)
    reseeded = _excite(source, 200.0, training=False)
    assert not torch.equal(default, reseeded)


@pytest.mark.parametrize("harmonics", [16, 32])
def test_deterministic_inference_is_in_distribution(harmonics):
    """The peakiness at inference has to be one the decoder has seen.

    Stated as membership of the training spread rather than as a ratio to one
    draw: the training distribution is wide (sd 0.31 at 16 harmonics, 0.37 at
    32), so comparing a fixed value against a single sample is mostly measuring
    that sample.  Against the mean, deterministic inference now sits at z=-1.37
    and z=-0.80, comfortably inside; aligning every harmonic put it at +2.4 and
    +3.9, and further out with every harmonic added.
    """
    source = _source(harmonics)
    draws = [_crest(_excite(source, 200.0, training=True)) for _ in range(40)]
    inference = _crest(_excite(source, 200.0, training=False))
    assert min(draws) <= inference <= max(draws), (
        f"inference crest factor {inference:.2f} is outside the "
        f"[{min(draws):.2f}, {max(draws):.2f}] the decoder trained on"
    )


def test_deterministic_inference_is_reproducible():
    """In-distribution must not have cost reproducibility.

    The offsets come from a fixed-seed generator, the same way the excitation
    noise already did, so two inference passes agree exactly.
    """
    source = _source(32)
    first = _excite(source, 200.0, training=False)
    second = _excite(source, 200.0, training=False)
    assert torch.equal(first, second)


def test_training_still_randomises_the_phase():
    """The regularisation itself is untouched: training keeps redrawing."""
    source = _source(32)
    assert not torch.equal(
        _excite(source, 200.0, training=True),
        _excite(source, 200.0, training=True),
    )


def _top_band_share(source: BandLimitedNSFSource, f0_hz: float) -> float:
    signal = _excite(source, f0_hz, training=False).squeeze()
    spectrum = torch.fft.rfft(signal * torch.hann_window(signal.numel())).abs()
    frequency = torch.fft.rfftfreq(signal.numel(), 1.0 / SR)
    return float(spectrum[frequency > 20000].pow(2).sum() / spectrum.pow(2).sum())


def test_aliasing_is_suppressed_at_a_singing_f0():
    """f0=800 Hz with 32 harmonics reaches 25.6 kHz, well past Nyquist.

    Centred at 0.48 the mask left 0.662% of the excitation's energy above
    20 kHz; at 0.45 it is 0.120%.  The bound below passes the second and fails
    the first.
    """
    assert _top_band_share(_source(32), 800.0) < 0.003


def test_a_speech_f0_is_unaffected():
    """Nothing below Nyquist should be touched by the mask at all."""
    # 32 harmonics of 120 Hz reach 3840 Hz -- nowhere near the transition.
    assert _top_band_share(_source(32), 120.0) < 1e-4


# -- FP16: the phase must not follow the autocast dtype ----------------------
#
# Under ``autocast(fp16)`` the HiFi-GAN source used to stay FP32 only because
# none of ``cumsum``/``fmod``/``sin`` are on autocast's op list and ``pitchf``
# happens to arrive as FP32.  That is an accident of the op list, and the
# failure mode if it ever stops holding is a detuned excitation, not a NaN --
# so no loss, and not the GradScaler, would report it.  These pin the fence.


def _sine_generator():
    from rvc.lib.algorithm.generators.hifigan_nsf import SineGenerator

    # harmonic_num=0 is what HiFiGANNSFGenerator builds.
    return SineGenerator(SR, num_harmonics=0)


@pytest.mark.parametrize("f0_dtype", [torch.float32, torch.float16])
def test_hifigan_source_stays_fp32_under_autocast(f0_dtype):
    if not torch.cuda.is_available():
        pytest.skip("autocast(fp16) is a CUDA path")
    generator = _sine_generator().cuda()
    f0 = (torch.rand(2, 64, device="cuda") * 300 + 80).to(f0_dtype)

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        waves, _voiced, _noise = generator(f0, 480)

    assert waves.dtype is torch.float32, (
        f"the excitation came back {waves.dtype}: the phase accumulation is "
        "running in reduced precision"
    )
    assert torch.isfinite(waves).all()


def test_autocast_does_not_change_the_excitation():
    """Same seed, autocast on and off: the samples must be identical."""
    if not torch.cuda.is_available():
        pytest.skip("autocast(fp16) is a CUDA path")
    generator = _sine_generator().cuda()
    f0 = torch.rand(2, 64, device="cuda") * 300 + 80

    # The generator draws a fresh noise term per call, so the seed has to be
    # reset or this measures the noise rather than the phase.
    torch.manual_seed(7)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        under_autocast, _v, _n = generator(f0, 480)
    torch.manual_seed(7)
    with torch.autocast(device_type="cuda", enabled=False):
        plain, _v, _n = generator(f0, 480)

    assert torch.equal(under_autocast, plain)
