"""The preprocessing DC-offset filter, and its agreement with inference.

The filter is easy to get wrong in a way that nothing reports: a causal IIR
removes the offset exactly as well as a zero-phase one and passes every
magnitude check, while rotating the phase of everything near the cutoff.  The
model then learns waveforms that inference never produces, because
``rvc/infer/pipeline.py`` filters its input with ``filtfilt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy", reason="needs numpy", exc_type=ImportError)
signal = pytest.importorskip(
    "scipy.signal", reason="needs scipy", exc_type=ImportError
)


def _preprocess():
    return pytest.importorskip(
        "rvc.train.preprocess.preprocess",
        reason="needs the preprocessing backend",
        exc_type=ImportError,
    )


def _voice(sr, seconds=2.0, f0=125.0):
    """A harmonic tone: energy at the fundamental and above, none below it.

    ``f0`` divides ``sr`` and the length is a whole number of its periods, so
    every harmonic completes an integer number of cycles and the tone is
    exactly zero-mean.  Otherwise the signal carries a DC component of its own
    that the filter is right to remove and the test would read as leakage.
    """
    assert sr % f0 == 0, "f0 must divide the sample rate"
    length = int(seconds * sr)
    length -= length % int(sr // f0)
    t = np.arange(length) / sr
    out = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    # Faded in and out, because a zero-phase filter has to invent context at
    # the signal edges and a tone that starts and ends at full amplitude is not
    # a recording -- it is a step, and the transient it produces is the test's
    # own artefact.  Measured on the un-faded version: the DC residual is
    # 5.7e-5 over the whole signal and 1.4e-6 once 500 samples of edge are
    # dropped.  Real files start and end near silence; this matches them.
    return (0.2 * out * np.hanning(length)).astype(np.float32)


def test_the_offset_goes_and_the_phase_stays():
    """Both halves of the contract, on the same signal.

    The reference is the input with its DC removed by subtraction, which is
    what the filter should converge to on material that has no energy below the
    cutoff.  A causal filter of the same design lands ~60 dB short of it, and
    all of that shortfall is phase.
    """
    preprocess = _preprocess()
    sr = 32000
    clean = _voice(sr)
    clean = clean - clean.mean()
    noisy = (clean + 0.02).astype(np.float32)

    processor = preprocess.PreProcess(sr, "/tmp")
    filtered = processor.high_pass(noisy)

    # 0.02 in, and what comes out has to be gone rather than merely reduced:
    # 80 dB of attenuation, against a filter that is -260 dB at 0.1 Hz.
    assert abs(float(filtered.mean())) < 0.02 * 1e-4, "the DC offset must be gone"

    def error_db(x):
        return 20 * np.log10(
            np.sqrt(np.mean((x - clean) ** 2)) / np.sqrt(np.mean(clean**2)) + 1e-30
        )

    assert error_db(filtered) < -60.0

    zi = signal.sosfilt_zi(processor.hp_sos)
    causal, _ = signal.sosfilt(processor.hp_sos, noisy, zi=zi * float(noisy[0]))
    assert error_db(causal) > error_db(filtered) + 30.0, (
        "a causal filter should be far worse here; if it is not, the test "
        "signal no longer isolates phase"
    )


def test_no_sample_is_worse_than_the_causal_filter():
    """The change was adopted as strictly-better, so pin that.

    A zero-phase filter pays for its phase response at the signal edges, where
    it has to invent context.  On a clip that starts and ends near silence --
    every real recording -- that cost does not materialise, and this is what
    would catch it coming back.
    """
    preprocess = _preprocess()
    sr = 32000
    clean = _voice(sr)
    clean = clean - clean.mean()
    noisy = (clean + 0.02).astype(np.float32)

    processor = preprocess.PreProcess(sr, "/tmp")
    zi = signal.sosfilt_zi(processor.hp_sos)
    causal, _ = signal.sosfilt(processor.hp_sos, noisy, zi=zi * float(noisy[0]))

    zero_phase_error = np.abs(processor.high_pass(noisy) - clean)
    assert zero_phase_error.max() <= np.abs(causal - clean).max()


def test_short_clips_survive():
    """``filtfilt`` raises when padlen >= len(signal); nothing may reach it."""
    preprocess = _preprocess()
    processor = preprocess.PreProcess(32000, "/tmp")

    for length in (0, 1, 2, 3, 64, 4001):
        out = processor.high_pass(np.full(length, 0.02, dtype=np.float32))
        assert out.shape[0] == length
        assert out.dtype == np.float32


def test_preprocessing_and_inference_filter_the_same_way():
    """Same design, same application.  Divergence here is a silent train /
    inference mismatch, not a bug either side reports."""
    preprocess = _preprocess()

    infer_source = (ROOT / "rvc" / "infer" / "pipeline.py").read_text(encoding="utf-8")
    assert "FILTER_ORDER = 5" in infer_source
    assert "CUTOFF_FREQUENCY = 48" in infer_source
    assert "signal.filtfilt(bh, ah, audio)" in infer_source

    assert preprocess.HIGH_PASS_CUTOFF == 48
    train_source = (
        ROOT / "rvc" / "train" / "preprocess" / "preprocess.py"
    ).read_text(encoding="utf-8")
    assert "signal.sosfiltfilt(" in train_source
    assert "signal.sosfilt(" not in train_source.replace("signal.sosfiltfilt(", "")
