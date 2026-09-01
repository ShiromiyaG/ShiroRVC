"""The output gate that keys on the input, and what it must not touch.

Digital silence has no level as far as the content encoder is concerned: its
first conv layer is group-normalised over the whole chunk and its transformer
attends across all of it, so silence comes out as a full-magnitude embedding
whose direction depends on the rest of the chunk, and the decoder renders that
faithfully -- as hiss over passages where the input was exactly 0.0. The
decoder itself is not at fault (it tracks level accurately when handed
training-scope features), so the gate lives at the end of the pipeline, the
last place that still knows what the input was.

What these pin is the part that is easy to get wrong: a gate that also touches
quiet-but-real audio is just ``change_rms`` with extra steps, and a gate that
steps rather than fades is a click.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.infer.pipeline import AudioProcessor

RATE = 16000
TARGET_RATE = 32000


def _noise(seconds, level_db, rate=RATE, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)) * 10 ** (level_db / 20)).astype(
        np.float32
    )


def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _gate(source, target, **kwargs):
    return AudioProcessor.gate_to_source(source, RATE, target, TARGET_RATE, **kwargs)


def test_hiss_over_digital_silence_is_removed():
    """The case from the Angra stem: input exactly 0.0, output at -56 dBFS."""
    source = np.zeros(RATE * 2, dtype=np.float32)
    target = _noise(2, -56, rate=TARGET_RATE)
    gated = _gate(source, target)
    assert _db(gated) < -100


def test_loud_audio_is_returned_untouched():
    """Not "almost untouched": the same array, so a normal file costs nothing."""
    source = _noise(2, -20)
    target = _noise(2, -20, rate=TARGET_RATE, seed=1)
    assert _gate(source, target) is target


def test_quiet_but_real_audio_is_left_alone():
    """A -45 dBFS passage is a passage, not silence.

    This is the line between the gate and ``change_rms``: the gate has nothing
    to say about audio the input actually contains, however quiet, so whatever
    dynamics the model gave it survive.
    """
    source = _noise(2, -45)
    target = _noise(2, -30, rate=TARGET_RATE, seed=2)
    assert _db(_gate(source, target)) == pytest.approx(_db(target), abs=0.1)


def test_the_knee_is_a_fade_not_a_switch():
    levels = [-40, -55, -62, -66, -75, -90]
    gains = []
    for level in levels:
        source = _noise(1.0, level)
        target = np.ones(TARGET_RATE, dtype=np.float32)
        gains.append(float(np.mean(_gate(source, target)[TARGET_RATE // 2 :])))
    assert gains[0] == pytest.approx(1.0)
    assert gains[-1] == pytest.approx(0.0, abs=1e-3)
    # Monotone, and something strictly between the ends: a hard switch would
    # give only ones and zeros here.
    assert all(a >= b - 1e-9 for a, b in zip(gains, gains[1:]))
    assert any(0.01 < g < 0.99 for g in gains)


def test_speech_keeps_its_tail():
    """A word ends into its own decay; the gate must not chop it.

    ``hold_ms`` is what buys that, and it is also what stops the gate
    chattering on the gaps between syllables.
    """
    source = np.concatenate([_noise(0.4, -20), np.zeros(int(0.05 * RATE), np.float32)])
    target = np.ones(int(len(source) / RATE * TARGET_RATE), dtype=np.float32)
    gated = _gate(source, target)
    assert gated[-1] == pytest.approx(1.0, abs=1e-3)


def test_the_gain_never_steps():
    """A step in the gain is an audible click, whatever else is right."""
    source = np.concatenate([_noise(0.5, -15), np.zeros(RATE, np.float32)])
    target = np.ones(int(len(source) / RATE * TARGET_RATE), dtype=np.float32)
    gated = _gate(source, target)
    # Per sample at 32 kHz; the release alone spans thousands of samples.
    assert np.abs(np.diff(gated)).max() < 0.01


def test_the_gate_can_be_turned_off():
    source = np.zeros(RATE, dtype=np.float32)
    target = _noise(1, -56, rate=TARGET_RATE)
    for disabled in (None, float("-inf")):
        assert _gate(source, target, threshold_db=disabled) is target


def test_it_survives_input_shorter_than_one_window():
    source = np.zeros(64, dtype=np.float32)
    target = _noise(0.004, -56, rate=TARGET_RATE)
    assert _gate(source, target) is target


def test_stereo_input_is_accepted():
    """``pipeline`` hands over mono, but the helper is callable on its own."""
    source = np.zeros((RATE, 2), dtype=np.float32)
    target = _noise(1, -56, rate=TARGET_RATE)
    assert _db(_gate(source, target)) < -100
