"""BS.1770-4 loudness normalisation, and why it replaced the RMS mode.

``post_rms`` measures the RMS of samples above a *fixed* -40 dBFS threshold.
A fixed threshold on a relative measurement is scale-dependent, and past the
point where no sample clears it the code takes a ``gain = 1.0`` branch and
normalises nothing -- silently.  These tests pin the calibration of the
replacement, that defect in the thing it replaces, and the wiring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.train.preprocess.loudness import (  # noqa: E402
    ABSOLUTE_GATE_LUFS,
    apply_gain_with_ceiling,
    integrated_lufs,
    loudness_gain,
)


def _sine(freq, seconds, sample_rate, amplitude=1.0):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * math.pi * freq * t)


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", [16000, 32000, 40000, 44100, 48000])
def test_a_full_scale_1k_sine_reads_minus_three_lufs(sample_rate):
    """BS.1770's calibration point, and the one number that says the filters
    are right at a rate other than the 48 kHz the spec tabulates."""

    measured = integrated_lufs(_sine(1000.0, 5.0, sample_rate), sample_rate)
    assert measured == pytest.approx(-3.01, abs=0.15)


@pytest.mark.parametrize("gain_db", [-40.0, -20.0, -6.0, 0.0, 6.0])
def test_the_measurement_is_exactly_scale_linear(gain_db):
    """The property ``post_rms`` does not have.  Scaling the input by g dB must
    move the measurement by g dB, or normalising to a target is guesswork."""

    sample_rate = 32000
    signal = _sine(1000.0, 5.0, sample_rate, 0.5)
    base = integrated_lufs(signal, sample_rate)
    moved = integrated_lufs(signal * 10 ** (gain_db / 20), sample_rate)
    assert moved - base == pytest.approx(gain_db, abs=1e-6)


def test_silence_is_not_a_loudness():
    """``-inf`` rather than a number: "unmeasurable" and "quiet" are different
    answers, and conflating them is how the RMS mode ends up doing nothing."""

    assert integrated_lufs(np.zeros(32000), 32000) == -math.inf
    assert integrated_lufs(np.array([]), 32000) == -math.inf
    # ...and a signal under the absolute gate is unmeasurable too.
    tiny = _sine(1000.0, 2.0, 32000, 10 ** ((ABSOLUTE_GATE_LUFS - 20) / 20))
    assert integrated_lufs(tiny, 32000) == -math.inf


def test_the_relative_gate_ignores_the_quiet_stretches():
    """The gate is the reason room tone does not drag the level down.  A file
    that is half speech and half a 45 dB quieter tail must measure close to the
    speech alone, not to their average."""

    sample_rate = 32000
    speech = _sine(1000.0, 4.0, sample_rate, 0.5)
    tail = _sine(1000.0, 4.0, sample_rate, 0.5 * 10 ** (-45 / 20))
    alone = integrated_lufs(speech, sample_rate)
    with_tail = integrated_lufs(np.concatenate([speech, tail]), sample_rate)
    assert abs(with_tail - alone) < 0.5

    # Without the relative gate this would be the answer instead, and it is
    # 3 dB out -- half the blocks at a negligible power halve the mean.
    ungated = 10 * math.log10(
        (np.concatenate([speech, tail]) ** 2).mean()
        / (speech ** 2).mean()
    )
    assert ungated < -2.5


# --------------------------------------------------------------------------
# what it replaces
# --------------------------------------------------------------------------


def _post_rms_level_db(audio):
    """``post_rms``'s level estimate, lifted out of ``preprocess`` verbatim."""
    mask = np.abs(audio) > 10 ** (-40.0 / 20)
    if not mask.any():
        return -math.inf
    return 20 * math.log10(math.sqrt((audio[mask] ** 2).mean()) + 1e-9)


def test_the_rms_estimate_is_scale_dependent_and_this_one_is_not():
    """The defect, as a measurement.  A quiet slice is an ordinary thing in a
    dataset -- mic distance, a soft phrase -- and ``post_rms`` mis-reads it by
    tens of dB before giving up entirely."""

    sample_rate = 32000
    # An amplitude envelope, because that is what makes the gate bite: on a
    # constant-amplitude tone almost every sample clears -40 dBFS whatever the
    # level, and speech is not a constant-amplitude tone.
    t = np.arange(3 * sample_rate) / sample_rate
    envelope = (0.5 + 0.5 * np.sin(2 * math.pi * 1.5 * t)) ** 2
    signal = _sine(220.0, 3.0, sample_rate, 0.5) * envelope

    rms_base = _post_rms_level_db(signal)
    lufs_base = integrated_lufs(signal, sample_rate)

    errors = {}
    for gain_db in (-10.0, -20.0, -30.0):
        quiet = signal * 10 ** (gain_db / 20)
        errors[gain_db] = (_post_rms_level_db(quiet) - rms_base) - gain_db
        assert abs((integrated_lufs(quiet, sample_rate) - lufs_base) - gain_db) < 1e-6

    # Measured: 0.72, 2.08, 5.00 dB.  Not rounding -- it grows with the
    # attenuation, because more of the waveform falls under the fixed gate.
    assert abs(errors[-10.0]) > 0.5
    assert abs(errors[-30.0]) > 4.0
    assert abs(errors[-30.0]) > abs(errors[-20.0]) > abs(errors[-10.0])
    # And far enough down, no sample clears the gate at all: the caller's
    # ``gain = 1.0`` branch then normalises nothing.
    assert _post_rms_level_db(signal * 10 ** (-35 / 20)) == -math.inf


def test_the_gain_gives_up_loudly_rather_than_silently():
    """``loudness_gain`` returns 1.0 for an unmeasurable slice, same as the RMS
    branch -- but ``integrated_lufs`` is available separately, so a caller that
    wants to *report* the case can, which is what the dry run does."""

    assert loudness_gain(np.zeros(32000), 32000, -16.0) == 1.0
    assert integrated_lufs(np.zeros(32000), 32000) == -math.inf


# --------------------------------------------------------------------------
# the gain and its ceiling
# --------------------------------------------------------------------------


def test_the_gain_hits_the_target():
    sample_rate = 32000
    signal = _sine(440.0, 3.0, sample_rate, 0.1)
    for target in (-23.0, -18.0, -14.0):
        gain = loudness_gain(signal, sample_rate, target)
        assert integrated_lufs(signal * gain, sample_rate) == pytest.approx(target, abs=1e-6)


def test_the_ceiling_reports_what_it_cost():
    """A file that cannot reach the target has to say so by how much, or the
    dataset ends up with a silent spread the dry run cannot warn about."""

    signal = _sine(440.0, 2.0, 32000, 0.5)
    ceiling = 10 ** (-1 / 20)

    # A gain that overshoots: the peak is pulled back and the shortfall is
    # reported, so the dry run can turn it into a message.
    kept, limited = apply_gain_with_ceiling(signal, 4.0, ceiling_db=-1.0)
    assert limited == pytest.approx(20 * math.log10(2.0 / ceiling), abs=1e-6)
    assert np.abs(kept).max() == pytest.approx(ceiling, rel=1e-9)

    # A gain that fits: nothing touched, nothing reported.
    kept, limited = apply_gain_with_ceiling(signal, 0.5, ceiling_db=-1.0)
    assert limited == 0.0
    assert np.abs(kept).max() == pytest.approx(0.25, rel=1e-9)


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------


def test_the_retired_modes_still_run_but_are_not_offered():
    """Three modes level every slice independently and are no longer offered;
    all three stay reachable so an experiment whose config names one re-runs
    unchanged."""

    from gui.services import catalog

    assert catalog.NORMALIZATION_MODES == ["none", "post_peak", "pre_loudness"]

    offered = '"none", "post_peak", "pre_loudness"'
    assert offered in (ROOT / "core.py").read_text(encoding="utf-8")
    assert offered in (ROOT / "tabs" / "train" / "train.py").read_text(encoding="utf-8")

    preprocess = (ROOT / "rvc" / "train" / "preprocess" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    for retired in ('"post_rms"', '"post_peak_rvc"', '"post_loudness"'):
        assert retired in preprocess, f"{retired} must still run"
        assert retired.strip('"') not in catalog.NORMALIZATION_MODES


def test_the_two_copies_are_normalised_by_the_same_gain():
    """The 16 kHz feature input and the ground truth must stay level-matched:
    the gain is measured once, on the ground truth, and applied to both."""

    preprocess = _preprocess()
    _apply_post_norm = preprocess._apply_post_norm
    _apply_post_norm_from_gain = preprocess._apply_post_norm_from_gain

    gt_sr, k16_sr = 32000, 16000
    ground_truth = _sine(220.0, 2.0, gt_sr, 0.05)
    sixteen = _sine(220.0, 2.0, k16_sr, 0.05)

    gt_out, _ = _apply_post_norm(ground_truth, gt_sr, "post_loudness", -18.0)
    k16_out = _apply_post_norm_from_gain(
        sixteen, ground_truth, "post_loudness", -18.0, gt_sample_rate=gt_sr
    )

    gt_gain = np.abs(gt_out).max() / np.abs(ground_truth).max()
    k16_gain = np.abs(k16_out).max() / np.abs(sixteen).max()
    assert gt_gain == pytest.approx(k16_gain, rel=1e-6)
    assert integrated_lufs(gt_out, gt_sr) == pytest.approx(-18.0, abs=0.05)


# --------------------------------------------------------------------------
# one gain per source recording
# --------------------------------------------------------------------------


def _preprocess():
    """Import ``preprocess``, standing in for ``ffmpeg`` if it is absent.

    ``preprocess`` imports ffmpeg-python for one audio-loading path that none
    of these tests touch, and skipping the whole group over it would leave the
    normalisation untested wherever that package is not installed.  The stub is
    inserted only when the real module is missing, so an environment that has
    it is unaffected.
    """

    import sys
    import types

    try:
        import ffmpeg  # noqa: F401
    except ImportError:
        sys.modules.setdefault("ffmpeg", types.ModuleType("ffmpeg"))
    import rvc.train.preprocess.preprocess as preprocess

    return preprocess


def test_the_source_key_is_the_speaker_and_the_file_index():
    """Slices are ``{sid}_{idx0}_{idx1}``.  ``idx0`` restarts per speaker
    directory, so the speaker has to be part of the key or two speakers' first
    recordings would be normalised as one."""

    source_key = _preprocess().source_key
    assert source_key("0_3_17.wav") == "0_3"
    assert source_key("2_3_17.flac") == "2_3"
    assert source_key("0_3_17.wav") != source_key("1_3_17.wav")
    # Every slice of one recording lands on the same key.
    assert len({source_key(f"0_3_{i}.wav") for i in range(50)}) == 1
    # And something that does not match the scheme still returns a key rather
    # than raising: a stray file must not take the run down.
    assert source_key("odd-name.wav") == "odd-name"


def test_pooling_blocks_equals_measuring_the_whole():
    """The property the per-source mode rests on.  The gates are defined over
    the whole measurement, so the slices' block powers are pooled and gated
    once -- gating each slice and averaging the results is a different number.
    """

    from rvc.train.preprocess.loudness import block_powers, loudness_from_blocks

    sample_rate = 32000
    whole = _sine(440.0, 6.0, sample_rate, 0.4)
    pieces = [whole[:2 * sample_rate], whole[2 * sample_rate:]]

    pooled = loudness_from_blocks(
        np.concatenate([block_powers(p, sample_rate) for p in pieces])
    )
    assert pooled == pytest.approx(integrated_lufs(whole, sample_rate), abs=0.01)


def _write_dataset(tmp_path, sample_rate, recordings):
    """``recordings`` is ``{(sid, idx0): [slice_gain_db, ...]}``."""
    import soundfile as sf

    names = []
    for (sid, idx0), gains in recordings.items():
        for index, gain_db in enumerate(gains):
            t = np.arange(int(1.2 * sample_rate)) / sample_rate
            audio = (
                0.5
                * np.sin(2 * math.pi * 220.0 * t)
                * (0.4 + 0.6 * np.sin(2 * math.pi * 2.0 * t) ** 2)
                * 10 ** (gain_db / 20)
            )
            name = f"{sid}_{idx0}_{index}.wav"
            sf.write(tmp_path / name, audio.astype(np.float32), sample_rate)
            names.append(name)
    return sorted(names)


def test_recordings_are_matched_and_their_inner_dynamics_survive(tmp_path):
    """The whole point of the mode, as two numbers.

    Two recordings 18 dB apart, each with 12 dB of dynamics across its own
    slices.  Afterwards the recordings must sit on top of each other and the
    12 dB must still be there -- which is exactly what per-slice normalisation
    destroys.
    """

    preprocess = _preprocess()
    sample_rate = 32000
    inner = [0.0, -4.0, -8.0, -12.0]
    files = _write_dataset(tmp_path, sample_rate, {
        (0, 0): [-12.0 + d for d in inner],
        (0, 1): [-30.0 + d for d in inner],
    })

    import soundfile as sf

    before = {f: integrated_lufs(sf.read(tmp_path / f)[0], sample_rate) for f in files}
    gains, effective, summary = preprocess.plan_source_gains(
        str(tmp_path), files, -18.0, 1
    )
    assert summary is None, "there is headroom here; nothing should be limited"
    assert effective == -18.0

    after = {
        f: before[f] + 20 * math.log10(gains[preprocess.source_key(f)])
        for f in files
    }

    def group(values, key):
        return [values[f] for f in files if preprocess.source_key(f) == key]

    # Between recordings: 18 dB apart, then together.
    gap_before = abs(np.mean(group(before, "0_0")) - np.mean(group(before, "0_1")))
    gap_after = abs(np.mean(group(after, "0_0")) - np.mean(group(after, "0_1")))
    assert gap_before == pytest.approx(18.0, abs=0.5)
    assert gap_after < 0.01

    # Within a recording: untouched, because one gain scales all of its slices.
    for key in ("0_0", "0_1"):
        span_before = max(group(before, key)) - min(group(before, key))
        span_after = max(group(after, key)) - min(group(after, key))
        assert span_before == pytest.approx(12.0, abs=0.5)
        assert span_after == pytest.approx(span_before, abs=1e-6)


def test_per_slice_normalisation_is_what_flattens_them(tmp_path):
    """The contrast that makes the per-source mode worth having."""

    preprocess = _preprocess()
    sample_rate = 32000
    files = _write_dataset(tmp_path, sample_rate, {
        (0, 0): [-12.0, -16.0, -20.0, -24.0],
    })

    import soundfile as sf

    levelled = [
        integrated_lufs(
            preprocess._apply_post_norm(
                sf.read(tmp_path / f)[0], sample_rate, "post_loudness", -18.0
            )[0],
            sample_rate,
        )
        for f in files
    ]
    assert max(levelled) - min(levelled) < 0.01


def test_a_recording_short_of_headroom_pulls_the_target_down_for_everyone(tmp_path):
    """A dataset where some sources reached the target and others were quietly
    limited is not level-matched, which was the entire point.  So the target
    comes down for all of them instead."""

    preprocess = _preprocess()
    sample_rate = 32000
    # One ordinary recording, one whose peaks leave almost no headroom.
    files = _write_dataset(tmp_path, sample_rate, {
        (0, 0): [-24.0, -24.0],
        (0, 1): [-1.0, -1.0],
    })

    gains, effective, summary = preprocess.plan_source_gains(
        str(tmp_path), files, -6.0, 1, ceiling_db=-1.0
    )
    assert summary is not None
    assert summary["num_sources"] == 2
    assert effective < -6.0

    import soundfile as sf

    reached = [
        integrated_lufs(
            sf.read(tmp_path / f)[0] * gains[preprocess.source_key(f)], sample_rate
        )
        for f in files
    ]
    # Still matched -- that is what lowering the target bought.
    assert max(reached) - min(reached) < 0.05
    # And nothing exceeds the ceiling.
    for f in files:
        peak = np.abs(sf.read(tmp_path / f)[0] * gains[preprocess.source_key(f)]).max()
        assert peak <= 10 ** (-1.0 / 20) + 1e-6


def test_the_per_source_mode_is_the_default():
    """``pre_loudness`` is named for what its result is equivalent to --
    normalising the recording before it was sliced -- not for when it runs,
    which is after slicing like every other mode here."""

    from gui.services import catalog

    assert catalog.NORMALIZATION_MODES[-1] == "pre_loudness"
    core = (ROOT / "core.py").read_text(encoding="utf-8")
    assert 'normalization_mode: str = "pre_loudness"' in core


def test_measuring_before_slicing_would_be_worse():
    """Why ``pre_loudness`` is not literally "pre", pinned as a measurement.

    Measuring the whole recording -- what the name describes -- reads *lower*
    than pooling the kept slices, always, and by an amount that depends on how
    much the recording pauses and how loud its room tone is.  That is the worst
    possible dependence for this mode: the error tracks exactly the
    between-recording variation it exists to remove.

    The numbers asserted below are from the synthetic here, -0.69 to -0.97 dB.
    On real singing the same comparison reached -4.69 dB (3 s gaps, -35 dBFS
    room tone), because real material has quiet passages inside the segments
    that fall under the relative gate once noise is added; that measurement is
    in the session notes, not reproducible from a sine, and is not asserted.

    Even with digitally silent gaps the error is ~0.7 dB, because blocks
    straddling an onset are half-filled and still clear the relative gate.
    """

    from rvc.train.preprocess.loudness import block_powers, loudness_from_blocks

    sample_rate = 32000
    rng = np.random.default_rng(0)
    t = np.arange(int(1.2 * sample_rate)) / sample_rate
    segment = (
        0.4
        * np.sin(2 * math.pi * 220.0 * t)
        * (0.15 + 0.85 * np.sin(2 * math.pi * 1.7 * t) ** 2)
    )

    def error_for(gap_seconds, floor_db):
        gap = np.zeros(int(gap_seconds * sample_rate))
        pieces, starts, cursor = [], [], 0
        for _ in range(6):
            starts.append(cursor)
            pieces += [segment, gap]
            cursor += len(segment) + len(gap)
        whole = np.concatenate(pieces)
        whole = whole + rng.normal(0, 10 ** (floor_db / 20), len(whole))
        kept = [whole[s:s + len(segment)] for s in starts]
        pooled = loudness_from_blocks(
            np.concatenate([block_powers(k, sample_rate) for k in kept])
        )
        return integrated_lufs(whole, sample_rate) - pooled

    quiet_gaps = error_for(0.3, -90.0)
    noisy_gaps = error_for(1.0, -45.0)

    # Always an under-read, never a wash -- so a genuine "pre" would put every
    # recording a little too loud.
    assert quiet_gaps < -0.4
    assert noisy_gaps < -0.4
    # And it is not a constant offset that could simply be calibrated out: it
    # moves with the recording's own pause and noise structure.
    assert noisy_gaps < quiet_gaps - 0.15
