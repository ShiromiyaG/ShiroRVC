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
    block_powers,
    integrated_lufs,
    limit_peaks,
    loudness_from_blocks,
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
    unchanged.

    ``pre_peak_rvc`` is the recording-scope version of ``post_peak_rvc`` and is
    offered; the per-slice one it replaces is not.  The ``pre_``/``post_``
    prefix is about the scope of the gain, not about when the pass runs.
    """

    from gui.services import catalog

    assert catalog.NORMALIZATION_MODES == [
        "none",
        "post_peak",
        "pre_peak_rvc",
        "pre_loudness",
    ]

    offered = '"none", "post_peak", "pre_peak_rvc", "pre_loudness"'
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


def _level_recordings(preprocess, tmp_path, files, sample_rate, target_lufs,
                      ceiling_db=-1.0):
    """Run the shipped ``pre_loudness`` path over ``files`` in ``tmp_path``.

    The 16 kHz copies the worker also rewrites are made here rather than
    faked, so the ceiling assertions cover both.  Returns
    ``{source_key: (loudness, peak)}`` measured from the files on disk
    afterwards, plus the overshoots the worker reported.
    """
    import soundfile as sf

    gt_dir = tmp_path / "sliced_audios"
    k16_dir = tmp_path / "sliced_audios_16k"
    gt_dir.mkdir(exist_ok=True)
    k16_dir.mkdir(exist_ok=True)
    by_source = {}
    for name in files:
        audio, rate = sf.read(tmp_path / name)
        sf.write(gt_dir / name, audio.astype(np.float32), rate)
        sf.write(k16_dir / name, audio[::2].astype(np.float32), rate // 2)
        by_source.setdefault(preprocess.source_key(name), []).append(name)

    overshoots = {}
    for key, names in sorted(by_source.items()):
        _written, overshoot, _short = preprocess._apply_source_gain_worker(
            (key, sorted(names), str(gt_dir), str(k16_dir), ceiling_db, target_lufs)
        )
        overshoots[key] = overshoot

    result = {}
    for key, names in by_source.items():
        powers, peak = [], 0.0
        for name in sorted(names):
            for directory in (gt_dir, k16_dir):
                peak = max(peak, float(np.abs(sf.read(directory / name)[0]).max()))
            audio, rate = sf.read(gt_dir / name)
            powers.append(block_powers(audio, rate))
        result[key] = (loudness_from_blocks(np.concatenate(powers)), peak)
    return result, overshoots


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

    def group(values, key):
        return [values[f] for f in files if preprocess.source_key(f) == key]

    gap_before = abs(np.mean(group(before, "0_0")) - np.mean(group(before, "0_1")))
    assert gap_before == pytest.approx(18.0, abs=0.5)

    result, overshoots = _level_recordings(
        preprocess, tmp_path, files, sample_rate, -18.0
    )
    assert set(overshoots.values()) == {None}, "there is headroom here"

    # Between recordings: 18 dB apart, then on top of each other at the target.
    for key in ("0_0", "0_1"):
        assert result[key][0] == pytest.approx(-18.0, abs=0.05)

    # Within a recording: untouched, because one gain scales all of its slices.
    gt_dir = tmp_path / "sliced_audios"
    for key in ("0_0", "0_1"):
        after = {
            f: integrated_lufs(sf.read(gt_dir / f)[0], sample_rate)
            for f in files if preprocess.source_key(f) == key
        }
        span_before = max(group(before, key)) - min(group(before, key))
        span_after = max(after.values()) - min(after.values())
        assert span_before == pytest.approx(12.0, abs=0.5)
        assert span_after == pytest.approx(span_before, abs=1e-4)


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


def test_a_recording_short_of_headroom_is_limited_not_matched_down(tmp_path):
    """A recording whose peaks leave no headroom used to pull the target down
    for the whole dataset, so that nothing had to be limited.  It now reaches
    the target like everything else and its peaks are ducked instead.

    The old rule traded 22 dB of dataset level for 0.03% of samples; this is
    the same guarantee -- matched, under the ceiling -- bought the other way.
    """

    import soundfile as sf

    preprocess = _preprocess()
    sample_rate = 32000
    # One ordinary recording, one whose peaks leave almost no headroom.
    files = _write_dataset(tmp_path, sample_rate, {
        (0, 0): [-24.0, -24.0],
        (0, 1): [-1.0, -1.0],
    })

    result, overshoots = _level_recordings(
        preprocess, tmp_path, files, sample_rate, -6.0
    )
    # Both peak past the ceiling at this target, and both are limited for it.
    assert all(value is not None for value in overshoots.values())

    reached = {key: value[0] for key, value in result.items()}
    peaks = [value[1] for value in result.values()]

    # Matched, and at the target rather than 22 dB under it: the solver puts
    # them there despite the limiter taking energy back out.
    assert max(reached.values()) - min(reached.values()) < 0.1
    for value in reached.values():
        assert value == pytest.approx(-6.0, abs=0.1)
    # And nothing exceeds the ceiling, in either copy.
    assert max(peaks) <= 10 ** (-1.0 / 20) + 1e-6


def test_one_clicky_outlier_does_not_drag_the_whole_dataset_down(tmp_path):
    """The headroom limit is set by a recording's *peak*, so a single click
    used to decide the level for everyone.

    Measured on a real 116 h set: taking the worst of 24119 recordings gave
    -40.6 LUFS against a -18 target, 22 dB thrown away for one outlier with a
    39.6 dB crest.  The click is now ducked and nobody moves.
    """

    import soundfile as sf

    preprocess = _preprocess()
    sample_rate = 32000
    files = _write_dataset(
        tmp_path, sample_rate, {(0, i): [-24.0] for i in range(9)}
    )
    t = np.arange(int(1.2 * sample_rate)) / sample_rate
    clicky = 0.5 * np.sin(2 * math.pi * 220.0 * t) * 10 ** (-24.0 / 20)
    clicky[len(clicky) // 2] = 0.99          # one sample, ~30 dB above the rest
    sf.write(tmp_path / "0_9_0.wav", clicky.astype(np.float32), sample_rate)
    files = sorted(files + ["0_9_0.wav"])

    result, overshoots = _level_recordings(
        preprocess, tmp_path, files, sample_rate, -10.0
    )

    # The ordinary signal has a 6.6 dB crest, so at -10 LUFS it is clear of a
    # -1 dBFS ceiling; the clicky one has 35.3 dB and is not.  Under the old
    # rule that one recording would have set -36.3 LUFS for all ten.
    assert overshoots["0_9"] is not None
    assert [k for k, v in overshoots.items() if v is not None] == ["0_9"]

    # Every one of them lands on target -- the outlier included, because the
    # gain is solved against what the limiter leaves rather than against the
    # measurement the click inflated.
    for key, (loudness, _peak) in result.items():
        assert loudness == pytest.approx(-10.0, abs=0.1), key
    assert max(peak for _l, peak in result.values()) <= 10 ** (-1.0 / 20) + 1e-6


def test_the_gain_is_solved_against_the_limited_result(tmp_path):
    """Ducking a peak removes energy the loudness measurement had counted, so
    the gain that hits the target before limiting lands under it afterwards.

    On the real dataset the tail of that is severe: a recording of eating
    sounds came out 10.0 dB short, whispers 0.7-1.1 dB.  The worker solves for
    the gain whose *limited* result measures the target, so the recording that
    needs the limiter most is not the one left quietest.
    """

    import soundfile as sf

    preprocess = _preprocess()
    sample_rate = 32000
    gt_dir = tmp_path / "sliced_audios"
    k16_dir = tmp_path / "sliced_audios_16k"
    gt_dir.mkdir()
    k16_dir.mkdir()

    # A recording whose energy is mostly in transients: the limiter has to take
    # a lot out, which is exactly the case one round of gain gets wrong.
    t = np.arange(int(1.2 * sample_rate)) / sample_rate
    audio = 0.5 * np.sin(2 * math.pi * 220.0 * t) * 10 ** (-30.0 / 20)
    audio[::1500] = 0.4                       # a spike every 47 ms
    names = ["0_0_0.wav", "0_0_1.wav"]
    for name in names:
        sf.write(gt_dir / name, audio.astype(np.float32), sample_rate)
        sf.write(k16_dir / name, audio[::2].astype(np.float32), 16000)

    # What one round of gain would have reached: the gain that puts the
    # *unlimited* recording on target, which is where the worker starts.
    powers = [block_powers(sf.read(gt_dir / n)[0], sample_rate) for n in names]
    unsolved = 10 ** ((-18.0 - loudness_from_blocks(np.concatenate(powers))) / 20)
    powers = [
        block_powers(
            limit_peaks(
                sf.read(gt_dir / n)[0] * unsolved, sample_rate, ceiling_db=-1.0
            ),
            sample_rate,
        )
        for n in names
    ]
    naive = loudness_from_blocks(np.concatenate(powers))
    assert naive < -18.0 - 1.0, "the fixture is not exercising the solver"

    preprocess._apply_source_gain_worker(
        ("0_0", names, str(gt_dir), str(k16_dir), -1.0, -18.0)
    )

    powers = [
        block_powers(sf.read(gt_dir / n)[0], sample_rate) for n in names
    ]
    assert loudness_from_blocks(np.concatenate(powers)) == pytest.approx(
        -18.0, abs=preprocess.GAIN_SOLVE_TOLERANCE_DB + 0.01
    )
    # The ceiling still holds, in both copies.
    for directory in (gt_dir, k16_dir):
        for name in names:
            assert np.abs(sf.read(directory / name)[0]).max() <= 10 ** (-1 / 20) + 1e-6


def test_a_recording_that_saturates_below_the_target_says_so(tmp_path):
    """Past a point the limiter has flattened a recording and no further gain
    raises its loudness, so a target above that point is simply unreachable.

    Measured on one recording of the real dataset -- 34.3 dB of crest, sparse
    transients over near-silence -- which tops out at -17.3 LUFS however hard
    it is driven.  The worker reports the shortfall rather than leaving it to
    be noticed later as a quiet outlier.
    """

    import soundfile as sf

    preprocess = _preprocess()
    sample_rate = 32000
    gt_dir = tmp_path / "sliced_audios"
    k16_dir = tmp_path / "sliced_audios_16k"
    gt_dir.mkdir()
    k16_dir.mkdir()

    # Short bursts separated by silence: the peaks carry essentially all of
    # the energy, so once the limiter has flattened them there is nothing left
    # for more gain to raise.  (Spikes over a noise floor do *not* saturate --
    # the floor keeps scaling linearly for ever.)
    audio = np.zeros(int(2.0 * sample_rate), dtype=np.float64)
    burst = int(0.005 * sample_rate)
    for start in range(0, len(audio) - burst, int(0.2 * sample_rate)):
        t = np.arange(burst) / sample_rate
        audio[start:start + burst] = (
            0.5 * np.sin(2 * math.pi * 300.0 * t) * np.hanning(burst)
        )
    names = ["0_0_0.wav"]
    for name in names:
        sf.write(gt_dir / name, audio.astype(np.float32), sample_rate)
        sf.write(k16_dir / name, audio[::2].astype(np.float32), 16000)

    _written, _overshoot, shortfall = preprocess._apply_source_gain_worker(
        ("0_0", names, str(gt_dir), str(k16_dir), -0.3, -3.0)
    )
    assert shortfall is not None and shortfall > 0.1

    # It still went as loud as it goes, and still under the ceiling.
    powers = [block_powers(sf.read(gt_dir / n)[0], sample_rate) for n in names]
    assert loudness_from_blocks(np.concatenate(powers)) == pytest.approx(
        -3.0 - shortfall, abs=0.2
    )
    for directory in (gt_dir, k16_dir):
        assert np.abs(sf.read(directory / names[0])[0]).max() <= 10 ** (-0.3 / 20) + 1e-6


def test_a_recording_scope_mode_is_the_default_everywhere():
    """The default is ``pre_peak_rvc``, and it has to be the same one in all
    four places that carry one.

    Both offered ``pre_`` modes are named for what their result is equivalent
    to -- normalising the recording before it was sliced -- not for when they
    run, which is after slicing like every other mode here.  What matters for
    the default is that it is *some* recording-scope mode: a per-slice default
    would flatten the dynamics between phrases of every dataset built without
    touching the setting.

    Pinned across all four surfaces because they used to disagree -- the CLI
    signature said ``pre_loudness`` while its own flag and both UIs said
    ``post_peak``, so the default depended on how preprocessing was started.
    """

    from gui.services import catalog

    assert catalog.NORMALIZATION_MODES[-2] == "pre_peak_rvc"

    core = (ROOT / "core.py").read_text(encoding="utf-8")
    assert 'normalization_mode: str = "pre_peak_rvc"' in core
    assert "default='pre_peak_rvc'" in core

    gradio = (ROOT / "tabs" / "train" / "train.py").read_text(encoding="utf-8")
    assert 'value="pre_peak_rvc"' in gradio

    gui = (ROOT / "gui" / "views" / "training.py").read_text(encoding="utf-8")
    assert 'set_text("pre_peak_rvc")' in gui


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


def test_recording_scope_modes_do_not_get_a_second_pass():
    """A recording-scope mode writes its slices back in its own branch.

    The generic per-slice pass that follows has to skip those modes.  It used
    to skip only ``pre_loudness`` by name, so adding ``pre_peak_rvc`` sent it
    through both: harmless numerically -- ``_apply_post_norm`` has no branch
    for it and returns the audio untouched -- but a full re-read and re-write
    of the dataset to apply nothing, which on a 100-hour set is not free.
    """

    source = (ROOT / "rvc" / "train" / "preprocess" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    assert 'RECORDING_SCOPE_MODES = ("pre_loudness", "pre_peak_rvc")' in source
    assert "if normalization_mode not in RECORDING_SCOPE_MODES:" in source
    assert 'if normalization_mode != "pre_loudness":' not in source
