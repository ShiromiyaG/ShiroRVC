"""Residual augmentation for the pretrain.  The coarse melody is kept, so the
descriptors recomputed afterwards describe the same notes sung differently."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy import signal

from .descriptors import DescriptorConfig, _filt, _runs, analyze, descriptor_vector, detect_notes
from .f0_repr import ReprConfig, butter_sos, compose_f0, hz_to_cents


@dataclass(frozen=True)
class AugmentConfig:
    prob: float = 0.8
    vibrato_extent: tuple = (0.5, 2.0)
    vibrato_rate: tuple = (0.8, 1.25)
    scoop_prob: float = 0.5
    #: 0 removes the scoop, above 1 exaggerates it.
    scoop_gain: tuple = (0.0, 2.5)
    #: Every attack flattened instead, so a converter learns to add the scoops
    #: its descriptors ask for rather than only to keep or drop the source's.
    scoop_flat_prob: float = 0.0
    #: Attacks off their note flattened one at a time, each with a chance drawn per clip.
    scoop_remove_prob: float = 0.0
    #: Plain attacks turned into scoops, so the scoop fraction varies apart from the depth.
    scoop_add_prob: float = 0.5
    scoop_add_cents: tuple = (60.0, 600.0)
    scoop_add_seconds: tuple = (0.05, 0.35)
    drop_prob: float = 0.5
    #: 0 flattens the phrase end, above 1 exaggerates its drop or rise.
    drop_gain: tuple = (0.0, 2.5)
    #: Every phrase end flattened instead, for the same reason as ``scoop_flat_prob``.
    drop_flat_prob: float = 0.0
    #: Phrase ends off their note flattened one at a time, each with a chance drawn per clip.
    drop_remove_prob: float = 0.0
    #: Flat phrase ends given a drop.
    drop_add_prob: float = 0.5
    drop_add_cents: tuple = (40.0, 400.0)
    drop_add_seconds: tuple = (0.1, 0.3)
    #: Glides between notes stretched or shortened by this factor.
    transition_prob: float = 0.0
    transition_time: tuple = (0.5, 2.0)
    #: 0 removes the overshoot past the new note, above 1 exaggerates it.
    overshoot_prob: float = 0.0
    overshoot_gain: tuple = (0.0, 2.0)
    #: Transitions without an overshoot given one.
    overshoot_add_prob: float = 0.0
    overshoot_add_cents: tuple = (40.0, 300.0)
    overshoot_add_seconds: tuple = (0.05, 0.15)
    coarse_shift_semitones: float = 3.0
    #: The clip's level swing scaled by this, since a source at inference
    #: can be mixed or compressed unlike the training singer.
    loudness_scale: tuple = (1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AugmentConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _reshape_vibrato(res, notes, fr, dcfg, extent, rate):
    """Scale the vibrato band of each long note in amplitude and, through its
    instantaneous phase, in rate, leaving the envelope and timing alone."""
    sos = butter_sos(2, dcfg.vibrato_band_hz, fr, "band")
    for note in notes:
        if note.end - note.start < dcfg.vibrato_min_note_s * fr:
            continue
        seg = res[note.start:note.end]
        vib = _filt(seg, sos)
        analytic = signal.hilbert(vib)
        phase = np.unwrap(np.angle(analytic))
        new = extent * np.abs(analytic) * np.cos(phase[0] + rate * (phase - phase[0]))
        res[note.start:note.end] = seg - vib + new


def _attacks(notes, voiced_runs, fr, dcfg):
    """``(note, start, stop)`` of each attack after silence inside the clip."""
    for i, note in enumerate(notes):
        prev = notes[i - 1] if i > 0 else None
        if prev is not None and prev.run == note.run:
            continue
        start = voiced_runs[note.run][0]
        stop = min(note.start + int(dcfg.attack_window_s * fr), note.end)
        if start > 0 and stop - start >= 3:
            yield note, start, stop


def _phrase_ends(notes, voiced_runs, n_frames, fr, dcfg):
    """``(note, start, end)`` of the last attack window of each phrase that
    ends inside the clip."""
    for i, note in enumerate(notes):
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        if nxt is not None and nxt.run == note.run:
            continue
        end = voiced_runs[note.run][1]
        start = max(note.start, end - int(dcfg.attack_window_s * fr))
        if end < n_frames and end - start >= 3:
            yield note, start, end


def _taper(n, gain):
    """``gain`` at the first frame, easing to 1 by the last."""
    return gain + (1.0 - gain) * 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))


def _reshape_attacks(res, notes, voiced_runs, fr, dcfg, gain):
    """Scale the deviation at each attack after silence, tapering back to the
    original by the end of the attack window."""
    for note, start, stop in _attacks(notes, voiced_runs, fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        res[start:stop] = level + (res[start:stop] - level) * _taper(stop - start, gain)


def _remove_scoops(res, notes, voiced_runs, fr, dcfg, rng):
    """Flatten attacks that start off their note, each with a chance drawn per clip."""
    chance = rng.random()
    for note, start, stop in _attacks(notes, voiced_runs, fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        if abs(np.median(res[start:start + 3]) - level) < dcfg.scoop_min_cents or rng.random() >= chance:
            continue
        res[start:stop] = level + (res[start:stop] - level) * _taper(stop - start, 0.0)


def _add_scoops(res, notes, voiced_runs, fr, dcfg, acfg, rng):
    """Start plain attacks from below, each with a chance drawn per clip."""
    chance = rng.random()
    for note, start, _ in _attacks(notes, voiced_runs, fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        if abs(np.median(res[start:start + 3]) - level) >= dcfg.scoop_min_cents or rng.random() >= chance:
            continue
        n = min(int(rng.uniform(*acfg.scoop_add_seconds) * fr), note.end - start)
        if n >= 3:
            res[start:start + n] -= _log_uniform(rng, *acfg.scoop_add_cents) * (1.0 - _taper(n, 0.0))


def _reshape_drops(res, notes, voiced_runs, fr, dcfg, gain):
    """Scale the deviation at each phrase end, from the original at the start
    of the window to ``gain`` at the last voiced frame."""
    for note, start, end in _phrase_ends(notes, voiced_runs, len(res), fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        res[start:end] = level + (res[start:end] - level) * _taper(end - start, gain)[::-1]


def _remove_drops(res, notes, voiced_runs, fr, dcfg, rng):
    """Flatten phrase ends that leave their note, each with a chance drawn per clip."""
    chance = rng.random()
    tail = int(dcfg.phrase_tail_s * fr)
    for note, start, end in _phrase_ends(notes, voiced_runs, len(res), fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        if abs(np.mean(res[max(note.start, end - tail):end]) - level) < dcfg.drop_min_cents or rng.random() >= chance:
            continue
        res[start:end] = level + (res[start:end] - level) * _taper(end - start, 0.0)[::-1]


def _add_drops(res, notes, voiced_runs, fr, dcfg, acfg, rng):
    """Bend flat phrase ends down, each with a chance drawn per clip."""
    chance = rng.random()
    tail = int(dcfg.phrase_tail_s * fr)
    for note, _, end in _phrase_ends(notes, voiced_runs, len(res), fr, dcfg):
        level = float(np.median(res[note.start:note.end]))
        if abs(np.mean(res[max(note.start, end - tail):end]) - level) >= dcfg.drop_min_cents or rng.random() >= chance:
            continue
        n = min(int(rng.uniform(*acfg.drop_add_seconds) * fr), end - note.start)
        if n >= 3:
            res[end - n:end] -= _log_uniform(rng, *acfg.drop_add_cents) * (1.0 - _taper(n, 0.0))[::-1]


def _transitions(notes, dcfg):
    """``(prev, note)`` of each glide between two notes of one voiced run."""
    for prev, note in zip(notes, notes[1:]):
        if prev.run == note.run and abs(note.pitch - prev.pitch) >= dcfg.merge_cents:
            yield prev, note


def _ease(n_frames, start, stop):
    """0 before ``start``, a raised cosine to 1 at ``stop``, 1 after."""
    t = np.clip((np.arange(n_frames) - start) / max(stop - start, 1), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _retime_transitions(res, notes, dcfg, factor):
    """Stretch each glide around its midpoint, into at most half of either
    note, keeping the detail riding on it."""
    for prev, note in _transitions(notes, dcfg):
        lo = prev.end - (prev.end - prev.start) // 2
        hi = note.start + (note.end - note.start) // 2
        mid = (prev.end + note.start) / 2
        half = min(factor * (note.start - prev.end) / 2, mid - lo, hi - mid)
        s, e = int(round(mid - half)), int(round(mid + half))
        a, b = min(s, prev.end), max(e, note.start)
        jump = note.pitch - prev.pitch
        res[a:b] += jump * (_ease(b - a, s - a, e - a) - _ease(b - a, prev.end - a, note.start - a))


def _reshape_overshoots(res, cents, notes, fr, dcfg, gain):
    """Scale how far each note is overshot past its pitch, easing back to the
    original by the end of the overshoot window."""
    for prev, note in _transitions(notes, dcfg):
        n = min(int(dcfg.overshoot_window_s * fr), note.end - note.start)
        if n < 3:
            continue
        direction = np.sign(note.pitch - prev.pitch)
        seg = slice(note.start, note.start + n)
        past = np.maximum(0.0, (cents[seg] + res[seg] - note.pitch) * direction)
        res[seg] += direction * past * (_taper(n, gain) - 1.0)


def _add_overshoots(res, cents, notes, fr, dcfg, acfg, rng):
    """Carry glides past the new note and back, each with a chance drawn per clip."""
    chance = rng.random()
    for prev, note in _transitions(notes, dcfg):
        direction = np.sign(note.pitch - prev.pitch)
        window = slice(note.start, min(note.start + int(dcfg.overshoot_window_s * fr), note.end))
        past = (cents[window] + res[window] - note.pitch) * direction
        if (past.size and past.max() >= dcfg.scoop_min_cents) or rng.random() >= chance:
            continue
        n = min(int(rng.uniform(*acfg.overshoot_add_seconds) * fr), note.end - note.start)
        if n >= 3:
            res[note.start:note.start + n] += direction * _log_uniform(rng, *acfg.overshoot_add_cents) * np.sin(np.pi * np.arange(n) / n)


def reshape(coarse_hz, residual, vuv, rcfg: ReprConfig, dcfg: DescriptorConfig, acfg: AugmentConfig, rng):
    """``residual`` with its vibrato, attacks and phrase ends re-sung as
    ``acfg`` draws them, and its transitions when ``acfg`` asks; the notes
    (``coarse_hz``) are left alone."""
    fr = rcfg.frame_rate
    res = np.asarray(residual, dtype=np.float64).copy()
    vuv = np.asarray(vuv, dtype=bool)
    notes, voiced_runs, _ = detect_notes(compose_f0(coarse_hz, res, vuv), rcfg, dcfg)
    if notes:
        extent = _log_uniform(rng, *acfg.vibrato_extent)
        rate = _log_uniform(rng, *acfg.vibrato_rate)
        _reshape_vibrato(res, notes, fr, dcfg, extent, rate)
        # A zero probability draws nothing, so configs without these keep their random stream.
        if acfg.scoop_flat_prob and rng.random() < acfg.scoop_flat_prob:
            _reshape_attacks(res, notes, voiced_runs, fr, dcfg, 0.0)
        elif rng.random() < acfg.scoop_prob:
            _reshape_attacks(res, notes, voiced_runs, fr, dcfg, rng.uniform(*acfg.scoop_gain))
        if acfg.scoop_remove_prob and rng.random() < acfg.scoop_remove_prob:
            _remove_scoops(res, notes, voiced_runs, fr, dcfg, rng)
        if rng.random() < acfg.scoop_add_prob:
            _add_scoops(res, notes, voiced_runs, fr, dcfg, acfg, rng)
        if acfg.drop_flat_prob and rng.random() < acfg.drop_flat_prob:
            _reshape_drops(res, notes, voiced_runs, fr, dcfg, 0.0)
        elif rng.random() < acfg.drop_prob:
            _reshape_drops(res, notes, voiced_runs, fr, dcfg, rng.uniform(*acfg.drop_gain))
        if acfg.drop_remove_prob and rng.random() < acfg.drop_remove_prob:
            _remove_drops(res, notes, voiced_runs, fr, dcfg, rng)
        if rng.random() < acfg.drop_add_prob:
            _add_drops(res, notes, voiced_runs, fr, dcfg, acfg, rng)
        if rng.random() < acfg.transition_prob:
            _retime_transitions(res, notes, dcfg, _log_uniform(rng, *acfg.transition_time))
        cents = hz_to_cents(np.asarray(coarse_hz, dtype=np.float64))
        if rng.random() < acfg.overshoot_prob:
            _reshape_overshoots(res, cents, notes, fr, dcfg, rng.uniform(*acfg.overshoot_gain))
        if rng.random() < acfg.overshoot_add_prob:
            _add_overshoots(res, cents, notes, fr, dcfg, acfg, rng)
    clip = rcfg.residual_clip_cents
    return np.where(vuv, np.clip(res, -clip, clip), 0.0).astype(np.float32)


def augment(coarse_hz, residual, vuv, rcfg: ReprConfig, dcfg: DescriptorConfig, acfg: AugmentConfig, rng):
    """``(coarse_hz, residual, descriptors)`` after augmentation; descriptors
    are ``None`` when nothing but the coarse shift was applied."""
    coarse_hz = np.asarray(coarse_hz, dtype=np.float64)
    if acfg.coarse_shift_semitones > 0:
        coarse_hz = coarse_hz * 2.0 ** (rng.uniform(-1, 1) * acfg.coarse_shift_semitones / 12.0)
    if rng.random() >= acfg.prob:
        return coarse_hz.astype(np.float32), residual, None
    res = reshape(coarse_hz, residual, vuv, rcfg, dcfg, acfg, rng)
    f0 = compose_f0(coarse_hz, res, np.asarray(vuv, dtype=bool))
    return coarse_hz.astype(np.float32), res, descriptor_vector(analyze(f0, rcfg, dcfg, res))
