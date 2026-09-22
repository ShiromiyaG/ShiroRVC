"""Residual augmentation for the pretrain.  The coarse melody is kept, so the
descriptors recomputed afterwards describe the same notes sung differently."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy import signal

from .descriptors import DescriptorConfig, _filt, _runs, analyze, descriptor_vector, detect_notes
from .f0_repr import ReprConfig, compose_f0


@dataclass(frozen=True)
class AugmentConfig:
    prob: float = 0.8
    vibrato_extent: tuple = (0.5, 2.0)
    vibrato_rate: tuple = (0.8, 1.25)
    scoop_prob: float = 0.5
    #: 0 removes the scoop, above 1 exaggerates it.
    scoop_gain: tuple = (0.0, 2.5)
    #: Plain attacks turned into scoops, so the scoop fraction varies apart from the depth.
    scoop_add_prob: float = 0.5
    scoop_add_cents: tuple = (60.0, 600.0)
    scoop_add_seconds: tuple = (0.05, 0.35)
    drop_prob: float = 0.5
    #: 0 flattens the phrase end, above 1 exaggerates its drop or rise.
    drop_gain: tuple = (0.0, 2.5)
    #: Flat phrase ends given a drop.
    drop_add_prob: float = 0.5
    drop_add_cents: tuple = (40.0, 400.0)
    drop_add_seconds: tuple = (0.1, 0.3)
    coarse_shift_semitones: float = 3.0

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
    sos = signal.butter(2, dcfg.vibrato_band_hz, btype="band", fs=fr, output="sos")
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


def augment(coarse_hz, residual, vuv, rcfg: ReprConfig, dcfg: DescriptorConfig, acfg: AugmentConfig, rng):
    """``(coarse_hz, residual, descriptors)`` after augmentation; descriptors
    are ``None`` when nothing but the coarse shift was applied."""
    coarse_hz = np.asarray(coarse_hz, dtype=np.float64)
    if acfg.coarse_shift_semitones > 0:
        coarse_hz = coarse_hz * 2.0 ** (rng.uniform(-1, 1) * acfg.coarse_shift_semitones / 12.0)
    if rng.random() >= acfg.prob:
        return coarse_hz.astype(np.float32), residual, None

    fr = rcfg.frame_rate
    res = np.asarray(residual, dtype=np.float64).copy()
    vuv = np.asarray(vuv, dtype=bool)
    notes, voiced_runs, _ = detect_notes(compose_f0(coarse_hz, res, vuv), rcfg, dcfg)
    if notes:
        extent = _log_uniform(rng, *acfg.vibrato_extent)
        rate = _log_uniform(rng, *acfg.vibrato_rate)
        _reshape_vibrato(res, notes, fr, dcfg, extent, rate)
        if rng.random() < acfg.scoop_prob:
            _reshape_attacks(res, notes, voiced_runs, fr, dcfg, rng.uniform(*acfg.scoop_gain))
        if rng.random() < acfg.scoop_add_prob:
            _add_scoops(res, notes, voiced_runs, fr, dcfg, acfg, rng)
        if rng.random() < acfg.drop_prob:
            _reshape_drops(res, notes, voiced_runs, fr, dcfg, rng.uniform(*acfg.drop_gain))
        if rng.random() < acfg.drop_add_prob:
            _add_drops(res, notes, voiced_runs, fr, dcfg, acfg, rng)

    clip = rcfg.residual_clip_cents
    res = np.where(vuv, np.clip(res, -clip, clip), 0.0).astype(np.float32)
    f0 = compose_f0(coarse_hz, res, vuv)
    return coarse_hz.astype(np.float32), res, descriptor_vector(analyze(f0, rcfg, dcfg, res))
