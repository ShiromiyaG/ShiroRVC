"""Style descriptors: a small vector that summarises how a singer shapes F0.

``analyze`` turns one contour into events (notes, vibrato, attacks,
transitions, phrase ends); ``pool`` merges the events of many clips; each
descriptor is its own function over the pooled events, so a singer's value is
pooled over notes rather than averaged over per-clip values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy import signal

from .f0_repr import ReprConfig, _runs, bridge_gaps, hz_to_cents, interpolate_unvoiced, voiced_mask


@dataclass(frozen=True)
class DescriptorConfig:
    bridge_gap_s: float = 0.03
    stable_slope_cents_s: float = 200.0
    min_note_s: float = 0.12
    merge_gap_s: float = 0.1
    merge_cents: float = 50.0
    vibrato_min_note_s: float = 0.35
    vibrato_band_hz: tuple = (3.5, 9.0)
    vibrato_peak_hz: tuple = (4.0, 8.5)
    vibrato_min_ratio: float = 0.5
    vibrato_min_extent_cents: float = 15.0
    attack_window_s: float = 0.3
    scoop_min_cents: float = 40.0
    settle_cents: float = 20.0
    overshoot_window_s: float = 0.15
    phrase_gap_s: float = 0.2
    phrase_tail_s: float = 0.1
    drop_min_cents: float = 40.0
    jitter_highpass_hz: float = 12.0
    jitter_min_note_s: float = 0.2

    @classmethod
    def from_dict(cls, data: dict | None) -> "DescriptorConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass
class Events:
    """Everything the descriptors read.  Lists pool by concatenation."""

    vibrato_rate: list = field(default_factory=list)
    vibrato_extent: list = field(default_factory=list)
    vibrato_onset: list = field(default_factory=list)
    vibrato_eligible: int = 0
    attack_kind: list = field(default_factory=list)  # -1 scoop, +1 fall-in, 0 plain
    scoop_depth: list = field(default_factory=list)
    scoop_duration: list = field(default_factory=list)
    transition_time: list = field(default_factory=list)
    transition_overshoot: list = field(default_factory=list)
    phrase_end_drop: list = field(default_factory=list)
    jitter_sq: float = 0.0
    jitter_n: int = 0
    residual_sq: float = 0.0
    residual_n: int = 0


def pool(events_list) -> Events:
    out = Events()
    for ev in events_list:
        for f in fields(Events):
            value = getattr(ev, f.name)
            if isinstance(value, list):
                getattr(out, f.name).extend(value)
            else:
                setattr(out, f.name, getattr(out, f.name) + value)
    return out


# --- detection -------------------------------------------------------------


def _filt(x, sos):
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def smooth_cents(f0_hz: np.ndarray, cfg: ReprConfig) -> np.ndarray | None:
    """The coarse contour before quantization, which is what note detection
    reads: quantization steps would read as slope."""
    vuv = voiced_mask(f0_hz)
    if vuv.sum() < 4:
        return None
    cents = interpolate_unvoiced(hz_to_cents(f0_hz), vuv)
    sos = signal.butter(4, cfg.coarse_cutoff_hz, fs=cfg.frame_rate, output="sos")
    padlen = min(len(cents) - 1, int(3 * cfg.frame_rate / cfg.coarse_cutoff_hz))
    return signal.sosfiltfilt(sos, cents, padtype="even", padlen=padlen)


@dataclass
class Note:
    start: int
    end: int
    pitch: float
    run: int


def detect_notes(f0_hz: np.ndarray, cfg: ReprConfig, dcfg: DescriptorConfig):
    """Notes from the smoothed coarse contour: voiced stretches where it is
    flat, merged when they are the same pitch.  Returns ``(notes, voiced_runs,
    smooth)``."""
    fr = cfg.frame_rate
    vuv = bridge_gaps(voiced_mask(f0_hz), int(round(dcfg.bridge_gap_s * fr)))
    voiced_runs = _runs(vuv)
    smooth = smooth_cents(f0_hz, cfg)
    if smooth is None:
        return [], voiced_runs, None

    run_id = np.full(len(vuv), -1)
    for i, (s, e) in enumerate(voiced_runs):
        run_id[s:e] = i
    slope = np.abs(np.gradient(smooth)) * fr
    stable = vuv & (slope < dcfg.stable_slope_cents_s)

    notes: list[Note] = []
    for s, e in _runs(stable):
        pitch = float(np.median(smooth[s:e]))
        prev = notes[-1] if notes else None
        if (
            prev is not None
            and prev.run == run_id[s]
            and s - prev.end <= dcfg.merge_gap_s * fr
            and abs(prev.pitch - pitch) < dcfg.merge_cents
        ):
            prev.end = e
            prev.pitch = float(np.median(smooth[prev.start:e]))
            continue
        notes.append(Note(int(s), int(e), pitch, int(run_id[s])))
    min_len = dcfg.min_note_s * fr
    return [n for n in notes if n.end - n.start >= min_len], voiced_runs, smooth


def _vibrato(dev: np.ndarray, fr: float, dcfg: DescriptorConfig):
    """``(rate_hz, extent_cents, onset_s)`` or ``None`` when the note has none."""
    dev = signal.detrend(dev)
    sos = signal.butter(2, dcfg.vibrato_band_hz, btype="band", fs=fr, output="sos")
    vib = _filt(dev, sos)
    total = float(np.mean(dev**2))
    if total <= 0:
        return None
    ratio = float(np.mean(vib**2)) / total

    spec = np.abs(np.fft.rfft(dev * np.hanning(len(dev)), n=1024)) ** 2
    freqs = np.fft.rfftfreq(1024, 1.0 / fr)
    lo, hi = dcfg.vibrato_peak_hz
    band = (freqs >= lo) & (freqs <= hi)
    peak = freqs[band][np.argmax(spec[band])]
    if not (lo < peak < hi) or ratio < dcfg.vibrato_min_ratio:
        return None

    env = np.abs(signal.hilbert(vib))
    active = np.flatnonzero(env >= 0.5 * env.max())
    first, last = active[0], active[-1] + 1
    extent = float(np.sqrt(2.0) * np.sqrt(np.mean(vib[first:last] ** 2)))
    if extent < dcfg.vibrato_min_extent_cents:
        return None
    seg = vib[first:last]
    crossings = np.count_nonzero(np.diff(np.signbit(seg)))
    duration = (last - first) / fr
    rate = crossings / (2.0 * duration) if crossings >= 2 else float(peak)
    return float(rate), extent, float(first / fr)


def analyze(f0_hz: np.ndarray, cfg: ReprConfig, dcfg: DescriptorConfig, residual=None) -> Events:
    """Events of one contour.  ``residual`` (cents, from ``f0_repr``) feeds the
    residual RMS; without it that descriptor stays empty for this clip."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    ev = Events()
    fr = cfg.frame_rate
    vuv = voiced_mask(f0_hz)
    if residual is not None and vuv.any():
        r = np.asarray(residual, dtype=np.float64)[vuv]
        ev.residual_sq, ev.residual_n = float(np.sum(r**2)), int(r.size)

    notes, voiced_runs, smooth = detect_notes(f0_hz, cfg, dcfg)
    if not notes:
        return ev
    fine = interpolate_unvoiced(hz_to_cents(f0_hz), vuv)
    n_frames = len(f0_hz)
    # Deviations past what the residual can hold are pitch-tracking errors.
    max_dev = cfg.residual_clip_cents
    jitter_sos = signal.butter(2, dcfg.jitter_highpass_hz, btype="high", fs=fr, output="sos")

    for i, note in enumerate(notes):
        dev = fine[note.start:note.end] - note.pitch
        length = note.end - note.start

        if length >= dcfg.vibrato_min_note_s * fr:
            ev.vibrato_eligible += 1
            found = _vibrato(dev, fr, dcfg)
            if found is not None:
                ev.vibrato_rate.append(found[0])
                ev.vibrato_extent.append(found[1])
                ev.vibrato_onset.append(found[2])

        if length >= dcfg.jitter_min_note_s * fr:
            hp = _filt(dev, jitter_sos)
            ev.jitter_sq += float(np.sum(hp**2))
            ev.jitter_n += int(hp.size)

        prev = notes[i - 1] if i > 0 else None
        run_start, run_end = voiced_runs[note.run]
        if prev is None or prev.run != note.run:
            # Attack after silence.  Skipped when the run starts at the clip
            # edge: the onset may lie before the clip.
            if run_start > 0:
                stop = min(note.start + int(dcfg.attack_window_s * fr), note.end)
                attack = fine[run_start:stop] - note.pitch
                d0 = float(np.median(attack[:3])) if attack.size >= 3 else np.nan
                if abs(d0) <= max_dev:
                    kind = 0
                    if d0 <= -dcfg.scoop_min_cents:
                        kind = -1
                        settled = np.flatnonzero(attack >= -dcfg.settle_cents)
                        dur = settled[0] / fr if settled.size else attack.size / fr
                        ev.scoop_depth.append(-d0)
                        ev.scoop_duration.append(float(dur))
                    elif d0 >= dcfg.scoop_min_cents:
                        kind = 1
                    ev.attack_kind.append(kind)
        else:
            ev.transition_time.append((note.start - prev.end) / fr)
            direction = np.sign(note.pitch - prev.pitch)
            stop = min(note.start + int(dcfg.overshoot_window_s * fr), note.end)
            window = (fine[prev.end:stop] - note.pitch) * direction
            ev.transition_overshoot.append(float(max(0.0, window.max())) if window.size else 0.0)

        nxt = notes[i + 1] if i + 1 < len(notes) else None
        if nxt is None or nxt.run != note.run:
            # Phrase end: needs a real gap after the run, inside the clip.
            gap_end = voiced_runs[note.run + 1][0] if note.run + 1 < len(voiced_runs) else n_frames
            if run_end < n_frames and gap_end - run_end >= dcfg.phrase_gap_s * fr:
                tail = int(dcfg.phrase_tail_s * fr)
                body = fine[note.start:note.end] - note.pitch
                end_dev = fine[max(note.start, run_end - tail):run_end] - note.pitch
                drop = float(np.mean(end_dev) - np.median(body)) if end_dev.size else np.nan
                if abs(drop) <= max_dev:
                    ev.phrase_end_drop.append(drop)
    return ev


# --- descriptors -----------------------------------------------------------


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _median(values) -> float:
    return float(np.median(values)) if len(values) else np.nan


def vibrato_rate_hz(ev: Events) -> float:
    return _mean(ev.vibrato_rate)


def vibrato_extent_cents(ev: Events) -> float:
    return _mean(ev.vibrato_extent)


def vibrato_onset_s(ev: Events) -> float:
    return _mean(ev.vibrato_onset)


def vibrato_fraction(ev: Events) -> float:
    return len(ev.vibrato_rate) / ev.vibrato_eligible if ev.vibrato_eligible else np.nan


def vibrato_rate_std_hz(ev: Events) -> float:
    return float(np.std(ev.vibrato_rate)) if len(ev.vibrato_rate) >= 2 else np.nan


def scoop_fraction(ev: Events) -> float:
    return float(np.mean(np.asarray(ev.attack_kind) == -1)) if ev.attack_kind else np.nan


def scoop_depth_cents(ev: Events) -> float:
    return _median(ev.scoop_depth)


def scoop_duration_s(ev: Events) -> float:
    return _mean(ev.scoop_duration)


def fall_in_fraction(ev: Events) -> float:
    return float(np.mean(np.asarray(ev.attack_kind) == 1)) if ev.attack_kind else np.nan


def transition_time_s(ev: Events) -> float:
    return _mean(ev.transition_time)


def transition_overshoot_cents(ev: Events) -> float:
    return _mean(ev.transition_overshoot)


def phrase_end_drop_cents(ev: Events) -> float:
    return _median(ev.phrase_end_drop)


def phrase_end_drop_fraction(ev: Events, min_cents: float = DescriptorConfig.drop_min_cents) -> float:
    if not ev.phrase_end_drop:
        return np.nan
    return float(np.mean(np.asarray(ev.phrase_end_drop) <= -min_cents))


def jitter_cents(ev: Events) -> float:
    return float(np.sqrt(ev.jitter_sq / ev.jitter_n)) if ev.jitter_n else np.nan


def residual_rms_cents(ev: Events) -> float:
    return float(np.sqrt(ev.residual_sq / ev.residual_n)) if ev.residual_n else np.nan


def legato_fraction(ev: Events) -> float:
    total = len(ev.transition_time) + len(ev.attack_kind)
    return len(ev.transition_time) / total if total else np.nan


DESCRIPTORS = {
    "vibrato_rate_hz": vibrato_rate_hz,
    "vibrato_extent_cents": vibrato_extent_cents,
    "vibrato_onset_s": vibrato_onset_s,
    "vibrato_fraction": vibrato_fraction,
    "vibrato_rate_std_hz": vibrato_rate_std_hz,
    "scoop_fraction": scoop_fraction,
    "scoop_depth_cents": scoop_depth_cents,
    "scoop_duration_s": scoop_duration_s,
    "fall_in_fraction": fall_in_fraction,
    "transition_time_s": transition_time_s,
    "transition_overshoot_cents": transition_overshoot_cents,
    "phrase_end_drop_cents": phrase_end_drop_cents,
    "phrase_end_drop_fraction": phrase_end_drop_fraction,
    "jitter_cents": jitter_cents,
    "residual_rms_cents": residual_rms_cents,
    "legato_fraction": legato_fraction,
}
DESCRIPTOR_NAMES = tuple(DESCRIPTORS)


def descriptor_vector(ev: Events) -> np.ndarray:
    """``(len(DESCRIPTOR_NAMES),)`` float32, NaN where there were no events."""
    return np.array([fn(ev) for fn in DESCRIPTORS.values()], dtype=np.float32)


def describe(f0_list, cfg: ReprConfig, dcfg: DescriptorConfig, residual_list=None) -> np.ndarray:
    """Pooled descriptor vector of a set of contours."""
    if residual_list is None:
        residual_list = [None] * len(f0_list)
    return descriptor_vector(pool(analyze(f0, cfg, dcfg, r) for f0, r in zip(f0_list, residual_list)))
