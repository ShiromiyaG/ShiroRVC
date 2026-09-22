"""F0 representation: coarse melody, fine residual and voicing on RVC's frame grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy import signal

#: RVC's pitch grid: 16 kHz audio, hop 160.
FRAME_RATE = 100.0
#: Cents are measured from here so every audible F0 is positive.
CENTS_REF_HZ = 10.0


@dataclass(frozen=True)
class ReprConfig:
    frame_rate: float = FRAME_RATE
    coarse_rate_hz: float = 10.0
    #: Zero-phase low-pass applied before the coarse resampling.  The 10 Hz
    #: rate alone leaves 5-7 Hz vibrato under its Nyquist, so this is what
    #: actually keeps vibrato out of the coarse contour.
    coarse_cutoff_hz: float = 3.0
    quant_cents: float = 25.0
    residual_clip_cents: float = 300.0
    #: "lowpass" keeps the smoothed contour; "notes" holds each note's pitch
    #: from its attack to the next note, so scoops, glides and phrase-end
    #: drops land in the residual the model generates.
    coarse_mode: str = "lowpass"
    note_slope_cents_s: float = 300.0
    note_min_s: float = 0.1
    #: How far a note's pitch reaches past its stable part, each way; beyond
    #: it, e.g. a drifting note that wasn't detected, the smoothed contour stays.
    note_reach_s: float = 0.35
    note_merge_cents: float = 50.0
    note_merge_gap_s: float = 0.1
    note_bridge_gap_s: float = 0.03

    @classmethod
    def from_dict(cls, data: dict | None) -> "ReprConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: str(v) if k == "coarse_mode" else float(v) for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def hz_to_cents(f0_hz: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(np.maximum(f0_hz, 1e-6) / CENTS_REF_HZ)


def cents_to_hz(cents: np.ndarray) -> np.ndarray:
    return CENTS_REF_HZ * np.power(2.0, cents / 1200.0)


def voiced_mask(f0_hz: np.ndarray) -> np.ndarray:
    return np.asarray(f0_hz) > 0


def _runs(mask: np.ndarray):
    """``[(start, end), ...]`` of the True runs, end exclusive."""
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate([[0], mask, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bridge_gaps(vuv: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill unvoiced gaps of at most ``max_gap`` frames between voiced frames."""
    out = np.asarray(vuv, dtype=bool).copy()
    for s, e in _runs(~out):
        if s > 0 and e < len(out) and e - s <= max_gap:
            out[s:e] = True
    return out


def interpolate_unvoiced(cents: np.ndarray, vuv: np.ndarray) -> np.ndarray:
    """Linear interpolation across unvoiced frames; the edges hold the nearest
    voiced value.  ``cents`` is only read where ``vuv`` is set."""
    idx = np.flatnonzero(vuv)
    if idx.size == 0:
        return np.zeros_like(cents, dtype=np.float64)
    return np.interp(np.arange(len(cents)), idx, cents[idx].astype(np.float64))


def _lowpass(x: np.ndarray, cutoff_hz: float, frame_rate: float) -> np.ndarray:
    if len(x) < 4:
        return np.full_like(x, x.mean())
    sos = signal.butter(4, cutoff_hz, fs=frame_rate, output="sos")
    padlen = min(len(x) - 1, int(3 * frame_rate / cutoff_hz))
    return signal.sosfiltfilt(sos, x, padtype="even", padlen=padlen)


def coarse_cents(f0_hz: np.ndarray, cfg: ReprConfig) -> np.ndarray | None:
    """Coarse melody in cents on every frame (unvoiced frames interpolated).

    ``None`` when the clip has fewer than two voiced frames.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    vuv = voiced_mask(f0_hz)
    if vuv.sum() < 2:
        return None
    n = len(f0_hz)
    smooth = _lowpass(interpolate_unvoiced(hz_to_cents(f0_hz), vuv), cfg.coarse_cutoff_hz, cfg.frame_rate)

    step = cfg.frame_rate / cfg.coarse_rate_hz
    grid = np.arange(0.0, n, step)
    if grid[-1] < n - 1:
        grid = np.append(grid, n - 1)
    knots = np.interp(grid, np.arange(n), smooth)
    knots = np.round(knots / cfg.quant_cents) * cfg.quant_cents
    cents = np.interp(np.arange(n), grid, knots)
    if cfg.coarse_mode == "notes":
        cents = _hold_notes(smooth, cents, vuv, cfg)
    return cents


def _hold_notes(smooth: np.ndarray, fallback: np.ndarray, vuv: np.ndarray, cfg: ReprConfig) -> np.ndarray:
    """Step contour: within each voiced run, frames take the pitch of their
    note; a note owns its attack and release up to ``note_reach_s``, and
    hands over to the next where ``smooth`` crosses the midpoint between
    them.  Frames no note reaches keep ``fallback``."""
    fr = cfg.frame_rate
    voiced = bridge_gaps(vuv, int(round(cfg.note_bridge_gap_s * fr)))
    stable = voiced & (np.abs(np.gradient(smooth)) * fr < cfg.note_slope_cents_s)
    out = fallback.copy()
    for s0, e0 in _runs(voiced):
        notes = []
        for s, e in _runs(stable[s0:e0]):
            s, e = s + s0, e + s0
            pitch = float(np.median(smooth[s:e]))
            prev = notes[-1] if notes else None
            if prev and s - prev[1] <= cfg.note_merge_gap_s * fr and abs(prev[2] - pitch) < cfg.note_merge_cents:
                notes[-1] = [prev[0], e, float(np.median(smooth[prev[0]:e]))]
            else:
                notes.append([s, e, pitch])
        notes = [n for n in notes if n[1] - n[0] >= cfg.note_min_s * fr]
        bounds = [s0]
        for (_, e, pitch), (nxt_start, _, nxt_pitch) in zip(notes, notes[1:]):
            mid = 0.5 * (pitch + nxt_pitch)
            crossed = np.flatnonzero((smooth[e:nxt_start] - mid) * (pitch - mid) <= 0)
            bounds.append(e + (int(crossed[0]) if crossed.size else (nxt_start - e) // 2))
        bounds.append(e0)
        reach = int(round(cfg.note_reach_s * fr))
        for i, (s, e, pitch) in enumerate(notes):
            start, end = max(bounds[i], s - reach), min(bounds[i + 1], e + reach)
            out[start:end] = np.round(pitch / cfg.quant_cents) * cfg.quant_cents
    return out


def coarse_f0(f0_hz: np.ndarray, cfg: ReprConfig) -> np.ndarray:
    """Coarse melody in Hz on every frame; zeros for an unvoiced clip."""
    cents = coarse_cents(f0_hz, cfg)
    if cents is None:
        return np.zeros(len(f0_hz), dtype=np.float32)
    return cents_to_hz(cents).astype(np.float32)


def residual_cents(f0_hz: np.ndarray, coarse_hz: np.ndarray, cfg: ReprConfig) -> np.ndarray:
    """1200*log2(f0/coarse) on voiced frames, clipped; zero elsewhere."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    res = np.zeros(len(f0_hz), dtype=np.float32)
    ok = voiced_mask(f0_hz) & (np.asarray(coarse_hz) > 0)
    res[ok] = 1200.0 * np.log2(f0_hz[ok] / coarse_hz[ok])
    clip = cfg.residual_clip_cents
    return np.clip(res, -clip, clip)


def compose_f0(coarse_hz: np.ndarray, residual: np.ndarray, vuv: np.ndarray) -> np.ndarray:
    """Inverse of the split: F0 in Hz, zero where unvoiced."""
    f0 = np.asarray(coarse_hz, dtype=np.float64) * np.power(2.0, np.asarray(residual) / 1200.0)
    return np.where(np.asarray(vuv, dtype=bool), f0, 0.0).astype(np.float32)


def decompose(f0_hz: np.ndarray, cfg: ReprConfig):
    """``(coarse_hz, residual_cents, vuv)`` for one contour."""
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    coarse = coarse_f0(f0_hz, cfg)
    return coarse, residual_cents(f0_hz, coarse, cfg), voiced_mask(f0_hz)


@dataclass
class Normalizer:
    """Global statistics that map the representation to unit scale and back."""

    residual_mean: float = 0.0
    residual_std: float = 1.0
    coarse_mean: float = 0.0
    coarse_std: float = 1.0
    descriptor_mean: np.ndarray | None = None
    descriptor_std: np.ndarray | None = None

    def encode_target(self, residual: np.ndarray, vuv: np.ndarray) -> np.ndarray:
        """``(2, T)``: normalized residual and voicing as -1/+1."""
        vuv = np.asarray(vuv, dtype=bool)
        res = (np.asarray(residual, dtype=np.float32) - self.residual_mean) / self.residual_std
        res = np.where(vuv, res, 0.0)
        return np.stack([res, np.where(vuv, 1.0, -1.0)]).astype(np.float32)

    def decode_target(self, target: np.ndarray):
        """``(residual_cents, vuv)`` from a ``(2, T)`` target."""
        target = np.asarray(target, dtype=np.float32)
        vuv = target[1] > 0
        res = target[0] * self.residual_std + self.residual_mean
        return np.where(vuv, res, 0.0).astype(np.float32), vuv

    def encode_coarse(self, coarse_hz: np.ndarray) -> np.ndarray:
        coarse_hz = np.asarray(coarse_hz, dtype=np.float32)
        out = (hz_to_cents(coarse_hz) - self.coarse_mean) / self.coarse_std
        return np.where(coarse_hz > 0, out, 0.0).astype(np.float32)

    def decode_coarse(self, coarse_norm: np.ndarray) -> np.ndarray:
        return cents_to_hz(np.asarray(coarse_norm) * self.coarse_std + self.coarse_mean).astype(np.float32)

    def encode_descriptors(self, values: np.ndarray) -> np.ndarray:
        """Missing descriptors (NaN) land on the dataset mean, i.e. zero."""
        values = np.asarray(values, dtype=np.float32)
        out = (values - self.descriptor_mean) / self.descriptor_std
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def decode_descriptors(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) * self.descriptor_std + self.descriptor_mean).astype(np.float32)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in ("descriptor_mean", "descriptor_std"):
            if out[key] is not None:
                out[key] = np.asarray(out[key]).tolist()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        data = dict(data)
        for key in ("descriptor_mean", "descriptor_std"):
            if data.get(key) is not None:
                data[key] = np.asarray(data[key], dtype=np.float32)
        return cls(**data)
