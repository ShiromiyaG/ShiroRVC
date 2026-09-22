"""Style inference: replace the fine F0 of a source with a style model's.

The coarse melody always comes from the source; only the residual (vibrato,
scoops, drops) is generated.  Voicing stays the RVC pipeline's.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy import signal

from . import checkpoint
from .clips import split_frames
from .descriptors import DESCRIPTOR_NAMES, analyze, descriptor_vector, detect_notes, pool
from .f0_repr import _runs, coarse_f0, decompose, interpolate_unvoiced
from .flow import sample
from .frontend import HOP, StyleFrontend
from .units import units_to_frames

#: Clips generated per batch; with CFG the model sees twice as many.
BATCH_CLIPS = 8

#: Descriptors exposed as sliders, as offsets in dataset standard deviations.
MAIN_DESCRIPTORS = (
    "vibrato_extent_cents",
    "vibrato_rate_hz",
    "vibrato_fraction",
    "scoop_fraction",
    "scoop_depth_cents",
    "phrase_end_drop_cents",
)


@dataclass
class StyleOptions:
    """Everything a conversion needs to apply a style model; plain values so
    the CLI, Gradio and Qt can all build it from a dict."""

    model: str
    strength: float = 1.0
    rate: float = 1.0
    steps: int = 32
    cfg: float = 2.0
    #: How far to move toward the model's descriptors, before ``descriptors``
    #: is added: 0 is the source's own (``relative``) or the dataset mean, 1
    #: the model's, above 1 past them.
    intensity: float = 1.0
    #: Condition each clip on the source's own descriptors there, shifted by
    #: the gap between the model's and the whole source's, instead of on the
    #: model's everywhere.
    relative: bool = True
    #: Keep the source's intonation: each note centred on the source's
    #: pitch, phrases without a stable note on its slow contour.
    recenter: bool = True
    #: ``{descriptor_name: offset}`` in dataset standard deviations, added to
    #: the model's own descriptors (the dataset mean for a base).
    descriptors: dict = field(default_factory=dict)
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "StyleOptions | None":
        if not data or not data.get("model"):
            return None
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})


def blend(coarse, residual, plain, rate: float) -> np.ndarray:
    """F0 in Hz between ``plain`` (the source's smoothed melody, at ``rate``
    0) and the generated ``coarse`` + ``residual`` (at 1); above 1
    exaggerates the style.  Scaling the residual alone would, with a step
    contour, turn glides into steps."""
    generated = np.asarray(coarse, dtype=np.float64) * np.power(2.0, np.asarray(residual) / 1200.0)
    plain = np.asarray(plain, dtype=np.float64)
    ok = plain > 0
    out = generated.copy()
    out[ok] = plain[ok] * np.power(generated[ok] / plain[ok], float(rate))
    return out.astype(np.float32)


def fill_unvoiced(residual, generated, vuv) -> np.ndarray:
    """``residual`` interpolated, within each voiced run of the source, over
    the frames the model generated as unvoiced: left at zero they snap the
    pitch to the coarse note mid-phrase."""
    out = np.asarray(residual, dtype=np.float32).copy()
    for s, e in _runs(vuv):
        ok = generated[s:e]
        if ok.any() and not ok.all():
            idx = np.flatnonzero(ok)
            out[s:e] = np.interp(np.arange(e - s), idx, out[s:e][idx])
    return out


def recenter_notes(residual, src_res, f0, vuv, rcfg, dcfg, slow_hz: float = 2.0) -> np.ndarray:
    """``residual`` shifted so the stable part of each source note averages
    the source's residual there.  Between notes of a phrase the shift is
    interpolated, so attacks and glides move with their notes instead of
    stepping.  Phrases with no stable note follow the source below
    ``slow_hz`` instead, keeping only the generated detail above it."""
    notes, runs, _ = detect_notes(f0, rcfg, dcfg)
    knots: dict = {}
    for note in notes:
        m = vuv[note.start : note.end]
        if m.sum() >= 3:
            offset = float(np.mean((residual[note.start : note.end] - src_res[note.start : note.end])[m]))
            knots.setdefault(note.run, []).extend([(note.start, offset), (note.end - 1, offset)])
    out = np.asarray(residual, dtype=np.float32).copy()
    sos = signal.butter(2, slow_hz, fs=rcfg.frame_rate, output="sos")
    for run, (s, e) in enumerate(runs):
        m = vuv[s:e]
        if run in knots:
            x, y = zip(*knots[run])
            shift = np.interp(np.arange(s, e), x, y)
        elif m.sum() >= 3:
            diff = interpolate_unvoiced(out[s:e] - src_res[s:e], m)
            # Too short for the filter to settle: its mean is all there is.
            shift = signal.sosfiltfilt(sos, diff) if e - s >= 30 else np.full(e - s, diff[m].mean())
        else:
            continue
        out[s:e] -= np.where(m, shift, 0.0).astype(np.float32)
    return out


class StyleEngine:
    def __init__(self, path: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self.style = checkpoint.load(path, device)
        self.frontend = StyleFrontend(device, self.style.embedder, voicing_threshold=self.style.voicing_threshold)

    def _encode(self, values):
        """Normalized descriptors and presence of raw ``values`` (None or NaN
        for unknown)."""
        raw = np.array([np.nan if v is None else v for v in values], dtype=np.float32)
        present = np.isfinite(raw)
        return np.where(present, self.style.normalizer.encode_descriptors(raw), 0.0), present

    def conditioning(self, options: StyleOptions, clip_events=()) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized descriptors and presence masks, one row per clip of
        ``clip_events`` (the source's events in each clip); see
        ``StyleOptions.relative``.  ``options.descriptors`` is added last."""
        own, own_ok = self._encode(self.style.style_descriptors or [None] * len(DESCRIPTOR_NAMES))
        k = float(options.intensity)
        rows = max(1, len(clip_events))
        if options.relative and clip_events:
            song, song_ok = self._encode(descriptor_vector(pool(clip_events)))
            shift = np.where(own_ok & song_ok, own - song, 0.0)
            desc, present = np.zeros((rows, len(own))), np.zeros((rows, len(own)), dtype=bool)
            for i, events in enumerate(clip_events):
                clip, clip_ok = self._encode(descriptor_vector(events))
                # A descriptor the clip lacks (e.g. no vibrato in it) falls back
                # to the whole source's, then to the model's own.
                start = np.where(clip_ok, clip, np.where(song_ok, song, 0.0))
                desc[i] = np.where(clip_ok | song_ok, start + k * shift, own * k)
                present[i] = clip_ok | song_ok | own_ok
            # Past the range the model was trained on, clip values stop meaning much.
            desc = np.clip(desc, -3.0, 3.0)
        else:
            desc = np.tile(np.where(own_ok, own * k, 0.0), (rows, 1))
            present = np.tile(own_ok, (rows, 1))
        for name, offset in (options.descriptors or {}).items():
            if name in DESCRIPTOR_NAMES and offset:
                j = DESCRIPTOR_NAMES.index(name)
                desc[:, j] += float(offset)
                present[:, j] = True
        return (
            torch.from_numpy(desc.astype(np.float32)).to(self.device),
            torch.from_numpy(present.astype(np.float32)).to(self.device),
        )

    def _autocast(self):
        """bf16, or fp16 without it, on CUDA; fp32 elsewhere."""
        if not str(self.device).startswith("cuda"):
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    @torch.no_grad()
    def residual(self, audio: np.ndarray, transpose: float, options: StyleOptions):
        """``(coarse_hz, residual_cents, style_vuv, plain_hz)`` over
        ``len(audio) // HOP`` frames of 16 kHz, high-passed ``audio``;
        ``plain_hz`` is the source's smoothed melody, which ``blend`` fades
        from.  Everything is transposed by ``transpose`` semitones."""
        rcfg, norm = self.style.representation, self.style.normalizer
        n = len(audio) // HOP
        f0 = self.frontend.f0(audio)[:n]
        f0 = np.pad(f0, (0, n - len(f0))) * 2.0 ** (transpose / 12.0)
        coarse, src_res, vuv = decompose(f0, rcfg)
        plain = coarse if rcfg.coarse_mode != "notes" else coarse_f0(f0, replace(rcfg, coarse_mode="lowpass"))
        units = units_to_frames(self.style.codebook.assign(self.frontend.features(audio)), n)

        # Generated in clips like the training ones, cut at the longest
        # silences, and batched.
        fr = rcfg.frame_rate
        spans = split_frames(vuv, min(int(5 * fr), n), int(15 * fr)) or [(0, n)]
        if spans[-1][1] < n:
            spans.append((spans[-1][1], n))
        coarse_n, source = norm.encode_coarse(coarse), norm.encode_target(src_res, vuv)
        dcfg = self.style.descriptor_config
        clip_events = [analyze(f0[s:e], rcfg, dcfg, src_res[s:e]) for s, e in spans] if options.relative else ()
        desc, desc_mask = self.conditioning(options, clip_events)
        if desc.shape[0] == 1:
            desc, desc_mask = desc.expand(len(spans), -1), desc_mask.expand(len(spans), -1)
        generator = torch.Generator(device=self.device).manual_seed(int(options.seed) or 0)
        residual = np.zeros(n, dtype=np.float32)
        generated = np.zeros(n, dtype=bool)
        # A fixed number of clips at a time, so peak VRAM doesn't grow with the
        # song's length.
        for b in range(0, len(spans), BATCH_CLIPS):
            chunk = spans[b : b + BATCH_CLIPS]
            T = max(e - s for s, e in chunk)
            B = len(chunk)
            batch = {
                "units": torch.full((B, T), 0, dtype=torch.long),
                "coarse": torch.zeros(B, T),
                "mask": torch.zeros(B, T, dtype=torch.bool),
                "source": torch.zeros(B, 2, T),
            }
            for i, (s, e) in enumerate(chunk):
                batch["units"][i, : e - s] = torch.from_numpy(units[s:e].astype(np.int64))
                batch["coarse"][i, : e - s] = torch.from_numpy(coarse_n[s:e])
                batch["source"][i, :, : e - s] = torch.from_numpy(source[:, s:e])
                batch["mask"][i, : e - s] = True
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._autocast():
                x = sample(
                    self.style.model, batch["units"], batch["coarse"],
                    desc[b : b + B], desc_mask[b : b + B], batch["mask"],
                    steps=int(options.steps), cfg_scale=float(options.cfg),
                    source=batch["source"], strength=float(options.strength), generator=generator,
                )
            x = x.float().cpu().numpy()
            for i, (s, e) in enumerate(chunk):
                residual[s:e], generated[s:e] = norm.decode_target(x[i, :, : e - s])
        residual = fill_unvoiced(residual, generated, vuv)
        if options.recenter:
            residual = recenter_notes(residual, src_res, f0, vuv, rcfg, dcfg)
        return coarse, residual, vuv, plain

    def contour(self, audio: np.ndarray, transpose: float, options: StyleOptions):
        """``(styled_hz, vuv)`` per frame of ``audio``: the style's F0 and
        where the style path sees voice."""
        coarse, residual, vuv, plain = self.residual(audio, transpose, options)
        return blend(coarse, residual, plain, options.rate), vuv

    def apply(self, audio: np.ndarray, pitchf: np.ndarray, transpose: float, options: StyleOptions) -> np.ndarray:
        """The pipeline's ``pitchf`` (Hz, already transposed) with the
        style's F0 merged in; see ``merge``."""
        return merge(pitchf, *self.contour(audio, transpose, options))


def merge(pitchf, styled, vuv) -> np.ndarray:
    """``pitchf`` with ``styled`` wherever both it and ``vuv`` see voice;
    elsewhere unchanged."""
    n = min(len(pitchf), len(styled))
    out = np.asarray(pitchf, dtype=np.float32).copy()
    use = (out[:n] > 0) & np.asarray(vuv[:n], dtype=bool)
    out[:n][use] = np.asarray(styled[:n], dtype=np.float32)[use]
    return out


_engines: dict = {}


def get_engine(path: str, device: str) -> StyleEngine:
    """One loaded style model at a time, reused across conversions."""
    key = (path, str(device))
    if key not in _engines:
        _engines.clear()
        _engines[key] = StyleEngine(path, device)
    return _engines[key]


def unload():
    _engines.clear()
