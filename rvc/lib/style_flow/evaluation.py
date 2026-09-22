"""Metrics shared by ``evaluate.py``, the training validation and the plots."""

from __future__ import annotations

import os
import re
from dataclasses import replace

import numpy as np

from .clips import AUDIO_EXTENSIONS
from .data import CLIP_DIR, MANIFEST, load_clip, read_manifest
from .descriptors import DESCRIPTOR_NAMES, DescriptorConfig, analyze, descriptor_vector, pool
from .f0_repr import Normalizer, ReprConfig, coarse_cents, decompose, voiced_mask


class ContourLoader:
    """F0 contours from ``.npz`` (a style clip or a generated contour, both
    with an ``f0`` key) or from audio, analysed on first use of the frontend."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._frontend = None

    def __call__(self, path: str) -> np.ndarray:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as data:
                return data["f0"].astype(np.float32)
        from rvc.lib.audio_io import load_audio_16k

        from .frontend import HOP, StyleFrontend, highpass

        if self._frontend is None:
            self._frontend = StyleFrontend(self.device)
        audio = highpass(load_audio_16k(path))
        return self._frontend.f0(audio)[: len(audio) // HOP]


def list_inputs(path: str):
    """A file, a style dataset directory (its clips) or a folder of audio/npz."""
    if os.path.isfile(path):
        return [path]
    if os.path.exists(os.path.join(path, MANIFEST)):
        path = os.path.join(path, CLIP_DIR)
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(".npz") or f.lower().endswith(AUDIO_EXTENSIONS)
    )


def melody_error_cents(f0_out, f0_src, cfg: ReprConfig, transpose: float = 0.0) -> dict:
    """Coarse-melody error of the output against the (transposed) source,
    over frames voiced in both."""
    n = min(len(f0_out), len(f0_src))
    f0_out, f0_src = np.asarray(f0_out[:n]), np.asarray(f0_src[:n])
    # The smoothed contour, whatever the model's: a step contour turns a note
    # boundary a few frames off into a whole interval of error.
    cfg = replace(cfg, coarse_mode="lowpass")
    c_out, c_src = coarse_cents(f0_out, cfg), coarse_cents(f0_src, cfg)
    both = voiced_mask(f0_out) & voiced_mask(f0_src)
    if c_out is None or c_src is None or not both.any():
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "frames": 0}
    err = np.abs(c_out[both] - c_src[both] - 100.0 * transpose)
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "frames": int(both.sum()),
    }


def events_of(paths, loader: ContourLoader, cfg: ReprConfig, dcfg: DescriptorConfig):
    """Events per input; style clips reuse the events stored at extraction."""
    out = []
    for path in paths:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as data:
                stored = "events" in data.files
            if stored:
                out.append(load_clip(path)["events"])
                continue
        f0 = loader(path)
        _, residual, _ = decompose(f0, cfg)
        out.append(analyze(f0, cfg, dcfg, residual))
    return out


def pooled_descriptors(paths, loader, cfg, dcfg) -> np.ndarray:
    return descriptor_vector(pool(events_of(paths, loader, cfg, dcfg)))


def descriptor_distance(a: np.ndarray, b: np.ndarray, normalizer: Normalizer) -> float:
    """Mean absolute difference in normalized units over descriptors both have."""
    za = (np.asarray(a) - normalizer.descriptor_mean) / normalizer.descriptor_std
    zb = (np.asarray(b) - normalizer.descriptor_mean) / normalizer.descriptor_std
    ok = np.isfinite(za) & np.isfinite(zb)
    return float(np.mean(np.abs(za[ok] - zb[ok]))) if ok.any() else np.nan


def normalizer_from(path: str | None) -> Normalizer | None:
    if path and os.path.exists(os.path.join(path, MANIFEST)):
        return Normalizer.from_dict(read_manifest(path)["normalizer"])
    return None


def descriptor_rows(columns: dict[str, np.ndarray]):
    """``[(name, [values...]), ...]`` in descriptor order, for printing."""
    return [(name, [float(col[i]) for col in columns.values()]) for i, name in enumerate(DESCRIPTOR_NAMES)]


# --- optional perceptual checks ---------------------------------------------


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize_text(reference), _normalize_text(hypothesis)
    if not ref:
        return np.nan
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


class Transcriber:
    """Whisper, when ``openai-whisper`` is installed."""

    def __init__(self, model_name: str = "small", device: str = "cuda"):
        import whisper

        self.model = whisper.load_model(model_name, device=device)

    def __call__(self, path: str) -> str:
        return self.model.transcribe(path)["text"]


class SpeakerEmbedder:
    """Resemblyzer d-vectors, when ``resemblyzer`` is installed."""

    def __init__(self):
        from resemblyzer import VoiceEncoder, preprocess_wav

        self.encoder = VoiceEncoder()
        self._prep = preprocess_wav

    def __call__(self, paths) -> np.ndarray:
        embeds = [self.encoder.embed_utterance(self._prep(p)) for p in paths]
        mean = np.mean(embeds, axis=0)
        return mean / np.linalg.norm(mean)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
