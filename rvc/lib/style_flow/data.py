"""On-disk style dataset: one ``.npz`` per clip plus ``style_data.json``."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .descriptors import DESCRIPTOR_NAMES, Events, descriptor_vector, pool
from .f0_repr import Normalizer, hz_to_cents

MANIFEST = "style_data.json"
CODEBOOK = "codebook.npy"
CLIP_DIR = "clips"


def save_clip(path: str, *, f0, coarse, residual, vuv, units, descriptors, events: Events, speaker: int) -> None:
    np.savez(
        path,
        f0=np.asarray(f0, dtype=np.float32),
        coarse=np.asarray(coarse, dtype=np.float32),
        residual=np.asarray(residual, dtype=np.float32),
        vuv=np.asarray(vuv, dtype=bool),
        units=np.asarray(units, dtype=np.int16),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        events=np.array(json.dumps(asdict(events))),
        speaker=np.int32(speaker),
    )


def load_clip(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        clip = {key: data[key] for key in data.files}
    clip["events"] = Events(**json.loads(str(clip["events"])))
    clip["speaker"] = int(clip["speaker"])
    return clip


def clip_paths(data_dir: str):
    clip_dir = os.path.join(data_dir, CLIP_DIR)
    return sorted(os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith(".npz"))


def compute_normalizer(paths) -> Normalizer:
    """Residual and coarse statistics over voiced frames; descriptor
    statistics over clips, ignoring missing values."""
    res_sum = res_sq = res_n = 0.0
    c_sum = c_sq = c_n = 0.0
    desc = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            vuv = data["vuv"]
            res = data["residual"][vuv].astype(np.float64)
            cents = hz_to_cents(data["coarse"][vuv].astype(np.float64))
            desc.append(data["descriptors"])
        res_sum, res_sq, res_n = res_sum + res.sum(), res_sq + (res**2).sum(), res_n + res.size
        c_sum, c_sq, c_n = c_sum + cents.sum(), c_sq + (cents**2).sum(), c_n + cents.size
    desc = np.stack(desc)
    res_mean = res_sum / max(res_n, 1)
    c_mean = c_sum / max(c_n, 1)
    d_std = np.nanstd(desc, axis=0)
    return Normalizer(
        residual_mean=float(res_mean),
        residual_std=float(np.sqrt(max(res_sq / max(res_n, 1) - res_mean**2, 1e-6))),
        coarse_mean=float(c_mean),
        coarse_std=float(np.sqrt(max(c_sq / max(c_n, 1) - c_mean**2, 1e-6))),
        descriptor_mean=np.nan_to_num(np.nanmean(desc, axis=0), nan=0.0).astype(np.float32),
        descriptor_std=np.where(np.isfinite(d_std) & (d_std > 1e-6), d_std, 1.0).astype(np.float32),
    )


def _events(path) -> Events:
    with np.load(path, allow_pickle=False) as data:
        return Events(**json.loads(str(data["events"])))


def _as_json(vector) -> list:
    return [None if np.isnan(v) else float(v) for v in vector]


def pooled_descriptors(paths) -> list:
    """Descriptor vector pooled over all of ``paths`` (NaN kept as ``None``)."""
    return _as_json(descriptor_vector(pool(_events(p) for p in paths)))


def speaker_descriptors(paths) -> dict[int, list]:
    """Pooled descriptor vector per speaker (NaN kept as ``None``)."""
    by_speaker: dict[int, list] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            by_speaker.setdefault(int(data["speaker"]), []).append(path)
    return {speaker: pooled_descriptors(group) for speaker, group in sorted(by_speaker.items())}


def write_manifest(data_dir: str, **payload) -> None:
    payload = {"descriptor_names": list(DESCRIPTOR_NAMES), **payload}
    with open(os.path.join(data_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_manifest(data_dir: str) -> dict:
    with open(os.path.join(data_dir, MANIFEST), "r", encoding="utf-8") as f:
        return json.load(f)
