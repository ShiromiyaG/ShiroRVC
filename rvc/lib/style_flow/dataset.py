"""Training batches: clips encoded for the model, with a cap on speech per batch."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .augment import AugmentConfig, augment
from .descriptors import DescriptorConfig
from .f0_repr import Normalizer, ReprConfig


def parse_speakers(spec) -> set[int]:
    """``["0-4", 7]`` -> ``{0, 1, 2, 3, 4, 7}``."""
    out = set()
    for item in spec or []:
        text = str(item)
        if "-" in text:
            lo, hi = map(int, text.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(text))
    return out


def scan_clips(paths):
    """``(lengths, speakers)`` of each clip, reading only two small arrays."""
    lengths, speakers = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            lengths.append(int(data["vuv"].shape[0]))
            speakers.append(int(data["speaker"]))
    return np.array(lengths), np.array(speakers)


class StyleDataset(Dataset):
    def __init__(
        self,
        paths,
        normalizer: Normalizer,
        rcfg: ReprConfig,
        dcfg: DescriptorConfig,
        acfg: AugmentConfig | None = None,
        max_frames: int = 1500,
        descriptor_dim_dropout: float = 0.0,
        random_crop: bool = True,
    ):
        self.paths = list(paths)
        self.normalizer = normalizer
        self.rcfg, self.dcfg, self.acfg = rcfg, dcfg, acfg
        self.max_frames = max_frames
        self.descriptor_dim_dropout = descriptor_dim_dropout
        self.random_crop = random_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        with np.load(self.paths[idx], allow_pickle=False) as data:
            coarse, residual, vuv = data["coarse"], data["residual"], data["vuv"]
            units, desc = data["units"].astype(np.int64), data["descriptors"]
        if len(vuv) > self.max_frames:
            # Descriptors were measured on the whole clip; a crop only shifts them a little.
            s = int(rng.integers(0, len(vuv) - self.max_frames + 1)) if self.random_crop else 0
            e = s + self.max_frames
            coarse, residual, vuv, units = coarse[s:e], residual[s:e], vuv[s:e], units[s:e]
        if self.acfg is not None:
            coarse, residual, new_desc = augment(coarse, residual, vuv, self.rcfg, self.dcfg, self.acfg, rng)
            if new_desc is not None:
                desc = new_desc
        present = np.isfinite(desc)
        if self.descriptor_dim_dropout > 0:
            present &= rng.random(present.shape) >= self.descriptor_dim_dropout
        return {
            "target": self.normalizer.encode_target(residual, vuv),
            "coarse": self.normalizer.encode_coarse(coarse),
            "units": units,
            "desc": np.where(present, self.normalizer.encode_descriptors(desc), 0.0).astype(np.float32),
            "desc_mask": present.astype(np.float32),
        }


def collate(items):
    T = max(item["units"].shape[0] for item in items)
    B = len(items)
    C = items[0]["target"].shape[0]
    out = {
        "target": torch.zeros(B, C, T),
        "coarse": torch.zeros(B, T),
        "units": torch.zeros(B, T, dtype=torch.long),
        "mask": torch.zeros(B, T, dtype=torch.bool),
        "desc": torch.from_numpy(np.stack([item["desc"] for item in items])),
        "desc_mask": torch.from_numpy(np.stack([item["desc_mask"] for item in items])),
    }
    for i, item in enumerate(items):
        n = item["units"].shape[0]
        out["target"][i, :, :n] = torch.from_numpy(item["target"])
        out["coarse"][i, :n] = torch.from_numpy(item["coarse"])
        out["units"][i, :n] = torch.from_numpy(item["units"])
        out["mask"][i, :n] = True
    return out


class MixedBatchSampler(Sampler):
    """Endless batches with ``floor(max_speech_fraction * batch_size)`` speech
    clips each.  Batches are drawn ``pool`` at a time and grouped by length
    so little of each batch is padding."""

    def __init__(self, singing, speech, lengths, batch_size, max_speech_fraction=0.15, pool=50, seed=0):
        self.singing = np.asarray(singing)
        self.speech = np.asarray(speech)
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.n_speech = int(np.floor(max_speech_fraction * batch_size)) if len(self.speech) else 0
        self.pool = pool
        self.seed = seed

    def _stream(self, indices, rng):
        while True:
            yield from rng.permutation(indices)

    def _take(self, stream, n):
        return np.array([next(stream) for _ in range(n)], dtype=np.int64)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        sing = self._stream(self.singing, rng)
        speech = self._stream(self.speech, rng) if self.n_speech else None
        n_sing = self.batch_size - self.n_speech
        while True:
            a = self._take(sing, n_sing * self.pool)
            a = a[np.argsort(self.lengths[a], kind="stable")].reshape(self.pool, n_sing)
            if speech is not None:
                b = self._take(speech, self.n_speech * self.pool)
                b = b[np.argsort(self.lengths[b], kind="stable")].reshape(self.pool, self.n_speech)
                a = np.concatenate([a, b], axis=1)
            for i in rng.permutation(self.pool):
                yield a[i].tolist()
