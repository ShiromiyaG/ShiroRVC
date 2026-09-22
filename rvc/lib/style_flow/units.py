"""Discrete content units: k-means over ContentVec features."""

from __future__ import annotations

import numpy as np


class UnitCodebook:
    def __init__(self, centroids: np.ndarray):
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self._sq = np.sum(self.centroids**2, axis=1)

    @property
    def size(self) -> int:
        return int(self.centroids.shape[0])

    @classmethod
    def fit(cls, samples: np.ndarray, clusters: int, iterations: int = 25, seed: int = 0) -> "UnitCodebook":
        import faiss

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        kmeans = faiss.Kmeans(samples.shape[1], clusters, niter=iterations, seed=seed, verbose=False)
        kmeans.train(samples)
        return cls(kmeans.centroids)

    def assign(self, feats: np.ndarray) -> np.ndarray:
        """Nearest centroid per row of ``(T, D)`` features, as int16."""
        feats = np.asarray(feats, dtype=np.float32)
        dist = self._sq[None, :] - 2.0 * feats @ self.centroids.T
        return np.argmin(dist, axis=1).astype(np.int16)

    def save(self, path: str) -> None:
        np.save(path, self.centroids, allow_pickle=False)

    @classmethod
    def load(cls, path: str) -> "UnitCodebook":
        return cls(np.load(path, allow_pickle=False))


def units_to_frames(units: np.ndarray, n_frames: int) -> np.ndarray:
    """ContentVec runs at half the pitch rate; RVC repeats each feature twice
    (``F.interpolate`` nearest, ``scale_factor=2``), and so does this."""
    out = np.repeat(np.asarray(units), 2)[:n_frames]
    if out.size < n_frames:
        fill = out[-1] if out.size else 0
        out = np.concatenate([out, np.full(n_frames - out.size, fill, dtype=out.dtype)])
    return out
