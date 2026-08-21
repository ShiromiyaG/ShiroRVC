"""Nearest-neighbour retrieval against a speaker's feature index.

The retrieval replaces each frame of the source's embedding with a weighted
average of the frames nearest to it in the target speaker's index.  What that
can and cannot do is worth stating, because the knobs here are often expected to
carry more than they do: the frame ordering, the F0 contour and the energy
envelope all come from the source audio and are never touched, so retrieval
moves *articulation and timbre* toward the training voice and leaves prosody,
rhythm and delivery where they were.

Three things are tunable, and all three default to the behaviour every existing
RVC model was tuned against:

``k``
    How many neighbours are averaged.  Fewer keeps whatever is idiosyncratic
    about the matched frames; more averages toward the dataset's mean voice.

``power``
    Exponent of the inverse-distance weighting, ``w = 1/d**power``.  Zero is a
    flat average of the k neighbours, large values approach picking the single
    nearest.  ``2.0`` reproduces the classic ``1/d**2``.

``continuity``
    Rewards candidates that are consecutive frames of the same recording as the
    candidates chosen for the neighbouring source frames.  Frame-independent
    search is free to jump between unrelated parts of the dataset twenty times a
    second, and that jitter smears exactly the fine articulation the retrieval
    exists to sharpen.  Needs the sidecar's provenance, so it is inactive for
    indexes built elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from rvc.lib import index_meta
from rvc.lib.terminal import error as print_error, warning

#: Distances below this are treated as this.  An exact match scores zero, and
#: ``1/0`` propagates NaN through the generator and out as silence.
_MIN_DISTANCE = 1e-8

#: Above this, the device copy of the index is held in half precision.  Only
#: reachable with ``RVC_INDEX_MAX_VECTORS`` raised past its default, and only
#: applied to cosine indexes -- squared-L2 is computed by expansion, whose
#: cancellation is precisely what half precision handles badly.
_DEVICE_FP32_BUDGET = 512 * 1024**2

#: Cap on the distance matrix materialized per query block (elements, fp32).
#: 64M elements is 256 MB, which keeps the search off the critical path
#: without competing with the generator for VRAM.
_BLOCK_ELEMENTS = 64 * 1024 * 1024

#: A candidate counts as following another if its frame position is within this
#: many frames after it.  More than one because selection thins the index: the
#: literal next frame often did not survive, and refusing to bridge a one-frame
#: hole would switch continuity off for most of the index.
_CONTINUITY_GAPS = (1, 2)

#: Candidates examined before continuity re-ranks them.  Searching exactly ``k``
#: would leave the bonus nothing to promote.
_CONTINUITY_CANDIDATES = 32


@dataclass
class RetrievalConfig:
    """Inference-time retrieval knobs.  Defaults reproduce classic RVC."""

    k: int = 8
    power: float = 2.0
    continuity: float = 0.5

    @classmethod
    def build(cls, k=None, power=None, continuity=None) -> "RetrievalConfig":
        """Construct from optional values, ignoring the ones left unset."""
        config = cls()
        if k is not None:
            config.k = max(1, int(k))
        if power is not None:
            config.power = max(0.0, float(power))
        if continuity is not None:
            config.continuity = max(0.0, float(continuity))
        return config


class IndexRetriever:
    """Holds an index, its sidecar, and the device-side copies of both."""

    def __init__(self, index, meta=None, device="cpu"):
        self.index = index
        self.device = device
        self.meta = index_meta.validate(
            meta if meta is not None else index_meta.legacy_meta(index), index
        )
        # The FAISS structure is only consulted on the CPU path; CUDA searches
        # the reconstructed vectors exactly, so they are needed either way.
        self.vectors = np.ascontiguousarray(
            index.reconstruct_n(0, index.ntotal), dtype=np.float32
        )
        self.ready = self.vectors.shape[0] > 0
        self._device_vectors = None
        self._device_norms = None
        #: Provenance keys, cached per device: the CPU fallback needs its own
        #: copy and rebuilding it for every chunk of every file adds up.
        self._keys = {}
        self._warned = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_path(cls, index_path, device="cpu"):
        import faiss

        index = faiss.read_index(index_path)
        return cls(index, index_meta.read(index_path), device)

    @classmethod
    def from_loaded(cls, index, meta_payload=None, device="cpu"):
        """Build from an already-deserialised index, as model bundles carry."""
        meta = None
        if isinstance(meta_payload, (bytes, bytearray)):
            meta = index_meta.from_bytes(bytes(meta_payload))
        elif isinstance(meta_payload, index_meta.IndexMeta):
            meta = meta_payload
        return cls(index, meta, device)

    def describe(self) -> str:
        return self.meta.describe()

    # -- device-side state -------------------------------------------------

    def _ensure_device_state(self):
        if self._device_vectors is not None:
            return

        dtype = torch.float32
        footprint = self.vectors.size * 4
        if self.meta.is_cosine and footprint > _DEVICE_FP32_BUDGET:
            dtype = torch.float16

        vectors = torch.from_numpy(self.vectors).to(self.device).to(dtype).contiguous()
        self._device_vectors = vectors
        if not self.meta.is_cosine:
            # ||b||^2 for every index vector, reused by every query block.
            self._device_norms = (
                (vectors.float() ** 2).sum(dim=-1).unsqueeze(0).contiguous()
            )
    def _keys_for(self, device):
        """Provenance keys on ``device``, or ``None`` when the index has none.

        One sortable identity per frame -- recording in the high bits, frame
        position in the low ones -- so "same recording, a frame or two later" is
        a single integer comparison.
        """
        if not self.meta.has_provenance:
            return None
        cached = self._keys.get(str(device))
        if cached is None:
            keys = self.meta.utt.astype(np.int64) * (1 << 32) + self.meta.pos.astype(
                np.int64
            )
            cached = torch.from_numpy(keys).to(device)
            self._keys[str(device)] = cached
        return cached

    def _dimension_ok(self, dims: int) -> bool:
        if dims == self.vectors.shape[1]:
            return True
        if not self._warned:
            self._warned = True
            warning(
                f"Index has {self.vectors.shape[1]}-dimensional vectors but "
                f"the embedder produces {dims}. This index belongs to a model built "
                "with a different embedder; retrieval is disabled for this run.",
                tag="[INFER]",
            )
        self.ready = False
        return False

    # -- search ------------------------------------------------------------

    def _search_torch(self, query, candidates):
        """Exact brute-force search on the accelerator.

        Returns squared-L2-like distances in both metrics: for cosine, unit
        vectors give ``||a-b||^2 == 2 - 2*cos``, so a single weighting formula
        covers both and ``power`` means the same thing either way.
        """
        self._ensure_device_state()
        vectors = self._device_vectors
        count = vectors.shape[0]
        block = max(1, _BLOCK_ELEMENTS // max(1, count))

        if self.meta.is_cosine:
            query = torch.nn.functional.normalize(query, dim=-1)
        transposed = vectors.T.contiguous()

        all_distances, all_ids = [], []
        for start in range(0, query.shape[0], block):
            chunk = query[start : start + block]
            if self.meta.is_cosine:
                similarity = (chunk.to(vectors.dtype) @ transposed).float()
                distances = 2.0 - 2.0 * similarity
            else:
                distances = torch.addmm(
                    self._device_norms,
                    chunk,
                    transposed,
                    alpha=-2.0,
                    beta=1.0,
                ) + (chunk**2).sum(dim=-1, keepdim=True)
            # Rounding can push exact matches slightly negative, which would
            # blow up the inverse-distance weighting below.
            distances.clamp_(min=_MIN_DISTANCE)
            found, ids = torch.topk(distances, k=candidates, dim=-1, largest=False)
            all_distances.append(found)
            all_ids.append(ids)

        return torch.cat(all_distances, dim=0), torch.cat(all_ids, dim=0)

    def _search_faiss(self, query, candidates):
        """CPU search through the IVF structure, where its pruning earns its keep."""
        probe = np.ascontiguousarray(query.detach().cpu().numpy(), dtype=np.float32)
        if self.meta.is_cosine:
            import faiss

            probe = probe.copy()
            faiss.normalize_L2(probe)
            similarity, ids = self.index.search(probe, candidates)
            distances = 2.0 - 2.0 * similarity
        else:
            distances, ids = self.index.search(probe, candidates)

        # FAISS pads with -1 when a probe finds fewer than k neighbours; those
        # would index the last vector in the array and weight it as a match.
        missing = ids < 0
        if missing.any():
            ids = np.where(missing, 0, ids)
            distances = np.where(missing, np.inf, distances)

        distances = np.maximum(distances, _MIN_DISTANCE)
        return (
            torch.from_numpy(np.ascontiguousarray(distances, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64)),
        )

    # -- weighting ---------------------------------------------------------

    @staticmethod
    def _continuity_bonus(keys):
        """0, 1 or 2 per candidate: does it continue a neighbouring frame's match.

        Credit is given in both directions -- a candidate that follows one of the
        previous frame's candidates, and one that is followed by one of the next
        frame's -- so a run is reinforced from both ends instead of depending on
        whichever frame happened to be searched first.
        """
        if keys.shape[0] < 2:
            return torch.zeros(keys.shape, dtype=torch.float32, device=keys.device)

        backward = torch.zeros(keys.shape, dtype=torch.bool, device=keys.device)
        forward = torch.zeros(keys.shape, dtype=torch.bool, device=keys.device)
        previous = keys[:-1].unsqueeze(-2)
        current = keys[1:].unsqueeze(-1)
        for gap in _CONTINUITY_GAPS:
            follows = (current - gap) == previous
            backward[1:] |= follows.any(dim=-1)
            forward[:-1] |= follows.any(dim=-2)

        return backward.float() + forward.float()

    def _weights(self, distances, ids, config):
        weights = distances.clamp_min(_MIN_DISTANCE).pow(-config.power)

        keys = self._keys_for(ids.device) if config.continuity > 0 else None
        if keys is not None:
            bonus = self._continuity_bonus(keys[ids])
            weights = weights * (1.0 + config.continuity * bonus)

        if weights.shape[-1] > config.k:
            weights, order = torch.topk(weights, k=config.k, dim=-1)
            ids = torch.gather(ids, -1, order)

        total = weights.sum(dim=-1, keepdim=True)
        # A row where FAISS found no neighbour at all has nothing to normalise.
        # Dividing anyway yields a zero vector, which is not a quiet degradation
        # -- at index_rate 1 it replaces the frame with silence -- so the caller
        # is told to keep the query for those frames instead.
        usable = total > 0
        weights = torch.where(usable, weights / total.clamp_min(_MIN_DISTANCE), weights)
        return weights, ids, usable

    # -- public ------------------------------------------------------------

    def retrieve(self, feats, index_rate, config=None):
        """Blend ``feats`` with its neighbours in the index.

        ``feats`` is ``(1, T, D)``.  Returns the same shape and dtype.
        """
        if not self.ready or index_rate <= 0:
            return feats
        if not self._dimension_ok(feats.shape[-1]):
            return feats

        config = config or RetrievalConfig()
        query = feats[0].float().contiguous()

        candidates = config.k
        if config.continuity > 0 and self.meta.has_provenance:
            candidates = max(config.k, _CONTINUITY_CANDIDATES)
        candidates = min(candidates, self.vectors.shape[0])

        on_accelerator = str(self.device) != "cpu"
        if on_accelerator:
            try:
                self._ensure_device_state()
                distances, ids = self._search_torch(query, candidates)
            except Exception as error:
                warning(
                    f"Torch index search failed ({error}); falling back to FAISS.",
                    tag="[INFER]",
                )
                on_accelerator = False

        if not on_accelerator:
            distances, ids = self._search_faiss(query, candidates)
            vectors = torch.from_numpy(self.vectors)
            query = query.cpu()
        else:
            vectors = self._device_vectors

        weights, ids, usable = self._weights(distances, ids, config)
        retrieved = (vectors[ids].float() * weights.unsqueeze(-1)).sum(dim=1)
        retrieved = torch.where(usable, retrieved, query)

        if self.meta.is_cosine:
            # The index stores directions.  Rescaling to the query's own norm
            # keeps the source's energy envelope intact and takes only the
            # articulation from the dataset -- and it means a lost sidecar can
            # never hand the generator a unit vector where a feature belongs.
            retrieved = torch.nn.functional.normalize(retrieved, dim=-1) * query.norm(
                dim=-1, keepdim=True
            )

        retrieved = retrieved.unsqueeze(0).to(feats.device).to(feats.dtype)
        return retrieved * index_rate + (1 - index_rate) * feats


def load_retriever(index_path, device="cpu"):
    """Read an index and its sidecar, or return ``None`` if there is nothing to read."""
    if not index_path or not os.path.exists(index_path):
        return None
    try:
        return IndexRetriever.from_path(index_path, device)
    except Exception as error:
        print_error(f"Could not read the index, retrieval is off: {error}", tag="[INFER]")
        return None
