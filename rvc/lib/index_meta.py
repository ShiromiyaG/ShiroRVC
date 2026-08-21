"""Sidecar metadata for FAISS retrieval indexes.

A ``.index`` file describes vectors and nothing else.  What this fork wants at
inference time and cannot express inside it is **where each vector came from**
-- which utterance, which frame -- because the temporal-continuity bonus has to
know whether two candidates are consecutive frames of the same recording.

That lives in a sidecar next to the index, ``<name>.index.meta.npz``.  The
separation is deliberate: an index produced by upstream RVC v2 simply has no
sidecar, :func:`read` returns ``None``, and the caller falls back to
:func:`legacy_meta`, which describes exactly what the bare file can support --
L2 search over the stored vectors, no provenance.  Copying a ``.index`` around
without its sidecar therefore degrades rather than breaks.

Nothing here is *required* to interpret the vectors, which is the property that
makes the degradation safe.  A cosine index stores unit vectors, and the
retrieval rescales each one it returns to the norm of the query that matched it
rather than to a recorded original norm -- so a lost sidecar costs continuity,
never a silently mis-scaled feature.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np

#: Bumped when the sidecar's meaning changes in a way older readers cannot
#: absorb.  :func:`read` refuses anything newer than it understands rather than
#: guessing.
FORMAT_VERSION = 1

SIDECAR_SUFFIX = ".meta.npz"

#: Squared L2 over the stored vectors.  What upstream RVC has always used, and
#: what every index built before this module speaks.
METRIC_L2 = "l2"
#: Inner product over unit-normalised vectors.  Ranks by direction alone, which
#: is the better-motivated match for embeddings whose magnitude tracks loudness
#: rather than content.
METRIC_COSINE = "cosine"

METRICS = (METRIC_L2, METRIC_COSINE)


@dataclass
class IndexMeta:
    """What the retrieval needs to know about an index beyond its vectors."""

    metric: str = METRIC_L2
    dim: int = 0
    count: int = 0
    #: Selection strategy that produced the vectors, for the log line only.
    selection: str = "all"
    nprobe: int = 1
    #: Measured top-k recall of the IVF search against an exact one, or ``None``
    #: when it was not measured.
    recall: float | None = None
    format_version: int = FORMAT_VERSION

    #: Per-vector provenance.  ``utt`` identifies the source recording and
    #: ``pos`` the frame's position inside it, both in the *original* feature
    #: stream -- so ``pos`` is not contiguous after selection thins it.
    utt: np.ndarray | None = None
    pos: np.ndarray | None = None

    #: Free-form, written for humans reading the file later.
    extra: dict = field(default_factory=dict)

    @property
    def has_provenance(self) -> bool:
        return self.utt is not None and self.pos is not None

    @property
    def is_cosine(self) -> bool:
        return self.metric == METRIC_COSINE

    def describe(self) -> str:
        parts = [f"{self.count} vectors", self.metric, self.selection]
        if self.recall is not None:
            parts.append(f"recall {self.recall:.0%}")
        if self.has_provenance:
            parts.append("with provenance")
        return ", ".join(parts)


def sidecar_path(index_path: str | os.PathLike) -> str:
    """``logs/x/x.index`` -> ``logs/x/x.index.meta.npz``."""
    return f"{os.fspath(index_path)}{SIDECAR_SUFFIX}"


def legacy_meta(index) -> IndexMeta:
    """Describe an index that arrived without a sidecar.

    Everything claimed here is read off the FAISS object itself, so it is true
    of upstream RVC v2 files as much as of anything else: there is no
    provenance, and the metric is whatever the file says it is.
    """
    import faiss

    metric = (
        METRIC_COSINE
        if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_INNER_PRODUCT
        else METRIC_L2
    )
    return IndexMeta(
        metric=metric,
        dim=int(index.d),
        count=int(index.ntotal),
        selection="unknown",
        nprobe=int(getattr(index, "nprobe", 1)),
        format_version=0,
    )


def _payload(meta: IndexMeta) -> dict:
    arrays = {
        "meta": np.array(
            json.dumps(
                {
                    "format_version": meta.format_version,
                    "metric": meta.metric,
                    "dim": meta.dim,
                    "count": meta.count,
                    "selection": meta.selection,
                    "nprobe": meta.nprobe,
                    "recall": meta.recall,
                    "extra": meta.extra,
                }
            )
        )
    }
    if meta.has_provenance:
        arrays["utt"] = np.asarray(meta.utt, dtype=np.int32)
        arrays["pos"] = np.asarray(meta.pos, dtype=np.int32)
    return arrays


def _from_npz(data) -> IndexMeta | None:
    raw = json.loads(str(data["meta"].item()))
    if int(raw.get("format_version", 0)) > FORMAT_VERSION:
        return None

    meta = IndexMeta(
        metric=raw.get("metric", METRIC_L2),
        dim=int(raw.get("dim", 0)),
        count=int(raw.get("count", 0)),
        selection=raw.get("selection", "all"),
        nprobe=int(raw.get("nprobe", 1)),
        recall=raw.get("recall"),
        format_version=int(raw.get("format_version", FORMAT_VERSION)),
        extra=raw.get("extra", {}),
    )
    if "utt" in data and "pos" in data:
        meta.utt = np.ascontiguousarray(data["utt"], dtype=np.int32)
        meta.pos = np.ascontiguousarray(data["pos"], dtype=np.int32)
    return meta


def write(index_path: str | os.PathLike, meta: IndexMeta) -> str:
    path = sidecar_path(index_path)
    np.savez_compressed(path, **_payload(meta))
    return path


def read(index_path: str | os.PathLike) -> IndexMeta | None:
    """Load the sidecar for ``index_path``, or ``None`` if there is not one.

    A sidecar that exists but cannot be parsed is treated the same as a missing
    one.  Retrieval still works without it, so a corrupt sidecar should cost the
    user the improvements, not the inference.
    """
    path = sidecar_path(index_path)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return _from_npz(data)
    except Exception:
        return None


def to_bytes(meta: IndexMeta) -> bytes:
    """Serialise for embedding in a model bundle, which has no file to sit next to."""
    buffer = BytesIO()
    np.savez_compressed(buffer, **_payload(meta))
    return buffer.getvalue()


def from_bytes(payload: bytes) -> IndexMeta | None:
    if not payload:
        return None
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            return _from_npz(data)
    except Exception:
        return None


def validate(meta: IndexMeta, index) -> IndexMeta:
    """Reconcile a sidecar with the index it claims to describe.

    The two are separate files and can be separated further by a careless copy,
    so anything the sidecar asserts that the index contradicts is dropped rather
    than trusted -- provenance arrays of the wrong length would pair every
    candidate with another vector's timeline.
    """
    count, dim = int(index.ntotal), int(index.d)
    if (meta.dim and meta.dim != dim) or (meta.count and meta.count != count):
        return legacy_meta(index)
    # The metric is a property of the index file, not of the sidecar, so read it
    # from the authority rather than believing a claim that could be stale.
    meta.metric = legacy_meta(index).metric
    if meta.has_provenance and (
        meta.utt.shape[0] != count or meta.pos.shape[0] != count
    ):
        meta.utt = meta.pos = None
    meta.dim, meta.count = dim, count
    return meta
