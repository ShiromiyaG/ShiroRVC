"""Metadata for FAISS retrieval indexes, embedded after the index data.

A FAISS index describes vectors and nothing else.  What this fork wants at
inference time and cannot express inside it is **where each vector came from**
-- which utterance, which frame -- because the temporal-continuity bonus has to
know whether two candidates are consecutive frames of the same recording.

That is appended to the ``.index`` file as a footer.  ``faiss.read_index``
stops at the end of the index structure, so upstream RVC and Applio read the
file as a plain index, and copying it carries the metadata along.  Indexes
built before the footer have it in a sidecar, ``<name>.index.meta.npz``, which
:func:`read` falls back to.  With neither -- an upstream index, or one re-saved
by a tool that drops the footer -- :func:`read` returns ``None`` and the caller
uses :func:`legacy_meta`, which describes exactly what the bare file supports:
the metric the file says, no provenance.  Losing the metadata therefore
degrades rather than breaks.

Nothing here is *required* to interpret the vectors, which is the property that
makes the degradation safe.  A cosine index stores unit vectors, and the
retrieval rescales each one it returns to the norm of the query that matched it
rather than to a recorded original norm -- so a lost sidecar costs continuity,
never a silently mis-scaled feature.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np

#: Bumped when the sidecar's meaning changes in a way older readers cannot
#: absorb.  :func:`read` refuses anything newer than it understands rather than
#: guessing.
FORMAT_VERSION = 1

SIDECAR_SUFFIX = ".meta.npz"

#: The footer's last bytes: ``<payload><payload length, u64 LE><magic>``.
FOOTER_MAGIC = b"RVCIMETA"
_FOOTER = struct.Struct("<Q8s")

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


def _footer(handle) -> tuple[int, int] | None:
    """``(offset, length)`` of the embedded payload, or ``None`` if there is none."""
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    if size < _FOOTER.size:
        return None
    handle.seek(size - _FOOTER.size)
    length, magic = _FOOTER.unpack(handle.read(_FOOTER.size))
    if magic != FOOTER_MAGIC or length > size - _FOOTER.size:
        return None
    return size - _FOOTER.size - length, length


def write(index_path: str | os.PathLike, meta: IndexMeta) -> str:
    """Embed ``meta`` at the end of the index file ``faiss.write_index`` wrote.

    A footer already there is replaced, and a sidecar from an older build is
    removed: it would describe a previous index, not this one.
    """
    payload = to_bytes(meta)
    with open(index_path, "r+b") as handle:
        footer = _footer(handle)
        if footer is not None:
            handle.truncate(footer[0])
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.write(_FOOTER.pack(len(payload), FOOTER_MAGIC))
    stale = sidecar_path(index_path)
    if os.path.exists(stale):
        os.remove(stale)
    return os.fspath(index_path)


def read(index_path: str | os.PathLike) -> IndexMeta | None:
    """Load the metadata for ``index_path``: its footer, else a sidecar beside it.

    Either one that cannot be parsed is treated as missing.  Retrieval still
    works without them, so a corrupt one should cost the user the
    improvements, not the inference.
    """
    try:
        with open(index_path, "rb") as handle:
            footer = _footer(handle)
            if footer is not None:
                handle.seek(footer[0])
                meta = from_bytes(handle.read(footer[1]))
                if meta is not None:
                    return meta
    except OSError:
        pass

    path = sidecar_path(index_path)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return _from_npz(data)
    except Exception:
        return None


def from_index_bytes(data) -> IndexMeta | None:
    """The footer of an index held in memory, as a model bundle carries it."""
    buffer = np.asarray(data, dtype=np.uint8).reshape(-1)
    if buffer.size < _FOOTER.size:
        return None
    length, magic = _FOOTER.unpack(buffer[-_FOOTER.size :].tobytes())
    start = buffer.size - _FOOTER.size - length
    if magic != FOOTER_MAGIC or start < 0:
        return None
    return from_bytes(buffer[start : start + length].tobytes())


def to_bytes(meta: IndexMeta) -> bytes:
    """Serialise, as the footer and model bundles both carry it."""
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
