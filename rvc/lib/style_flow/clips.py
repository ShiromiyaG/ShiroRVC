"""Audio sources for extraction: stitched RVC slices or a folder of files."""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
_SLICE_NAME = re.compile(r"^(\d+)_(\d+)_(\d+)$")


@dataclass
class Stretch:
    """Contiguous 16 kHz audio.  ``slices`` are ``(slice_file_name,
    start_sample, overlap_samples)`` of the RVC slices it was stitched from
    (overlap with the previous one), empty for plain files."""

    first: int
    audio: np.ndarray
    slices: list = field(default_factory=list)


@dataclass
class Run:
    """One recording; ``load`` returns its ``Stretch`` list."""

    key: str
    speaker: int
    load: Callable[[], list]


def _overlap_matches(a: np.ndarray, b: np.ndarray, min_corr: float) -> bool:
    if a.size == 0 or a.size != b.size:
        return False
    ra, rb = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
    if ra < 1e-4 and rb < 1e-4:
        return True
    if ra < 1e-4 or rb < 1e-4:
        return False
    return float(np.corrcoef(a, b)[0, 1]) >= min_corr


def _find_overlap(current: np.ndarray, audio: np.ndarray, probe: int, min_corr: float) -> int:
    """Samples by which ``audio`` overlaps the end of ``current``, found by
    locating its opening ``probe`` samples in ``current``'s tail; 0 when it
    does not continue ``current``."""
    span = min(len(current), len(audio)) - 1
    head = audio[:probe]
    if span <= probe or np.sqrt(np.mean(head**2)) < 1e-4:
        return 0
    tail = current[-span:].astype(np.float64)
    corr = signal.correlate(tail, head, mode="valid", method="fft")
    energy = signal.convolve(tail**2, np.ones(probe), mode="valid", method="fft")
    pos = int(np.argmax(corr / np.sqrt(np.maximum(energy, 1e-12))))
    found = span - pos
    return found if _overlap_matches(current[-found:], audio[:found], min_corr) else 0


def stitch(chunks, overlap: int, min_corr: float, probe: int = 1600):
    """Join consecutive ``(index, file_name, audio)`` slices that continue
    each other into ``Stretch``es.

    RVC's slicer numbers slices across VAD segments without marking where a
    segment ends, so continuity is checked on the audio itself.  ``overlap``
    is tried first; otherwise the real overlap is searched for, since
    "Simple" cutting's overlap follows the sample rate and it re-anchors each
    file's last slice to the file's end.
    """
    current = None
    for index, stem, audio in chunks:
        found = 0
        if current is not None:
            if len(current.audio) >= overlap and len(audio) > overlap and _overlap_matches(
                current.audio[-overlap:], audio[:overlap], min_corr
            ):
                found = overlap
            else:
                found = _find_overlap(current.audio, audio, probe, min_corr)
        if found:
            current.slices.append((stem, len(current.audio) - found, found))
            current.audio = np.concatenate([current.audio, audio[found:]])
            continue
        if current is not None:
            yield current
        current = Stretch(first=index, audio=audio, slices=[(stem, 0, 0)])
    if current is not None:
        yield current


def stitch_frames(arrays, starts, overlaps, total: int, hop: int) -> np.ndarray:
    """Per-slice frame arrays (``hop`` samples per frame) joined the way
    ``stitch`` joined the audio, switching slices mid-overlap so neither
    side's edge frames are used."""
    n = total // hop
    edges = [0] + [(s + o // 2) // hop for s, o in zip(starts[1:], overlaps[1:])] + [n]
    out = []
    for arr, start, lo, hi in zip(arrays, starts, edges[:-1], edges[1:]):
        local = lo - start // hop
        seg = arr[local : local + hi - lo]
        if len(seg) < hi - lo:
            seg = np.concatenate([seg, np.repeat(seg[-1:], hi - lo - len(seg), axis=0)])
        out.append(seg)
    return np.concatenate(out)


def experiment_runs(exp_dir: str, overlap_s: float, min_corr: float):
    """Runs from ``<exp>/sliced_audios``, grouped by ``<sid>_<recording>`` and
    stitched.  Stitching needs the audio, so it happens in ``Run.load``."""
    from rvc.lib.audio_io import load_audio_16k

    slice_dir = os.path.join(exp_dir, "sliced_audios")
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(slice_dir, "*")):
        stem = os.path.splitext(os.path.basename(path))[0]
        match = _SLICE_NAME.match(stem)
        if match:
            sid, rec, idx = map(int, match.groups())
            groups[(sid, rec)].append((idx, path))

    overlap = int(round(overlap_s * 16000))
    runs = []
    for (sid, rec), items in sorted(groups.items()):
        items.sort()

        def load(items=items):
            chunks = ((idx, os.path.basename(path), load_audio_16k(path)) for idx, path in items)
            return list(stitch(chunks, overlap, min_corr))

        runs.append(Run(key=f"{sid}_{rec}", speaker=sid, load=load))
    return runs


_SPEAKER_DIR = re.compile(r"^(\d+)_")


def _audio_files(root: str):
    return sorted(
        p
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
        if p.lower().endswith(AUDIO_EXTENSIONS)
    )


def folder_runs(audio_dir: str, speaker: int | None = None):
    """One run per audio file under ``audio_dir`` (recursive), whole files of
    any length.  Subfolders named ``<id>_<name>`` are speakers, as in an RVC
    multi-speaker dataset, unless ``speaker`` forces one id for everything."""
    from rvc.lib.audio_io import load_audio_16k

    groups = []
    if speaker is None:
        for name in sorted(os.listdir(audio_dir)):
            match = _SPEAKER_DIR.match(name)
            if match and os.path.isdir(os.path.join(audio_dir, name)):
                groups.append((int(match.group(1)), os.path.join(audio_dir, name)))
    if not groups:
        groups = [(speaker or 0, audio_dir)]

    runs = []
    for sid, root in groups:
        for path in _audio_files(root):
            rel = os.path.splitext(os.path.relpath(path, audio_dir))[0]
            key = re.sub(r"[^\w.-]+", "_", rel)
            runs.append(Run(key=key, speaker=sid, load=lambda path=path: [Stretch(0, load_audio_16k(path))]))
    return runs


def split_frames(vuv: np.ndarray, min_frames: int, max_frames: int):
    """``[(start, end), ...]`` clips of ``min_frames``..``max_frames``, cut at
    the middle of the longest unvoiced stretch in the allowed window so no
    clip starts or ends mid-note.  A tail shorter than ``min_frames`` is
    dropped."""
    n = len(vuv)
    unvoiced = ~np.asarray(vuv, dtype=bool)
    clips, pos = [], 0
    while n - pos >= min_frames:
        if n - pos <= max_frames:
            clips.append((pos, n))
            break
        lo, hi = pos + min_frames, pos + max_frames
        window = unvoiced[lo:hi].astype(np.int8)
        edges = np.diff(np.concatenate([[0], window, [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if starts.size:
            best = int(np.argmax(ends - starts))
            cut = lo + (starts[best] + ends[best]) // 2
        else:
            cut = hi
        clips.append((pos, cut))
        pos = cut
    return clips
