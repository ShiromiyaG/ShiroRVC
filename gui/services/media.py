"""Validation artifacts a training run leaves on disk.

``rvc/train/utils.log_validation_preview`` writes each preview twice: into the
TensorBoard event file, and as loose files under the run's log directory::

    logs/<model>/validation_samples/
        epoch_0006/
            mel/sample_00.png
            audio/sample_00_generated.wav
            audio/sample_00_original.wav

The loose copies are what this module reads.  Pulling the same images and audio
back out of the event file would mean decoding TensorBoard's re-encoded PNG and
WAV payloads, and the event file for a long run is hundreds of megabytes of
mostly scalars -- while the originals are sitting right there, already grouped
by epoch and already at full resolution.

Nothing here touches Qt, so it can be tested without a display.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Written by ``TENSORBOARD_VALIDATION_PREVIEW_DIR`` on the training side.
PREVIEW_DIR_NAME = "validation_samples"

_EPOCH_DIR = re.compile(r"^epoch_(\d+)$")
_SAMPLE_STEM = re.compile(r"^sample_(\d+)$")
_AUDIO_STEM = re.compile(r"^sample_(\d+)_(generated|original)$")

#: Formats ``QMediaPlayer`` will open that the trainer might have written.
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")


@dataclass(frozen=True)
class Sample:
    """One validation sample within one epoch."""

    index: int
    mel: Path | None = None
    generated: Path | None = None
    original: Path | None = None

    @property
    def label(self) -> str:
        return f"Sample {self.index:02d}"

    @property
    def has_audio(self) -> bool:
        return self.generated is not None or self.original is not None


@dataclass(frozen=True)
class EpochPreviews:
    """Every sample written for one epoch, ordered by sample index."""

    epoch: int
    directory: Path
    samples: tuple[Sample, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"Epoch {self.epoch}"

    @property
    def mel_count(self) -> int:
        return sum(1 for s in self.samples if s.mel is not None)

    @property
    def audio_count(self) -> int:
        return sum(1 for s in self.samples if s.has_audio)


def preview_root(run_dir: str | os.PathLike[str] | None) -> Path | None:
    """Locate ``validation_samples/`` for a run directory.

    The monitor identifies a run by its *event* directory, which is usually
    ``logs/<model>/eval`` but is ``logs/<model>`` for a run copied in from
    elsewhere (see ``catalog.list_runs``).  Previews live beside the events in
    the first case and inside the same directory in the second, so both shapes
    have to be tried rather than assuming the ``eval`` suffix.
    """
    if not run_dir:
        return None
    base = Path(run_dir)
    for candidate in (base / PREVIEW_DIR_NAME, base.parent / PREVIEW_DIR_NAME):
        if candidate.is_dir():
            return candidate
    return None


def _index_epoch(directory: Path) -> tuple[Sample, ...]:
    mels: dict[int, Path] = {}
    audio: dict[tuple[int, str], Path] = {}

    mel_dir = directory / "mel"
    if mel_dir.is_dir():
        for entry in mel_dir.iterdir():
            if entry.suffix.lower() != ".png":
                continue
            match = _SAMPLE_STEM.match(entry.stem)
            if match:
                mels[int(match.group(1))] = entry

    audio_dir = directory / "audio"
    if audio_dir.is_dir():
        for entry in audio_dir.iterdir():
            if entry.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            match = _AUDIO_STEM.match(entry.stem)
            if match:
                audio[(int(match.group(1)), match.group(2))] = entry

    indices = sorted({i for i in mels} | {i for i, _ in audio})
    return tuple(
        Sample(
            index=index,
            mel=mels.get(index),
            generated=audio.get((index, "generated")),
            original=audio.get((index, "original")),
        )
        for index in indices
    )


def list_previews(run_dir: str | os.PathLike[str] | None) -> list[EpochPreviews]:
    """Every epoch with validation artifacts, oldest first.

    Ordered by epoch number rather than by name: ``epoch_10`` sorts before
    ``epoch_9`` as a string, and a run that passes epoch 9999 loses the
    zero-padding that would otherwise hide the problem.
    """
    root = preview_root(run_dir)
    if root is None:
        return []

    found: list[EpochPreviews] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.is_dir():
            continue
        match = _EPOCH_DIR.match(entry.name)
        if not match:
            continue
        samples = _index_epoch(entry)
        if samples:
            found.append(
                EpochPreviews(
                    epoch=int(match.group(1)), directory=entry, samples=samples
                )
            )

    found.sort(key=lambda item: item.epoch)
    return found


def latest_preview(run_dir: str | os.PathLike[str] | None) -> EpochPreviews | None:
    previews = list_previews(run_dir)
    return previews[-1] if previews else None


def total_bytes(previews: list[EpochPreviews]) -> int:
    """Disk cost of the previews, for the "these are not free" line in the UI."""
    total = 0
    for epoch in previews:
        for sample in epoch.samples:
            for path in (sample.mel, sample.generated, sample.original):
                if path is None:
                    continue
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    return total


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
