"""Incremental scalar reader for TensorBoard event files.

The obvious implementation imports ``tensorboard.backend.event_processing``,
but that pulls a second web stack into the GUI process, costs a second or two
at import, and re-reads the whole file on every refresh.  A training run writes
these files for days; the live chart needs to tail one, not reload it.

So this parses the two formats involved directly.  Both are simple:

* the container is TFRecord -- ``uint64`` length, ``uint32`` CRC, payload,
  ``uint32`` CRC.  The CRCs are masked CRC32C, which we skip: a torn record
  from a writer mid-flush is detected by the length running past the end of
  the file, which is the only corruption that actually occurs here.
* the payload is an ``Event`` protobuf, and only two of its fields matter
  (``step`` and ``summary``), so a ~40 line wire-format walk replaces the
  generated bindings.
"""

from __future__ import annotations

import bisect
import os
import struct
from pathlib import Path

_HEADER = struct.Struct("<QI")
_FOOTER = struct.Struct("<I")


def _read_varint(buffer: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(buffer):
        byte = buffer[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
    raise ValueError("truncated varint")


def _iter_fields(buffer: bytes, start: int = 0, end: int | None = None):
    """Yield ``(field_number, wire_type, value)`` over one protobuf message.

    ``value`` is an int for varint/fixed fields and a ``memoryview`` slice for
    length-delimited ones, so nested messages cost no copying.
    """
    pos = start
    end = len(buffer) if end is None else end
    while pos < end:
        key, pos = _read_varint(buffer, pos)
        field, wire = key >> 3, key & 0x7
        if wire == 0:
            value, pos = _read_varint(buffer, pos)
            yield field, wire, value
        elif wire == 1:
            yield field, wire, buffer[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buffer, pos)
            yield field, wire, buffer[pos:pos + length]
            pos += length
        elif wire == 5:
            yield field, wire, buffer[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def _scalar_from_value(payload) -> tuple[str, float] | None:
    """Extract ``(tag, value)`` from one ``Summary.Value``."""
    tag = None
    value = None
    for field, _wire, raw in _iter_fields(payload):
        if field == 1:
            tag = bytes(raw).decode("utf-8", "replace")
        elif field == 2:
            value = struct.unpack("<f", raw)[0]
        elif field == 8 and value is None:
            # A scalar written as a TensorProto: either one packed float in
            # tensor_content (field 4) or one entry in float_val (field 5).
            for tfield, _twire, traw in _iter_fields(raw):
                if tfield in (4, 5) and len(traw) >= 4:
                    value = struct.unpack("<f", traw[:4])[0]
                    break
    if tag is None or value is None:
        return None
    return tag, value


def _scalars_from_event(payload) -> tuple[int, list[tuple[str, float]]]:
    """``(step, [(tag, value), ...])`` of one ``Event``.

    ``payload`` is a ``memoryview`` and stays one all the way down: most of an
    event file by size is preview images and audio, and handing ``bytes``
    copies down copied every one of them just to read the tag in front of it.
    """
    step = 0
    scalars: list[tuple[str, float]] = []
    for field, _wire, raw in _iter_fields(payload):
        if field == 2:
            step = raw
        elif field == 5:
            for sfield, _swire, sraw in _iter_fields(raw):
                if sfield == 1:
                    found = _scalar_from_value(sraw)
                    if found:
                        scalars.append(found)
    return step, scalars


#: Bytes read from an event file per ``read`` call.  The first read of a long
#: run covers hundreds of megabytes, nearly all of it images and audio, and
#: this is what keeps that from being held in memory a whole file at a time.
_CHUNK = 8 * 1024 * 1024


def event_files(run_dir: str | os.PathLike[str]) -> list[Path]:
    """The run's event files, in the order they were started.

    ``events.out.tfevents.<unix time>.<host>...`` -- the creation time is in
    the name, which is what TensorBoard orders by too.  Modification time would
    not do: a file left by a crashed run can be touched after its successor was
    created.
    """
    directory = Path(run_dir)
    try:
        candidates = [
            p for p in directory.iterdir()
            if p.is_file() and p.name.startswith("events.out.tfevents")
        ]
    except OSError:
        return []

    def started(path: Path) -> tuple[int, str]:
        parts = path.name.split(".")
        try:
            return int(parts[3]), path.name
        except (IndexError, ValueError):
            try:
                return int(path.stat().st_mtime), path.name
            except OSError:
                return 0, path.name

    return sorted(candidates, key=started)


class RunReader:
    """Every event file in a run directory, merged, with the newest tailed.

    Call :meth:`poll` on a timer; it returns only what appeared since the last
    call, so the cost is proportional to new data rather than to run length.

    A run resumed from a checkpoint writes a new file that starts at the
    checkpoint's step, while the file before it usually ran on past that point
    before the stop.  Both are read, in order, and for each metric the newer
    file wins from the step it starts at -- TensorBoard's own rule.  This used
    to read only the newest file, so a resumed run showed its last segment
    alone: 73k-98k of a 98k-step pretrain.  A run restarted from scratch
    starts its file at step 0 or 1, which the same rule turns into starting
    over.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        #: Files read so far, in order.  The last is the one being tailed.
        self._files: list[Path] = []
        self._offset = 0
        #: Metrics already seen in the file being tailed; the first value of
        #: each is where the older files' data for it gives way.
        self._file_tags: set[str] = set()
        #: tag -> (steps, values), accumulated across polls.
        self.series: dict[str, tuple[list[int], list[float]]] = {}

    @property
    def active_file(self) -> Path | None:
        return self._files[-1] if self._files else None

    def reset(self) -> None:
        self._files = []
        self._offset = 0
        self._file_tags = set()
        self.series.clear()

    def poll(self) -> dict[str, tuple[list[int], list[float]]]:
        """Read newly appended records.  Returns ``{tag: (steps, values)}``."""
        files = event_files(self.run_dir)
        if not files:
            return {}
        if files[: len(self._files)] != self._files:
            # Not an extension of what has been read: a file went away, or one
            # older than the newest appeared -- a run copied in.  Merging from
            # the middle would be guesswork, so start over.
            self.reset()

        fresh: dict[str, tuple[list[int], list[float]]] = {}
        if not self._files:
            self._open(files[0])
        while True:
            # Finish the file being tailed first: it may have been written up
            # to the moment its successor started.
            self._offset = self._consume(self._files[-1], self._offset, fresh)
            if len(self._files) == len(files):
                return fresh
            self._open(files[len(self._files)])

    def _open(self, path: Path) -> None:
        self._files.append(path)
        self._offset = 0
        self._file_tags = set()

    def _consume(self, path: Path, offset: int, fresh: dict) -> int:
        """Parse every complete record after ``offset``; returns the new offset."""
        try:
            if path.stat().st_size <= offset:
                return offset
            handle = open(path, "rb")
        except OSError:
            return offset
        with handle:
            handle.seek(offset)
            carry = b""
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                data = carry + chunk if carry else chunk
                consumed = self._parse(memoryview(data), fresh)
                offset += consumed
                carry = data[consumed:]
        # What is left in ``carry`` is a record the writer is still in the
        # middle of; the next poll starts at it again.
        return offset

    def _parse(self, view: memoryview, fresh: dict) -> int:
        """Records in ``view`` up to the last complete one; returns bytes used."""
        pos = 0
        while pos + _HEADER.size + _FOOTER.size <= len(view):
            length, _crc = _HEADER.unpack_from(view, pos)
            record_end = pos + _HEADER.size + length + _FOOTER.size
            if record_end > len(view):
                break  # the writer is mid-record, or the record spans chunks
            payload = view[pos + _HEADER.size: pos + _HEADER.size + length]
            pos = record_end
            try:
                step, scalars = _scalars_from_event(payload)
            except (ValueError, struct.error):
                continue
            for tag, value in scalars:
                self._add(tag, step, value, fresh)
        return pos

    def _add(self, tag: str, step: int, value: float, fresh: dict) -> None:
        steps, values = self.series.setdefault(tag, ([], []))
        if tag not in self._file_tags:
            self._file_tags.add(tag)
            # The older files hold this metric past where this one starts: a
            # stretch the run went back over when it resumed.
            cut = bisect.bisect_left(steps, step)
            if cut < len(steps):
                del steps[cut:]
                del values[cut:]
        steps.append(step)
        values.append(value)
        fsteps, fvalues = fresh.setdefault(tag, ([], []))
        fsteps.append(step)
        fvalues.append(value)

    def tags(self) -> list[str]:
        return sorted(self.series)

    def latest(self, tag: str) -> float | None:
        entry = self.series.get(tag)
        return entry[1][-1] if entry and entry[1] else None
