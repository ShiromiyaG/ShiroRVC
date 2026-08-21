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


def _scalar_from_value(payload: bytes) -> tuple[str, float] | None:
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
            for tfield, _twire, traw in _iter_fields(bytes(raw)):
                if tfield in (4, 5) and len(traw) >= 4:
                    value = struct.unpack("<f", bytes(traw)[:4])[0]
                    break
    if tag is None or value is None:
        return None
    return tag, value


def _scalars_from_event(payload: bytes) -> tuple[int, list[tuple[str, float]]]:
    step = 0
    scalars: list[tuple[str, float]] = []
    for field, _wire, raw in _iter_fields(payload):
        if field == 2:
            step = raw
        elif field == 5:
            for sfield, _swire, sraw in _iter_fields(bytes(raw)):
                if sfield == 1:
                    found = _scalar_from_value(bytes(sraw))
                    if found:
                        scalars.append(found)
    return step, scalars


class RunReader:
    """Tails the newest event file in a run directory.

    Call :meth:`poll` on a timer; it returns only what appeared since the last
    call, so the cost is proportional to new data rather than to run length.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self._path: Path | None = None
        self._offset = 0
        #: tag -> (steps, values), accumulated across polls.
        self.series: dict[str, tuple[list[int], list[float]]] = {}

    @property
    def active_file(self) -> Path | None:
        return self._path

    def _newest_event_file(self) -> Path | None:
        if not self.run_dir.is_dir():
            return None
        candidates = [
            p for p in self.run_dir.iterdir()
            if p.is_file() and p.name.startswith("events.out.tfevents")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def reset(self) -> None:
        self._path = None
        self._offset = 0
        self.series.clear()

    def poll(self) -> dict[str, tuple[list[int], list[float]]]:
        """Read newly appended records.  Returns ``{tag: (steps, values)}``."""
        newest = self._newest_event_file()
        if newest is None:
            return {}
        if newest != self._path:
            # A restarted run writes a fresh file; start over rather than
            # splicing two runs into one line.
            self._path = newest
            self._offset = 0
            self.series.clear()

        try:
            size = newest.stat().st_size
            if size <= self._offset:
                return {}
            with open(newest, "rb") as handle:
                handle.seek(self._offset)
                blob = handle.read(size - self._offset)
        except OSError:
            return {}

        fresh: dict[str, tuple[list[int], list[float]]] = {}
        pos = 0
        consumed = 0
        while pos + _HEADER.size + _FOOTER.size <= len(blob):
            length, _crc = _HEADER.unpack_from(blob, pos)
            record_end = pos + _HEADER.size + length + _FOOTER.size
            if record_end > len(blob):
                break  # writer is mid-record; pick it up next poll
            payload = blob[pos + _HEADER.size: pos + _HEADER.size + length]
            pos = record_end
            consumed = pos
            try:
                step, scalars = _scalars_from_event(payload)
            except (ValueError, struct.error):
                continue
            for tag, value in scalars:
                steps, values = self.series.setdefault(tag, ([], []))
                steps.append(step)
                values.append(value)
                fsteps, fvalues = fresh.setdefault(tag, ([], []))
                fsteps.append(step)
                fvalues.append(value)

        self._offset += consumed
        return fresh

    def tags(self) -> list[str]:
        return sorted(self.series)

    def latest(self, tag: str) -> float | None:
        entry = self.series.get(tag)
        return entry[1][-1] if entry and entry[1] else None
