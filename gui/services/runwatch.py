"""Reading a run's event file without blocking the UI thread.

The first read of a long run parses the whole file -- half a second on a 36 MB
pretrain -- and doing that between a click on a sidebar row and the page
appearing is indistinguishable from the application hanging.  So the parse goes
to the thread pool and the result comes back by signal.

Two screens need this (the training page's monitor and the diagnostics page),
and the subtleties are not obvious enough to reimplement twice: only one read
may be in flight per reader, a result for a run the user has since switched
away from has to be recognised and dropped, and the window can close while a
read is still running, taking the receiving object with it.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from .tbreader import RunReader

#: How many runs keep a parsed reader alive.  Each holds every scalar series
#: it has read -- a long pretrain is a couple of megabytes of Python lists --
#: so this bounds memory while still covering "compare the last few runs".
READER_CACHE_LIMIT = 4

_readers: "OrderedDict[str, RunReader]" = OrderedDict()
#: Readers with a poll in flight, mapped to everyone waiting for its result.
#: Touched from the UI thread (:func:`start_poll`) and from the pool thread
#: when a read finishes, so it is guarded.
_inflight: dict[int, list["ReadSignals"]] = {}
_lock = threading.Lock()


def _key(run_dir: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(str(run_dir)))


def reader_for(run_dir: str | os.PathLike[str]) -> RunReader:
    """The shared reader for a run directory.

    Two screens watch the same event files, and a :class:`RunReader` is only
    cheap after its first read -- that one parses the file end to end.  Handing
    both the same instance means the run is parsed once no matter how many
    views are open, and they stay on identical data.
    """
    key = _key(run_dir)
    with _lock:
        reader = _readers.get(key)
        if reader is None:
            reader = RunReader(run_dir)
            _readers[key] = reader
            while len(_readers) > READER_CACHE_LIMIT:
                evicted, _ = _readers.popitem(last=False)
                if evicted == key:  # never evict what we just handed out
                    _readers[key] = reader
                    break
        else:
            _readers.move_to_end(key)
        return reader


def forget_readers() -> None:
    """Drop every cached reader.  For tests, and for a hard rescan."""
    with _lock:
        _readers.clear()


def start_poll(reader: RunReader, signals: "ReadSignals") -> bool:
    """Poll ``reader`` off the UI thread, coalescing concurrent callers.

    Returns whether a new read was started.  When one is already running for
    this reader -- the other screen asked a moment ago -- the caller is added
    to that read's listeners instead of starting a second parse over the same
    file.  Both hear about it, so neither is left with its "reading" flag stuck
    on and its timer effectively dead.
    """
    key = id(reader)
    with _lock:
        waiting = _inflight.get(key)
        if waiting is not None:
            if signals not in waiting:
                waiting.append(signals)
            return False
        _inflight[key] = [signals]

    QThreadPool.globalInstance().start(MetricsRead(reader, None))
    return True


def _listeners_for(reader: RunReader) -> list["ReadSignals"]:
    with _lock:
        return _inflight.pop(id(reader), [])


class ReadSignals(QObject):
    """Carries a background read's outcome back to the UI thread."""

    #: ``(reader, whether anything new arrived)``.  The reader is passed back
    #: so a result for a run the user has since switched away from can be
    #: recognised and dropped.
    done = Signal(object, bool)
    failed = Signal(str)


class MetricsRead(QRunnable):
    """One :meth:`RunReader.poll`, off the UI thread.

    Only one of these is ever in flight per reader, so nothing else touches it
    while this runs.  With ``signals`` supplied the result goes to that one
    listener; with ``None`` it goes to everyone :func:`start_poll` collected,
    which is how two screens share a single parse.
    """

    def __init__(self, reader: RunReader, signals: ReadSignals | None):
        super().__init__()
        self._reader = reader
        self._signals = signals

    def _targets(self) -> list[ReadSignals]:
        if self._signals is not None:
            return [self._signals]
        return _listeners_for(self._reader)

    def run(self) -> None:  # noqa: D102 - QRunnable's entry point
        try:
            fresh = bool(self._reader.poll())
        except Exception as error:  # noqa: BLE001 - a torn file must not kill the pool
            for signals in self._targets():
                self._deliver(signals.failed, str(error))
            return
        for signals in self._targets():
            self._deliver(signals.done, self._reader, fresh)

    @staticmethod
    def _deliver(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The window closed while this read was in flight, taking the page
            # that owns the signals with it.  There is nobody left to tell.
            pass
