"""GUI-side client for the backend worker process.

Owns a single :class:`QProcess` running :mod:`gui.services.worker`, turns its
stdout into signals, and hands results back to callers as callbacks on the UI
thread.  Views never touch ``QProcess`` themselves -- they call :meth:`Engine.call`
and get a callback, which is the only concurrency primitive the rest of the GUI
is allowed to know about.
"""

from __future__ import annotations

import codecs
import json
from typing import Any, Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from . import paths

SENTINEL = "\x1e"

#: How long :meth:`Engine.shutdown` waits for a worker that is stopping a
#: training run.  The trainer defers a stop until any checkpoint write in
#: flight is on disk, and ``core.TRAINING_STOP_GRACE_SECONDS`` (45 s) plus the
#: tree kill after it is what the worker may spend; mirrored, since gui/ may
#: not import core.  Killing the worker sooner orphans the trainer on Windows.
TRAINING_SHUTDOWN_MS = 60_000
SHUTDOWN_MS = 3_000

#: NTSTATUS values a crashed Windows process exits with.  Mirrors
#: ``core.describe_exit_code``; gui/ may not import core.
_NTSTATUS = {
    0xC0000005: "access violation - a native crash, usually a driver or a CUDA kernel",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC0000135: "a required DLL was not found",
    0xC000013A: "interrupted with Ctrl+C",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun / fatal abort",
    0xE06D7363: "unhandled C++ exception",
}


def describe_exit(exit_code: int, crashed: bool) -> str:
    unsigned = exit_code & 0xFFFFFFFF
    name = _NTSTATUS.get(unsigned)
    if name:
        return f"exit code 0x{unsigned:08X} ({name})"
    if exit_code == 0 and not crashed:
        return "exit code 0 (a clean exit: the worker was told to stop or lost its command pipe)"
    return f"exit code {exit_code}" + (" (crashed)" if crashed else "")


ResultCallback = Callable[[dict], None]
ErrorCallback = Callable[[str], None]


class Engine(QObject):
    """Lifecycle and request routing for the backend process."""

    #: A line of backend output that is not a control message.
    log = Signal(str)
    #: The worker announced itself; commands may now be sent.
    ready = Signal()
    #: The worker exited without being asked to.  Carries a human explanation.
    crashed = Signal(str)
    #: A job started or the last one finished.
    busy_changed = Signal(bool)
    #: The worker began running this job id (it may have waited in its queue).
    job_started = Signal(int)
    #: This job id got its answer, or never will because the worker died.
    job_finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        # Incremental, because a pipe read ends wherever it ends: a character
        # split across two reads -- any accented one in a Portuguese log line
        # or a path -- decoded chunk by chunk came out as two "�".
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._next_id = 1
        self._pending: dict[int, tuple[str, ResultCallback | None, ErrorCallback | None]] = {}
        self._running: set[int] = set()
        self._is_ready = False
        self._shutting_down = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def busy(self) -> bool:
        return bool(self._running)

    def is_pending(self, job_id: int) -> bool:
        """Whether this job was sent and still awaits its answer."""
        return job_id in self._pending

    @property
    def is_training(self) -> bool:
        """Whether a training job was sent and has not answered yet."""
        return any(cmd == "train" for cmd, _result, _error in self._pending.values())

    def start(self) -> None:
        if self._process is not None:
            return

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.setWorkingDirectory(str(paths.ROOT))

        environment = QProcessEnvironment.systemEnvironment()
        # Unbuffered so log lines reach the console as they happen rather than
        # in 8 KB bursts, and UTF-8 so model names with non-ASCII survive the
        # trip on a Windows console defaulting to cp1252.
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        # rich checks this before deciding to emit colour; the console widget
        # strips ANSI anyway, but not generating it keeps the logs cheaper.
        environment.insert("NO_COLOR", "1")
        environment.insert("TERM", "dumb")
        process.setProcessEnvironment(environment)

        self._stdout_decoder.reset()
        self._stderr_decoder.reset()
        process.readyReadStandardOutput.connect(self._drain_stdout)
        process.readyReadStandardError.connect(self._drain_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)

        self._process = process
        process.start(paths.python_executable(), ["-u", "-m", "gui.services.worker"])

    def shutdown(self) -> None:
        """Ask the worker to exit, then make sure it did.

        A training run can hold the GPU for hours; leaving the worker behind
        because the window closed is how people end up rebooting to reclaim
        VRAM.  Terminate is the polite ask, kill is the guarantee.

        With a run going the worker stops it first, which can take most of a
        minute; killing the worker three seconds in would leave the trainer
        running without it.
        """
        self._shutting_down = True
        process = self._process
        if process is None:
            return
        if process.state() != QProcess.NotRunning:
            try:
                process.write(json.dumps({"id": 0, "cmd": "shutdown"}).encode() + b"\n")
                process.waitForBytesWritten(200)
            except (OSError, RuntimeError):
                pass
            grace = TRAINING_SHUTDOWN_MS if self.is_training else SHUTDOWN_MS
            if not process.waitForFinished(grace):
                process.kill()
                process.waitForFinished(1000)
        self._process = None
        self._is_ready = False

    def restart(self) -> None:
        self.shutdown()
        self._shutting_down = False
        self._pending.clear()
        self._running.clear()
        self.start()

    # -- requests ----------------------------------------------------------

    def call(
        self,
        cmd: str,
        args: dict[str, Any] | None = None,
        on_result: ResultCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> int:
        """Send one command.  Returns the job id the reply will carry."""
        process = self._process
        job_id = self._next_id
        self._next_id += 1

        if process is None or process.state() == QProcess.NotRunning:
            if on_error:
                on_error("The backend is not running. Use Restart backend to recover.")
            return job_id

        self._pending[job_id] = (cmd, on_result, on_error)
        payload = json.dumps({"id": job_id, "cmd": cmd, "args": args or {}})
        process.write(payload.encode("utf-8") + b"\n")
        return job_id

    # -- stream handling ---------------------------------------------------

    def _drain_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        chunk = self._stdout_decoder.decode(bytes(process.readAllStandardOutput()))
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        chunk = self._stderr_decoder.decode(bytes(process.readAllStandardError()))
        self._stderr_buffer += chunk
        while "\n" in self._stderr_buffer:
            line, self._stderr_buffer = self._stderr_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.log.emit(line)

    def _handle_line(self, line: str) -> None:
        # Searched for, not expected at column zero.  Only the worker's own
        # messages go through its write lock; a library writing a line in
        # pieces -- or straight to the descriptor from C -- can leave a
        # fragment ahead of one.  Read as a log line, that message was lost,
        # and a lost ``result`` leaves its job running forever.
        head, sentinel, payload = line.partition(SENTINEL)
        if not sentinel:
            if line:
                self.log.emit(line)
            return
        if head.strip():
            self.log.emit(head)
        try:
            message = json.loads(payload)
        except ValueError:
            self.log.emit(line)
            return
        self._handle_message(message)

    def _handle_message(self, message: dict) -> None:
        kind = message.get("type")

        if kind == "ready":
            self._is_ready = True
            self.ready.emit()
            return

        job_id = message.get("id")
        was_busy = self.busy

        if kind == "started":
            self._running.add(job_id)
            if isinstance(job_id, int):
                self.job_started.emit(job_id)
        elif kind in ("result", "error"):
            self._running.discard(job_id)
            cmd, on_result, on_error = self._pending.pop(job_id, ("?", None, None))
            if kind == "result":
                if on_result:
                    on_result(message.get("data") or {})
            else:
                detail = message.get("traceback") or ""
                if detail:
                    self.log.emit(detail.rstrip())
                self.log.emit(f"✖ {cmd}: {message.get('error', 'unknown backend error')}")
                if on_error:
                    on_error(message.get("error", "unknown backend error"))
            if isinstance(job_id, int):
                self.job_finished.emit(job_id)

        if self.busy != was_busy:
            self.busy_changed.emit(self.busy)

    def _on_finished(self, exit_code: int, status) -> None:
        # Whatever the worker wrote last -- usually the interesting part -- may
        # still be unread or sitting in a buffer without its trailing newline.
        self._drain_stdout()
        self._drain_stderr()
        for attr in ("_stdout_buffer", "_stderr_buffer"):
            rest = getattr(self, attr).strip()
            setattr(self, attr, "")
            if rest:
                self.log.emit(rest)

        self._is_ready = False
        reason = describe_exit(exit_code, status == QProcess.CrashExit)
        for job_id, (cmd, _on_result, on_error) in list(self._pending.items()):
            if on_error:
                on_error(f"The backend exited while running {cmd}: {reason}.")
            self.job_finished.emit(job_id)
        self._pending.clear()
        had_jobs = bool(self._running)
        self._running.clear()
        if had_jobs:
            self.busy_changed.emit(False)
        if not self._shutting_down:
            self.crashed.emit(
                f"The backend process stopped: {reason}. "
                "The log above usually says why."
            )

    def _on_process_error(self, error) -> None:
        if self._shutting_down:
            return
        if error == QProcess.FailedToStart:
            self.crashed.emit(
                f"Could not start the backend with {paths.python_executable()!r}. "
                "Is the environment installed?"
            )


_instance: Engine | None = None


def instance() -> Engine:
    """The process-wide engine.

    One worker per window is the point -- a second one would mean a second copy
    of every model in VRAM.
    """
    global _instance
    if _instance is None:
        _instance = Engine()
    return _instance
