"""GUI-side client for the backend worker process.

Owns a single :class:`QProcess` running :mod:`gui.services.worker`, turns its
stdout into signals, and hands results back to callers as callbacks on the UI
thread.  Views never touch ``QProcess`` themselves -- they call :meth:`Engine.call`
and get a callback, which is the only concurrency primitive the rest of the GUI
is allowed to know about.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from . import paths

SENTINEL = "\x1e"

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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._next_id = 1
        self._pending: dict[int, tuple[ResultCallback | None, ErrorCallback | None]] = {}
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
            if not process.waitForFinished(3000):
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

        self._pending[job_id] = (on_result, on_error)
        payload = json.dumps({"id": job_id, "cmd": cmd, "args": args or {}})
        process.write(payload.encode("utf-8") + b"\n")
        return job_id

    # -- stream handling ---------------------------------------------------

    def _drain_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        chunk = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buffer += chunk
        while "\n" in self._stderr_buffer:
            line, self._stderr_buffer = self._stderr_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.log.emit(line)

    def _handle_line(self, line: str) -> None:
        if not line.startswith(SENTINEL):
            if line:
                self.log.emit(line)
            return
        try:
            message = json.loads(line[len(SENTINEL):])
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
        elif kind in ("result", "error"):
            self._running.discard(job_id)
            on_result, on_error = self._pending.pop(job_id, (None, None))
            if kind == "result":
                if on_result:
                    on_result(message.get("data") or {})
            else:
                detail = message.get("traceback") or ""
                if detail:
                    self.log.emit(detail.rstrip())
                if on_error:
                    on_error(message.get("error", "unknown backend error"))

        if self.busy != was_busy:
            self.busy_changed.emit(self.busy)

    def _on_finished(self, exit_code: int, _status) -> None:
        self._is_ready = False
        for _job_id, (_on_result, on_error) in list(self._pending.items()):
            if on_error:
                on_error("The backend exited before answering.")
        self._pending.clear()
        had_jobs = bool(self._running)
        self._running.clear()
        if had_jobs:
            self.busy_changed.emit(False)
        if not self._shutting_down:
            self.crashed.emit(
                f"The backend process exited with code {exit_code}. "
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
