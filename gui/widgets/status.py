"""Status bar: backend state, GPU telemetry, activity.

VRAM is the resource that decides whether a batch size works, and the usual way
to find out is a crash twenty minutes in.  Polling the driver puts the number in
front of the user while they are still choosing the batch size.

The driver is read through NVML (``nvidia-ml-py``, imported as ``pynvml``).
``nvidia-smi`` is itself a front end to that library, so calling it directly
returns the same numbers in well under a millisecond, with no process spawn and
no CSV to parse.  The query still runs on the thread pool: a busy driver can be
slow to answer, and that must stall a timer, never the UI thread.

``nvidia-smi`` through a detached ``QProcess`` remains as the fallback, for an
install that predates the dependency or a driver NVML cannot open.
"""

from __future__ import annotations

import shutil
import threading
from typing import NamedTuple

from PySide6.QtCore import QObject, QProcess, QRunnable, QSize, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from . import icons
from .effects import ChromePanel
from .forms import ghost_button

from ..i18n import _


def _short_gpu_name(name: str) -> str:
    """Trim the vendor boilerplate off an nvidia-smi device name.

    "NVIDIA GeForce RTX 4090" is 23 characters of which 15 are the same on
    every consumer card; the status bar is 34 px tall and shares its row with
    three other readouts, so only the part that identifies the card is worth
    the space.
    """
    for prefix in ("NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip()


class GpuReading(NamedTuple):
    name: str
    utilization: int
    used_bytes: int
    total_bytes: int
    temperature: int


_nvml_lock = threading.Lock()
#: ``None`` until the first query, then the device handle -- or ``False`` once
#: NVML has failed to start, so a failure is remembered rather than retried.
_nvml_handle = None


def _nvml_installed() -> bool:
    """Whether the binding is importable; says nothing about the driver yet."""
    try:
        import pynvml  # noqa: F401
    except ImportError:
        return False
    return True


def _read_nvml() -> GpuReading:
    """One reading of the first device.  Raises if NVML is unusable."""
    global _nvml_handle
    import pynvml

    with _nvml_lock:
        if _nvml_handle is None:
            try:
                pynvml.nvmlInit()
                _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except pynvml.NVMLError:
                _nvml_handle = False
                raise
        if _nvml_handle is False:
            raise RuntimeError("NVML is unavailable")
        handle = _nvml_handle

    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):  # bindings before 11.5 return bytes
        name = name.decode("utf-8", "replace")
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return GpuReading(
        name=name,
        utilization=int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
        used_bytes=int(memory.used),
        total_bytes=int(memory.total),
        temperature=int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
    )


class _NvmlSignals(QObject):
    done = Signal(object)
    failed = Signal()


class _NvmlJob(QRunnable):
    """One NVML reading, off the UI thread."""

    def __init__(self, signals: _NvmlSignals):
        super().__init__()
        self.signals = signals

    def run(self) -> None:  # noqa: D102 - QRunnable's entry point
        try:
            reading = _read_nvml()
        except Exception:  # noqa: BLE001 - the meter falls back, never raises into Qt
            self._deliver(self.signals.failed)
            return
        self._deliver(self.signals.done, reading)

    @staticmethod
    def _deliver(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The window closed while the query was in flight.
            pass


class GpuMeter(QWidget):
    """Live utilisation and memory for the first CUDA device."""

    #: NVML answers in well under a millisecond, so it can afford a live tick.
    #: ``nvidia-smi`` spawns a process each time (~66 ms of system work), so it
    #: keeps the slower one -- still live enough to size a batch against.
    NVML_INTERVAL_MS = 1000
    SMI_INTERVAL_MS = 3000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._smi_available = shutil.which("nvidia-smi") is not None
        self._use_nvml = _nvml_installed()
        self._available = self._use_nvml or self._smi_available
        self._nvml_busy = False
        self._nvml_signals = _NvmlSignals(self)
        self._nvml_signals.done.connect(self._on_nvml)
        self._nvml_signals.failed.connect(self._on_nvml_failed)
        #: Filled by the first successful poll and reused: the device name does
        #: not change, and keeping it means a failed tick does not blank it.
        self._name = ""

        self.label = QLabel("GPU —")
        self.label.setObjectName("StatusText")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedWidth(70)
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)

        if not self._available:
            self.label.setText(_("GPU not detected"))
            self.bar.hide()
            return

        self._timer = QTimer(self)
        self._timer.setInterval(self.NVML_INTERVAL_MS if self._use_nvml else self.SMI_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    def _poll(self) -> None:
        # Nothing to report to a window nobody is looking at.
        window = self.window()
        if window is not None and (window.isMinimized() or not window.isVisible()):
            return
        if self._use_nvml:
            self._poll_nvml()
        else:
            self._poll_smi()

    def _poll_nvml(self) -> None:
        # Skip the tick rather than queue a second query: on a busy GPU the
        # driver can take longer to answer than the poll interval.
        if self._nvml_busy:
            return
        self._nvml_busy = True
        QThreadPool.globalInstance().start(_NvmlJob(self._nvml_signals))

    def _on_nvml(self, reading: GpuReading) -> None:
        self._nvml_busy = False
        self._name = _short_gpu_name(reading.name) or self._name
        self._show(
            reading.utilization,
            reading.used_bytes / 2**30,
            reading.total_bytes / 2**30,
            reading.temperature,
        )

    def _on_nvml_failed(self) -> None:
        self._nvml_busy = False
        # NVML could not open the driver.  nvidia-smi may still manage, or will
        # at least fail in its own words; either way, switch for good.
        self._use_nvml = False
        if not self._smi_available:
            self._timer.stop()
            self.label.setText(_("GPU not detected"))
            self.bar.hide()
            return
        self._timer.setInterval(self.SMI_INTERVAL_MS)
        self._poll_smi()

    def _poll_smi(self) -> None:
        # Same skip-don't-queue rule as the NVML path.
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        process = QProcess(self)
        process.finished.connect(lambda *_: self._read(process))
        process.errorOccurred.connect(lambda *_: self.label.setText(_("GPU query failed")))
        self._process = process
        process.start(
            "nvidia-smi",
            [
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
        )

    def _read(self, process: QProcess) -> None:
        output = bytes(process.readAllStandardOutput()).decode("utf-8", "replace").strip()
        if not output:
            return
        first = output.splitlines()[0]
        # Split from the right: the four telemetry fields are always last, and
        # taking the name as "everything before them" survives a device name
        # that contains a comma rather than silently shifting every column.
        parts = [part.strip() for part in first.split(",")]
        if len(parts) < 5:
            return
        try:
            used_percent, used_mb, total_mb, temperature = [
                int(float(part)) for part in parts[-4:]
            ]
        except ValueError:
            return
        self._name = _short_gpu_name(", ".join(parts[:-4])) or self._name
        self._show(used_percent, used_mb / 1024, total_mb / 1024, temperature)

    def _show(self, percent: int, used_gb: float, total_gb: float, temperature: int) -> None:
        self.bar.setValue(percent)
        self.label.setText(
            f"{self._name or 'GPU'}   {percent}%   "
            f"{used_gb:.1f}/{total_gb:.1f} GB   {temperature}°C"
        )


class StatusBar(ChromePanel):
    """The strip along the bottom of the window."""

    #: Flat, not a gradient: 34 px is not enough height for one to read as
    #: anything but a banding artefact.
    TOP_TOKEN = BOTTOM_TOKEN = "bg_alt"
    DIVIDER_EDGE = "top"

    restartRequested = Signal()
    consoleToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(34)

        self.dot = QLabel("●")
        self.dot.setObjectName("StatusDot")
        self._set_state("idle")

        self.message = QLabel(_("Starting backend…"))
        self.message.setObjectName("StatusText")

        self.activity = QProgressBar()
        self.activity.setRange(0, 0)  # indeterminate
        self.activity.setFixedWidth(110)
        self.activity.setFixedHeight(6)
        self.activity.setTextVisible(False)
        self.activity.hide()

        self.gpu = GpuMeter()
        self.gpu_icon = QLabel()
        self.gpu_icon.setFixedWidth(18)

        self.console_button = ghost_button(_("Log"))
        self.console_button.setIconSize(QSize(15, 15))
        self.console_button.setCheckable(True)
        self.console_button.setChecked(True)
        self.console_button.toggled.connect(self.consoleToggled)

        self.restart_button = ghost_button(_("Restart backend"))
        self.restart_button.clicked.connect(self.restartRequested)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 12, 0)
        layout.setSpacing(10)
        layout.addWidget(self.dot)
        layout.addWidget(self.message)
        layout.addWidget(self.activity)
        layout.addStretch(1)
        layout.addWidget(self.gpu_icon)
        layout.addWidget(self.gpu)
        layout.addWidget(self.console_button)
        layout.addWidget(self.restart_button)

    def apply_theme(self, tokens: dict[str, str]) -> None:
        self.apply_chrome(tokens)
        self.gpu_icon.setPixmap(icons.pixmap("chip", tokens["text_faint"], 15, 1.6))
        self.console_button.setIcon(icons.icon("terminal", tokens["text_dim"], size=15))

    def _set_state(self, state: str) -> None:
        self.dot.setProperty("state", state)
        # Property-driven QSS needs an explicit repolish; Qt does not watch
        # dynamic properties for style invalidation.
        self.dot.style().unpolish(self.dot)
        self.dot.style().polish(self.dot)

    def set_idle(self, text: str = "Ready") -> None:
        self._set_state("ok")
        self.message.setText(text)
        self.activity.hide()

    def set_busy(self, text: str) -> None:
        self._set_state("busy")
        self.message.setText(text)
        self.activity.show()

    def set_error(self, text: str) -> None:
        self._set_state("error")
        self.message.setText(text)
        self.activity.hide()

    def set_starting(self, text: str = "Starting backend…") -> None:
        self._set_state("idle")
        self.message.setText(text)
        self.activity.show()
