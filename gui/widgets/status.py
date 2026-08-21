"""Status bar: backend state, GPU telemetry, activity.

VRAM is the resource that decides whether a batch size works, and the usual way
to find out is a crash twenty minutes in.  Polling ``nvidia-smi`` puts the
number in front of the user while they are still choosing the batch size.

The poll runs as a detached ``QProcess`` rather than ``subprocess.run`` so a
hung driver query stalls a timer, never the UI thread.
"""

from __future__ import annotations

import shutil

from PySide6.QtCore import QProcess, QSize, QTimer, Signal
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


class GpuMeter(QWidget):
    """Live utilisation and memory for the first CUDA device."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._available = shutil.which("nvidia-smi") is not None
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

        # Each tick spawns a process, which costs ~66 ms of system work even
        # though it is off the UI thread.  Three seconds is still live enough
        # to size a batch against, at two thirds of the churn.
        self._timer = QTimer(self)
        self._timer.setInterval(3000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    def _poll(self) -> None:
        # Skip the tick rather than queue a second query: on a busy GPU the
        # driver can take longer to answer than the poll interval.
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        # Nothing to report to a window nobody is looking at.
        window = self.window()
        if window is not None and (window.isMinimized() or not window.isVisible()):
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

        self.bar.setValue(used_percent)
        self.label.setText(
            f"{self._name or 'GPU'}   {used_percent}%   "
            f"{used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB   {temperature}°C"
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
