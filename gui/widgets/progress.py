"""Progress widgets, driven by the backend's own progress lines.

The trainer prints ``[PROGRESS] epoch=7/500 batch=3600/8083 step=48500 ...``
about once a second whenever stdout is not a terminal.  Everything here is
derived from that: the two bars, the counters, and the estimate.

The estimate is deliberately conservative.  A rate measured over the last few
seconds swings wildly -- a checkpoint write, a validation pass or another
process taking the GPU all show up as a stall -- so it is smoothed over a long
window and only shown once there is enough history to mean something.

Preprocessing and extraction report through ``[TASK]`` lines instead, which
:class:`ProgressButton` turns into a fill inside the button that started them.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from . import QWIDGETSIZE_MAX

from ..i18n import _

#: Emitted by ``_emit_machine_progress`` in rvc/train/train.py.
PROGRESS_LINE = re.compile(
    r"\[PROGRESS\]\s+epoch=(\d+)/(\d+)\s+batch=(\d+)/(\d+)\s+step=(\d+)(?P<rest>.*)"
)

#: Samples kept for the rate estimate, at ~1 Hz from the trainer.
_RATE_WINDOW = 120
#: Below this many samples the estimate is too noisy to show.
_RATE_MINIMUM = 15


def parse_progress(line: str) -> dict | None:
    """Decode one trainer progress line, or ``None`` if it is not one."""
    match = PROGRESS_LINE.search(line)
    if not match:
        return None
    return {
        "epoch": int(match.group(1)),
        "total_epochs": int(match.group(2)),
        "batch": int(match.group(3)),
        "total_batches": int(match.group(4)),
        "step": int(match.group(5)),
        "metrics": match.group("rest").strip(),
    }


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 86400:
        return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


class TrainingProgress(QWidget):
    """Overall and per-epoch progress for a running trainer."""

    stopRequested = Signal()

    #: Long enough to be seen as a movement rather than a flicker, short
    #: enough that pressing Start does not feel like waiting for something.
    REVEAL_MS = 240
    DISMISS_MS = 260

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._height_animation: QPropertyAnimation | None = None
        self.setObjectName("Card")
        # The other cards are QFrames, which paint a stylesheet background on
        # their own; a plain QWidget needs telling.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._samples: list[tuple[float, int]] = []
        self._started = 0.0
        self._latest: dict | None = None

        self.title = QLabel(_("Training"))
        self.title.setObjectName("CardTitle")

        self.epoch_badge = QLabel("—")
        self.epoch_badge.setObjectName("Badge")

        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("FieldHint")

        # This card is the only training control visible once it floats over
        # another page, so it carries its own stop.
        self.stop_button = QPushButton(_("Stop"))
        self.stop_button.setObjectName("Chip")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(self._request_stop)
        self.stop_button.hide()

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self.title)
        header.addWidget(self.epoch_badge)
        header.addStretch(1)
        header.addWidget(self.elapsed_label)
        header.addWidget(self.stop_button)

        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setValue(0)
        self.overall.setTextVisible(False)
        self.overall.setFixedHeight(8)

        self.epoch_bar = QProgressBar()
        self.epoch_bar.setRange(0, 1000)
        self.epoch_bar.setValue(0)
        self.epoch_bar.setTextVisible(False)
        self.epoch_bar.setFixedHeight(4)
        self.epoch_bar.setObjectName("Secondary")

        self.detail = QLabel(_("Waiting for the first batch…"))
        self.detail.setObjectName("FieldHint")
        self.detail.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.overall)
        layout.addWidget(self.epoch_bar)
        layout.addWidget(self.detail)

    # -- lifecycle ---------------------------------------------------------

    def begin(self, total_epochs: int) -> None:
        """Reset for a new run and show the indeterminate state."""
        self._samples.clear()
        self._started = time.monotonic()
        self._latest = None
        self.epoch_badge.setText(f"0 / {total_epochs}")
        self.elapsed_label.setText("")
        # Indeterminate until the first line arrives: a bar sitting at 0% while
        # the trainer loads its dataset reads as a hang.
        self.overall.setRange(0, 0)
        self.epoch_bar.setValue(0)
        self.detail.setText(_("Starting the trainer… (loading models and dataset)"))

    # -- appearing and going away ------------------------------------------

    def reveal(self) -> None:
        """Grow into place rather than shoving the layout aside in one frame.

        The height is what is animated, not the opacity: a fade needs a
        graphics effect, and this widget's only effect slot is spoken for by
        the card shadow the light theme puts on it.  Growing also keeps the
        panel below from jumping, which a fade in place would not.
        """
        target = self.sizeHint().height()
        self.setMaximumHeight(0)
        self.show()
        self._animate_height(
            target, self.REVEAL_MS, QEasingCurve.OutCubic,
            # Give the ceiling back afterwards: the detail line wraps to two
            # rows on a narrow column, and a card frozen at its opening height
            # would clip it.
            on_done=lambda: self.setMaximumHeight(QWIDGETSIZE_MAX),
        )

    def dismiss(self, on_done: Callable[[], None] | None = None) -> None:
        """Collapse away, then hide.  ``on_done`` runs once it is gone."""
        if not self.isVisible():
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            if on_done is not None:
                on_done()
            return

        def landed() -> None:
            self.hide()
            # Restored so the next run's reveal starts from a sane widget
            # rather than one pinned to zero.
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            if on_done is not None:
                on_done()

        self.setMaximumHeight(self.height())
        self._animate_height(0, self.DISMISS_MS, QEasingCurve.InCubic, on_done=landed)

    def _animate_height(
        self,
        target: int,
        duration: int,
        curve: QEasingCurve.Type,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        if self._height_animation is not None:
            # Its own stateChanged handler runs, so whatever it was going to
            # restore is restored before this one starts.
            self._height_animation.stop()

        animation = QPropertyAnimation(self, b"maximumHeight", self)
        animation.setDuration(duration)
        animation.setStartValue(self.maximumHeight())
        animation.setEndValue(target)
        animation.setEasingCurve(curve)

        def on_state(new_state, _old_state) -> None:
            if new_state != QAbstractAnimation.Stopped:
                return
            self._height_animation = None
            animation.deleteLater()
            if on_done is not None:
                on_done()

        animation.stateChanged.connect(on_state)
        self._height_animation = animation
        animation.start()

    def set_stoppable(self, stoppable: bool) -> None:
        """Whether there is a run to stop."""
        self.stop_button.setVisible(stoppable)
        self.stop_button.setEnabled(stoppable)
        self.stop_button.setText(_("Stop"))

    def _request_stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.stop_button.setText(_("Stopping…"))
        self.stopRequested.emit()

    def finish(self, message: str = "") -> None:
        self.set_stoppable(False)
        self.overall.setRange(0, 1000)
        self.overall.setValue(1000 if not message else self.overall.value())
        self.detail.setText(message or "Finished.")

    # -- updates -----------------------------------------------------------

    def consume(self, line: str) -> bool:
        """Feed a log line.  Returns whether it was a progress line."""
        update = parse_progress(line)
        if update is None:
            return False
        self._apply(update)
        return True

    def _apply(self, update: dict) -> None:
        self._latest = update
        epoch = update["epoch"]
        total_epochs = max(1, update["total_epochs"])
        batch = update["batch"]
        total_batches = max(1, update["total_batches"])

        if self.overall.maximum() == 0:
            self.overall.setRange(0, 1000)

        # Completed epochs plus the fraction of the current one.
        fraction = ((epoch - 1) + batch / total_batches) / total_epochs
        fraction = max(0.0, min(1.0, fraction))
        self.overall.setValue(int(fraction * 1000))
        self.epoch_bar.setValue(int(batch / total_batches * 1000))
        self.epoch_badge.setText(f"{epoch} / {total_epochs}")

        now = time.monotonic()
        self._samples.append((now, self._absolute_batch(update)))
        if len(self._samples) > _RATE_WINDOW:
            del self._samples[: len(self._samples) - _RATE_WINDOW]

        parts = [
            f"{fraction * 100:.1f}%",
            f"batch {batch:,} / {total_batches:,}",
            f"step {update['step']:,}",
        ]
        if update["metrics"]:
            parts.append(update["metrics"])
        remaining = self._remaining_seconds(update)
        if remaining is not None:
            parts.append(f"~{_clock(remaining)} left")
        self.detail.setText("   ·   ".join(parts))

        if self._started:
            self.elapsed_label.setText(
                _("{duration} elapsed").format(duration=_clock(now - self._started))
            )

    @staticmethod
    def _absolute_batch(update: dict) -> int:
        """Batches completed since the run started, across epochs."""
        return (update["epoch"] - 1) * update["total_batches"] + update["batch"]

    def _remaining_seconds(self, update: dict) -> float | None:
        if len(self._samples) < _RATE_MINIMUM:
            return None
        (first_time, first_batch) = self._samples[0]
        (last_time, last_batch) = self._samples[-1]
        elapsed = last_time - first_time
        done = last_batch - first_batch
        if elapsed <= 0 or done <= 0:
            return None
        rate = done / elapsed
        total = update["total_epochs"] * update["total_batches"]
        return max(0.0, (total - last_batch) / rate)


#: Emitted by ``_ReportingProgress`` in rvc/lib/terminal.py for every Rich bar
#: when the output is piped: ``[TASK] 120/5166 Slicing & resampling``.
TASK_LINE = re.compile(r"\[TASK\]\s+(\d+)/(\d+|\?)\s+(.*)")


def parse_task(line: str) -> tuple[str, int, int | None] | None:
    """``(description, done, total)`` for one task line, or ``None``."""
    match = TASK_LINE.search(line)
    if not match:
        return None
    total = None if match.group(2) == "?" else int(match.group(2))
    return match.group(3).strip(), int(match.group(1)), total


class ProgressButton(QPushButton):
    """A run button that fills up with the progress of the job it started.

    While a job runs the button is disabled, so its face is free to become the
    progress bar: the fill is painted between the stylesheet's bevel and the
    label, and the label carries the stage and the percentage.

    Several bars can run at once -- extraction starts one per device -- so
    tasks are grouped by the first word of their description ("F0", "Features")
    and the most recent group is summed.  A group that has not reported a total
    yet, or no line at all, shows a sweeping band instead of a fake 0%.
    """

    _SWEEP_MS = 30

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._idle_text = text
        self._active = False
        self._fraction: float | None = None
        self._sweep = 0.0
        self._tasks: dict[str, tuple[int, int | None]] = {}
        self._phase = ""
        self._fill = QColor(255, 255, 255, 60)
        self._timer = QTimer(self)
        self._timer.setInterval(self._SWEEP_MS)
        self._timer.timeout.connect(self._advance_sweep)

    # -- lifecycle ---------------------------------------------------------

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        """Remember the caller's label; the running label is ours."""
        if not self._active:
            self._idle_text = text
        super().setText(text)

    def queue(self, text: str) -> None:
        """Submitted, possibly waiting behind another job in the worker."""
        self._active = True
        self._tasks.clear()
        self._phase = ""
        self._fraction = None
        self._sweep = 0.0
        super().setText(text)
        self._set_running(True)
        self._timer.start()
        self.update()

    def start(self, text: str) -> None:
        """The worker has begun this job."""
        if self._active and not self._tasks:
            super().setText(text)

    def finish(self) -> None:
        self._active = False
        self._timer.stop()
        self._tasks.clear()
        self._fraction = None
        super().setText(self._idle_text)
        self._set_running(False)
        self.update()

    def consume(self, line: str) -> bool:
        """Feed a log line.  Returns whether it was a task line."""
        if not self._active:
            return False
        parsed = parse_task(line)
        if parsed is None:
            return False
        description, done, total = parsed
        phase = description.split(" ", 1)[0]
        if phase != self._phase:
            # A new stage replaces the old one rather than adding to it: the
            # stages run one after another, and summing them would make the
            # bar jump backwards each time a new one reported its total.
            self._phase = phase
            self._tasks = {key: value for key, value in self._tasks.items()
                           if key.split(" ", 1)[0] == phase}
        self._tasks[description] = (done, total)

        totals = [task_total for _done, task_total in self._tasks.values()]
        finished = sum(task_done for task_done, _total in self._tasks.values())
        if any(value is None for value in totals) or not sum(totals):
            self._fraction = None
            label = f"{description} · {finished}"
            if not self._timer.isActive():
                self._timer.start()
        else:
            total = sum(totals)
            self._fraction = max(0.0, min(1.0, finished / total))
            self._timer.stop()
            name = description if len(self._tasks) == 1 else phase
            label = f"{name} · {finished}/{total} · {self._fraction * 100:.0f}%"
        super().setText(label)
        self.update()
        return True

    # -- appearance --------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        """Flip the ``running`` property style.qss keys the label colour on.

        A disabled button's label is faint, which is right for "not now" and
        wrong for "this is what is happening".
        """
        self.setProperty("running", running)
        self.style().unpolish(self)
        self.style().polish(self)

    def apply_theme(self, tokens: dict[str, str]) -> None:
        colour = QColor(tokens.get("accent", "#8b5cf6"))
        colour.setAlpha(110)
        self._fill = colour
        self.update()

    def _advance_sweep(self) -> None:
        self._sweep = (self._sweep + 0.012) % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._active:
            super().paintEvent(event)
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        painter = QStylePainter(self)
        painter.drawControl(QStyle.CE_PushButtonBevel, option)

        inner = self.rect().adjusted(1, 1, -1, -1)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        # 8 px in style.qss, less the 1 px border.
        clip.addRoundedRect(QRectF(inner), 7, 7)
        painter.setClipPath(clip)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._fill)
        if self._fraction is None:
            band = inner.width() * 0.3
            left = inner.left() - band + (inner.width() + band) * self._sweep
            painter.drawRect(QRectF(left, inner.top(), band, inner.height()))
        else:
            painter.drawRect(QRectF(inner.left(), inner.top(),
                                    inner.width() * self._fraction, inner.height()))
        painter.restore()

        painter.drawControl(QStyle.CE_PushButtonLabel, option)


def progress_button(text: str) -> ProgressButton:
    """A :func:`~.forms.primary_button` that can show a job's progress."""
    button = ProgressButton(text)
    button.setObjectName("Primary")
    button.setMinimumHeight(38)
    button.setCursor(Qt.PointingHandCursor)
    return button
