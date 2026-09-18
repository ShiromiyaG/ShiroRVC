"""Waveform display and playback.

Being able to hear the result without leaving the window is the main thing a
native front-end buys over a browser page, and seeing the waveform is how you
spot a clipped or truncated conversion in one glance instead of one listen.

Peak extraction runs on a thread pool: decoding a five-minute file takes long
enough to drop frames if it happens on the UI thread.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import (
    QLoggingCategory,
    QObject,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..i18n import _
from . import icons

#: FFmpeg's ``AV_LOG_WARNING``.
_AV_LOG_WARNING = 24


def quiet_media_logs() -> None:
    """Keep the media backend's per-file chatter out of the terminal.

    Qt leaves FFmpeg at its default INFO level, so every file the player opens
    prints a full stream dump.  Warnings still come through: they are how a
    malformed file shows up.  ``QT_LOGGING_RULES`` still overrides the Qt half.
    """
    QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg.info=false")
    try:
        import ctypes

        import PySide6

        # The copy PySide6 bundles is the one Qt's plugin links against, and
        # loading it first means the plugin gets this same instance.
        base = Path(PySide6.__file__).parent
        found = sorted(
            [
                *base.glob("Qt/lib/libavutil.so.*"),
                *base.glob("Qt/lib/libavutil.*.dylib"),
                *base.glob("avutil-*.dll"),
            ],
            key=lambda path: len(path.name),
        )
        if found:
            ctypes.CDLL(str(found[0])).av_log_set_level(_AV_LOG_WARNING)
    except Exception:  # noqa: BLE001 - cosmetic; never block startup
        pass


#: Envelope resolution.  More than this and the extra detail is sub-pixel on
#: any window a person actually uses.
_PEAK_BUCKETS = 1200


def _same_path(a: str, b: str) -> bool:
    """Whether two paths name one file.  ``samefile`` needs both to exist,
    and the point of asking is often a file that is about to be written."""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class _PeakSignals(QObject):
    done = Signal(str, list, float)
    failed = Signal(str, str)


class _PeakJob(QRunnable):
    """Decode a file down to a min/max envelope."""

    def __init__(self, path: str, signals: _PeakSignals):
        super().__init__()
        self.path = path
        self.signals = signals

    @Slot()
    def run(self) -> None:
        try:
            import numpy as np
            import soundfile as sf

            with sf.SoundFile(self.path) as handle:
                frames = len(handle)
                rate = handle.samplerate
                # Read as float32 mono; blocks keep memory flat for long files.
                block = max(1, frames // _PEAK_BUCKETS)
                peaks: list[tuple[float, float]] = []
                for chunk in handle.blocks(blocksize=block, dtype="float32", always_2d=True):
                    if not len(chunk):
                        continue
                    mono = chunk.mean(axis=1)
                    peaks.append((float(np.min(mono)), float(np.max(mono))))
            duration = frames / rate if rate else 0.0
            self._deliver(self.signals.done, self.path, peaks, duration)
        except Exception as error:  # noqa: BLE001 - shown, not raised into Qt
            self._deliver(self.signals.failed, self.path, str(error))

    @staticmethod
    def _deliver(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The player was destroyed while this scan was in flight, taking
            # the signal object with it.  Routine when a page switches epochs
            # quickly or closes mid-load, and there is nobody left to tell.
            pass


class Waveform(QWidget):
    """Envelope view with a playhead; click or drag to seek."""

    seekRequested = Signal(float)  # 0..1
    #: A local audio file dropped on the view; see :meth:`accept_files`.
    fileDropped = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._peaks: list[tuple[float, float]] = []
        self._progress = 0.0
        self._empty_message = _("No audio loaded")
        self._message = self._empty_message
        self._drop_suffixes: tuple[str, ...] = ()
        self._drop_hover = False
        self.colours = {
            "bg": QColor("#14141a"),
            "wave": QColor("#3d3d4d"),
            "played": QColor("#60a5fa"),
            "head": QColor("#e8e8ef"),
            "text": QColor("#6b6b7d"),
            "mid": QColor("#2e2e3a"),
        }

    def apply_theme(self, tokens: dict[str, str]) -> None:
        self.colours["bg"] = QColor(tokens["input"])
        self.colours["wave"] = QColor(tokens["border_strong"])
        self.colours["played"] = QColor(tokens["accent"])
        self.colours["head"] = QColor(tokens["text"])
        self.colours["text"] = QColor(tokens["text_faint"])
        self.colours["mid"] = QColor(tokens["border"])
        self.update()

    def set_peaks(self, peaks: list[tuple[float, float]]) -> None:
        self._peaks = peaks
        self._message = "" if peaks else self._empty_message
        self.update()

    def accept_files(self, suffixes: tuple[str, ...], empty_message: str) -> None:
        """Take drops of files with these suffixes, and say so while empty."""
        self._drop_suffixes = tuple(suffix.lower() for suffix in suffixes)
        if self._message == self._empty_message:
            self._message = empty_message
        self._empty_message = empty_message
        self.setAcceptDrops(True)

    def _dropped_path(self, event) -> str:
        urls = event.mimeData().urls()
        path = os.path.normpath(urls[0].toLocalFile()) if urls and urls[0].isLocalFile() else ""
        return path if os.path.isfile(path) and path.lower().endswith(self._drop_suffixes) else ""

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._dropped_path(event):
            event.acceptProposedAction()
            self._drop_hover = True
            self.update()

    def dragLeaveEvent(self, _event) -> None:  # noqa: N802
        self._drop_hover = False
        self.update()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._drop_hover = False
        self.update()
        path = self._dropped_path(event)
        if path:
            event.acceptProposedAction()
            self.fileDropped.emit(path)

    def set_message(self, text: str) -> None:
        self._peaks = []
        self._message = text
        self.update()

    def set_progress(self, fraction: float) -> None:
        self._progress = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        # Smoothed for the rounded box only; the columns below stay crisp.
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        # An accent outline while a droppable file hovers over the box.
        if self._drop_hover:
            rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(QPen(self.colours["played"], 2))
        else:
            painter.setPen(QPen(self.colours["mid"], 1))
        painter.setBrush(self.colours["bg"])
        painter.drawRoundedRect(rect, 8, 8)
        painter.setRenderHint(QPainter.Antialiasing, False)

        if not self._peaks:
            painter.setPen(self.colours["text"])
            painter.drawText(self.rect(), Qt.AlignCenter, self._message)
            return

        middle = rect.center().y()
        painter.setPen(QPen(self.colours["mid"], 1))
        painter.drawLine(rect.left() + 6, middle, rect.right() - 6, middle)

        usable = rect.width() - 12
        half = rect.height() / 2 - 6
        played_until = rect.left() + 6 + usable * self._progress
        count = len(self._peaks)

        for column in range(int(usable)):
            index = int(column / usable * count)
            low, high = self._peaks[min(index, count - 1)]
            x = rect.left() + 6 + column
            top = middle - high * half
            bottom = middle - low * half
            if bottom - top < 1:
                top, bottom = middle - 0.5, middle + 0.5
            painter.setPen(
                QPen(self.colours["played"] if x <= played_until else self.colours["wave"], 1)
            )
            painter.drawLine(x, top, x, bottom)

        if self._progress > 0:
            painter.setPen(QPen(self.colours["head"], 1))
            painter.drawLine(played_until, rect.top() + 4, played_until, rect.bottom() - 4)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._seek_from(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton:
            self._seek_from(event)

    def _seek_from(self, event: QMouseEvent) -> None:
        if not self._peaks:
            return
        usable = max(1.0, self.width() - 12)
        fraction = (event.position().x() - 6) / usable
        self.seekRequested.emit(max(0.0, min(1.0, fraction)))


class AudioPlayer(QWidget):
    """Waveform, transport and volume for a single file."""

    #: The save icon was pressed.  The owner decides where the copy goes; the
    #: player has no business opening a file dialog of its own.
    saveRequested = Signal()
    #: An audio file dropped on the waveform, once :meth:`accept_drops` is on.
    #: The owner loads it, since what loading means is its decision.
    fileDropped = Signal(str)

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._path = ""
        self._pool = QThreadPool.globalInstance()
        self._signals = _PeakSignals()
        self._signals.done.connect(self._on_peaks)
        self._signals.failed.connect(self._on_peaks_failed)

        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.9)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)

        self.waveform = Waveform()
        self.waveform.seekRequested.connect(self._seek_fraction)
        self.waveform.fileDropped.connect(self.fileDropped)

        # Drawn icons rather than "▶"/"⏸"/"✕"/"🔊" as button text.  Those came
        # from whatever font on the machine happened to carry them -- four
        # different designs at four weights, and the speaker was a colour emoji
        # on Windows -- sitting next to an otherwise hand-drawn icon set.
        self._icon_colour = theme.tokens()["text_dim"]
        self._accent_colour = theme.tokens()["accent"]

        self.play_button = QPushButton()
        self.play_button.setObjectName("PlayButton")
        self.play_button.setFixedSize(32, 32)
        self.play_button.setIconSize(QSize(14, 14))
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.clicked.connect(self.toggle)
        self.play_button.setEnabled(False)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("FieldHint")

        self.name_label = QLabel(title or "—")
        self.name_label.setObjectName("FieldHint")
        self.name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(90)
        self.volume.setFixedWidth(90)
        self.volume.valueChanged.connect(
            lambda value: self._audio_output.setVolume(value / 100)
        )

        # Nothing else empties this player: a result stays loaded across tab
        # switches and until the next conversion replaces it, because coming
        # back to compare two takes is the normal thing to want.  This is the
        # way out when you are done with one.
        self.clear_button = QPushButton()
        self.clear_button.setObjectName("IconButton")
        self.clear_button.setFixedWidth(28)
        self.clear_button.setIconSize(QSize(13, 13))
        self.clear_button.setToolTip(_("Unload this audio."))
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(False)

        # Keeping a copy somewhere else belongs on the player rather than in
        # the card around it: it acts on *this* file, next to the other two
        # things you can do with it once you have heard it.  Hidden unless an
        # owner asks for it -- offering to save a copy of the input the user
        # just picked would be noise.
        self.save_button = QPushButton()
        self.save_button.setObjectName("IconButton")
        self.save_button.setFixedWidth(28)
        self.save_button.setIconSize(QSize(13, 13))
        self.save_button.setToolTip(_("Save a copy…"))
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.clicked.connect(self.saveRequested)
        self.save_button.setEnabled(False)
        self.save_button.hide()

        self.open_button = QPushButton(_("Show in folder"))
        self.open_button.setObjectName("Ghost")
        self.open_button.setIconSize(QSize(14, 14))
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.open_button.clicked.connect(self._reveal)
        self.open_button.setEnabled(False)

        transport = QHBoxLayout()
        transport.setSpacing(8)
        transport.addWidget(self.play_button)
        transport.addWidget(self.time_label)
        transport.addWidget(self.name_label, 1)
        self.volume_icon = QLabel()
        self.volume_icon.setFixedWidth(16)
        transport.addWidget(self.volume_icon)
        transport.addWidget(self.volume)
        transport.addWidget(self.clear_button)
        transport.addWidget(self.save_button)
        transport.addWidget(self.open_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.waveform)
        layout.addLayout(transport)

        self._paint_icons()

    # -- api ---------------------------------------------------------------

    def load(self, path: str) -> None:
        self._path = path or ""
        if not self._path or not os.path.isfile(self._path):
            self.clear()
            return
        self.name_label.setText(os.path.basename(self._path))
        self.play_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.waveform.set_message(_("Reading waveform…"))
        url = QUrl.fromLocalFile(os.path.abspath(self._path))
        if self._player.source() == url:
            # ``setSource`` with the URL it already has stops and returns
            # without reopening the file.  A result rewritten in place -- the
            # same input converted again, every TTS run -- got its new
            # waveform while playback kept the old decode and duration.
            self._player.setSource(QUrl())
        self._player.setSource(url)
        self._pool.start(_PeakJob(self._path, self._signals))

    def release(self, *paths: str) -> None:
        """Unload, if what is loaded is one of ``paths``, before it is rewritten.

        The media backend holds a loaded file open.  On Windows that handle
        lets another process write the file but not delete it, so a conversion
        rewrote the result under a player still decoding it, and the pipeline's
        cleanup of its intermediate WAV failed without a word.  Let go first,
        and :meth:`load` the new file once it exists.
        """
        if self._path and any(path and _same_path(self._path, path) for path in paths):
            self.clear()

    def clear(self) -> None:
        self._path = ""
        self._player.stop()
        self._player.setSource(QUrl())
        self.waveform.set_peaks([])
        self.waveform.set_progress(0.0)
        self.name_label.setText("—")
        self.time_label.setText("0:00 / 0:00")
        self.play_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def enable_saving(self, enabled: bool = True) -> None:
        """Show the save icon.  Off by default; only results are worth keeping."""
        self.save_button.setVisible(bool(enabled))

    def accept_drops(self, suffixes: tuple[str, ...]) -> None:
        """Let the waveform take a dropped audio file.  Off by default: a
        result player has nothing to receive."""
        self.waveform.accept_files(suffixes, _("Drop an audio file here"))

    def path(self) -> str:
        """What is loaded, for an owner that needs to act on the file."""
        return self._path

    def toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def stop(self) -> None:
        self._player.stop()

    def apply_theme(self, tokens: dict[str, str]) -> None:
        self.waveform.apply_theme(tokens)
        self._icon_colour = tokens["text_dim"]
        self._accent_colour = tokens["accent"]
        self._paint_icons()

    def _paint_icons(self) -> None:
        """(Re)render the transport icons; QSS cannot reach a pixmap."""
        self.play_button.setIcon(
            icons.icon(
                "pause" if self._player.playbackState() == QMediaPlayer.PlayingState
                else "play",
                self._accent_colour,
                size=14,
            )
        )
        self.open_button.setIcon(icons.icon("folder", self._icon_colour, size=14))
        self.clear_button.setIcon(icons.icon("close", self._icon_colour, size=13))
        # "download" rather than a floppy: an arrow into a tray is what this
        # means now, and it is the glyph the rest of the set already carries.
        self.save_button.setIcon(icons.icon("download", self._icon_colour, size=13))
        self.volume_icon.setPixmap(
            icons.pixmap("volume", self._icon_colour, 15, 1.7)
        )

    # -- internals ---------------------------------------------------------

    def _on_peaks(self, path: str, peaks: list, _duration: float) -> None:
        if path == self._path:
            self.waveform.set_peaks(peaks)

    def _on_peaks_failed(self, path: str, error: str) -> None:
        if path == self._path:
            # Playback may still work even when we cannot decode for display,
            # so this is a degraded view rather than a failure.
            self.waveform.set_message(_("Preview unavailable ({error})").format(error=error))

    def _on_position(self, position: int) -> None:
        duration = self._player.duration()
        if duration > 0:
            self.waveform.set_progress(position / duration)
        self.time_label.setText(
            f"{self._clock(position)} / {self._clock(duration)}"
        )

    def _on_duration(self, duration: int) -> None:
        self.time_label.setText(f"0:00 / {self._clock(duration)}")

    def _on_state(self, state) -> None:
        self._paint_icons()

    def _seek_fraction(self, fraction: float) -> None:
        duration = self._player.duration()
        if duration > 0:
            self._player.setPosition(int(duration * fraction))

    def _reveal(self) -> None:
        if not self._path:
            return
        folder = os.path.dirname(os.path.abspath(self._path))
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    @staticmethod
    def _clock(milliseconds: int) -> str:
        seconds = max(0, milliseconds) // 1000
        return f"{seconds // 60}:{seconds % 60:02d}"
