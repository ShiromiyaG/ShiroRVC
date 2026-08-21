"""Viewer for the mel preview images a training run writes.

The previews are wide and short -- three panels side by side, 1608x388 at the
default geometry, more if the run raised ``validation_preview_dpi``.  Scaled to
fit a normal window they become an unreadable strip, so this offers both: fit
for scanning across epochs, and 1:1 with scrollbars for actually looking at a
formant.

Loading is deferred to :meth:`show_path` and the pixmap is cached by path, so
stepping back and forth through epochs does not re-decode a 400 KB PNG each
time.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .forms import ghost_button
from ..i18n import _

#: Past this many decoded previews the cache is cleared.  Each is a few
#: megabytes uncompressed; a long run has hundreds of epochs.
CACHE_LIMIT = 24


class PreviewImage(QWidget):
    """One preview image, fit to width or shown at native resolution."""

    #: Emitted with the new zoom state whenever the user toggles it.
    zoomChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._cache: dict[str, QPixmap] = {}
        self._actual_size = False
        self._path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._label.setMinimumHeight(160)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._label)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._scroll)

        # Fit scaling depends on the viewport's size, and the viewport reaches
        # its real size late: this widget lives on a tab that is not the
        # current one when the page is built, so the image is loaded against a
        # placeholder geometry and the resize that fixes it arrives afterwards,
        # on the viewport rather than on this widget.  Watching the viewport
        # directly is what makes the image fill the tab the first time it is
        # opened instead of sitting at a third of its width.
        self._scroll.viewport().installEventFilter(self)

    # -- content -----------------------------------------------------------

    def show_path(self, path: Path | None) -> None:
        self._path = path
        if path is None:
            self._pixmap = None
            self._label.setText(_("No preview image for this epoch."))
            return

        key = str(path)
        pixmap = self._cache.get(key)
        if pixmap is None:
            pixmap = QPixmap(key)
            if pixmap.isNull():
                self._pixmap = None
                self._label.setText(_("Could not read {name}.").format(name=path.name))
                return
            if len(self._cache) >= CACHE_LIMIT:
                self._cache.clear()
            self._cache[key] = pixmap

        self._pixmap = pixmap
        self._rescale()

    def clear(self) -> None:
        self._pixmap = None
        self._path = None
        self._label.clear()
        self._label.setText(_("Select a run to see its validation previews."))

    @property
    def actual_size(self) -> bool:
        return self._actual_size

    @actual_size.setter
    def actual_size(self, value: bool) -> None:
        value = bool(value)
        if value == self._actual_size:
            return
        self._actual_size = value
        # Only the native-size mode needs the label to stop tracking the
        # viewport; fit mode must track it or there is nothing to fit to.
        self._scroll.setWidgetResizable(not value)
        self._rescale()
        self.zoomChanged.emit(value)

    def native_size(self) -> tuple[int, int] | None:
        if self._pixmap is None:
            return None
        return self._pixmap.width(), self._pixmap.height()

    # -- painting ----------------------------------------------------------

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        if self._actual_size:
            self._label.setPixmap(self._pixmap)
            self._label.resize(self._pixmap.size())
            return

        available = self._scroll.viewport().size()
        if available.width() <= 1 or available.height() <= 1:
            return
        self._label.setPixmap(
            self._pixmap.scaled(
                available,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's spelling
        if (
            watched is self._scroll.viewport()
            and event.type() in (QEvent.Resize, QEvent.Show)
            and not self._actual_size
        ):
            self._rescale()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        super().resizeEvent(event)
        if not self._actual_size:
            self._rescale()


class PreviewGallery(QWidget):
    """A preview image with epoch stepping and a zoom toggle."""

    #: The visible epoch changed; carries the index into the epoch list.
    epochChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._count = 0
        self._index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.image = PreviewImage()
        layout.addWidget(self.image, 1)

        # Scrubbing beats stepping for the thing this is actually used for:
        # dragging across fifty epochs to see when a formant appeared, which
        # is fifty clicks otherwise.  Ticks mark the epochs so a long run still
        # shows where the stops are.
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.set_index)
        layout.addWidget(self.slider)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self._previous = ghost_button(_("Previous"))
        self._next = ghost_button(_("Next"))
        self._latest = ghost_button(_("Latest"))
        self._zoom = ghost_button(_("Actual size"))
        self._zoom.setCheckable(True)
        self._caption = QLabel()
        self._caption.setObjectName("previewCaption")

        self._previous.clicked.connect(lambda: self.step(-1))
        self._next.clicked.connect(lambda: self.step(1))
        self._latest.clicked.connect(lambda: self.set_index(self._count - 1))
        self._zoom.toggled.connect(self._set_zoom)

        bar.addWidget(self._previous)
        bar.addWidget(self._next)
        bar.addWidget(self._latest)
        bar.addSpacing(12)
        bar.addWidget(self._caption, 1)
        bar.addWidget(self._zoom)
        layout.addLayout(bar)

        self.set_count(0)

    # -- navigation --------------------------------------------------------

    def set_count(self, count: int) -> None:
        self._count = max(0, int(count))
        self._index = min(self._index, self._count - 1)
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self._count - 1))
        # One tick per epoch up to the point where they merge into a smear.
        self.slider.setTickInterval(1 if self._count <= 40 else 0)
        self.slider.setValue(max(0, self._index))
        self.slider.blockSignals(blocked)
        self._sync_buttons()

    def set_index(self, index: int) -> None:
        if self._count <= 0:
            return
        index = max(0, min(int(index), self._count - 1))
        if index == self._index:
            return
        self._index = index
        # Blocked because this is also reached *from* the slider; letting it
        # signal back would be a loop that only terminates by luck.
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(blocked)
        self._sync_buttons()
        self.epochChanged.emit(index)

    def step(self, delta: int) -> None:
        self.set_index(self._index + delta)

    @property
    def index(self) -> int:
        return self._index

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)

    def _set_zoom(self, enabled: bool) -> None:
        self.image.actual_size = enabled
        size = self.image.native_size()
        self._zoom.setText(
            _("Fit to window") if enabled else _("Actual size")
        )
        if enabled and size:
            self._zoom.setToolTip(_("Native size: {w} x {h}").format(w=size[0], h=size[1]))

    def _sync_buttons(self) -> None:
        has_any = self._count > 0
        self.slider.setEnabled(self._count > 1)
        self._previous.setEnabled(has_any and self._index > 0)
        self._next.setEnabled(has_any and self._index < self._count - 1)
        self._latest.setEnabled(has_any and self._index != self._count - 1)
        self._zoom.setEnabled(has_any)
