"""The classic (VITS) RVC pipeline: inference and training as tabs of one page,
as the rectified page holds its own."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QWidget

from .base import Page
from .inference import InferencePage
from .training import TrainingPage

from ..i18n import _, N_


class _Inference(InferencePage):
    title = subtitle = ""


class _Training(TrainingPage):
    title = subtitle = ""


class ClassicRvcPage(Page):
    title = N_("Classic RVC")
    subtitle = N_("Convert with and train VITS-based RVC models.")
    scrollable = False

    #: The visible tab changed; the window re-checks where the progress card goes.
    subpageChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.inference = _Inference()
        self.training = _Training()
        self.subpages = [self.inference, self.training]
        for page in self.subpages:
            page.busy.connect(self.busy)
            page.notify.connect(self.notify)
            page.log.connect(self.log)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.inference, _("Inference"))
        self.tabs.addTab(self.training, _("Training"))
        self._current = 0
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.content.addWidget(self.tabs, 1)

    @property
    def _training(self) -> bool:
        """Read by the window before closing."""
        return self.training._training

    def shows(self, page: QWidget) -> bool:
        """Whether ``page`` is the tab on screen here."""
        return self.tabs.currentWidget() is page

    def _on_tab_changed(self, index: int) -> None:
        if self.isVisible():
            self.subpages[self._current].on_hidden()
            self.subpages[index].on_shown()
        self._current = index
        self.subpageChanged.emit()

    def on_shown(self) -> None:
        self.tabs.currentWidget().on_shown()

    def on_hidden(self) -> None:
        self.tabs.currentWidget().on_hidden()

    def populate_gpus(self, devices: list[dict]) -> None:
        self.training.populate_gpus(devices)
