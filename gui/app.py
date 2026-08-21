"""Main window: navigation, status, console, and the engine's lifecycle."""

from __future__ import annotations

import sys

from PySide6.QtCore import QByteArray, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QIcon,
    QKeySequence,
    QPaintEvent,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME, __version__, theme
from .services import engine, paths, prefs
from . import native
from .widgets import dock, effects, icons, scrollguard
from .widgets.dock import FloatingDock
from .widgets.navlist import NavList
from .widgets.console import LogConsole
from .widgets.status import StatusBar
from .views.inference import InferencePage
from .views.monitor import MonitorPage
from .views.tools import ToolsPage
from .views.training import TrainingPage
from .views.tts import TtsPage

from . import i18n
from .i18n import _, N_

#: Marked, not translated: this table is built at import time, which is before
#: :func:`run` installs the catalog.  The labels go through ``_()`` where they
#: are used, in :class:`Sidebar`.
NAV = [
    (N_("Inference"), "waveform", N_("Convert audio to a trained voice"), "Ctrl+1"),
    (N_("Text to speech"), "speech", N_("Synthesise a line and convert it"), "Ctrl+2"),
    (N_("Training"), "trend", N_("Prepare data and train a model"), "Ctrl+3"),
    (N_("Diagnostics"), "spark", N_("Metrics, previews and audio from a run"), "Ctrl+4"),
    (N_("Utilities"), "sliders", N_("Inspect, blend, download, analyze"), "Ctrl+5"),
]


class BrandMark(QWidget):
    """The wordmark, with a small painted glyph instead of an image asset."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.glyph = QLabel()
        self.glyph.setFixedSize(30, 30)
        self.glyph.setObjectName("BrandGlyph")
        self.glyph.setAlignment(Qt.AlignCenter)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setObjectName("BrandName")
        self.version = QLabel(f"v{__version__}")
        self.version.setObjectName("BrandVersion")
        text.addWidget(name)
        text.addWidget(self.version)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.glyph)
        layout.addLayout(text)
        layout.addStretch(1)

    def apply_theme(self, tokens: dict[str, str]) -> None:
        # The shiba is drawn rather than scaled from assets/logo.png: the logo
        # is a detailed illustration and turns to mush at 20 px. The raster is
        # used where it has room -- the window and taskbar icon.
        self.glyph.setPixmap(icons.pixmap("shiba", tokens["accent"], 20, 1.8))


class Sidebar(effects.ChromePanel):
    """Brand block, navigation, and the theme switch."""

    DIVIDER_EDGE = "right"

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(216)

        self.brand = BrandMark()

        self.list = NavList()
        self.list.setObjectName("NavList")
        self.list.setFrameShape(QListWidget.NoFrame)
        self.list.setIconSize(icons.default_size())
        self.list.setUniformItemSizes(True)
        for label, _glyph, hint, shortcut in NAV:
            item = QListWidgetItem(_(label))
            item.setToolTip(f"{_(hint)}  ({shortcut})")
            item.setSizeHint(QSize(0, 40))
            self.list.addItem(item)
        self.list.setCurrentRow(0)
        self.list.currentRowChanged.connect(self.currentChanged)

        self.theme_button = QPushButton(_("Light theme"))
        self.theme_button.setObjectName("Ghost")
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.setIconSize(QSize(16, 16))

        self.backdrop_button = QPushButton(_("Backdrop: off"))
        self.backdrop_button.setObjectName("Ghost")
        self.backdrop_button.setCursor(Qt.PointingHandCursor)
        self.backdrop_button.setToolTip(
            _("Blur the desktop behind the window (Windows 11).\n"
            "Cards stay opaque, so nothing you have to read sits on a wallpaper.")
        )

        # A button rather than a combo box: with a handful of languages the
        # cycle is one click instead of two, and it matches the two switches
        # already in this footer.
        self.language_button = QPushButton()
        self.language_button.setObjectName("Ghost")
        self.language_button.setCursor(Qt.PointingHandCursor)
        self.language_button.setToolTip(
            _(
                "Interface language. Remembered between sessions and applied "
                "when the window is rebuilt."
            )
        )

        footer = QVBoxLayout()
        footer.setContentsMargins(10, 8, 10, 12)
        footer.setSpacing(2)
        footer.addWidget(self.language_button)
        footer.addWidget(self.backdrop_button)
        footer.addWidget(self.theme_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.brand)
        layout.addWidget(self.list, 1)
        layout.addLayout(footer)

        # Only once the button has a parent, which is this line above and not
        # ``footer.addWidget``: a layout adopts its widgets when it is itself
        # installed on one.  Showing a widget that still has no parent makes it
        # a top-level window, and it flashes on screen for the frame it takes
        # to be reparented.
        self.backdrop_button.setVisible(native.supports_backdrop())

    def apply_theme(self, tokens: dict[str, str], mode: str) -> None:
        self.apply_chrome(tokens)
        self.brand.apply_theme(tokens)
        self.list.apply_theme(tokens["accent"], 46 if mode == "dark" else 34)
        for row, (_label, glyph, _hint, _shortcut) in enumerate(NAV):
            self.list.item(row).setIcon(
                icons.icon(glyph, tokens["text_faint"], tokens["accent"])
            )
        going_light = mode == "dark"
        self.theme_button.setText(_("Light theme") if going_light else _("Dark theme"))
        self.theme_button.setIcon(
            icons.icon("sun" if going_light else "moon", tokens["text_dim"], size=16)
        )

    def update_backdrop_label(self, backdrop: str) -> None:
        # "mica" and "acrylic" are the DWM's own names for these effects and
        # stay as they are; only the frame around them is translated.
        name = _("off") if backdrop == "none" else backdrop
        self.backdrop_button.setText(_("Backdrop: {name}").format(name=name))

    def update_language_label(self) -> None:
        self.language_button.setText(
            _("Language: {name}").format(name=i18n.language_name(i18n.current_language()))
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setMinimumSize(1120, 720)

        self._mode = prefs.get("theme", "dark")
        self._accent = prefs.get("accent", "violet")
        self._backdrop = prefs.get("backdrop", "none")
        if not native.supports_backdrop():
            self._backdrop = "none"
        # The window fills itself in paintEvent rather than letting Qt erase it
        # from the palette.  That fill is what the backdrop replaces, so owning
        # it is what makes the toggle a repaint instead of either a stylesheet
        # swap (0.3 s of re-polish) or a WA_TranslucentBackground change, which
        # is creation-time and would mean rebuilding the whole window.
        self._window_fill = QColor(theme.tokens(self._mode, self._accent)["bg"])
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self.engine = engine.instance()

        self.sidebar = Sidebar()
        self.sidebar.currentChanged.connect(self._on_nav)
        self.sidebar.theme_button.clicked.connect(self._toggle_theme)
        self.sidebar.backdrop_button.clicked.connect(self._cycle_backdrop)
        self.sidebar.language_button.clicked.connect(self._cycle_language)
        self.sidebar.update_backdrop_label(self._backdrop)
        self.sidebar.update_language_label()

        self.stack = QStackedWidget()
        # Order must match NAV: the sidebar drives the stack by index.
        self.pages = [
            InferencePage(), TtsPage(), TrainingPage(), MonitorPage(), ToolsPage(),
        ]
        for page in self.pages:
            page.busy.connect(self._on_page_busy)
            page.notify.connect(self._on_notify)
            page.log.connect(self._on_page_log)
            self.stack.addWidget(page)

        self.training_page = next(
            page for page in self.pages if isinstance(page, TrainingPage)
        )
        self._progress_live = False
        self._progress_docked = False
        self._card_flight = None
        self.training_page.progressActive.connect(self._on_progress_active)
        # A run keeps going when the user navigates away -- it is a process in
        # the backend, not something this page owns -- so its progress card
        # follows them instead of disappearing with the page.
        self.progress_dock = FloatingDock(self.stack)

        self.console = LogConsole(max_lines=prefs.get("console_max_lines", 5000))

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.stack)
        self.splitter.addWidget(self.console)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([620, 190])

        self.status = StatusBar()
        self.status.restartRequested.connect(self._restart_engine)
        self.status.consoleToggled.connect(self._toggle_console)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.splitter, 1)
        body.addWidget(self.status)

        root = QWidget()
        root.setObjectName("Root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addLayout(body, 1)
        self.setCentralWidget(root)

        self._install_shortcuts()
        self._connect_engine()
        self._apply_theme()
        self._restore_state()

        # Start the backend after the first paint so the window appears
        # immediately -- importing torch behind it takes several seconds and
        # there is no reason to stare at nothing while it happens.
        QTimer.singleShot(0, self._start_engine)

    # -- engine ------------------------------------------------------------

    def _connect_engine(self) -> None:
        self.engine.log.connect(self.console.append)
        self.engine.ready.connect(self._on_engine_ready)
        self.engine.crashed.connect(self._on_engine_crashed)
        self.engine.busy_changed.connect(self._on_engine_busy)

    def _start_engine(self) -> None:
        self.status.set_starting()
        self.console.append_notice(
            _("Starting backend with {executable}").format(
                executable=paths.python_executable()
            )
        )
        self.engine.start()

    def _restart_engine(self) -> None:
        self.console.append_notice(_("Restarting the backend."))
        self.status.set_starting(_("Restarting backend…"))
        self.engine.restart()

    def _on_engine_ready(self) -> None:
        self.status.set_idle(_("Ready"))
        self.console.append_notice(_("Backend ready."))
        self.engine.call(
            "gpu_info",
            {},
            on_result=self._on_gpu_info,
            on_error=lambda error: self.console.append_notice(
                _("GPU query failed: {error}").format(error=error)
            ),
        )
        if prefs.get("warmup_on_launch", True):
            # Pay the torch import now, while the user is still reading the
            # first screen, rather than on their first conversion.
            self.engine.call("warmup", {}, on_error=lambda _error: None)

    def _on_gpu_info(self, data: dict) -> None:
        devices = data.get("devices", [])
        if not data.get("cuda"):
            self.console.append_notice(
                _("CUDA is not available to the backend; everything will run on CPU.")
            )
            return
        names = ", ".join(device["name"] for device in devices)
        self.console.append_notice(f"CUDA {data.get('version', '?')} · {names}")
        for page in self.pages:
            if hasattr(page, "populate_gpus"):
                page.populate_gpus(devices)

    def _on_engine_crashed(self, message: str) -> None:
        self.status.set_error(_("Backend stopped"))
        self.console.append_notice(message)

    def _on_engine_busy(self, busy: bool) -> None:
        if not busy:
            self.status.set_idle(_("Ready"))

    # -- page plumbing -----------------------------------------------------

    def _on_nav(self, index: int) -> None:
        previous = self.stack.currentIndex()
        if previous != index and 0 <= previous < len(self.pages):
            self.pages[previous].on_hidden()
        self.stack.setCurrentIndex(index)
        page = self.pages[index]
        page.on_shown()
        self._sync_progress_dock()

    def _on_progress_active(self, active: bool) -> None:
        self._progress_live = active
        self._sync_progress_dock()

    def _sync_progress_dock(self) -> None:
        """Float the training progress card whenever its page is not showing."""
        away = self.stack.currentWidget() is not self.training_page
        docked = self._progress_live and away
        if docked == self._progress_docked:
            return
        self._progress_docked = docked

        card = self.training_page.progress
        # Snapshot before the move, at the size it has where it is leaving
        # from: the two homes are different widths, and the flight morphs one
        # into the other.
        animate = self.isVisible() and self.training_page.has_progress
        start = self._card_rect(card) if animate else None
        snapshot = card.grab() if start is not None else None

        if docked:
            self.progress_dock.adopt(card)
        else:
            self.progress_dock.release()
            self.training_page.reclaim_progress()

        if start is None:
            return
        # The reparent only queues a layout; the destination rectangle is not
        # real until it runs, and the flight needs somewhere to land.
        layout = card.parentWidget().layout() if card.parentWidget() else None
        if layout is not None:
            layout.activate()
        end = self._card_rect(card)
        if end is None or end == start:
            return
        self._fly_card(card, snapshot, start, end)

    def _card_rect(self, card: QWidget) -> QRect | None:
        """Where a card sits in the page area, or None if it has no size yet."""
        if card.width() <= 0 or card.height() <= 0:
            return None
        return QRect(card.mapTo(self.stack, QPoint(0, 0)), card.size())

    def _fly_card(self, card: QWidget, snapshot, start: QRect, end: QRect) -> None:
        """Move the card between its two homes instead of teleporting it."""
        # Whichever end it just landed in is hidden for the flight: the ghost
        # is the only copy the user should see moving.
        hidden = self.progress_dock if self._progress_docked else card
        hidden.hide()

        def landed() -> None:
            hidden.show()
            if self._progress_docked:
                self.progress_dock.raise_()
            self._card_flight = None

        if self._card_flight is not None:
            self._card_flight.stop()  # its own handler puts its widget back
        self._card_flight = dock.fly(self.stack, snapshot, start, end, on_finished=landed)

    def _on_page_busy(self, active: bool, description: str) -> None:
        if active:
            self.status.set_busy(description)
        elif not self.engine.busy:
            self.status.set_idle(_("Ready"))

    def _on_page_log(self, message: str) -> None:
        self.console.append_notice(message)

    def _on_notify(self, level: str, message: str) -> None:
        self.console.append_notice(message)
        if level == "error":
            self.status.set_error(message.splitlines()[0][:120])
            QMessageBox.warning(self, _("Something went wrong"), message)
        elif level == "success":
            self.status.set_idle(message.splitlines()[0][:120])

    # -- chrome ------------------------------------------------------------

    def _install_shortcuts(self) -> None:
        for index in range(len(NAV)):
            action = QAction(self)
            action.setShortcut(QKeySequence(f"Ctrl+{index + 1}"))
            action.triggered.connect(lambda _checked=False, i=index: self.sidebar.list.setCurrentRow(i))
            self.addAction(action)

        toggle_console = QAction(self)
        toggle_console.setShortcut(QKeySequence("Ctrl+L"))
        toggle_console.triggered.connect(
            lambda: self.status.console_button.setChecked(not self.status.console_button.isChecked())
        )
        self.addAction(toggle_console)

        toggle_theme = QAction(self)
        toggle_theme.setShortcut(QKeySequence("Ctrl+T"))
        toggle_theme.triggered.connect(self._toggle_theme)
        self.addAction(toggle_theme)

    def _toggle_console(self, visible: bool) -> None:
        self.console.setVisible(visible)
        prefs.set("console_visible", visible)

    def _toggle_theme(self) -> None:
        self._mode = "light" if self._mode == "dark" else "dark"
        prefs.set("theme", self._mode)
        # Written now rather than at close: a theme is a deliberate choice, and
        # losing it because the process was killed (or because a training run
        # took the machine down with it) is the kind of small betrayal people
        # remember. Dark stays the first-run default -- see prefs.DEFAULTS.
        prefs.save()
        self._apply_theme()

    def _apply_theme(self) -> None:
        application = QApplication.instance()

        # On the window, not on the application: same widgets, same result,
        # 0.3 s instead of 2.1 s -- see theme.stylesheet.  Re-applying still
        # unpolishes and repolishes every widget, and Qt shows the unstyled
        # state while that runs, so updates stay frozen on the last painted
        # frame and the cursor says it is working rather than hung.
        application.setOverrideCursor(Qt.WaitCursor)
        self.setUpdatesEnabled(False)
        try:
            theme.apply_palette(application, self._mode)
            self.setStyleSheet(theme.stylesheet(self._mode, self._accent))
        finally:
            self.setUpdatesEnabled(True)
            application.restoreOverrideCursor()

        tokens = theme.tokens(self._mode, self._accent)
        self._window_fill = QColor(tokens["bg"])
        self._apply_native_effects()
        self._elevate_cards()
        # Custom-painted children never see QSS, so they are handed the palette
        # directly.  Missing one of these is how a chart ends up black-on-white.
        self.sidebar.apply_theme(tokens, self._mode)
        self.status.apply_theme(tokens)
        for page in self.pages:
            page.apply_theme(tokens)

    def _elevate_cards(self) -> None:
        """Attach or drop card shadows, according to the theme.

        Measured: 25 shadowed cards take a full-window repaint from 7.8 ms to
        24.5 ms -- 3.1x -- because a graphics effect forces its widget through
        an offscreen buffer every time.  On the light theme that buys visible
        elevation and is worth it.  On the dark one a black shadow against a
        near-black background is invisible, so it is pure cost; depth there
        comes from the card's gradient sitting lighter than the window, which
        is free.
        """
        light = self._mode != "dark"
        for card in self.findChildren(QWidget, "Card"):
            has_shadow = card.graphicsEffect() is not None
            if light and not has_shadow:
                effects.elevate(card, alpha=34)
            elif not light and has_shadow:
                card.setGraphicsEffect(None)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Paint the window's own background.

        ``WA_OpaquePaintEvent`` is set, so Qt does not erase first and this is
        the only thing that touches these pixels.  With a backdrop on the fill
        is fully transparent, which is what the compositor draws its blur into;
        ``CompositionMode_Source`` rather than the default blend because
        painting transparent *over* the last frame would leave it untouched.
        """
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(
            event.rect(),
            Qt.transparent if self._backdrop != "none" else self._window_fill,
        )

    def _apply_native_effects(self) -> None:
        """Title bar theme and compositor backdrop, where the platform has them."""
        native.apply_titlebar_theme(self, self._mode == "dark")
        # The frame lets the backdrop through as well, at a fraction of the
        # opacity -- the blur belongs to the whole window, but nav labels and
        # status text still have to be readable over a wallpaper.
        translucent = self._backdrop != "none"
        self.sidebar.set_translucent(translucent)
        self.status.set_translucent(translucent)

        # The order is not symmetric, and the repaint is deliberately
        # synchronous.  Dropping the extended frame while the window is still
        # painted transparent leaves the compositor a frame with nothing to
        # composite, and it fills that with black -- a flash that is obvious on
        # the light theme.  So going out, paint the opaque fill first and tell
        # the DWM after; going in, extend the frame before anything becomes
        # transparent.  Re-applied on theme changes too: the DWM attributes do
        # not survive every window state transition, and the backdrop has a
        # light and a dark rendition that follows the title bar's.
        if translucent:
            native.apply_backdrop(self, self._backdrop)
            self.repaint()
        else:
            self.repaint()
            native.apply_backdrop(self, self._backdrop)

    #: Acrylic first: it is the one that is unmistakably visible.  Mica tints
    #: with the wallpaper and is deliberately subtle -- on a dark desktop it is
    #: almost nothing, which surprises people who expect a blur.
    BACKDROP_ORDER = ["none", "acrylic", "mica"]

    def _cycle_backdrop(self) -> None:
        """Step through the available compositor backdrops.

        No window recreation and no restart: the frame extension in native.py
        works on a live window and the fill it replaces is ours, so this is two
        DWM calls and a repaint rather than the multi-second white flash that
        rebuilding several hundred widgets produced.
        """
        order = self.BACKDROP_ORDER
        self._backdrop = order[(order.index(self._backdrop) + 1) % len(order)]
        prefs.set("backdrop", self._backdrop)
        prefs.save()

        self._apply_native_effects()  # repaints, in the right order
        self.sidebar.update_backdrop_label(self._backdrop)
        self.console.append_notice(
            {
                "none": "Window backdrop off.",
                "acrylic": "Window backdrop: acrylic (blurs what is behind the window).",
                "mica": "Window backdrop: mica (tints with the wallpaper; subtle by design).",
            }[self._backdrop]
        )

    def _cycle_language(self) -> None:
        """Step to the next shipped language and offer to apply it now.

        The choice is written immediately, the way the theme is: it is a
        deliberate decision, and losing it to a crash is a small betrayal.  It
        is stored as an explicit tag rather than as "system", so switching
        Windows to another language later will not silently undo it.
        """
        codes = list(i18n.LANGUAGES)
        current = i18n.current_language()
        following = codes[(codes.index(current) + 1) % len(codes)] if current in codes else codes[0]

        prefs.set("language", following)
        prefs.save()

        # Installed now so the confirmation below is already in the language
        # being asked about -- that is the only preview available before the
        # window is rebuilt.
        i18n.install(following)
        self.sidebar.update_language_label()
        self.console.append_notice(
            _("Language set to {name}.").format(name=i18n.language_name(following))
        )

        # Every widget took its text at construction, so nothing already on
        # screen changes.  Rather than pretend otherwise, offer the restart.
        if any(getattr(page, "_training", False) for page in self.pages):
            self.console.append_notice(
                _("Training is running; the new language applies at the next start.")
            )
            return
        answer = QMessageBox.question(
            self,
            _("Restart to change language"),
            _(
                "The interface is built when the window opens, so the new "
                "language needs a restart.\n\nRestart now?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._restart_application()

    def _restart_application(self) -> None:
        """Relaunch this process, then close.

        Detached, and started *before* closing: a child of a process that is
        already exiting can be reaped with it on some shells, and the user
        would be left with nothing.
        """
        from PySide6.QtCore import QProcess

        prefs.set("window_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        prefs.save()
        QProcess.startDetached(sys.executable, sys.argv)
        self.close()

    # -- state -------------------------------------------------------------

    def _restore_state(self) -> None:
        geometry = prefs.get("window_geometry")
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        else:
            self.resize(1280, 840)

        visible = prefs.get("console_visible", True)
        self.console.setVisible(visible)
        self.status.console_button.setChecked(visible)

        # Always Inference, never the last view used.  Converting is what this
        # application is opened to do; landing on the training form because
        # that is where the last session ended is a step backwards every time.
        self.sidebar.list.setCurrentRow(0)
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        training = any(
            getattr(page, "_training", False) for page in self.pages
        )
        if training:
            answer = QMessageBox.question(
                self,
                "Training is running",
                "A training run is still active. Closing the window stops it.\n\n"
                "Stop training and quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        prefs.set("window_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        prefs.save()
        self.engine.shutdown()
        event.accept()


def _claim_taskbar_identity() -> None:
    """Let Windows group our windows under our own icon.

    Without an explicit AppUserModelID the shell attributes the process to the
    interpreter that launched it, so the taskbar shows the Python icon no
    matter what ``setWindowIcon`` says.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"Shiromiya.{APP_NAME}.{__version__}"
        )
    except Exception:
        pass  # cosmetic only; never worth failing a launch over


def _application_icon() -> QIcon:
    """The window and taskbar icon, from ``assets/logo.png``."""
    if paths.LOGO_PATH.is_file():
        icon = icons.raster_icon(str(paths.LOGO_PATH))
        if not icon.isNull():
            return icon
    # A missing asset should not leave the window wearing the interpreter's
    # icon; the drawn shiba stands in.
    fallback = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        fallback.addPixmap(icons.pixmap("shiba", theme.ACCENTS["violet"], size))
    return fallback


def _install_qt_translations(application: QApplication, language: str) -> None:
    """Translate the widgets Qt owns rather than we do.

    ``QFileDialog``, the standard ``QMessageBox`` buttons and the line-edit
    context menu draw their text from Qt's own catalogs.  Without this the
    interface is in Portuguese right up to the moment someone clicks Browse and
    gets an English file dialog with an English "Cancel".
    """
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    if language == i18n.DEFAULT_LANGUAGE:
        return  # Qt's built-in source language; there is nothing to load
    translations = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for catalog in ("qtbase", "qt"):
        translator = QTranslator(application)
        if translator.load(QLocale(language), catalog, "_", translations):
            application.installTranslator(translator)
            return  # qtbase carries the widget strings; qt is the older bundle


def run() -> int:
    """Create the application and show the window."""
    # Before the QApplication, and long before any view: every widget resolves
    # its text in its constructor, so a catalog installed later reaches nothing
    # that is already on screen.
    language = i18n.install_resolved(
        explicit=i18n.language_from_argv(),
        stored=prefs.get("language"),
    )

    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(__version__)
    QApplication.setOrganizationName("Shiromiya")

    _claim_taskbar_identity()
    application = QApplication(sys.argv)
    application.setWindowIcon(_application_icon())
    application.setStyle("Fusion")  # the only style that themes consistently across platforms
    _install_qt_translations(application, language)
    # Set once and never again: it carries no colours, only palette roles, so
    # a theme change reaches it through the palette.  Everything else is
    # applied to the window instead -- see theme.stylesheet.
    application.setStyleSheet(theme.global_stylesheet())

    # Before any window exists, so no control is ever briefly wheel-editable.
    scrollguard.install(application)

    paths.ensure_dirs()

    if not paths.is_backend_available():
        QMessageBox.critical(
            None,
            APP_NAME,
            f"core.py was not found next to this package.\n\nExpected it at:\n{paths.CORE_SCRIPT}\n\n"
            "The GUI has to sit inside the application folder to reach the backend.",
        )
        return 1

    window = MainWindow()
    window.show()
    return application.exec()
