"""Setup window for ShiroRVC.

A three-page wizard over :mod:`installer.steps`: choose, install, done.  It
carries its own copy of the palette rather than importing ``gui.theme``,
because the installer runs before the application exists on disk.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import config, steps

#: For the one message that has to override the muted Hint colour: a refused
#: install location.  The rest of the palette lives in STYLE below.
DANGER = "#f87171"

STYLE = """
QWidget { background: #101014; color: #e8e8ef; font-size: 13px; }
/* Without this, labels inherit the window colour and read as dark boxes once
 * they sit on a lighter card. */
QLabel, QRadioButton, QCheckBox { background: transparent; }
#Title { font-size: 24px; font-weight: 600; }
#Subtitle { color: #9a9aab; font-size: 13px; }
#Hint { color: #6b6b7d; font-size: 11px; }
#Card { background: #1c1c24; border: 1px solid #2e2e3a; border-radius: 12px; }
#StepLabel { color: #8b7cf6; font-weight: 600; font-size: 12px; letter-spacing: 1px; }
QLineEdit {
    background: #14141a; border: 1px solid #2e2e3a; border-radius: 8px; padding: 8px 10px;
}
QLineEdit:focus { border-color: #8b7cf6; }
QPushButton {
    background: #22222c; border: 1px solid #2e2e3a; border-radius: 8px; padding: 8px 16px;
}
QPushButton:hover { background: #2a2a36; }
QPushButton#Primary {
    background: #8b7cf6; border-color: #8b7cf6; color: #ffffff;
    font-weight: 600; padding: 11px 26px;
}
QPushButton#Primary:hover { background: #9d90f8; }
QPushButton#Primary:disabled { background: #22222c; border-color: #2e2e3a; color: #6b6b7d; }
QPlainTextEdit {
    background: #0c0c10; border: 1px solid #2e2e3a; border-radius: 10px;
    color: #9a9aab; padding: 8px;
}
QProgressBar {
    background: #22222c; border: none; border-radius: 4px; height: 8px; text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #8b7cf6; border-radius: 4px; }
QRadioButton, QCheckBox { spacing: 8px; padding: 4px 0; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #3d3d4d; background: #14141a;
}
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator { border-radius: 5px; }
/* The radius has to be restated: re-declaring the border in the checked state
 * drops it, and a square "radio" is worse than no styling at all. */
QRadioButton::indicator:checked {
    border: 5px solid #8b7cf6; border-radius: 8px; background: #14141a;
}
QCheckBox::indicator:checked { background: #8b7cf6; border-color: #8b7cf6; }
QScrollBar:vertical { background: transparent; width: 9px; }
QScrollBar::handle:vertical { background: #3d3d4d; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


class Worker(QObject):
    """Runs the pipeline off the UI thread."""

    logged = Signal(str)
    progressed = Signal(float)
    progress_cleared = Signal()
    stepped = Signal(int, int, str)
    finished = Signal(bool, str)

    def __init__(self, options: steps.Options):
        super().__init__()
        self.options = options
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        reporter = steps.Reporter(
            log=self.logged.emit,
            progress=lambda fraction: (
                self.progressed.emit(fraction) if fraction is not None
                else self.progress_cleared.emit()
            ),
            step=self.stepped.emit,
            cancelled=lambda: self._cancelled,
        )
        try:
            steps.install(self.options, reporter)
        except steps.InstallError as error:
            self.finished.emit(False, str(error))
        except Exception as error:  # noqa: BLE001 - surfaced, not swallowed
            self.finished.emit(False, f"{type(error).__name__}: {error}")
        else:
            self.finished.emit(True, str(self.options.install_dir))


def _title(text: str, object_name: str = "Title") -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    label.setWordWrap(True)
    return label


class ChoicePage(QWidget):
    """Install location and build variant."""

    #: Emitted with whether the current folder can be installed into.
    targetChecked = Signal(bool)

    def __init__(self, gpu: dict, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(_title(f"Install {config.APP_NAME}"))
        layout.addWidget(_title(
            "This downloads Python, PyTorch and the application. "
            "Expect 6–9 GB and a few minutes on a decent connection.",
            "Subtitle",
        ))

        location = QFrame()
        location.setObjectName("Card")
        location_layout = QVBoxLayout(location)
        location_layout.setContentsMargins(16, 14, 16, 16)
        location_layout.setSpacing(8)
        location_layout.addWidget(_title("Install to", "StepLabel"))

        row = QHBoxLayout()
        row.setSpacing(8)
        self.path_edit = QLineEdit(str(config.default_install_dir()))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        location_layout.addLayout(row)
        location_layout.addWidget(_title(
            "Avoid Program Files and synced folders such as OneDrive: the "
            "application writes models and training data here.",
            "Hint",
        ))
        # Verdict on the chosen folder, updated as it is typed.  Checking only
        # on the Install click would let someone pick their Desktop, read an
        # error a second later, and have no idea which part was wrong.
        self.location_status = _title("", "Hint")
        self.location_status.setWordWrap(True)
        location_layout.addWidget(self.location_status)
        layout.addWidget(location)

        self.path_edit.textChanged.connect(self._revalidate)

        variant = QFrame()
        variant.setObjectName("Card")
        variant_layout = QVBoxLayout(variant)
        variant_layout.setContentsMargins(16, 14, 16, 16)
        variant_layout.setSpacing(6)
        variant_layout.addWidget(_title("Build", "StepLabel"))

        self.cuda_radio = QRadioButton("NVIDIA GPU (CUDA 13.0)")
        self.cpu_radio = QRadioButton("CPU only")
        variant_layout.addWidget(self.cuda_radio)
        variant_layout.addWidget(self.cpu_radio)

        if gpu.get("cuda"):
            device = gpu["devices"][0]
            self.cuda_radio.setChecked(True)
            detail = f"Detected {device['name']}, driver {device['driver']}."
        else:
            self.cpu_radio.setChecked(True)
            self.cuda_radio.setEnabled(False)
            detail = (
                f"No NVIDIA GPU detected ({gpu.get('reason', 'unknown')}). "
                "Training will not be practical on CPU."
            )
        variant_layout.addWidget(_title(detail, "Hint"))
        layout.addWidget(variant)

        self.shortcut_check = QCheckBox("Create a Desktop shortcut")
        self.shortcut_check.setChecked(True)
        self.launch_check = QCheckBox("Launch when the install finishes")
        self.launch_check.setChecked(True)
        layout.addWidget(self.shortcut_check)
        layout.addWidget(self.launch_check)
        layout.addStretch(1)

        # Last: the verdict label has to exist before it can be filled in.
        self._revalidate()

    def _browse(self) -> None:
        # Opened at the *parent* of the current target, since that is what the
        # dialog will hand back and what gets the app name appended again.
        current = Path(self.path_edit.text().strip() or ".")
        chosen = QFileDialog.getExistingDirectory(
            self, "Install location", str(current.parent if current.name else current)
        )
        if chosen:
            # The picker returns the folder to install *into*, not the install
            # folder itself.  The field then shows the full final path, so what
            # the user reads is exactly where the files will go.
            self.path_edit.setText(os.path.normpath(steps.suggest_target(Path(chosen))))

    def _revalidate(self) -> None:
        raw = self.path_edit.text().strip()
        state, explanation = steps.inspect_target(Path(raw))
        usable = state in (
            steps.TargetState.EMPTY, steps.TargetState.EXISTING_INSTALL,
        )
        if state is steps.TargetState.EMPTY:
            # Browse appends the application name, so the field often holds a
            # path the user did not type in full.  Saying it back is what makes
            # that visible rather than surprising.
            verb = "will be created" if not Path(raw).exists() else "is empty"
            explanation = f"Everything goes into {raw} — the folder {verb}."
        self.location_status.setText(explanation)
        # Reusing the Hint style for the good cases and a red for the bad ones,
        # rather than a second widget that is usually invisible.
        self.location_status.setStyleSheet(
            "" if usable else f"color: {DANGER};"
        )
        self.targetChecked.emit(usable)

    def target_is_usable(self) -> bool:
        state, _ = steps.inspect_target(Path(self.path_edit.text().strip()))
        return state in (
            steps.TargetState.EMPTY, steps.TargetState.EXISTING_INSTALL,
        )

    def options(self) -> steps.Options:
        return steps.Options(
            install_dir=Path(self.path_edit.text().strip()),
            variant="cu130" if self.cuda_radio.isChecked() else "cpu",
            create_shortcut=self.shortcut_check.isChecked(),
            launch_when_done=self.launch_check.isChecked(),
        )


class ProgressPage(QWidget):
    """Step counter, progress bar and the running log."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.step_label = _title("STEP 1 OF 7", "StepLabel")
        self.headline = _title("Preparing…")
        layout.addWidget(self.step_label)
        layout.addWidget(self.headline)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        layout.addWidget(self.bar)

        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(font)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log, 1)

    def append(self, line: str) -> None:
        self.log.appendPlainText(line)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class DonePage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self.headline = _title("Done")
        self.detail = _title("", "Subtitle")
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.headline)
        layout.addWidget(self.detail)
        layout.addStretch(1)


class Installer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{config.APP_NAME} Setup")
        self.setMinimumSize(720, 620)
        self.resize(780, 680)

        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._install_dir: Path | None = None
        self._launch_when_done = True

        self.choice_page = ChoicePage(steps.detect_gpu())
        self.progress_page = ProgressPage()
        self.done_page = DonePage()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.choice_page)
        self.stack.addWidget(self.progress_page)
        self.stack.addWidget(self.done_page)

        self.back_button = QPushButton("Cancel")
        self.back_button.clicked.connect(self._on_secondary)
        self.next_button = QPushButton("Install")
        self.next_button.setObjectName("Primary")
        self.next_button.setCursor(Qt.PointingHandCursor)
        self.next_button.clicked.connect(self._on_primary)
        # Refused locations grey the button out rather than failing on click.
        self.choice_page.targetChecked.connect(self._on_target_checked)
        self._on_target_checked(self.choice_page.target_is_usable())

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(18)
        layout.addWidget(self.stack, 1)
        layout.addLayout(buttons)

    def _on_primary(self) -> None:
        page = self.stack.currentIndex()
        if page == 0:
            self._start()
        elif page == 2:
            if self._launch_when_done and self._install_dir:
                self._launch()
            self.close()

    def _on_secondary(self) -> None:
        if self.stack.currentIndex() == 1 and self._worker:
            self._worker.cancel()
            self.back_button.setEnabled(False)
            self.back_button.setText("Cancelling…")
            return
        self.close()

    def _on_target_checked(self, usable: bool) -> None:
        # Only while the choice page is showing: on the progress and done pages
        # this same button means Cancel and Close.
        if self.stack.currentIndex() == 0:
            self.next_button.setEnabled(usable)

    def _start(self) -> None:
        options = self.choice_page.options()
        # Re-checked rather than trusted: the folder can have been filled by
        # something else between the last keystroke and this click, and the
        # button state is a convenience, not the guarantee.
        try:
            steps.check_target(options.install_dir)
        except steps.InstallError as error:
            QMessageBox.warning(self, "Cannot install here", str(error))
            self.choice_page._revalidate()
            return
        self._launch_when_done = options.launch_when_done

        self.stack.setCurrentIndex(1)
        self.next_button.setEnabled(False)
        self.back_button.setText("Cancel")

        self._thread = QThread(self)
        self._worker = Worker(options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.logged.connect(self.progress_page.append)
        self._worker.stepped.connect(self._on_step)
        self._worker.progressed.connect(self._on_progress)
        self._worker.progress_cleared.connect(self._on_progress_cleared)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_step(self, index: int, total: int, title: str) -> None:
        self.progress_page.step_label.setText(f"STEP {index} OF {total}")
        self.progress_page.headline.setText(title)

    def _on_progress(self, fraction: float) -> None:
        self.progress_page.bar.setRange(0, 1000)
        self.progress_page.bar.setValue(int(fraction * 1000))

    def _on_progress_cleared(self) -> None:
        # Back to the indeterminate sweep: uv is working but not reporting a
        # fraction, and a bar frozen at 100% reads as a hang.
        self.progress_page.bar.setRange(0, 0)

    def _on_finished(self, ok: bool, detail: str) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None

        self.stack.setCurrentIndex(2)
        self.next_button.setEnabled(True)
        if ok:
            self._install_dir = Path(detail)
            self.done_page.headline.setText(f"{config.APP_NAME} is installed")
            self.done_page.detail.setText(
                f"Installed to:\n{detail}\n\n"
                f"Start it any time with {config.APP_NAME}.exe in that folder."
            )
            self.next_button.setText("Launch" if self._launch_when_done else "Close")
            self.back_button.hide()
        else:
            self.done_page.headline.setText("The install did not finish")
            self.done_page.detail.setText(
                f"{detail}\n\nThe full log is on the previous screen."
            )
            self.next_button.setText("Close")
            self.back_button.setText("Back to log")
            self.back_button.setEnabled(True)
            self.back_button.clicked.disconnect()
            self.back_button.clicked.connect(lambda: self.stack.setCurrentIndex(1))

    def _launch(self) -> None:
        if self._install_dir is None:
            return
        executable = self._install_dir / f"{config.APP_NAME}.exe"
        try:
            if executable.is_file():
                subprocess.Popen(
                    [str(executable)],
                    cwd=str(self._install_dir),
                    creationflags=steps.NO_WINDOW,
                )
            elif sys.platform == "win32":
                # The batch fallback is a console program; without this the
                # last thing the wizard does is flash a black window.
                subprocess.Popen(
                    [str(self._install_dir / "start-gui.bat")],
                    cwd=str(self._install_dir), shell=True,
                    creationflags=steps.NO_WINDOW,
                )
        except OSError:
            pass  # the user can start it themselves; nothing worth blocking on

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        event.accept()


def main() -> int:
    # ``--self-test`` builds the wizard and exits without showing it: a CI
    # runner has no display, so this is what the packaging smoke test uses to
    # catch a frozen bundle that cannot import its own package or is missing a
    # Qt plugin.
    self_test = "--self-test" in sys.argv
    if self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    application = QApplication([arg for arg in sys.argv if arg != "--self-test"])
    application.setStyle("Fusion")
    application.setApplicationName(f"{config.APP_NAME} Setup")
    application.setStyleSheet(STYLE)

    window = Installer()
    if self_test:
        return 0
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
