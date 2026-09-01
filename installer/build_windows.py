"""Build the two Windows executables.

``ShiroRVC.exe``       the launcher shim, copied into every install
``ShiroRVC-Setup.exe`` the wizard, with the launcher bundled inside it

Run locally with::

    python installer/build_windows.py --tag v1.0.0

The launcher is built first because the setup executable embeds it: that is
what lets a finished install have a double-clickable icon without the
bootstrapper having to fetch a second download.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "installer"
BUILD = ROOT / "build" / "pyinstaller"
DIST = ROOT / "dist"

APP_NAME = "ShiroRVC"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{command[0]} failed with exit code {result.returncode}")


def write_release_info(repo: str, tag: str, source_url: str | None) -> Path:
    target = INSTALLER / "release_info.json"
    payload = {"repo": repo, "tag": tag}
    if source_url:
        payload["source_url"] = source_url
    elif tag:
        payload["source_url"] = (
            f"https://github.com/{repo}/releases/download/{tag}/{APP_NAME}-source-{tag}.zip"
        )
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"release_info.json -> {payload}", flush=True)
    return target


def _trim(image, probe: int = 64):
    """Square bounding box of the non-transparent pixels.

    Deliberately a copy of ``gui.widgets.icons._opaque_bounds`` rather than an
    import of it: ``installer/`` is standalone by design, and importing
    ``gui`` here would break the promise that deleting ``gui/`` leaves
    everything else working.
    """
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QImage

    if not image.hasAlphaChannel():
        return image.rect()

    thumbnail = image.scaled(
        probe, probe, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
    ).convertToFormat(QImage.Format_ARGB32)
    left, top, right, bottom = probe, probe, -1, -1
    for y in range(probe):
        for x in range(probe):
            if (thumbnail.pixel(x, y) >> 24) & 0xFF > 8:
                left, top = min(left, x), min(top, y)
                right, bottom = max(right, x), max(bottom, y)
    if right < 0:
        return image.rect()

    scale_x, scale_y = image.width() / probe, image.height() / probe
    x, y = int(left * scale_x), int(top * scale_y)
    width = max(1, int((right - left + 1) * scale_x))
    height = max(1, int((bottom - top + 1) * scale_y))
    side = min(max(width, height), image.width(), image.height())
    return QRect(
        max(0, min(x - (side - width) // 2, image.width() - side)),
        max(0, min(y - (side - height) // 2, image.height() - side)),
        side, side,
    )


def ensure_icon(explicit: Path) -> Path | None:
    """The .ico for both executables, generated from ``assets/logo.png``.

    A hand-made ``installer/icon.ico`` wins if one is committed. Otherwise the
    brand PNG is converted here. Note Qt's writer stores only one image, so
    this produces a single 256 px rendition that Windows downscales for the
    16 and 32 px slots; commit a real multi-resolution .ico if that matters.
    """
    if explicit.is_file():
        print(f"using {explicit}", flush=True)
        return explicit

    logo = ROOT / "assets" / "logo.png"
    if not logo.is_file():
        print("No assets/logo.png and no installer/icon.ico; using the default icon.")
        return None

    try:
        # Offscreen: the build runs on a CI machine with no display.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication, QImage

        _app = QGuiApplication.instance() or QGuiApplication([])

        image = QImage(str(logo))
        if image.isNull():
            raise ValueError("logo.png could not be read")
        image = image.copy(_trim(image))

        target = BUILD / "icon.ico"
        target.parent.mkdir(parents=True, exist_ok=True)
        scaled = image.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not scaled.save(str(target), "ICO"):
            raise ValueError("Qt declined to write the .ico")
        print(f"generated {target} from {logo.name}", flush=True)
        return target
    except Exception as error:  # noqa: BLE001 - cosmetic, never fail the build
        print(f"Could not generate an icon from logo.png ({error}); using the default.")
        return None


def build_launcher(icon: Path | None) -> Path:
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", APP_NAME,
        "--distpath", str(DIST / "launcher"),
        "--workpath", str(BUILD / "launcher"),
        "--specpath", str(BUILD),
        # The shim only spawns a subprocess; excluding these keeps it at
        # ~10 MB instead of dragging in half of Qt through a stray import.
        "--exclude-module", "PySide6",
        "--exclude-module", "tkinter",
        "--exclude-module", "numpy",
    ]
    if icon and icon.is_file():
        command += ["--icon", str(icon)]
    command.append(str(INSTALLER / "launcher.py"))
    run(command)

    built = DIST / "launcher" / f"{APP_NAME}.exe"
    if not built.is_file():
        raise SystemExit(f"PyInstaller did not produce {built}")
    return built


def build_setup(launcher: Path, icon: Path | None) -> Path:
    separator = ";" if sys.platform == "win32" else ":"
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", f"{APP_NAME}-Setup",
        "--distpath", str(DIST / "setup"),
        "--workpath", str(BUILD / "setup"),
        "--specpath", str(BUILD),
        "--add-data", f"{launcher}{separator}.",
        "--add-data", f"{INSTALLER / 'release_info.json'}{separator}.",
        # PySide6 ships far more than a wizard needs, and PyInstaller bundles
        # every module it can reach.  Dropping these halves the setup binary.
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.QtQuick",
        "--exclude-module", "PySide6.QtQml",
        "--exclude-module", "PySide6.Qt3DCore",
        "--exclude-module", "PySide6.QtCharts",
        "--exclude-module", "PySide6.QtDataVisualization",
        "--exclude-module", "PySide6.QtMultimedia",
        "--exclude-module", "PySide6.QtPdf",
        "--exclude-module", "numpy",
        "--exclude-module", "tkinter",
        # The entry script imports the package absolutely; ROOT has to be on
        # the analysis path for `installer.bootstrap` to resolve.
        "--paths", str(ROOT),
    ]
    if icon and icon.is_file():
        command += ["--icon", str(icon)]
    command.append(str(INSTALLER / "setup_entry.py"))
    run(command)

    built = DIST / "setup" / f"{APP_NAME}-Setup.exe"
    if not built.is_file():
        raise SystemExit(f"PyInstaller did not produce {built}")
    return built


def package(setup_exe: Path, tag: str) -> Path:
    stage = DIST / "package"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    shutil.copy2(setup_exe, stage / setup_exe.name)
    (stage / "README.txt").write_text(
        f"{APP_NAME} {tag or 'development build'}\n"
        f"{'=' * 40}\n\n"
        f"Run {setup_exe.name} and follow the wizard.\n\n"
        "It downloads Python, PyTorch and the application itself, so the\n"
        "install needs a network connection and about 14 GB of free disk.\n"
        "A CUDA build is chosen automatically when an NVIDIA GPU is present.\n\n"
        "Nothing is written outside the folder you choose, and no\n"
        "administrator rights are required.\n",
        encoding="utf-8",
    )

    archive = DIST / f"{APP_NAME}-Setup-win64{('-' + tag) if tag else ''}.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", str(stage))
    print(f"packaged -> {archive}", flush=True)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows setup executable.")
    parser.add_argument("--repo", default="ShiromiyaG/ShiroRVC")
    parser.add_argument("--tag", default="")
    parser.add_argument("--source-url", default=None,
                        help="Overrides the release asset URL the installer downloads.")
    parser.add_argument("--icon", type=Path, default=INSTALLER / "icon.ico")
    arguments = parser.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)
    write_release_info(arguments.repo, arguments.tag, arguments.source_url)

    icon = ensure_icon(arguments.icon)

    launcher = build_launcher(icon)
    setup = build_setup(launcher, icon)
    package(setup, arguments.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
