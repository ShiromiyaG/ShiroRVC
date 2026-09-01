"""The install pipeline, with no Qt anywhere in it.

Keeping the logic free of widgets means the whole install can be exercised from
a terminal (``python -m installer.steps --dry-run``) and that the UI is only
responsible for showing what it reports.

Speed is the design goal.  The previous installer downloads a ~90 MB Miniconda,
runs its silent installer, creates an environment, then bootstraps pip and uv
inside it -- several minutes before the first package is fetched.  This uses uv
for all of it: uv downloads a standalone CPython in seconds and resolves and
installs the torch stack in parallel.  Miniconda buys nothing here, since no
package in this project needs a conda-only build.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

from . import config

LogFn = Callable[[str], None]
ProgressFn = Callable[[float | None], None]
StepFn = Callable[[int, int, str], None]

#: The wizard is a windowed executable with no console of its own, so every
#: console program it starts would otherwise be handed a brand new console
#: window -- a black rectangle flashing over the wizard for each of the dozen
#: commands an install runs.  Output is on a pipe either way.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class InstallError(RuntimeError):
    """The message is shown to the user verbatim."""


@dataclass
class Options:
    install_dir: Path
    variant: str = "cu130"
    create_shortcut: bool = True
    launch_when_done: bool = True
    source_zip: Path | None = None  # local override, used by the test harness


@dataclass
class Reporter:
    """Where progress goes.  Defaults print, so the module is runnable alone."""

    log: LogFn = lambda message: print(message, flush=True)
    progress: ProgressFn = lambda _fraction: None
    step: StepFn = lambda index, total, title: print(f"[{index}/{total}] {title}", flush=True)
    cancelled: Callable[[], bool] = lambda: False


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------


def detect_gpu() -> dict:
    """What ``nvidia-smi`` says, if it is there at all.

    Only the driver matters: the CUDA wheels ship their own cuBLAS, cuDNN and
    cuFFT, so a CUDA Toolkit install is not required.
    """
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"cuda": False, "reason": "nvidia-smi was not found"}
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"cuda": False, "reason": str(error)}

    if output.returncode != 0 or not output.stdout.strip():
        return {"cuda": False, "reason": "nvidia-smi returned no devices"}

    devices = []
    for line in output.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            devices.append({"name": parts[0], "driver": parts[1], "vram_mb": parts[2]})
    return {"cuda": bool(devices), "devices": devices}


#: Files a folder can contain and still count as empty.  All are created by
#: the shell or a sync client rather than by anyone putting something there,
#: and refusing a folder because Explorer left a ``desktop.ini`` in it would
#: read as a bug.
IGNORABLE_ENTRIES = frozenset({
    "desktop.ini", "thumbs.db", ".ds_store", ".localized", "$recycle.bin",
    "system volume information",
})

#: Written by :func:`write_manifest`.  Its presence is what distinguishes "our
#: install, being reinstalled" from "somebody else's folder".
MANIFEST_NAME = "install-manifest.json"


class TargetState(str, Enum):
    """What is already at a chosen install location."""

    #: Does not exist yet, or exists with nothing meaningful in it.
    EMPTY = "empty"
    #: A previous install of this application: reinstalling is the point.
    EXISTING_INSTALL = "existing_install"
    #: Somebody else's files.  Installing here would interleave a Python
    #: distribution and a source tree with them, and no uninstall could ever
    #: separate the two again.
    OCCUPIED = "occupied"
    #: A drive root.  Technically empty-ish sometimes, never the right answer.
    DRIVE_ROOT = "drive_root"
    #: The path names an existing file, or cannot be read.
    UNUSABLE = "unusable"


def inspect_target(path: Path) -> tuple[TargetState, str]:
    """Classify an install location.  Returns ``(state, explanation)``.

    Kept out of the Qt layer so the same rule applies to the wizard and to
    ``python -m installer.steps``, and so it can be tested without a display.
    """
    raw = str(path).strip()
    # ``Path("")`` stringifies to ``"."``, so an empty field arrives here
    # looking like the current directory rather than like nothing.
    if not raw or raw == ".":
        return TargetState.UNUSABLE, "Choose a folder to install into."

    try:
        resolved = Path(os.path.expandvars(raw)).expanduser()
    except (OSError, ValueError):
        return TargetState.UNUSABLE, "That path cannot be read."

    # A relative path would be resolved against wherever the setup executable
    # happens to have been started from -- usually the user's Downloads folder.
    if not resolved.is_absolute():
        return (
            TargetState.UNUSABLE,
            "Enter a full path, such as C:\\ShiroRVC, rather than a relative one.",
        )

    if resolved.exists() and not resolved.is_dir():
        return TargetState.UNUSABLE, f"{resolved} is a file, not a folder."

    # ``C:\`` and ``/``.  The installer writes its source tree straight into
    # the folder it is given, so a root would scatter core.py, env/ and logs/
    # across the drive with no way to undo it.
    if resolved.parent == resolved:
        return (
            TargetState.DRIVE_ROOT,
            "Installing into a drive root would scatter files across the whole "
            "drive. Choose a folder inside it instead.",
        )

    if not resolved.exists():
        return TargetState.EMPTY, ""

    try:
        entries = [
            entry for entry in resolved.iterdir()
            if entry.name.lower() not in IGNORABLE_ENTRIES
        ]
    except OSError as error:
        return TargetState.UNUSABLE, f"That folder cannot be read ({error.strerror})."

    if not entries:
        return TargetState.EMPTY, ""

    if (resolved / MANIFEST_NAME).is_file():
        return (
            TargetState.EXISTING_INSTALL,
            f"An existing {config.APP_NAME} install is here. Continuing will "
            "update it in place; models and logs under logs/ are kept.",
        )

    shown = ", ".join(sorted(entry.name for entry in entries)[:3])
    more = f" and {len(entries) - 3} more" if len(entries) > 3 else ""
    return (
        TargetState.OCCUPIED,
        f"This folder is not empty ({shown}{more}). Choose an empty folder or "
        f"a new one -- the install writes a Python environment and the whole "
        f"application here, and could not later be removed without taking "
        f"whatever else is in it.",
    )


def suggest_target(chosen: Path) -> Path:
    """Where to install given a folder the user browsed to.

    A folder picker returns what the user thinks of as the *parent* -- "put it
    on D:" -- so taking it literally would turn a Downloads folder into a
    refused install; appending the application name avoids that. Idempotent:
    browsing to a folder already named after the application returns it
    unchanged rather than nesting a second copy inside it.
    """
    chosen = Path(os.path.expandvars(str(chosen).strip())).expanduser()
    if chosen.name.lower() == config.APP_NAME.lower():
        return chosen
    return chosen / config.APP_NAME


def check_target(path: Path) -> None:
    """Raise :class:`InstallError` unless ``path`` is safe to install into."""
    state, explanation = inspect_target(path)
    if state in (TargetState.EMPTY, TargetState.EXISTING_INSTALL):
        return
    raise InstallError(explanation)


def check_disk_space(target: Path) -> None:
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # cannot tell; let the install try
    if free < config.REQUIRED_DISK_BYTES:
        raise InstallError(
            f"{target.drive or probe} has {free / 1024**3:.1f} GB free, and the "
            f"install needs about {config.REQUIRED_DISK_BYTES / 1024**3:.0f} GB. "
            "Free some space or choose another drive."
        )


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def download(url: str, destination: Path, reporter: Reporter, label: str = "") -> Path:
    """Fetch a URL to disk, reporting progress when the server declares a length."""
    reporter.log(f"Downloading {label or url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": f"{config.APP_NAME}-installer"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            # Written to a sibling first so an interrupted download can never
            # be mistaken for a complete one on the next run.
            partial = destination.with_suffix(destination.suffix + ".part")
            with open(partial, "wb") as handle:
                while True:
                    if reporter.cancelled():
                        raise InstallError("Cancelled.")
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        reporter.progress(done / total)
            partial.replace(destination)
    except urllib.error.URLError as error:
        raise InstallError(f"Could not download {label or url}: {error.reason}") from error
    reporter.progress(None)
    # ASCII only: the default reporter prints to a console that may be cp1252,
    # where an em dash is a UnicodeEncodeError waiting to happen.
    reporter.log(f"  {destination.name} - {destination.stat().st_size / 1024**2:.1f} MB")
    return destination


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def ensure_uv(tools_dir: Path, reporter: Reporter) -> Path:
    name = "uv.exe" if sys.platform == "win32" else "uv"
    target = tools_dir / name
    if target.exists():
        reporter.log(f"Reusing {target}")
        return target

    archive = download(config.UV_URL, tools_dir / "uv.zip", reporter, "uv (package installer)")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if Path(member).name == name:
                with bundle.open(member) as source, open(target, "wb") as handle:
                    shutil.copyfileobj(source, handle)
                break
        else:
            raise InstallError("The uv archive did not contain uv.exe.")
    archive.unlink(missing_ok=True)
    target.chmod(0o755)
    return target


def fetch_source(install_dir: Path, reporter: Reporter, options: Options) -> None:
    if options.source_zip:
        archive = options.source_zip
        reporter.log(f"Using local source archive {archive}")
    else:
        archive = download(
            config.source_url(),
            install_dir / "_tools" / "source.zip",
            reporter,
            f"{config.APP_NAME} source",
        )

    reporter.log("Unpacking...")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        # A GitHub source archive nests everything under <repo>-<ref>/; a
        # release asset we built ourselves does not.  Strip whichever applies.
        roots = {name.split("/")[0] for name in names if "/" in name}
        strip = len(roots) == 1 and not any("/" not in name for name in names)
        prefix = f"{roots.pop()}/" if strip else ""

        for index, name in enumerate(names):
            if reporter.cancelled():
                raise InstallError("Cancelled.")
            if name.endswith("/"):
                continue
            relative = name[len(prefix):] if prefix else name
            if not relative:
                continue
            destination = install_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, open(destination, "wb") as handle:
                shutil.copyfileobj(source, handle)
            if index % 40 == 0:
                reporter.progress(index / max(1, len(names)))
    reporter.progress(None)

    if not (install_dir / "core.py").is_file():
        raise InstallError(
            "The downloaded archive does not look like the application "
            "(core.py is missing)."
        )
    if not options.source_zip:
        archive.unlink(missing_ok=True)


def create_environment(uv: Path, install_dir: Path, reporter: Reporter) -> Path:
    # Absolute, because run() sets cwd to install_dir: a relative "env" would
    # be resolved against it a second time and land in install_dir/install_dir.
    env_dir = (install_dir / "env").resolve()
    run(
        [str(uv), "venv", "--python", config.PYTHON_VERSION, str(env_dir)],
        cwd=install_dir,
        reporter=reporter,
    )
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        raise InstallError(f"uv created {env_dir} but {python.name} is not in it.")
    return python


def torch_pins(install_dir: Path) -> dict[str, str]:
    """Read the torch pins out of requirements.txt.

    Taking them from the file rather than from a constant here means the
    installer cannot drift from what the application actually requires.
    """
    pins = dict(config.FALLBACK_TORCH_PINS)
    requirements = install_dir / "requirements.txt"
    try:
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" not in line:
                continue
            name, _, version = line.partition("==")
            name = name.strip().lower()
            if name in pins:
                pins[name] = version.split(";")[0].strip()
    except OSError:
        pass  # no requirements.txt to read: the constants stand in
    return pins


def install_torch(uv: Path, python: Path, install_dir: Path, variant: str,
                  reporter: Reporter) -> None:
    index = config.TORCH_INDEXES.get(variant, config.TORCH_INDEXES["cpu"])
    pins = torch_pins(install_dir)
    packages = [f"{name}=={version}" for name, version in pins.items()]
    reporter.log(
        f"Installing {', '.join(packages)} from {index}\n"
        "This is the large download — several GB for the CUDA build."
    )
    run(
        [str(uv), "pip", "install", "--python", str(python), *packages,
         "--index-url", index],
        cwd=install_dir,
        reporter=reporter,
    )


def install_requirements(uv: Path, python: Path, install_dir: Path,
                         reporter: Reporter) -> None:
    arguments = [str(uv), "pip", "install", "--python", str(python),
                 "-r", str(install_dir / "requirements.txt")]

    gui_requirements = install_dir / "gui" / "requirements-gui.txt"
    if gui_requirements.is_file():
        arguments += ["-r", str(gui_requirements)]
    else:
        # The GUI is optional by design; a source tree without it still
        # installs, it just has no Qt front-end to launch.
        reporter.log("No gui/requirements-gui.txt found; skipping the Qt interface.")

    run(arguments, cwd=install_dir, reporter=reporter)


def install_launcher(install_dir: Path, reporter: Reporter) -> Path | None:
    """Place ``ShiroRVC.exe`` next to the application.

    The launcher is bundled inside the setup executable rather than built here,
    so this is a copy when frozen and a no-op when running from source.
    """
    source = config.bundle_dir() / f"{config.APP_NAME}.exe"
    if not source.is_file():
        reporter.log("No launcher executable in this build; use start-gui.bat.")
        return None
    target = install_dir / f"{config.APP_NAME}.exe"
    shutil.copy2(source, target)
    reporter.log(f"Installed {target.name}")
    return target


def create_shortcut(install_dir: Path, reporter: Reporter) -> None:
    """Put a Desktop shortcut in place, via PowerShell's WScript.Shell."""
    if sys.platform != "win32":
        return
    target = install_dir / f"{config.APP_NAME}.exe"
    if not target.is_file():
        return
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    if not desktop.is_dir():
        return
    link = desktop / f"{config.APP_NAME}.lnk"
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut('{link}'); "
        f"$link.TargetPath = '{target}'; "
        f"$link.WorkingDirectory = '{install_dir}'; "
        "$link.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, timeout=30, check=False,
            creationflags=NO_WINDOW,
        )
        reporter.log(f"Created {link}")
    except (OSError, subprocess.SubprocessError) as error:
        reporter.log(f"Could not create the shortcut: {error}")


def write_manifest(install_dir: Path, options: Options) -> None:
    """Record what was installed, so a repair or update knows what it found."""
    manifest = {
        "app": config.APP_NAME,
        "variant": options.variant,
        "python": config.PYTHON_VERSION,
        "release": config.release_info(),
    }
    try:
        with open(install_dir / "install-manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------


def run(command: list[str], cwd: Path, reporter: Reporter) -> None:
    reporter.log("$ " + " ".join(_quote(part) for part in command))

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    # uv paints progress bars that turn into thousands of unreadable lines once
    # the carriage returns are gone.
    environment["NO_COLOR"] = "1"
    environment["UV_NO_PROGRESS"] = "1"

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=NO_WINDOW,
        )
    except OSError as error:
        raise InstallError(f"Could not run {command[0]}: {error}") from error

    assert process.stdout is not None
    for line in process.stdout:
        if reporter.cancelled():
            process.kill()
            raise InstallError("Cancelled.")
        line = line.rstrip()
        if line:
            reporter.log(line)
    code = process.wait()
    if code != 0:
        raise InstallError(
            f"{Path(command[0]).name} failed with exit code {code}. "
            "The log above has the details."
        )


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part else part


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PIPELINE: list[tuple[str, str]] = [
    ("Checking the system", "checking"),
    ("Fetching uv", "uv"),
    ("Downloading the application", "source"),
    ("Creating the Python environment", "env"),
    ("Installing PyTorch", "torch"),
    ("Installing dependencies", "requirements"),
    ("Finishing up", "finish"),
]


def install(options: Options, reporter: Reporter | None = None) -> Path:
    """Run the whole pipeline. Returns the install directory."""
    reporter = reporter or Reporter()
    total = len(PIPELINE)
    install_dir = options.install_dir
    tools = install_dir / "_tools"

    reporter.step(1, total, PIPELINE[0][0])
    # Before mkdir, and before anything is downloaded: the wizard checks this
    # too, but the CLI entry point and any future caller reach here directly.
    check_target(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(install_dir)
    gpu = detect_gpu()
    if options.variant != "cpu" and not gpu.get("cuda"):
        reporter.log(
            f"No NVIDIA GPU visible ({gpu.get('reason', 'unknown')}); "
            "falling back to the CPU build."
        )
        options.variant = "cpu"
    elif gpu.get("cuda"):
        for device in gpu.get("devices", []):
            reporter.log(f"Found {device['name']} (driver {device['driver']})")

    reporter.step(2, total, PIPELINE[1][0])
    uv = ensure_uv(tools, reporter)

    reporter.step(3, total, PIPELINE[2][0])
    fetch_source(install_dir, reporter, options)

    reporter.step(4, total, PIPELINE[3][0])
    python = create_environment(uv, install_dir, reporter)

    reporter.step(5, total, PIPELINE[4][0])
    install_torch(uv, python, install_dir, options.variant, reporter)

    reporter.step(6, total, PIPELINE[5][0])
    install_requirements(uv, python, install_dir, reporter)

    reporter.step(7, total, PIPELINE[6][0])
    install_launcher(install_dir, reporter)
    if options.create_shortcut:
        create_shortcut(install_dir, reporter)
    write_manifest(install_dir, options)
    reporter.log(f"\n{config.APP_NAME} is installed in {install_dir}")
    return install_dir


def main(argv: Iterable[str] | None = None) -> int:
    """Terminal entry point, mostly for testing the pipeline without Qt."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Install {config.APP_NAME}.")
    parser.add_argument("--dir", type=Path, default=config.default_install_dir())
    parser.add_argument("--variant", choices=sorted(config.TORCH_INDEXES), default="cu130")
    parser.add_argument("--source-zip", type=Path, default=None)
    parser.add_argument("--no-shortcut", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="Report what would be installed and exit.")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.probe:
        print(json.dumps({
            "gpu": detect_gpu(),
            "install_dir": str(arguments.dir),
            "source_url": config.source_url(),
            "uv_url": config.UV_URL,
        }, indent=2))
        return 0

    try:
        install(Options(
            install_dir=arguments.dir,
            variant=arguments.variant,
            create_shortcut=not arguments.no_shortcut,
            source_zip=arguments.source_zip,
        ))
    except InstallError as error:
        print(f"\nInstall failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
