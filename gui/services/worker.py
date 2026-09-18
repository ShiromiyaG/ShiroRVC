"""Backend process for the Qt front-end.

Runs inside the application environment with ``core`` importable and serves one
JSON command per stdin line.  Living out of process is what lets the GUI stay
responsive while torch loads, and lets it survive a CUDA OOM or a segfault
inside a native kernel -- the window reports a dead worker and offers to
restart it instead of disappearing.

It also keeps the model warm: ``core.import_voice_converter`` is ``lru_cache``d,
so consecutive conversions reuse a loaded checkpoint the way the Gradio server
does, which a fresh subprocess per file could not.

Protocol
--------
stdin   one JSON object per line: ``{"id": int, "cmd": str, "args": {...}}``
stdout  lines starting with :data:`SENTINEL` are JSON control messages; every
        other line is log output from core/rvc and is forwarded verbatim.

The sentinel is ASCII RS (0x1e), which never appears in human-readable log
output, so no escaping of the log stream is required in either direction.
"""

from __future__ import annotations

import faulthandler
import json
import os
import queue
import sys
import threading
import time
import traceback

SENTINEL = "\x1e"

# The GUI launches us with the application root as cwd, but core resolves a few
# paths against sys.path[0]; make the root importable regardless.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Filesystem and JSON only, so it costs the worker nothing to share the GUI's
# rule for which files a conversion writes.
from . import catalog  # noqa: E402

_write_lock = threading.Lock()
#: Jobs waiting for the single job thread: ``(id, cmd, args)``.
_jobs: queue.Queue = queue.Queue()
#: Name of the command the job thread is running, for queue and exit notes.
_current_cmd: str | None = None


def note(text: str) -> None:
    """A line of the worker's own narration, written to stderr for the console."""
    with _write_lock:
        sys.stderr.write(f"[backend] {text}\n")
        sys.stderr.flush()


def emit(payload: dict) -> None:
    """Write one control message.  Safe to call from any thread."""
    line = SENTINEL + json.dumps(payload, default=str)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
# Each handler takes the decoded ``args`` dict and returns something JSON
# serialisable.  Import core lazily inside the handlers: the worker announces
# itself as ready long before torch has finished loading, so the GUI can paint
# a live window instead of a spinner.


def _core():
    import core

    return core


def cmd_ping(args):
    return {"pid": os.getpid(), "python": sys.version.split()[0]}


def cmd_warmup(args):
    """Pay the torch import cost up front, while the user is still clicking."""
    _core()
    return {"ready": True}


def _snapshot(paths: list[str]) -> dict[str, tuple[int, int] | None]:
    """``(mtime_ns, size)`` per path, ``None`` where there is no file yet."""
    found: dict[str, tuple[int, int] | None] = {}
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            found[path] = None
        else:
            found[path] = (stat.st_mtime_ns, stat.st_size)
    return found


def _require_written(before: dict, produced: str) -> None:
    """Raise unless ``produced`` was written since ``before`` was taken.

    ``VoiceConverter.convert_audio`` catches every error and prints it, so
    ``core`` reports success for a conversion that wrote nothing.  The window
    then said "converted" and loaded whatever was already on disk -- the
    *previous* take, when the same input is converted twice.
    """
    try:
        stat = os.stat(produced)
    except OSError:
        raise RuntimeError("The conversion failed; the log above says why.") from None
    if before.get(produced) == (stat.st_mtime_ns, stat.st_size):
        raise RuntimeError("The conversion failed; the log above says why.")


def cmd_infer(args):
    before = _snapshot(
        catalog.conversion_outputs(args["output_path"], args.get("export_format", "WAV"))
    )
    message, preview = _core().run_infer_script(**args)
    _require_written(before, preview)
    return {"message": message, "preview": preview}


def cmd_batch_infer(args):
    return {"message": _core().run_batch_infer_script(**args)}


def cmd_tts(args):
    before = _snapshot(
        catalog.conversion_outputs(args["output_rvc_path"], args.get("export_format", "WAV"))
    )
    message, output = _core().run_tts_script(**args)
    _require_written(before, output)
    return {"message": message, "output": output}


def cmd_preprocess(args):
    return {"message": _core().run_preprocess_script(**args)}


def cmd_extract(args):
    return {"message": _core().run_extract_script(**args)}


def cmd_train(args):
    return {"message": _core().run_train_script(**args)}


def cmd_stop_train(args):
    return {"message": _core().stop_train_script()}


def cmd_index(args):
    return {"message": _core().run_index_script(**args)}


def cmd_model_info(args):
    return {"info": _core().run_model_information_script(**args)}


def cmd_blend(args):
    message, blended = _core().run_model_blender_script(**args)
    return {"message": message, "output": str(blended) if blended else ""}


def cmd_download(args):
    # The pipeline rather than core's wrapper: that one returns a failure as a
    # message, which would reach the window as a success.
    from rvc.lib.extras.model_download import model_download_pipeline

    folder = model_download_pipeline(args["model_link"])
    return {"message": f"Model downloaded to {os.path.relpath(folder)}."}


def cmd_prerequisites(args):
    return {"message": _core().run_prerequisites_script(**args)}


def cmd_analyze(args):
    info, plot = _core().run_audio_analyzer_script(**args)
    return {"info": str(info), "plot": plot}


def cmd_speakers(args):
    """Speaker ids stored in a checkpoint.

    The same ``rvc.lib`` helper the Gradio tab calls, so bundles and plain
    ``.pth`` files behave identically -- without importing that tab, which
    would bring Gradio and its folder scans into the worker.
    """
    from rvc.lib.model_bundle import speaker_ids

    model = args.get("model")
    if not model or not os.path.isfile(model):
        return {"speakers": [0]}
    try:
        return {"speakers": speaker_ids(model, args.get("sub_model"))}
    except Exception as error:  # noqa: BLE001 - as the Gradio tab: fall back to one speaker
        note(f"Could not read the model's speaker IDs: {error}")
        return {"speakers": [0]}


def cmd_bundle_models(args):
    from rvc.lib.model_bundle import bundle_model_names, is_model_bundle

    model = args.get("model")
    if not model or not is_model_bundle(model) or not os.path.isfile(model):
        return {"names": []}
    try:
        return {"names": bundle_model_names(model)}
    except Exception as error:  # noqa: BLE001 - as the Gradio tab: no sub-models to offer
        note(f"Could not inspect the model bundle: {error}")
        return {"names": []}


def cmd_bundle_create(args):
    from pathlib import Path

    from rvc.lib import model_bundle

    pth_paths = [Path(path) for path in args.get("pth_paths") or []]
    output = model_bundle.resolve_bundle_path(
        args.get("output_path"), args["logs_dir"], pth_paths
    )
    details = model_bundle.create_model_bundle(
        pth_paths,
        args.get("index_paths") or [],
        output,
        bool(args.get("single_index")),
        int(args.get("compression", 3)),
    )
    return {"message": f"Bundle saved to {output}", "output": str(output), "details": details}


def cmd_bundle_extract(args):
    from pathlib import Path

    from rvc.lib import model_bundle

    bundle = Path(args["bundle"])
    output = Path(args.get("output_dir") or Path(args["logs_dir"]) / f"{bundle.stem}_extracted")
    reports = model_bundle.extract_model_bundle(bundle, output, bool(args.get("overwrite")))
    return {
        "message": f"Extracted to {output}",
        "output": str(output),
        "details": model_bundle.extraction_report(reports),
    }


def cmd_gpu_info(args):
    """Device names and total VRAM, for the picker and the status bar."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_vram": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return {"cuda": True, "devices": devices, "version": torch.version.cuda}


HANDLERS = {
    "ping": cmd_ping,
    "warmup": cmd_warmup,
    "infer": cmd_infer,
    "batch_infer": cmd_batch_infer,
    "tts": cmd_tts,
    "preprocess": cmd_preprocess,
    "extract": cmd_extract,
    "train": cmd_train,
    "index": cmd_index,
    "model_info": cmd_model_info,
    "blend": cmd_blend,
    "download": cmd_download,
    "prerequisites": cmd_prerequisites,
    "analyze": cmd_analyze,
    "speakers": cmd_speakers,
    "bundle_models": cmd_bundle_models,
    "bundle_create": cmd_bundle_create,
    "bundle_extract": cmd_bundle_extract,
    "gpu_info": cmd_gpu_info,
}

#: Commands answered on the reader thread rather than queued behind the running
#: job.  Stopping a training run is the whole point: it has to be dispatchable
#: precisely while a job is occupying the worker.
CONTROL = {"ping", "stop_train"}

#: Commands that get a thread of their own instead of a place in the queue.
#: ``run_train_script`` waits on the trainer process for the whole run, and a
#: conversion clicked during training used to queue behind it -- "Converting…"
#: for hours.  The trainer is a process of its own, so converting beside it is
#: what the web interface has always allowed; the queue is for work that would
#: actually collide.
DETACHED = {"train"}
#: Detached commands running now, so a second one of the same kind is refused
#: rather than started on top of the first.
_detached: set[str] = set()
_detached_lock = threading.Lock()


def run_job(job_id: int, cmd: str, args: dict) -> None:
    started = time.perf_counter()
    with _detached_lock:
        alongside = sorted(_detached - {cmd})
    note(f"{cmd} started" + (
        f" while {', '.join(alongside)} runs; both share the GPU" if alongside else ""
    ))
    emit({"id": job_id, "type": "started"})
    try:
        result = HANDLERS[cmd](args)
        note(f"{cmd} finished in {time.perf_counter() - started:.1f}s")
        emit({"id": job_id, "type": "result", "data": result})
    except BaseException as error:  # noqa: BLE001 - reported, never swallowed
        note(f"{cmd} failed after {time.perf_counter() - started:.1f}s: "
             f"{type(error).__name__}: {error}")
        emit(
            {
                "id": job_id,
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def job_loop() -> None:
    """Run queued jobs one at a time, in the order they arrived.

    Jobs used to be refused while another ran, and the first one after launch
    is ``gpu_info``, which pays the torch import: the GUI's own ``warmup``
    was always turned away, and so was anything the user clicked in those
    first twenty seconds.  Waiting in line is what the user meant.
    """
    global _current_cmd
    while True:
        job_id, cmd, args = _jobs.get()
        _current_cmd = cmd
        try:
            run_job(job_id, cmd, args)
        finally:
            _current_cmd = None


def _run_detached(job_id: int, cmd: str, args: dict) -> None:
    try:
        run_job(job_id, cmd, args)
    finally:
        with _detached_lock:
            _detached.discard(cmd)


def _running() -> list[str]:
    """Every command running now, queued or detached, for exit notes."""
    with _detached_lock:
        detached = sorted(_detached)
    return [cmd for cmd in (_current_cmd, *detached) if cmd]


def _stop_training_before_exit() -> None:
    """Take a trainer this worker started down with it.

    Every exit here is ``os._exit``, which skips ``atexit`` -- and ``core``'s
    ``_stop_training_at_exit`` is an ``atexit`` hook, so it never ran.  On
    Linux the trainer's ``PR_SET_PDEATHSIG`` still ended it; on Windows a child
    outlives its parent, so closing the window left training running on the
    GPU with nothing left to stop it, while the dialog had just said closing
    stops it.

    Only the trainer this process launched: ``stop_train_script`` with none
    tracked goes looking for any trainer by command line, which would include
    one started from the web interface.
    """
    core = sys.modules.get("core")
    process = getattr(core, "training_process", None) if core else None
    if process is None or process.poll() is not None:
        return
    note("stopping the training run before exiting")
    try:
        note(core.stop_train_script())
    except Exception as error:  # noqa: BLE001 - exiting regardless
        note(f"could not stop the training run: {type(error).__name__}: {error}")


def dispatch(message: dict) -> None:
    job_id = message.get("id", -1)
    cmd = message.get("cmd", "")
    args = message.get("args") or {}

    if cmd == "shutdown":
        running = _running()
        note("shutdown requested by the GUI"
             + (f" while {', '.join(running)} was running" if running else ""))
        _stop_training_before_exit()
        emit({"id": job_id, "type": "result", "data": {"bye": True}})
        os._exit(0)

    if cmd not in HANDLERS:
        emit({"id": job_id, "type": "error", "error": f"unknown command: {cmd!r}"})
        return

    if cmd in CONTROL:
        try:
            emit({"id": job_id, "type": "result", "data": HANDLERS[cmd](args)})
        except BaseException as error:  # noqa: BLE001
            emit({"id": job_id, "type": "error", "error": f"{type(error).__name__}: {error}"})
        return

    if cmd in DETACHED:
        with _detached_lock:
            already = cmd in _detached
            _detached.add(cmd)
        if already:
            emit({"id": job_id, "type": "error", "error": f"{cmd} is already running"})
            return
        threading.Thread(
            target=_run_detached, args=(job_id, cmd, args), name=cmd, daemon=True
        ).start()
        return

    busy_with = _current_cmd
    waiting = _jobs.qsize()
    _jobs.put((job_id, cmd, args))
    if busy_with:
        ahead = f" and {waiting} more" if waiting else ""
        note(f"{cmd} queued behind {busy_with}{ahead}")


def _take_command_pipe():
    """Move the GUI's command pipe off stdin, where children cannot inherit it.

    Every child the backend starts -- the trainer, preprocess, extraction,
    ffmpeg inside a library -- inherits stdin unless told otherwise, and on
    Windows a child holding this pipe was followed by the worker reading EOF
    on it and exiting with code 0 in the middle of the job ("command pipe
    closed ... abandoning train").  Passing ``stdin=DEVNULL`` at each call site
    only covers the call sites we know about.

    ``os.dup`` returns a non-inheritable descriptor (PEP 446), so the commands
    are read from a private copy, and descriptor 0 -- together with the Win32
    standard input handle that CreateProcess hands to children -- is pointed
    at the null device.
    """
    private = os.dup(sys.stdin.fileno())
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":
        try:
            import ctypes
            import msvcrt

            STD_INPUT_HANDLE = -10
            ctypes.windll.kernel32.SetStdHandle(STD_INPUT_HANDLE, msvcrt.get_osfhandle(0))
        except (OSError, AttributeError):
            pass
    sys.stdin = open(os.devnull, encoding="utf-8")
    return open(private, encoding="utf-8", errors="replace", closefd=True)


def main() -> None:
    # Children inherit these; without them a crash inside a spawned trainer can
    # strand a full pipe buffer of output that the GUI never gets to show.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except AttributeError:
        pass

    # A segfault or abort inside a native kernel otherwise ends the process
    # with nothing but an exit code; this prints every thread's Python stack.
    faulthandler.enable(file=sys.stderr, all_threads=True)
    threading.excepthook = lambda hook: note(
        "uncaught error in thread {}:\n{}".format(
            hook.thread.name if hook.thread else "?",
            "".join(traceback.format_exception(hook.exc_type, hook.exc_value, hook.exc_traceback)),
        )
    )

    commands = _take_command_pipe()

    threading.Thread(target=job_loop, name="jobs", daemon=True).start()
    emit({"type": "ready", "pid": os.getpid()})

    for line in commands:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            emit({"type": "error", "error": f"malformed command: {line[:200]!r}"})
            continue
        dispatch(message)

    # stdin closed: the GUI is gone.  Do not linger holding the GPU.
    running = _running()
    note("command pipe closed, exiting"
         + (f" (abandoning {', '.join(running)})" if running else ""))
    _stop_training_before_exit()
    os._exit(0)


if __name__ == "__main__":
    main()
