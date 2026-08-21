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

import json
import os
import sys
import threading
import traceback

SENTINEL = "\x1e"

# The GUI launches us with the application root as cwd, but core resolves a few
# paths against sys.path[0]; make the root importable regardless.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_write_lock = threading.Lock()
_job_lock = threading.Lock()
_current_job: threading.Thread | None = None


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


def cmd_infer(args):
    message, preview = _core().run_infer_script(**args)
    return {"message": message, "preview": preview}


def cmd_batch_infer(args):
    return {"message": _core().run_batch_infer_script(**args)}


def cmd_tts(args):
    message, output = _core().run_tts_script(**args)
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
    return {"message": _core().run_download_script(**args)}


def cmd_prerequisites(args):
    return {"message": _core().run_prerequisites_script(**args)}


def cmd_analyze(args):
    info, plot = _core().run_audio_analyzer_script(**args)
    return {"info": str(info), "plot": plot}


def cmd_speakers(args):
    """Speaker ids stored in a checkpoint.

    Reads through the same helper the Gradio tab uses so bundles and plain
    ``.pth`` files behave identically.
    """
    from tabs.inference.inference import get_speakers_id

    return {"speakers": list(get_speakers_id(args.get("model"), args.get("sub_model")))}


def cmd_bundle_models(args):
    from tabs.inference.inference import get_bundle_model_names

    return {"names": list(get_bundle_model_names(args.get("model")))}


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
    "gpu_info": cmd_gpu_info,
}

#: Commands answered on the reader thread rather than queued behind the running
#: job.  Stopping a training run is the whole point: it has to be dispatchable
#: precisely while a job is occupying the worker.
CONTROL = {"ping", "stop_train"}


def run_job(job_id: int, cmd: str, args: dict) -> None:
    try:
        result = HANDLERS[cmd](args)
        emit({"id": job_id, "type": "result", "data": result})
    except BaseException as error:  # noqa: BLE001 - reported, never swallowed
        emit(
            {
                "id": job_id,
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        _job_lock.release()


def dispatch(message: dict) -> None:
    global _current_job

    job_id = message.get("id", -1)
    cmd = message.get("cmd", "")
    args = message.get("args") or {}

    if cmd == "shutdown":
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

    if not _job_lock.acquire(blocking=False):
        emit({"id": job_id, "type": "error", "error": "worker is busy"})
        return

    emit({"id": job_id, "type": "started"})
    _current_job = threading.Thread(
        target=run_job, args=(job_id, cmd, args), name=f"job-{job_id}", daemon=True
    )
    _current_job.start()


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

    emit({"type": "ready", "pid": os.getpid()})

    for line in sys.stdin:
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
    os._exit(0)


if __name__ == "__main__":
    main()
