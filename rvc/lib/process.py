"""Running the pipeline scripts as child processes, and stopping the trainer."""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import time

import psutil

from rvc.lib.terminal import info, warning

# The trainer defers a stop until any checkpoint write in flight has reached
# the disk, so this grace has to comfortably outlast one save (a couple of
# hundred MB) before the force-kill takes over.
TRAINING_STOP_GRACE_SECONDS = 45
TRAINING_SCRIPT_MARKER = "rvc/train/train.py"

_NTSTATUS = {
    0xC0000005: "access violation - a native crash, usually a driver or a CUDA kernel",
    0xC000001D: "illegal instruction",
    0xC0000094: "integer division by zero",
    0xC00000FD: "stack overflow",
    0xC0000135: "a required DLL was not found",
    0xC0000139: "entry point not found in a DLL",
    0xC000013A: "interrupted with Ctrl+C",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun / fatal abort",
    0xC0000417: "invalid parameter passed to the C runtime",
    0xE06D7363: "unhandled C++ exception",
    0x40010004: "the process was killed",
}


def describe_exit_code(code: int) -> str:
    """``code 1``, or the NTSTATUS name for a native crash on Windows."""
    unsigned = code & 0xFFFFFFFF
    name = _NTSTATUS.get(unsigned)
    if name:
        return f"code 0x{unsigned:08X} ({name})"
    if code < 0 and os.name != "nt":
        try:
            return f"signal {signal.Signals(-code).name}"
        except ValueError:
            pass
    return f"code {code}"


def run_stage(command, stage: str) -> None:
    """Run one pipeline script and raise if it fails.

    ``stdin`` is detached because the Qt backend reads its commands from its
    own stdin, and a child sharing that pipe has no business with it.
    """
    result = subprocess.run(command, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed: {os.path.basename(command[1])} exited with "
            f"{describe_exit_code(result.returncode)}. The traceback is in the log above."
        )


def _trainer_preexec():
    """Child setup on POSIX: own session, but tied to this interface's lifetime.

    `setsid` keeps the trainer off the terminal's SIGHUP path, which is what a
    long run wants.  On its own, though, that also means closing the terminal
    kills the interface and leaves the trainer orphaned on the GPU with no way
    left to reach it.  PR_SET_PDEATHSIG asks the kernel to SIGTERM the child
    when its parent goes away, which closes that hole -- including when the
    interface is killed outright and no handler of ours could run.
    """
    os.setsid()
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGTERM
        )
    except Exception:
        # No glibc prctl (macOS, musl): the atexit handler is the fallback.
        pass


def spawn_trainer(command) -> subprocess.Popen:
    """Start the trainer in its own process group, so it can be stopped gracefully."""
    if platform.system() == "Windows":
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        preexec_fn=_trainer_preexec,
    )


def find_trainer_processes():
    """Every live process running the training script, whoever started it.

    Used instead of matching on the `python.exe` image name, which also matched
    this interface and TensorBoard.
    """
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            cmdline = process.info["cmdline"] or []
        except psutil.Error:
            continue
        # Match the interpreter actually running the script, not any process
        # whose command line merely mentions the path -- a grep, an editor or a
        # shell would otherwise qualify.  The launcher always builds the command
        # as [python, train_script_path, ...], so the script is argv[1].
        if len(cmdline) < 2:
            continue
        executable = os.path.basename(str(cmdline[0])).lower()
        if not executable.startswith("python"):
            continue
        if str(cmdline[1]).replace("\\", "/").endswith(TRAINING_SCRIPT_MARKER):
            found.append(process)
    return found


def _request_graceful_stop(process):
    """Ask the trainer to unwind rather than shooting it.

    The launcher puts the trainer in its own process group precisely so this is
    possible.  Both signals reach the DataLoader workers as well, and give
    Python the chance to run its `finally` blocks and flush the TensorBoard
    writer on the way out.
    """
    if platform.system() == "Windows":
        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def _kill_process_tree(pid):
    """Last resort: kill the process and every descendant. Returns survivors."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []

    # Snapshot the children before killing the parent, otherwise the reparented
    # DataLoader workers become unreachable through it.
    try:
        victims = parent.children(recursive=True) + [parent]
    except psutil.Error:
        victims = [parent]

    for victim in victims:
        try:
            victim.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(victims, timeout=5)
    return alive


def _wait_for_exit(process, timeout):
    """Poll rather than `wait()`, which the progress watcher may already hold."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.5)
    return process.poll() is not None


def stop_trainer(process: subprocess.Popen | None) -> str:
    """Stop ``process``, or any orphaned trainer when it is not running.

    Returns the message for the interface.
    """
    if process is None or process.poll() is not None:
        # Nothing tracked -- but a run from a previous session of this interface
        # may still be alive, so look for it by command line.
        orphans = find_trainer_processes()
        if not orphans:
            return "No training process is running."
        for orphan in orphans:
            try:
                orphan.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(orphans, timeout=5)
        if alive:
            return f"Could not stop PID(s): {', '.join(str(p.pid) for p in alive)}."
        return f"Stopped {len(orphans)} orphaned training process(es)."

    pid = process.pid
    try:
        _request_graceful_stop(process)
    except (OSError, psutil.Error) as error:
        warning(f"Graceful stop failed ({error}); killing instead.", tag="[TRAINING]")
    else:
        info(f"Asked PID {pid} to stop, waiting up to {TRAINING_STOP_GRACE_SECONDS}s...", tag="[TRAINING]")
        if _wait_for_exit(process, TRAINING_STOP_GRACE_SECONDS):
            return f"Training stopped (PID {pid})."
        warning(f"PID {pid} did not exit in time; killing the tree.", tag="[TRAINING]")

    alive = _kill_process_tree(pid)
    if alive:
        return f"Could not stop PID(s): {', '.join(str(p.pid) for p in alive)}."
    return f"Training force-stopped (PID {pid})."
