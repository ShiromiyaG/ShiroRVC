import click
import psutil
import os
import sys

import atexit
import platform
import signal
import subprocess
import time


from functools import lru_cache


now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import (
    DEFAULT_CPU_THREADS,
    info,
    install_rich_print,
    print_error_panel,
    print_settings_panel,
    success,
    warning,
)

install_rich_print()

current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

from rvc.lib.extras.prerequisites_download import (
    download_vocoder_pretraineds,
    prequisites_download_pipeline,
)
from rvc.configs.vocoders import (
    get_vocoder_sample_rates,
    normalize_vocoder,
)
from rvc.train.run_spec import TrainRunSpec
from rvc.cli_options import (
    AUDIO_ANALYZER_OWN,
    BATCH_INFER_DEFAULTS,
    BATCH_INFER_OWN,
    DOWNLOAD_OWN,
    EXTRACT_OWN,
    FORMANT_OPTIONS,
    INDEX_OWN,
    INFER_DEFAULTS,
    INFER_OWN,
    MODEL_BLENDER_OWN,
    MODEL_INFORMATION_OWN,
    PREPROCESS_OWN,
    PREREQUISITES_OWN,
    STYLE_EXTRACT_OWN,
    STYLE_OPTIONS,
    STYLE_TRAIN_OWN,
    TRAIN_OWN,
    TTS_DEFAULTS,
    TTS_OWN,
    apply_options,
    inference_options,
    load_voices_data,
)
from rvc.train.messages import (
    TORCH_COMPILE_MODE_CLI_HELP,
    TORCH_COMPILE_MODES,
    VOCODER_COMPILE_CLI_HELP,
)

python = sys.executable
training_process = None

voices_data = load_voices_data()
locales = list({voice["ShortName"] for voice in voices_data})


@lru_cache(maxsize=None)
def import_voice_converter():
    from rvc.infer.infer import VoiceConverter

    return VoiceConverter()


@lru_cache(maxsize=1)
def get_config():
    from rvc.configs.config import Config

    return Config()


# Pipeline scripts
def _run_stage(command, stage: str) -> None:
    """Run one pipeline script and fail loudly when it fails.

    These used to be bare ``subprocess.run`` calls, so a script that crashed
    still reported "... successfully" and the only trace was whatever it had
    printed.  ``stdin`` is detached because the Qt backend reads its commands
    from its own stdin, and a child sharing that pipe has no business with it.
    """
    result = subprocess.run(command, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed: {os.path.basename(command[1])} exited with "
            f"{describe_exit_code(result.returncode)}. The traceback is in the log above."
        )


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


# Infer
def run_infer_script(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    sid: int = 0,
    seed: int = 0,
    bundle_submodel: str = None,
    index_k: int = 8,
    index_power: float = 2.0,
    index_continuity: float = 0.5,
    silence_gate_db: float = -60.0,
    *,
    noise_scale: float = None,
    style: dict = None,
):
    kwargs = {
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": pth_path,
        "index_path": index_path,
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "volume_envelope": volume_envelope,
        "silence_gate_db": silence_gate_db,
        "protect": protect,
        "f0_method": f0_method,
        "pth_path": pth_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "f0_file": f0_file,
        "embedder_model": embedder_model,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "sid": sid,
        "seed": seed,
        "bundle_submodel": bundle_submodel,
        "index_k": index_k,
        "index_power": index_power,
        "index_continuity": index_continuity,
        "noise_scale": noise_scale,
        "style": style,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(
        **kwargs,
    )

    export_path = output_path.replace(".wav", f".{export_format.lower()}")
    if not os.path.exists(export_path):
        export_path = output_path

    return f"File {input_path} inferred successfully. Saved to: {export_path}", export_path


# Batch infer
def run_batch_infer_script(
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    sid: int = 0,
    seed: int = 0,
    index_k: int = 8,
    index_power: float = 2.0,
    index_continuity: float = 0.5,
    silence_gate_db: float = -60.0,
    *,
    noise_scale: float = None,
    style: dict = None,
):
    kwargs = {
        "audio_input_paths": input_folder,
        "audio_output_path": output_folder,
        "model_path": pth_path,
        "index_path": index_path,
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "volume_envelope": volume_envelope,
        "silence_gate_db": silence_gate_db,
        "protect": protect,
        "f0_method": f0_method,
        "pth_path": pth_path,
        "index_path": index_path,
        "split_audio": split_audio,
        "f0_autotune": f0_autotune,
        "f0_autotune_strength": f0_autotune_strength,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "f0_file": f0_file,
        "embedder_model": embedder_model,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "sid": sid,
        "seed": seed,
        "index_k": index_k,
        "index_power": index_power,
        "index_continuity": index_continuity,
        "noise_scale": noise_scale,
        "style": style,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio_batch(
        **kwargs,
    )

    return f"Files from {input_folder} inferred successfully."


# TTS
def run_tts_script(
    tts_file: str,
    tts_text: str,
    tts_voice: str,
    tts_rate: int,
    pitch: int,
    filter_radius: int,
    index_rate: float,
    volume_envelope: int,
    protect: float,
    f0_method: str,
    output_tts_path: str,
    output_rvc_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    f0_autotune: bool,
    f0_autotune_strength: float,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    f0_file: str,
    embedder_model: str,
    sid: int = 0,
    seed: int = 0,
    index_k: int = 8,
    index_power: float = 2.0,
    index_continuity: float = 0.5,
    silence_gate_db: float = -60.0,
    *,
    noise_scale: float = None,
):

    tts_script_path = os.path.join("rvc", "lib", "extras", "tts.py")

    if os.path.exists(output_tts_path) and os.path.abspath(output_tts_path).startswith(os.path.abspath("assets")):
        os.remove(output_tts_path)

    command_tts = [
        *map(
            str,
            [
                python,
                tts_script_path,
                tts_file,
                tts_text,
                tts_voice,
                tts_rate,
                output_tts_path,
            ],
        ),
    ]
    _run_stage(command_tts, "Text to speech")
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(
        pitch=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        silence_gate_db=silence_gate_db,
        protect=protect,
        f0_method=f0_method,
        audio_input_path=output_tts_path,
        audio_output_path=output_rvc_path,
        model_path=pth_path,
        index_path=index_path,
        split_audio=split_audio,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        f0_file=f0_file,
        embedder_model=embedder_model,
        sid=sid,
        seed=seed,
        formant_shifting=None,
        formant_qfrency=None,
        formant_timbre=None,
        index_k=index_k,
        index_power=index_power,
        index_continuity=index_continuity,
        noise_scale=noise_scale,
    )

    return f"Text {tts_text} synthesized successfully.", output_rvc_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Preprocess
def run_preprocess_script(
    model_name: str,
    dataset_path: str,
    sample_rate: int,
    cpu_threads: int,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    clean_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str = "pre_peak_rvc",
    loading_resampling: str = "ffmpeg",
    dataset_format: str = "WAV",
    rms_norm_db: float = -16.0
):
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(logs_path, model_name),
                dataset_path,
                sample_rate,
                cpu_threads,
                cut_preprocess,
                process_effects,
                noise_reduction,
                clean_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
                loading_resampling,
                dataset_format,
                rms_norm_db,
            ],
        ),
    ]
    _run_stage(command, "Preprocessing")
    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(
    model_name: str,
    f0_method: str,
    cpu_threads: int,
    gpu: int,
    sample_rate: int,
    vocoder_arch: str,
    embedder_model: str,
    include_mutes: int = 5,
    feature_precision: str = "fp32",
):
    vocoder_arch = normalize_vocoder(vocoder_arch)
    if int(sample_rate) not in get_vocoder_sample_rates(vocoder_arch):
        raise ValueError(
            f"{vocoder_arch} does not provide a configuration for {sample_rate} Hz."
        )

    model_path = os.path.join(logs_path, model_name)
    extract = os.path.join("rvc", "train", "extract", "extract.py")

    command_1 = [
        python,
        extract,
        *map(
            str,
            [
                model_path,
                f0_method,
                cpu_threads,
                gpu,
                sample_rate,
                vocoder_arch,
                embedder_model,
                include_mutes,
                feature_precision,
            ],
        ),
    ]

    _run_stage(command_1, "Extraction")

    return f"Model {model_name} extracted successfully."


# Train
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


def run_train_script(
    model_name: str,
    epoch_save_frequency: int,
    save_only_latest_net_models: bool,
    save_weight_models: bool,
    total_epoch_count: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    use_warmup: bool,
    warmup_duration: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "hifi",
    use_checkpointing: bool = False,
    precision: str = "fp32",
    compile_vocoder: bool = False,
    torch_compile_mode: str = "default",
    overtrain_detector: bool = False,
    stop_on_overtrain: bool = False,
):
    global training_process

    vocoder = normalize_vocoder(vocoder)
    if int(sample_rate) not in get_vocoder_sample_rates(vocoder):
        raise ValueError(
            f"{vocoder} does not provide a configuration for {sample_rate} Hz."
        )
    if pretrained == True:
        from rvc.lib.extras.pretrained_selector import pretrained_selector

        if custom_pretrained == False:
            # A link added after the app's startup download would otherwise be
            # missed, and a missing file silently means training from scratch.
            download_vocoder_pretraineds(vocoder, int(sample_rate))
            pg, pd = pretrained_selector(str(vocoder), int(sample_rate))
        else:
            pg = g_pretrained_path if g_pretrained_path is not None else ""
            pd = d_pretrained_path if d_pretrained_path is not None else ""
    else:
        pg, pd = "", ""

    # The trainer derives the training phase from the pretrained paths, so it
    # is not passed: two fields that must agree are one field too many.
    spec = TrainRunSpec(
        model_name=model_name,
        sample_rate=int(sample_rate),
        vocoder=str(vocoder),
        total_epoch_count=int(total_epoch_count),
        epoch_save_frequency=int(epoch_save_frequency),
        batch_size=int(batch_size),
        gpus=str(gpu),
        save_only_latest_net_models=bool(save_only_latest_net_models),
        save_weight_models=bool(save_weight_models),
        cleanup=bool(cleanup),
        pretrain_g=str(pg),
        pretrain_d=str(pd),
        use_warmup=bool(use_warmup),
        warmup_duration=int(warmup_duration),
        use_checkpointing=bool(use_checkpointing),
        precision=str(precision).lower(),
        compile_vocoder=bool(compile_vocoder),
        torch_compile_mode=str(torch_compile_mode),
        overtrain_detector=bool(overtrain_detector),
        stop_on_overtrain=bool(stop_on_overtrain),
    )
    # Written into the run's own log directory, so it survives the process and
    # answers "what was this trained with?" long after the fact.
    spec_path = spec.save(
        os.path.join(now_dir, "logs", model_name, "run_spec.json")
    )

    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [python, train_script_path, str(spec_path)]
    if platform.system() == "Windows":
        global training_process
        training_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        training_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            preexec_fn=_trainer_preexec
        )

    training_process.wait()
    return f"Training has been successfully completed or stopped."


# Stopping the training
# The trainer defers a stop until any checkpoint write in flight has reached
# the disk, so this grace has to comfortably outlast one save (a couple of
# hundred MB) before the force-kill takes over.
TRAINING_STOP_GRACE_SECONDS = 45
TRAINING_SCRIPT_MARKER = "rvc/train/train.py"


def _find_trainer_processes():
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


def stop_train_script():
    global training_process

    if training_process is None or training_process.poll() is not None:
        # Nothing tracked -- but a run from a previous session of this interface
        # may still be alive, so look for it by command line.
        orphans = _find_trainer_processes()
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

    pid = training_process.pid
    try:
        _request_graceful_stop(training_process)
    except (OSError, psutil.Error) as error:
        warning(f"Graceful stop failed ({error}); killing instead.", tag="[TRAINING]")
    else:
        info(f"Asked PID {pid} to stop, waiting up to {TRAINING_STOP_GRACE_SECONDS}s...", tag="[TRAINING]")
        if _wait_for_exit(training_process, TRAINING_STOP_GRACE_SECONDS):
            return f"Training stopped (PID {pid})."
        warning(f"PID {pid} did not exit in time; killing the tree.", tag="[TRAINING]")

    alive = _kill_process_tree(pid)
    if alive:
        return f"Could not stop PID(s): {', '.join(str(p.pid) for p in alive)}."
    return f"Training force-stopped (PID {pid})."


def _stop_training_at_exit():
    """Take the trainer down when this interface exits normally."""
    if training_process is not None and training_process.poll() is None:
        info("Interface is exiting; stopping the training run.", tag="[TRAINING]")
        stop_train_script()


atexit.register(_stop_training_at_exit)


# Index
def list_experiment_speakers(model_name: str):
    """Speaker ids with extracted features in ``logs/<model_name>/``.

    Imported lazily: the module reaches faiss and scikit-learn at import time,
    and this is called to populate a dropdown.
    """
    from rvc.train.process.extract_index import available_speakers

    return available_speakers(os.path.join(logs_path, model_name))


def run_index_script(
    model_name: str, index_algorithm: str, index_metric: str = "l2",
    index_speaker: int | str | None = None,
):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    # "all" travels as a word rather than as an empty argument, which a shell
    # can drop and which would otherwise be read back as speaker 0.
    speaker = "all" if index_speaker in (None, "", "all") else str(int(index_speaker))
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        index_algorithm,
        index_metric,
        speaker,
    ]

    # Checked, because it has not always succeeded: the script's ``rvc.``
    # import failed under the subprocess's sys.path and this reported success
    # regardless, so a missing index looked like a working one.
    result = subprocess.run(command, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        return (
            f"Index generation for {model_name} failed "
            f"({describe_exit_code(result.returncode)}). "
            "See the terminal for the traceback."
        )
    return f"Index file for {model_name} generated successfully."


# Style model
style_models_path = os.path.join(current_script_directory, "rvc", "models", "style")
_style_process = None


def list_style_models(kind: str = "all") -> list[str]:
    """Style checkpoints: bases (``rvc/models/style/*.pt``, ``logs/*/style_base.pt``)
    and fine-tuned singers (``logs/*/*_style.pt``).  ``kind`` is "base",
    "singer" or "all"."""
    import glob

    bases = glob.glob(os.path.join(style_models_path, "*.pt")) + glob.glob(
        os.path.join(logs_path, "*", "style_base.pt")
    )
    singers = glob.glob(os.path.join(logs_path, "*", "*_style.pt"))
    chosen = {"base": bases, "singer": singers}.get(kind, bases + singers)
    return sorted(os.path.relpath(p, current_script_directory) for p in chosen)


def style_data_summary(model_name: str) -> str | None:
    """One line on ``logs/<model_name>/style_data``, or None when missing."""
    import json

    manifest = os.path.join(logs_path, model_name, "style_data", "style_data.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        f"{data.get('clips', 0)} clips, {data.get('hours', 0.0):.1f} h, "
        f"{len(data.get('speaker_descriptors', {}))} speaker(s), embedder {data.get('embedder', '?')}."
    )


def _run_style_process(command) -> int:
    """Run a style script, remembered so ``stop_style_script`` can stop it."""
    global _style_process
    _style_process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        return _style_process.wait()
    finally:
        _style_process = None


def stop_style_script():
    """Interrupt the running style extraction or training; training saves a
    checkpoint on the way out and resumes from it next time."""
    process = _style_process
    if process is None or process.poll() is not None:
        return "No style job is running."
    if os.name == "nt":
        process.terminate()
    else:
        process.send_signal(signal.SIGINT)
    return "Stopping the style job; training saves a checkpoint first."


def _style_device(gpu) -> str:
    return "cpu" if str(gpu).strip() in ("-", "") else f"cuda:{str(gpu).split('-')[0]}"


def run_style_extract_script(
    model_name: str,
    dataset_path: str = None,
    embedder_model: str = "contentvec",
    base_path: str = None,
    gpu: str = "0",
    recompute: bool = False,
):
    """Build ``logs/<model_name>/style_data`` from a folder of whole audio
    files (``dataset_path``; ``<id>_<name>`` subfolders are speakers) or, with
    no path, from the RVC experiment ``logs/<model_name>``.  ``base_path``
    ties it to a base for fine-tuning; otherwise ``embedder_model`` is used
    for a new base."""
    exp_dir = os.path.join(logs_path, model_name)
    command = [
        python,
        os.path.join(current_script_directory, "tools", "style_flow", "extract.py"),
        "--out", os.path.join(exp_dir, "style_data"),
        "--device", _style_device(gpu),
    ]
    if dataset_path:
        if not os.path.isdir(dataset_path):
            return f"Dataset folder not found: {dataset_path}"
        command += ["--audio-dir", dataset_path]
    elif os.path.isdir(os.path.join(exp_dir, "sliced_audios")):
        command += ["--experiment", exp_dir]
        if recompute:
            command.append("--recompute")
    else:
        return f"Pick a dataset folder; logs/{model_name} is not a preprocessed RVC experiment."
    if base_path:
        if not os.path.exists(base_path):
            return f"Style base not found: {base_path}"
        command += ["--reference", base_path, "--config",
                    os.path.join(current_script_directory, "rvc", "configs", "style_flow", "finetune.yaml")]
    else:
        command += ["--embedder", embedder_model, "--config",
                    os.path.join(current_script_directory, "rvc", "configs", "style_flow", "pretrain.yaml")]
    code = _run_style_process(command)
    if code != 0:
        return f"Style extraction for {model_name} stopped ({describe_exit_code(code)}); rerun to resume."
    return f"Style data for {model_name}: {style_data_summary(model_name)}"


def run_style_train_script(
    model_name: str,
    base_path: str = None,
    steps: int = 0,
    precision: str = "bf16",
    gpu: str = "0",
    speech_speakers: str = None,
    batch_size: int = 0,
    checkpointing: bool = False,
    compile_model: bool = False,
):
    """Train a style model on ``logs/<model_name>``, extracting its style
    dataset from the RVC experiment there when it has none yet.  With
    ``base_path`` it fine-tunes LoRA adapters on that base into
    ``<model_name>_style.pt``;
    without it, it pretrains a new base, ``style_base.pt``.  ``steps`` and
    ``batch_size`` 0 keep the config's; ``speech_speakers`` ("0-4", "" for
    none) only applies to a pretrain."""
    exp_dir = os.path.join(logs_path, model_name)
    if not os.path.isdir(exp_dir):
        return f"Nothing to train on: extract style data for {model_name} first."
    pretrain = not base_path
    if not pretrain and not os.path.exists(base_path):
        return f"Style base not found: {base_path}"
    script = "pretrain.py" if pretrain else "finetune.py"
    command = [
        python,
        os.path.join(current_script_directory, "tools", "style_flow", script),
        "--experiment", exp_dir,
        "--precision", precision,
        "--device", _style_device(gpu),
    ]
    if steps:
        command += ["--steps", str(int(steps))]
    if batch_size:
        command += ["--batch-size", str(int(batch_size))]
    if checkpointing:
        command.append("--checkpointing")
    if compile_model:
        command.append("--compile")
    if pretrain:
        if speech_speakers is not None:
            command += ["--speech-speakers", speech_speakers]
    else:
        command += ["--base", base_path]
    code = _run_style_process(command)
    if code != 0:
        return (
            f"Style training for {model_name} stopped ({describe_exit_code(code)}). "
            "Rerun to resume; see the terminal if it failed."
        )
    output = "style_base.pt" if pretrain else f"{model_name}_style.pt"
    return f"Style model saved to {os.path.join(exp_dir, output)}."


def style_options(kwargs: dict) -> dict | None:
    """Pops the CLI's ``style_*`` options out of ``kwargs`` into the ``style``
    argument of the infer scripts."""
    import json

    keys = (
        "model", "strength", "rate", "vibrato_gain", "scoop_gain", "drop_gain", "intensity", "relative", "recenter",
        "steps", "cfg", "cfg_until", "descriptors",
    )
    values = {key: kwargs.pop(f"style_{key}", None) for key in keys}
    if not values["model"]:
        return None
    values["descriptors"] = json.loads(values["descriptors"]) if values["descriptors"] else {}
    return {k: v for k, v in values.items() if v is not None}


# Model information
def run_model_information_script(pth_path: str):
    from rvc.train.process.model_information import model_information

    # Loading the checkpoint is not free, so read it once and render the same
    # text the caller gets back.
    information = model_information(pth_path)
    print_settings_panel(
        (line.split(": ", 1) for line in information.splitlines() if ": " in line),
        title=os.path.basename(pth_path),
    )
    return information


# Model blender
def run_model_blender_script(
    model_name: str, pth_path_1: str, pth_path_2: str, ratio: float
):
    from rvc.train.process.model_blender import model_blender

    message, model_blended = model_blender(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Tensorboard
def run_tensorboard_script():
    from rvc.lib.extras.launch_tensorboard import launch_tensorboard_pipeline

    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link: str):
    from rvc.lib.extras.model_download import DownloadError, model_download_pipeline

    # Checked, like run_index_script: this used to report success whatever the
    # pipeline returned, so a failed download looked like a finished one.
    try:
        folder = model_download_pipeline(model_link)
    except DownloadError as error:
        return f"Model download failed: {error}"
    return f"Model downloaded to {os.path.relpath(folder)}."


# Prerequisites
def run_prerequisites_script(
    pretraineds_hifigan: bool,
    models: bool,
    exe: bool,
):
    prequisites_download_pipeline(
        pretraineds_hifigan,
        models,
        exe,
    )
    return "Prerequisites installed successfully."


# Audio analyzer
def run_audio_analyzer_script(
    input_path: str, save_plot_path: str = "logs/audio_analysis.png"
):
    from rvc.lib.extras.analyzer import analyze_audio

    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print_settings_panel(
        (line.split(": ", 1) for line in str(audio_info).splitlines() if ": " in line),
        title=os.path.basename(input_path),
    )
    success(f"Plot saved at '{plot_path}'.", tag="[ANALYZE]")
    return audio_info, plot_path


# Parse arguments
# =====================================================================
# Command line interface
# =====================================================================
# Options shared by several commands are declared once in rvc/cli_options.py
# and applied with @apply_options.  The previous argparse layer repeated them
# per subparser, which let the accepted ranges drift apart between commands.


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Command line interface for the RVC fork."""


@cli.command("infer")
@apply_options(INFER_OWN, inference_options(INFER_DEFAULTS), FORMANT_OPTIONS, STYLE_OPTIONS)
def infer(**kwargs):
    """Run inference on a single audio file."""
    style = style_options(kwargs)
    run_infer_script(**kwargs, style=style)


@cli.command("batch_infer")
@apply_options(BATCH_INFER_OWN, inference_options(BATCH_INFER_DEFAULTS), FORMANT_OPTIONS, STYLE_OPTIONS)
def batch_infer(**kwargs):
    """Run inference on every audio file in a folder."""
    style = style_options(kwargs)
    run_batch_infer_script(**kwargs, style=style)


@cli.command("tts")
@apply_options(TTS_OWN, inference_options(TTS_DEFAULTS))
def tts(**kwargs):
    """Synthesize speech with edge-tts and convert it."""
    run_tts_script(**kwargs)


@cli.command("preprocess")
@apply_options(PREPROCESS_OWN)
def preprocess(**kwargs):
    """Preprocess a dataset for training."""
    # run_preprocess_script names this argument differently from its flag:
    # the flag is --noise_reduction_strength, the parameter is clean_strength.
    kwargs["clean_strength"] = kwargs.pop("noise_reduction_strength")
    run_preprocess_script(**kwargs)


@cli.command("extract")
@apply_options(EXTRACT_OWN)
def extract(**kwargs):
    """Extract features and F0 from a preprocessed dataset."""
    run_extract_script(**kwargs)


@cli.command("train")
@apply_options(TRAIN_OWN)
def train(**kwargs):
    """Train a model."""
    run_train_script(**kwargs)


@cli.command("index")
@apply_options(INDEX_OWN)
def index(**kwargs):
    """Build the FAISS index for a trained model."""
    run_index_script(**kwargs)


@cli.command("style_extract")
@apply_options(STYLE_EXTRACT_OWN)
def style_extract(**kwargs):
    """Build a style dataset from an audio folder or an RVC experiment."""
    print(run_style_extract_script(**kwargs))


@cli.command("style_train")
@apply_options(STYLE_TRAIN_OWN)
def style_train(**kwargs):
    """Pretrain a style base, or fine-tune one on a singer."""
    print(run_style_train_script(**kwargs))


@cli.command("model_information")
@apply_options(MODEL_INFORMATION_OWN)
def model_information(**kwargs):
    """Print the metadata stored in a .pth file."""
    run_model_information_script(**kwargs)


@cli.command("model_blender")
@apply_options(MODEL_BLENDER_OWN)
def model_blender(**kwargs):
    """Blend two models into one."""
    run_model_blender_script(**kwargs)


@cli.command("tensorboard")
def tensorboard():
    """Launch TensorBoard."""
    run_tensorboard_script()


@cli.command("download")
@apply_options(DOWNLOAD_OWN)
def download(**kwargs):
    """Download a model from a link."""
    from rvc.lib.extras.model_download import DownloadError, model_download_pipeline

    try:
        model_download_pipeline(kwargs["model_link"])
    except DownloadError:
        raise SystemExit(1)  # the pipeline has printed why


@cli.command("prerequisites")
@apply_options(PREREQUISITES_OWN)
def prerequisites(**kwargs):
    """Download the prerequisite models and executables."""
    run_prerequisites_script(**kwargs)


@cli.command("audio_analyzer")
@apply_options(AUDIO_ANALYZER_OWN)
def audio_analyzer(**kwargs):
    """Analyze an audio file and print a report."""
    run_audio_analyzer_script(**kwargs)


def main():
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as error:
        error.show()
        sys.exit(error.exit_code)
    except click.Abort:
        print("Aborted.")
        sys.exit(1)
    except Exception as error:
        import traceback

        print_error_panel(
            error,
            title="Command failed",
            details=traceback.format_exc(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
