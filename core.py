import click
import psutil
import os
import sys
import json
import shutil

import atexit
import platform
import signal
import subprocess
import time
from multiprocessing import cpu_count


from functools import lru_cache


now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import (
    DEFAULT_CPU_THREADS,
    install_rich_print,
    print_error_panel,
    print_settings_panel,
    success,
)

install_rich_print()

current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
from rvc.configs.vocoders import (
    get_all_vocoder_sample_rates,
    get_vocoder_cli_choices,
    get_vocoder_sample_rates,
    normalize_vocoder,
)
from rvc.train.run_spec import TrainRunSpec
from rvc.train.messages import (
    TORCH_COMPILE_MODE_CLI_HELP,
    TORCH_COMPILE_MODES,
    VOCODER_COMPILE_CLI_HELP,
)

python = sys.executable
training_process = None

# Get TTS Voices -> https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4
@lru_cache(maxsize=1)  # Cache only one result since the file is static
def load_voices_data():
    with open(
        os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
    ) as file:
        return json.load(file)


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
    embedder_model_custom: str = None,
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
        "embedder_model_custom": embedder_model_custom,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "sid": sid,
        "seed": seed,
        "bundle_submodel": bundle_submodel,
        "index_k": index_k,
        "index_power": index_power,
        "index_continuity": index_continuity,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(
        **kwargs,
    )

    export_path = output_path.replace(".wav", f".{export_format.lower()}")
    if not os.path.exists(export_path):
        export_path = output_path

    try:
        tmp_dir = os.path.join(os.environ.get("TEMP", os.path.join(now_dir, "temp")), "infer_preview")
        os.makedirs(tmp_dir, exist_ok=True)
        preview_path = os.path.join(tmp_dir, os.path.basename(export_path))
        shutil.copy2(export_path, preview_path)
        return f"File {input_path} inferred successfully. Saved to: {export_path}", preview_path
    except Exception:
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
    embedder_model_custom: str = None,
    formant_shifting: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    sid: int = 0,
    seed: int = 0,
    index_k: int = 8,
    index_power: float = 2.0,
    index_continuity: float = 0.5,
    silence_gate_db: float = -60.0,
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
        "embedder_model_custom": embedder_model_custom,
        "formant_shifting": formant_shifting,
        "formant_qfrency": formant_qfrency,
        "formant_timbre": formant_timbre,
        "sid": sid,
        "seed": seed,
        "index_k": index_k,
        "index_power": index_power,
        "index_continuity": index_continuity,
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
    embedder_model_custom: str = None,
    sid: int = 0,
    seed: int = 0,
    index_k: int = 8,
    index_power: float = 2.0,
    index_continuity: float = 0.5,
    silence_gate_db: float = -60.0,
):

    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")

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
    subprocess.run(command_tts)
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
        embedder_model_custom=embedder_model_custom,
        sid=sid,
        seed=seed,
        formant_shifting=None,
        formant_qfrency=None,
        formant_timbre=None,
        index_k=index_k,
        index_power=index_power,
        index_continuity=index_continuity,
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
    normalization_mode: str = "post_rms",
    loading_resampling: str = "librosa",
    use_smart_cutter: bool = False,
    dataset_format: str = "WAV",
    rms_norm_db: float = -18.0
):
    if int(sample_rate) == 44100 and use_smart_cutter:
        raise ValueError("SmartCutter is not available for the 44.1 kHz vocoder configuration.")

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
                use_smart_cutter,
                dataset_format,
                rms_norm_db,
            ],
        ),
    ]
    subprocess.run(command)
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
    embedder_model_custom: str = None,
    include_mutes: int = 2,
    remove_16k_slices: bool = False,
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
                embedder_model_custom,
                include_mutes,
                remove_16k_slices,
                feature_precision,
            ],
        ),
    ]

    subprocess.run(command_1)

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
    optimizer_choice: str = "AdamW",
    use_checkpointing: bool = False,
    use_tf32: bool = False,
    use_fp16: bool = False,
    use_benchmark: bool = True,
    lr_scheduler: str = "exp decay step",
    use_custom_lr: bool = False,
    custom_lr_g: float = 1e-4,
    custom_lr_d: float = 1e-4,
    compile_vocoder: bool = False,
    torch_compile_mode: str = "default",
    overtrain_detector: bool = False,
    stop_on_overtrain: bool = False,
    use_ema: bool = True,
):
    global training_process

    vocoder = normalize_vocoder(vocoder)
    if int(sample_rate) not in get_vocoder_sample_rates(vocoder):
        raise ValueError(
            f"{vocoder} does not provide a configuration for {sample_rate} Hz."
        )
    if pretrained == True:
        from rvc.lib.tools.pretrained_selector import pretrained_selector

        if custom_pretrained == False:
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
        optimizer_choice=str(optimizer_choice),
        lr_scheduler=str(lr_scheduler),
        use_warmup=bool(use_warmup),
        warmup_duration=int(warmup_duration),
        use_custom_lr=bool(use_custom_lr),
        # The UIs send the slider value even with the override off, but a CLI
        # caller can pass nothing at all.
        custom_lr_g=float(custom_lr_g if custom_lr_g is not None else 1e-4),
        custom_lr_d=float(custom_lr_d if custom_lr_d is not None else 1e-4),
        use_checkpointing=bool(use_checkpointing),
        use_tf32=bool(use_tf32),
        use_fp16=bool(use_fp16),
        use_benchmark=bool(use_benchmark),
        compile_vocoder=bool(compile_vocoder),
        torch_compile_mode=str(torch_compile_mode),
        overtrain_detector=bool(overtrain_detector),
        stop_on_overtrain=bool(stop_on_overtrain),
        use_ema=bool(use_ema),
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
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        training_process = subprocess.Popen(
            command,
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
        print(f"[TRAINING] Graceful stop failed ({error}); killing instead.")
    else:
        print(f"[TRAINING] Asked PID {pid} to stop, waiting up to {TRAINING_STOP_GRACE_SECONDS}s...")
        if _wait_for_exit(training_process, TRAINING_STOP_GRACE_SECONDS):
            return f"Training stopped (PID {pid})."
        print(f"[TRAINING] PID {pid} did not exit in time; killing the tree.")

    alive = _kill_process_tree(pid)
    if alive:
        return f"Could not stop PID(s): {', '.join(str(p.pid) for p in alive)}."
    return f"Training force-stopped (PID {pid})."


def _stop_training_at_exit():
    """Take the trainer down when this interface exits normally."""
    if training_process is not None and training_process.poll() is None:
        print("[TRAINING] Interface is exiting; stopping the training run.")
        stop_train_script()


atexit.register(_stop_training_at_exit)


# Index
def run_index_script(
    model_name: str, index_algorithm: str, index_metric: str = "l2"
):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        index_algorithm,
        index_metric,
    ]

    # Checked, because it has not always succeeded: the script's ``rvc.``
    # import failed under the subprocess's sys.path and this reported success
    # regardless, so a missing index looked like a working one.
    result = subprocess.run(command)
    if result.returncode != 0:
        return (
            f"Index generation for {model_name} failed (exit {result.returncode}). "
            "See the terminal for the traceback."
        )
    return f"Index file for {model_name} generated successfully."


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
    from rvc.lib.tools.launch_tensorboard import launch_tensorboard_pipeline

    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link: str):
    from rvc.lib.tools.model_download import model_download_pipeline

    model_download_pipeline(model_link)
    return f"Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(
    pretraineds_hifigan: bool,
    models: bool,
    exe: bool,
    smartcutter: bool,
):
    prequisites_download_pipeline(
        pretraineds_hifigan,
        models,
        exe,
        smartcutter,
    )
    return "Prerequisites installed successfully."


# Audio analyzer
def run_audio_analyzer_script(
    input_path: str, save_plot_path: str = "logs/audio_analysis.png"
):
    from rvc.lib.tools.analyzer import analyze_audio

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
# Options shared by several commands are declared once in the groups below and
# applied with @apply_options.  The previous argparse layer repeated them per
# subparser, which let the accepted ranges drift apart between commands.


def _tts_voice_choices():
    """Voice short names accepted by the tts command."""
    return [voice["ShortName"] for voice in load_voices_data()]


def apply_options(*groups):
    """Apply one or more lists of click.option decorators to a command."""
    def decorator(function):
        for group in reversed(groups):
            for option in reversed(group):
                function = option(function)
        return function

    return decorator


# ---- Shared option groups -------------------------------------------

#: Per-command overrides for the conversion options below.  Single-file, batch
#: and TTS ship genuinely different numbers in the Gradio tabs -- TTS leans on
#: the index harder and protects consonants more, because a synthesised source
#: has no breath to preserve -- so one shared set would match none of them.
#: Anything absent here keeps the value in the list.
INFER_DEFAULTS = {"index_rate": 0.5, "clean_strength": 0.3}
BATCH_INFER_DEFAULTS = {"index_rate": 0.5, "clean_strength": 0.5, "protect": 0.3}
TTS_DEFAULTS = {
    "index_rate": 0.75,
    "clean_strength": 0.5,
    "protect": 0.5,
    "clean_audio": True,
}


def inference_options(overrides: dict | None = None) -> list:
    """The conversion knobs, built with one command's defaults applied."""
    chosen = overrides or {}

    def default(name, fallback):
        return chosen.get(name, fallback)

    return [
        click.option(
            "--pitch",
            type=click.IntRange(-24, 24),
            default=0,
            show_default=True,
            help="Set the pitch of the audio. Higher values result in a higher pitch.",
        ),
        click.option(
            "--filter_radius",
            type=click.IntRange(0, 10),
            default=3,
            show_default=True,
            help="Apply median filtering to the extracted pitch values if this value is greater than or equal to three. This can help reduce breathiness in the output audio.",
        ),
        click.option(
            "--index_rate",
            type=click.FloatRange(0, 1),
            default=default("index_rate", 0.3),
            show_default=True,
            help="Control the influence of the index file on the output. Higher values mean stronger influence. Lower values can help reduce artifacts but may result in less accurate voice cloning.",
        ),
        click.option(
            "--index_k",
            type=click.IntRange(1, 64),
            default=8,
            show_default=True,
            help="Number of index neighbours averaged per frame. Fewer keeps what is idiosyncratic about the matched frames; more averages toward the dataset's mean voice.",
        ),
        click.option(
            "--index_power",
            type=click.FloatRange(0, 8),
            default=2.0,
            show_default=True,
            help="Exponent of the inverse-distance weighting of those neighbours. 0 is a flat average, larger values approach picking only the nearest.",
        ),
        click.option(
            "--index_continuity",
            type=click.FloatRange(0, 4),
            default=0.5,
            show_default=True,
            help="Reward for neighbours that continue the frame the previous one matched, which stops the retrieval jumping between unrelated parts of the dataset. Needs an index built by this fork; ignored otherwise.",
        ),
        click.option(
            "--volume_envelope",
            type=click.FloatRange(0, 1),
            default=1,
            show_default=True,
            help="Control the blending of the output's volume envelope. A value of 1 means the output envelope is fully used.",
        ),
        click.option(
            "--silence_gate_db",
            type=click.FloatRange(-120, 0),
            default=-60.0,
            show_default=True,
            help="Input level, in dBFS, under which the output is faded out. The content encoder gives digital silence a full-magnitude embedding that the decoder renders as hiss, so passages the input says are empty come back noisy; this gates them. -120 disables it.",
        ),
        click.option(
            "--protect",
            type=click.FloatRange(0, 0.5),
            default=default("protect", 0.33),
            show_default=True,
            help="Protect consonants and breathing sounds from artifacts. A value of 0.5 offers the strongest protection, while lower values may reduce the protection level but potentially mitigate the indexing effect.",
        ),
        click.option(
            "--f0_method",
            type=click.Choice(["crepe", "crepe-tiny", "rmvpe", "fcpe"]),
            default='rmvpe',
            show_default=True,
            help="Choose the pitch extraction algorithm for the conversion. 'rmvpe' is the default and generally recommended.",
        ),
        click.option(
            "--pth_path",
            type=str,
            required=True,
            help="Full path to the RVC model file (.pth).",
        ),
        click.option(
            "--index_path",
            type=str,
            required=True,
            help="Full path to the index file (.index).",
        ),
        click.option(
            "--split_audio",
            type=click.BOOL,
            default=False,
            show_default=True,
            help="Split the audio into smaller segments before inference. This can improve the quality of the output for longer audio files.",
        ),
        click.option(
            "--f0_autotune",
            type=click.BOOL,
            default=False,
            show_default=True,
            help="Apply a light autotune to the inferred audio. Particularly useful for singing voice conversions.",
        ),
        click.option(
            "--f0_autotune_strength",
            type=click.FloatRange(0, 1),
            default=1.0,
            show_default=True,
            help="Set the autotune strength - the more you increase it the more it will snap to the chromatic grid.",
        ),
        click.option(
            "--clean_audio",
            type=click.BOOL,
            default=default("clean_audio", False),
            show_default=True,
            help="Clean the output audio using noise reduction algorithms. Recommended for speech conversions.",
        ),
        click.option(
            "--clean_strength",
            type=click.FloatRange(0, 1),
            default=default("clean_strength", 0.7),
            show_default=True,
            help="Adjust the intensity of the audio cleaning process. Higher values result in stronger cleaning, but may lead to a more compressed sound.",
        ),
        click.option(
            "--export_format",
            type=click.Choice(["WAV", "MP3", "FLAC", "OGG", "M4A"]),
            default='WAV',
            show_default=True,
            help="Select the desired output audio format.",
        ),
        click.option(
            "--embedder_model",
            type=click.Choice(["contentvec", "spin_v1", "spin_v2", "custom"]),
            default='contentvec',
            show_default=True,
            help="Choose the model used for generating speaker embeddings.",
        ),
        click.option(
            "--embedder_model_custom",
            type=str,
            default=None,
            help="Specify the path to a custom model for speaker embedding. Only applicable if 'embedder_model' is set to 'custom'.",
        ),
        click.option(
            "--f0_file",
            type=str,
            default=None,
            help="Full path to an external F0 file (.f0). This allows you to use pre-computed pitch values for the input audio.",
        ),
    ]

FORMANT_OPTIONS = [
    click.option(
        "--formant_shifting",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Apply formant shifting to the input audio. This can help adjust the timbre of the voice.",
    ),
    click.option(
        "--formant_qfrency",
        type=float,
        default=1.0,
        show_default=True,
        help="Control the frequency of the formant shifting effect. Higher values result in a more pronounced effect.",
    ),
    click.option(
        "--formant_timbre",
        type=float,
        default=1.0,
        show_default=True,
        help="Control the timbre of the formant shifting effect. Higher values result in a more pronounced effect.",
    ),
    click.option(
        "--sid",
        type=int,
        default=0,
        show_default=True,
        help="Speaker ID for multi-speaker models.",
    ),
]

# ---- infer ----
INFER_OWN = [
    click.option(
        "--input_path",
        type=str,
        required=True,
        help="Full path to the input audio file.",
    ),
    click.option(
        "--output_path",
        type=str,
        required=True,
        help="Full path to the output audio file.",
    ),
]

# ---- batch_infer ----
BATCH_INFER_OWN = [
    click.option(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input audio files.",
    ),
    click.option(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder for saving output audio files.",
    ),
]

# ---- tts ----
TTS_OWN = [
    click.option(
        "--tts_file",
        type=str,
        required=True,
        help="File with a text to be synthesized",
    ),
    click.option("--tts_text", type=str, required=True, help="Text to be synthesized"),
    click.option(
        "--tts_voice",
        type=click.Choice(_tts_voice_choices()),
        required=True,
        help="Voice to be used for TTS synthesis.",
    ),
    click.option(
        "--tts_rate",
        type=click.IntRange(-100, 100),
        default=0,
        show_default=True,
        help="Control the speaking rate of the TTS. Values range from -100 (slower) to 100 (faster).",
    ),
    click.option(
        "--output_tts_path",
        type=str,
        required=True,
        help="Full path to save the synthesized TTS audio.",
    ),
    click.option(
        "--output_rvc_path",
        type=str,
        required=True,
        help="Full path to save the voice-converted audio using the synthesized TTS.",
    ),
]

# ---- preprocess ----
PREPROCESS_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be trained.",
    ),
    click.option(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Target sampling rate for the audio data.",
    ),
    click.option(
        "--cpu_threads",
        type=click.IntRange(1, min(cpu_count(), 192)),
        default=4,
        show_default=True,
        help="Number of CPU threads to use for preprocessing.",
    ),
    click.option(
        "--cut_preprocess",
        type=click.Choice(["Skip", "Simple", "Automatic"]),
        default='Simple',
        show_default=True,
        help="Cut the dataset into smaller segments for faster preprocessing.",
    ),
    click.option(
        "--process_effects",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Enable high-pass filtering during preprocessing.",
    ),
    click.option(
        "--noise_reduction",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enable noise reduction during preprocessing.",
    ),
    click.option(
        "--noise_reduction_strength",
        type=click.FloatRange(0, 1),
        default=0.7,
        show_default=True,
        help="Strength of the noise reduction filter.",
    ),
    click.option(
        "--chunk_len",
        type=click.FloatRange(0.5, 5.0),
        default=3.0,
        show_default=True,
        help="Chunk length.",
    ),
    click.option(
        "--overlap_len",
        type=click.FloatRange(0.0, 0.4),
        default=0.36,
        show_default=True,
        help="Overlap length.",
    ),
    click.option(
        "--normalization_mode",
        type=click.Choice(["none", "post_peak", "post_peak_rvc", "post_rms"]),
        default='post_peak',
        show_default=True,
        help="Normalization mode.",
    ),
    click.option(
        "--loading_resampling",
        type=click.Choice(["librosa", "ffmpeg"]),
        default='librosa',
        show_default=True,
        help="Librosa's using SoXr, FFmpeg's using Windowed Sinc filter with Blackman-Nuttall window.",
    ),
    click.option(
        "--use_smart_cutter",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enable SmartCutter silence-truncation during preprocessing.",
    ),
]

# ---- extract ----
EXTRACT_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the model."),
    click.option(
        "--f0_method",
        type=click.Choice(["crepe", "crepe-tiny", "rmvpe", "fcpe"]),
        default='rmvpe',
        show_default=True,
        help="Pitch extraction method to use.",
    ),
    click.option(
        "--cpu_threads",
        type=click.IntRange(1, min(cpu_count(), 192)),
        default=4,
        show_default=True,
        help="Number of CPU threads to use for feature extraction (optional).",
    ),
    click.option(
        "--gpu",
        type=str,
        default='-',
        show_default=True,
        help="GPU device to use for feature extraction (optional).",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Target sampling rate for the audio data.",
    ),
    click.option(
        "--vocoder_arch",
        type=click.Choice(get_vocoder_cli_choices()),
        default='hifi',
        show_default=True,
        help="Choose the vocoder architecture",
    ),
    click.option(
        "--embedder_model",
        type=click.Choice(["contentvec", "spin_v1", "spin_v2", "custom"]),
        default='contentvec',
        show_default=True,
        help="Choose the model used for generating speaker embeddings.",
    ),
    click.option(
        "--embedder_model_custom",
        type=str,
        default=None,
        help="Specify the path to a custom model for speaker embedding. Only applicable if 'embedder_model' is set to 'custom'.",
    ),
    click.option(
        "--include_mutes",
        type=click.IntRange(0, 10),
        default=2,
        show_default=True,
        help="Number of silent files to include.",
    ),
    click.option(
        "--remove_16k_slices",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Delete the 16 kHz slices once extraction has consumed them.",
    ),
    click.option(
        "--feature_precision",
        type=click.Choice(["fp32", "fp16"]),
        default="fp32",
        show_default=True,
        help="Precision the extracted embeddings are stored at. fp32 doubles the feature cache on disk but keeps the retrieval index free of a quantisation floor; fp16 halves it. Either can be read back without re-extracting.",
    ),
]

# ---- train ----
TRAIN_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be trained.",
    ),
    click.option(
        "--vocoder",
        type=click.Choice(get_vocoder_cli_choices()),
        default='hifi',
        show_default=True,
        help="Vocoder name",
    ),
    click.option(
        "--optimizer_choice",
        type=click.Choice(["AdamW", "Sched-Free AdamW", "Muon", "Lion"]),
        default='AdamW',
        show_default=True,
        help="Optimizer for the generator and discriminator. Mirrors rvc.train.optimizers.OPTIMIZER_CHOICES; kept as a literal so --help does not import torch.",
    ),
    click.option(
        "--use_checkpointing",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enables usage of checkpointing.",
    ),
    click.option(
        "--compile_vocoder",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Compile the selected vocoder decoder during training.",
    ),
    click.option(
        "--torch_compile_mode",
        type=click.Choice(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]),
        default='default',
        show_default=True,
        help="Torch compile mode used for the vocoder decoder.",
    ),
    click.option(
        "--custom_lr_g",
        type=float,
        default=0.0001,
        show_default=True,
        help="Custom learning rate for generator.",
    ),
    click.option(
        "--custom_lr_d",
        type=float,
        default=0.0001,
        show_default=True,
        help="Custom learning rate for discriminator.",
    ),
    click.option(
        "--overtrain_detector",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Holds a few whole source recordings out of training and scores them periodically. Training loss cannot see overtraining; this is the only signal that can. Auto-disables on datasets too small to give any away.",
    ),
    click.option(
        "--stop_on_overtrain",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Ends the run once held-out quality has stopped improving. Off by default: the pre-overtrain weights are exported either way, this only decides whether training keeps going.",
    ),
    click.option(
        "--use_ema",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Keeps an exponential moving average of the generator weights and exports that instead of a single step. Usually better than any one step of a GAN vocoder, and it makes the overtrain curve far less noisy. Costs one extra copy of the generator in VRAM.",
    ),
    click.option(
        "--epoch_save_frequency",
        type=click.IntRange(1, 100),
        required=True,
        help="Save the model every specified number of epochs.",
    ),
    click.option(
        "--save_only_latest_net_models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Save only the latest G/D files.",
    ),
    click.option(
        "--save_weight_models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Save model weights every epoch.",
    ),
    click.option(
        "--total_epoch_count",
        type=click.IntRange(1, 10000),
        default=500,
        show_default=True,
        help="Total number of epochs to train for.",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Sampling rate of the training data.",
    ),
    click.option(
        "--batch_size",
        type=click.IntRange(1, 50),
        default=8,
        show_default=True,
        help="Batch size for training.",
    ),
    click.option(
        "--gpu",
        type=str,
        default='0',
        show_default=True,
        help="GPU device to use for training (e.g., '0').",
    ),
    click.option(
        "--pretrained",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Use a pretrained model for initialization.",
    ),
    click.option(
        "--custom_pretrained",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Use a custom pretrained model.",
    ),
    click.option(
        "--g_pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained generator model file.",
    ),
    click.option(
        "--d_pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained discriminator model file.",
    ),
    click.option(
        "--use_warmup",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enables usage of warmup.",
    ),
    click.option(
        "--warmup_duration",
        type=click.IntRange(1, 999),
        default=5,
        show_default=True,
        help="Duration of warmup phase (in epochs).",
    ),
    click.option(
        "--use_tf32",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Lets you choose between FP32 and TF32 precision used in training.",
    ),
    click.option(
        "--use_fp16",
        type=click.BOOL,
        default=False,
        show_default=True,
        help=(
            "Run the forward pass under FP16 autocast with a GradScaler. Master "
            "weights stay FP32; distribution math and the NSF source stay FP32. "
            "Off means plain FP32 (with TF32 tensor cores if --use_tf32)."
        ),
    ),
    click.option(
        "--use_benchmark",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Enable cuDNN benchmark mode for potential speedup.",
    ),
    click.option(
        "--lr_scheduler",
        type=click.Choice(["exp decay step", "exp decay epoch", "cosine annealing", "none"]),
        default='exp decay epoch',
        show_default=True,
        help="Pick the shared LR scheduler for generator and discriminator.",
    ),
    click.option(
        "--use_custom_lr",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enables customization of learning rate for Generator and Discriminator.",
    ),
    click.option(
        "--cleanup",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Cleanup previous training attempt.",
    ),
    click.option(
        "--index_algorithm",
        type=click.Choice(["Auto", "Faiss", "KMeans"]),
        default='Auto',
        show_default=True,
        help="Choose the method for generating the index file.",
    ),
]

# ---- index ----
INDEX_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the model."),
    click.option(
        "--index_algorithm",
        type=click.Choice(["Auto", "Faiss", "KMeans"]),
        default='Auto',
        show_default=True,
        help="Choose the method for generating the index file.",
    ),
    click.option(
        "--index_metric",
        type=click.Choice(["l2", "cosine"]),
        default="l2",
        show_default=True,
        help="Similarity used to find neighbours. l2 reproduces what upstream RVC builds; cosine ranks by direction alone, which suits embeddings whose magnitude tracks loudness.",
    ),
]

# ---- model_information ----
MODEL_INFORMATION_OWN = [
    click.option(
        "--pth_path",
        type=str,
        required=True,
        help="Path to the .pth model file.",
    ),
]

# ---- model_blender ----
MODEL_BLENDER_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the new fused model.",
    ),
    click.option(
        "--pth_path_1",
        type=str,
        required=True,
        help="Path to the first .pth model file.",
    ),
    click.option(
        "--pth_path_2",
        type=str,
        required=True,
        help="Path to the second .pth model file.",
    ),
    click.option(
        "--ratio",
        type=click.FloatRange(0, 1),
        default=0.5,
        show_default=True,
        help="Ratio for blending the two models (0.0 to 1.0).",
    ),
]

# ---- download ----
DOWNLOAD_OWN = [
    click.option(
        "--model_link",
        type=str,
        required=True,
        help="Direct link to the model file.",
    ),
]

# ---- prerequisites ----
PREREQUISITES_OWN = [
    click.option(
        "--pretraineds_hifigan",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download pretrained models for RVC v2.",
    ),
    click.option(
        "--models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download additional models.",
    ),
    click.option(
        "--exe",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download required executables.",
    ),
    click.option(
        "--smartcutter",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download required SmartCutter models.",
    ),
]

# ---- audio_analyzer ----
AUDIO_ANALYZER_OWN = [
    click.option(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input audio file.",
    ),
]

# ---- Commands -------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Command line interface for the RVC fork."""


@cli.command("infer")
@apply_options(INFER_OWN, inference_options(INFER_DEFAULTS), FORMANT_OPTIONS)
def infer(**kwargs):
    """Run inference on a single audio file."""
    run_infer_script(**kwargs)


@cli.command("batch_infer")
@apply_options(BATCH_INFER_OWN, inference_options(BATCH_INFER_DEFAULTS), FORMANT_OPTIONS)
def batch_infer(**kwargs):
    """Run inference on every audio file in a folder."""
    run_batch_infer_script(**kwargs)


@cli.command("tts")
@apply_options(TTS_OWN, inference_options(TTS_DEFAULTS))
def tts(**kwargs):
    """Synthesize speech with edge-tts and convert it."""
    run_tts_script(**kwargs)


@cli.command("preprocess")
@apply_options(PREPROCESS_OWN)
def preprocess(**kwargs):
    """Preprocess a dataset for training."""
    # run_preprocess_script names this argument differently from its flag.
    kwargs["noise_reduction_strength"] = kwargs.pop("clean_strength")
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
    run_download_script(**kwargs)


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
