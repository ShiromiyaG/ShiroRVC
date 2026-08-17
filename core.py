import psutil
import os
import sys
import json
import argparse
import shutil

import platform
import subprocess
import signal
from multiprocessing import cpu_count


from functools import lru_cache
from distutils.util import strtobool


now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import DEFAULT_CPU_THREADS, install_rich_print

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
            ],
        ),
    ]

    subprocess.run(command_1)

    return f"Model {model_name} extracted successfully."


# Train
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
    use_benchmark: bool = True,
    use_deterministic: bool = False,
    spectral_loss: str = "L1 Mel Loss",
    lr_scheduler: str = "exp decay step",
    exp_decay_gamma: str = "0.999875",
    use_kl_annealing: bool = False,
    kl_annealing_cycle_duration: int = 3,
    rolling_loss_steps: int = 50,
    grad_clip_scheduling: bool = False,
    grad_clip_steps_duration: int = 0,
    grad_clip_value_g_cap: int = 0,
    grad_clip_value_d_cap: int = 0,
    grad_clip_value_g_release: int = 0,
    grad_clip_value_d_release: int = 0,
    use_custom_lr: bool = False,
    custom_lr_g: float = 1e-4,
    custom_lr_d: float = 1e-4,
    use_2_sample_kl: bool = False,
    use_best_step: bool = False,
    double_d_updates: bool = False,
    compile_vocoder: bool = False,
    torch_compile_mode: str = "default",
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

    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [
        python,
        train_script_path,
        *map(
            str,
            [
                model_name,
                epoch_save_frequency,
                total_epoch_count,
                pg,
                pd,
                gpu,
                batch_size,
                sample_rate,
                save_only_latest_net_models,
                save_weight_models,
                use_warmup,
                warmup_duration,
                cleanup,
                vocoder,
                "RVC",
                optimizer_choice,
                optimizer_choice,
                use_checkpointing,
                use_tf32,
                use_benchmark,
                use_deterministic,
                spectral_loss,
                lr_scheduler,
                exp_decay_gamma,
                use_kl_annealing,
                kl_annealing_cycle_duration,
                rolling_loss_steps,
                grad_clip_scheduling,
                grad_clip_steps_duration,
                grad_clip_value_g_cap,
                grad_clip_value_d_cap,
                grad_clip_value_g_release,
                grad_clip_value_d_release,
                use_custom_lr,
                custom_lr_g,
                custom_lr_d,
                use_2_sample_kl,
                use_best_step,
                double_d_updates,
                compile_vocoder,
                torch_compile_mode,
            ],
        ),
    ]
    if platform.system() == "Windows":
        global training_process
        training_process = subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        training_process = subprocess.Popen(
            command,
            preexec_fn=os.setsid
        )

    training_process.wait()
    return f"Training has been successfully completed or stopped."


# Stopping the training
def stop_train_script():
    global training_process
    if training_process and training_process.poll() is None:
        try:
            pid = training_process.pid
            parent_process = psutil.Process(pid)

            worker_pids = []
            for child in parent_process.children(recursive=True):
                worker_pids.append(child.pid)

            for pid in worker_pids:
                worker_process = psutil.Process(pid)
                worker_process.terminate()
                print(f"[TRAINING] Terminated child worker process PID: {pid}")

            parent_process.terminate()
            print(f"[TRAINING] Terminated parent process PID: {pid}")

            return ""

        except psutil.NoSuchProcess as e:
            print(f"No such process: {e}")
            return "No running process found."
        except psutil.AccessDenied as e:
            print(f"Permission denied: {e}")
            return "Failed to terminate process due to permission issues."
        except Exception as e:
            print(f"Error while stopping process: {e}")
            return f"Error stopping process: {e}"
    else:
    # Emergency-Nuke if everything else failed.
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/IM", "python.exe"])
        else:
            subprocess.run(["pkill", "-f", "rvc/train/train.py"])
        return "Emergency-Nuke issued."

# Saving the models at given step count and stopping the training
def early_save_stop():
    global training_process

    if training_process and training_process.poll() is None:
        print(f"[TRAINING]  Sending Early Stopping signal to PID: {training_process.pid}")

        try:
            if platform.system() == "Windows":
                os.kill(training_process.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.kill(training_process.pid, signal.SIGINT)

            try:
                training_process.wait(timeout=10)
                print("[TRAINING] Early Stopping completed.")
                return ""

            except subprocess.TimeoutExpired:
                print("[TRAINING] Save timed out! Issuing Hard-Stop.")
                return stop_train_script()

        except Exception as e:
            print(f"[TRAINING] Signal Error: {e}")
            return f"Error during the Early Stopping-stop: {e}"
    else:
        return "No active training process to early stop."


# Index
def run_index_script(model_name: str, index_algorithm: str):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        index_algorithm,
    ]

    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."


# Model information
def run_model_information_script(pth_path: str):
    from rvc.train.process.model_information import model_information

    print(model_information(pth_path))
    return model_information(pth_path)


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
    print(
        f"Audio info of {input_path}: {audio_info}",
        f"Audio file {input_path} analyzed successfully. Plot saved at: {plot_path}",
    )
    return audio_info, plot_path


# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the main.py script with specific parameters."
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="mode", help="Choose a mode"
    )

    # Parser for 'infer' mode
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    pitch_description = (
        "Set the pitch of the audio. Higher values result in a higher pitch."
    )
    infer_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    filter_radius_description = "Apply median filtering to the extracted pitch values if this value is greater than or equal to three. This can help reduce breathiness in the output audio."
    infer_parser.add_argument(
        "--filter_radius",
        type=int,
        help=filter_radius_description,
        choices=range(11),
        default=3,
    )
    index_rate_description = "Control the influence of the index file on the output. Higher values mean stronger influence. Lower values can help reduce artifacts but may result in less accurate voice cloning."
    infer_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=0.3,
    )
    volume_envelope_description = "Control the blending of the output's volume envelope. A value of 1 means the output envelope is fully used."
    infer_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=1,
    )
    protect_description = "Protect consonants and breathing sounds from artifacts. A value of 0.5 offers the strongest protection, while lower values may reduce the protection level but potentially mitigate the indexing effect."
    infer_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[i / 1000.0 for i in range(0, 501)],
        default=0.33,
    )
    f0_method_description = "Choose the pitch extraction algorithm for the conversion. 'rmvpe' is the default and generally recommended."
    infer_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
        ],
        default="rmvpe",
    )
    infer_parser.add_argument(
        "--input_path",
        type=str,
        help="Full path to the input audio file.",
        required=True,
    )
    infer_parser.add_argument(
        "--output_path",
        type=str,
        help="Full path to the output audio file.",
        required=True,
    )
    pth_path_description = "Full path to the RVC model file (.pth)."
    infer_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    index_path_description = "Full path to the index file (.index)."
    infer_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    split_audio_description = "Split the audio into smaller segments before inference. This can improve the quality of the output for longer audio files."
    infer_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    f0_autotune_description = "Apply a light autotune to the inferred audio. Particularly useful for singing voice conversions."
    infer_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    f0_autotune_strength_description = "Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."
    infer_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=f0_autotune_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    clean_audio_description = "Clean the output audio using noise reduction algorithms. Recommended for speech conversions."
    infer_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    clean_strength_description = "Adjust the intensity of the audio cleaning process. Higher values result in stronger cleaning, but may lead to a more compressed sound."
    infer_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    export_format_description = "Select the desired output audio format."
    infer_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    embedder_model_description = (
        "Choose the model used for generating speaker embeddings."
    )
    infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin_v1",
            "spin_v2",
            "custom",
        ],
        default="contentvec",
    )
    embedder_model_custom_description = "Specify the path to a custom model for speaker embedding. Only applicable if 'embedder_model' is set to 'custom'."
    infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    f0_file_description = "Full path to an external F0 file (.f0). This allows you to use pre-computed pitch values for the input audio."
    infer_parser.add_argument(
        "--f0_file",
        type=str,
        help=f0_file_description,
        default=None,
    )
    formant_shifting_description = "Apply formant shifting to the input audio. This can help adjust the timbre of the voice."
    infer_parser.add_argument(
        "--formant_shifting",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=formant_shifting_description,
        default=False,
    )
    formant_qfrency_description = "Control the frequency of the formant shifting effect. Higher values result in a more pronounced effect."
    infer_parser.add_argument(
        "--formant_qfrency",
        type=float,
        help=formant_qfrency_description,
        default=1.0,
    )
    formant_timbre_description = "Control the timbre of the formant shifting effect. Higher values result in a more pronounced effect."
    infer_parser.add_argument(
        "--formant_timbre",
        type=float,
        help=formant_timbre_description,
        default=1.0,
    )
    sid_description = "Speaker ID for multi-speaker models."
    infer_parser.add_argument(
        "--sid",
        type=int,
        help=sid_description,
        default=0,
    )
    # Parser for 'batch_infer' mode
    batch_infer_parser = subparsers.add_parser(
        "batch_infer",
        help="Run batch inference",
    )
    batch_infer_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    batch_infer_parser.add_argument(
        "--filter_radius",
        type=int,
        help=filter_radius_description,
        choices=range(11),
        default=3,
    )
    batch_infer_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=0.3,
    )
    batch_infer_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[i / 100.0 for i in range(0, 101)],
        default=1,
    )
    batch_infer_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[i / 1000.0 for i in range(0, 501)],
        default=0.33,
    )
    batch_infer_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
        ],
        default="rmvpe",
    )
    batch_infer_parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input audio files.",
        required=True,
    )
    batch_infer_parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder for saving output audio files.",
        required=True,
    )
    batch_infer_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    batch_infer_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    batch_infer_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    batch_infer_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    batch_infer_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    batch_infer_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin_v1",
            "spin_v2",
            "custom",
        ],
        default="contentvec",
    )
    batch_infer_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    batch_infer_parser.add_argument(
        "--f0_file",
        type=str,
        help=f0_file_description,
        default=None,
    )
    batch_infer_parser.add_argument(
        "--formant_shifting",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=formant_shifting_description,
        default=False,
    )
    batch_infer_parser.add_argument(
        "--formant_qfrency",
        type=float,
        help=formant_qfrency_description,
        default=1.0,
    )
    batch_infer_parser.add_argument(
        "--formant_timbre",
        type=float,
        help=formant_timbre_description,
        default=1.0,
    )
    batch_infer_parser.add_argument(
        "--sid",
        type=int,
        help=sid_description,
        default=0,
    )
    # Parser for 'tts' mode
    tts_parser = subparsers.add_parser("tts", help="Run TTS inference")
    tts_parser.add_argument(
        "--tts_file", type=str, help="File with a text to be synthesized", required=True
    )
    tts_parser.add_argument(
        "--tts_text", type=str, help="Text to be synthesized", required=True
    )
    tts_parser.add_argument(
        "--tts_voice",
        type=str,
        help="Voice to be used for TTS synthesis.",
        choices=locales,
        required=True,
    )
    tts_parser.add_argument(
        "--tts_rate",
        type=int,
        help="Control the speaking rate of the TTS. Values range from -100 (slower) to 100 (faster).",
        choices=range(-100, 101),
        default=0,
    )
    tts_parser.add_argument(
        "--pitch",
        type=int,
        help=pitch_description,
        choices=range(-24, 25),
        default=0,
    )
    tts_parser.add_argument(
        "--filter_radius",
        type=int,
        help=filter_radius_description,
        choices=range(11),
        default=3,
    )
    tts_parser.add_argument(
        "--index_rate",
        type=float,
        help=index_rate_description,
        choices=[(i / 10) for i in range(11)],
        default=0.3,
    )
    tts_parser.add_argument(
        "--volume_envelope",
        type=float,
        help=volume_envelope_description,
        choices=[(i / 10) for i in range(11)],
        default=1,
    )
    tts_parser.add_argument(
        "--protect",
        type=float,
        help=protect_description,
        choices=[(i / 10) for i in range(6)],
        default=0.33,
    )
    tts_parser.add_argument(
        "--f0_method",
        type=str,
        help=f0_method_description,
        choices=[
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
        ],
        default="rmvpe",
    )
    tts_parser.add_argument(
        "--output_tts_path",
        type=str,
        help="Full path to save the synthesized TTS audio.",
        required=True,
    )
    tts_parser.add_argument(
        "--output_rvc_path",
        type=str,
        help="Full path to save the voice-converted audio using the synthesized TTS.",
        required=True,
    )
    tts_parser.add_argument(
        "--pth_path", type=str, help=pth_path_description, required=True
    )
    tts_parser.add_argument(
        "--index_path", type=str, help=index_path_description, required=True
    )
    tts_parser.add_argument(
        "--split_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=split_audio_description,
        default=False,
    )
    tts_parser.add_argument(
        "--f0_autotune",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=f0_autotune_description,
        default=False,
    )
    tts_parser.add_argument(
        "--f0_autotune_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=1.0,
    )
    tts_parser.add_argument(
        "--clean_audio",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=clean_audio_description,
        default=False,
    )
    tts_parser.add_argument(
        "--clean_strength",
        type=float,
        help=clean_strength_description,
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    tts_parser.add_argument(
        "--export_format",
        type=str,
        help=export_format_description,
        choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
        default="WAV",
    )
    tts_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin_v1",
            "spin_v2",
            "custom",
        ],
        default="contentvec",
    )
    tts_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    tts_parser.add_argument(
        "--f0_file",
        type=str,
        help=f0_file_description,
        default=None,
    )

    # Parser for 'preprocess' mode
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess a dataset for training."
    )
    preprocess_parser.add_argument(
        "--model_name", type=str, help="Name of the model to be trained.", required=True
    )
    preprocess_parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset directory.", required=True
    )
    preprocess_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Target sampling rate for the audio data.",
        choices=get_all_vocoder_sample_rates(),
        required=True,
    )
    preprocess_parser.add_argument(
        "--cpu_threads",
        type=int,
        help="Number of CPU threads to use for preprocessing.",
        choices=range(1, min(cpu_count(), 192) + 1),
        default=DEFAULT_CPU_THREADS
    )
    preprocess_parser.add_argument(
        "--cut_preprocess",
        type=str,
        choices=["Skip", "Simple", "Automatic"],
        help="Cut the dataset into smaller segments for faster preprocessing.",
        default="Simple",
    )
    preprocess_parser.add_argument(
        "--process_effects",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable high-pass filtering during preprocessing.",
        default=True,
    )
    preprocess_parser.add_argument(
        "--noise_reduction",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable noise reduction during preprocessing.",
        default=False,
    )
    preprocess_parser.add_argument(
        "--noise_reduction_strength",
        type=float,
        help="Strength of the noise reduction filter.",
        choices=[(i / 10) for i in range(11)],
        default=0.7,
    )
    preprocess_parser.add_argument(
        "--chunk_len",
        type=float,
        help="Chunk length.",
        choices=[i * 0.5 for i in range(1, 11)],
        default=3.0,
    )
    preprocess_parser.add_argument(
        "--overlap_len",
        type=float,
        help="Overlap length.",
        choices=[0.0, 0.1, 0.2, 0.3, 0.4],
        default=0.3,
    )
    preprocess_parser.add_argument(
        "--normalization_mode",
        type=str,
        help="Normalization mode.",
        choices=["none", "post_peak", "post_peak_rvc", "post_rms"],
        default="post_rms",
    )
    preprocess_parser.add_argument(
        "--loading_resampling",
        type=str,
        help="Librosa's using SoXr, FFmpeg's using Windowed Sinc filter with Blackman-Nuttall window.",
        choices=["librosa", "ffmpeg"],
        default="librosa",
    )
    preprocess_parser.add_argument(
        "--use_smart_cutter",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable SmartCutter silence-truncation during preprocessing.",
        default=False,
    )
    # Parser for 'extract' mode
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from a dataset."
    )
    extract_parser.add_argument(
        "--model_name", type=str, help="Name of the model.", required=True
    )
    extract_parser.add_argument(
        "--f0_method",
        type=str,
        help="Pitch extraction method to use.",
        choices=[
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
        ],
        default="rmvpe",
    )
    extract_parser.add_argument(
        "--cpu_threads",
        type=int,
        help="Number of CPU threads to use for feature extraction (optional).",
        choices=range(1, min(cpu_count(), 192) + 1),
        default=DEFAULT_CPU_THREADS,
    )
    extract_parser.add_argument(
        "--gpu",
        type=str,
        help="GPU device to use for feature extraction (optional).",
        default="-",
    )
    extract_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Target sampling rate for the audio data.",
        choices=get_all_vocoder_sample_rates(),
        required=True,
    )
    extract_parser.add_argument(
        "--vocoder_arch",
        type=str,
        help="Choose the vocoder architecture",
        choices=get_vocoder_cli_choices(),
        default="hifi",
    )
    extract_parser.add_argument(
        "--embedder_model",
        type=str,
        help=embedder_model_description,
        choices=[
            "contentvec",
            "spin_v1",
            "spin_v2",
            "custom",
        ],
        default="contentvec",
    )
    extract_parser.add_argument(
        "--embedder_model_custom",
        type=str,
        help=embedder_model_custom_description,
        default=None,
    )
    extract_parser.add_argument(
        "--include_mutes",
        type=int,
        help="Number of silent files to include.",
        choices=range(0, 11),
        default=2,
    )

    # Parser for 'train' mode
    train_parser = subparsers.add_parser("train", help="Train an RVC model.")
    train_parser.add_argument(
        "--model_name", type=str, help="Name of the model to be trained.", required=True
    )
    train_parser.add_argument(
        "--vocoder",
        type=str,
        help="Vocoder name",
        choices=get_vocoder_cli_choices(),
        default="hifi",
    )
    train_parser.add_argument(
        "--optimizer_choice",
        type=str,
        choices=["AdamW", "AdaBelief", "RAdam", "Ranger21", "Sched-Free AdamW", "Sched-Free RAdam"],
        help="Choose the optimizer used for Generator and Discriminator.",
        default="AdamW",
    )
    train_parser.add_argument(
        "--use_checkpointing",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enables usage of checkpointing.",
        default=False,
    )
    train_parser.add_argument(
        "--compile_vocoder",
        "--compile_chouwagan",
        dest="compile_vocoder",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help=VOCODER_COMPILE_CLI_HELP,
        default=False,
    )
    train_parser.add_argument(
        "--torch_compile_mode",
        type=str,
        choices=TORCH_COMPILE_MODES,
        help=TORCH_COMPILE_MODE_CLI_HELP,
        default="default",
    )
    train_parser.add_argument(
        "--custom_lr_g",
        type=float,
        help="Custom learning rate for generator.",
        default=1e-4,
    )
    train_parser.add_argument(
        "--custom_lr_d",
        type=float,
        help="Custom learning rate for discriminator.",
        default=1e-4,
    )
    train_parser.add_argument(
        "--use_2_sample_kl",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="uses 2 samples to calculate KL Loss.",
        default=False,
    )
    train_parser.add_argument(
        "--use_best_step",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Tracks the step with lowest FM+Mel loss each epoch and uses those weights for eval preview and model extraction.",
        default=False,
    )
    train_parser.add_argument(
        "--double_d_updates",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Runs the discriminator backward/update step twice per batch. Gives D more gradient signal on small datasets.",
        default=False,
    )
    train_parser.add_argument(
        "--epoch_save_frequency",
        type=int,
        help="Save the model every specified number of epochs.",
        choices=range(1, 101),
        required=True,
    )
    train_parser.add_argument(
        "--save_only_latest_net_models",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Save only the latest G/D files.",
        default=False,
    )
    train_parser.add_argument(
        "--save_weight_models",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Save model weights every epoch.",
        default=True,
    )
    train_parser.add_argument(
        "--total_epoch_count",
        type=int,
        help="Total number of epochs to train for.",
        choices=range(1, 10001),
        default=1000,
    )
    train_parser.add_argument(
        "--sample_rate",
        type=int,
        help="Sampling rate of the training data.",
        choices=get_all_vocoder_sample_rates(),
        required=True,
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training.",
        choices=range(1, 51),
        default=8,
    )
    train_parser.add_argument(
        "--gpu",
        type=str,
        help="GPU device to use for training (e.g., '0').",
        default="0",
    )
    train_parser.add_argument(
        "--pretrained",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Use a pretrained model for initialization.",
        default=True,
    )
    train_parser.add_argument(
        "--custom_pretrained",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Use a custom pretrained model.",
        default=False,
    )
    train_parser.add_argument(
        "--g_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained generator model file.",
    )
    train_parser.add_argument(
        "--d_pretrained_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the pretrained discriminator model file.",
    )
    train_parser.add_argument(
        "--use_warmup",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enables usage of warmup.",
        default=False,
    )
    train_parser.add_argument(
        "--warmup_duration",
        type=int,
        help="Duration of warmup phase (in epochs).",
        choices=range(1, 1000),
        default=10,
    )
    train_parser.add_argument(
        "--use_tf32",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Lets you choose between FP32 and TF32 precision used in training.",
        default=False,
    )
    train_parser.add_argument(
        "--use_benchmark",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enable cuDNN benchmark mode for potential speedup.",
        default=True,
    )
    train_parser.add_argument(
        "--use_deterministic",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Toggle deterministic mode for reproducibility at possible performance cost.",
        default=False,
    )
    train_parser.add_argument(
        "--spectral_loss",
        type=str,
        choices=["L1 Mel Loss", "Multi-Scale Mel Loss", "Hybrid L1"],
        help="Available types of spectral loss functions. ",
        default="L1 Mel Loss",
    )
    train_parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["exp decay step", "exp decay epoch", "cosine annealing", "none"],
        help="Pick the shared LR scheduler for generator and discriminator.",
        default="exp decay step",
    )
    train_parser.add_argument(
        "--exp_decay_gamma",
        type=str,
        choices=["0.9999996", "0.999875", "0.999", "0.9975", "0.995"],
        help="Gamma for the shared exponential decay scheduler.",
        default="0.999875",
    )
    train_parser.add_argument(
        "--use_kl_annealing",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Whether you wanna use kl annealing.",
        default=False,
    )
    train_parser.add_argument(
        "--kl_annealing_cycle_duration",
        type=int,
        help="Duration of kl annealing phase (in epochs).",
        default=3,
    )
    train_parser.add_argument(
        "--rolling_loss_steps",
        type=int,
        help="interval for rolling avg loss (in steps).",
        default=50,
    )
    train_parser.add_argument(
        "--grad_clip_scheduling",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Whether you wanna enable grads clipping scheduling",
        default=False,
    )
    train_parser.add_argument(
        "--grad_clip_steps_duration",
        type=int,
        help="Duration ( in steps ) for grads clipping",
        default=0,
    )
    train_parser.add_argument(
        "--grad_clip_value_g_cap",
        type=int,
        help="Specify clipping value for Generator's clip_grad_norm",
        default=0,
    )
    train_parser.add_argument(
        "--grad_clip_value_d_cap",
        type=int,
        help="Specify clipping value for Discriminator's clip_grad_norm",
        default=0,
    )
    train_parser.add_argument(
        "--grad_clip_value_g_release",
        type=int,
        help="Specify what kind of clipping value you want after the scheduling, for G. Set to 0 to leave unconstrained.",
        default=0,
    )
    train_parser.add_argument(
        "--grad_clip_value_d_release",
        type=int,
        help="Specify what kind of clipping value you want after the scheduling, for D. Set to 0 to leave unconstrained.",
        default=0,
    )
    train_parser.add_argument(
        "--use_custom_lr",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Enables customization of learning rate for Generator and Discriminator.",
        default=False,
    )
    train_parser.add_argument(
        "--cleanup",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        help="Cleanup previous training attempt.",
        default=False,
    )
    train_parser.add_argument(
        "--index_algorithm",
        type=str,
        choices=["Auto", "Faiss", "KMeans"],
        help="Choose the method for generating the index file.",
        default="Auto",
    )

    # Parser for 'index' mode
    index_parser = subparsers.add_parser(
        "index", help="Generate an index file for an RVC model."
    )
    index_parser.add_argument(
        "--model_name", type=str, help="Name of the model.", required=True
    )
    index_parser.add_argument(
        "--index_algorithm",
        type=str,
        choices=["Auto", "Faiss", "KMeans"],
        help="Choose the method for generating the index file.",
        default="Auto",
    )

    # Parser for 'model_information' mode
    model_information_parser = subparsers.add_parser(
        "model_information", help="Display information about a trained model."
    )
    model_information_parser.add_argument(
        "--pth_path", type=str, help="Path to the .pth model file.", required=True
    )

    # Parser for 'model_blender' mode
    model_blender_parser = subparsers.add_parser(
        "model_blender", help="Fuse two RVC models together."
    )
    model_blender_parser.add_argument(
        "--model_name", type=str, help="Name of the new fused model.", required=True
    )
    model_blender_parser.add_argument(
        "--pth_path_1",
        type=str,
        help="Path to the first .pth model file.",
        required=True,
    )
    model_blender_parser.add_argument(
        "--pth_path_2",
        type=str,
        help="Path to the second .pth model file.",
        required=True,
    )
    model_blender_parser.add_argument(
        "--ratio",
        type=float,
        help="Ratio for blending the two models (0.0 to 1.0).",
        choices=[(i / 10) for i in range(11)],
        default=0.5,
    )

    # Parser for 'tensorboard' mode
    subparsers.add_parser(
        "tensorboard", help="Launch TensorBoard for monitoring training progress."
    )

    # Parser for 'download' mode
    download_parser = subparsers.add_parser(
        "download", help="Download a model from a provided link."
    )
    download_parser.add_argument(
        "--model_link", type=str, help="Direct link to the model file.", required=True
    )

    # Parser for 'prerequisites' mode
    prerequisites_parser = subparsers.add_parser(
        "prerequisites", help="Install prerequisites for RVC."
    )
    prerequisites_parser.add_argument(
        "--pretraineds_hifigan",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download pretrained models for RVC v2.",
    )
    prerequisites_parser.add_argument(
        "--models",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download additional models.",
    )
    prerequisites_parser.add_argument(
        "--exe",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download required executables.",
    )
    prerequisites_parser.add_argument(
        "--smartcutter",
        type=lambda x: bool(strtobool(x)),
        choices=[True, False],
        default=True,
        help="Download required SmartCutter models.",
    )
    # Parser for 'audio_analyzer' mode
    audio_analyzer = subparsers.add_parser(
        "audio_analyzer", help="Analyze an audio file."
    )
    audio_analyzer.add_argument(
        "--input_path", type=str, help="Path to the input audio file.", required=True
    )

    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        print("Please run the script with '-h' for more information.")
        sys.exit(1)

    args = parse_arguments()

    try:
        if args.mode == "infer":
            run_infer_script(
                pitch=args.pitch,
                filter_radius=args.filter_radius,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                input_path=args.input_path,
                output_path=args.output_path,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                f0_file=args.f0_file,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                formant_shifting=args.formant_shifting,
                formant_qfrency=args.formant_qfrency,
                formant_timbre=args.formant_timbre,
                sid=args.sid,
            )
        elif args.mode == "batch_infer":
            run_batch_infer_script(
                pitch=args.pitch,
                filter_radius=args.filter_radius,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                f0_file=args.f0_file,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                formant_shifting=args.formant_shifting,
                formant_qfrency=args.formant_qfrency,
                formant_timbre=args.formant_timbre,
                sid=args.sid,
            )
        elif args.mode == "tts":
            run_tts_script(
                tts_file=args.tts_file,
                tts_text=args.tts_text,
                tts_voice=args.tts_voice,
                tts_rate=args.tts_rate,
                pitch=args.pitch,
                filter_radius=args.filter_radius,
                index_rate=args.index_rate,
                volume_envelope=args.volume_envelope,
                protect=args.protect,
                f0_method=args.f0_method,
                output_tts_path=args.output_tts_path,
                output_rvc_path=args.output_rvc_path,
                pth_path=args.pth_path,
                index_path=args.index_path,
                split_audio=args.split_audio,
                f0_autotune=args.f0_autotune,
                f0_autotune_strength=args.f0_autotune_strength,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                export_format=args.export_format,
                f0_file=args.f0_file,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
            )
        elif args.mode == "preprocess":
            run_preprocess_script(
                model_name=args.model_name,
                dataset_path=args.dataset_path,
                sample_rate=args.sample_rate,
                cpu_threads=args.cpu_threads,
                cut_preprocess=args.cut_preprocess,
                process_effects=args.process_effects,
                noise_reduction=args.noise_reduction,
                clean_strength=args.noise_reduction_strength,
                chunk_len=args.chunk_len,
                overlap_len=args.overlap_len,
                normalization_mode=args.normalization_mode,
                loading_resampling=args.loading_resampling,
                use_smart_cutter=args.use_smart_cutter,
            )
        elif args.mode == "extract":
            run_extract_script(
                model_name=args.model_name,
                f0_method=args.f0_method,
                cpu_threads=args.cpu_threads,
                gpu=args.gpu,
                sample_rate=args.sample_rate,
                vocoder_arch=args.vocoder_arch,
                embedder_model=args.embedder_model,
                embedder_model_custom=args.embedder_model_custom,
                include_mutes=args.include_mutes,
            )
        elif args.mode == "train":
            run_train_script(
                model_name=args.model_name,
                epoch_save_frequency=args.epoch_save_frequency,
                save_only_latest_net_models=args.save_only_latest_net_models,
                save_weight_models=args.save_weight_models,
                total_epoch_count=args.total_epoch_count,
                sample_rate=args.sample_rate,
                batch_size=args.batch_size,
                gpu=args.gpu,
                use_warmup=args.use_warmup,
                warmup_duration=args.warmup_duration,
                pretrained=args.pretrained,
                cleanup=args.cleanup,
                index_algorithm=args.index_algorithm,
                custom_pretrained=args.custom_pretrained,
                g_pretrained_path=args.g_pretrained_path,
                d_pretrained_path=args.d_pretrained_path,
                vocoder=args.vocoder,
                optimizer_choice=args.optimizer_choice,
                use_checkpointing=args.use_checkpointing,
                compile_vocoder=args.compile_vocoder,
                torch_compile_mode=args.torch_compile_mode,
                use_tf32=args.use_tf32,
                use_benchmark=args.use_benchmark,
                use_deterministic=args.use_deterministic,
                spectral_loss=args.spectral_loss,
                lr_scheduler=args.lr_scheduler,
                exp_decay_gamma=args.exp_decay_gamma,
                rolling_loss_steps=args.rolling_loss_steps,
                grad_clip_scheduling=args.grad_clip_scheduling,
                grad_clip_steps_duration=args.grad_clip_steps_duration,
                grad_clip_value_g_cap=args.grad_clip_value_g_cap,
                grad_clip_value_d_cap=args.grad_clip_value_d_cap,
                grad_clip_value_g_release=args.grad_clip_value_g_release,
                grad_clip_value_d_release=args.grad_clip_value_d_release,
                use_custom_lr=args.use_custom_lr,
                custom_lr_g=args.custom_lr_g,
                custom_lr_d=args.custom_lr_d,
                use_2_sample_kl=args.use_2_sample_kl,
                use_best_step=args.use_best_step,
                double_d_updates=args.double_d_updates,
            )
        elif args.mode == "index":
            run_index_script(
                model_name=args.model_name,
                index_algorithm=args.index_algorithm,
            )
        elif args.mode == "model_information":
            run_model_information_script(
                pth_path=args.pth_path,
            )
        elif args.mode == "model_blender":
            run_model_blender_script(
                model_name=args.model_name,
                pth_path_1=args.pth_path_1,
                pth_path_2=args.pth_path_2,
                ratio=args.ratio,
            )
        elif args.mode == "tensorboard":
            run_tensorboard_script()
        elif args.mode == "download":
            run_download_script(
                model_link=args.model_link,
            )
        elif args.mode == "prerequisites":
            run_prerequisites_script(
                pretraineds_hifigan=args.pretraineds_hifigan,
                models=args.models,
                exe=args.exe,
                smartcutter=args.smartcutter,
            )
        elif args.mode == "audio_analyzer":
            run_audio_analyzer_script(
                input_path=args.input_path,
            )
    except Exception as error:
        print(f"An error occurred during execution: {error}")

        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
