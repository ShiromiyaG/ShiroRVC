import click
import os
import sys

import atexit
import subprocess


from functools import lru_cache

from rvc.lib.paths import LOGS_DIR, ROOT
from rvc.lib.process import (
    describe_exit_code,
    run_stage,
    spawn_trainer,
    stop_trainer,
)
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

    tts_script_path = os.path.join(ROOT, "rvc", "lib", "extras", "tts.py")

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
    run_stage(command_tts, "Text to speech")
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
    preprocess_script_path = os.path.join(ROOT, "rvc", "train", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(LOGS_DIR, model_name),
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
    run_stage(command, "Preprocessing")
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

    model_path = os.path.join(LOGS_DIR, model_name)
    extract = os.path.join(ROOT, "rvc", "train", "extract", "extract.py")

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

    run_stage(command_1, "Extraction")

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
        os.path.join(LOGS_DIR, model_name, "run_spec.json")
    )

    train_script_path = os.path.join(ROOT, "rvc", "train", "train.py")
    command = [python, train_script_path, str(spec_path)]
    training_process = spawn_trainer(command)
    training_process.wait()
    return f"Training has been successfully completed or stopped."


# Stopping the training
def stop_train_script():
    return stop_trainer(training_process)


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

    return available_speakers(os.path.join(LOGS_DIR, model_name))


def run_index_script(
    model_name: str, index_algorithm: str, index_metric: str = "l2",
    index_speaker: int | str | None = None,
):
    index_script_path = os.path.join(ROOT, "rvc", "train", "process", "extract_index.py")
    # "all" travels as a word rather than as an empty argument, which a shell
    # can drop and which would otherwise be read back as speaker 0.
    speaker = "all" if index_speaker in (None, "", "all") else str(int(index_speaker))
    command = [
        python,
        index_script_path,
        os.path.join(LOGS_DIR, model_name),
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
