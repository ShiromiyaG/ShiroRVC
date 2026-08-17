import os
import signal

process_pids = []

import shutil
import sys
import json
from multiprocessing import cpu_count

import gradio as gr

from core import (
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    run_prerequisites_script,
    run_train_script,
    stop_train_script,
    early_save_stop,
)
from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu, microarchitecture_capability_checker, check_if_fp16
from rvc.configs.vocoders import (
    get_vocoder_choices,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.text import format_title
from rvc.lib.terminal import DEFAULT_CPU_THREADS
from tabs.train.descs import *

now_dir = os.getcwd()
sys.path.append(now_dir)

supported_audio_ext = { "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3", }

saved_components = [] # List of components that should have their states saved ~ For presets


# Custom Pretraineds
pretraineds_custom_path = os.path.join(now_dir, "rvc", "models", "pretraineds", "custom")
pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)
# Custom embedders
custom_embedder_root = os.path.join(now_dir, "rvc", "models", "embedders", "embedders_custom")
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
# Training presets
presets_path = os.path.join(now_dir, 'assets', 'training_presets')
presets_path_relative = os.path.relpath(presets_path, now_dir)

# Ensure dirs existence
os.makedirs(pretraineds_custom_path_relative, exist_ok=True)
os.makedirs(custom_embedder_root, exist_ok=True)
os.makedirs(presets_path, exist_ok=True)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]

pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")

def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )

datasets_path = os.path.join(now_dir, "assets", "datasets")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

datasets_path_relative = os.path.relpath(datasets_path, now_dir)
DATASET_METADATA_NAME = ".rvc_dataset.json"

def get_datasets_list():
    dataset_roots = set()
    for dirpath, _, filenames in os.walk(datasets_path_relative):
        if DATASET_METADATA_NAME in filenames:
            dataset_roots.add(os.path.normcase(os.path.abspath(dirpath)))

    datasets = []
    for dirpath, _, filenames in os.walk(datasets_path_relative):
        absolute_dir = os.path.normcase(os.path.abspath(dirpath))
        nested_in_dataset = any(
            absolute_dir != dataset_root
            and os.path.commonpath([absolute_dir, dataset_root]) == dataset_root
            for dataset_root in dataset_roots
        )
        if nested_in_dataset:
            continue
        has_audio = any(
            filename.lower().endswith(tuple(supported_audio_ext))
            for filename in filenames
        )
        if has_audio or DATASET_METADATA_NAME in filenames:
            datasets.append(dirpath)
    return sorted(set(datasets))

def refresh_datasets():
    return {"choices": sorted(get_datasets_list()), "__type__": "update"}

# Model Names
models_path = os.path.join(now_dir, "logs")

def get_models_list():
    return [
        os.path.basename(dirpath)
        for dirpath in os.listdir(models_path)
        if os.path.isdir(os.path.join(models_path, dirpath))
        and all(excluded not in dirpath for excluded in ["zips", "mute", "reference"])
    ]

def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}

# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )

# Refresh Custom Embedders
def get_embedder_custom_list():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]

def refresh_custom_embedder_list():
    return {"choices": sorted(get_embedder_custom_list()), "__type__": "update"}

# Retrieve presets
def get_presets_list():
    return [os.path.splitext(s)[0] for s in os.listdir(presets_path) if s.endswith('.json')]

# Drop Model
def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info("Invalid pretrained file.")
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info("Refresh the list to use the uploaded pretrained file.")
    return None

# Drop Dataset
def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info("Enter a valid dataset name.")
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in supported_audio_ext:
            gr.Info("Invalid audio file.")
        else:
            dataset_name = format_title(dataset_name)
            audio_file = format_title(os.path.basename(dropbox))
            dataset_path = os.path.join(now_dir, "assets", "datasets", dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            destination_path = os.path.join(dataset_path, audio_file)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            shutil.copy(dropbox, destination_path)
            gr.Info(
                "Audio added. Run preprocessing when ready."
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path

# Drop Custom Embedder
def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)
    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))

    if config_file:
        shutil.copy(config_file, os.path.join(target_folder, os.path.basename(config_file)))

    return f"Files moved to folder {target_folder}"

def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

# Export
def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]

def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )

# Export Pth and Index Files
def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        print(f"Attempted to export invalid pth path: {pth_path}")
        return None

def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        print(f"Attempted to export invalid index path: {index_path}")
        return None

# Upload to Google Drive
def upload_to_google_drive(pth_path, index_path):
    def upload_file(file_path):
        if file_path:
            try:
                gr.Info(f"Uploading {pth_path} to Google Drive...")
                google_drive_folder = "/content/drive/MyDrive/Shiromiya-RVC-Fork-Exported"
                if not os.path.exists(google_drive_folder):
                    os.makedirs(google_drive_folder)
                google_drive_file_path = os.path.join(
                    google_drive_folder, os.path.basename(file_path)
                )
                if os.path.exists(google_drive_file_path):
                    os.remove(google_drive_file_path)
                shutil.copy2(file_path, google_drive_file_path)
                gr.Info("File uploaded successfully.")
            except Exception as error:
                print(f"An error occurred uploading to Google Drive: {error}")
                gr.Info("Error uploading to Google Drive")

    upload_file(pth_path)
    upload_file(index_path)

# Enable checkpointing for gpus with memory 
def auto_enable_checkpointing():
    try:
        return max_vram_gpu(0) < 6
    except:
        return False

# Init state for certain options.
initial_sample_rate_choices = [
    str(rate) for rate in get_vocoder_sample_rates("hifi")
]
initial_sample_rate = "48000"


def update_vocoder_settings(vocoder_id, current_sample_rate, use_smartcutter):
    vocoder_id = normalize_vocoder(vocoder_id)
    spec = get_vocoder_spec(vocoder_id)
    sample_rate_choices = [
        str(rate) for rate in get_vocoder_sample_rates(vocoder_id)
    ]
    current_sample_rate = str(current_sample_rate)
    selected_sample_rate = (
        current_sample_rate
        if current_sample_rate in sample_rate_choices
        else sample_rate_choices[0]
    )
    smartcutter_enabled = bool(spec.get("supports_smartcutter", False))
    return (
        {
            "choices": sample_rate_choices,
            "value": selected_sample_rate,
            "__type__": "update",
        },
        {
            "interactive": smartcutter_enabled,
            "value": use_smartcutter if smartcutter_enabled else False,
            "__type__": "update",
        },
    )

# Microarch. dependent features, options, functionalities etc.. Might expand in future.
fp16_check = None

initial_optimizer = "AdamW"
initial_optimizer_choices = [("AdamW", "AdamW"), ("AdaBelief", "AdaBelief"), ("RAdam", "RAdam"), ("Ranger21", "Ranger21"), ("Sched-Free AdamW", "Sched-Free AdamW"), ("Sched-Free RAdam", "Sched-Free RAdam")]
fp16_check = True

# FP16 checker
if fp16_check:
    if check_if_fp16():
        initial_optimizer = "AdamW"
        initial_optimizer_choices = [("AdamW", "AdamW"), ("AdaBelief", "AdaBelief"), ("RAdam", "RAdam"), ("Ranger21", "Ranger21"), ("Sched-Free AdamW", "Sched-Free AdamW"), ("Sched-Free RAdam", "Sched-Free RAdam")]


# Train Tab
def train_tab():
    # Training presets section
    with gr.Accordion("Training Presets", open=False):
        with gr.Row():
            refresh_presets_button = gr.Button("Refresh Presets")
        with gr.Row():
            with gr.Column():
                preset_dropdown = gr.Dropdown(
                    choices=get_presets_list(),
                    label="Preset Name",
                    allow_custom_value=True,
                    interactive=True
                )
            with gr.Column():
                save_preset_button = gr.Button("Save to preset")
                load_preset_button = gr.Button("Load from preset")

    # Model settings section
    with gr.Accordion("Model Settings"):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="Model Name",
                    info="Name of the new model.",
                    choices=get_models_list(),
                    value="example-model-name",
                    interactive=True,
                    allow_custom_value=True,
                    key='model_name'
                )
                optimizer_choice = gr.Radio(
                    label="Optimizer (G/D)",
                    info=OPTIMIZER_INFO,
                    choices=initial_optimizer_choices,
                    value=initial_optimizer,
                    interactive=True,
                    visible=True,
                    key='optimizer_choice'
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label="Sampling Rate",
                    info="Target sample rate. Match it to the dataset when possible.",
                    choices=initial_sample_rate_choices,
                    value=initial_sample_rate,
                    interactive=True,
                    key='sampling_rate'
                )
                vocoder = gr.Radio(
                    label="Vocoder",
                    info=VOCODER_INFO_RVC,
                    choices=get_vocoder_choices(),
                    value="hifi",
                    interactive=True,
                    visible=True,
                    key='vocoder'
                )
        with gr.Accordion(
            "CPU / GPU settings for ' f0 ' and ' features ' extraction.",
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_threads = gr.Slider(
                        1,
                        min(cpu_count(), 192),  # max 192 parallel processes
                        DEFAULT_CPU_THREADS,
                        step=1,
                        label="CPU Threads",
                        info="CPU threads used during extraction.",
                        interactive=True,
                        key='cpu_threads'
                    )
                with gr.Column():
                    extract_gpu = gr.Textbox(
                        label="GPU ID",
                        info="GPU IDs for extraction, separated by '-'.",
                        placeholder="0 to ∞ separated by -",
                        value=str(get_number_of_gpus()),
                        interactive=True,
                        key='extract_gpu'
                    )
                    gr.Textbox(
                        label="GPU Information",
                        info="Detected GPU information.",
                        value=get_gpu_info(),
                        interactive=False,
                    )

    # Preprocess section
    with gr.Accordion("Preprocessing"):
        dataset_path = gr.Dropdown(
            label="Dataset Path",
            info="Folder containing the training audio.",
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
            key='dataset_path'
        )
        refresh = gr.Button("Refresh")

        with gr.Accordion("Advanced Settings for the preprocessing step", open=True):
            gr.Markdown()
            with gr.Row(elem_classes=["rvc-preprocess-options"]):
                with gr.Column(min_width=0):
                    dataset_format = gr.Radio(
                        label="Dataset Format",
                        info=DATASET_FORMAT_INFO,
                        choices=["WAV", "FLAC"],
                        value="WAV",
                        interactive=True,
                        key='dataset_format'
                    )
                with gr.Column(min_width=0):
                    loading_resampling = gr.Radio(
                        label="Resampling & Loading Handler",
                        info=RESAMPLER_INFO,
                        choices=["librosa", "ffmpeg"],
                        value="librosa",
                        interactive=True,
                        key='loading_resampling'
                    )
                with gr.Column(min_width=0):
                    use_smart_cutter = gr.Checkbox(
                        label="SmartCutter",
                        info=SMARTCUTTER_INFO,
                        value=False,
                        interactive=True,
                        visible=True,
                        key='use_smart_cutter'
                    )
                with gr.Column(min_width=0):
                    normalization_mode = gr.Radio(
                        label="Loudness Normalization",
                        info=NORMALIZATION_INFO,
                        choices=["none", "post_peak", "post_peak_rvc", "post_rms"],
                        value="post_peak",
                        interactive=True,
                        visible=True,
                        key='normalization_mode'
                    )
            with gr.Row():
                rms_norm_db = gr.Slider(
                    -24.0, -3.0, -18.0, step=1.0,
                    label="RMS Target (dBFS)",
                    info=PREPROCESS_RMS_VALUE_INFO,
                    interactive=True,
                    visible=False,
                    key='rms_norm_db'
                )
            with gr.Row():
                cut_preprocess = gr.Radio(
                    label="Audio cutting",
                    info=AUDIO_FILE_SLICING_INFO,
                    choices=["Skip", "Simple", "Automatic"],
                    value="Simple",
                    interactive=True,
                    key='cut_preprocess'
                )
                chunk_len = gr.Slider(
                    0.5,
                    30.0,
                    3.0,
                    step=0.1,
                    label="Chunk length (sec)",
                    info="Chunk length for Simple cutting.",
                    interactive=True,
                    scale=46,
                    key='chunk_len'
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.42,
                    0.36,
                    step=0.01,
                    label="Overlap length",
                    info="Overlap between Simple chunks, in seconds.",
                    interactive=True,
                    scale=57,
                    key='overlap_len'
                )
            with gr.Column():
                process_effects = gr.Checkbox(
                    label="DC / high-pass filtering",
                    info="Remove DC offset and low-frequency noise.",
                    value=True,
                    interactive=True,
                    visible=True,
                    key='process_effects'
                )
            with gr.Column():
                noise_reduction = gr.Checkbox(
                    label="Noise Reduction",
                    info="Apply spectral-gating noise reduction.",
                    value=False,
                    interactive=True,
                    visible=True,
                    key='noise_reduction'
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Noise Reduction Strength",
                    info="Higher values apply stronger cleanup.",
                    visible=False,
                    value=0.5,
                    interactive=True,
                    key='clean_strength'
                )
        preprocess_output_info = gr.Textbox(
            label="Output Information",
            info="Preprocessing status.",
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button("Preprocess Dataset")
            preprocess_button.click(
                fn=run_preprocess_script,
                inputs=[
                    model_name,
                    dataset_path,
                    sampling_rate,
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
                outputs=[preprocess_output_info],
            )

    # Extract section
    with gr.Accordion("Extraction"):
        with gr.Row():
            f0_method = gr.Radio(
                label="Pitch extraction algorithm",
                info=PITCH_EXTRACTION_INFO,
                choices=["crepe", "crepe-tiny", "rmvpe", "fcpe"],
                value="rmvpe",
                interactive=True,
                key='f0_method'
            )

            embedder_model = gr.Radio(
                label="Embedder Model",
                info="Model used for speaker features.",
                choices=[
                    "contentvec",
                    "spin_v1",
                    "spin_v2",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
                key='embedder_model'
            )
        include_mutes = gr.Slider(
            0,
            10,
            2,
            step=1,
            label="Silent ( 'mute' ) files for training.",
            info="Add silent examples so the model can reproduce silence.",
            value=True,
            interactive=True,
            key='include_mutes'
        )
        with gr.Row(visible=False) as embedder_custom:
            with gr.Accordion("Custom Embedder", open=True):
                with gr.Row():
                    embedder_model_custom = gr.Dropdown(
                        label="Select Custom Embedder",
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                        key='embedder_model_custom'
                    )
                    refresh_embedders_button = gr.Button("Refresh embedders")
                folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                with gr.Row():
                    bin_file_upload = gr.File(
                        label="Upload .bin", type="filepath", interactive=True
                    )
                    config_file_upload = gr.File(
                        label="Upload .json", type="filepath", interactive=True
                    )
                move_files_button = gr.Button("Move files to custom embedder")

        extract_output_info = gr.Textbox(
            label="Output Information",
            info="Extraction status.",
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button("Extract Features")
        extract_button.click(
            fn=run_extract_script,
            inputs=[
                model_name,
                f0_method,
                cpu_threads,
                extract_gpu,
                sampling_rate,
                vocoder,
                embedder_model,
                embedder_model_custom,
                include_mutes,
            ],
            outputs=[extract_output_info],
        )

    # Training section
    with gr.Accordion("Training"):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                128,
                max_vram_gpu(0),
                step=1,
                label="Batch Size",
                info=BATCH_SIZE_INFO,
                interactive=True,
                key='batch_size'
            )
            epoch_save_frequency = gr.Slider(
                1,
                100,
                1,
                step=1,
                label="Saving frequency",
                info="Save a checkpoint every N epochs.",
                interactive=True,
                key='epoch_save_frequency'
            )
            total_epoch_count = gr.Slider(
                1,
                10000,
                500,
                step=1,
                label="Total Epochs",
                info="Total training epochs.",
                interactive=True,
                key='total_epoch_count'
            )
        with gr.Accordion("Advanced Settings for training", open=False):
            with gr.Row():
                with gr.Column(scale=9):
                    save_only_latest_net_models = gr.Checkbox(
                        label="Save Only Latest G/D",
                        info="Keep only the latest generator and discriminator checkpoints.",
                        value=True,
                        interactive=True,
                        key='save_only_latest_net_models'
                    )
                    save_weight_models = gr.Checkbox(
                        label="Save weight models",
                        info="Save the compact voice model files.",
                        value=True,
                        interactive=True,
                        key='save_weight_models'
                    )
                    pretrained = gr.Checkbox(
                        label="Pretrained",
                        info="Use pretrained weights for fine-tuning.",
                        value=True,
                        interactive=True,
                        key='pretrained'
                    )
                    cleanup = gr.Checkbox(
                        label="Fresh Training",
                        info="Clear previous weights and logs before training.",
                        value=False,
                        interactive=True,
                        key='cleanup'
                    )
                    use_checkpointing = gr.Checkbox(
                        label="Checkpointing",
                        info="Reduce VRAM use at the cost of speed.",
                        value=auto_enable_checkpointing,
                        interactive=True,
                        key='use_checkpointing'
                    )
                    compile_vocoder = gr.Checkbox(
                        label=VOCODER_COMPILE_LABEL,
                        info=VOCODER_COMPILE_INFO,
                        value=False,
                        interactive=True,
                        key='compile_vocoder'
                    )
                    torch_compile_mode = gr.Radio(
                        label=TORCH_COMPILE_MODE_LABEL,
                        info=TORCH_COMPILE_MODE_INFO,
                        choices=TORCH_COMPILE_MODE_CHOICES,
                        value=TORCH_COMPILE_MODE_CHOICES[0],
                        interactive=True,
                        visible=False,
                        key='torch_compile_mode'
                    )
                    use_tf32 = gr.Checkbox(
                        label="use 'TF32' precision",
                        info="Use TF32 on supported GPUs for faster training.",
                        value=microarchitecture_capability_checker(),
                        interactive=microarchitecture_capability_checker(),
                        key='use_tf32'
                    )
                    use_benchmark = gr.Checkbox(
                        label="Use 'cuDNN benchmark' mode",
                        info="Enable cuDNN benchmark mode.",
                        value=True,
                        interactive=True,
                        key='use_benchmark'
                    )
                    use_deterministic = gr.Checkbox(
                        label="Use 'cuDNN deterministic' mode",
                        info="Improve reproducibility, possibly reducing speed.",
                        value=False,
                        interactive=True,
                        key='use_deterministic'
                    )
                with gr.Column(scale=7):
                    rolling_loss_steps = gr.Slider(
                        3,
                        1000,
                        50,
                        step=1,
                        label="Rolling avg loss steps",
                        info="Steps between rolling loss and gradient logs.",
                        interactive=True,
                        visible=True,
                        key='rolling_loss_steps'
                    )
                    grad_clip_scheduling = gr.Checkbox(
                        label="Grad clipping scheduling",
                        info="Schedule gradient clipping values.",
                        value=False,
                        interactive=True,
                        key='grad_clip_scheduling'
                    )
                    grad_clip_steps_duration = gr.Number(
                        label="Clipping duration",
                        info="Initial clipping duration, in steps.",
                        value=0,
                        interactive=True,
                        visible=False,
                        key='grad_clip_steps_duration'
                    )
                    grad_clip_value_g_cap = gr.Number(
                        label="G grads initial clip",
                        info="Initial generator gradient cap.",
                        value=0,
                        interactive=True,
                        visible=False,
                        key='grad_clip_value_g_cap'
                    )
                    grad_clip_value_d_cap = gr.Number(
                        label="D grads initial clip",
                        info="Initial discriminator gradient cap.",
                        value=0,
                        interactive=True,
                        visible=False,
                        key='grad_clip_value_d_cap'
                    )
                    grad_clip_value_g_release = gr.Number(
                        label="G grads secondary clip",
                        info="Generator gradient cap after the schedule.",
                        value=0,
                        interactive=True,
                        visible=False,
                        key='grad_clip_value_g_release'
                    )
                    grad_clip_value_d_release = gr.Number(
                        label="D grads secondary clip",
                        info="Discriminator gradient cap after the schedule.",
                        value=0,
                        interactive=True,
                        visible=False,
                        key='grad_clip_value_d_release'
                    )
                with gr.Column(scale=9):
                    spectral_loss = gr.Radio(
                        label="Spectral loss",
                        info=SPECTRAL_LOSS_INFO,
                        choices=["L1 Mel Loss", "Multi-Scale Mel Loss", "Hybrid L1"],
                        value="L1 Mel Loss",
                        interactive=True,
                        key='spectral_loss'
                    )
                    lr_scheduler = gr.Radio(
                        label="LR scheduler (G/D)",
                        info=LR_SCHEDULER_INFO,
                        choices=["exp decay step", "exp decay epoch", "cosine annealing", "none"],
                        value="exp decay epoch",
                        interactive=True,
                        key='lr_scheduler'
                    )
                    exp_decay_gamma = gr.Radio(
                        label="Exp decay gamma (G/D)",
                        info="Decay factor for exponential scheduling.",
                        choices=["0.9999996", "0.999875", "0.999", "0.9975", "0.995"],
                        value="0.999875",
                        interactive=True,
                        visible=True,
                        key='exp_decay_gamma'
                    )
                    use_kl_annealing = gr.Checkbox(
                        label="KL loss annealing",
                        info=KL_ANNEALING_INFO,
                        value=False,
                        interactive=True,
                        key='use_kl_annealing'
                    )
                    kl_annealing_cycle_duration = gr.Slider(
                        1,
                        100,
                        3,
                        step=1,
                        label="KL annealing cycle duration",
                        info=KL_ANNEALING_CYCLE_INFO,
                        interactive=True,
                        visible=False,
                        key='kl_annealing_cycle_duration'
                    )
                    use_2_sample_kl = gr.Checkbox(
                        label="Use 2-sample KL",
                        info="Use two samples for KL loss. Experimental.",
                        value=False,
                        interactive=True,
                        key='use_2_sample_kl'
                    )
                    use_best_step = gr.Checkbox(
                        label=BEST_STEP_LABEL,
                        info=BEST_STEP_INFO,
                        value=False,
                        interactive=True,
                        key='use_best_step'
                    )
                    double_d_updates = gr.Checkbox(
                        label="Double Discriminator Update",
                        info="Update the discriminator twice per batch.",
                        value=False,
                        interactive=True,
                        key='double_d_updates'
                    )
            with gr.Column():
                custom_pretrained = gr.Checkbox(
                    label="Custom Pretrained",
                    info="Use custom generator and discriminator pretrained files.",
                    value=False,
                    interactive=True,
                    key='custom_pretrained'
                )
                with gr.Column(visible=False) as pretrained_custom_settings:
                        with gr.Accordion("Pretrained Custom Settings"):
                            upload_pretrained = gr.File(
                                label="Upload Pretrained Model",
                                type="filepath",
                                interactive=True,
                            )
                            refresh_custom_pretaineds_button = gr.Button("Refresh Custom Pretraineds")
                            g_pretrained_path = gr.Dropdown(
                                label="Custom Pretrained G",
                                info="Generator pretrained file.",
                                choices=sorted(pretraineds_list_g),
                                interactive=True,
                                allow_custom_value=True,
                                key='g_pretrained_path'
                            )
                            d_pretrained_path = gr.Dropdown(
                                label="Custom Pretrained D",
                                info="Discriminator pretrained file.",
                                choices=sorted(pretraineds_list_d),
                                interactive=True,
                                allow_custom_value=True,
                                key='d_pretrained_path'
                            )
                multiple_gpu = gr.Checkbox(
                    label="GPU Settings",
                    info=(
                        "Enable multi-GPU training.",
                    ),
                    value=False,
                    interactive=True,
                    key='multiple_gpu'
                )
                with gr.Column(visible=False) as gpu_custom_settings:
                    with gr.Accordion("GPU ID override / Multi-gpu-training configuration"):
                        training_gpu = gr.Textbox(
                            label="GPU Number",
                            info="GPU IDs for training, separated by '-'.",
                            placeholder="0 to ∞ separated by -",
                            value=str(get_number_of_gpus()),
                            interactive=True,
                            key="training_gpu"
                        )
                        gr.Textbox(
                            label="GPU Information",
                            info="Detected GPU information.",
                            value=get_gpu_info(),
                            interactive=False,
                        )
                use_warmup = gr.Checkbox(
                    label="Warmup phase for training",
                    info="Use linear learning-rate warmup.",
                    value=False,
                    interactive=True,
                    key='use_warmup'
                )
                with gr.Column(visible=False) as warmup_settings:
                    with gr.Accordion("Warmup settings"):
                        warmup_duration = gr.Slider(
                            1,
                            100,
                            5,
                            step=1,
                            label="Duration of the warmup phase",
                            info="Warmup duration, in epochs.",
                            interactive=True,
                            key='warmup_duration'
                        )

                use_custom_lr = gr.Checkbox(
                    label="Custom lr for gen and disc",
                    info="Set separate generator and discriminator learning rates.",
                    value=False,
                    interactive=True,
                    key='use_custom_lr'
                )
                with gr.Column(visible=False) as custom_lr_settings:
                    with gr.Accordion("Custom lr settings"):
                        custom_lr_g = gr.Textbox(
                            label="Learning rate for Generator",
                            placeholder="Default is 1e-4 / 0.0001",
                            info="Generator learning rate. Both rates are required.",
                            interactive=True,
                            key='custom_lr_g'
                        )
                        custom_lr_d = gr.Textbox(
                            label="Learning rate for Discriminator",
                            placeholder="Default is 1e-4 / 0.0001",
                            info="Discriminator learning rate. Both rates are required.",
                            interactive=True,
                            key='custom_lr_d'
                        )

                with gr.Row():
                    index_algorithm = gr.Radio(
                    label="Index Algorithm",
                    info="Index method for large datasets.",
                    choices=["Auto", "Faiss", "KMeans"],
                    value="Auto",
                    interactive=True,
                    key='index_algorithm'
                )

        train_output_info = gr.Textbox(
            label="Output Information",
            info="Training status.",
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            train_button = gr.Button("Start Training")
            train_button.click(
                fn=run_train_script,
                inputs=[
                    model_name,
                    epoch_save_frequency,
                    save_only_latest_net_models,
                    save_weight_models,
                    total_epoch_count,
                    sampling_rate,
                    batch_size,
                    training_gpu,
                    use_warmup,
                    warmup_duration,
                    pretrained,
                    cleanup,
                    index_algorithm,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                    vocoder,
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
                outputs=[train_output_info],
            )

            stop_train_button = gr.Button("Stop Training", visible=True)
            stop_train_button.click(
                fn=stop_train_script,
                inputs=[],
                outputs=[train_output_info],
            )

            early_stop_button = gr.Button("Early Stopping", visible=True)
            early_stop_button.click(
                fn=early_save_stop,
                inputs=[],
                outputs=[train_output_info],
            )

            index_button = gr.Button("Generate Index")
            index_button.click(
                fn=run_index_script,
                inputs=[model_name, index_algorithm],
                outputs=[train_output_info],
            )

    # Export Model section
    with gr.Accordion("Export Model", open=False):
        if not os.name == "nt":
            gr.Markdown(
                "Upload is available on Google Colab and saves exported files to your Google Drive."
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label="Exported Pth file",
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label="Pth file",
                info="PTH file to export.",
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label="Exported Index File",
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label="Index File",
                info="Index file to export.",
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button("Refresh")
                if not os.name == "nt":
                    upload_exported = gr.Button("Upload")
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return gr.update(visible=bool(checkbox))

            def toggle_compile_mode(compile_enabled):
                return gr.update(visible=bool(compile_enabled))

            def toggle_visible_gamma(lr_scheduler):
                return gr.update(
                    visible=lr_scheduler in ["exp decay step", "exp decay epoch"]
                )

            def download_prerequisites():
                    gr.Info(
                        "Checking for prerequisites with pitch guidance... Missing files will be downloaded. If you already have them, this step will be skipped."
                    )
                    run_prerequisites_script(
                        pretraineds_hifigan=True,
                        models=False,
                        exe=False,
                    )
                    gr.Info(
                        "Prerequisites check complete. Missing files were downloaded, and you may now start preprocessing."
                    )

            def toggle_visible_embedder_custom(embedder_model):
                return gr.update(visible=embedder_model == "custom")

            def update_noise_reduce_slider_visibility(noise_reduction):
                return gr.update(visible=bool(noise_reduction))

            def toggle_rms_norm_slider(norm_mode):
                return gr.update(visible=norm_mode == "post_rms")

            saved_components.extend([
                # Model settings
                optimizer_choice, vocoder, sampling_rate, cpu_threads, extract_gpu,

                # Preprocessing
                dataset_path, dataset_format, loading_resampling, use_smart_cutter,
                normalization_mode, rms_norm_db, cut_preprocess, chunk_len, overlap_len,
                process_effects, noise_reduction, clean_strength,

                # Feature extract
                f0_method, embedder_model, include_mutes,
                embedder_model_custom,

                # Training
                batch_size, epoch_save_frequency, total_epoch_count,
                save_only_latest_net_models, save_weight_models, pretrained,
                cleanup, use_checkpointing, compile_vocoder, torch_compile_mode,
                use_tf32, use_benchmark, use_deterministic, spectral_loss,
                lr_scheduler, exp_decay_gamma,
                custom_pretrained, g_pretrained_path,
                d_pretrained_path, multiple_gpu, training_gpu, use_warmup,
                warmup_duration, use_custom_lr, custom_lr_g,
                custom_lr_d, use_kl_annealing, kl_annealing_cycle_duration,
                rolling_loss_steps, grad_clip_scheduling, grad_clip_steps_duration,
                grad_clip_value_g_cap, grad_clip_value_d_cap, grad_clip_value_g_release,
                grad_clip_value_d_release, index_algorithm, use_2_sample_kl, use_best_step,
                double_d_updates
            ])

            def save_training_preset(inputs):
                settings = {}
                for component in saved_components:
                    settings[component.key] = inputs[component]

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, inputs[preset_dropdown] + '.json')))

                if not preset_path.startswith(presets_path):
                    raise gr.Error(f"Invalid training preset name: {inputs[preset_dropdown]}", duration=5)

                with open(preset_path, 'w', encoding='utf-8') as of:
                    json.dump(settings, of, indent=4, ensure_ascii=False)

            def load_training_preset(preset_name):
                if preset_name not in get_presets_list():
                    raise gr.Error(f'Preset does not exist: {preset_name}')

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, preset_name + '.json')))

                with open(preset_path, 'r', encoding='utf-8') as ifile:
                    settings = json.loads(ifile.read())

                return [
                    (
                        settings[component.key]
                        if component.key in settings
                        else settings.get("compile_chouwagan", gr.skip())
                        if component.key == "compile_vocoder"
                        else gr.skip()
                    )
                    for component in saved_components
                ]

            refresh_presets_button.click(
                fn=lambda: gr.Dropdown(choices=get_presets_list()), 
                outputs=[preset_dropdown]
            )

            save_preset_button.click(
                fn=save_training_preset,
                inputs=set(saved_components) | {preset_dropdown}
            ).then(
                fn=lambda: gr.Dropdown(choices=get_presets_list()), 
                outputs=[preset_dropdown]
            )

            load_preset_button.click(
                fn=load_training_preset,
                inputs=[preset_dropdown],
                outputs=saved_components
            ).then(  # update twice so components depending on "change" events get updated
                fn=load_training_preset,
                inputs=[preset_dropdown],
                outputs=saved_components
            )

            noise_reduction.change(
                fn=update_noise_reduce_slider_visibility,
                inputs=noise_reduction,
                outputs=clean_strength,
            )
            normalization_mode.change(
                fn=toggle_rms_norm_slider,
                inputs=normalization_mode,
                outputs=rms_norm_db,
            )
            sampling_rate.change(
                fn=lambda sr: {
                    "48000": 0.36,
                    "40000": 0.38,
                    "32000": 0.40,
                    "44100": 0.37,
                }.get(sr, 0.36),
                inputs=[sampling_rate],
                outputs=[overlap_len],
            )
            vocoder.change(
                fn=update_vocoder_settings,
                inputs=[vocoder, sampling_rate, use_smart_cutter],
                outputs=[sampling_rate, use_smart_cutter],
            )
            refresh.click(
                fn=refresh_models_and_datasets,
                inputs=[],
                outputs=[model_name, dataset_path],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            embedder_model.change(
                fn=toggle_visible_embedder_custom,
                inputs=[embedder_model],
                outputs=[embedder_custom],
            )
            move_files_button.click(
                fn=create_folder_and_move_files,
                inputs=[folder_name_input, bin_file_upload, config_file_upload],
                outputs=[],
            )
            refresh_embedders_button.click(
                fn=refresh_embedders_folders, inputs=[], outputs=[embedder_model_custom]
            )
            pretrained.change(
                fn=lambda pretrained_val, custom_val: (
                    gr.update(visible=bool(pretrained_val)),
                    gr.update(visible=bool(pretrained_val and custom_val)),
                ),
                inputs=[pretrained, custom_pretrained],
                outputs=[custom_pretrained, pretrained_custom_settings],
            )
            # placeholder_trigger.change(
                # fn=lambda value: {"visible": not value, "__type__": "update"},
                # inputs=[placeholder_trigger], # element to be unchecked / disabled
                # outputs=[placeholder_result] # element to appear to appear
            # )
            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
            )
            refresh_custom_pretaineds_button.click(
                fn=refresh_custom_pretraineds,
                inputs=[],
                outputs=[g_pretrained_path, d_pretrained_path],
            )
            upload_pretrained.upload(
                fn=save_drop_model,
                inputs=[upload_pretrained],
                outputs=[upload_pretrained],
            )
            use_warmup.change(
                fn=toggle_visible,
                inputs=[use_warmup],
                outputs=[warmup_settings],
            )
            compile_vocoder.change(
                fn=toggle_compile_mode,
                inputs=[compile_vocoder],
                outputs=[torch_compile_mode],
            )
            use_custom_lr.change(
                fn=toggle_visible,
                inputs=[use_custom_lr],
                outputs=[custom_lr_settings],
            )
            use_kl_annealing.change(
                fn=lambda v: gr.update(visible=bool(v)),
                inputs=[use_kl_annealing],
                outputs=[kl_annealing_cycle_duration]
            )
            grad_clip_scheduling.change(
                fn=lambda v: [gr.update(visible=bool(v)) for _ in range(5)],
                inputs=[grad_clip_scheduling],
                outputs=[grad_clip_steps_duration, grad_clip_value_g_cap, grad_clip_value_d_cap, grad_clip_value_g_release, grad_clip_value_d_release]
            )
            lr_scheduler.change(
                fn=toggle_visible_gamma,
                inputs=[lr_scheduler],
                outputs=[exp_decay_gamma],
            )
            multiple_gpu.change(
                fn=toggle_visible,
                inputs=[multiple_gpu],
                outputs=[gpu_custom_settings],
            )
            pth_dropdown_export.change(
                fn=export_pth,
                inputs=[pth_dropdown_export],
                outputs=[pth_file_export],
            )
            index_dropdown_export.change(
                fn=export_index,
                inputs=[index_dropdown_export],
                outputs=[index_file_export],
            )
            refresh_export.click(
                fn=refresh_pth_and_index_list,
                inputs=[],
                outputs=[pth_dropdown_export, index_dropdown_export],
            )
