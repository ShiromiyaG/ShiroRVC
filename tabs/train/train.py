import os
import signal

from rvc.lib.i18n import _

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
)
from rvc.configs.config import (
    get_gpu_info,
    get_number_of_gpus,
    get_use_fp16,
    max_vram_gpu,
    microarchitecture_capability_checker,
)
from rvc.configs.vocoders import (
    get_vocoder_choices,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.text import format_title
from rvc.lib.terminal import (
    DEFAULT_CPU_THREADS,
    error as print_error,
    warning,
)
from rvc.lib.model_bundle import walk_models
from tabs.train.descs import *

now_dir = os.getcwd()
sys.path.append(now_dir)

supported_audio_ext = { "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3", }

saved_components = []  # components whose state is saved/restored by presets


pretraineds_custom_path = os.path.join(now_dir, "rvc", "models", "pretraineds", "custom")
pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)
custom_embedder_root = os.path.join(now_dir, "rvc", "models", "embedders", "embedders_custom")
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
presets_path = os.path.join(now_dir, 'assets', 'training_presets')
presets_path_relative = os.path.relpath(presets_path, now_dir)

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

def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )

def get_embedder_custom_list():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]

def refresh_custom_embedder_list():
    return {"choices": sorted(get_embedder_custom_list()), "__type__": "update"}

def get_presets_list():
    return [os.path.splitext(s)[0] for s in os.listdir(presets_path) if s.endswith('.json')]

def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(_("Invalid pretrained file."))
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info(_("Refresh the list to use the uploaded pretrained file."))
    return None

def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info(_("Enter a valid dataset name."))
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in supported_audio_ext:
            gr.Info(_("Invalid audio file."))
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
                _("Audio added. Run preprocessing when ready.")
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path

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

def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in walk_models(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]

def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in walk_models(models_path)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )

def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        warning(f"Not a valid .pth path, skipping: {pth_path}", tag="[EXPORT]")
        return None

def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        warning(f"Not a valid index path, skipping: {index_path}", tag="[EXPORT]")
        return None

def upload_to_google_drive(pth_path, index_path):
    def upload_file(file_path):
        if file_path:
            try:
                gr.Info(_("Uploading {path} to Google Drive...").format(path=pth_path))
                google_drive_folder = "/content/drive/MyDrive/Shiromiya-RVC-Fork-Exported"
                if not os.path.exists(google_drive_folder):
                    os.makedirs(google_drive_folder)
                google_drive_file_path = os.path.join(
                    google_drive_folder, os.path.basename(file_path)
                )
                if os.path.exists(google_drive_file_path):
                    os.remove(google_drive_file_path)
                shutil.copy2(file_path, google_drive_file_path)
                gr.Info(_("File uploaded successfully."))
            except Exception as error:
                print_error(f"Upload to Google Drive failed: {error}", tag="[EXPORT]")
                gr.Info(_("Error uploading to Google Drive"))

    upload_file(pth_path)
    upload_file(index_path)

def auto_enable_checkpointing():
    try:
        return max_vram_gpu(0) < 6
    except:
        return False

def start_train_from_ui(
    model_name,
    epoch_save_frequency,
    save_only_latest_net_models,
    save_weight_models,
    total_epoch_count,
    sample_rate,
    batch_size,
    gpu,
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
    lr_scheduler,
    use_custom_lr,
    custom_lr_g,
    custom_lr_d,
    compile_vocoder,
    torch_compile_mode,
    overtrain_detector,
    stop_on_overtrain,
    use_ema,
):
    """Launch a run from this tab's controls.

    Gradio binds ``inputs`` to parameters *by position*, so calling
    ``run_train_script`` directly makes the order of that list part of the
    launcher's signature -- inserting one argument in ``core.py`` silently
    shifts every later flag.  Naming the parameters here and calling with
    keywords confines the positional coupling to this one function.

    ``use_fp16`` is not among the controls: it is a machine-level setting under
    Settings -> Precision, read at launch so it still lands in the run spec.
    """
    return run_train_script(
        model_name=model_name,
        epoch_save_frequency=epoch_save_frequency,
        save_only_latest_net_models=save_only_latest_net_models,
        save_weight_models=save_weight_models,
        total_epoch_count=total_epoch_count,
        sample_rate=sample_rate,
        batch_size=batch_size,
        gpu=gpu,
        use_warmup=use_warmup,
        warmup_duration=warmup_duration,
        pretrained=pretrained,
        cleanup=cleanup,
        index_algorithm=index_algorithm,
        custom_pretrained=custom_pretrained,
        g_pretrained_path=g_pretrained_path,
        d_pretrained_path=d_pretrained_path,
        vocoder=vocoder,
        optimizer_choice=optimizer_choice,
        use_checkpointing=use_checkpointing,
        use_tf32=use_tf32,
        use_fp16=get_use_fp16(),
        use_benchmark=use_benchmark,
        lr_scheduler=lr_scheduler,
        use_custom_lr=use_custom_lr,
        custom_lr_g=custom_lr_g,
        custom_lr_d=custom_lr_d,
        compile_vocoder=compile_vocoder,
        torch_compile_mode=torch_compile_mode,
        overtrain_detector=overtrain_detector,
        stop_on_overtrain=stop_on_overtrain,
        use_ema=use_ema,
    )


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

initial_optimizer = "AdamW"
# Mirrors rvc.train.optimizers.OPTIMIZER_CHOICES.
initial_optimizer_choices = [
    ("AdamW", "AdamW"),
    ("Sched-Free AdamW", "Sched-Free AdamW"),
    ("Muon", "Muon"),
    ("Lion", "Lion"),
]

def train_tab():
    with gr.Accordion(_("Training Presets"), open=False):
        with gr.Row():
            refresh_presets_button = gr.Button(_("Refresh Presets"))
        with gr.Row():
            with gr.Column():
                preset_dropdown = gr.Dropdown(
                    choices=get_presets_list(),
                    label=_("Preset Name"),
                    allow_custom_value=True,
                    interactive=True
                )
            with gr.Column():
                save_preset_button = gr.Button(_("Save to preset"))
                load_preset_button = gr.Button(_("Load from preset"))

    with gr.Accordion(_("Model Settings")):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label=_("Model Name"),
                    info=_("Name of the new model."),
                    choices=get_models_list(),
                    value="example-model-name",
                    interactive=True,
                    allow_custom_value=True,
                    key='model_name'
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label=_("Sampling Rate"),
                    info=_("Target sample rate. Match it to the dataset when possible."),
                    choices=initial_sample_rate_choices,
                    value=initial_sample_rate,
                    interactive=True,
                    key='sampling_rate'
                )
                vocoder = gr.Radio(
                    label=_("Vocoder"),
                    info=_(VOCODER_INFO_RVC),
                    choices=get_vocoder_choices(),
                    value="hifi",
                    interactive=True,
                    visible=True,
                    key='vocoder'
                )
        with gr.Accordion(
            _("CPU / GPU settings for ' f0 ' and ' features ' extraction."),
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_threads = gr.Slider(
                        1,
                        min(cpu_count(), 192),  # max 192 parallel processes
                        DEFAULT_CPU_THREADS,
                        step=1,
                        label=_("CPU Threads"),
                        info=_("CPU threads used during extraction."),
                        interactive=True,
                        key='cpu_threads'
                    )
                with gr.Column():
                    extract_gpu = gr.Textbox(
                        label=_("GPU ID"),
                        info=_("GPU IDs for extraction, separated by '-'."),
                        placeholder=_("0 to ∞ separated by -"),
                        value=str(get_number_of_gpus()),
                        interactive=True,
                        key='extract_gpu'
                    )
                    gr.Textbox(
                        label=_("GPU Information"),
                        info=_("Detected GPU information."),
                        value=get_gpu_info(),
                        interactive=False,
                    )

    with gr.Accordion(_("Preprocessing")):
        dataset_path = gr.Dropdown(
            label=_("Dataset Path"),
            info=_("Folder containing the training audio."),
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
            key='dataset_path'
        )
        refresh = gr.Button(_("Refresh"))

        with gr.Accordion(_("Advanced Settings for the preprocessing step"), open=True):
            gr.Markdown()
            with gr.Row(elem_classes=["rvc-preprocess-options"]):
                with gr.Column(min_width=0):
                    dataset_format = gr.Radio(
                        label=_("Dataset Format"),
                        info=_(DATASET_FORMAT_INFO),
                        choices=["WAV", "FLAC"],
                        value="WAV",
                        interactive=True,
                        key='dataset_format'
                    )
                with gr.Column(min_width=0):
                    loading_resampling = gr.Radio(
                        label=_("Resampling & Loading Handler"),
                        info=_(RESAMPLER_INFO),
                        choices=["librosa", "ffmpeg"],
                        value="librosa",
                        interactive=True,
                        key='loading_resampling'
                    )
                with gr.Column(min_width=0):
                    use_smart_cutter = gr.Checkbox(
                        label=_("SmartCutter"),
                        info=_(SMARTCUTTER_INFO),
                        value=False,
                        interactive=True,
                        visible=True,
                        key='use_smart_cutter'
                    )
                with gr.Column(min_width=0):
                    normalization_mode = gr.Radio(
                        label=_("Loudness Normalization"),
                        info=_(NORMALIZATION_INFO),
                        choices=["none", "post_peak", "post_peak_rvc", "post_rms"],
                        value="post_peak",
                        interactive=True,
                        visible=True,
                        key='normalization_mode'
                    )
            with gr.Row():
                rms_norm_db = gr.Slider(
                    -24.0, -3.0, -18.0, step=1.0,
                    label=_("RMS Target (dBFS)"),
                    info=_(PREPROCESS_RMS_VALUE_INFO),
                    interactive=True,
                    visible=False,
                    key='rms_norm_db'
                )
            with gr.Row():
                cut_preprocess = gr.Radio(
                    label=_("Audio cutting"),
                    info=_(AUDIO_FILE_SLICING_INFO),
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
                    label=_("Chunk length (sec)"),
                    info=_("Chunk length for Simple cutting."),
                    interactive=True,
                    scale=46,
                    key='chunk_len'
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.42,
                    0.36,
                    step=0.01,
                    label=_("Overlap length"),
                    info=_("Overlap between Simple chunks, in seconds."),
                    interactive=True,
                    scale=57,
                    key='overlap_len'
                )
            with gr.Column():
                process_effects = gr.Checkbox(
                    label=_("DC / high-pass filtering"),
                    info=_("Remove DC offset and low-frequency noise."),
                    value=True,
                    interactive=True,
                    visible=True,
                    key='process_effects'
                )
            with gr.Column():
                noise_reduction = gr.Checkbox(
                    label=_("Noise Reduction"),
                    info=_("Apply spectral-gating noise reduction."),
                    value=False,
                    interactive=True,
                    visible=True,
                    key='noise_reduction'
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Noise Reduction Strength"),
                    info=_("Higher values apply stronger cleanup."),
                    visible=False,
                    value=0.5,
                    interactive=True,
                    key='clean_strength'
                )
        preprocess_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Preprocessing status."),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button(_("Preprocess Dataset"))
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

    with gr.Accordion(_("Extraction")):
        with gr.Row():
            f0_method = gr.Radio(
                label=_("Pitch extraction algorithm"),
                info=_(PITCH_EXTRACTION_INFO),
                choices=["crepe", "crepe-tiny", "rmvpe", "fcpe"],
                value="rmvpe",
                interactive=True,
                key='f0_method'
            )

            embedder_model = gr.Radio(
                label=_("Embedder Model"),
                info=_("Model used for speaker features."),
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
            label=_("Silent ( 'mute' ) files for training."),
            info=_("Add silent examples so the model can reproduce silence."),
            interactive=True,
            key='include_mutes'
        )
        remove_16k_slices = gr.Checkbox(
            label=_("Delete 16 kHz slices after extraction"),
            info=(
                _("The 16 kHz copies only feed pitch and embedder extraction; training "
                "never reads them. Deleting frees roughly a third of what preprocessing "
                "wrote. Re-extracting with another f0 method or embedder needs them "
                "back, which means running preprocessing again.")
            ),
            value=False,
            interactive=True,
            key='remove_16k_slices'
        )
        feature_precision = gr.Radio(
            label=_("Feature Precision"),
            info=(
                _("How the extracted embeddings are stored. These are both the "
                "training input and what the retrieval index is built from, and "
                "fp16 puts a quantisation floor under every index vector while "
                "inference queries with fp32 ones. fp32 doubles the feature "
                "cache on disk (a 2 h dataset goes from roughly 550 MB to "
                "1.1 GB); fp16 halves it. Either can be read back without "
                "re-extracting.")
            ),
            choices=["fp32", "fp16"],
            value="fp32",
            interactive=True,
            key="feature_precision",
        )
        with gr.Row(visible=False) as embedder_custom:
            with gr.Accordion(_("Custom Embedder"), open=True):
                with gr.Row():
                    embedder_model_custom = gr.Dropdown(
                        label=_("Select Custom Embedder"),
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                        key='embedder_model_custom'
                    )
                    refresh_embedders_button = gr.Button(_("Refresh embedders"))
                folder_name_input = gr.Textbox(label=_("Folder Name"), interactive=True)
                with gr.Row():
                    bin_file_upload = gr.File(
                        label=_("Upload .bin"), type="filepath", interactive=True
                    )
                    config_file_upload = gr.File(
                        label=_("Upload .json"), type="filepath", interactive=True
                    )
                move_files_button = gr.Button(_("Move files to custom embedder"))

        extract_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Extraction status."),
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button(_("Extract Features"))
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
                remove_16k_slices,
                feature_precision,
            ],
            outputs=[extract_output_info],
        )

    with gr.Accordion(_("Training")):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                128,
                max_vram_gpu(0),
                step=1,
                label=_("Batch Size"),
                info=_(BATCH_SIZE_INFO),
                interactive=True,
                key='batch_size'
            )
            epoch_save_frequency = gr.Slider(
                1,
                100,
                1,
                step=1,
                label=_("Saving frequency"),
                info=_("Save a checkpoint every N epochs."),
                interactive=True,
                key='epoch_save_frequency'
            )
            total_epoch_count = gr.Slider(
                1,
                10000,
                500,
                step=1,
                label=_("Total Epochs"),
                info=_("Total training epochs."),
                interactive=True,
                key='total_epoch_count'
            )
        with gr.Accordion(_("Advanced Settings for training"), open=False):
            # Grouped by what the setting decides, not by widget type.
            gr.Markdown(f"#### {_('Starting point')}")
            with gr.Row():
                with gr.Column(min_width=0):
                    pretrained = gr.Checkbox(
                        label=_("Pretrained"),
                        info=_("Use pretrained weights for fine-tuning."),
                        value=True,
                        interactive=True,
                        key='pretrained'
                    )
                with gr.Column(min_width=0):
                    custom_pretrained = gr.Checkbox(
                        label=_("Custom Pretrained"),
                        info=_("Use custom generator and discriminator pretrained files."),
                        value=False,
                        interactive=True,
                        key='custom_pretrained'
                    )
                with gr.Column(min_width=0):
                    cleanup = gr.Checkbox(
                        label=_("Fresh Training"),
                        info=_("Clear previous weights and logs before training."),
                        value=False,
                        interactive=True,
                        key='cleanup'
                    )
            with gr.Column(visible=False) as pretrained_custom_settings:
                with gr.Accordion(_("Pretrained Custom Settings")):
                    upload_pretrained = gr.File(
                        label=_("Upload Pretrained Model"),
                        type="filepath",
                        interactive=True,
                    )
                    refresh_custom_pretaineds_button = gr.Button(_("Refresh Custom Pretraineds"))
                    g_pretrained_path = gr.Dropdown(
                        label=_("Custom Pretrained G"),
                        info=_("Generator pretrained file."),
                        choices=sorted(pretraineds_list_g),
                        interactive=True,
                        allow_custom_value=True,
                        key='g_pretrained_path'
                    )
                    d_pretrained_path = gr.Dropdown(
                        label=_("Custom Pretrained D"),
                        info=_("Discriminator pretrained file."),
                        choices=sorted(pretraineds_list_d),
                        interactive=True,
                        allow_custom_value=True,
                        key='d_pretrained_path'
                    )

            gr.Markdown(f"#### {_('Optimisation')}")
            with gr.Row():
                with gr.Column(min_width=0):
                    optimizer_choice = gr.Radio(
                        label=_("Optimizer (G/D)"),
                        info=_(OPTIMIZER_INFO),
                        choices=initial_optimizer_choices,
                        value=initial_optimizer,
                        interactive=True,
                        visible=True,
                        key='optimizer_choice'
                    )
                with gr.Column(min_width=0):
                    lr_scheduler = gr.Radio(
                        label=_("LR scheduler (G/D)"),
                        info=_(LR_SCHEDULER_INFO),
                        choices=["exp decay step", "exp decay epoch", "cosine annealing", "none"],
                        value="exp decay epoch",
                        interactive=True,
                        key='lr_scheduler'
                    )
            with gr.Row():
                with gr.Column(min_width=0):
                    use_warmup = gr.Checkbox(
                        label=_("Warmup phase for training"),
                        info=_("Use linear learning-rate warmup."),
                        value=False,
                        interactive=True,
                        key='use_warmup'
                    )
                    with gr.Column(visible=False) as warmup_settings:
                        with gr.Accordion(_("Warmup settings")):
                            warmup_duration = gr.Slider(
                                1,
                                100,
                                5,
                                step=1,
                                label=_("Duration of the warmup phase"),
                                info=_("Warmup duration, in epochs."),
                                interactive=True,
                                key='warmup_duration'
                            )
                with gr.Column(min_width=0):
                    use_custom_lr = gr.Checkbox(
                        label=_("Custom lr for gen and disc"),
                        info=_("Set separate generator and discriminator learning rates."),
                        value=False,
                        interactive=True,
                        key='use_custom_lr'
                    )
                    with gr.Column(visible=False) as custom_lr_settings:
                        with gr.Accordion(_("Custom lr settings")):
                            custom_lr_g = gr.Textbox(
                                label=_("Learning rate for Generator"),
                                placeholder=_("Default is 1e-4 / 0.0001"),
                                info=_("Generator learning rate. Both rates are required."),
                                interactive=True,
                                key='custom_lr_g'
                            )
                            custom_lr_d = gr.Textbox(
                                label=_("Learning rate for Discriminator"),
                                placeholder=_("Default is 1e-4 / 0.0001"),
                                info=_("Discriminator learning rate. Both rates are required."),
                                interactive=True,
                                key='custom_lr_d'
                            )

            gr.Markdown(f"#### {_('Checkpoints and quality')}")
            with gr.Row():
                with gr.Column(min_width=0):
                    save_only_latest_net_models = gr.Checkbox(
                        label=_("Save Only Latest G/D"),
                        info=_("Keep only the latest generator and discriminator checkpoints."),
                        value=True,
                        interactive=True,
                        key='save_only_latest_net_models'
                    )
                    save_weight_models = gr.Checkbox(
                        label=_("Save weight models"),
                        info=_("Save the compact voice model files."),
                        value=True,
                        interactive=True,
                        key='save_weight_models'
                    )
                with gr.Column(min_width=0):
                    use_ema = gr.Checkbox(
                        label=_(USE_EMA_LABEL),
                        info=_(USE_EMA_INFO),
                        value=True,
                        interactive=True,
                        key='use_ema'
                    )
                with gr.Column(min_width=0):
                    overtrain_detector = gr.Checkbox(
                        label=_(OVERTRAIN_DETECTOR_LABEL),
                        info=_(OVERTRAIN_DETECTOR_INFO),
                        value=False,
                        interactive=True,
                        key='overtrain_detector'
                    )
                    stop_on_overtrain = gr.Checkbox(
                        label=_(STOP_ON_OVERTRAIN_LABEL),
                        info=_(STOP_ON_OVERTRAIN_INFO),
                        value=False,
                        interactive=True,
                        # Hidden with the detector off, which is the default.
                        visible=False,
                        key='stop_on_overtrain'
                    )
                    # Meaningless without the detector that produces the signal.
                    # ``interactive`` only greys a Gradio checkbox out, leaving
                    # a dead control sitting there; hide it instead.
                    overtrain_detector.change(
                        fn=lambda enabled: gr.update(visible=enabled),
                        inputs=[overtrain_detector],
                        outputs=[stop_on_overtrain],
                        # Without this the default "full" progress tracker paints
                        # a spinner on the output component, and a component that
                        # was hidden when the request started never receives the
                        # completion status -- it unhides stuck on "processing".
                        show_progress="hidden",
                    )

            gr.Markdown(f"#### {_('Performance')}")
            with gr.Row():
                with gr.Column(min_width=0):
                    use_checkpointing = gr.Checkbox(
                        label=_("Checkpointing"),
                        info=_("Reduce VRAM use at the cost of speed."),
                        value=auto_enable_checkpointing,
                        interactive=True,
                        key='use_checkpointing'
                    )
                    use_benchmark = gr.Checkbox(
                        label=_("Use 'cuDNN benchmark' mode"),
                        info=_("Enable cuDNN benchmark mode."),
                        value=True,
                        interactive=True,
                        key='use_benchmark'
                    )
                with gr.Column(min_width=0):
                    use_tf32 = gr.Checkbox(
                        label=_("use 'TF32' precision"),
                        info=_("Use TF32 on supported GPUs for faster training."),
                        value=microarchitecture_capability_checker(),
                        interactive=microarchitecture_capability_checker(),
                        key='use_tf32'
                    )
                with gr.Column(min_width=0):
                    compile_vocoder = gr.Checkbox(
                        label=_(VOCODER_COMPILE_LABEL),
                        info=_(VOCODER_COMPILE_INFO),
                        value=False,
                        interactive=True,
                        key='compile_vocoder'
                    )
                    torch_compile_mode = gr.Radio(
                        label=_(TORCH_COMPILE_MODE_LABEL),
                        info=_(TORCH_COMPILE_MODE_INFO),
                        choices=TORCH_COMPILE_MODE_CHOICES,
                        value=TORCH_COMPILE_MODE_CHOICES[0],
                        interactive=True,
                        visible=False,
                        key='torch_compile_mode'
                    )

            gr.Markdown(f"#### {_('Hardware')}")
            with gr.Row():
                with gr.Column(min_width=0):
                    multiple_gpu = gr.Checkbox(
                        label=_("GPU Settings"),
                        # Must stay a plain string: a trailing comma turning
                        # this into a tuple breaks translation and Gradio's
                        # info serialization.
                        info=_("Choose which GPUs to train on, and enable "
                               "multi-GPU training."),
                        value=False,
                        interactive=True,
                        key='multiple_gpu'
                    )
                    with gr.Column(visible=False) as gpu_custom_settings:
                        with gr.Accordion(_("GPU ID override / Multi-gpu-training configuration")):
                            training_gpu = gr.Textbox(
                                label=_("GPU Number"),
                                info=_("GPU IDs for training, separated by '-'."),
                                placeholder=_("0 to ∞ separated by -"),
                                # Despite the name this returns the ID list
                                # ("0", "0-1", ...), not a count, so it is
                                # already the right shape for this field.
                                value=str(get_number_of_gpus()),
                                interactive=True,
                                key="training_gpu"
                            )
                            gr.Textbox(
                                label=_("GPU Information"),
                                info=_("Detected GPU information."),
                                value=get_gpu_info(),
                                interactive=False,
                            )

            # -- the retrieval index -----------------------------------------
            # Last, and labelled, because these two are not training options at
            # all: they are read by the "Generate Index" button further down.
            gr.Markdown(f"#### {_('Index')}")
            gr.Markdown(
                f"<sub>{_('Used by the Generate Index button below, after training.')}</sub>"
            )
            with gr.Row():
                with gr.Column(min_width=0):
                    index_algorithm = gr.Radio(
                        label=_("Index Algorithm"),
                        info=_("Index method for large datasets."),
                        choices=["Auto", "Faiss", "KMeans"],
                        value="Auto",
                        interactive=True,
                        key='index_algorithm'
                    )
                with gr.Column(min_width=0):
                    index_metric = gr.Radio(
                        label=_("Index Similarity"),
                        info=(
                            _("How neighbours are ranked. L2 is what upstream RVC "
                            "builds. Cosine compares direction only, so a quiet and "
                            "a loud take of the same sound match equally well.")
                        ),
                        choices=["l2", "cosine"],
                        value="l2",
                        interactive=True,
                        key="index_metric",
                    )

        train_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Training status."),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            train_button = gr.Button(_("Start Training"))
            # Announce first so the box says something straight away, then run
            # the blocking call with Gradio's spinner off: a run lasts hours,
            # and an overlay counting seconds on an empty box reads as a hang.
            train_button.click(
                fn=lambda: (
                    "Training started. Epoch, step and loss progress is shown "
                    "in the terminal window."
                ),
                inputs=[],
                outputs=[train_output_info],
                show_progress="hidden",
            ).then(
                fn=start_train_from_ui,
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
                    lr_scheduler,
                    use_custom_lr,
                    custom_lr_g,
                    custom_lr_d,
                    compile_vocoder,
                    torch_compile_mode,
                    overtrain_detector,
                    stop_on_overtrain,
                    use_ema,
                ],
                outputs=[train_output_info],
                show_progress="hidden",
            )

            stop_train_button = gr.Button(_("Stop Training"), visible=True)
            # Stopping can take a moment while a checkpoint write finishes, so
            # the same announce-then-run treatment applies.
            stop_train_button.click(
                fn=lambda: "Stopping training - letting any checkpoint write finish first...",
                inputs=[],
                outputs=[train_output_info],
                show_progress="hidden",
            ).then(
                fn=stop_train_script,
                inputs=[],
                outputs=[train_output_info],
                show_progress="hidden",
            )

            index_button = gr.Button(_("Generate Index"))
            index_button.click(
                fn=run_index_script,
                inputs=[model_name, index_algorithm, index_metric],
                outputs=[train_output_info],
            )

    with gr.Accordion(_("Export Model"), open=False):
        if not os.name == "nt":
            gr.Markdown(
                _("Upload is available on Google Colab and saves exported files to your Google Drive.")
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label=_("Exported Pth file"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label=_("Pth file"),
                info=_("PTH file to export."),
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label=_("Exported Index File"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label=_("Index File"),
                info=_("Index file to export."),
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button(_("Refresh"))
                if not os.name == "nt":
                    upload_exported = gr.Button(_("Upload"))
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return gr.update(visible=bool(checkbox))

            def toggle_compile_mode(compile_enabled):
                return gr.update(visible=bool(compile_enabled))

            def download_prerequisites():
                    gr.Info(
                        _("Checking for prerequisites with pitch guidance... Missing files will be downloaded. If you already have them, this step will be skipped.")
                    )
                    run_prerequisites_script(
                        pretraineds_hifigan=True,
                        models=False,
                        exe=False,
                    )
                    gr.Info(
                        _("Prerequisites check complete. Missing files were downloaded, and you may now start preprocessing.")
                    )

            def toggle_visible_embedder_custom(embedder_model):
                return gr.update(visible=embedder_model == "custom")

            def update_noise_reduce_slider_visibility(noise_reduction):
                return gr.update(visible=bool(noise_reduction))

            def toggle_rms_norm_slider(norm_mode):
                return gr.update(visible=norm_mode == "post_rms")

            saved_components.extend([
                # Model settings
                vocoder, sampling_rate, cpu_threads, extract_gpu,

                # Preprocessing
                dataset_path, dataset_format, loading_resampling, use_smart_cutter,
                normalization_mode, rms_norm_db, cut_preprocess, chunk_len, overlap_len,
                process_effects, noise_reduction, clean_strength,

                # Feature extract
                f0_method, embedder_model, include_mutes,
                embedder_model_custom, remove_16k_slices, feature_precision,

                # Training
                batch_size, epoch_save_frequency, total_epoch_count,
                save_only_latest_net_models, save_weight_models, pretrained,
                cleanup, use_checkpointing, compile_vocoder, torch_compile_mode,
                use_tf32, use_benchmark,
                optimizer_choice, lr_scheduler,
                custom_pretrained, g_pretrained_path,
                d_pretrained_path, multiple_gpu, training_gpu, use_warmup,
                warmup_duration, use_custom_lr, custom_lr_g,
                custom_lr_d,
                index_algorithm, index_metric,
                overtrain_detector, stop_on_overtrain, use_ema
            ])

            def save_training_preset(inputs):
                settings = {}
                for component in saved_components:
                    settings[component.key] = inputs[component]

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, inputs[preset_dropdown] + '.json')))

                if not preset_path.startswith(presets_path):
                    raise gr.Error(
                        _("Invalid training preset name: {name}").format(
                            name=inputs[preset_dropdown]
                        ),
                        duration=5,
                    )

                with open(preset_path, 'w', encoding='utf-8') as of:
                    json.dump(settings, of, indent=4, ensure_ascii=False)

            def load_training_preset(preset_name):
                if preset_name not in get_presets_list():
                    raise gr.Error(
                        _("Preset does not exist: {name}").format(name=preset_name)
                    )

                preset_path = os.path.normpath(os.path.abspath(os.path.join(presets_path, preset_name + '.json')))

                with open(preset_path, 'r', encoding='utf-8') as ifile:
                    settings = json.loads(ifile.read())

                return [
                    (
                        settings[component.key]
                        if component.key in settings
                        # Historical preset key, kept only so a preset saved
                        # before the option was renamed still restores the
                        # toggle.  Nothing writes it any more.
                        else settings.get("compile_chouwagan", gr.skip())
                        if component.key == "compile_vocoder"
                        else gr.skip()
                    )
                    for component in saved_components
                ]

            refresh_presets_button.click(
                fn=lambda: gr.Dropdown(choices=get_presets_list()), 
                outputs=[preset_dropdown],
                show_progress="hidden",
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
                show_progress="hidden",
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
                show_progress="hidden",
            )
            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
                show_progress="hidden",
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
                show_progress="hidden",
            )
            compile_vocoder.change(
                fn=toggle_compile_mode,
                inputs=[compile_vocoder],
                outputs=[torch_compile_mode],
                show_progress="hidden",
            )
            use_custom_lr.change(
                fn=toggle_visible,
                inputs=[use_custom_lr],
                outputs=[custom_lr_settings],
                show_progress="hidden",
            )
            multiple_gpu.change(
                fn=toggle_visible,
                inputs=[multiple_gpu],
                outputs=[gpu_custom_settings],
                show_progress="hidden",
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
