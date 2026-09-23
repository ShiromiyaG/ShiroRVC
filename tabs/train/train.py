import os
import shutil
import json
from multiprocessing import cpu_count

import gradio as gr

from core import (
    run_extract_script,
    list_experiment_speakers,
    run_index_script,
    run_preprocess_script,
    run_prerequisites_script,
    run_train_script,
    stop_train_script,
)
from rvc.configs.config import (
    get_gpu_info,
    get_number_of_gpus,
    get_training_precision,
    max_vram_gpu,
)
from rvc.configs.vocoders import (
    get_default_vocoder,
    get_vocoder_choices,
    get_vocoder_description,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.i18n import _
from rvc.lib.terminal import (
    DEFAULT_CPU_THREADS,
    error as print_error,
    warning,
)
from rvc.lib import catalog
from rvc.lib.paths import (
    CUSTOM_PRETRAINED_DIR,
    DATASET_DIR,
    ROOT,
    TRAINING_PRESET_DIR,
)
from tabs.train.descs import (
    AUDIO_FILE_SLICING_INFO,
    BATCH_SIZE_INFO,
    DATASET_FORMAT_INFO,
    INDEX_SINGLE_SPEAKER_INFO,
    NORMALIZATION_INFO,
    OVERTRAIN_DETECTOR_INFO,
    OVERTRAIN_DETECTOR_LABEL,
    PITCH_EXTRACTION_INFO,
    PREPROCESS_RMS_VALUE_INFO,
    RESAMPLER_INFO,
    STOP_ON_OVERTRAIN_INFO,
    STOP_ON_OVERTRAIN_LABEL,
    TORCH_COMPILE_MODE_CHOICES,
    TORCH_COMPILE_MODE_INFO,
    TORCH_COMPILE_MODE_LABEL,
    VOCODER_COMPILE_INFO,
    VOCODER_COMPILE_LABEL,
    VOCODER_INFO_RVC,
)

saved_components = []  # components whose state is saved/restored by presets


presets_path = str(TRAINING_PRESET_DIR)

os.makedirs(CUSTOM_PRETRAINED_DIR, exist_ok=True)
os.makedirs(presets_path, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


def get_presets_list():
    return [os.path.splitext(s)[0] for s in os.listdir(presets_path) if s.endswith('.json')]

def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(_("Invalid pretrained file."))
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(CUSTOM_PRETRAINED_DIR, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info(_("Pretrained file added."))
    return None

def get_pth_list():
    return catalog.list_models(bundles=False, training_checkpoints=True)


def refresh_lists():
    """New choices for every list in the tab, in ``train_tab``'s ``list_outputs`` order."""
    return (
        gr.update(choices=catalog.list_training_models()),
        gr.update(choices=catalog.list_dataset_folders()),
        gr.update(choices=catalog.list_custom_pretraineds("G")),
        gr.update(choices=catalog.list_custom_pretraineds("D")),
        gr.update(choices=get_presets_list()),
        gr.update(choices=get_pth_list()),
        gr.update(choices=catalog.list_indexes()),
    )

def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [os.path.abspath(os.path.join(ROOT, p)) for p in allowed_paths]
    normalized_pth_path = os.path.abspath(os.path.join(ROOT, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        warning(f"Not a valid .pth path, skipping: {pth_path}", tag="[EXPORT]")
        return None

def export_index(index_path):
    allowed_paths = catalog.list_indexes()
    normalized_allowed_paths = [os.path.abspath(os.path.join(ROOT, p)) for p in allowed_paths]
    normalized_index_path = os.path.abspath(os.path.join(ROOT, index_path))

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
    except Exception:
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
    use_checkpointing,
    compile_vocoder,
    torch_compile_mode,
    overtrain_detector,
    stop_on_overtrain,
):
    """Launch a run from this tab's controls.

    Gradio binds ``inputs`` to parameters *by position*, so calling
    ``run_train_script`` directly makes the order of that list part of the
    launcher's signature -- inserting one argument in ``core.py`` silently
    shifts every later flag.  Naming the parameters here and calling with
    keywords confines the positional coupling to this one function.

    ``precision`` is not among the controls: it is a machine-level setting under
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
        use_checkpointing=use_checkpointing,
        precision=get_training_precision(),
        compile_vocoder=compile_vocoder,
        torch_compile_mode=torch_compile_mode,
        overtrain_detector=overtrain_detector,
        stop_on_overtrain=stop_on_overtrain,
    )


initial_vocoder = get_default_vocoder()
initial_sample_rate_choices = [
    str(rate) for rate in get_vocoder_sample_rates(initial_vocoder)
]
initial_sample_rate = (
    "48000" if "48000" in initial_sample_rate_choices else initial_sample_rate_choices[0]
)


def vocoder_description_text(vocoder_id):
    description = get_vocoder_description(vocoder_id)
    return _(description)


def update_vocoder_settings(vocoder_id, current_sample_rate):
    vocoder_id = normalize_vocoder(vocoder_id)
    sample_rate_choices = [
        str(rate) for rate in get_vocoder_sample_rates(vocoder_id)
    ]
    current_sample_rate = str(current_sample_rate)
    selected_sample_rate = (
        current_sample_rate
        if current_sample_rate in sample_rate_choices
        else sample_rate_choices[0]
    )
    return {
        "choices": sample_rate_choices,
        "value": selected_sample_rate,
        "__type__": "update",
    }

def train_tab(tab=None):
    """Build the tab; with ``tab`` given, its lists are refreshed on selection."""
    with gr.Row(equal_height=True):
        model_name = gr.Dropdown(
            label=_("Model Name"),
            info=_("Name of the new model."),
            choices=catalog.list_training_models(),
            value="example-model-name",
            interactive=True,
            allow_custom_value=True,
            scale=4,
            key='model_name'
        )
        refresh_button = gr.Button(_("Refresh"), scale=1)

    with gr.Row():
        with gr.Column():
            sampling_rate = gr.Radio(
                label=_("Sampling Rate"),
                info=_("Target sample rate. Match it to the dataset when possible."),
                choices=initial_sample_rate_choices,
                value=initial_sample_rate,
                interactive=True,
                key='sampling_rate'
            )
        with gr.Column():
            vocoder = gr.Radio(
                label=_("Vocoder"),
                info=_(VOCODER_INFO_RVC),
                choices=get_vocoder_choices(),
                value=initial_vocoder,
                interactive=True,
                visible=True,
                key='vocoder'
            )
            vocoder_description = gr.Markdown(
                value=vocoder_description_text(initial_vocoder),
                elem_classes=["rvc-vocoder-description"],
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

    with gr.Accordion(_("Training Presets"), open=False):
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

    with gr.Tab(f"1. {_('Preprocessing')}"):
        dataset_path = gr.Dropdown(
            label=_("Dataset Path"),
            info=_("Folder containing the training audio."),
            choices=catalog.list_dataset_folders(),
            allow_custom_value=True,
            interactive=True,
            key='dataset_path'
        )

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
                    choices=["ffmpeg", "librosa"],
                    value="ffmpeg",
                    interactive=True,
                    key='loading_resampling'
                )
            with gr.Column(min_width=0):
                normalization_mode = gr.Radio(
                    label=_("Loudness Normalization"),
                    info=_(NORMALIZATION_INFO),
                    choices=["none", "post_peak", "pre_peak_rvc", "pre_loudness"],
                    value="pre_peak_rvc",
                    interactive=True,
                    visible=True,
                    key='normalization_mode'
                )
        with gr.Row():
            rms_norm_db = gr.Slider(
                -24.0, -3.0, -16.0, step=1.0,
                label=_("Target Level (LUFS)"),
                info=_(PREPROCESS_RMS_VALUE_INFO),
                interactive=True,
                visible=False,
                key='rms_norm_db'
            )
        # The radio gets a row to itself: sharing one with the sliders below
        # left it a sliver of the width, so its four choices stacked into a
        # column three rows tall.
        with gr.Row():
            cut_preprocess = gr.Radio(
                label=_("Audio cutting"),
                info=_(AUDIO_FILE_SLICING_INFO),
                choices=["Skip", "Simple", "Automatic", "New Automatic"],
                value="New Automatic",
                interactive=True,
                key='cut_preprocess'
            )
        with gr.Row():
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
        preprocess_button = gr.Button(_("Preprocess Dataset"), variant="primary")
        preprocess_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Preprocessing status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(f"2. {_('Extraction')}"):
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
                choices=["contentvec", "spin_v2"],
                value="contentvec",
                interactive=True,
                key='embedder_model'
            )
        include_mutes = gr.Slider(
            0,
            10,
            5,
            step=1,
            label=_("Silent ( 'mute' ) files for training."),
            info=_("Add silent examples so the model can reproduce silence."),
            interactive=True,
            key='include_mutes'
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
        extract_button = gr.Button(_("Extract Features"), variant="primary")
        extract_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Extraction status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(f"3. {_('Training')}"):
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
                10,
                step=1,
                label=_("Saving frequency"),
                info=_("Save a checkpoint every N epochs."),
                interactive=True,
                key='epoch_save_frequency'
            )
            total_epoch_count = gr.Slider(
                1,
                10000,
                250,
                step=1,
                label=_("Total Epochs"),
                info=_("Total training epochs."),
                interactive=True,
                key='total_epoch_count'
            )
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
        # A Column, not a Group: a hidden Group keeps its border as a stray line.
        with gr.Column(visible=False) as pretrained_custom_settings:
            with gr.Row():
                g_pretrained_path = gr.Dropdown(
                    label=_("Custom Pretrained G"),
                    info=_("Generator pretrained file."),
                    choices=catalog.list_custom_pretraineds("G"),
                    interactive=True,
                    allow_custom_value=True,
                    key='g_pretrained_path'
                )
                d_pretrained_path = gr.Dropdown(
                    label=_("Custom Pretrained D"),
                    info=_("Discriminator pretrained file."),
                    choices=catalog.list_custom_pretraineds("D"),
                    interactive=True,
                    allow_custom_value=True,
                    key='d_pretrained_path'
                )
            with gr.Row():
                upload_pretrained = gr.UploadButton(
                    _("Upload Pretrained Model"),
                    file_types=[".pth"],
                    type="filepath",
                    size="sm",
                    scale=0,
                    elem_classes=["rvc-fit-button"],
                )

        gr.Markdown(f"#### {_('Optimisation')}")
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

        with gr.Row():
            train_button = gr.Button(_("Start Training"), variant="primary")
            stop_train_button = gr.Button(_("Stop Training"), variant="stop")
        train_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Training status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(f"4. {_('Index')}"):
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
        with gr.Row():
            with gr.Column(min_width=0):
                index_single_speaker = gr.Checkbox(
                    label=_("Index one speaker only"),
                    info=_(INDEX_SINGLE_SPEAKER_INFO),
                    value=False,
                    interactive=True,
                    key="index_single_speaker",
                )
            # The container carries the visibility, not the dropdown: toggling
            # the dropdown's own ``visible`` alongside its ``choices`` did not
            # take effect until the checkbox was cycled a second time.
            with gr.Column(min_width=0, visible=False) as index_speaker_row:
                index_speaker = gr.Dropdown(
                    label=_("Speaker to index"),
                    info=_("Read from the extracted features. Refresh after extracting."),
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                    key="index_speaker",
                )
        index_button = gr.Button(_("Generate Index"), variant="primary")
        index_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Index status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(f"5. {_('Export Model')}"):
        if not os.name == "nt":
            gr.Markdown(
                _("Upload is available on Google Colab and saves exported files to your Google Drive.")
            )
        with gr.Row():
            with gr.Column():
                pth_dropdown_export = gr.Dropdown(
                    label=_("Pth file"),
                    info=_("PTH file to export."),
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
                pth_file_export = gr.File(
                    label=_("Exported Pth file"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
            with gr.Column():
                index_dropdown_export = gr.Dropdown(
                    label=_("Index File"),
                    info=_("Index file to export."),
                    choices=catalog.list_indexes(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
                index_file_export = gr.File(
                    label=_("Exported Index File"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
        if not os.name == "nt":
            upload_exported = gr.Button(_("Upload"))
            upload_exported.click(
                fn=upload_to_google_drive,
                inputs=[pth_dropdown_export, index_dropdown_export],
                outputs=[],
            )

    # -- events ----------------------------------------------------------------

    def toggle_visible(checkbox):
        return gr.update(visible=bool(checkbox))

    def toggle_rms_norm_slider(norm_mode):
        return gr.update(
            visible=norm_mode in ("post_rms", "post_loudness", "pre_loudness")
        )

    def fill_index_speakers(name):
        """The picker's contents, from the selected model's features.

        Kept separate from showing it: one update that both reveals a
        component and repopulates it is what failed to apply on the first click.
        """
        speakers = [str(sid) for sid in list_experiment_speakers(name)] if name else []
        return gr.update(
            choices=speakers,
            value=speakers[0] if speakers else None,
        )

    def generate_index(name, algorithm, metric, single, speaker):
        if not single:
            return run_index_script(name, algorithm, metric, "all")
        if speaker in (None, ""):
            return _("Pick a speaker, or turn off 'Index one speaker only'.")
        return run_index_script(name, algorithm, metric, speaker)

    list_outputs = [
        model_name, dataset_path, g_pretrained_path, d_pretrained_path,
        preset_dropdown, pth_dropdown_export, index_dropdown_export,
    ]
    refresh_button.click(fn=refresh_lists, inputs=[], outputs=list_outputs)
    if tab is not None:
        tab.select(
            fn=refresh_lists, inputs=[], outputs=list_outputs, show_progress="hidden"
        )

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
            dataset_format,
            rms_norm_db,
        ],
        outputs=[preprocess_output_info],
    )
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
            include_mutes,
            feature_precision,
        ],
        outputs=[extract_output_info],
    )

    # Announce first so the box says something straight away, then run the
    # blocking call with Gradio's spinner off: a run lasts hours, and an
    # overlay counting seconds on an empty box reads as a hang.
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
            use_checkpointing,
            compile_vocoder,
            torch_compile_mode,
            overtrain_detector,
            stop_on_overtrain,
        ],
        outputs=[train_output_info],
        show_progress="hidden",
    )
    # Stopping can take a moment while a checkpoint write finishes.
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

    index_single_speaker.change(
        fn=toggle_visible,
        inputs=[index_single_speaker],
        outputs=[index_speaker_row],
    ).then(
        # Refilled on reveal too, so features extracted while this tab was
        # open are not missing from a list built before they existed.
        fn=fill_index_speakers,
        inputs=[model_name],
        outputs=[index_speaker],
    )
    # Switching models would otherwise leave the previous one's speakers in the
    # dropdown, and the build fails on an id that is not there.
    model_name.change(
        fn=fill_index_speakers,
        inputs=[model_name],
        outputs=[index_speaker],
    )
    index_button.click(
        fn=generate_index,
        inputs=[model_name, index_algorithm, index_metric,
                index_single_speaker, index_speaker],
        outputs=[index_output_info],
    )

    saved_components.extend([
        # Model settings
        vocoder, sampling_rate, cpu_threads, extract_gpu,

        # Preprocessing
        dataset_path, dataset_format, loading_resampling,
        normalization_mode, rms_norm_db, cut_preprocess, chunk_len, overlap_len,
        process_effects, noise_reduction, clean_strength,

        # Feature extract
        f0_method, embedder_model, include_mutes, feature_precision,

        # Training
        batch_size, epoch_save_frequency, total_epoch_count,
        save_only_latest_net_models, save_weight_models, pretrained,
        cleanup, use_checkpointing, compile_vocoder, torch_compile_mode,
        custom_pretrained, g_pretrained_path,
        d_pretrained_path, multiple_gpu, training_gpu, use_warmup,
        warmup_duration,
        index_algorithm, index_metric, index_single_speaker,
        overtrain_detector, stop_on_overtrain
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

    # ``show_progress="hidden"`` on these: the default tracker paints a spinner
    # on the output, and a component hidden when the request started never
    # receives the completion status -- it unhides stuck on "processing".
    for checkbox, target in (
        (noise_reduction, clean_strength),
        (custom_pretrained, pretrained_custom_settings),
        (use_warmup, warmup_settings),
        (compile_vocoder, torch_compile_mode),
        (multiple_gpu, gpu_custom_settings),
        # Meaningless without the detector that produces the signal.
        (overtrain_detector, stop_on_overtrain),
    ):
        checkbox.change(
            fn=toggle_visible,
            inputs=[checkbox],
            outputs=[target],
            show_progress="hidden",
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
        }.get(sr, 0.36),
        inputs=[sampling_rate],
        outputs=[overlap_len],
    )
    vocoder.change(
        fn=update_vocoder_settings,
        inputs=[vocoder, sampling_rate],
        outputs=[sampling_rate],
    )
    vocoder.change(
        fn=vocoder_description_text,
        inputs=[vocoder],
        outputs=[vocoder_description],
        show_progress="hidden",
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
    upload_pretrained.upload(
        fn=save_drop_model,
        inputs=[upload_pretrained],
        outputs=[],
    ).then(
        fn=refresh_lists, inputs=[], outputs=list_outputs, show_progress="hidden"
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
