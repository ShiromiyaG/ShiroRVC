import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import json


from core import (
    run_infer_script,
    run_batch_infer_script,
    import_voice_converter,
)

from rvc.lib.terminal import warning
from rvc.lib.text import format_title
from rvc.lib.model_bundle import (
    get_bundle_models,
    get_bundle_model_state,
    is_model_bundle,
    is_model_file,
    load_model_bundle,
    walk_models,
)
from tabs.settings.sections.restart import stop_infer
from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in walk_models(model_root_relative)
    for file in files
    if (
        is_model_file(file)
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

default_weight = names[0] if names else None

indexes_list = [
    os.path.join(root, name)
    for root, _, files in walk_models(model_root_relative)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]


def update_sliders(preset):
    with open(
        os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["filter_radius"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
    )


def update_sliders_formant(preset):
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)


def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets


def get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect):
    return {
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }


def export_presets_button(
    preset_name, pitch, filter_radius, index_rate, rms_mix_rate, protect
):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(
            pitch, filter_radius, index_rate, rms_mix_rate, protect
        )
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"


def import_presets_button(file_path):
    """Copy a preset file into ``PRESETS_DIR`` and reselect it in the dropdown.

    The dropdown lists the *files* in ``PRESETS_DIR`` and ``update_sliders``
    loads the selection back as ``PRESETS_DIR/<name>.json``, so an entry that
    was never written there cannot be loaded.  This used to return three values
    -- names, the parsed dict and a status string -- into a single dropdown
    output, which handed the dropdown the whole tuple as its value; and it read
    ``file_path.name`` although the picker is ``type="filepath"`` and hands
    over a plain string.
    """
    if not file_path:
        return gr.update()

    name = format_title(os.path.basename(file_path))
    if not name.endswith(".json"):
        name = f"{name}.json"
    try:
        # Parsed before copying, so an unreadable file is refused rather than
        # landing in the folder as a preset that breaks on selection.
        import_presets(file_path)
    except (OSError, ValueError) as error:
        gr.Warning(_("Could not read that preset file: {}").format(error))
        return gr.update()

    target_path = os.path.join(PRESETS_DIR, name)
    if os.path.abspath(target_path) != os.path.abspath(file_path):
        shutil.copyfile(file_path, target_path)

    return gr.update(
        choices=list_json_files(PRESETS_DIR), value=name.rsplit(".", 1)[0]
    )


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(audio_root, new_name)
    return output_path


def change_choices(model):
    names = [
        os.path.join(root, file)
        for root, _, files in walk_models(model_root_relative)
        for file in files
        if (
            is_model_file(file)
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in walk_models(model_root_relative)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {"__type__": "update"},
        {"__type__": "update"},
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in walk_models(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        # Clearing a recording fires this with nothing; both outputs still need
        # a value or Gradio rejects the response for being short.
        return gr.skip(), gr.skip()
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, gr.update()


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, gr.update()


def delete_outputs():
    gr.Info(_("Inference outputs cleared!"))
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and "_output" in name:
                os.remove(os.path.join(root, name))

def match_index(model_file_value):
    if not model_file_value or is_model_bundle(model_file_value):
        return ""

    model_dir = os.path.dirname(model_file_value)
    if not os.path.exists(model_dir):
        return ""

    try:
        files_in_dir = os.listdir(model_dir)
        index_files = [f for f in files_in_dir if f.endswith(".index")]
    except:
        return ""

    if not index_files:
        return ""

    model_name = os.path.basename(model_file_value)
    model_base = os.path.splitext(model_name)[0]
    core_name = model_base.split('_')[0]

    for index_file in index_files:
        if core_name.lower() in index_file.lower():
            return os.path.join(model_dir, index_file)
    return ""


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


def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

def get_speakers_id(model, sub_model_name=None):
    if not model or not os.path.exists(os.path.join(now_dir, model)):
        return [0]
    try:
        if is_model_bundle(model):
            model_data = load_model_bundle(os.path.join(now_dir, model))
            model_state = get_bundle_model_state(model_data, sub_model_name)
            speakers_id = model_state.get("speakers_id") if model_state else None
        else:
            import torch

            model_data = torch.load(os.path.join(now_dir, model), map_location="cpu", weights_only=True)
            speakers_id = model_data.get("speakers_id") if isinstance(model_data, dict) else None
        if speakers_id:
            return list(range(speakers_id))
        else:
            return [0]
    except Exception as e:
        warning(f"Could not read the model's speaker IDs: {e}", tag="[INFER]")
        return [0]

def get_bundle_model_names(model):
    """Return the speaker names stored in a multi-model bundle."""
    if not model or not is_model_bundle(model) or not os.path.exists(os.path.join(now_dir, model)):
        return []
    try:
        model_data = load_model_bundle(os.path.join(now_dir, model))
        return sorted(get_bundle_models(model_data).keys())
    except Exception as e:
        warning(f"Could not inspect the model bundle: {e}", tag="[INFER]")
        return []

def inference_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=_("Voice Model"),
                info=_("Voice model used for inference."),
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            bundle_submodel = gr.Dropdown(
                label=_("Bundle sub-model"),
                info=_("Sub-model inside the model bundle."),
                choices=[],
                value=None,
                interactive=True,
                visible=False
            )
            index_file = gr.Dropdown(
                label=_("Index File"),
                info=_("Optional index file; unavailable for model bundles."),
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button(_("Unload the voice model"))
            refresh_button = gr.Button(_("Refresh models, indexes and audios"))

            def _unload_and_cleanup():
                import_voice_converter().cleanup_model()
                return {"value": "", "__type__": "update"}, {"value": "", "__type__": "update"}

            unload_button.click(
                fn=_unload_and_cleanup,
                inputs=[],
                outputs=[model_file, index_file],
            )

        def run_single_infer(
            pitch, filter_radius, index_rate, rms_mix_rate, protect,
            f0_method, audio, output_path, model_file, index_file,
            split_audio, autotune, autotune_strength,
            clean_audio, clean_strength, export_format,
            embedder_model, embedder_model_custom,
            formant_shifting, formant_qfrency, formant_timbre,
            sid, seed, bundle_submodel,
            index_k, index_power, index_continuity,
            silence_gate_db,
        ):
            if not output_path or not output_path.strip():
                output_path = output_path_fn(audio)
            else:
                if os.path.isdir(output_path):
                    default_name = os.path.splitext(os.path.basename(output_path_fn(audio)))[0]
                    output_path = os.path.join(output_path, default_name + f".{export_format.lower()}")

                _, ext = os.path.splitext(output_path)
                valid_formats = {"wav", "mp3", "flac", "ogg", "m4a"}
                if ext and ext.lower().lstrip(".") in valid_formats:
                    export_format = ext.lower().lstrip(".").upper()

                    output_path = output_path[: -len(ext)] + ".wav"
                elif not ext:
                    output_path += ".wav"

            return run_infer_script(
                pitch, filter_radius, index_rate, rms_mix_rate, protect,
                f0_method, audio, output_path, model_file, index_file,
                split_audio, autotune, autotune_strength,
                clean_audio, clean_strength, export_format,
                None,
                embedder_model, embedder_model_custom,
                formant_shifting, formant_qfrency, formant_timbre,
                sid, seed, bundle_submodel,
                index_k, index_power, index_continuity,
                silence_gate_db,
            )

        def on_model_change(model_path):
            bundle_models = get_bundle_model_names(model_path)

            if bundle_models:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(choices=[0], value=0, visible=True),
                    gr.update(visible=True, choices=bundle_models, value=bundle_models[0])
                )
            else:
                speakers = get_speakers_id(model_path)
                speaker_val = speakers[0] if speakers else 0
                is_bundle = is_model_bundle(model_path)

                return (
                    gr.update(
                        choices=get_indexes(),
                        value=match_index(model_path) if not is_bundle else "",
                        interactive=not is_bundle,
                        visible=True,
                    ),
                    gr.update(visible=True, choices=speakers, value=speaker_val),
                    gr.update(visible=False, choices=[], value=None)
                )

        def on_submodel_change(model_path, sub_model_name):
            if not model_path or not sub_model_name:
                return gr.update(choices=[0], value=0)
            
            speakers = get_speakers_id(model_path, sub_model_name)
            speaker_val = speakers[0] if speakers else 0
            return gr.update(choices=speakers, value=speaker_val)

            
        def sync_speaker_id(model_path, repurposed_index_value):
            if model_path and is_model_bundle(model_path):
                return gr.update(value=repurposed_index_value)
            return gr.update()

    with gr.Tab(_("Single input infer")):
        with gr.Column():
            upload_audio = gr.Audio(
                label=_("Upload Audio"), type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Dropdown(
                    label=_("Select Audio Input"),
                    info=_("Audio to convert."),
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion(_("Advanced Settings for inference"), open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button(_("Clear '_output' audio files ( infer outputs ) from 'assets/audios' "))
                output_path = gr.Textbox(
                    label=_("Path for infer outputs"),
                    placeholder=os.path.join("assets", "audios", "filename_output.wav"),
                    info=_("Optional output path. Empty uses assets/audios."),
                    value="",
                    interactive=True,
                )
                export_format = gr.Radio(
                    label=_("Export Format"),
                    info=_("Output audio format."),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                seed = gr.Number(
                    label=_("Inference Seed"),
                    info=_("Seed for reproducible output. Use 0 for random output."),
                    value=0,
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label=_("Speaker ID"),
                    info=_("Speaker ID for multi-speaker models."),
                    choices=[0],
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label=_("Audio splitting"),
                    info=_("Split input at silence regions."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label=_("Autotuning"),
                    info=_("Apply autotune."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Strength of autotuning"),
                    info=_("Higher values snap pitch to the chromatic grid."),
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label=_("Audio cleanup"),
                    info=_("Reduce detected noise in speech."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Strength of cleaning"),
                    info=_("Higher values apply stronger cleanup."),
                    visible=False,
                    value=0.3,
                    interactive=True,
                )
                formant_shifting = gr.Checkbox(
                    label=_("Formant Shifting"),
                    info=_("Shift vocal formants when needed."),
                    value=False,
                    visible=True,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row:
                    formant_preset = gr.Dropdown(
                        label=_("Browse presets for formant shifting"),
                        info=_("Presets from assets/formant_shift."),
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency = gr.Slider(
                    value=1.0,
                    info=_("Formant quefrency. Default: 1.0."),
                    label=_("Formant Quefrency."),
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre = gr.Slider(
                    value=1.0,
                    info=_("Formant timbre. Default: 1.0."),
                    label=_("Formant Timbre"),
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                with gr.Accordion(_("Preset Settings"), open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label=_("Select Custom Preset"),
                            choices=list_json_files(PRESETS_DIR),
                            interactive=True,
                        )
                        presets_refresh_button = gr.Button(_("Refresh Presets"))
                    import_file = gr.File(
                        label=_("Select file to import"),
                        file_count="single",
                        type="filepath",
                        interactive=True,
                    )
                    import_file.change(
                        import_presets_button,
                        inputs=import_file,
                        outputs=[preset_dropdown],
                    )
                    presets_refresh_button.click(
                        refresh_presets, outputs=preset_dropdown
                    )
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label=_("Preset Name"),
                            placeholder=_("Enter preset name"),
                        )
                        export_button = gr.Button(_("Export Preset"))
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=_("Pitch"),
                    info=_("Pitch shift in semitones. 12 = one octave up."),
                    value=0,
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Filter Radius"),
                    info=_("Smooth the extracted pitch curve. Default: 0.006."),
                    value=0.006,
                    step=0.001,
                    interactive=False,
                    visible=False,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Search Feature Ratio"),
                    info=_("Index influence. Lower values can reduce artifacts."),
                    value=0.5,
                    interactive=True,
                )
                index_k = gr.Slider(
                    minimum=1,
                    maximum=32,
                    step=1,
                    label=_("Index Neighbours"),
                    info=(
                        _("Frames averaged per match. Fewer keeps the training "
                        "voice's idiosyncratic articulation; more averages "
                        "toward its mean voice.")
                    ),
                    value=8,
                    interactive=True,
                )
                index_power = gr.Slider(
                    minimum=0,
                    maximum=8,
                    step=0.25,
                    label=_("Index Sharpness"),
                    info=(
                        _("How strongly closer neighbours outweigh further ones. "
                        "0 averages them equally; high values use the nearest alone.")
                    ),
                    value=2.0,
                    interactive=True,
                )
                index_continuity = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=0.1,
                    label=_("Index Continuity"),
                    info=(
                        _("Favours matches that continue the previous frame's, so "
                        "the retrieval stops jumping between unrelated parts of "
                        "the dataset. Needs an index built by this fork.")
                    ),
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("RMS Volume Envelope"),
                    info=_("Mix the converted and input loudness envelopes."),
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=_("Protect Voiceless Consonants"),
                    info=_("Protect voiceless consonants. Higher values reduce index influence."),
                    value=0.33,
                    interactive=True,
                )
                silence_gate_db = gr.Slider(
                    minimum=-120,
                    maximum=0,
                    step=1,
                    label=_("Silence Gate"),
                    info=_("Fade the output out where the input is quieter than this, in dBFS. Silence has no level for the content encoder, so the model fills it with hiss. -120 turns the gate off."),
                    value=-60,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                    show_progress="hidden",
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                f0_method = gr.Radio(
                    label=_("Pitch extraction algorithm"),
                    info=_("Pitch algorithm. RMVPE is the recommended default."),
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
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
                )
                with gr.Column(visible=False) as embedder_custom:
                    with gr.Accordion(_("Custom Embedder"), open=True):
                        with gr.Row():
                            embedder_model_custom = gr.Dropdown(
                                label=_("Select Custom Embedder"),
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button = gr.Button(_("Refresh embedders"))
                        folder_name_input = gr.Textbox(label=_("Folder Name"), interactive=True)
                        with gr.Row():
                            bin_file_upload = gr.File(
                                label=_("Upload .bin"),
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload = gr.File(
                                label=_("Upload .json"),
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button = gr.Button(_("Move files to custom embedder"))

        convert_button1 = gr.Button(_("Convert"))

        with gr.Row():
            vc_output1 = gr.Textbox(
                label=_("Output Information"),
                info=_("Inference status."),
            )
            vc_output2 = gr.Audio("Export Audio")

    with gr.Tab(_("Batch")):
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label=_("Input Folder"),
                    info=_("Folder containing input audio."),
                    placeholder=_("Enter input path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label=_("Output Folder"),
                    info=_("Folder for converted audio."),
                    placeholder=_("Enter output path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion(_("Advanced Settings"), open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button(_("Clear Outputs"))
                export_format_batch = gr.Radio(
                    label=_("Export Format"),
                    info=_("Output audio format."),
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid_batch = gr.Dropdown(
                    label=_("Speaker ID"),
                    info=_("Speaker ID for conversion."),
                    choices=[0],
                    value=0,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label=_("Split Audio"),
                    info=_("Split input into chunks."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label=_("Autotune"),
                    info=_("Apply autotune for singing."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Autotune Strength"),
                    info=_("Higher values snap pitch to the chromatic grid."),
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label=_("Clean Audio"),
                    info=_("Reduce detected noise in speech."),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Clean Strength"),
                    info=_("Higher values apply stronger cleanup."),
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                formant_shifting_batch = gr.Checkbox(
                    label=_("Formant Shifting"),
                    info=_("Shift vocal formants when needed."),
                    value=False,
                    visible=True,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row_batch:
                    formant_preset_batch = gr.Dropdown(
                        label=_("Browse presets for formanting"),
                        info=_("Presets from assets/formant_shift."),
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button_batch = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency_batch = gr.Slider(
                    value=1.0,
                    info=_("Default: 1.0."),
                    label=_("Quefrency for formant shifting"),
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre_batch = gr.Slider(
                    value=1.0,
                    info=_("Default: 1.0."),
                    label=_("Timbre for formant shifting"),
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=_("Pitch"),
                    info=_("Pitch shift in semitones."),
                    value=0,
                    interactive=True,
                )
                filter_radius_batch = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=_("Filter Radius"),
                    info=_("Median filtering for pitch smoothing."),
                    value=3,
                    step=1,
                    interactive=False,
                    visible=False,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Search Feature Ratio"),
                    info=_("Index influence. Lower values can reduce artifacts."),
                    value=0.5,
                    interactive=True,
                )
                index_k_batch = gr.Slider(
                    minimum=1,
                    maximum=32,
                    step=1,
                    label=_("Index Neighbours"),
                    info=(
                        _("Frames averaged per match. Fewer keeps the training "
                        "voice's idiosyncratic articulation; more averages "
                        "toward its mean voice.")
                    ),
                    value=8,
                    interactive=True,
                )
                index_power_batch = gr.Slider(
                    minimum=0,
                    maximum=8,
                    step=0.25,
                    label=_("Index Sharpness"),
                    info=(
                        _("How strongly closer neighbours outweigh further ones. "
                        "0 averages them equally; high values use the nearest alone.")
                    ),
                    value=2.0,
                    interactive=True,
                )
                index_continuity_batch = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=0.1,
                    label=_("Index Continuity"),
                    info=(
                        _("Favours matches that continue the previous frame's, so "
                        "the retrieval stops jumping between unrelated parts of "
                        "the dataset. Needs an index built by this fork.")
                    ),
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Volume Envelope"),
                    info=_("Mix the converted and input loudness envelopes."),
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=_("Protect Voiceless Consonants"),
                    info=_("Protect voiceless consonants. Higher values reduce index influence."),
                    value=0.3,
                    interactive=True,
                )
                silence_gate_db_batch = gr.Slider(
                    minimum=-120,
                    maximum=0,
                    step=1,
                    label=_("Silence Gate"),
                    info=_("Fade the output out where the input is quieter than this, in dBFS. Silence has no level for the content encoder, so the model fills it with hiss. -120 turns the gate off."),
                    value=-60,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch_batch,
                        filter_radius_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                    ],
                    show_progress="hidden",
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                    outputs=[],
                )
                f0_method_batch = gr.Radio(
                    label=_("Pitch extraction algorithm"),
                    info=_("Pitch algorithm. RMVPE is the recommended default."),
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model_batch = gr.Radio(
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
                )
                f0_file_batch = gr.File(
                    label=_("Edited F0 curve"),
                    visible=True,
                )
                with gr.Column(visible=False) as embedder_custom_batch:
                    with gr.Accordion(_("Custom Embedder"), open=True):
                        with gr.Row():
                            embedder_model_custom_batch = gr.Dropdown(
                                label=_("Select Custom Embedder"),
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button_batch = gr.Button(_("Refresh embedders"))
                        folder_name_input_batch = gr.Textbox(
                            label=_("Folder Name"), interactive=True
                        )
                        with gr.Row():
                            bin_file_upload_batch = gr.File(
                                label=_("Upload .bin"),
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload_batch = gr.File(
                                label=_("Upload .json"),
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button_batch = gr.Button(_("Move files to custom embedder"))

        convert_button_batch = gr.Button(_("Convert"))
        stop_button = gr.Button(_("Stop convert"), visible=False)
        stop_button.click(fn=stop_infer, inputs=[], outputs=[])

        with gr.Row():
            vc_output3 = gr.Textbox(
                label=_("Output Information"),
            info=_("Batch status."),
            )

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def toggle_visible_formant_shifting(checkbox):
        if checkbox:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    model_file.change(
        fn=on_model_change,
        inputs=[model_file],
        outputs=[index_file, sid, bundle_submodel],
        show_progress="hidden",
    )
    bundle_submodel.change(
        fn=on_submodel_change,
        inputs=[model_file, bundle_submodel],
        outputs=[sid]
    )
    index_file.change(
        fn=sync_speaker_id,
        inputs=[model_file, index_file],
        outputs=[sid],
    )
    autotune.change(
        fn=toggle_visible,
        inputs=[autotune],
        outputs=[autotune_strength],
        show_progress="hidden",
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
        show_progress="hidden",
    )
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row,
            formant_preset,
            formant_refresh_button,
            formant_qfrency,
            formant_timbre,
        ],
        show_progress="hidden",
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row_batch,
            formant_preset_batch,
            formant_refresh_button_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
        ],
        show_progress="hidden",
    )
    formant_refresh_button.click(
        fn=refresh_formant,
        inputs=[],
        outputs=[formant_preset],
        show_progress="hidden",
    )
    formant_preset.change(
        fn=update_sliders_formant,
        inputs=[formant_preset],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
        show_progress="hidden",
    )
    formant_preset_batch.change(
        fn=update_sliders_formant,
        inputs=[formant_preset_batch],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
        show_progress="hidden",
    )
    autotune_batch.change(
        fn=toggle_visible,
        inputs=[autotune_batch],
        outputs=[autotune_strength_batch],
        show_progress="hidden",
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
        show_progress="hidden",
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio, sid, sid_batch],
    )

    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
        show_progress="hidden",
    )
    embedder_model_batch.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model_batch],
        outputs=[embedder_custom_batch],
        show_progress="hidden",
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
        show_progress="hidden",
    )
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[
            folder_name_input_batch,
            bin_file_upload_batch,
            config_file_upload_batch,
        ],
        outputs=[],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom_batch],
        show_progress="hidden",
    )
    convert_button1.click(
        fn=run_single_infer,
        inputs=[
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            sid,
            seed,
            bundle_submodel,
            index_k,
            index_power,
            index_continuity,
            silence_gate_db,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=run_batch_infer_script,
        inputs=[
            pitch_batch,
            filter_radius_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            autotune_strength_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            f0_file_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            sid_batch,
            seed,
            index_k_batch,
            index_power_batch,
            index_continuity_batch,
            silence_gate_db_batch,
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
        show_progress="hidden",
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
        show_progress="hidden",
    )
