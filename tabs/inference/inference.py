import os
import gradio as gr
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
from rvc.lib import catalog
from rvc.lib.model_bundle import (
    bundle_model_names,
    is_model_bundle,
    speaker_ids,
)
from rvc.lib.paths import AUDIO_DIR, FORMANT_DIR, INFERENCE_PRESET_DIR, ROOT
from tabs.settings.sections.restart import stop_infer
from rvc.lib import inference_presets as presets_lib
from rvc.lib.i18n import _

PRESETS_DIR = str(INFERENCE_PRESET_DIR)
FORMANTSHIFT_DIR = str(FORMANT_DIR)

os.makedirs(AUDIO_DIR, exist_ok=True)

names = catalog.list_models()
default_weight = names[0] if names else None

# The preset validation checks choices against these same sets, so a preset
# can never name one the controls below do not offer.
F0_METHODS = list(presets_lib.F0_METHODS)
EMBEDDER_MODELS = list(presets_lib.EMBEDDER_MODELS)
EXPORT_FORMATS = list(presets_lib.EXPORT_FORMATS)


def update_sliders_formant(preset):
    if not preset:
        return gr.skip(), gr.skip()
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def import_presets_button(file_path):
    """Copy a preset file into ``PRESETS_DIR`` and reselect it in the dropdown.

    The dropdown lists the *files* in ``PRESETS_DIR`` and ``read_preset``
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
        presets_lib.parse_preset(file_path)
    except (OSError, ValueError) as error:
        gr.Warning(_("Could not read that preset file: {}").format(error))
        return gr.update()

    os.makedirs(PRESETS_DIR, exist_ok=True)
    target_path = os.path.join(PRESETS_DIR, name)
    if os.path.abspath(target_path) != os.path.abspath(file_path):
        shutil.copyfile(file_path, target_path)

    return gr.update(
        choices=presets_lib.list_presets(PRESETS_DIR), value=name.rsplit(".", 1)[0]
    )


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    return gr.update(choices=presets_lib.list_presets(PRESETS_DIR))


def refresh_formant():
    return gr.update(choices=list_json_files(FORMANTSHIFT_DIR))


def save_to_wav(record_button):
    if record_button is None:
        # Clearing a recording fires this with nothing; both outputs still need
        # a value or Gradio rejects the response for being short.
        return gr.skip(), gr.skip()
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(AUDIO_DIR, new_name)

        os.makedirs(AUDIO_DIR, exist_ok=True)
        shutil.move(path_to_file, target_path)
        return catalog.relative(target_path), gr.update()


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(AUDIO_DIR, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    shutil.copy(file_path, target_path)
    return catalog.relative(target_path), gr.update()


def delete_outputs():
    gr.Info(_("Inference outputs cleared!"))
    for path in catalog.list_outputs():
        os.remove(path)


def get_speakers_id(model, sub_model_name=None):
    if not model or not os.path.exists(os.path.join(ROOT, model)):
        return [0]
    try:
        return speaker_ids(os.path.join(ROOT, model), sub_model_name)
    except Exception as e:
        warning(f"Could not read the model's speaker IDs: {e}", tag="[INFER]")
        return [0]

def get_bundle_model_names(model):
    """Return the speaker names stored in a multi-model bundle."""
    if not model or not is_model_bundle(model) or not os.path.exists(os.path.join(ROOT, model)):
        return []
    try:
        return bundle_model_names(os.path.join(ROOT, model))
    except Exception as e:
        warning(f"Could not inspect the model bundle: {e}", tag="[INFER]")
        return []

#: Preset keys, in the order the preset controls are passed around.  What a
#: preset leaves out, and why, is in ``rvc.lib.inference_presets``.
PRESET_ORDER = (
    "export_format", "seed", "split_audio", "autotune", "autotune_strength",
    "clean_audio", "clean_strength", "formant_shifting", "formant_qfrency",
    "formant_timbre", "pitch", "index_rate", "index_k", "index_power",
    "index_continuity", "rms_mix_rate", "protect", "silence_gate_db",
    "f0_method", "embedder_model",
)


def _conversion_settings(batch: bool) -> dict:
    """Build the conversion controls shared by the single and batch tabs.

    ``batch`` selects that tab's defaults and its extra F0-file input.
    Returns the components by name; the preset and visibility events are wired
    here, the conversion itself by the caller.
    """
    c = {}

    # One group visible at a time, each laid out in two columns.
    with gr.Tabs(elem_classes=["rvc-settings-tabs"]):
        with gr.Tab(_("Pitch")):
            with gr.Row():
                c["pitch"] = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=_("Pitch"),
                    info=_("Pitch shift in semitones. 12 = one octave up."),
                    value=0,
                    interactive=True,
                )
                c["f0_method"] = gr.Radio(
                    label=_("Pitch extraction algorithm"),
                    info=_("Pitch algorithm. RMVPE is the recommended default."),
                    choices=F0_METHODS,
                    value="rmvpe",
                    interactive=True,
                )
            with gr.Row():
                c["autotune"] = gr.Checkbox(
                    label=_("Autotuning"),
                    info=_("Apply autotune."),
                    value=False,
                    interactive=True,
                )
                c["autotune_strength"] = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Strength of autotuning"),
                    info=_("Higher values snap pitch to the chromatic grid."),
                    visible=False,
                    value=1,
                    interactive=True,
                )
            # Hidden in both tabs, each with the range and value it has always sent.
            c["filter_radius"] = gr.Slider(
                minimum=0,
                maximum=7 if batch else 1,
                step=1 if batch else 0.001,
                value=3 if batch else 0.006,
                label=_("Filter Radius"),
                interactive=False,
                visible=False,
            )

        with gr.Tab(_("Index")):
            with gr.Row():
                c["index_rate"] = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Search Feature Ratio"),
                    info=_("Index influence. Lower values can reduce artifacts."),
                    value=0.5,
                    interactive=True,
                )
                c["index_k"] = gr.Slider(
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
            with gr.Row():
                c["index_power"] = gr.Slider(
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
                c["index_continuity"] = gr.Slider(
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

        with gr.Tab(_("Voice")):
            with gr.Row():
                c["sid"] = gr.Dropdown(
                    label=_("Speaker ID"),
                    info=_("Speaker ID for multi-speaker models."),
                    choices=[0],
                    value=0,
                    interactive=True,
                )
                c["embedder_model"] = gr.Radio(
                    label=_("Embedder Model"),
                    info=_("Model used for speaker features."),
                    choices=EMBEDDER_MODELS,
                    value="contentvec",
                    interactive=True,
                )
            with gr.Row():
                c["protect"] = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=_("Protect Voiceless Consonants"),
                    info=_("Protect voiceless consonants. Higher values reduce index influence."),
                    value=0.3 if batch else 0.33,
                    interactive=True,
                )
                c["rms_mix_rate"] = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Volume Envelope"),
                    info=_("Mix the converted and input loudness envelopes."),
                    value=1,
                    interactive=True,
                )

        with gr.Tab(_("Audio processing")):
            with gr.Row():
                c["split_audio"] = gr.Checkbox(
                    label=_("Audio splitting"),
                    info=_("Split input at silence regions."),
                    value=False,
                    interactive=True,
                )
                c["clean_audio"] = gr.Checkbox(
                    label=_("Audio cleanup"),
                    info=_("Reduce detected noise in speech."),
                    value=False,
                    interactive=True,
                )
            with gr.Row():
                c["silence_gate_db"] = gr.Slider(
                    minimum=-120,
                    maximum=0,
                    step=1,
                    label=_("Silence Gate"),
                    info=_("Fade the output out where the input is quieter than this, in dBFS. Silence has no level for the content encoder, so the model fills it with hiss. -120 turns the gate off."),
                    value=-60,
                    interactive=True,
                )
                c["clean_strength"] = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=_("Strength of cleaning"),
                    info=_("Higher values apply stronger cleanup."),
                    visible=False,
                    value=0.5 if batch else 0.3,
                    interactive=True,
                )

        with gr.Tab(_("Formants")):
            c["formant_shifting"] = gr.Checkbox(
                label=_("Formant Shifting"),
                info=_("Shift vocal formants when needed."),
                value=False,
                interactive=True,
            )
            with gr.Column(visible=False) as formant_settings:
                c["formant_preset"] = gr.Dropdown(
                    label=_("Browse presets for formant shifting"),
                    info=_("Presets from assets/formant_shift."),
                    choices=list_json_files(FORMANTSHIFT_DIR),
                    interactive=True,
                )
                with gr.Row():
                    c["formant_qfrency"] = gr.Slider(
                        value=1.0,
                        info=_("Formant quefrency. Default: 1.0."),
                        label=_("Formant Quefrency"),
                        minimum=0.0,
                        maximum=16.0,
                        step=0.1,
                        interactive=True,
                    )
                    c["formant_timbre"] = gr.Slider(
                        value=1.0,
                        info=_("Formant timbre. Default: 1.0."),
                        label=_("Formant Timbre"),
                        minimum=0.0,
                        maximum=16.0,
                        step=0.1,
                        interactive=True,
                    )

        with gr.Tab(_("Output")):
            if not batch:
                c["output_path"] = gr.Textbox(
                    label=_("Path for infer outputs"),
                    placeholder=os.path.join("assets", "audios", "filename_output.wav"),
                    info=_("Optional output path. Empty uses assets/audios."),
                    value="",
                    interactive=True,
                )
            with gr.Row():
                c["export_format"] = gr.Radio(
                    label=_("Export Format"),
                    info=_("Output audio format."),
                    choices=EXPORT_FORMATS,
                    value="WAV",
                    interactive=True,
                )
                c["seed"] = gr.Number(
                    label=_("Inference Seed"),
                    info=_("Seed for reproducible output. Use 0 for random output."),
                    value=0,
                    interactive=True,
                )
            if batch:
                c["f0_file"] = gr.File(label=_("Edited F0 curve"))
            clear_outputs = gr.Button(_("Clear Outputs"))

        with gr.Tab(_("Preset Settings")):
            with gr.Row():
                c["preset_dropdown"] = gr.Dropdown(
                    label=_("Select Custom Preset"),
                    choices=presets_lib.list_presets(PRESETS_DIR),
                    interactive=True,
                )
                import_file = gr.File(
                    label=_("Select file to import"),
                    file_count="single",
                    type="filepath",
                    interactive=True,
                )
            with gr.Row(equal_height=True):
                preset_name_input = gr.Textbox(
                    label=_("Preset Name"),
                    placeholder=_("Enter preset name"),
                )
                export_button = gr.Button(_("Export Preset"))

    for checkbox, target in (
        (c["autotune"], c["autotune_strength"]),
        (c["clean_audio"], c["clean_strength"]),
        (c["formant_shifting"], formant_settings),
    ):
        checkbox.change(
            fn=lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[checkbox],
            outputs=[target],
            show_progress="hidden",
        )
    c["formant_preset"].change(
        fn=update_sliders_formant,
        inputs=[c["formant_preset"]],
        outputs=[c["formant_qfrency"], c["formant_timbre"]],
        show_progress="hidden",
    )
    clear_outputs.click(fn=delete_outputs, inputs=[], outputs=[])

    preset_controls = [c[key] for key in PRESET_ORDER]

    def apply_preset(preset):
        if not preset:
            return [gr.skip()] * len(preset_controls)
        try:
            values = presets_lib.read_preset(PRESETS_DIR, preset)
        except (OSError, ValueError) as error:
            gr.Warning(_("Could not read that preset file: {}").format(error))
            return [gr.skip()] * len(preset_controls)
        # The checkboxes' own ``.change`` events then show or hide their
        # strength and formant controls.
        return [
            gr.update(value=values[key]) if key in values else gr.skip()
            for key in PRESET_ORDER
        ]

    def save_preset(preset_name, *values):
        name = (preset_name or "").strip()
        if not name:
            gr.Warning(_("Enter a preset name first."))
            return gr.skip()
        if not presets_lib.is_valid_name(name):
            gr.Warning(_('A preset name cannot contain \\ / : * ? " < > |'))
            return gr.skip()
        presets_lib.write_preset(PRESETS_DIR, name, dict(zip(PRESET_ORDER, values)))
        gr.Info(_("Preset saved: {}").format(name))
        # Only the list is refreshed: selecting the new preset would load it
        # straight back over the form.
        return gr.update(choices=presets_lib.list_presets(PRESETS_DIR))

    c["preset_dropdown"].change(
        apply_preset,
        inputs=c["preset_dropdown"],
        outputs=preset_controls,
        show_progress="hidden",
    )
    import_file.change(
        import_presets_button,
        inputs=import_file,
        outputs=[c["preset_dropdown"]],
    )
    export_button.click(
        save_preset,
        inputs=[preset_name_input, *preset_controls],
        outputs=c["preset_dropdown"],
    )
    return c


def inference_tab(tab=None):
    """Build the tab; with ``tab`` given, its lists are refreshed on selection."""
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=_("Voice Model"),
                info=_("Voice model used for inference."),
                choices=names,
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
                choices=catalog.list_indexes(),
                value=catalog.guess_index_for(default_weight),
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button(_("Unload the voice model"))
            refresh_button = gr.Button(_("Refresh models, indexes and audios"))

    with gr.Tab(_("Single input infer")):
        with gr.Column():
            upload_audio = gr.Audio(
                label=_("Upload Audio"), type="filepath", editable=False
            )
            audio_paths = catalog.list_audios()
            audio = gr.Dropdown(
                label=_("Select Audio Input"),
                info=_("Audio to convert."),
                choices=audio_paths,
                value=audio_paths[0] if audio_paths else "",
                interactive=True,
                allow_custom_value=True,
            )

        single = _conversion_settings(batch=False)

        convert_button1 = gr.Button(_("Convert"), variant="primary")

        with gr.Row():
            vc_output1 = gr.Textbox(
                label=_("Output Information"),
                info=_("Inference status."),
            )
            vc_output2 = gr.Audio(label=_("Export Audio"))

    with gr.Tab(_("Batch")):
        with gr.Row():
            input_folder_batch = gr.Textbox(
                label=_("Input Folder"),
                info=_("Folder containing input audio."),
                placeholder=_("Enter input path"),
                value=str(AUDIO_DIR),
                interactive=True,
            )
            output_folder_batch = gr.Textbox(
                label=_("Output Folder"),
                info=_("Folder for converted audio."),
                placeholder=_("Enter output path"),
                value=str(AUDIO_DIR),
                interactive=True,
            )

        batch = _conversion_settings(batch=True)

        with gr.Row():
            convert_button_batch = gr.Button(_("Convert"), variant="primary")
            stop_button = gr.Button(_("Stop convert"), visible=False)

        vc_output3 = gr.Textbox(
            label=_("Output Information"),
            info=_("Batch status."),
        )

    def run_single_infer(
        pitch, filter_radius, index_rate, rms_mix_rate, protect,
        f0_method, audio, output_path, model_file, index_file,
        split_audio, autotune, autotune_strength,
        clean_audio, clean_strength, export_format,
        embedder_model,
        formant_shifting, formant_qfrency, formant_timbre,
        sid, seed, bundle_submodel,
        index_k, index_power, index_continuity,
        silence_gate_db,
    ):
        if not output_path or not output_path.strip():
            output_path = catalog.default_output_path(audio)
        else:
            if os.path.isdir(output_path):
                default_name = os.path.splitext(os.path.basename(catalog.default_output_path(audio)))[0]
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
            embedder_model,
            formant_shifting, formant_qfrency, formant_timbre,
            sid, seed, bundle_submodel,
            index_k, index_power, index_continuity,
            silence_gate_db,
        )

    def on_model_change(model_path):
        bundle_models = get_bundle_model_names(model_path)

        if bundle_models:
            sid_update = gr.update(choices=[0], value=0)
            return (
                gr.update(visible=False, value=""),
                sid_update,
                sid_update,
                gr.update(visible=True, choices=bundle_models, value=bundle_models[0])
            )
        speakers = get_speakers_id(model_path)
        sid_update = gr.update(choices=speakers, value=speakers[0] if speakers else 0)
        is_bundle = is_model_bundle(model_path)
        return (
            gr.update(
                choices=catalog.list_indexes(),
                value=catalog.guess_index_for(model_path),
                interactive=not is_bundle,
                visible=True,
            ),
            sid_update,
            sid_update,
            gr.update(visible=False, choices=[], value=None)
        )

    def on_submodel_change(model_path, sub_model_name):
        if not model_path or not sub_model_name:
            sid_update = gr.update(choices=[0], value=0)
        else:
            speakers = get_speakers_id(model_path, sub_model_name)
            sid_update = gr.update(choices=speakers, value=speakers[0] if speakers else 0)
        return sid_update, sid_update

    def sync_speaker_id(model_path, repurposed_index_value):
        if model_path and is_model_bundle(model_path):
            return gr.update(value=repurposed_index_value), gr.update(value=repurposed_index_value)
        return gr.update(), gr.update()

    def refresh_lists():
        return (
            gr.update(choices=catalog.list_models()),
            gr.update(choices=catalog.list_indexes()),
            gr.update(choices=catalog.list_audios()),
            refresh_formant(),
            refresh_formant(),
            refresh_presets(),
            refresh_presets(),
        )

    def _unload_and_cleanup():
        import_voice_converter().cleanup_model()
        return gr.update(value=""), gr.update(value="")

    sids = [single["sid"], batch["sid"]]
    list_outputs = [
        model_file, index_file, audio,
        single["formant_preset"], batch["formant_preset"],
        single["preset_dropdown"], batch["preset_dropdown"],
    ]

    unload_button.click(
        fn=_unload_and_cleanup,
        inputs=[],
        outputs=[model_file, index_file],
    )
    refresh_button.click(fn=refresh_lists, inputs=[], outputs=list_outputs)
    if tab is not None:
        tab.select(
            fn=refresh_lists, inputs=[], outputs=list_outputs, show_progress="hidden"
        )
    model_file.change(
        fn=on_model_change,
        inputs=[model_file],
        outputs=[index_file, *sids, bundle_submodel],
        show_progress="hidden",
    )
    bundle_submodel.change(
        fn=on_submodel_change,
        inputs=[model_file, bundle_submodel],
        outputs=sids,
    )
    index_file.change(
        fn=sync_speaker_id,
        inputs=[model_file, index_file],
        outputs=sids,
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, single["output_path"]],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, single["output_path"]],
    )
    convert_button1.click(
        fn=run_single_infer,
        inputs=[
            single["pitch"],
            single["filter_radius"],
            single["index_rate"],
            single["rms_mix_rate"],
            single["protect"],
            single["f0_method"],
            audio,
            single["output_path"],
            model_file,
            index_file,
            single["split_audio"],
            single["autotune"],
            single["autotune_strength"],
            single["clean_audio"],
            single["clean_strength"],
            single["export_format"],
            single["embedder_model"],
            single["formant_shifting"],
            single["formant_qfrency"],
            single["formant_timbre"],
            single["sid"],
            single["seed"],
            bundle_submodel,
            single["index_k"],
            single["index_power"],
            single["index_continuity"],
            single["silence_gate_db"],
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=run_batch_infer_script,
        inputs=[
            batch["pitch"],
            batch["filter_radius"],
            batch["index_rate"],
            batch["rms_mix_rate"],
            batch["protect"],
            batch["f0_method"],
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            batch["split_audio"],
            batch["autotune"],
            batch["autotune_strength"],
            batch["clean_audio"],
            batch["clean_strength"],
            batch["export_format"],
            batch["f0_file"],
            batch["embedder_model"],
            batch["formant_shifting"],
            batch["formant_qfrency"],
            batch["formant_timbre"],
            batch["sid"],
            batch["seed"],
            batch["index_k"],
            batch["index_power"],
            batch["index_continuity"],
            batch["silence_gate_db"],
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[convert_button_batch, stop_button],
        show_progress="hidden",
    )
    stop_button.click(fn=stop_infer, inputs=[], outputs=[])
    stop_button.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[convert_button_batch, stop_button],
        show_progress="hidden",
    )
