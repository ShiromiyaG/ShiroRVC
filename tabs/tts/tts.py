import json
import os
import random
import sys

import gradio as gr

from rvc.lib.i18n import _

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_tts_script, import_voice_converter
from tabs.inference.inference import (
    change_choices,
    create_folder_and_move_files,
    get_indexes,
    get_speakers_id,
    match_index,
    refresh_embedders_folders,
    extract_model_and_epoch,
    names,
    default_weight,
)


with open(
    os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
) as file:
    tts_voices_data = json.load(file)

short_names = [voice.get("ShortName", "") for voice in tts_voices_data]


def process_input(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file.read()
        gr.Info(_("The file has been loaded!"))
        return file_path, file_path
    except UnicodeDecodeError:
        gr.Info(_("The file has to be in UTF-8 encoding."))
        return None, None


# TTS tab
def tts_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=_("Voice Model"),
            info=_("Voice model used for conversion."),
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            best_default_index_path = match_index(model_file.value)
            index_file = gr.Dropdown(
                label=_("Index File"),
            info=_("Optional index file."),
                choices=get_indexes(),
                value=best_default_index_path,
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button(_("Unload Voice"))
            refresh_button = gr.Button(_("Refresh"))

            def _unload_and_cleanup():
                import_voice_converter().cleanup_model()
                return {"value": "", "__type__": "update"}, {"value": "", "__type__": "update"}

            unload_button.click(
                fn=_unload_and_cleanup,
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    gr.Markdown(_("Generate speech with EdgeTTS, then convert it with the selected RVC model."))
    tts_voice = gr.Dropdown(
        label=_("TTS Voices"),
        info=_("Select the TTS voice to use for the conversion."),
        choices=short_names,
        interactive=True,
        value=random.choice(short_names),
    )

    tts_rate = gr.Slider(
        minimum=-100,
        maximum=100,
        step=1,
        label=_("TTS Speed"),
        info=_("Increase or decrease TTS speed."),
        value=0,
        interactive=True,
    )

    with gr.Tabs():
        with gr.Tab(label=_("Text to Speech")):
            tts_text = gr.Textbox(
                label=_("Text to Synthesize"),
                info=_("Enter the text to synthesize."),
                placeholder=_("Enter text to synthesize"),
                lines=3,
            )
        with gr.Tab(label=_("File to Speech")):
            txt_file = gr.File(
                label=_("Upload a .txt file"),
                type="filepath",
            )
            input_tts_path = gr.Textbox(
                label=_("Input path for text file"),
                placeholder=_("The path to the text file that contains content for text to speech."),
                value="",
                interactive=True,
            )

    with gr.Accordion(_("Advanced Settings"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=_("Output Path for TTS Audio"),
                placeholder=_("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=_("Output Path for RVC Audio"),
                placeholder=_("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=_("Export Format"),
                info=_("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            seed = gr.Number(
                label=_("Inference Seed"),
                info=_("Specify any seed to be used for inference or leave at '0' for random outputs. ( Classic RVC behavior. ) \n **Ensure you don't leave this field empty.**"),
                value=0,
                interactive=True,
            )
            sid = gr.Dropdown(
                label=_("Speaker ID"),
                info=_("Select the speaker ID to use for the conversion."),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=_("Split Audio"),
                info=_("Split the audio into chunks for inference to obtain better results in some cases."),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=_("Autotune"),
                info=_("Apply a soft autotune to your inferences, recommended for singing conversions."),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=_("Autotune Strength"),
                info=_("Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."),
                visible=False,
                value=1,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=_("Clean Audio"),
                info=_("Clean your audio output using noise detection algorithms, recommended for speaking audios."),
                visible=True,
                value=True,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=_("Clean Strength"),
                info=_("Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."),
                visible=True,
                value=0.5,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=_("Pitch"),
                info=_("Set the pitch of the audio, the higher the value, the higher the pitch."),
                value=0,
                interactive=True,
            )
            filter_radius = gr.Slider(
                minimum=0,
                maximum=7,
                label=_("Filter Radius"),
                info=_("If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."),
                value=3,
                step=1,
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=_("Search Feature Ratio"),
                info=_("Index influence. Lower values can reduce artifacts."),
                value=0.75,
                interactive=True,
            )
            index_k = gr.Slider(
                minimum=1,
                maximum=32,
                step=1,
                label=_("Index Neighbours"),
                info=_("Frames averaged per match. Fewer keeps the training voice's idiosyncratic articulation; more averages toward its mean voice."),
                value=8,
                interactive=True,
            )
            index_power = gr.Slider(
                minimum=0,
                maximum=8,
                step=0.25,
                label=_("Index Sharpness"),
                info=_("How strongly closer neighbours outweigh further ones. 0 averages them equally; high values use the nearest alone."),
                value=2.0,
                interactive=True,
            )
            index_continuity = gr.Slider(
                minimum=0,
                maximum=4,
                step=0.1,
                label=_("Index Continuity"),
                info=_("Favours matches that continue the previous frame's, so the retrieval stops jumping between unrelated parts of the dataset. Needs an index built by this fork."),
                value=0.5,
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=_("Volume Envelope"),
                info=_("Mix the converted and input loudness envelopes."),
                value=1,
                interactive=True,
            )
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=_("Protect Voiceless Consonants"),
                info=_("Protect voiceless consonants. Higher values reduce index influence."),
                value=0.5,
                interactive=True,
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
                info=_("Model used for learning speaker embedding."),
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
                    folder_name_input = gr.Textbox(
                        label=_("Folder Name"), interactive=True
                    )
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
            f0_file = gr.File(
                label=_("Edited F0 curve"),
                visible=True,
            )

    convert_button = gr.Button(_("Convert"))

    with gr.Row():
        vc_output1 = gr.Textbox(
            label=_("Output Information"),
            info=_("TTS status."),
        )
        vc_output2 = gr.Audio(label=_("Export Audio"))

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

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
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, sid, sid],
    )
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[input_tts_path, txt_file],
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
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
        show_progress="hidden",
    )
    convert_button.click(
        fn=run_tts_script,
        inputs=[
            input_tts_path,
            tts_text,
            tts_voice,
            tts_rate,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            f0_method,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            f0_file,
            embedder_model,
            embedder_model_custom,
            sid,
            seed,
            index_k,
            index_power,
            index_continuity,
        ],
        outputs=[vc_output1, vc_output2],
    )
