import os
import shutil
from multiprocessing import cpu_count

import gradio as gr

from core import (
    list_experiment_speakers,
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    run_rectified_flow_train_script,
    run_rectified_infer_script,
    run_rectified_vocoder_train_script,
    stop_rectified_train_script,
    unload_rectified_models,
)
from rvc.configs.config import (
    get_gpu_info,
    get_number_of_gpus,
    get_training_precision,
    max_vram_gpu,
)
from rvc.configs.vocoders import RECTIFIED_EXTRACTION
from rvc.lib import catalog
from rvc.lib.i18n import _
from rvc.lib.model_bundle import is_model_bundle
from rvc.lib.terminal import DEFAULT_CPU_THREADS
from rvc.rectified.common import PRETRAINED_DIR, default_pretrained, describe_flow, list_exports, list_pretrained
from rvc.rectified.flow_model import RESCALE_MODES, SAMPLERS, SCHEDULES
from tabs.inference.inference import F0_METHODS, EXPORT_FORMATS, save_to_wav, save_to_wav2
from tabs.train.descs import (
    AUDIO_FILE_SLICING_INFO,
    DATASET_FORMAT_INFO,
    INDEX_SINGLE_SPEAKER_INFO,
    NORMALIZATION_INFO,
    PITCH_EXTRACTION_INFO,
    PREPROCESS_RMS_VALUE_INFO,
    RESAMPLER_INFO,
    TORCH_COMPILE_MODE_CHOICES,
    TORCH_COMPILE_MODE_INFO,
    TORCH_COMPILE_MODE_LABEL,
)

#: The rectified recipe ships one configuration, at 44.1 kHz: SingingVocoders' mel.
SAMPLE_RATE = 44100

TRAINING_STARTED = (
    "Training started. Epoch, step and loss progress is shown in the terminal window."
)


def flow_info(flow_path, submodel=None):
    """``describe_flow`` of a flow export or bundle, read without building it."""
    try:
        return describe_flow(flow_path, submodel)
    except Exception:
        return {"submodels": [], "submodel": "", "speakers": [0], "vocoder": ""}


def save_uploaded_pretrained(path):
    """Copy an uploaded .pth into the rectified pretrain folder."""
    if not path or not path.endswith(".pth"):
        gr.Info(_("Invalid pretrained file."))
        return
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    shutil.copy(path, os.path.join(PRETRAINED_DIR, os.path.basename(path)))
    gr.Info(_("Pretrained file added."))


def _first(items):
    return items[0] if items else None


def index_for(flow_path):
    """The RVC index of the experiment a flow export was trained in, or ""."""
    if not flow_path:
        return ""
    folder = os.path.dirname(os.path.dirname(os.path.abspath(flow_path)))
    return catalog.guess_index_for(os.path.join(folder, os.path.basename(flow_path)))


def rectified_inference_tab():
    flows = list_exports("flow")
    vocoders = list_exports("vocoder")
    initial = flow_info(_first(flows))
    with gr.Column():
        with gr.Row():
            flow_model = gr.Dropdown(
                label=_("Flow Model"),
                info=_("Rectified-flow model, or a model bundle holding one: content, pitch and speaker to mel."),
                choices=flows,
                value=_first(flows),
                interactive=True,
                allow_custom_value=True,
            )
            flow_submodel = gr.Dropdown(
                label=_("Bundle Model"),
                info=_("The flow to use inside the bundle."),
                choices=initial["submodels"],
                value=initial["submodel"] or None,
                visible=bool(initial["submodels"]),
                interactive=True,
            )
            vocoder_model = gr.Dropdown(
                label=_("Vocoder Model"),
                info=_("Rectified NSF-BigVGAN export or OpenVPI NSF-HiFiGAN checkpoint that renders the mel."),
                choices=vocoders,
                value=_first(vocoders),
                interactive=True,
                allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label=_("Index File"),
                info=_("Optional RVC index over the voice model's training features. "
                       "A bundle's own index is used when it has one."),
                choices=catalog.list_indexes(),
                value=index_for(_first(flows)),
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button(_("Unload the models"))
            refresh_button = gr.Button(_("Refresh models, indexes and audios"))

    with gr.Column():
        upload_audio = gr.Audio(label=_("Upload Audio"), type="filepath", editable=False)
        audio_paths = catalog.list_audios()
        audio = gr.Dropdown(
            label=_("Select Audio Input"),
            info=_("Audio to convert."),
            choices=audio_paths,
            value=_first(audio_paths) or "",
            interactive=True,
            allow_custom_value=True,
        )

    with gr.Tabs(elem_classes=["rvc-settings-tabs"]):
        with gr.Tab(_("Pitch")):
            with gr.Row():
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=_("Pitch"),
                    info=_("Pitch shift in semitones. 12 = one octave up."),
                    value=0,
                    interactive=True,
                )
                f0_method = gr.Radio(
                    label=_("Pitch extraction algorithm"),
                    info=_("Pitch algorithm. RMVPE is the recommended default."),
                    choices=F0_METHODS,
                    value="rmvpe",
                    interactive=True,
                )
            with gr.Row():
                autotune = gr.Checkbox(
                    label=_("Autotuning"),
                    info=_("Apply autotune."),
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
            formant_shift = gr.Slider(
                minimum=-5,
                maximum=5,
                step=0.5,
                label=_("Formant Shift"),
                info=_("Moves the formants by semitones, apart from the pitch. 0 keeps the voice's own."),
                value=0,
                interactive=True,
            )
            with gr.Row():
                f0_median = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    label=_("Pitch Median Filter"),
                    info=_("Median over this many 10 ms frames either side of each voiced frame, against pitch jitter. 0 turns it off."),
                    value=0,
                    interactive=True,
                )
                f0_octave_fix = gr.Checkbox(
                    label=_("Fix Octave Errors"),
                    info=_("Folds pitch that jumps an octave away from its surroundings back into place."),
                    value=False,
                    interactive=True,
                )

        with gr.Tab(_("Sampling")):
            with gr.Row():
                steps = gr.Slider(
                    minimum=1,
                    maximum=64,
                    step=1,
                    label=_("Steps"),
                    info=_("ODE steps from noise to mel. More is slower and usually cleaner."),
                    value=16,
                    interactive=True,
                )
                sampler = gr.Radio(
                    label=_("Sampler"),
                    info=_("Heun costs two model passes per step and is more accurate per step."),
                    choices=list(SAMPLERS),
                    value="euler",
                    interactive=True,
                )
            with gr.Row():
                schedule = gr.Radio(
                    label=_("Step Schedule"),
                    info=_("Spacing of the steps. Sway puts more of them early, where the mel's structure is decided; logit-normal matches the times the model was trained on most."),
                    choices=list(SCHEDULES),
                    value="uniform",
                    interactive=True,
                )
            with gr.Row():
                noise_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    label=_("Noise Temperature"),
                    info=_("Scale of the starting noise. Lower is steadier and less hissy but can sound flatter; 1 is what the model was trained on."),
                    value=1.0,
                    interactive=True,
                )
                flow_start = gr.Slider(
                    minimum=0.0,
                    maximum=0.95,
                    step=0.05,
                    label=_("Flow Start"),
                    info=_("Flow time sampling starts from, on the aux decoder's mel. Later is steadier and closer to that blurrier mel. 0 or anything before the model's own start uses the model's own."),
                    value=0.0,
                    interactive=True,
                )

        with gr.Tab(_("Guidance")):
            with gr.Row():
                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    step=0.1,
                    label=_("Speaker Guidance"),
                    info=_("Classifier-free guidance on the speaker. 1 turns it off; higher pushes toward the target voice."),
                    value=1.0,
                    interactive=True,
                )
                content_guidance = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label=_("Content Guidance"),
                    info=_("Pushes away from a blurred copy of the content for clearer articulation. 0 turns it off."),
                    value=0.0,
                    interactive=True,
                )
            with gr.Row():
                guidance_from = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label=_("Guidance From"),
                    info=_("Flow time (0 noise, 1 mel) guidance starts at."),
                    value=0.0,
                    interactive=True,
                )
                guidance_until = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label=_("Guidance Until"),
                    info=_("Flow time guidance stops at. Stopping early lets stronger guidance through without oversaturating the fine detail."),
                    value=1.0,
                    interactive=True,
                )
            with gr.Row():
                guidance_rescale = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label=_("Guidance Rescale"),
                    info=_("Pulls guided output back to the unguided level so strong guidance does not oversaturate."),
                    value=0.7,
                    interactive=True,
                )
                rescale_mode = gr.Radio(
                    label=_("Rescale Over"),
                    info=_("Global measures the level over the whole pass, silences included; frame measures it per frame, so it does not depend on how much silence there is."),
                    choices=list(RESCALE_MODES),
                    value="global",
                    interactive=True,
                )

        with gr.Tab(_("Index")):
            with gr.Row():
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
                    info=_("Frames averaged per match. Fewer keeps the training "
                           "voice's idiosyncratic articulation; more averages "
                           "toward its mean voice."),
                    value=8,
                    interactive=True,
                )
            with gr.Row():
                index_power = gr.Slider(
                    minimum=0,
                    maximum=8,
                    step=0.25,
                    label=_("Index Sharpness"),
                    info=_("How strongly closer neighbours outweigh further ones. "
                           "0 averages them equally; high values use the nearest alone."),
                    value=2.0,
                    interactive=True,
                )
                index_continuity = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=0.1,
                    label=_("Index Continuity"),
                    info=_("Favours matches that continue the previous frame's, so "
                           "the retrieval stops jumping between unrelated parts of "
                           "the dataset. Needs an index built by this fork."),
                    value=0.5,
                    interactive=True,
                )

        with gr.Tab(_("Voice")):
            with gr.Row():
                sid = gr.Dropdown(
                    label=_("Speaker ID"),
                    info=_("Speaker ID for multi-speaker models."),
                    choices=initial["speakers"],
                    value=0,
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

        with gr.Tab(_("Output")):
            output_path = gr.Textbox(
                label=_("Path for infer outputs"),
                placeholder=os.path.join("assets", "audios", "filename_output.wav"),
                info=_("Optional output path. Empty uses assets/audios."),
                value="",
                interactive=True,
            )
            with gr.Row():
                export_format = gr.Radio(
                    label=_("Export Format"),
                    info=_("Output audio format."),
                    choices=EXPORT_FORMATS,
                    value="WAV",
                    interactive=True,
                )
                seed = gr.Number(
                    label=_("Inference Seed"),
                    info=_("Seed for reproducible output. Use 0 for random output."),
                    value=0,
                    interactive=True,
                )
            with gr.Row():
                split_audio = gr.Checkbox(
                    label=_("Audio splitting"),
                    info=_("Split input at silence regions."),
                    value=False,
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
                content_context = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.5,
                    label=_("Content Context"),
                    info=_("Seconds of audio the content encoder sees either side of each 30 s pass. Raise it if the joins between passes are audible."),
                    value=2.0,
                    interactive=True,
                )

    convert_button = gr.Button(_("Convert"), variant="primary")
    with gr.Row():
        output_info = gr.Textbox(label=_("Output Information"), info=_("Inference status."))
        output_audio = gr.Audio(label=_("Export Audio"))

    def convert(
        audio, output_path, flow_model, vocoder_model, sid, pitch, f0_method,
        steps, sampler, cfg_scale, autotune, autotune_strength, seed, export_format,
        index_path, index_rate, index_k, index_power, index_continuity, protect,
        formant_shift, content_guidance, guidance_rescale, split_audio, silence_gate_db,
        noise_temperature, flow_start, guidance_from, guidance_until, rescale_mode, schedule,
        f0_median, f0_octave_fix, content_context, flow_submodel,
    ):
        if not flow_model or not vocoder_model:
            return _("Pick a flow model and a vocoder model."), None
        if not audio:
            return _("Pick an audio file to convert."), None
        if not output_path or not output_path.strip():
            output_path = catalog.default_output_path(audio)
        elif os.path.isdir(output_path):
            name = os.path.basename(catalog.default_output_path(audio))
            output_path = os.path.join(output_path, name)
        else:
            stem, ext = os.path.splitext(output_path)
            if ext.lower().lstrip(".") in {"wav", "mp3", "flac", "ogg", "m4a"}:
                export_format = ext.lstrip(".").upper()
            output_path = stem + ".wav"
        return run_rectified_infer_script(
            input_path=audio,
            output_path=output_path,
            flow_path=flow_model,
            vocoder_path=vocoder_model,
            sid=sid,
            pitch=pitch,
            f0_method=f0_method,
            steps=steps,
            sampler=sampler,
            cfg_scale=cfg_scale,
            f0_autotune=autotune,
            f0_autotune_strength=autotune_strength,
            seed=seed,
            export_format=export_format,
            index_path=index_path,
            index_rate=index_rate,
            index_k=index_k,
            index_power=index_power,
            index_continuity=index_continuity,
            protect=protect,
            formant_shift=formant_shift,
            content_guidance=content_guidance,
            guidance_rescale=guidance_rescale,
            split_audio=split_audio,
            silence_gate_db=silence_gate_db,
            noise_temperature=noise_temperature,
            flow_start=flow_start,
            guidance_from=guidance_from,
            guidance_until=guidance_until,
            rescale_mode=rescale_mode,
            schedule=schedule,
            f0_median=f0_median,
            f0_octave_fix=f0_octave_fix,
            content_context=content_context,
            flow_submodel=flow_submodel if is_model_bundle(flow_model) else "",
        )

    def refresh():
        return (
            gr.update(choices=list_exports("flow")),
            gr.update(choices=list_exports("vocoder")),
            gr.update(choices=catalog.list_indexes()),
            gr.update(choices=catalog.list_audios()),
        )

    def voice_updates(described):
        """Speakers of the flow and the vocoder it was trained with, when a
        file of that name is here."""
        speakers = described["speakers"]
        vocoder = gr.update(value=described["vocoder"]) if described["vocoder"] else gr.update()
        return gr.update(choices=speakers, value=speakers[0]), vocoder

    def on_flow_change(path):
        """The bundle's flows, the speakers and vocoder of the flow, and its
        experiment's index."""
        described = flow_info(path)
        index = gr.update(choices=catalog.list_indexes(), value=index_for(path))
        submodels = gr.update(
            choices=described["submodels"], value=described["submodel"] or None,
            visible=bool(described["submodels"]),
        )
        return (*voice_updates(described), index, submodels)

    refresh_button.click(fn=refresh, inputs=[], outputs=[flow_model, vocoder_model, index_file, audio])
    unload_button.click(fn=unload_rectified_models, inputs=[], outputs=[])
    flow_model.change(
        fn=on_flow_change, inputs=[flow_model], outputs=[sid, vocoder_model, index_file, flow_submodel],
        show_progress="hidden",
    )
    flow_submodel.change(
        fn=lambda path, submodel: voice_updates(flow_info(path, submodel)),
        inputs=[flow_model, flow_submodel],
        outputs=[sid, vocoder_model],
        show_progress="hidden",
    )
    autotune.change(
        fn=lambda enabled: gr.update(visible=bool(enabled)),
        inputs=[autotune],
        outputs=[autotune_strength],
        show_progress="hidden",
    )
    upload_audio.upload(fn=save_to_wav2, inputs=[upload_audio], outputs=[audio, output_path])
    upload_audio.stop_recording(fn=save_to_wav, inputs=[upload_audio], outputs=[audio, output_path])
    convert_button.click(
        fn=convert,
        inputs=[
            audio, output_path, flow_model, vocoder_model, sid, pitch, f0_method,
            steps, sampler, cfg_scale, autotune, autotune_strength, seed, export_format,
            index_file, index_rate, index_k, index_power, index_continuity, protect,
            formant_shift, content_guidance, guidance_rescale, split_audio, silence_gate_db,
            noise_temperature, flow_start, guidance_from, guidance_until, rescale_mode, schedule,
            f0_median, f0_octave_fix, content_context, flow_submodel,
        ],
        outputs=[output_info, output_audio],
    )
    return refresh, [flow_model, vocoder_model, index_file, audio]


def _run_controls(prefix: str):
    """Batch size, epochs and save frequency, as in the RVC training tab."""
    with gr.Row():
        batch_size = gr.Slider(
            1,
            128,
            max_vram_gpu(0),
            step=1,
            label=_("Batch Size"),
            info=_("Clips per step. Lower it if training runs out of VRAM."),
            interactive=True,
        )
        save_every = gr.Slider(
            1,
            100,
            10,
            step=1,
            label=_("Saving frequency"),
            info=_("Save a checkpoint every N epochs."),
            interactive=True,
        )
        total_epochs = gr.Slider(
            1,
            10000,
            250,
            step=1,
            label=_("Total Epochs"),
            info=_("Total training epochs."),
            interactive=True,
        )
    return batch_size, save_every, total_epochs


def _checkpoint_mode():
    """What a run keeps of its training checkpoints; ``CHECKPOINT_MODES``."""
    return gr.Radio(
        label=_("Keep Checkpoints"),
        info=_("Training checkpoints hold the optimizer, so a stopped run resumes from them; "
               "the flow's are close to 1 GB each. Without them only the exports are saved, "
               "and a stopped run starts over. Exported models are always kept."),
        choices=[(_("Latest only"), "latest"), (_("All"), "all"),
                 (_("Don't save"), "none")],
        value="latest",
        interactive=True,
    )


def _checkpoint_controls(prefix: str):
    gr.Markdown(f"#### {_('Checkpoints')}")
    with gr.Row():
        with gr.Column(min_width=0):
            checkpoints = _checkpoint_mode()
        with gr.Column(min_width=0):
            fresh = gr.Checkbox(
                label=_("Fresh Training"),
                info=_("Ignore this run's checkpoints and start over."),
                value=False,
                interactive=True,
            )
    return checkpoints, fresh


def _start_stop(start_fn, inputs):
    with gr.Row():
        start = gr.Button(_("Start Training"), variant="primary")
        stop = gr.Button(_("Stop Training"), variant="stop")
    output = gr.Textbox(
        label=_("Output Information"),
        info=_("Training status."),
        value="",
        max_lines=8,
        interactive=False,
    )
    start.click(
        fn=lambda: TRAINING_STARTED, inputs=[], outputs=[output], show_progress="hidden"
    ).then(fn=start_fn, inputs=inputs, outputs=[output], show_progress="hidden")
    stop.click(
        fn=lambda: "Stopping training - letting any checkpoint write finish first...",
        inputs=[],
        outputs=[output],
        show_progress="hidden",
    ).then(fn=stop_rectified_train_script, inputs=[], outputs=[output], show_progress="hidden")


def rectified_training_tab():
    with gr.Row(equal_height=True):
        model_name = gr.Dropdown(
            label=_("Model Name"),
            info=_("Name of the new model. Its data lives in logs/<name>, like an RVC model."),
            choices=catalog.list_training_models(),
            value="example-rectified-model",
            interactive=True,
            allow_custom_value=True,
            scale=4,
        )
        refresh_button = gr.Button(_("Refresh"), scale=1)
    gr.Markdown(_("Sample rate: 44100 Hz. The rectified recipe has one configuration."))

    with gr.Accordion(_("CPU / GPU settings"), open=False):
        with gr.Row():
            with gr.Column():
                cpu_threads = gr.Slider(
                    1,
                    min(cpu_count(), 192),
                    DEFAULT_CPU_THREADS,
                    step=1,
                    label=_("CPU Threads"),
                    info=_("CPU threads used during preprocessing and extraction."),
                    interactive=True,
                )
            with gr.Column():
                gpu = gr.Textbox(
                    label=_("GPU ID"),
                    info=_("GPU IDs separated by '-', for extraction and training. Batch size is per GPU."),
                    placeholder=_("0 to ∞ separated by -"),
                    value=str(get_number_of_gpus()),
                    interactive=True,
                )
                gr.Textbox(
                    label=_("GPU Information"),
                    info=_("Detected GPU information."),
                    value=get_gpu_info(),
                    interactive=False,
                )

    with gr.Tab(f"1. {_('Preprocessing')}"):
        dataset_path = gr.Dropdown(
            label=_("Dataset Path"),
            info=_("Folder containing the training audio."),
            choices=catalog.list_dataset_folders(),
            allow_custom_value=True,
            interactive=True,
        )
        with gr.Row(elem_classes=["rvc-preprocess-options"]):
            with gr.Column(min_width=0):
                dataset_format = gr.Radio(
                    label=_("Dataset Format"),
                    info=_(DATASET_FORMAT_INFO),
                    choices=["WAV", "FLAC"],
                    value="WAV",
                    interactive=True,
                )
            with gr.Column(min_width=0):
                loading_resampling = gr.Radio(
                    label=_("Resampling & Loading Handler"),
                    info=_(RESAMPLER_INFO),
                    choices=["ffmpeg", "librosa"],
                    value="ffmpeg",
                    interactive=True,
                )
            with gr.Column(min_width=0):
                normalization_mode = gr.Radio(
                    label=_("Loudness Normalization"),
                    info=_(NORMALIZATION_INFO),
                    choices=["none", "post_peak", "pre_peak_rvc", "pre_loudness"],
                    value="pre_peak_rvc",
                    interactive=True,
                )
        rms_norm_db = gr.Slider(
            -24.0, -3.0, -16.0, step=1.0,
            label=_("Target Level (LUFS)"),
            info=_(PREPROCESS_RMS_VALUE_INFO),
            interactive=True,
            visible=False,
        )
        cut_preprocess = gr.Radio(
            label=_("Audio cutting"),
            info=_(AUDIO_FILE_SLICING_INFO),
            choices=["Skip", "Simple", "Automatic", "New Automatic"],
            value="New Automatic",
            interactive=True,
        )
        with gr.Row():
            chunk_len = gr.Slider(
                0.5, 30.0, 3.0, step=0.1,
                label=_("Chunk length (sec)"),
                info=_("Chunk length for Simple cutting."),
                interactive=True,
            )
            overlap_len = gr.Slider(
                0.0, 0.42, 0.40, step=0.01,
                label=_("Overlap length"),
                info=_("Overlap between Simple chunks, in seconds."),
                interactive=True,
            )
        with gr.Row():
            process_effects = gr.Checkbox(
                label=_("DC / high-pass filtering"),
                info=_("Remove DC offset and low-frequency noise."),
                value=True,
                interactive=True,
            )
            noise_reduction = gr.Checkbox(
                label=_("Noise Reduction"),
                info=_("Apply spectral-gating noise reduction."),
                value=False,
                interactive=True,
            )
        clean_strength = gr.Slider(
            minimum=0,
            maximum=1,
            label=_("Noise Reduction Strength"),
            info=_("Higher values apply stronger cleanup."),
            visible=False,
            value=0.5,
            interactive=True,
        )
        preprocess_button = gr.Button(_("Preprocess Dataset"), variant="primary")
        preprocess_output = gr.Textbox(
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
            )
            embedder_model = gr.Radio(
                label=_("Embedder Model"),
                info=_("Content features the flow model is conditioned on."),
                choices=["contentvec", "spin_v2"],
                value="contentvec",
                interactive=True,
            )
        include_mutes = gr.Slider(
            0, 10, 5, step=1,
            label=_("Silent ( 'mute' ) files for training."),
            info=_("Add silent examples so the model can reproduce silence."),
            interactive=True,
        )
        feature_precision = gr.Radio(
            label=_("Feature Precision"),
            info=_("How the extracted embeddings are stored. fp16 halves the cache on disk."),
            choices=["fp32", "fp16"],
            value="fp32",
            interactive=True,
        )
        extract_button = gr.Button(_("Extract Features"), variant="primary")
        extract_output = gr.Textbox(
            label=_("Output Information"),
            info=_("Extraction status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(f"3. {_('Voice Model')}"):
        gr.Markdown(
            _("The rectified-flow voice model: content, pitch, loudness and "
              "speaker to a mel spectrogram.")
        )
        flow_batch, flow_save, flow_epochs = _run_controls("flow")

        gr.Markdown(f"#### {_('Starting point')}")
        with gr.Row():
            with gr.Column(min_width=0):
                flow_pretrained = gr.Checkbox(
                    label=_("Pretrained"),
                    info=_("Fine-tune the default pretrained flow for the embedder the experiment "
                           "was extracted with: pretrain_flow_contentvec.pth or pretrain_flow_spin_v2.pth "
                           "in rvc/models/pretraineds/rectified."),
                    value=True,
                    interactive=True,
                )
            with gr.Column(min_width=0):
                flow_custom = gr.Checkbox(
                    label=_("Custom Pretrained"),
                    info=_("Pick the flow pretrain yourself."),
                    value=False,
                    interactive=True,
                )
            with gr.Column(min_width=0):
                flow_fresh = gr.Checkbox(
                    label=_("Fresh Training"),
                    info=_("Ignore this run's checkpoints and start over."),
                    value=False,
                    interactive=True,
                )
        # A Column, not a Group: a hidden Group keeps its border as a stray line.
        with gr.Column(visible=False) as flow_custom_settings:
            with gr.Row():
                custom_flow = gr.Dropdown(
                    label=_("Custom Pretrained Flow"),
                    info=_("Flow to fine-tune. Its speakers are replaced by this dataset's."),
                    choices=list_pretrained("flow"),
                    interactive=True,
                    allow_custom_value=True,
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

        gr.Markdown(f"#### {_('Vocoder')}")
        with gr.Row():
            default_vocoder = default_pretrained("vocoder")
            custom_vocoder = gr.Dropdown(
                label=_("Vocoder"),
                info=_("Renders the audio previews and is paired with the exported model. "
                       "Empty uses the newest in rvc/models/pretraineds/rectified."),
                choices=list_exports("vocoder"),
                value=catalog.relative(default_vocoder) if default_vocoder else None,
                interactive=True,
                allow_custom_value=True,
            )

        gr.Markdown(f"#### {_('Optimisation')}")
        with gr.Row():
            with gr.Column(min_width=0):
                flow_compile = gr.Checkbox(
                    label=_("Compile backbone"),
                    info=_("torch.compile the flow backbone for faster training. The first steps take longer while the graph is built. Needs CUDA and Triton."),
                    value=False,
                    interactive=True,
                )
            with gr.Column(min_width=0):
                flow_compile_mode = gr.Radio(
                    label=_(TORCH_COMPILE_MODE_LABEL),
                    info=_(TORCH_COMPILE_MODE_INFO),
                    choices=list(TORCH_COMPILE_MODE_CHOICES),
                    value=TORCH_COMPILE_MODE_CHOICES[0],
                    interactive=True,
                    visible=False,
                )
        flow_compile.change(
            fn=lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[flow_compile],
            outputs=[flow_compile_mode],
            show_progress="hidden",
        )

        gr.Markdown(f"#### {_('Checkpoints')}")
        with gr.Row():
            flow_checkpoints = _checkpoint_mode()

        def start_flow(name, epochs, save, batch, gpu_ids, pretrained, custom,
                       flow_path, vocoder_path, checkpoints, fresh, compile_backbone, compile_mode):
            if pretrained and not custom:
                # The pretrain has to match the features the experiment was extracted with.
                embedder = catalog.experiment_embedder(name)
                flow_path = catalog.default_flow_pretrain(embedder)
                if not flow_path:
                    return _("No pretrained flow for {embedder} ({file}) in "
                             "rvc/models/pretraineds/rectified. Add it, pick a custom "
                             "pretrain, or turn off Pretrained.").format(
                        embedder=embedder, file=catalog.RECTIFIED_FLOW_PRETRAINS.get(embedder, "?"))
            elif not pretrained:
                flow_path = None
            elif not flow_path:
                return _("Pick a custom pretrained flow, or turn off Custom Pretrained.")
            return run_rectified_flow_train_script(
                model_name=name,
                total_epochs=epochs,
                save_every=save,
                batch_size=batch,
                gpu=gpu_ids,
                pretrained_flow=catalog.relative(flow_path) if flow_path else "",
                vocoder=catalog.relative(vocoder_path) if vocoder_path else "",
                checkpoints=checkpoints,
                fresh=fresh,
                precision=get_training_precision(),
                compile=compile_backbone,
                torch_compile_mode=compile_mode,
            )

        _start_stop(
            start_flow,
            [model_name, flow_epochs, flow_save, flow_batch, gpu, flow_pretrained,
             flow_custom, custom_flow, custom_vocoder, flow_checkpoints,
             flow_fresh, flow_compile, flow_compile_mode],
        )

    with gr.Tab(f"4. {_('Index')}"):
        gr.Markdown(
            _("The same retrieval index as RVC, built from this experiment's "
              "content features. An index built in the RVC Training tab for the "
              "same experiment works as is.")
        )
        with gr.Row():
            with gr.Column(min_width=0):
                index_algorithm = gr.Radio(
                    label=_("Index Algorithm"),
                    info=_("Index method for large datasets."),
                    choices=["Auto", "Faiss", "KMeans"],
                    value="Auto",
                    interactive=True,
                )
            with gr.Column(min_width=0):
                index_metric = gr.Radio(
                    label=_("Index Similarity"),
                    info=_("How neighbours are ranked. L2 is what upstream RVC "
                           "builds. Cosine compares direction only, so a quiet and "
                           "a loud take of the same sound match equally well."),
                    choices=["l2", "cosine"],
                    value="l2",
                    interactive=True,
                )
        with gr.Row():
            with gr.Column(min_width=0):
                index_single_speaker = gr.Checkbox(
                    label=_("Index one speaker only"),
                    info=_(INDEX_SINGLE_SPEAKER_INFO),
                    value=False,
                    interactive=True,
                )
            with gr.Column(min_width=0, visible=False) as index_speaker_row:
                index_speaker = gr.Dropdown(
                    label=_("Speaker to index"),
                    info=_("Read from the extracted features. Refresh after extracting."),
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        index_button = gr.Button(_("Generate Index"), variant="primary")
        index_output = gr.Textbox(
            label=_("Output Information"),
            info=_("Index status."),
            value="",
            max_lines=8,
            interactive=False,
        )

    with gr.Tab(_("Vocoder Pretrain")):
        gr.Markdown(
            _("Builds the NSF-BigVGAN vocoder pretrain, on a large multi-speaker "
              "dataset. Not needed to fine-tune a voice: the vocoder has no "
              "speaker input, so one pretrain renders every voice model.")
        )
        voc_batch, voc_save, voc_epochs = _run_controls("vocoder")
        gr.Markdown(f"#### {_('Starting point')}")
        with gr.Row():
            pretrained_g = gr.Dropdown(
                label=_("Pretrained Generator"),
                info=_("Optional. A vocoder checkpoint or export to fine-tune from."),
                choices=list_pretrained("vocoder_g"),
                value=None,
                interactive=True,
                allow_custom_value=True,
            )
            pretrained_d = gr.Dropdown(
                label=_("Pretrained Discriminator"),
                info=_("Optional. The matching D checkpoint."),
                choices=list_pretrained("vocoder_d"),
                value=None,
                interactive=True,
                allow_custom_value=True,
            )
        voc_checkpoints, voc_fresh = _checkpoint_controls("vocoder")

        def start_vocoder(name, epochs, save, batch, gpu_ids, g_path, d_path, checkpoints, fresh):
            return run_rectified_vocoder_train_script(
                model_name=name,
                total_epochs=epochs,
                save_every=save,
                batch_size=batch,
                gpu=gpu_ids,
                pretrained_g=g_path,
                pretrained_d=d_path,
                checkpoints=checkpoints,
                fresh=fresh,
                precision=get_training_precision(),
            )

        _start_stop(
            start_vocoder,
            [model_name, voc_epochs, voc_save, voc_batch, gpu, pretrained_g,
             pretrained_d, voc_checkpoints, voc_fresh],
        )

    def refresh():
        return (
            gr.update(choices=catalog.list_training_models()),
            gr.update(choices=catalog.list_dataset_folders()),
            gr.update(choices=list_pretrained("vocoder_g")),
            gr.update(choices=list_pretrained("vocoder_d")),
            gr.update(choices=list_pretrained("flow")),
            gr.update(choices=list_exports("vocoder")),
        )

    list_outputs = [model_name, dataset_path, pretrained_g, pretrained_d, custom_flow, custom_vocoder]
    refresh_button.click(fn=refresh, inputs=[], outputs=list_outputs)

    flow_custom.change(
        fn=lambda enabled: gr.update(visible=bool(enabled)),
        inputs=[flow_custom],
        outputs=[flow_custom_settings],
        show_progress="hidden",
    )
    flow_pretrained.change(
        fn=lambda pretrained, custom: (
            gr.update(visible=bool(pretrained)),
            gr.update(visible=bool(pretrained and custom)),
        ),
        inputs=[flow_pretrained, flow_custom],
        outputs=[flow_custom, flow_custom_settings],
        show_progress="hidden",
    )
    upload_pretrained.upload(
        fn=save_uploaded_pretrained, inputs=[upload_pretrained], outputs=[]
    ).then(fn=refresh, inputs=[], outputs=list_outputs, show_progress="hidden")

    for checkbox, target in ((noise_reduction, clean_strength),):
        checkbox.change(
            fn=lambda enabled: gr.update(visible=bool(enabled)),
            inputs=[checkbox],
            outputs=[target],
            show_progress="hidden",
        )
    normalization_mode.change(
        fn=lambda mode: gr.update(visible=mode == "pre_loudness"),
        inputs=[normalization_mode],
        outputs=[rms_norm_db],
    )
    def fill_index_speakers(name):
        speakers = [str(sid) for sid in list_experiment_speakers(name)] if name else []
        return gr.update(choices=speakers, value=speakers[0] if speakers else None)

    def generate_index(name, algorithm, metric, single, speaker):
        if not single:
            return run_index_script(name, algorithm, metric, "all")
        if speaker in (None, ""):
            return _("Pick a speaker, or turn off 'Index one speaker only'.")
        return run_index_script(name, algorithm, metric, speaker)

    index_single_speaker.change(
        fn=lambda enabled: gr.update(visible=bool(enabled)),
        inputs=[index_single_speaker],
        outputs=[index_speaker_row],
    ).then(fn=fill_index_speakers, inputs=[model_name], outputs=[index_speaker])
    model_name.change(fn=fill_index_speakers, inputs=[model_name], outputs=[index_speaker])
    index_button.click(
        fn=generate_index,
        inputs=[model_name, index_algorithm, index_metric, index_single_speaker, index_speaker],
        outputs=[index_output],
    )
    preprocess_button.click(
        fn=lambda *args: run_preprocess_script(args[0], args[1], SAMPLE_RATE, *args[2:]),
        inputs=[
            model_name, dataset_path, cpu_threads, cut_preprocess, process_effects,
            noise_reduction, clean_strength, chunk_len, overlap_len,
            normalization_mode, loading_resampling, dataset_format, rms_norm_db,
        ],
        outputs=[preprocess_output],
    )
    extract_button.click(
        fn=lambda name, method, threads, gpu_ids, embedder, mutes, precision: run_extract_script(
            name, method, threads, gpu_ids, SAMPLE_RATE, RECTIFIED_EXTRACTION, embedder, mutes, precision
        ),
        inputs=[model_name, f0_method, cpu_threads, gpu, embedder_model, include_mutes, feature_precision],
        outputs=[extract_output],
    )
    return refresh, list_outputs


def rectified_tab(tab=None):
    """Build the tab; with ``tab`` given, its lists are refreshed on selection."""
    with gr.Tab(_("Inference")):
        refresh_inference, inference_lists = rectified_inference_tab()
    with gr.Tab(_("Training")):
        refresh_training, training_lists = rectified_training_tab()
    if tab is not None:
        tab.select(
            fn=lambda: (*refresh_inference(), *refresh_training()),
            inputs=[],
            outputs=[*inference_lists, *training_lists],
            show_progress="hidden",
        )
