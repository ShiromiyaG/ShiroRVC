import os

import gradio as gr

from core import (
    list_style_models,
    run_style_extract_script,
    run_style_train_script,
    stop_style_script,
    style_data_summary,
)
from rvc.configs.config import get_number_of_gpus
from rvc.lib.i18n import _
from rvc.lib.utils import EMBEDDER_FEATURE_DIMS

now_dir = os.getcwd()
datasets_path = os.path.join(now_dir, "assets", "datasets")
logs_path = os.path.join(now_dir, "logs")

PRETRAIN = "New base (pretrain)"
FINETUNE = "Fine-tune a base"
FOLDER = "Audio folder"
EXPERIMENT = "RVC experiment"


def list_datasets():
    if not os.path.isdir(datasets_path):
        return []
    return sorted(
        os.path.relpath(os.path.join(datasets_path, d), now_dir)
        for d in os.listdir(datasets_path)
        if os.path.isdir(os.path.join(datasets_path, d)) and not d.startswith((".", "_"))
    )


def list_runs():
    if not os.path.isdir(logs_path):
        return []
    return sorted(d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d)))


def describe_run(name):
    if not name:
        return ""
    summary = style_data_summary(name)
    return _("Style data: ") + summary if summary else _("No style data yet: run step 1.")


def style_tab():
    gr.Markdown(
        _("Style models learn how a singer shapes pitch (vibrato, scoops, phrase endings) and are used from the inference tab next to any voice model. They are trained on whole songs, so they have their own data step instead of the RVC preprocessing.")
    )
    with gr.Row(equal_height=True):
        model_name = gr.Dropdown(
            label=_("Name"),
            info=_("Everything goes to logs/<name>. Reusing an RVC model's name can reuse its extraction."),
            choices=list_runs(),
            value=None,
            allow_custom_value=True,
            interactive=True,
            scale=3,
        )
        mode = gr.Radio(
            label=_("Train"),
            info=_("A base learns style in general; a fine-tune learns one singer from a base."),
            choices=[PRETRAIN, FINETUNE],
            value=FINETUNE,
            interactive=True,
            scale=3,
        )
        gpu = gr.Textbox(
            label=_("GPU"),
            info=_("GPU id, or '-' for CPU."),
            value=str(get_number_of_gpus()).split("-")[0],
            interactive=True,
            scale=1,
        )
    with gr.Row(equal_height=True) as base_row:
        base_path = gr.Dropdown(
            label=_("Style base"),
            info=_("style_base.pt to fine-tune; its embedder is used."),
            choices=list_style_models("base"),
            value=None,
            allow_custom_value=True,
            interactive=True,
            scale=6,
        )
        refresh_bases = gr.Button(_("Refresh"), size="sm", scale=0, min_width=100)
    with gr.Row(visible=False) as embedder_row:
        embedder = gr.Radio(
            label=_("Embedder"),
            info=_("Content embedder of the new base; train one base per embedder."),
            choices=list(EMBEDDER_FEATURE_DIMS),
            value="contentvec",
            interactive=True,
        )
    run_info = gr.Markdown("")

    with gr.Accordion(_("1. Style data"), open=True):
        source = gr.Radio(
            label=_("Source"),
            info=_("An audio folder is read whole, any length. An RVC experiment reuses its slices, F0 and features."),
            choices=[FOLDER, EXPERIMENT],
            value=FOLDER,
            interactive=True,
        )
        with gr.Row(equal_height=True) as folder_row:
            dataset_path = gr.Dropdown(
                label=_("Dataset folder"),
                info=_("Subfolders named <id>_<name> are speakers; otherwise it is one speaker."),
                choices=list_datasets(),
                value=None,
                allow_custom_value=True,
                interactive=True,
                scale=6,
            )
            refresh_datasets = gr.Button(_("Refresh"), size="sm", scale=0, min_width=100)
        recompute = gr.Checkbox(
            label=_("Recompute F0 and features"),
            info=_("Ignore the experiment's own; needed when it was not extracted with RMVPE and this embedder."),
            value=False,
            interactive=True,
            visible=False,
        )
        extract_button = gr.Button(_("Extract style data"), variant="primary")

    with gr.Accordion(_("2. Training"), open=True):
        with gr.Row():
            steps = gr.Slider(
                0, 300000, 0, step=100,
                label=_("Steps"),
                info=_("0 uses the default: 200000 for a base, 5000 for a fine-tune."),
                interactive=True,
            )
            precision = gr.Radio(
                label=_("Precision"),
                info=_("fp16 adds a gradient scaler; TF32 is on."),
                choices=["bf16", "fp16", "fp32"],
                value="bf16",
                interactive=True,
            )
        with gr.Row():
            batch_size = gr.Slider(
                0, 128, 0, step=1,
                label=_("Batch size"),
                info=_("0 uses the default: 32 for a base, 16 for a fine-tune."),
                interactive=True,
            )
            checkpointing = gr.Checkbox(
                label=_("Gradient checkpointing"),
                info=_("Less VRAM for slower steps; try it before lowering the batch size."),
                value=False,
                interactive=True,
            )
        with gr.Row():
            speech = gr.Textbox(
                label=_("Speech speakers"),
                info=_("Speaker ids that are speech, e.g. 0-4 or 0-4,7; capped at 15% of each batch. Empty for none."),
                value="0-4",
                interactive=True,
                visible=False,
            )
        with gr.Row():
            train_button = gr.Button(_("Train style model"), variant="primary")
            stop_button = gr.Button(_("Stop"))

    output = gr.Textbox(label=_("Output Information"), value="", max_lines=6, interactive=False)

    def on_mode(value):
        pretrain = value == PRETRAIN
        return gr.update(visible=not pretrain), gr.update(visible=pretrain), gr.update(visible=pretrain)

    def on_source(value):
        return gr.update(visible=value == FOLDER), gr.update(visible=value == EXPERIMENT)

    def extract(name, mode_value, base, embedder_value, source_value, folder, recompute_value, gpu_value):
        if not name:
            return _("Give the run a name.")
        if mode_value == FINETUNE and not base:
            return _("Pick the style base to fine-tune, or switch to a new base.")
        if source_value == FOLDER and not folder:
            return _("Pick a dataset folder.")
        return run_style_extract_script(
            name,
            dataset_path=folder if source_value == FOLDER else None,
            embedder_model=embedder_value,
            base_path=base if mode_value == FINETUNE else None,
            gpu=gpu_value,
            recompute=recompute_value,
        )

    def train(
        name, mode_value, base, steps_value, precision_value, gpu_value, speech_value, batch_value, checkpointing_value,
    ):
        if not name:
            return _("Give the run a name.")
        if mode_value == FINETUNE and not base:
            return _("Pick the style base to fine-tune, or switch to a new base.")
        return run_style_train_script(
            name,
            base_path=base if mode_value == FINETUNE else None,
            steps=steps_value,
            precision=precision_value,
            gpu=gpu_value,
            speech_speakers=speech_value if mode_value == PRETRAIN else None,
            batch_size=batch_value,
            checkpointing=checkpointing_value,
        )

    mode.change(fn=on_mode, inputs=[mode], outputs=[base_row, embedder_row, speech], show_progress="hidden")
    source.change(fn=on_source, inputs=[source], outputs=[folder_row, recompute], show_progress="hidden")
    model_name.change(fn=describe_run, inputs=[model_name], outputs=[run_info], show_progress="hidden")
    refresh_bases.click(fn=lambda: gr.update(choices=list_style_models("base")), inputs=[], outputs=[base_path])
    refresh_datasets.click(fn=lambda: gr.update(choices=list_datasets()), inputs=[], outputs=[dataset_path])

    extract_button.click(
        fn=lambda: _("Extracting; progress is shown in the terminal window."),
        inputs=[],
        outputs=[output],
        show_progress="hidden",
    ).then(
        fn=extract,
        inputs=[model_name, mode, base_path, embedder, source, dataset_path, recompute, gpu],
        outputs=[output],
        show_progress="hidden",
    ).then(fn=describe_run, inputs=[model_name], outputs=[run_info], show_progress="hidden")

    train_button.click(
        fn=lambda: _("Training; progress is shown in the terminal window and in TensorBoard."),
        inputs=[],
        outputs=[output],
        show_progress="hidden",
    ).then(
        fn=train,
        inputs=[model_name, mode, base_path, steps, precision, gpu, speech, batch_size, checkpointing],
        outputs=[output],
        show_progress="hidden",
    ).then(fn=lambda: gr.update(choices=list_runs()), inputs=[], outputs=[model_name])

    stop_button.click(fn=stop_style_script, inputs=[], outputs=[output], show_progress="hidden")
