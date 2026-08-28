import os
import hashlib
import datetime
import json
from collections import OrderedDict
import gradio as gr
import traceback

from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_choices,
    get_vocoder_sample_rates,
    get_vocoder_spec,
    normalize_vocoder,
)

from rvc.lib.algorithm.synthesizers import vocoder_config_from_model
from rvc.lib.i18n import _
from rvc.lib.terminal import error as print_error


def extract_small_model(
    path: str,
    name: str,
    output_dir: str,
    sr: int,
    pitch_guidance: bool,
    version: str,
    vocoder: str = "hifi",
):
    import torch

    if not path:
        return "Error: Please upload a Generator checkpoint ( Big G network .pth file)."

    try:
        pitch_guidance = True
        if not output_dir:
            output_dir = "logs/EXTRACTED_SMALL_MODELS" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pth_file = f"{name}.pth"
        final_pth_path = os.path.join(output_dir, pth_file)

        ckpt = torch.load(path, map_location="cpu")

        if "model" in ckpt:
            ckpt = ckpt["model"]

        opt = OrderedDict(
            weight={
                key: value.half()
                for key, value in ckpt.items()
                if "enc_q" not in key
                and not key.startswith("chouwa_latent.posterior")

            }
        )

        vocoder_id = normalize_vocoder(vocoder)
        if int(sr) not in get_vocoder_sample_rates(vocoder_id):
            raise ValueError(
                f"{vocoder_id} does not provide a configuration for {sr} Hz."
            )
        with open(
            os.path.join("rvc", "configs", vocoder_id, f"{int(sr)}.json"),
            "r",
            encoding="utf-8",
        ) as config_file:
            vocoder_config = json.load(config_file)

        data_config = vocoder_config["data"]
        train_config = vocoder_config["train"]
        model_config = vocoder_config["model"]
        opt["config"] = [
            data_config["filter_length"] // 2 + 1,
            train_config["segment_size"] // data_config["hop_length"],
            model_config["inter_channels"],
            model_config["hidden_channels"],
            model_config["filter_channels"],
            model_config["n_heads"],
            model_config["n_layers"],
            model_config["kernel_size"],
            model_config["p_dropout"],
            model_config["resblock"],
            model_config["resblock_kernel_sizes"],
            model_config["resblock_dilation_sizes"],
            model_config["upsample_rates"],
            model_config["upsample_initial_channel"],
            model_config["upsample_kernel_sizes"],
            model_config["spk_embed_dim"],
            model_config["gin_channels"],
            data_config["sample_rate"],
        ]

        opt.update(
            {
                "sr": sr,
                "f0": int(pitch_guidance),
                "version": version,
                "creation_date": datetime.datetime.now().isoformat(),
                "speakers_id": opt["config"][15] if len(opt["config"]) > 15 else 1,
                "vocoder": get_vocoder_spec(vocoder_id)["label"],
                "vocoder_id": vocoder_id,
                "vocoder_architecture": vocoder_id,
                "vocoder_config": vocoder_config_from_model(model_config),
                "architecture_id": model_config.get(
                    "architecture_id", get_architecture_id(vocoder_id)
                ),
            }
        )

        torch.save(opt, final_pth_path)

        return f" Successfully extracted and saved model to {final_pth_path} ..."

    except Exception as error:
        print_error(f"Could not extract the model: {error}", tag="[EXPORT]")
        return f" Failed to extract model: {error}\n{traceback.format_exc()}"

def extract_small_model_tab():
    with gr.Column():
        gr.Markdown(
            _("""
            # Checkpoint Extractor ⚙️
            """)
        )

        with gr.Row():
            model_path_input = gr.File(
                label=_("1. Generator network checkpoint (.pth)"),
                file_types=[".pth"],
                file_count="single",
                interactive=True,
                scale=2
            )
            model_name_input = gr.Textbox(
                label=_("Output Model Name"),
                info=_("The output file will be saved as `<name>.pth`."),
                value="My_extracted_model_123",
                interactive=True,
                scale=1
            )
            output_dir_input = gr.Textbox(
                label=_("Output Directory"),
                info=_("The directory where the final .pth file will be saved."),
                value="logs/EXTRACTED_SMALL_MODELS",
                interactive=True,
                scale=1
            )

        with gr.Row():
            sr_input = gr.Dropdown(
                label=_("Sample Rate of the model (sr)"),
                choices=get_vocoder_sample_rates("hifi"),
                value=48000, 
                type="value",
                interactive=True,
                scale=1
            )
            vocoder_input = gr.Dropdown(
                label=_("Vocoder"),
                choices=get_vocoder_choices(),
                value="hifi",
                type="value",
                interactive=True,
                scale=1,
            )
            pitch_guidance_input = gr.Checkbox(
                label=_("F0-guided model"), 
                value=True,
                info=_("The selected vocoder requires F0 guidance."),
                interactive=False,
                scale=1
            )
            version_input = gr.Dropdown(
                label=_("Version"),
                info=_("Select one that corresponds to your training."),
                choices=['v1', 'v2'],
                value='v2',
                interactive=True,
                scale=1
            )

        vocoder_input.change(
            fn=lambda vocoder: {
                "choices": get_vocoder_sample_rates(vocoder),
                "value": get_vocoder_sample_rates(vocoder)[0],
                "__type__": "update",
            },
            inputs=[vocoder_input],
            outputs=[sr_input],
        )

        extract_button = gr.Button(_("Extract Small Model"), variant="primary")

        output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Status messages and final file path will be displayed here."),
            value="",
            max_lines=8,
            interactive=False 
        )

        extract_button.click(
            fn=extract_small_model,
            inputs=[
                model_path_input,
                model_name_input,
                output_dir_input,
                sr_input,
                pitch_guidance_input,
                version_input,
                vocoder_input,
            ],
            outputs=[output_info],
        )

if __name__ == "__main__":
    with gr.Blocks() as demo:
        extract_small_model_tab()
