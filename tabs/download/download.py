import os
import sys
import json
import shutil
import requests
import tempfile
import gradio as gr

from concurrent.futures import ThreadPoolExecutor

from rvc.lib.i18n import _


now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script
from rvc.lib.text import format_title
from rvc.lib.terminal import progress_handle, progress_task, success

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message=_(
                "The file you dropped is not a valid model file. Please try again."
            )
        )

    file_name = format_title(os.path.basename(dropbox))
    model_name = file_name

    if ".pth" in model_name:
        model_name = model_name.split(".pth")[0]
    elif ".index" in model_name:
        replacements = ["nprobe_1_", "_v1", "_v2", "added_"]
        for rep in replacements:
            model_name = model_name.replace(rep, "")
        model_name = model_name.split(".index")[0]

    model_path = os.path.join(now_dir, "logs", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(model_path, file_name)):
        os.remove(os.path.join(model_path, file_name))
    shutil.move(dropbox, os.path.join(model_path, file_name))
    success(f"Saved '{file_name}' in {model_path}.", tag="[DOWNLOAD]")
    gr.Info(_("{name} saved in {folder}").format(name=file_name, folder=model_path))

    return None


json_url = "https://huggingface.co/shiromiya/ShiroRVC-Resources/raw/main/pretrains.json"


def fetch_pretrained_data():
    pretraineds_custom_path = os.path.join(
        "rvc", "models", "pretraineds", "custom"
    )
    os.makedirs(pretraineds_custom_path, exist_ok=True)
    try:
        with open(
            os.path.join(pretraineds_custom_path, json_url.split("/")[-1]), "r"
        ) as f:
            data = json.load(f)
    except:
        try:
            response = requests.get(json_url)
            response.raise_for_status()
            data = response.json()
            with open(
                os.path.join(pretraineds_custom_path, json_url.split("/")[-1]),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    separators=(",", ": "),
                    ensure_ascii=False,
                )
        except:
            data = {
                "Titan": {
                    "32k": {"D": "null", "G": "null"},
                },
            }
    return data


def get_pretrained_list():
    data = fetch_pretrained_data()
    return list(data.keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    if model not in data:
        model = next(iter(data), None)
    return list(data[model].keys()) if model else []


def get_file_size(url):
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


def download_file(url, destination_path, progress_bar):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))


def download_pretrained_model(model, sample_rate):
    data = fetch_pretrained_data()
    if model not in data or sample_rate not in data[model]:
        raise gr.Error(
            message=_("{name} is not available at {rate}.").format(
                name=model, rate=sample_rate
            )
        )
    paths = data[model][sample_rate]
    pretraineds_custom_path = os.path.join(
        "rvc", "models", "pretraineds", "custom"
    )
    os.makedirs(pretraineds_custom_path, exist_ok=True)

    d_url = f"https://huggingface.co/{paths['D']}"
    g_url = f"https://huggingface.co/{paths['G']}"

    total_size = get_file_size(d_url) + get_file_size(g_url)

    gr.Info(_("Downloading pretrained model..."))

    with progress_task(
        total_size,
        "Downloading files",
        download=True,
        leave=True,
    ) as (progress, task_id):
        progress_bar = progress_handle(progress, task_id)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    download_file,
                    d_url,
                    os.path.join(pretraineds_custom_path, os.path.basename(paths["D"])),
                    progress_bar,
                ),
                executor.submit(
                    download_file,
                    g_url,
                    os.path.join(pretraineds_custom_path, os.path.basename(paths["G"])),
                    progress_bar,
                ),
            ]
            for future in futures:
                future.result()

    gr.Info(_("Pretrained model downloaded successfully!"))
    success("Pretrained model downloaded.", tag="[DOWNLOAD]")


def update_sample_rate_dropdown(model):
    sample_rates = get_pretrained_sample_rates(model)
    return {
        "choices": sample_rates,
        "value": sample_rates[0] if sample_rates else None,
        "__type__": "update",
    }


def download_tab():
    with gr.Column():
        gr.Markdown(value=_("## Download Model"))
        model_link = gr.Textbox(
            label=_("Model Link"),
            placeholder=_("Introduce the model link"),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=_("Output Information"),
            info=_("Download status."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(_("Download Model"))
        model_download_button.click(
            fn=run_download_script,
            inputs=[model_link],
            outputs=[model_download_output_info],
        )
        gr.Markdown(value=_("## Quick drag-and-drop"))
        dropbox = gr.File(
            label=_("Drop a .pth and .index file to copy them into the model folder."),
            type="filepath",
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
        gr.Markdown(value=_("## Download Pretrained Models"))
        pretrained_list = get_pretrained_list()
        default_pretrained = pretrained_list[0] if pretrained_list else None
        default_sample_rates = get_pretrained_sample_rates(default_pretrained)
        pretrained_model = gr.Dropdown(
            label=_("Pretrained"),
            info=_("Select the pretrained model you want to download."),
            choices=pretrained_list,
            value=default_pretrained,
            interactive=True,
        )
        pretrained_sample_rate = gr.Dropdown(
            label=_("Sampling Rate"),
            info=_("And select the sampling rate."),
            choices=default_sample_rates,
            value=default_sample_rates[0] if default_sample_rates else None,
            interactive=True,
            allow_custom_value=True,
        )
        pretrained_model.change(
            update_sample_rate_dropdown,
            inputs=[pretrained_model],
            outputs=[pretrained_sample_rate],
        )
        download_pretrained = gr.Button(_("Download"))
        download_pretrained.click(
            fn=download_pretrained_model,
            inputs=[pretrained_model, pretrained_sample_rate],
            outputs=[],
        )
