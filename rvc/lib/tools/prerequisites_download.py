import os
from concurrent.futures import ThreadPoolExecutor
import requests 

from rvc.lib.terminal import progress_handle, progress_task

RESOURCE_BASE = "https://huggingface.co/shiromiya/Shiromiya-RVC-Fork-Resources/resolve/main"
SMARTCUTTER_BASE = "https://huggingface.co/shiromiya/SmartCutter/resolve/main/SmartCutter-v3"
url_base = RESOURCE_BASE

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
        f"{RESOURCE_BASE}/pretrained_models/rvc_nsf_hifigan/v2",
    )
]

smartcutter_list = [
    (
        "smartcutter/",
        [
            "v3_model_32000.pth",
            "v3_model_40000.pth",
            "v3_model_48000.pth",
        ],
        SMARTCUTTER_BASE,
    )
]

models_list = [
    ("predictors/", ["rmvpe.pt"]),
    ("predictors/", ["fcpe_ddsp.pt"], f"{RESOURCE_BASE}/f0_predictors")
]

embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin_v1", ["pytorch_model.bin", "config.json"], f"{RESOURCE_BASE}/embedders/spin"),
    ("embedders/spin_v2", ["pytorch_model.bin", "config.json"], f"{RESOURCE_BASE}/embedders/spin_v2"),
]

executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "embedders/spin_v1": "rvc/models/embedders/spin_v1/",
    "embedders/spin_v2": "rvc/models/embedders/spin_v2/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
    "smartcutter/": "rvc/models/smartcutter/"
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    Supports optional third element (custom base URL) in the tuple.
    """
    total_size = 0
    for entry in file_list:
        if len(entry) == 2:
            remote_folder, files = entry
            base_url = url_base
        else:
            remote_folder, files, base_url = entry

        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                # Construct URL depending on whether it's using the shared base or custom one
                if base_url == url_base:
                    url = f"{base_url}/{remote_folder}{file}"
                else:
                    url = f"{base_url}/{file}"
                response = requests.head(url)
                total_size += int(response.headers.get("content-length", 0))
    return total_size



def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            global_bar.update(len(data))


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    Supports optional third element (custom base URL) in the tuple.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for entry in file_mapping_list:
            if len(entry) == 2:
                remote_folder, file_list = entry
                base_url = url_base
            else:
                remote_folder, file_list, base_url = entry

            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not os.path.exists(destination_path):
                    if base_url == url_base:
                        url = f"{base_url}/{remote_folder}{file}"
                    else:
                        url = f"{base_url}/{file}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, global_bar
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for entry in pretrained_list:
        if len(entry) == 3:
            folder, files, url = entry
        else:
            folder, files = entry
            url = None

        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]

        if f0_files:
            f0_list.append((folder, f0_files, url) if url else (folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files, url) if url else (folder, non_f0_files))

    return f0_list, non_f0_list

pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
    smartcutter,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0

    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)

    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)

    if smartcutter:
        total_size += get_file_size_if_missing(smartcutter_list)

    total_size += get_file_size_if_missing(pretraineds_hifigan)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
    smartcutter,
):
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
        smartcutter,
    )

    if total_size > 0:
        with progress_task(
            total_size,
            "Downloading all files",
            download=True,
            leave=True,
        ) as (progress, task_id):
            global_bar = progress_handle(progress, task_id)
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if smartcutter:
                download_mapping_files(smartcutter_list, global_bar)
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
    else:
        pass
