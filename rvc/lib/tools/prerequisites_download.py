import os
from concurrent.futures import ThreadPoolExecutor
import requests 

from rvc.lib.terminal import info, progress_handle, progress_task

RESOURCE_BASE = "https://huggingface.co/shiromiya/ShiroRVC-Resources/resolve/main"
FIREREDVAD_BASE = "https://huggingface.co/FireRedTeam/FireRedVAD/resolve/main/VAD"
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
        f"{RESOURCE_BASE}/RVC_v2_pretrains",
    )
]

models_list = [
    # Both live under ``predictors/`` in the resource repo, so both go through
    # the default path.  ``fcpe_ddsp.pt`` used to carry an override pointing at
    # ``f0_predictors/``, which does not exist and 404'd on every run.
    ("predictors/", ["rmvpe.pt", "fcpe_ddsp.pt"]),
    # FireRedVAD, for the "New Automatic" cutter.  Pulled straight from the
    # upstream repo rather than mirrored: it is Apache-2.0 and 2.3 MB, so there
    # is nothing to gain by copying it and a stale mirror to lose.
    ("fireredvad/VAD/", ["model.pth.tar", "cmvn.ark"], FIREREDVAD_BASE),
]

embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin_v1", ["pytorch_model.bin", "config.json"], f"{RESOURCE_BASE}/embedders/spin_v1"),
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
    "fireredvad/VAD/": "rvc/models/fireredvad/VAD/",
    "formant/": "rvc/models/formant/",
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    Supports optional third element (custom base URL) in the tuple.

    The HEAD has to follow redirects.  Hugging Face answers ``resolve/main``
    with a 302 to its CDN, and ``requests.head`` does not follow redirects by
    default, so this read the *redirect's* content-length -- about 950 bytes per
    file.  The progress bar was therefore sized at ~15 kB for a ~1.7 GB download
    and hit 100% on the first chunk.
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
                if base_url == url_base:
                    url = f"{base_url}/{remote_folder}{file}"
                else:
                    url = f"{base_url}/{file}"
                response = requests.head(url, allow_redirects=True, timeout=30)
                total_size += int(response.headers.get("content-length", 0))
    return total_size



def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.

    The status code is checked before anything is written, and the body lands
    in a temporary file that is only renamed into place once it is complete.
    Without both of those this wrote whatever came back: a 404 from Hugging Face
    carries the fifteen-byte body ``Entry not found``, which was duly saved as
    ``f0G40k.pth``.  Every later run then saw the file existing and skipped it,
    so six pretrained models stayed permanently broken and only failed much
    later, at ``torch.load``, with an error about serialization formats that had
    nothing to do with the real problem.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    expected = int(response.headers.get("content-length", 0))

    temporary_path = f"{destination_path}.part"
    written = 0
    try:
        with open(temporary_path, "wb") as file:
            for data in response.iter_content(1024):
                file.write(data)
                written += len(data)
                global_bar.update(len(data))
        if expected and written != expected:
            raise IOError(
                f"{os.path.basename(destination_path)}: expected {expected} bytes, got {written}."
            )
        os.replace(temporary_path, destination_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def download_mapping_files(file_mapping_list, global_bar):
    """Download entries in parallel. Supports an optional third (custom base URL) tuple element."""
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
):
    total_size = 0

    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)

    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)

    total_size += get_file_size_if_missing(pretraineds_hifigan)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
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
                    info("No executables needed.", tag="[DOWNLOAD]")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
    else:
        pass
