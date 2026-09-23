import os
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter

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

#: Folder URL holding ``f0G32k.pth`` and ``f0D32k.pth`` for each vocoder, keyed
#: as in ``rvc/configs/vocoders.json``.  Empty means none is published yet, and
#: the vocoder is skipped.
VOCODER_PRETRAINED_URLS = {
    "refinegan2": "",
    "hifi++": "",
}

vocoder_pretraineds = {
    "refinegan2": ("pretrained_refinegan2/", ["f0G32k.pth", "f0D32k.pth"]),
    "hifi++": ("pretrained_hifi++/", ["f0G32k.pth", "f0D32k.pth"]),
}


def vocoder_pretraineds_list(vocoder=None, sample_rate=None):
    """Download entries for the vocoder pretrains, optionally for one vocoder and rate."""
    entries = []
    for vocoder_id, (remote_folder, files) in vocoder_pretraineds.items():
        url = VOCODER_PRETRAINED_URLS.get(vocoder_id, "").rstrip("/")
        if not url or (vocoder is not None and vocoder_id != vocoder):
            continue
        if sample_rate is not None:
            files = [f for f in files if f.endswith(f"{str(sample_rate)[:2]}k.pth")]
        if files:
            entries.append((remote_folder, files, url))
    return entries

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
    ("embedders/spin_v2", ["pytorch_model.bin", "config.json"], f"{RESOURCE_BASE}/embedders/spin_v2"),
]

executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    # Must match ``pretrained_dir`` in vocoders.json: pretrained_selector reads there.
    "pretrained_refinegan2/": "rvc/models/pretraineds/refinegan2/",
    "pretrained_hifi++/": "rvc/models/pretraineds/hifi-gan++/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "embedders/spin_v2": "rvc/models/embedders/spin_v2/",
    "predictors/": "rvc/models/predictors/",
    "fireredvad/VAD/": "rvc/models/fireredvad/VAD/",
    "formant/": "rvc/models/formant/",
}


#: Files at least this big are fetched as ``SEGMENTS`` byte ranges at once:
#: the CDN caps each connection well below what a cloud machine can take.
SEGMENT_MIN = 32 * 1024 * 1024
SEGMENTS = 8
FILE_WORKERS = 8
CHUNK = 1024 * 1024

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_maxsize=FILE_WORKERS * SEGMENTS))


def _missing(file_list):
    """``(url, destination)`` of the files in ``file_list`` not on disk yet.
    An entry's optional third element is its own base URL."""
    for entry in file_list:
        remote_folder, files = entry[0], entry[1]
        base_url = entry[2] if len(entry) > 2 else url_base
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                if base_url == url_base:
                    yield f"{base_url}/{remote_folder}{file}", destination_path
                else:
                    yield f"{base_url}/{file}", destination_path


def _probe(url):
    """Size of the file at ``url`` and whether its server takes byte ranges.
    Follows redirects: Hugging Face answers with a 302 to its CDN, whose
    content-length is the file's."""
    response = _session.head(url, allow_redirects=True, timeout=30)
    size = int(response.headers.get("content-length", 0))
    ranges = response.ok and response.headers.get("accept-ranges") == "bytes"
    return size, ranges


def _plan(file_list):
    """``(url, destination, size, ranges)`` of each missing file, probed in
    parallel."""
    missing = list(_missing(file_list))
    if not missing:
        return []
    with ThreadPoolExecutor(min(len(missing), 16)) as executor:
        return [(*item, *probe) for item, probe in zip(missing, executor.map(_probe, [u for u, _ in missing]))]


def _fetch(url, path, global_bar, start=None, end=None):
    """Write ``url``, or its bytes ``start`` to ``end`` into the preallocated
    ``path``, and check the length."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    response = _session.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()
    name = os.path.basename(path).removesuffix(".part")
    if start is None:
        expected = int(response.headers.get("content-length", 0))
        mode = "wb"
    else:
        if response.status_code != 206:
            raise IOError(f"{name}: the server ignored the byte range.")
        expected = end - start + 1
        mode = "r+b"
    written = 0
    with open(path, mode) as file:
        if start is not None:
            file.seek(start)
        for data in response.iter_content(CHUNK):
            file.write(data)
            written += len(data)
            global_bar.update(len(data))
    if expected and written != expected:
        raise IOError(f"{name}: expected {expected} bytes, got {written}.")


def download_file(url, destination_path, global_bar, size=0, ranges=False):
    """Download ``url`` to ``destination_path``, in byte ranges when it is big
    and the server allows it.

    The status is checked before anything is written and the body only renamed
    into place once complete: a saved 404 body would otherwise pass as the file
    on every later run and fail only at ``torch.load``.
    """
    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temporary_path = f"{destination_path}.part"
    try:
        if ranges and size >= SEGMENT_MIN:
            with open(temporary_path, "wb") as file:
                file.truncate(size)
            step = -(-size // SEGMENTS)
            with ThreadPoolExecutor(SEGMENTS) as executor:
                futures = [
                    executor.submit(_fetch, url, temporary_path, global_bar, start, min(start + step, size) - 1)
                    for start in range(0, size, step)
                ]
                for future in futures:
                    future.result()
        else:
            _fetch(url, temporary_path, global_bar)
        os.replace(temporary_path, destination_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _download(file_list, description):
    """Download every missing file of ``file_list`` in one queue, biggest first,
    under one progress bar."""
    plan = sorted(_plan(file_list), key=lambda item: item[2], reverse=True)
    total_size = sum(item[2] for item in plan)
    if total_size <= 0:
        return
    with progress_task(total_size, description, download=True, leave=True) as (progress, task_id):
        global_bar = progress_handle(progress, task_id)
        with ThreadPoolExecutor(FILE_WORKERS) as executor:
            futures = [executor.submit(download_file, *item[:2], global_bar, *item[2:]) for item in plan]
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


def download_vocoder_pretraineds(vocoder, sample_rate):
    """Fetch one vocoder's missing G/D at ``sample_rate``."""
    _download(vocoder_pretraineds_list(vocoder, sample_rate), f"Downloading {vocoder} pretrained models")


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    entries = []
    if models:
        entries += models_list + embedders_list
    if exe:
        if os.name == "nt":
            entries += executables_list
        else:
            info("No executables needed.", tag="[DOWNLOAD]")
    if pretraineds_hifigan:
        entries += pretraineds_hifigan_list + vocoder_pretraineds_list()
    _download(entries, "Downloading all files")
