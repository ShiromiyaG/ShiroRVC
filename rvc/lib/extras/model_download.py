import os
import re
import shutil
import sys
import tempfile
import zipfile
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.text import format_title
from rvc.lib.terminal import (
    error as print_error,
    progress_handle,
    progress_task,
    success,
    warning,
)
from rvc.lib.extras import gdown


file_path = os.path.join(now_dir, "logs")
zips_path = os.path.join(file_path, "zips")
os.makedirs(zips_path, exist_ok=True)

#: What a link may hand over besides a .zip: the files a model consists of.
MODEL_SUFFIXES = (".pth", ".index", ".srvc")
#: Seconds to connect, then to wait for each chunk.  A stalled server fails the
#: job instead of holding the GUI's single worker thread forever.
TIMEOUT = (15, 60)


class DownloadError(RuntimeError):
    """A download that left no model behind; the message says why."""


def _safe_filename(name: str, fallback: str = "downloaded_file") -> str:
    """The last component of a server-supplied name, whichever separator it
    used -- ``os.path.sep`` alone left ``/`` through on Windows."""
    name = os.path.basename(name.replace("\\", "/")).strip()
    return name if name not in ("", ".", "..") else fallback


def _filename_from_response(response, url: str) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(
        r"filename\*=(?:[\w-]+'[\w-]*')?([^;]+)", disposition, re.IGNORECASE
    ) or re.search(r'filename="?([^";]+)"?', disposition, re.IGNORECASE)
    name = match.group(1).strip() if match else urlparse(url).path
    return _safe_filename(unquote(name))


def _model_name(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return format_title(stem).strip(".") or "downloaded_model"


def download_from_url(url: str, destination: str) -> None:
    """Fetch ``url`` into the folder ``destination``, by its host's route."""
    if "drive.google.com" in url:
        file_id = extract_google_drive_id(url)
        if not file_id:
            raise DownloadError("No file id in the Google Drive link.")
        gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=destination + os.sep,
            quiet=False,
            fuzzy=True,
        )
    elif "/blob/" in url or "/resolve/" in url:
        download_file(url.replace("/blob/", "/resolve/"), destination)
    elif "/tree/main" in url:
        download_from_huggingface(url, destination)
    else:
        download_file(url, destination)


def extract_google_drive_id(url):
    if "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    return None


def save_response_content(response, url: str, destination: str) -> None:
    file_name = _filename_from_response(response, url)
    total_size = int(response.headers.get("Content-Length", 0))

    with open(os.path.join(destination, file_name), "wb") as file:
        with progress_task(
            total_size,
            file_name,
            download=True,
            leave=True,
        ) as (progress, task_id):
            progress_bar = progress_handle(progress, task_id)
            for data in response.iter_content(1024 * 1024):
                file.write(data)
                progress_bar.update(len(data))


def download_from_huggingface(url: str, destination: str) -> None:
    response = requests.get(url, timeout=TIMEOUT)
    soup = BeautifulSoup(response.content, "html.parser")
    temp_url = next(
        (
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].endswith(".zip")
        ),
        None,
    )
    if not temp_url:
        raise DownloadError("No .zip file found on that Hugging Face page.")
    url = temp_url.replace("blob", "resolve")
    if "huggingface.co" not in url:
        url = "https://huggingface.co" + url
    download_file(url, destination)


def download_file(url: str, destination: str) -> None:
    response = requests.get(url, stream=True, timeout=TIMEOUT)
    if response.status_code != 200:
        raise DownloadError(f"The server answered with status {response.status_code}.")
    save_response_content(response, url, destination)


def clean_extracted_files(extract_folder_path, model_name):
    macosx_path = os.path.join(extract_folder_path, "__MACOSX")
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)

    subfolders = [
        f
        for f in os.listdir(extract_folder_path)
        if os.path.isdir(os.path.join(extract_folder_path, f))
    ]
    if len(subfolders) == 1:
        subfolder_path = os.path.join(extract_folder_path, subfolders[0])
        for item in os.listdir(subfolder_path):
            shutil.move(
                os.path.join(subfolder_path, item),
                os.path.join(extract_folder_path, item),
            )
        os.rmdir(subfolder_path)

    for item in os.listdir(extract_folder_path):
        source_path = os.path.join(extract_folder_path, item)
        if ".pth" in item:
            new_file_name = model_name + ".pth"
        elif ".index" in item:
            new_file_name = model_name + ".index"
        else:
            continue

        destination_path = os.path.join(extract_folder_path, new_file_name)
        if not os.path.exists(destination_path):
            os.rename(source_path, destination_path)


def _install_archive(zip_path: str) -> str:
    """Unpack a downloaded .zip into ``logs/<its name>/``."""
    model_name = _model_name(os.path.basename(zip_path))
    folder = os.path.join(file_path, model_name)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(folder)
    except zipfile.BadZipFile as error:
        raise DownloadError(f"{os.path.basename(zip_path)} is not a valid zip: {error}") from error
    clean_extracted_files(folder, model_name)
    return folder


def _install_file(path: str) -> str:
    """Move a bare model file into ``logs/<its name>/``, as an archive's would go."""
    suffix = os.path.splitext(path)[1].lower()
    model_name = _model_name(os.path.basename(path))
    folder = os.path.join(file_path, model_name)
    os.makedirs(folder, exist_ok=True)
    os.replace(path, os.path.join(folder, model_name + suffix))
    return folder


def _fetch_and_install(url: str, workdir: str) -> str:
    download_from_url(url, workdir)
    folders = []
    for name in sorted(os.listdir(workdir)):
        path = os.path.join(workdir, name)
        if name.lower().endswith(".zip"):
            folders.append(_install_archive(path))
        elif name.lower().endswith(MODEL_SUFFIXES):
            folders.append(_install_file(path))
        else:
            warning(f"Skipped {name}: not a .zip, .pth, .index or .srvc.", tag="[DOWNLOAD]")
    if not folders:
        raise DownloadError("The link gave no .zip, .pth, .index or .srvc file.")
    return folders[-1]


def model_download_pipeline(url: str) -> str:
    """Download a model into ``logs/`` and return its folder.

    Takes a .zip, unpacked into ``logs/<name>/``, or a bare .pth, .index or
    .srvc.  Each download lands in its own scratch folder under ``logs/zips``,
    so neither a concurrent download nor the leftovers of a failed one is
    mistaken for this one's file.  Raises :class:`DownloadError` after
    printing why.
    """
    workdir = tempfile.mkdtemp(dir=zips_path)
    try:
        folder = _fetch_and_install(url, workdir)
    except Exception as error:
        message = str(error) or type(error).__name__
        print_error(f"Download failed: {message}", tag="[DOWNLOAD]")
        raise DownloadError(message) from error
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    success(f"Downloaded '{os.path.basename(folder)}'.", tag="[DOWNLOAD]")
    return folder
