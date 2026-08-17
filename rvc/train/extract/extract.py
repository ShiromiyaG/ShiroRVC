import os
import sys
import glob
import time
import torch
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rvc.configs.vocoders import (
    get_vocoder_sample_rates,
    normalize_vocoder,
)
import numpy as np
import concurrent.futures
import multiprocessing as mp
import json

from rvc.lib.terminal import get_console, install_rich_print, progress_task

install_rich_print()

from rvc.lib.utils import load_audio_16k, load_embedder_model, extract_features
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.f0 import CREPE, RMVPE, FCPE
from rvc.configs.config import Config

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu"):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device

        if f0_method in ("crepe", "crepe-tiny"):
            self.model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "rmvpe":
            self.model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "fcpe":
            self.model = FCPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        self.f0_method = f0_method

    def compute_f0(self, x, p_len=None):
        if self.f0_method == "crepe":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
        elif self.f0_method == "crepe-tiny":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
        elif self.f0_method == "rmvpe":
            f0 = self.model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            f0 = self.model.get_f0(x, p_len, filter_radius=0.006, test_time_augmentation=True)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(np.uint8, copy=False)

    def process_file(self, file_info):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            feature_pit = np.asarray(self.compute_f0(np_arr), dtype=np.float32)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )


def process_files(files, f0_method, device, threads):
    if device == "cpu":
        torch.set_num_threads(max(1, threads))
    fe = FeatureInput(f0_method=f0_method, device=device)

    with progress_task(len(files), f"F0 {device}", leave=True) as (progress, task_id):
        for file_info in files:
            fe.process_file(file_info)
            progress.advance(task_id)


def run_pitch_extraction(files, devices, f0_method, threads):
    devices_str = ", ".join(devices)
    get_console().print(
        f"Starting pitch extraction with {num_processes} threads on {devices_str} using {f0_method}..."
    )
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i :: len(devices)],
                f0_method,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    get_console().print(
        f"Pitch extraction completed in {time.time() - start_time:.2f} seconds."
    )


def process_file_embedding(
    files, embedder_model, embedder_model_custom, device_num, device, n_threads
):
    model, do_normalize = load_embedder_model(embedder_model, embedder_model_custom)
    model = model.to(device).float()
    model.eval()
    if device == "cpu":
        torch.set_num_threads(max(1, n_threads))
    use_amp = str(device).startswith("cuda")

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).to(device).float()
        feats = feats.view(1, -1)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            result = extract_features(model, feats, "v2", do_normalize=do_normalize)
        feats_out = result.squeeze(0).float().cpu().numpy()
        if np.isfinite(feats_out).all():
            np.save(
                out_file_path,
                feats_out.astype(np.float16, copy=False),
                allow_pickle=False,
            )
        else:
            print(f"{wav_file_path} produced NaN values; skipping.")

    with progress_task(
        len(files),
        f"Features {device}",
        leave=True,
    ) as (progress, task_id):
        with torch.inference_mode():
            for file_info in files:
                worker(file_info)
                progress.advance(task_id)


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads
):
    devices_str = ", ".join(devices)
    get_console().print(
        f"Starting embedding extraction with {num_processes} threads on {devices_str}..."
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                embedder_model_custom,
                i,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    get_console().print(
        f"Embedding extraction completed in {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    num_processes = int(sys.argv[3])
    gpus = sys.argv[4]
    sample_rate = sys.argv[5]
    vocoder_arch = normalize_vocoder(sys.argv[6])
    if int(sample_rate) not in get_vocoder_sample_rates(vocoder_arch):
        raise ValueError(
            f"{vocoder_arch} does not provide a configuration for {sample_rate} Hz."
        )
    embedder_model = sys.argv[7]
    embedder_model_custom = sys.argv[8] if len(sys.argv) > 8 else None
    include_mutes = int(sys.argv[9]) if len(sys.argv) > 9 else 2

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = chosen_embedder_model
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    files = []
    audio_files = [
        path
        for path in glob.glob(os.path.join(wav_path, "*"))
        if os.path.isfile(path)
    ]
    audio_files.sort(key=os.path.getsize, reverse=True)
    for file in audio_files:
        file_name = os.path.basename(file)
        feature_name = os.path.splitext(file_name)[0] + ".npy"
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", feature_name),
        ]
        files.append(file_info)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(files, devices, f0_method, num_processes)

    run_embedding_extraction(
        files, devices, embedder_model, embedder_model_custom, num_processes
    )

    generate_config(sample_rate, exp_dir, vocoder_arch)
    generate_filelist(exp_dir, sample_rate, include_mutes, embedder_model, vocoder_arch)
