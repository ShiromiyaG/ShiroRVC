import os
import sys
import glob
import shutil
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
import soundfile as sf
import concurrent.futures
import multiprocessing as mp
import json

from rvc.lib.terminal import (
    error as print_error,
    info,
    install_rich_print,
    progress_task,
    success,
    warning,
)

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

    def process_batch(self, batched_model, group):
        """``process_file`` for several equal-length clips in one pass.

        Falls back to the single path for the whole group on any failure: a
        batch is all-or-nothing on the GPU, and one unreadable file must not
        cost the other fifteen.
        """
        try:
            audio = [load_audio_16k(info[0]) for info in group]
            contours = batched_model.infer_from_audio_batch(audio, thred=0.03)
        except Exception as error:
            print_error(
                f"Batch of {len(group)} failed on {self.device} ({error}); "
                f"retrying one at a time.",
                tag="[EXTRACT]",
            )
            for info in group:
                self.process_file(info)
            return

        for info, contour in zip(group, contours):
            _, opt_path_coarse, opt_path_full, _ = info
            feature_pit = np.asarray(contour, dtype=np.float32)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            np.save(opt_path_coarse, self.coarse_f0(feature_pit), allow_pickle=False)

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
            print_error(
                f"Could not extract {inp_path} on {self.device}: {error}",
                tag="[EXTRACT]",
            )


#: Clips per GPU pass.  A 3-second clip is far too small to fill a GPU on its
#: own, so both stages ran mostly between kernels rather than in them; 16 gets
#: ~8x the throughput and the little that 32 adds is not worth the extra
#: activation memory on an 8 GB card.
BATCH_SIZE = 16


def _grouped_by_length(files):
    """``files`` bucketed by exact sample count, longest bucket first.

    Exact, not approximate: a batch has to be one tensor, and padding clips to
    a common length would change the model's own padding and every frame that
    follows -- measured at up to 95% relative error on the embeddings.  Equal
    lengths make a batch the same arithmetic as the loop it replaces.

    This costs nothing to know.  Preprocessing cuts on a fixed grid, so the
    great majority of a dataset lands on one length (78% of a real experiment
    here, all 3.00 s); the ragged tails fall into small buckets and, at worst,
    into buckets of one, which is exactly the old path.
    """
    buckets = {}
    for file_info in files:
        try:
            frames = sf.info(file_info[0]).frames
        except Exception:
            frames = -1  # unreadable: give it its own bucket, fail it alone
        buckets.setdefault(frames, []).append(file_info)
    return sorted(buckets.values(), key=len, reverse=True)


def process_files(files, f0_method, device, threads):
    if device == "cpu":
        torch.set_num_threads(max(1, threads))
    fe = FeatureInput(f0_method=f0_method, device=device)
    # Only RMVPE has a batched path; the others keep the one-at-a-time loop.
    batched = getattr(fe.model, "model", None)
    batched = batched if hasattr(batched, "infer_from_audio_batch") else None

    with progress_task(len(files), f"F0 {device}", leave=True) as (progress, task_id):
        for bucket in _grouped_by_length(files):
            for start in range(0, len(bucket), BATCH_SIZE):
                group = [
                    info
                    for info in bucket[start : start + BATCH_SIZE]
                    if not (os.path.exists(info[1]) and os.path.exists(info[2]))
                ]
                done = len(bucket[start : start + BATCH_SIZE])
                if not group:
                    progress.advance(task_id, done)
                    continue
                if batched is None or len(group) == 1:
                    for info in group:
                        fe.process_file(info)
                else:
                    fe.process_batch(batched, group)
                progress.advance(task_id, done)


def run_pitch_extraction(files, devices, f0_method, threads):
    devices_str = ", ".join(devices)
    info(
        f"Pitch extraction: {f0_method}, {num_processes} threads on {devices_str}.",
        tag="[EXTRACT]",
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

    success(
        f"Pitch extraction finished in {time.time() - start_time:.2f}s.",
        tag="[EXTRACT]",
    )


#: fp16 puts a quantisation floor under the retrieval index vectors built from
#: these features while inference queries with float32 ones; fp32 doubles the
#: feature cache (~550MB -> ~1.1GB for 2h) but is the better default.
FEATURE_PRECISIONS = {"fp32": np.float32, "fp16": np.float16}


def process_file_embedding(
    files, embedder_model, embedder_model_custom, device_num, device, n_threads,
    feature_precision="fp32",
):
    dtype = FEATURE_PRECISIONS.get(feature_precision, np.float32)
    model, do_normalize = load_embedder_model(embedder_model, embedder_model_custom)
    model = model.to(device).float()
    model.eval()
    if device == "cpu":
        torch.set_num_threads(max(1, n_threads))
    use_amp = str(device).startswith("cuda")

    def save(file_info, feats_out):
        wav_file_path, _, _, out_file_path = file_info
        if np.isfinite(feats_out).all():
            np.save(
                out_file_path,
                feats_out.astype(dtype, copy=False),
                allow_pickle=False,
            )
        else:
            warning(f"{wav_file_path} produced NaN values; skipping.", tag="[EXTRACT]")

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
        save(file_info, result.squeeze(0).float().cpu().numpy())

    def batch_worker(group):
        """One forward pass for a group of equal-length clips.

        Same all-or-nothing fallback as the pitch stage: on any failure the
        group is redone one clip at a time, so a single bad file costs its
        neighbours a retry rather than their features.
        """
        try:
            audio = np.stack([load_audio_16k(info[0]) for info in group])
            feats = torch.from_numpy(audio).to(device).float()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                result = extract_features(model, feats, "v2", do_normalize=do_normalize)
            out = result.float().cpu().numpy()
        except Exception as error:
            print_error(
                f"Batch of {len(group)} failed on {device} ({error}); "
                f"retrying one at a time.",
                tag="[EXTRACT]",
            )
            for info in group:
                worker(info)
            return
        for info, feats_out in zip(group, out):
            save(info, feats_out)

    with progress_task(
        len(files),
        f"Features {device}",
        leave=True,
    ) as (progress, task_id):
        with torch.inference_mode():
            for bucket in _grouped_by_length(files):
                for start in range(0, len(bucket), BATCH_SIZE):
                    chunk = bucket[start : start + BATCH_SIZE]
                    group = [i for i in chunk if not os.path.exists(i[3])]
                    if not group:
                        progress.advance(task_id, len(chunk))
                        continue
                    if len(group) == 1:
                        worker(group[0])
                    else:
                        batch_worker(group)
                    progress.advance(task_id, len(chunk))


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads,
    feature_precision="fp32",
):
    devices_str = ", ".join(devices)
    info(
        f"Embedding extraction: {num_processes} threads on {devices_str}.",
        tag="[EXTRACT]",
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
                feature_precision,
            )
            for i in range(len(devices))
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    success(
        f"Embedding extraction finished in {time.time() - start_time:.2f}s.",
        tag="[EXTRACT]",
    )


def discard_16k_slices(wav_path):
    """Delete the 16 kHz slices once the features derived from them exist.

    They feed pitch and embedder extraction only, so are dead weight after
    that; re-extracting with a different f0 method or embedder needs them
    back, meaning preprocessing must run again.
    """
    if os.path.basename(os.path.normpath(wav_path)) != "sliced_audios_16k":
        return
    if not os.path.isdir(wav_path):
        return

    # The mute folders are a shared asset: `preparing_files` falls back to
    # `logs/mute/sliced_audios_16k/mute.wav` when building the mute sample for
    # a new sample rate, so never strip one of those.
    experiment_name = os.path.basename(os.path.dirname(os.path.normpath(wav_path)))
    if experiment_name.lower().startswith("mute"):
        info(
            f"Keeping 16 kHz slices: '{experiment_name}' is a shared mute asset.",
            tag="[EXTRACT]",
        )
        return

    # Only discard once the artifacts that replace them are actually on disk.
    exp_dir = os.path.dirname(os.path.normpath(wav_path))
    if not os.path.isfile(os.path.join(exp_dir, "filelist.txt")):
        warning(
            "Keeping 16 kHz slices: filelist.txt is missing, so extraction "
            "looks incomplete.",
            tag="[EXTRACT]",
        )
        return

    freed = 0
    count = 0
    for root, _, names in os.walk(wav_path):
        for name in names:
            try:
                freed += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass

    try:
        shutil.rmtree(wav_path)
    except OSError as error:
        warning(f"Could not remove the 16 kHz slices: {error}", tag="[EXTRACT]")
        return

    success(
        f"Removed {count} 16 kHz slices, freeing {freed / (1024 ** 3):.2f} GB.",
        tag="[EXTRACT]",
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
    remove_16k_slices = sys.argv[10].lower() == "true" if len(sys.argv) > 10 else False
    feature_precision = sys.argv[11] if len(sys.argv) > 11 else "fp32"
    if feature_precision not in FEATURE_PRECISIONS:
        warning(
            f"Unknown feature precision {feature_precision!r}; using fp32.",
            tag="[EXTRACT]",
        )
        feature_precision = "fp32"

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
        files,
        devices,
        embedder_model,
        embedder_model_custom,
        num_processes,
        feature_precision,
    )

    generate_config(sample_rate, exp_dir, vocoder_arch)
    generate_filelist(exp_dir, sample_rate, include_mutes, embedder_model, vocoder_arch)

    if remove_16k_slices:
        discard_16k_slices(wav_path)
