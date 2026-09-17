"""Preprocess and extract a dataset on a TPU host (torch_xla), RMVPE + embedder.

Preprocessing is CPU work (slicing, VAD, loudness), so it runs the fork's own
``preprocess.py`` across every host core with one BLAS/OpenMP thread per worker.

Extraction writes exactly what ``rvc/train/extract/extract.py`` writes.  Clips
are grouped by exact 16 kHz length, as there -- padding changes the features --
and each group large enough to be worth a compile runs on the TPU in fixed-size
batches, the last one filled with repeats.  The rare lengths go to a CPU pool
running the stock extractor, so they cost no compiles.

    python tools/TPU/prepare_dataset_tpu.py --model-name my-pretrain \
        --dataset /kaggle/input/my-audio --sample-rate 32000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

F0_BIN, F0_MIN, F0_MAX = 256, 50.0, 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)
RMVPE_THRESHOLD = 0.03


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", default="", help="Audio folder (speaker subfolders 0_name, 1_name... for multi-speaker).")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--vocoder", default="refinegan2", help="hifi, hifi++ or refinegan2 (disabled ones included); picks the config.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse logs/<model>/sliced_audios.")
    parser.add_argument("--skip-extract", action="store_true")

    group = parser.add_argument_group("preprocess (same meaning as core.py preprocess)")
    group.add_argument("--cut", default="New Automatic", choices=("Skip", "Simple", "Automatic", "New Automatic"))
    group.add_argument("--no-effects", action="store_true", help="Disable the high-pass filter.")
    group.add_argument("--noise-reduction", action="store_true")
    group.add_argument("--noise-reduction-strength", type=float, default=0.7)
    group.add_argument("--chunk-len", type=float, default=3.0)
    group.add_argument("--overlap-len", type=float, default=0.36)
    group.add_argument("--normalization", default="pre_peak_rvc", choices=("none", "post_peak", "pre_peak_rvc", "pre_loudness"))
    group.add_argument("--loading-resampling", default="librosa", choices=("librosa", "ffmpeg"))
    group.add_argument("--rms-norm-db", type=float, default=-16.0)

    group = parser.add_argument_group("extract")
    group.add_argument("--embedder", default="contentvec", choices=("contentvec", "spin_v2"))
    group.add_argument("--include-mutes", type=int, default=5)
    group.add_argument("--feature-precision", default="fp32", choices=("fp32", "fp16"))
    group.add_argument("--batch-size", type=int, default=32, help="Clips per TPU chip per step.")
    group.add_argument("--min-group", type=int, default=64, help="Smallest same-length group sent to the TPU; smaller ones run on the CPU.")
    group.add_argument("--embedder-bf16", action="store_true", help="Run the embedder under bf16 autocast.")
    group.add_argument("--single-process", action="store_true", help="One TPU chip only, for debugging.")
    return parser.parse_args(argv)


def run_preprocess(args, exp_dir):
    cpus = os.cpu_count() or 4
    # Each pool worker would otherwise start one BLAS/OpenMP thread per core.
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        SHIRO_VAD_DEVICE="cpu",
    )
    command = [
        sys.executable,
        os.path.join("rvc", "train", "preprocess", "preprocess.py"),
        exp_dir,
        args.dataset,
        str(args.sample_rate),
        str(cpus),
        args.cut,
        str(not args.no_effects),
        str(args.noise_reduction),
        str(args.noise_reduction_strength),
        str(args.chunk_len),
        str(args.overlap_len),
        args.normalization,
        args.loading_resampling,
        "WAV",
        str(args.rms_norm_db),
    ]
    print(f"[PREP] preprocess on {cpus} CPU workers", flush=True)
    subprocess.run(command, env=env, check=True)


def ensure_models(embedder):
    wanted = [
        os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
        os.path.join("rvc", "models", "embedders", embedder, "pytorch_model.bin"),
        os.path.join("rvc", "models", "fireredvad", "VAD", "model.pth.tar"),
    ]
    if all(os.path.isfile(path) for path in wanted):
        return
    from rvc.lib.extras.prerequisites_download import prequisites_download_pipeline

    prequisites_download_pipeline(False, True, False)


def coarse_f0(f0):
    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    f0_mel = np.clip((f0_mel - F0_MEL_MIN) * (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN) + 1, 1, F0_BIN - 1)
    return np.rint(f0_mel).astype(np.uint8, copy=False)


def split_groups(groups, min_group):
    """``(groups for the TPU, flat list of items for the CPU)``."""
    tpu = [group for group in groups if len(group) >= min_group]
    cpu = [item for group in groups if len(group) < min_group for item in group]
    return tpu, cpu


def _keep_rows(rows):
    return rows


class _Clips:
    """Map-style dataset over one same-length group: (index, 16 kHz audio)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # ``extract._clip_audio`` without importing the extractor into every worker.
        from rvc.lib.audio_io import load_audio_16k
        from rvc.train.extract.noise_mutes import is_noise_source, noise_audio

        source = self.items[index][0]
        audio = noise_audio(source) if is_noise_source(source) else load_audio_16k(source)
        return index, np.asarray(audio, dtype=np.float32)


def _batches(items, batch_size, workers):
    """Full-size batches of one group; the last is filled with repeats of its own items."""
    import torch

    loader = torch.utils.data.DataLoader(
        _Clips(items),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=_keep_rows,
        prefetch_factor=4 if workers else None,
    )
    for rows in loader:
        indices = [row[0] for row in rows]
        audio = [row[1] for row in rows]
        real = len(rows)
        while len(audio) < batch_size:
            audio.append(audio[len(audio) % real])
        yield indices, torch.from_numpy(np.stack(audio)), real


def _mp_fn(index, plan):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    import xla_patches

    xla_patches.apply_extract()

    from rvc.lib.predictors.rmvpe import RMVPE0Predictor
    from rvc.lib.utils import extract_features, load_embedder_model
    from rvc.train.extract.extract import FEATURE_PRECISIONS

    device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    sync = getattr(torch_xla, "sync", xm.mark_step)
    rank, world = xr.global_ordinal(), xr.world_size()
    batch_size = plan["batch_size"]
    workers = max(1, (os.cpu_count() or 8) // (2 * world))

    def shard(groups):
        return [group[rank::world] for group in groups if group[rank::world]]

    pitch_groups = shard(plan["pitch_groups"])
    if pitch_groups:
        predictor = RMVPE0Predictor(os.path.join("rvc", "models", "predictors", "rmvpe.pt"), device=device)
        # Decoded on the CPU: the boolean-mask write in ``decode`` is a dynamic shape on XLA.
        decoder = object.__new__(RMVPE0Predictor)
        decoder.cents_mapping_tensor = torch.from_numpy(predictor.cents_mapping).float()
        corrector = None
        if plan["high_register"]["enabled"]:
            from rvc.lib.predictors.f0 import RMVPE

            corrector = RMVPE(device="cpu", high_register=plan["high_register"])
        done = 0
        with torch.no_grad():
            for group in pitch_groups:
                for indices, audio, real in _batches(group, batch_size, workers):
                    mel = predictor.mel_extractor(audio.to(device), center=True)
                    hidden = predictor.mel2hidden(mel)
                    sync()
                    # Trimmed on the host: slicing on the device would compile the short last batch again.
                    hidden = hidden.cpu()[:real]
                    count, frames, classes = hidden.shape
                    f0 = decoder.decode(hidden.reshape(count * frames, classes), thred=RMVPE_THRESHOLD)
                    f0 = f0.reshape(count, frames).numpy()
                    for row, item_index in enumerate(indices):
                        source, coarse_path, full_path, _ = group[item_index]
                        contour = np.asarray(f0[row], dtype=np.float32)
                        if corrector is not None:
                            from rvc.lib.audio_io import load_audio_16k

                            contour = np.asarray(
                                corrector._fix_high_register(load_audio_16k(source), contour, RMVPE_THRESHOLD),
                                dtype=np.float32,
                            )
                        np.save(full_path, contour, allow_pickle=False)
                        np.save(coarse_path, coarse_f0(contour), allow_pickle=False)
                    done += real
                if rank == 0:
                    print(f"[PREP] pitch chip0: {done} clips", flush=True)
        del predictor

    embed_groups = shard(plan["embed_groups"])
    if embed_groups:
        dtype = FEATURE_PRECISIONS[plan["feature_precision"]]
        model, do_normalize = load_embedder_model(plan["embedder"])
        model = model.float().eval().to(device)
        done = 0
        with torch.no_grad():
            for group in embed_groups:
                for indices, audio, real in _batches(group, batch_size, workers):
                    with torch.autocast("xla", dtype=torch.bfloat16, enabled=plan["embedder_bf16"]):
                        feats = extract_features(model, audio.to(device), "v2", do_normalize=do_normalize)
                    sync()
                    feats = feats.float().cpu()[:real].numpy()
                    for row, item_index in enumerate(indices):
                        source, _, _, feature_path = group[item_index]
                        if np.isfinite(feats[row]).all():
                            np.save(feature_path, feats[row].astype(dtype, copy=False), allow_pickle=False)
                        else:
                            print(f"[PREP] {source} produced NaN values; skipped.", flush=True)
                    done += real
                if rank == 0:
                    print(f"[PREP] embeddings chip0: {done} clips", flush=True)


def run_cpu_leftovers(executor, workers, pitch_items, embed_items, args, threads):
    from rvc.train.extract import extract as stock

    futures = []
    for i in range(workers):
        if pitch_items[i::workers]:
            futures.append(executor.submit(stock.process_files, pitch_items[i::workers], "rmvpe", "cpu", threads))
    for i in range(workers):
        if embed_items[i::workers]:
            futures.append(
                executor.submit(
                    stock.process_file_embedding,
                    embed_items[i::workers],
                    args.embedder,
                    i,
                    "cpu",
                    threads,
                    args.feature_precision,
                )
            )
    return futures


def run_extract(args, exp_dir):
    from rvc.lib.predictors.f0 import load_high_register_settings
    from rvc.train.extract.extract import _grouped_by_length
    from rvc.train.extract.noise_mutes import prepare_noise_mutes
    from rvc.train.extract.preparing_files import generate_config, generate_filelist

    for name in ("f0", "f0_voiced", "extracted"):
        os.makedirs(os.path.join(exp_dir, name), exist_ok=True)
    info_path = os.path.join(exp_dir, "model_info.json")
    data = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as handle:
            data = json.load(handle)
    data["embedder_model"] = args.embedder
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)

    files = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "sliced_audios", "*"))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        files.append([
            path,
            os.path.join(exp_dir, "f0", name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", name + ".npy"),
            os.path.join(exp_dir, "extracted", os.path.splitext(name)[0] + ".npy"),
        ])
    if not files:
        sys.exit(f"No sliced audio in {exp_dir}/sliced_audios.")

    _, mute_files = prepare_noise_mutes(exp_dir, str(args.sample_rate), args.embedder, args.include_mutes)
    pitch_pending = [f for f in files if not (os.path.exists(f[1]) and os.path.exists(f[2]))]
    embed_pending = [f for f in files + mute_files if not os.path.exists(f[3])]

    pitch_tpu, pitch_cpu = split_groups(_grouped_by_length(pitch_pending), args.min_group)
    embed_tpu, embed_cpu = split_groups(_grouped_by_length(embed_pending), args.min_group)
    print(
        f"[PREP] pitch: {sum(map(len, pitch_tpu))} clips in {len(pitch_tpu)} TPU shapes, {len(pitch_cpu)} on CPU | "
        f"embeddings: {sum(map(len, embed_tpu))} clips in {len(embed_tpu)} TPU shapes, {len(embed_cpu)} on CPU",
        flush=True,
    )

    high_register = load_high_register_settings()
    high_register["mode"] = "true_pitch"
    plan = {
        "pitch_groups": pitch_tpu,
        "embed_groups": embed_tpu,
        "batch_size": args.batch_size,
        "embedder": args.embedder,
        "feature_precision": args.feature_precision,
        "embedder_bf16": args.embedder_bf16,
        "high_register": high_register,
    }

    started = time.time()
    cpus = os.cpu_count() or 4
    threads = 4
    # Half the host stays with the TPU processes' audio loaders.
    cpu_workers = max(1, (cpus // 2) // threads)
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers, mp_context=ctx) as executor:
        futures = run_cpu_leftovers(executor, cpu_workers, pitch_cpu, embed_cpu, args, threads)
        if pitch_tpu or embed_tpu:
            import torch_xla

            if hasattr(torch_xla, "launch"):
                torch_xla.launch(_mp_fn, args=(plan,), debug_single_process=args.single_process)
            else:
                import torch_xla.distributed.xla_multiprocessing as xmp

                xmp.spawn(_mp_fn, args=(plan,), nprocs=1 if args.single_process else None)
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"[PREP] extraction finished in {time.time() - started:.1f}s", flush=True)

    generate_config(str(args.sample_rate), exp_dir, args.vocoder, args.embedder)
    generate_filelist(exp_dir, str(args.sample_rate), args.include_mutes, args.embedder, args.vocoder)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.chdir(REPO_ROOT)

    from rvc.configs.vocoders import get_vocoder_sample_rates, normalize_vocoder

    args.vocoder = normalize_vocoder(args.vocoder)
    if args.sample_rate not in get_vocoder_sample_rates(args.vocoder):
        sys.exit(f"{args.vocoder} has no configuration for {args.sample_rate} Hz.")

    exp_dir = os.path.join(REPO_ROOT, "logs", args.model_name)
    os.makedirs(exp_dir, exist_ok=True)
    ensure_models(args.embedder)

    if not args.skip_preprocess:
        if not args.dataset:
            sys.exit("--dataset is required unless --skip-preprocess is given.")
        run_preprocess(args, exp_dir)
    if not args.skip_extract:
        run_extract(args, exp_dir)


if __name__ == "__main__":
    main()
