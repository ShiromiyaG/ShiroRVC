"""How does the rectified vocoder's ``source_noise_std`` change what it renders?

Renders clips from the ground-truth mel at several ``source_noise_std`` values
(it holds no weights, so it can change at inference) and a few noise draws,
and compares against the original audio:

``HNR``      harmonic peaks over the inter-harmonic floor per band, dB, on
             windows where f0 holds within 1%; the render minus the original.
``flicker``  mean |band energy - its 9-frame moving average| on voiced frames,
             dB, at a 256-sample hop (not the model's 512).
``MR-STFT``  mean |log magnitude| difference over three resolutions.

Usage::

    python tools/probes/rectified_noise_sweep.py \
        --vocoder logs/pretrain/vocoder/pretrain_vocoder_3e_64680s.pth \
        --experiments andre-rectified edu-rebirth-rectified
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rvc.rectified.common import f0_to_mel_rate, mel_frames, read_filelist  # noqa: E402
from rvc.rectified.mel import LogMel, normalize_mel  # noqa: E402
from rvc.rectified.vocoder import load_vocoder  # noqa: E402

HNR_BANDS = ((400, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))
FLICKER_BANDS = ((300, 2000), (2000, 8000), (8000, 16000))
HNR_N = 2048
FLICKER_N, FLICKER_HOP = 2048, 256


def stable_windows(f0, sr, n):
    """Window starts (hop n/2) whose 10 ms f0 is voiced and within 1%."""
    starts = []
    for start in range(0, int(len(f0) * sr / 100) - n, n // 2):
        seg = f0[start * 100 // sr : (start + n) * 100 // sr + 1]
        if len(seg) and (seg > 0).all() and seg.max() / seg.min() < 1.01:
            starts.append(start)
    return starts


def hnr(x, sr, f0, starts, lo, hi, n=HNR_N):
    df = sr / n
    window = np.hanning(n)
    ratios = []
    for start in starts:
        f = float(np.median(f0[start * 100 // sr : (start + n) * 100 // sr + 1]))
        power = np.abs(np.fft.rfft(x[start : start + n] * window)) ** 2
        peaks, floor = [], []
        for k in range(max(1, int(np.ceil(lo / f))), int(hi / f) + 1):
            c = int(round(k * f / df))
            a, b = int((k + 0.25) * f / df), int((k + 0.75) * f / df)
            if b <= a or b >= len(power) - 3:
                continue
            peaks.append(power[max(c - 2, 0) : c + 3].max())
            floor.append(np.median(power[a : b + 1]))
        if peaks:
            ratios.append(10 * np.log10(np.sum(peaks) / (np.sum(floor) + 1e-30)))
    return float(np.mean(ratios)) if ratios else float("nan")


def band_db(x, sr, lo, hi):
    count = (len(x) - FLICKER_N) // FLICKER_HOP + 1
    frames = np.lib.stride_tricks.sliding_window_view(x, FLICKER_N)[::FLICKER_HOP][:count]
    power = np.abs(np.fft.rfft(frames * np.hanning(FLICKER_N), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FLICKER_N, 1.0 / sr)
    return 10 * np.log10(power[:, (freqs >= lo) & (freqs < hi)].sum(1) + 1e-20)


def flicker(x, sr, voiced, lo, hi):
    db = band_db(x, sr, lo, hi)
    smooth = np.convolve(np.pad(db, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    keep = voiced[: len(db)]
    return float(np.abs(db - smooth)[keep].mean()) if keep.any() else float("nan")


def mr_stft(x, y):
    total = 0.0
    for n in (512, 1024, 2048):
        window = torch.hann_window(n, dtype=torch.float64)
        sx = torch.stft(torch.from_numpy(x), n, n // 4, window=window, return_complex=True).abs()
        sy = torch.stft(torch.from_numpy(y), n, n // 4, window=window, return_complex=True).abs()
        total += float((torch.log(sx + 1e-5) - torch.log(sy + 1e-5)).abs().mean())
    return total / 3


def metrics(x, sr, f0, starts, voiced):
    out = {f"hnr {lo // 1000 if lo >= 1000 else lo / 1000:g}-{hi // 1000}k": hnr(x, sr, f0, starts, lo, hi)
           for lo, hi in HNR_BANDS}
    out.update({f"flick {lo / 1000:g}-{hi // 1000}k": flicker(x, sr, voiced, lo, hi) for lo, hi in FLICKER_BANDS})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder", default="logs/pretrain/vocoder/pretrain_vocoder_3e_64680s.pth")
    ap.add_argument("--experiments", nargs="+", default=["andre-rectified", "edu-rebirth-rectified"])
    ap.add_argument("--clips", type=int, default=12, help="per experiment, most stable windows first")
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--values", type=float, nargs="+", default=[0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01])
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.vocoder, map_location="cpu", weights_only=True)
    data = checkpoint["config"]["data"]
    trained_std = checkpoint["config"]["vocoder"]["model"]["source_noise_std"]
    vocoder, _ = load_vocoder(args.vocoder, data)
    vocoder = vocoder.to(device)
    source = vocoder.m_source
    sr, hop = int(data["sample_rate"]), int(data["hop_length"])
    logmel = LogMel.from_config(data)

    import soundfile as sf

    clips = []
    for name in args.experiments:
        entries = [e for e in read_filelist(name) if "mute" not in Path(e[0]).name]
        random.Random(0).shuffle(entries)
        scored = []
        for entry in entries[: args.candidates]:
            audio, rate = sf.read(entry[0], dtype="float64")
            if rate != sr:
                raise SystemExit(f"{entry[0]} is {rate} Hz, the vocoder {sr} Hz.")
            audio = audio.mean(-1) if audio.ndim == 2 else audio
            f0 = np.load(entry[3]).astype(np.float64)
            starts = stable_windows(f0, sr, HNR_N)
            scored.append((len(starts), entry[0], audio, f0, starts))
        scored.sort(key=lambda s: -s[0])
        clips += scored[: args.clips]

    print(f"vocoder {Path(args.vocoder).name}, trained at source_noise_std {trained_std:g}; "
          f"{len(clips)} clips, {sum(c[0] for c in clips)} stable windows, {args.draws} draws\n")

    reference, renders = [], {value: [] for value in args.values}
    for _, path, audio, f0, starts in clips:
        frames = min(len(audio) // hop, mel_frames(len(f0), sr, hop))
        audio = audio[: frames * hop]
        with torch.no_grad():
            mel = logmel(torch.from_numpy(audio).float().unsqueeze(0))[:, :, :frames]
        f0_mel = f0_to_mel_rate(torch.from_numpy(f0).float(), frames, sr, hop).unsqueeze(0)
        voiced = np.interp(np.arange(0, len(audio) - FLICKER_N + 1, FLICKER_HOP) + FLICKER_N / 2,
                           np.arange(len(f0)) * sr / 100, f0) > 0
        reference.append(metrics(audio, sr, f0, starts, voiced))
        for value in args.values:
            source.noise_std = value
            for draw in range(args.draws):
                torch.manual_seed(draw)
                with torch.no_grad():
                    out = vocoder(normalize_mel(mel.to(device), data), f0_mel.to(device))[0, 0]
                out = out.double().cpu().numpy()[: len(audio)]
                row = metrics(out, sr, f0, starts, voiced)
                row["MR-STFT"] = mr_stft(out, audio)
                renders[value].append(row)

    keys = list(reference[0])
    gt = {k: np.nanmean([r[k] for r in reference]) for k in keys}
    print(f"{'noise_std':>10s}" + "".join(f"{k:>13s}" for k in keys) + f"{'MR-STFT':>10s}")
    print(f"{'original':>10s}" + "".join(f"{gt[k]:13.2f}" for k in keys))
    for value, rows in renders.items():
        mean = {k: np.nanmean([r[k] for r in rows]) for k in keys + ["MR-STFT"]}
        mark = " *" if value == trained_std else ""
        print(f"{value:10g}" + "".join(f"{mean[k] - gt[k]:+13.2f}" for k in keys)
              + f"{mean['MR-STFT']:10.4f}{mark}")
    print("\nRows below the original are render minus original: HNR under 0 means more "
          "inter-harmonic floor than the original, flicker over 0 more flicker. * = trained value.")


if __name__ == "__main__":
    main()
