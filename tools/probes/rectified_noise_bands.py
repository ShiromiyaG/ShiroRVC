"""Where across frequency does the rectified vocoder lack inter-harmonic floor,
and how does it pass its excitation noise on?

Per 500 Hz band, on windows where f0 holds within 1% (as
``rectified_noise_sweep``), render minus original:

``floor``   median power between harmonics, dB.
``peaks``   power at the harmonics, dB.

And the noise's transfer: renders with and without the voiced excitation
noise, drawn from the same seed, differ only by what that noise became. Its
PSD over voiced samples, against the white noise put in, in dB relative to
the 1-2 kHz bands.

Usage::

    python tools/probes/rectified_noise_bands.py \
        --vocoder logs/pretrain/vocoder/pretrain_vocoder_3e_64680s.pth
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "probes"))

from rectified_noise_sweep import HNR_N, stable_windows  # noqa: E402
from rvc.rectified.common import f0_to_mel_rate, mel_frames, read_filelist  # noqa: E402
from rvc.rectified.mel import LogMel, normalize_mel  # noqa: E402
from rvc.rectified.vocoder import load_vocoder  # noqa: E402

BAND = 500


def floor_and_peaks(x, sr, f0, starts, bands, n=HNR_N):
    """Per band, the mean dB of the inter-harmonic medians and of the peaks."""
    df = sr / n
    window = np.hanning(n)
    floor = [[] for _ in range(bands)]
    peaks = [[] for _ in range(bands)]
    for start in starts:
        f = float(np.median(f0[start * 100 // sr : (start + n) * 100 // sr + 1]))
        power = np.abs(np.fft.rfft(x[start : start + n] * window)) ** 2 + 1e-30
        for k in range(1, int(sr / 2 / f)):
            a, b = int((k + 0.25) * f / df), int((k + 0.75) * f / df)
            c = int(round(k * f / df))
            if b <= a or b >= len(power) - 3:
                continue
            band = int((k + 0.5) * f // BAND)
            if band < bands:
                floor[band].append(10 * np.log10(np.median(power[a : b + 1])))
                peaks[band].append(10 * np.log10(power[max(c - 2, 0) : c + 3].max()))
    return floor, peaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder", default="logs/pretrain/vocoder/pretrain_vocoder_3e_64680s.pth")
    ap.add_argument("--experiments", nargs="+", default=["andre-rectified", "edu-rebirth-rectified"])
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.vocoder, map_location="cpu", weights_only=True)
    data = checkpoint["config"]["data"]
    trained_std = float(checkpoint["config"]["vocoder"]["model"]["source_noise_std"])
    vocoder, _ = load_vocoder(args.vocoder, data)
    vocoder = vocoder.to(device)
    source = vocoder.m_source
    sr, hop = int(data["sample_rate"]), int(data["hop_length"])
    logmel = LogMel.from_config(data)
    bands = sr // 2 // BAND

    clips = []
    for name in args.experiments:
        entries = [e for e in read_filelist(name) if "mute" not in Path(e[0]).name]
        random.Random(0).shuffle(entries)
        scored = []
        for entry in entries[: args.candidates]:
            f0 = np.load(entry[3]).astype(np.float64)
            scored.append((len(stable_windows(f0, sr, HNR_N)), entry))
        scored.sort(key=lambda s: -s[0])
        clips += [entry for _, entry in scored[: args.clips]]

    def render(mel, f0_mel, noise_std):
        source.noise_std = noise_std
        torch.manual_seed(0)
        with torch.no_grad():
            out = vocoder(normalize_mel(mel.to(device), data), f0_mel.to(device))[0, 0]
        return out.double().cpu().numpy()

    gt_floor = [[] for _ in range(bands)]
    gt_peaks = [[] for _ in range(bands)]
    re_floor = [[] for _ in range(bands)]
    re_peaks = [[] for _ in range(bands)]
    noise_psd, noise_psd2 = 0.0, 0.0
    for entry in clips:
        audio, _ = sf.read(entry[0], dtype="float64")
        audio = audio.mean(-1) if audio.ndim == 2 else audio
        f0 = np.load(entry[3]).astype(np.float64)
        frames = min(len(audio) // hop, mel_frames(len(f0), sr, hop))
        audio = audio[: frames * hop]
        with torch.no_grad():
            mel = logmel(torch.from_numpy(audio).float().unsqueeze(0))[:, :, :frames]
        f0_mel = f0_to_mel_rate(torch.from_numpy(f0).float(), frames, sr, hop).unsqueeze(0)
        starts = stable_windows(f0, sr, HNR_N)

        noisy = render(mel, f0_mel, trained_std)[: len(audio)]
        clean = render(mel, f0_mel, 0.0)[: len(audio)]
        noisier = render(mel, f0_mel, 2 * trained_std)[: len(audio)]
        for target, x in ((gt_floor, audio), (re_floor, noisy)):
            floor, peaks = floor_and_peaks(x, sr, f0, starts, bands)
            for band in range(bands):
                target[band] += floor[band]
                (gt_peaks if target is gt_floor else re_peaks)[band] += peaks[band]

        voiced = np.interp(np.arange(len(audio)), np.arange(len(f0)) * sr / 100, f0) > 0
        for diff, acc in ((noisy - clean, 1), (noisier - clean, 2)):
            part = diff[voiced]
            if len(part) < 4096:
                continue
            freqs, psd = ss.welch(part, sr, nperseg=2048)
            if acc == 1:
                noise_psd = noise_psd + psd * len(part)
            else:
                noise_psd2 = noise_psd2 + psd * len(part)

    # The white noise put in has a flat PSD of std^2 / (sr / 2) per Hz.
    transfer = 10 * np.log10(noise_psd / noise_psd.sum() * len(noise_psd) + 1e-30)
    ratio = 10 * np.log10(noise_psd2 / noise_psd + 1e-30)
    ref = np.mean(transfer[(freqs >= 1000) & (freqs < 2000)])

    print(f"vocoder {Path(args.vocoder).name}, noise_std {trained_std:g}, {len(clips)} clips\n")
    print(f"{'band kHz':>10s}{'floor':>9s}{'peaks':>9s}{'noise gain':>12s}{'2x noise':>10s}   windows")
    for band in range(bands):
        lo, hi = band * BAND, (band + 1) * BAND
        sel = (freqs >= lo) & (freqs < hi)
        if not gt_floor[band]:
            continue
        floor = np.mean(re_floor[band]) - np.mean(gt_floor[band])
        peaks = np.mean(re_peaks[band]) - np.mean(gt_peaks[band])
        print(f"{lo / 1000:5.1f}-{hi / 1000:<4.1f}{floor:+9.1f}{peaks:+9.1f}"
              f"{np.mean(transfer[sel]) - ref:+12.1f}{np.mean(ratio[sel]):+10.1f}   {len(gt_floor[band])}")
    print("\nfloor/peaks: render minus original, dB. noise gain: what the voiced noise became, "
          "per Hz, relative to 1-2 kHz (white in). 2x noise: +6.0 is linear.")


if __name__ == "__main__":
    main()
