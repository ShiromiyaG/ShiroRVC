"""What does the decoder do to the excitation's noise in the high bands?

On the longest voiced stretch of ``logs/reference``, with ``m_p`` held at its
mean and f0 flat, the render is periodic without the excitation noise, so the
difference between a render with noise and one without is the noise's own
contribution.  Per band it reports, against the reference audio's same band:

``jitter``     mean |frame-to-frame change| of band energy, dB.
``shuffled``   the same after randomising the phases of the whole stretch:
               same spectrum, stationary.  Close to ``jitter`` means the
               fluctuation comes from a peaky spectrum; well under it means
               the noise is modulated in time.
``flatness``   spectral flatness of the band (1 = white, near 0 = few peaks).
``kurtosis``   of the band-passed signal; 3 is Gaussian, higher is bursty.
``env <50Hz``  share of the band envelope's modulation power between 2 and
               50 Hz, i.e. at the rate of the vertical lines.
``HNR``        harmonic peaks over the inter-harmonic floor, dB, at each
               frame's f0; higher is cleaner.  The noise's own row has no harmonics, so
               its HNR reads near zero.

Also reports how the contribution scales with ``noise_std``: 2x the noise
giving 2x the contribution means the decoder passes it through linearly.

Usage::

    python archive/probes/noise_render.py --log-dir logs/pretrain_bigvgan_teste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "probes"))

from decoder_determinism import build, reference_inputs  # noqa: E402
from hf_jitter_sources import longest_voiced_run  # noqa: E402
from rvc.lib.algorithm.energy import frame_energy  # noqa: E402
from rvc.lib.audio_io import load_audio  # noqa: E402

BANDS = ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def bandpass(x, sr, lo, hi):
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spectrum[(freqs < lo) | (freqs >= hi)] = 0
    return np.fft.irfft(spectrum, len(x))


def jitter(x, sr, hop, lo, hi):
    n_fft = 1024
    count = (len(x) - n_fft) // hop + 1
    frames = np.stack([x[i * hop : i * hop + n_fft] for i in range(count)])
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= lo) & (freqs < hi)
    db = 10 * np.log10(power[:, band].sum(axis=1) + 1e-20)
    return float(np.abs(np.diff(db)).mean())


def shuffled(x, seed=0):
    spectrum = np.fft.rfft(x)
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, spectrum.shape)
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phase), len(x))


def hnr(x, sr, hop, f0_frames, lo, hi, n=4096):
    """Harmonic peaks over the median between them, dB, averaged over frames;
    ``f0_frames`` is the f0 at the frame rate from the start of ``x``."""
    ratios = []
    df = sr / n
    for start in range(0, len(x) - n, n // 2):
        f0 = float(f0_frames[min((start + n // 2) // hop, len(f0_frames) - 1)])
        power = np.abs(np.fft.rfft(x[start : start + n] * np.hanning(n))) ** 2
        peaks, floor = [], []
        for k in range(int(np.ceil(lo / f0)), int(hi / f0) + 1):
            c = int(round(k * f0 / df))
            peaks.append(power[c - 2 : c + 3].max())
            floor.append(np.median(power[int((k + 0.25) * f0 / df) : int((k + 0.75) * f0 / df)]))
        if peaks:
            ratios.append(10 * np.log10(np.sum(peaks) / np.sum(floor)))
    return float(np.mean(ratios))


def stats(x, sr, hop, lo, hi, f0_frames):
    band = bandpass(x, sr, lo, hi)
    f, psd = ss.welch(x, sr, nperseg=2048)
    sel = (f >= lo) & (f < hi)
    flatness = float(np.exp(np.mean(np.log(psd[sel] + 1e-30))) / np.mean(psd[sel]))
    kurtosis = float(np.mean(band**4) / np.mean(band**2) ** 2)
    env = np.abs(ss.hilbert(band))
    env = env - env.mean()
    ef, ep = ss.periodogram(env, sr)
    low = float(ep[(ef >= 2) & (ef < 50)].sum() / ep[ef >= 2].sum())
    return {
        "jitter": jitter(x, sr, hop, lo, hi),
        "shuffled": jitter(shuffled(x), sr, hop, lo, hi),
        "flatness": flatness,
        "kurtosis": kurtosis,
        "env": low,
        "rms": float(np.sqrt(np.mean(band**2))),
        "hnr": hnr(x, sr, hop, f0_frames, lo, hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain_bigvgan_teste")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    # Training saves the EMA as ``model``, so these are the preview's weights.
    net_g, sr = build(log_dir, ckpt_path, ema=False)
    if net_g.latent_mode != "direct":
        raise SystemExit(f"{ckpt_path.name} is not a direct-mode model.")
    hop = sr // 100

    ref_dir = Path(args.ref_dir)
    phone, lengths, pitch, pitchf = reference_inputs(ref_dir, None, net_g.content_interpolation)
    frames = int(lengths[0])
    wave = load_audio(str(ref_dir / "ref_audio.wav"), sr)[: frames * hop].astype(np.float64)
    energy = None
    if getattr(net_g, "energy_embedding", None) is not None:
        energy = frame_energy(torch.from_numpy(wave).float().view(1, -1), sr, frames)
    sid = torch.tensor([args.sid])

    start, end = longest_voiced_run(pitchf[0].numpy() > 0)
    start, end = start + args.margin, end - args.margin
    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)
        m_p, _logs_p, x_mask = net_g.encode_content(phone, lengths, pitch, energy)
    held = m_p.clone()
    held[..., start:end] = m_p[..., start:end].mean(dim=-1, keepdim=True)
    held = held * x_mask
    flat = pitchf.clone()
    flat[:, start:end] = pitchf[:, start:end].log().mean().exp()

    source = net_g.dec.m_source
    base_std = source.noise_std

    def render(noise_std):
        source.noise_std = noise_std
        torch.manual_seed(args.seed)
        with torch.no_grad():
            out = net_g.dec(held, flat, g)[0, 0].double().numpy()
        source.noise_std = base_std
        # The stretch, off its edges by 10 frames, where every input is held.
        return out[(start + 10) * hop : (end - 10) * hop]

    clean = render(0.0)
    noisy = render(base_std)
    noisier = render(2 * base_std)
    part = noisy - clean
    part2 = noisier - clean
    real = wave[(start + 10) * hop : (end - 10) * hop]
    real_f0 = pitchf[0, start + 10 : end - 10].numpy()
    flat_f0 = flat[0, start + 10 : end - 10].numpy()

    print(
        f"checkpoint: {ckpt_path.name}   stretch frames {start + 10}-{end - 10} "
        f"({(end - start - 20) / 100:.2f} s)   f0 {float(flat[0, start]):.1f} Hz   "
        f"noise_std {base_std:g}\n"
    )
    columns = ("jitter", "shuffled", "flatness", "kurtosis", "env")
    print(f"{'':32s}" + "".join(f"{c:>10s}" for c in ("jitter", "shuffled", "flatness", "kurtosis", "env<50Hz", "rms dBFS", "HNR")))
    for lo, hi in BANDS:
        print(f"{lo // 1000}-{hi // 1000} kHz")
        for label, x, f0_frames in (
            ("original", real, real_f0),
            ("render, noise on", noisy, flat_f0),
            ("render, noise off", clean, flat_f0),
            ("noise's contribution", part, flat_f0),
        ):
            s = stats(x, sr, hop, lo, hi, f0_frames)
            print(
                f"  {label:30s}"
                + "".join(f"{s[c]:10.2f}" for c in columns)
                + f"{20 * np.log10(s['rms'] + 1e-20):10.1f}"
                + f"{s['hnr']:10.1f}"
            )
        a, b = bandpass(part, sr, lo, hi), bandpass(part2, sr, lo, hi)
        gain = np.sqrt(np.mean(b**2) / np.mean(a**2))
        corr = float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))
        print(f"  2x noise: contribution x{gain:.2f}, correlation {corr:.3f}\n")


if __name__ == "__main__":
    main()
