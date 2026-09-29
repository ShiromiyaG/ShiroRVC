"""Does a phase-coherent excitation give the high bands their glottal pulses?

Renders ``logs/reference`` through ``infer`` with the partials' starting phases
random (as trained) and coherent, and reports on the longest voiced stretch,
per band, against the reference audio:

``kurtosis``  of the band-passed signal; 3 is Gaussian, a pulse per period
              reads higher.
``env@f0``    share of the band envelope's modulation power at f0 and 2 f0,
              i.e. how much of the band arrives once per period.
``HNR``       harmonic peaks over the inter-harmonic floor, dB.

The model was trained on random phases, so the coherent render is off its
training distribution: it shows what the source can carry, not what a model
trained on it would do.  Renders go to ``--save-dir`` for listening.

Usage::

    python tools/probes/source_phase_ab.py --log-dir logs/pretrain_bigvgan_teste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "probes"))

from decoder_determinism import build, reference_inputs  # noqa: E402
from hf_jitter_sources import longest_voiced_run  # noqa: E402
from noise_render import hnr  # noqa: E402
from rvc.lib.algorithm.energy import frame_energy  # noqa: E402
from rvc.lib.audio_io import load_audio  # noqa: E402

BANDS = ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 15000))


def pulse_stats(x, sr, hop, f0_frames, lo, hi):
    band = ss.sosfiltfilt(ss.butter(8, [lo, hi], btype="band", fs=sr, output="sos"), x)
    kurtosis = float(np.mean(band**4) / np.mean(band**2) ** 2)
    env = np.abs(ss.hilbert(band))
    env = env - env.mean()
    freqs = np.fft.rfftfreq(len(env), 1 / sr)
    power = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    f0 = float(np.median(f0_frames))
    near = (np.abs(freqs - f0) < 0.07 * f0) | (np.abs(freqs - 2 * f0) < 0.07 * f0)
    share = float(power[near].sum() / power[freqs > 5].sum())
    return kurtosis, share, hnr(x, sr, hop, f0_frames, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain_bigvgan_teste")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-dir", default=None, help="default: <log-dir>/source_phase_ab")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    # Training saves the EMA as ``model``, so these are the preview's weights.
    net_g, sr = build(log_dir, ckpt_path, ema=False)
    source = net_g.dec.m_source
    if source.dim < 2:
        raise SystemExit(f"{ckpt_path.name} has a single-partial source; phase does nothing.")
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
    f0_frames = pitchf[0, start:end].numpy()
    lo_s, hi_s = start * hop, end * hop

    save_dir = Path(args.save_dir) if args.save_dir else log_dir / "source_phase_ab"
    save_dir.mkdir(parents=True, exist_ok=True)
    sf.write(save_dir / "original.wav", wave, sr)

    trained = source.harmonic_phase
    renders = {"original": wave}
    for phase in ("random", "coherent"):
        source.harmonic_phase = phase
        torch.manual_seed(args.seed)
        with torch.no_grad():
            out, *_ = net_g.infer(phone, lengths, pitch, pitchf, sid, energy=energy)
        out = out[0, 0, : len(wave)].double().numpy()
        renders[f"{phase} phase"] = out
        sf.write(save_dir / f"{phase}.wav", out, sr)
    source.harmonic_phase = trained

    print(
        f"checkpoint: {ckpt_path.name} (trained with {trained} phase)   "
        f"stretch {(end - start) / 100:.2f} s   f0 ~{np.median(f0_frames):.0f} Hz\n"
    )
    print(f"{'':16s}" + "".join(f"{f'{lo}-{hi}':>24s}" for lo, hi in BANDS))
    print(f"{'':16s}" + "".join(f"{'kurt':>8s}{'env@f0':>8s}{'HNR':>8s}" for _ in BANDS))
    for label, x in renders.items():
        seg = x[lo_s:hi_s]
        row = f"{label:16s}"
        for lo, hi in BANDS:
            k, share, h = pulse_stats(seg, sr, hop, f0_frames, lo, hi)
            row += f"{k:8.2f}{share:8.2f}{h:8.1f}"
        print(row)
    print(f"\nrenders in {save_dir}")


if __name__ == "__main__":
    main()
