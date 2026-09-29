"""Where do the preview's high-band vertical lines come from?

Renders the ``logs/reference`` clip through the direct path with one input
low-passed over frames at a time: the energy (or none at all, what the 10%
``energy_dropout`` trained for), the content features, and ``m_p``, the
decoder's input.  If smoothing ``m_p`` leaves the lines, the decoder makes them
itself; the last cases then smooth the f0 it is driven by (the encoder keeps
the raw pitch) or silence the excitation's noise.  Reports, per band over voiced frames, the mean frame-to-frame change
of band energy; the reference audio gets the same measurement, as the floor.

Usage::

    python tools/probes/energy_ab.py --log-dir logs/pretrain_bigvgan_teste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "probes"))

from decoder_determinism import build, reference_inputs  # noqa: E402
from rvc.lib.algorithm.energy import frame_energy  # noqa: E402
from rvc.lib.audio_io import load_audio  # noqa: E402

BANDS = ((500, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def smooth_frames(x: torch.Tensor, width: int) -> torch.Tensor:
    """Moving average over the last axis, ``x`` [batch, channels, frames];
    replicate padding keeps the ends."""
    if width <= 1:
        return x
    padded = F.pad(x, (width // 2, (width - 1) // 2), mode="replicate")
    return F.avg_pool1d(padded, width, stride=1)


def smooth_f0(pitchf: torch.Tensor, width: int) -> torch.Tensor:
    """Moving average of log f0 over voiced frames only; unvoiced stays 0."""
    if width <= 1:
        return pitchf
    voiced = (pitchf > 0).to(pitchf.dtype).unsqueeze(1)
    log_f0 = torch.log(pitchf.clamp_min(1.0)).unsqueeze(1) * voiced
    mean = smooth_frames(log_f0, width) / smooth_frames(voiced, width).clamp_min(1e-6)
    return (torch.exp(mean) * voiced).squeeze(1)


def render(
    net_g, phone, lengths, pitch, pitchf, sid, energy,
    content=1, latent=1, f0=1, source_noise=True,
):
    """``infer``'s direct path, with the content features, ``m_p`` and the
    decoder's f0 low-passed over ``content``, ``latent`` and ``f0`` frames."""
    noise_std = net_g.dec.m_source.noise_std
    if not source_noise:
        net_g.dec.m_source.noise_std = 0.0
    try:
        with torch.no_grad():
            g = net_g.emb_g(sid).unsqueeze(-1)
            phone = smooth_frames(phone.transpose(1, 2), content).transpose(1, 2)
            m_p, _logs_p, x_mask = net_g.encode_content(phone, lengths, pitch, energy)
            m_p = smooth_frames(m_p, latent)
            return net_g.dec(m_p * x_mask, smooth_f0(pitchf, f0), g)
    finally:
        net_g.dec.m_source.noise_std = noise_std


def band_jitter(wave: np.ndarray, sr: int, hop: int, voiced: np.ndarray):
    """Mean |frame-to-frame change| of each band's energy in dB, and its level."""
    n_fft = 1024
    count = min((len(wave) - n_fft) // hop + 1, len(voiced))
    frames = np.stack([wave[i * hop : i * hop + n_fft] for i in range(count)])
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Centre of analysis frame i is sample i*hop + n_fft/2, i.e. f0 frame i + 1.6.
    shift = int(round(n_fft / 2 / hop))
    v = voiced[shift : shift + count]
    pairs = v[1:] & v[:-1]
    jitter, level = [], []
    for lo, hi in BANDS:
        band = (freqs >= lo) & (freqs < min(hi, sr / 2))
        db = 10 * np.log10(power[:, band].sum(axis=1) + 1e-12)
        jitter.append(float(np.abs(np.diff(db))[pairs].mean()))
        level.append(float(db[v].mean()))
    return jitter, level


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain_bigvgan_teste")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=5, help="frames for the low-passed cases")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-dir", default=None, help="default: <log-dir>/energy_ab")
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
    has_energy = getattr(net_g, "energy_embedding", None) is not None
    device = torch.device(args.device)
    net_g = net_g.to(device)
    hop = sr // 100

    ref_dir = Path(args.ref_dir)
    phone, lengths, pitch, pitchf = (t.to(device) for t in reference_inputs(ref_dir, None, net_g.content_interpolation))
    frames = int(lengths[0])
    wave = load_audio(str(ref_dir / "ref_audio.wav"), sr)[: frames * hop].astype(np.float32)
    audio = torch.from_numpy(wave).view(1, -1).to(device)
    energy = frame_energy(audio, sr, frames)
    sid = torch.tensor([args.sid], device=device)
    voiced = pitchf[0].cpu().numpy() > 0

    if not has_energy:
        energy = None
    n = args.smooth
    cases = [("as preview", {})]
    if has_energy:
        cases += [
            (f"energy LP {n}fr", {"energy": smooth_frames(energy.unsqueeze(1), n).squeeze(1)}),
            ("energy=None", {"energy": None}),
        ]
    cases += [
        (f"content LP {n}fr", {"content": n}),
        (f"m_p LP {n}fr", {"latent": n}),
        (f"m_p LP {3 * n}fr", {"latent": 3 * n}),
        (f"m_p+f0 LP {n}fr", {"latent": n, "f0": n}),
        (f"m_p LP {n}fr, no noise", {"latent": n, "source_noise": False}),
    ]

    save_dir = Path(args.save_dir) if args.save_dir else log_dir / "energy_ab"
    save_dir.mkdir(parents=True, exist_ok=True)
    sf.write(save_dir / "original.wav", wave, sr)

    print(
        f"checkpoint: {ckpt_path.name}   sr={sr}   "
        f"frames={frames}   voiced={int(voiced.sum())}"
    )
    if has_energy:
        # The input's own jitter: 1 unit of the [-1, 1] map is 35 dB.
        energy_db = energy[0].cpu().numpy() * 35.0
        pairs = voiced[1:] & voiced[:-1]
        print(f"energy input |dE| over voiced frames: {np.abs(np.diff(energy_db))[pairs].mean():.2f} dB")
    print()

    header = "  ".join(f"{lo // 1000 if lo >= 1000 else lo}-{hi // 1000}k".rjust(8) for lo, hi in BANDS)
    print(f"{'frame-to-frame |dE| (dB)':26s}{header}   {'level vs original (dB)'}")

    ref_jitter, ref_level = band_jitter(wave, sr, hop, voiced)
    print(f"{'original':26s}" + "  ".join(f"{j:8.2f}" for j in ref_jitter))

    for label, case in cases:
        options = {"energy": energy, **case}
        out = render(net_g, phone, lengths, pitch, pitchf, sid, **options)
        out = out[0, 0, : len(wave)].float().cpu().numpy()
        jitter, level = band_jitter(out, sr, hop, voiced)
        gaps = "  ".join(f"{l - r:+5.1f}" for l, r in zip(level, ref_level))
        print(f"{label:26s}" + "  ".join(f"{j:8.2f}" for j in jitter) + f"   {gaps}")
        name = label.replace(" ", "_").replace("(", "").replace(")", "")
        name = name.replace("=", "_").replace("+", "_").replace(",", "")
        sf.write(save_dir / f"{name}.wav", out, sr)

    print(f"\nrenders in {save_dir}")


if __name__ == "__main__":
    main()
