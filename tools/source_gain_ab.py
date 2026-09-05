"""Does the prior's roughness reach the waveform through ``source_gain``?

``source_gain`` multiplies the excitation by an envelope projected from ``z``.
At inference ``z_p = m_p + exp(logs_p) * randn * noise_scale`` draws an
*independent* ``randn`` per frame, so that envelope jumps every 10 ms -- a
multiplication, not a pointwise nonlinearity, which is why no anti-aliasing
setting ever touched what it produces.

Test bed: content frozen to one tiled frame, f0 flat, excitation dither off.
The system is then exactly periodic in 1600 samples (11 pitch periods = 5 hops)
whenever ``noise_scale`` is 0, so every measurement below is against a known
zero.  The analysis window is 32 x 1600 samples, so 220 Hz lands on bin 352
exactly and no window function is needed: the comb is exact, and the skirt is
whatever is not on it.

Two numbers per case, because they move for different reasons:

``skirt``      out-of-comb power over comb power.  This is the artefact.
``diversity``  two renders with different prior seeds, rms of the difference.
               This is what ``noise_scale`` is *for*.  A case that lowers both
               has bought quiet with flatness; one that lowers only the first
               is a free win.

Usage::

    python tools/source_gain_ab.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from decoder_determinism import build  # noqa: E402
from render_constant_f0 import coarse, pick_feature  # noqa: E402

PERIOD = 1600
BLOCKS = 32


def lowpass_frames(gain: torch.Tensor, frame_rate: float, cutoff: float):
    """Zero-phase FIR lowpass along time, at the frame rate.

    Replicate padding, so a constant envelope stays exactly constant and the
    ends are not pulled toward zero.
    """
    if cutoff >= frame_rate / 2:
        return gain
    taps = int(4 * frame_rate / cutoff) | 1
    n = torch.arange(taps, dtype=gain.dtype) - (taps - 1) / 2
    kernel = torch.sinc(2 * cutoff / frame_rate * n) * torch.hamming_window(
        taps, periodic=False, dtype=gain.dtype
    )
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = F.pad(gain, (taps // 2, taps // 2), mode="replicate")
    return F.conv1d(padded, kernel)


def patched_source_gain(dec, mode: str, frame_rate: float, cutoff: float):
    """A drop-in ``_apply_source_gain`` for one A/B case."""

    def apply(har_source, mel):
        if mode == "off":
            return har_source
        gain = F.softplus(dec.source_gain(mel))
        if mode == "mean":
            gain = gain.mean(dim=-1, keepdim=True).expand_as(gain)
        elif mode == "lowpass":
            gain = lowpass_frames(gain, frame_rate, cutoff)
        for ups in dec.source_gain_ups:
            gain = ups(gain)
        length = har_source.shape[-1]
        if gain.shape[-1] > length:
            gain = gain[..., :length]
        elif gain.shape[-1] < length:
            gain = F.pad(gain, (0, length - gain.shape[-1]), mode="replicate")
        return har_source * gain

    return apply


def skirt_db(x, sr: int, f0: float, start: int):
    """(out-of-comb / comb) in dB, and the render's rms, on a coherent block."""
    length = BLOCKS * PERIOD
    seg = x[start : start + length]
    spectrum = np.abs(np.fft.rfft(seg)) ** 2
    step = length * f0 / sr
    assert abs(step - round(step)) < 1e-9, "f0 is not on an exact bin"
    comb = np.zeros(spectrum.size, dtype=bool)
    comb[:: int(round(step))] = True
    comb[0] = False
    return (
        10 * math.log10(spectrum[~comb].sum() / spectrum[comb].sum()),
        float(np.sqrt((seg**2).mean())),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--f0", type=float, default=220.0)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--freeze-frame", type=int, default=60)
    ap.add_argument("--cutoff", type=float, default=20.0, help="gain lowpass, Hz")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    net_g, sr = build(log_dir, ckpt_path, False)
    dec = net_g.dec
    dec.m_source.noise_std = 0.0  # the dither is measured elsewhere, not here
    frame_rate = sr / dec.upp

    feature_path, sid = pick_feature(log_dir, None)
    phone = np.load(feature_path)
    index = args.freeze_frame % len(phone)
    frames = args.frames
    phone = np.repeat(phone[index : index + 1], frames, axis=0)
    phone_t = torch.from_numpy(phone).float().unsqueeze(0)
    lengths = torch.tensor([frames])
    nsff0 = np.full(frames, float(args.f0), dtype=np.float32)
    pitch = torch.from_numpy(coarse(nsff0.copy())).unsqueeze(0)
    pitchf = torch.from_numpy(nsff0).unsqueeze(0)
    sid_t = torch.tensor([sid])

    with torch.no_grad():
        g = net_g.emb_g(sid_t).unsqueeze(-1)
        m_p, logs_p, x_mask = net_g.enc_p(phone=phone_t, pitch=pitch, lengths=lengths)

    original = dec._apply_source_gain

    def render(noise_scale: float, seed: int):
        with torch.no_grad():
            z_p = m_p
            if noise_scale:
                torch.manual_seed(seed)
                z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale
            z = net_g.flow(z_p * x_mask, x_mask, g=g, reverse=True) * x_mask
            return dec(z, pitchf, g)[0, 0].double().numpy()

    start = 2 * sr  # past the padding transient at the encoder and the decoder
    cases = [
        ("source_gain on   ns=0.667", "on", 0.66666),
        ("source_gain on   ns=0.4  ", "on", 0.4),
        ("source_gain on   ns=0.2  ", "on", 0.2),
        ("source_gain on   ns=0    ", "on", 0.0),
        ("source_gain OFF  ns=0.667", "off", 0.66666),
        ("gain -> its mean ns=0.667", "mean", 0.66666),
        (f"gain LP {args.cutoff:g}Hz   ns=0.667", "lowpass", 0.66666),
        (f"gain LP {args.cutoff:g}Hz   ns=0.4  ", "lowpass", 0.4),
    ]

    print(
        f"checkpoint: {ckpt_path.name}   sr={sr}   f0={args.f0:g} Hz   "
        f"frames={frames}   frame rate={frame_rate:g} Hz"
    )
    print(
        f"window: {BLOCKS} x {PERIOD} = {BLOCKS * PERIOD} samples "
        f"({BLOCKS * PERIOD / sr:.2f} s), coherent\n"
    )
    print(f"{'case':28s} {'skirt':>9s} {'rms':>8s} {'diversity':>11s}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for label, mode, noise_scale in cases:
        dec._apply_source_gain = (
            original
            if mode == "on"
            else patched_source_gain(dec, mode, frame_rate, args.cutoff)
        )
        a = render(noise_scale, 1234)
        skirt, rms = skirt_db(a, sr, args.f0, start)
        if noise_scale:
            b = render(noise_scale, 4321)
            d = a - b
            diversity = 20 * math.log10(
                math.sqrt((d**2).mean()) / math.sqrt((a**2).mean())
            )
            diversity_text = f"{diversity:7.1f} dB"
        else:
            diversity_text = "--"
        print(f"{label:28s} {skirt:6.1f} dB {rms:8.4f} {diversity_text:>11s}")
        if save_dir:
            name = label.replace(" ", "").replace("->", "to") + ".wav"
            sf.write(save_dir / name, a, sr)

    dec._apply_source_gain = original


if __name__ == "__main__":
    main()
