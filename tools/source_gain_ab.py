"""Does the prior's roughness reach the waveform through ``source_gain``?

``source_gain`` multiplies the excitation by envelopes projected from ``z``.
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
``ih floor``   median inter-harmonic over harmonic power, per one-period frame.
``burst``      the loudest 5% of those frames over their median (all bands,
               and above 4 kHz): steady variation reads near 1 dB, bursts
               read higher at the same ``diversity``.

``--posterior`` measures stage 1's path instead, where the prior is still at
its init: ``logs/reference`` audio -> ``enc_q`` -> ``m_q`` -> ``dec``, against
the reference itself.  Per case it reports how much the gain envelope jumps
from frame to frame, how much the render's level error does (the vertical
columns in a preview's Difference panel), and the mel L1, all over voiced
frames and 250 Hz - 4 kHz.

Usage::

    python tools/source_gain_ab.py
    python tools/source_gain_ab.py --log-dir logs/my-model --posterior
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from decoder_determinism import build  # noqa: E402
from render_constant_f0 import coarse, pick_feature  # noqa: E402
from rvc.lib.audio_io import load_audio  # noqa: E402
from rvc.train.mel_processing import mel_spectrogram_torch, spectrogram_torch  # noqa: E402

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
    channels = gain.shape[1]
    kernel = (kernel / kernel.sum()).view(1, 1, -1).expand(channels, 1, -1)
    padded = F.pad(gain, (taps // 2, taps // 2), mode="replicate")
    return F.conv1d(padded, kernel, groups=channels)


def smooth_noise(noise: torch.Tensor, width: int) -> torch.Tensor:
    """White noise correlated over ``width`` frames, still unit variance per frame."""
    if width <= 1:
        return noise
    kernel = torch.hann_window(width + 2, periodic=False, dtype=noise.dtype)[1:-1]
    kernel = (kernel / kernel.square().sum().sqrt()).view(1, 1, -1)
    channels = noise.shape[1]
    padded = F.pad(noise, (width // 2, (width - 1) // 2), mode="reflect")
    return F.conv1d(padded, kernel.expand(channels, 1, -1), groups=channels)


def frame_gain_logits(dec, mel, g, mode: str, frame_rate: float, cutoff: float):
    """The pre-softplus gains at the frame rate, as one A/B case shapes them."""
    gain = dec.source_gain(mel)
    if g is not None and hasattr(dec, "source_gain_cond"):
        gain = gain + dec.source_gain_cond(g)
    if mode == "mean":
        gain = gain.mean(dim=-1, keepdim=True).expand_as(gain)
    elif mode == "lowpass":
        gain = lowpass_frames(gain, frame_rate, cutoff)
    return gain


def patched_source_gain(
    dec, mode: str, frame_rate: float, cutoff: float, fixed_mel=None
):
    """A drop-in ``_source_gain`` for one A/B case.

    ``fixed_mel`` makes the gain read that ``z`` instead of the one the trunk
    decodes.
    """

    def apply(mel, g=None):
        if mode == "off":
            return None
        if fixed_mel is not None:
            mel = fixed_mel
        gain = frame_gain_logits(dec, mel, g, mode, frame_rate, cutoff)
        for ups in dec.source_gain_ups:
            gain = ups(gain)
        return F.softplus(gain).transpose(1, 2)

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


def burst_metrics(x, sr: int, f0: float, start: int):
    """Inter-harmonic floor and burstiness over one-period frames.

    Each frame is exactly one period under a rectangular window, so a periodic
    render puts all its power on harmonic bins and every frame is identical.
    Returns, in dB: the median inter-harmonic over harmonic power per frame,
    and the loudest 5% of frames' inter-harmonic power over its median, over
    0.3-15 kHz and over 4-15 kHz.
    """
    hop = PERIOD // 4
    seg = np.asarray(x[start:])
    count = (seg.size - PERIOD) // hop + 1
    frames = np.stack([seg[i * hop : i * hop + PERIOD] for i in range(count)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(PERIOD, 1.0 / sr)
    step = PERIOD * f0 / sr
    assert abs(step - round(step)) < 1e-9, "f0 is not on an exact bin"
    harmonic = np.zeros(freqs.size, dtype=bool)
    harmonic[:: int(round(step))] = True
    band = (freqs >= 300.0) & (freqs <= 15000.0)

    def burst(mask):
        energy = power[:, mask & ~harmonic].sum(axis=1)
        top = np.sort(energy)[-max(1, energy.size // 20) :].mean()
        return energy, 10 * math.log10(top / np.median(energy))

    inter, burst_full = burst(band)
    comb = power[:, band & harmonic].sum(axis=1)
    floor = 10 * math.log10(np.median(inter) / np.median(comb))
    _, burst_high = burst(band & (freqs >= 4000.0))
    return floor, burst_full, burst_high


def gain_envelope_db(dec, z, g, mode: str, frame_rate: float, cutoff: float):
    """The per-frame source gain each case applies, in dB, averaged over channels."""
    if mode == "off":
        return np.zeros(z.shape[-1])
    with torch.no_grad():
        gain = F.softplus(frame_gain_logits(dec, z, g, mode, frame_rate, cutoff))
    return (20 * torch.log10(gain[0].double().clamp_min(1e-8))).mean(dim=0).numpy()


def log_mel_db(wave: torch.Tensor, data: dict):
    """The training loss's log-mel, in dB, as (bins, frames)."""
    mel = mel_spectrogram_torch(
        wave.view(1, -1).float(),
        data["filter_length"],
        data["n_mel_channels"],
        data["sample_rate"],
        data["hop_length"],
        data["win_length"],
        data["mel_fmin"],
        data["mel_fmax"],
    )
    return mel[0].double().numpy() * (20 / math.log(10))


def run_posterior(args, net_g, sr: int, log_dir: Path, ckpt_path: Path) -> None:
    data = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))["data"]
    dec = net_g.dec
    hop = data["hop_length"]
    frame_rate = sr / dec.upp

    ref_dir = Path(args.ref_dir)
    wave = torch.from_numpy(load_audio(str(ref_dir / "ref_audio.wav"), sr)).float()
    f0 = torch.from_numpy(np.load(ref_dir / "ref_f0f.npy")).float().flatten()
    spec = spectrogram_torch(
        wave.view(1, -1), data["filter_length"], hop, data["win_length"], center=False
    )
    frames = min(spec.shape[-1], f0.shape[0], wave.shape[0] // hop)
    spec, f0 = spec[..., :frames], f0[:frames]
    target = wave[: frames * hop]

    with torch.no_grad():
        g = net_g.emb_g(torch.tensor([args.sid])).unsqueeze(-1)
        _z, m_q, logs_q, mask = net_g.enc_q(spec, torch.tensor([frames]), g=g)
        z = m_q
        if args.posterior_noise:
            torch.manual_seed(1234)
            z = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * args.posterior_noise
        z = z * mask

    fmax = data["mel_fmax"] or sr / 2
    centers = librosa.mel_frequencies(
        n_mels=data["n_mel_channels"] + 2, fmin=data["mel_fmin"], fmax=fmax
    )[1:-1]
    band = (centers >= 250.0) & (centers <= 4000.0)
    voiced = f0.numpy() > 0
    pairs = voiced[1:] & voiced[:-1]
    original_db = log_mel_db(target, data)

    print(
        f"checkpoint: {ckpt_path.name}   sr={sr}   frames={frames}   "
        f"voiced={int(voiced.sum())}   z={'m_q' if not args.posterior_noise else f'draw x{args.posterior_noise:g}'}"
    )
    print("path: reference -> enc_q -> dec (stage 1's), metrics over 250 Hz - 4 kHz\n")
    print(f"{'case':24s} {'gain jump':>10s} {'column jump':>12s} {'mel L1':>9s}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        sf.write(save_dir / "original.wav", target.numpy(), sr)

    original = dec._source_gain
    cases = [
        ("source_gain on", "on"),
        ("source_gain OFF", "off"),
        ("gain -> its mean", "mean"),
        (f"gain LP {args.cutoff:g}Hz", "lowpass"),
    ]
    for label, mode in cases:
        dec._source_gain = (
            original
            if mode == "on"
            else patched_source_gain(dec, mode, frame_rate, args.cutoff)
        )
        with torch.no_grad():
            out = dec(z, f0.view(1, -1), g)[0, 0][: target.shape[0]]
        render_db = log_mel_db(out, data)
        n = min(render_db.shape[1], original_db.shape[1], frames)
        diff = render_db[band, :n] - original_db[band, :n]
        # Mean error per frame across the band: a vertical column in the
        # Difference panel is a jump in this from one frame to the next.
        column = diff.mean(axis=0)
        mask_n = pairs[: n - 1]
        column_jump = float(np.sqrt((np.diff(column)[mask_n] ** 2).mean()))
        gain_db = gain_envelope_db(dec, z, g, mode, frame_rate, args.cutoff)[:n]
        gain_jump = float(np.sqrt((np.diff(gain_db)[mask_n] ** 2).mean()))
        mel_l1 = float(np.abs(diff[:, voiced[:n]]).mean())
        print(
            f"{label:24s} {gain_jump:7.2f} dB {column_jump:9.2f} dB {mel_l1:6.2f} dB"
        )
        if save_dir:
            name = label.replace(" ", "").replace("->", "to") + ".wav"
            sf.write(save_dir / name, out.double().numpy(), sr)

    dec._source_gain = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--f0", type=float, default=220.0)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--freeze-frame", type=int, default=60)
    ap.add_argument("--cutoff", type=float, default=20.0, help="gain lowpass, Hz")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument(
        "--posterior",
        action="store_true",
        help="render through enc_q from --ref-dir instead of the prior (stage 1)",
    )
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument(
        "--posterior-noise",
        type=float,
        default=0.0,
        help="with --posterior, decode m_q + exp(logs_q) * randn * this instead of m_q",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    net_g, sr = build(log_dir, ckpt_path, False, keep_posterior=args.posterior)
    dec = net_g.dec
    if not getattr(dec, "has_source_gain", False):
        raise SystemExit(f"{ckpt_path.name} has no source_gain to compare.")
    dec.m_source.noise_std = 0.0  # the dither is measured elsewhere, not here
    if args.posterior:
        run_posterior(args, net_g, sr, log_dir, ckpt_path)
        return
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
        z_prior_mean = net_g.flow(m_p * x_mask, x_mask, g=g, reverse=True) * x_mask

    original = dec._source_gain

    def render(noise_scale: float, seed: int, smooth: int = 1):
        with torch.no_grad():
            z_p = m_p
            if noise_scale:
                torch.manual_seed(seed)
                noise = smooth_noise(torch.randn_like(m_p), smooth)
                z_p = m_p + torch.exp(logs_p) * noise * noise_scale
            z = net_g.flow(z_p * x_mask, x_mask, g=g, reverse=True) * x_mask
            return dec(z, pitchf, g)[0, 0].double().numpy()

    start = 2 * sr  # past the padding transient at the encoder and the decoder
    cases = [
        ("source_gain on   ns=0.667", "on", 0.66666),
        ("source_gain on   ns=0.5  ", "on", 0.5),
        ("source_gain on   ns=0.4  ", "on", 0.4),
        ("source_gain on   ns=0.3  ", "on", 0.3),
        ("source_gain on   ns=0.2  ", "on", 0.2),
        ("source_gain on   ns=0    ", "on", 0.0),
        ("source_gain OFF  ns=0.667", "off", 0.66666),
        ("gain -> its mean ns=0.667", "mean", 0.66666),
        (f"gain LP {args.cutoff:g}Hz   ns=0.667", "lowpass", 0.66666),
        (f"gain LP {args.cutoff:g}Hz   ns=0.4  ", "lowpass", 0.4),
        # Trunk decodes the draw; the gain reads the prior mean through the flow.
        ("gain = prior mean ns=0.667", "prior_mean", 0.66666),
        ("gain = prior mean ns=0.5 ", "prior_mean", 0.5),
        ("gain = prior mean ns=0.4 ", "prior_mean", 0.4),
        # The prior draw correlated over N frames instead of white per frame.
        ("eps smooth 3fr   ns=0.667", "smooth3", 0.66666),
        ("eps smooth 5fr   ns=0.667", "smooth5", 0.66666),
        ("eps smooth 10fr  ns=0.667", "smooth10", 0.66666),
        ("eps smooth 20fr  ns=0.667", "smooth20", 0.66666),
    ]

    print(
        f"checkpoint: {ckpt_path.name}   sr={sr}   f0={args.f0:g} Hz   "
        f"frames={frames}   frame rate={frame_rate:g} Hz"
    )
    print(
        f"window: {BLOCKS} x {PERIOD} = {BLOCKS * PERIOD} samples "
        f"({BLOCKS * PERIOD / sr:.2f} s), coherent\n"
    )
    print(
        f"{'case':28s} {'skirt':>9s} {'rms':>8s} {'diversity':>11s} "
        f"{'ih floor':>10s} {'burst':>8s} {'burst>4k':>9s}"
    )

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for label, mode, noise_scale in cases:
        smooth = int(mode[len("smooth"):]) if mode.startswith("smooth") else 1
        dec._source_gain = (
            original
            if mode == "on" or smooth > 1
            else patched_source_gain(
                dec,
                mode,
                frame_rate,
                args.cutoff,
                fixed_mel=z_prior_mean if mode == "prior_mean" else None,
            )
        )
        a = render(noise_scale, 1234, smooth)
        skirt, rms = skirt_db(a, sr, args.f0, start)
        # The last second holds the decoder's end-of-signal padding transient.
        floor, burst, burst_high = burst_metrics(a[:-sr], sr, args.f0, start)
        if noise_scale:
            b = render(noise_scale, 4321, smooth)
            d = a - b
            diversity = 20 * math.log10(
                math.sqrt((d**2).mean()) / math.sqrt((a**2).mean())
            )
            diversity_text = f"{diversity:7.1f} dB"
        else:
            diversity_text = "--"
        print(
            f"{label:28s} {skirt:6.1f} dB {rms:8.4f} {diversity_text:>11s} "
            f"{floor:7.1f} dB {burst:5.1f} dB {burst_high:6.1f} dB"
        )
        if save_dir:
            name = label.replace(" ", "").replace("->", "to") + ".wav"
            sf.write(save_dir / name, a, sr)

    dec._source_gain = original


if __name__ == "__main__":
    main()
