"""Does the flow darken the high band because the pitch stands still?

Renders the preview clip (``logs/reference`` as speaker 0, as the trainer does)
with the export's EMA weights three ways, from the same noise and with content,
loudness and breathiness untouched:

- ``original``: the clip's own f0.
- ``A vibrato``: sine vibrato added to the f0 over ``--vibrato-span``.
- ``B flattened``: the f0's vibrato smoothed away over ``--flatten-span``.

Prints each variant's flow-mel error against the real mel in dB, per band, over
each half second, and writes the vocoded audio and a figure to ``--out``.

Usage::

    python tools/probes/pitch_motion_ab.py --model pretrain
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rvc.rectified.common import RectifiedDataset, read_filelist, run_dir  # noqa: E402
from rvc.rectified.flow_model import build_flow  # noqa: E402
from rvc.rectified.mel import denormalize_mel, normalize_mel  # noqa: E402
from rvc.rectified.vocoder import load_vocoder  # noqa: E402

BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))
DB = 20.0 / math.log(10.0)  # the mel is a natural log of amplitude


def latest_export(flow_dir: str) -> str:
    exports = glob.glob(os.path.join(flow_dir, "*_flow_*e_*s.pth"))
    if not exports:
        raise FileNotFoundError(f"No flow export in {flow_dir}; pass --export.")
    return max(exports, key=lambda p: int(p.rsplit("_", 1)[1][:-5]))


def span_weight(frames, fps, span, fade, device):
    """[frames] 1 inside ``span`` seconds, 0 outside, with ``fade``-second ramps."""
    t = torch.arange(frames, device=device) / fps
    start, end = span
    rise = ((t - start) / fade + 0.5).clamp(0, 1)
    fall = ((end - t) / fade + 0.5).clamp(0, 1)
    return rise * fall


def with_vibrato(f0, fps, span, cents, rate, fade):
    t = torch.arange(f0.shape[-1], device=f0.device) / fps
    weight = span_weight(f0.shape[-1], fps, span, fade, f0.device)
    shift = cents * weight * torch.sin(2 * math.pi * rate * (t - span[0]))
    return torch.where(f0 > 0, f0 * 2.0 ** (shift / 1200.0), f0)


def flattened(f0, fps, span, window, fade):
    """Log f0 moving-averaged over ``window`` seconds (voiced frames only),
    blended in over ``span``."""
    voiced = (f0 > 0).float()
    log_f0 = torch.log(f0.clamp_min(1.0)) * voiced
    width = max(1, int(round(window * fps))) | 1
    kernel = torch.ones(1, 1, width, device=f0.device)
    smooth = lambda x: F.conv1d(x[:, None], kernel, padding=width // 2)[:, 0]
    mean = smooth(log_f0) / smooth(voiced).clamp_min(1.0)
    weight = span_weight(f0.shape[-1], fps, span, fade, f0.device)
    blended = torch.exp(weight * mean + (1 - weight) * torch.log(f0.clamp_min(1.0)))
    return torch.where(f0 > 0, blended, f0)


def cents_spread(f0, fps, span):
    a, b = int(span[0] * fps), int(span[1] * fps)
    segment = f0[0, a:b]
    segment = segment[segment > 0]
    if segment.numel() < 2:
        return float("nan")
    return (1200 * torch.log2(segment / segment.median())).std().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pretrain")
    parser.add_argument("--export", default="", help="Flow export; the latest in the run by default.")
    parser.add_argument("--vocoder", default="", help="The export's vocoder by default.")
    parser.add_argument("--steps", type=int, default=16, help="Euler steps, as the preview.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vibrato-span", type=float, nargs=2, default=(2.0, 5.2))
    parser.add_argument("--vibrato-cents", type=float, default=40.0, help="Half the peak-to-peak depth.")
    parser.add_argument("--vibrato-rate", type=float, default=5.5, help="Hz.")
    parser.add_argument("--flatten-span", type=float, nargs=2, default=(5.3, 6.7))
    parser.add_argument("--flatten-window", type=float, default=0.4, help="Seconds; longer than a vibrato cycle.")
    parser.add_argument("--fade", type=float, default=0.2, help="Seconds of ramp at each span edge.")
    parser.add_argument("--slice", type=float, default=0.5, help="Seconds per column of the table.")
    parser.add_argument("--out", default="", help="probe_pitch_motion next to the export by default.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_dir = run_dir(args.model, "flow")
    export = args.export or latest_export(flow_dir)
    checkpoint = torch.load(export, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data = config["data"]
    model = build_flow(config, checkpoint["speaker_count"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    print(f"{os.path.basename(export)}, {args.steps} Euler steps, seed {args.seed}")

    dataset = RectifiedDataset(
        read_filelist(args.model), config, "flow", int(config["flow"]["segment_frames"]), augment=False
    )
    reference = dataset.reference()
    if reference is None:
        raise SystemExit("No preview clip.")
    ref_mel, content, f0, energy, breathiness, audio, sid, source = reference
    print(f"Clip: {source}, speaker {sid}")
    content, f0, energy, breathiness = (x.to(device) for x in (content, f0, energy, breathiness))
    ref_mel = ref_mel.to(device)
    frames = f0.shape[1]
    fps = int(data["sample_rate"]) / int(data["hop_length"])

    variants = {
        "original": f0,
        "A vibrato": with_vibrato(f0, fps, args.vibrato_span, args.vibrato_cents, args.vibrato_rate, args.fade),
        "B flattened": flattened(f0, fps, args.flatten_span, args.flatten_window, args.fade),
    }
    print("\nPitch spread (std, cents)      vibrato span  flatten span")
    for name, pitch in variants.items():
        print(
            f"  {name:28}{cents_spread(pitch, fps, args.vibrato_span):10.1f}"
            f"{cents_spread(pitch, fps, args.flatten_span):14.1f}"
        )

    mask = torch.ones(1, 1, frames, device=device)
    speaker = torch.tensor([sid], device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(1, model.n_mels, frames, device=device, generator=generator)
    mels = {}
    with torch.no_grad():
        for name, pitch in variants.items():
            generated = model.sample(
                content, pitch, energy, speaker, mask, steps=args.steps, noise=noise, breathiness=breathiness
            )
            mels[name] = denormalize_mel(generated, data)

    freqs = torch.from_numpy(
        librosa.mel_frequencies(data["n_mels"] + 2, fmin=data["mel_fmin"], fmax=data["mel_fmax"])[1:-1]
    )
    per_slice = max(1, int(round(args.slice * fps)))
    edges = list(range(0, frames, per_slice))
    header = "".join(f"{e / fps:7.1f}" for e in edges)
    print(f"\nFlow mel minus real mel, dB, per {args.slice:g} s (column = start time)")
    for lo, hi in BANDS:
        rows = ((freqs >= lo) & (freqs < hi)).to(device)
        if not rows.any():
            continue
        print(f"\n{lo / 1000:g}-{hi / 1000:g} kHz{'':15}{header}")
        for name, mel in mels.items():
            error = ((mel - ref_mel)[0][rows].mean(0) * DB).cpu()
            cells = "".join(f"{error[e:e + per_slice].mean().item():+7.1f}" for e in edges)
            print(f"  {name:26}{cells}")

    out = args.out or os.path.join(flow_dir, "probe_pitch_motion")
    os.makedirs(out, exist_ok=True)
    figure, axes = plt.subplots(len(mels) + 1, 1, figsize=(14, 3.2 * (len(mels) + 1)), sharex=True)
    extent = (0, frames / fps, 0, data["n_mels"])
    image = axes[0].imshow(ref_mel[0].cpu().numpy(), origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0].set_title("Real mel")
    figure.colorbar(image, ax=axes[0], pad=0.01)
    for axis, (name, mel) in zip(axes[1:], mels.items()):
        image = axis.imshow(
            ((mel - ref_mel)[0] * DB).cpu().numpy(), origin="lower", aspect="auto", extent=extent,
            cmap="coolwarm", vmin=-20, vmax=20,
        )
        axis.set_title(f"{name}: flow minus real (dB)")
        figure.colorbar(image, ax=axis, pad=0.01)
    for axis in axes:
        ticks = [0, 1000, 2000, 4000, 8000]
        axis.set_yticks([int(np.argmin(np.abs(freqs.numpy() - f))) for f in ticks], [f"{f // 1000}k" for f in ticks])
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(os.path.join(out, "difference.png"), dpi=110)
    plt.close(figure)

    vocoder_path = args.vocoder or checkpoint.get("vocoder", "")
    if not vocoder_path or not os.path.exists(vocoder_path):
        print(f"\nNo vocoder; figure only, in {out}")
        return
    vocoder, _ = load_vocoder(vocoder_path, data)
    vocoder = vocoder.to(device)
    sr = int(data["sample_rate"])
    sf.write(os.path.join(out, "real.wav"), audio.squeeze().numpy(), sr)
    with torch.no_grad():
        for name, mel in mels.items():
            wave = vocoder(normalize_mel(mel, data), variants[name]).float().squeeze().cpu().numpy()
            sf.write(os.path.join(out, name.replace(" ", "_") + ".wav"), wave, sr)
    print(f"\nAudio and figure in {out}")


if __name__ == "__main__":
    main()
