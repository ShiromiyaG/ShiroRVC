"""Does the shallow flow's start from the aux decoder darken the high band?

For each clip, renders the mel four ways with the export's EMA weights: the
aux decoder alone, the flow from the aux mel (what inference does), the flow
from the real mel mixed with noise at ``t_start`` (the state it was trained
on), and the flow from the aux mel with four times the steps. Reports, per
band, the mean log-mel error against the real mel (negative is too dark) and
the L1. The first clip is also rendered through the vocoder to ``--out``.

Usage::

    python tools/probes/flow_start_ab.py --model pretrain
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.commons import upsample_content  # noqa: E402
from rvc.rectified.common import (  # noqa: E402
    RectifiedDataset,
    read_filelist,
    run_dir,
    speaker_count,
    split_holdout,
)
from rvc.rectified.flow_model import build_flow  # noqa: E402
from rvc.rectified.mel import normalize_mel  # noqa: E402
from rvc.rectified.vocoder import load_vocoder  # noqa: E402

BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))


def latest_export(flow_dir: str) -> str:
    exports = glob.glob(os.path.join(flow_dir, "*_flow_*e_*s.pth"))
    if not exports:
        raise FileNotFoundError(f"No flow export in {flow_dir}.")
    return max(exports, key=lambda p: int(p.rsplit("_", 1)[1][:-5]))


def clip(dataset, entry, max_seconds):
    wav_path, content_path, _, f0_path, sid = entry
    audio = dataset._audio(wav_path)
    f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
    content = upsample_content(
        torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
        dataset.data["content_interpolation"],
    )
    max_frames = int(max_seconds * dataset.sample_rate) // dataset.hop
    return dataset._reference_item(audio, content, f0, sid, wav_path, max_frames)


def integrate(model, x, cond, voice, mask, steps):
    """Euler from ``t_start`` to 1, as ``RectifiedFlow.sample`` without guidance."""
    dt = (1.0 - model.t_start) / steps
    for index in range(steps):
        t = torch.full((x.shape[0],), model.t_start + index * dt, device=x.device)
        x = x + dt * model.backbone(x, t, cond, mask, voice)
    return x * mask


@torch.no_grad()
def render(model, item, data, device, steps, seed):
    """Normalised mels {variant: [1, n_mels, T]} and the real one."""
    inputs = item.inputs._replace(tension=None).to(device)
    f0, mask = inputs.f0, inputs.mask
    cond = model.encoder(inputs)
    voice = model.encoder.voice(inputs.speaker)
    real = normalize_mel(item.mel.to(device), data)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(real.shape, device=device, generator=generator)
    aux = model.aux(cond, mask, voice)
    ts = model.t_start
    return real, {
        "aux only": aux,
        "flow from aux": integrate(model, (1 - ts) * noise + ts * aux, cond, voice, mask, steps),
        "flow from real (oracle)": integrate(model, (1 - ts) * noise + ts * real, cond, voice, mask, steps),
        f"flow from aux, {4 * steps} steps": integrate(
            model, (1 - ts) * noise + ts * aux, cond, voice, mask, 4 * steps
        ),
    }, f0


def band_rows(data):
    freqs = torch.from_numpy(
        librosa.mel_frequencies(data["n_mels"] + 2, fmin=data["mel_fmin"], fmax=data["mel_fmax"])[1:-1]
    )
    return [((freqs >= lo) & (freqs < hi)) for lo, hi in BANDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pretrain")
    parser.add_argument("--export", default="", help="Flow export; the latest in the run by default.")
    parser.add_argument("--clips", type=int, default=16, help="Held-out clips, after the preview clip.")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--out", default="", help="Where the first clip's audio goes; next to the export by default.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_dir = run_dir(args.model, "flow")
    export = args.export or latest_export(flow_dir)
    checkpoint = torch.load(export, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model_data = config["data"]
    model = build_flow(config, checkpoint["speaker_count"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    print(f"{os.path.basename(export)}, t_start {model.t_start:g}, {args.steps} Euler steps")

    entries = read_filelist(args.model)
    assert speaker_count(entries) == checkpoint["speaker_count"]
    segment = int(config["flow"]["segment_frames"])
    train, holdout = split_holdout(entries, int(config["flow"].get("holdout_clips", 0)))
    dataset = RectifiedDataset(train, config, "flow", segment, augment=False)
    items = [("preview (train)", dataset.reference(args.max_seconds))]
    items += [(f"holdout {i}", clip(dataset, e, args.max_seconds)) for i, e in enumerate(holdout[: args.clips])]

    bands = band_rows(model_data)
    scale = float(model_data["mel_std"])
    totals, first = {}, None
    for index, (label, item) in enumerate(items):
        real, variants, f0 = render(model, item, model_data, device, args.steps, seed=index)
        if first is None:
            first = (real, variants, f0)
        for name, mel in variants.items():
            error = ((mel - real) * scale)[0].float().cpu()
            row = [error[b].mean().item() for b in bands] + [error.abs().mean().item()]
            totals.setdefault(name, []).append(row)
            if index == 0:
                totals.setdefault(f"{name} [preview]", []).append(row)

    header = "".join(f"{f'{lo // 1000}-{hi // 1000}k':>8}" for lo, hi in BANDS) + f"{'L1':>8}"
    print(f"\nMean log-mel error vs real (negative = too dark), {len(items)} clips\n{'':38}{header}")
    for name, rows in totals.items():
        values = np.mean(rows, axis=0)
        print(f"{name:38}" + "".join(f"{v:+8.3f}" for v in values[:-1]) + f"{values[-1]:8.3f}")

    vocoder_path = checkpoint.get("vocoder", "")
    if not vocoder_path or not os.path.exists(vocoder_path):
        print("\nNo vocoder in the export; no audio.")
        return
    vocoder, _ = load_vocoder(vocoder_path, model_data)
    vocoder = vocoder.to(device)
    out = args.out or os.path.join(flow_dir, "probe_flow_start")
    os.makedirs(out, exist_ok=True)
    real, variants, f0 = first
    sr = int(model_data["sample_rate"])
    with torch.no_grad():
        for name, mel in {"real mel": real, **variants}.items():
            audio = vocoder(mel, f0.to(device)).float().squeeze().cpu().numpy()
            sf.write(os.path.join(out, name.replace(" ", "_").replace(",", "") + ".wav"), audio, sr)
    with open(os.path.join(out, "source.json"), "w", encoding="utf-8") as handle:
        json.dump({"export": export, "clip": items[0][1].path}, handle, indent=2)
    print(f"\nPreview clip audio in {out}")


if __name__ == "__main__":
    main()
