"""How far each content embedder drifts through a held note.

Embeds one clip with each embedder and prints, per slice, the cosine of the
features to the note's onset (``--anchor``) and the drift relative to how far
the other syllables (``--others``) are from it: 0 is no drift, 1 is as far as
another syllable. Raw cosines do not compare across embedders, the ratio does.

Usage::

    python archive/probes/embedder_drift.py --audio logs/reference/ref_audio.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rvc.lib.audio_io import load_audio_16k  # noqa: E402
from rvc.lib.utils import extract_features, load_embedder_model  # noqa: E402

FPS = 50  # 16 kHz / hop 320


def embed(name, audio, device):
    model, do_normalize = load_embedder_model(name)
    model = model.to(device).float().eval()
    with torch.inference_mode():
        source = torch.from_numpy(audio).to(device).float().view(1, -1)
        return extract_features(model, source, "v2", do_normalize=do_normalize)[0].float()


def frames(span):
    return slice(int(span[0] * FPS), int(span[1] * FPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="logs/reference/ref_audio.wav")
    parser.add_argument("--embedders", nargs="+", default=["contentvec", "spin_v2"])
    parser.add_argument("--anchor", type=float, nargs=2, default=(1.6, 1.8), help="Seconds: the note's onset.")
    parser.add_argument("--others", type=float, nargs=2, default=(0.6, 1.3), help="Seconds: other syllables.")
    parser.add_argument("--note", type=float, nargs=2, default=(1.5, 6.9), help="Seconds: the held note.")
    parser.add_argument("--slice", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = load_audio_16k(str(ROOT / args.audio) if not Path(args.audio).is_absolute() else args.audio)
    print(f"{args.audio}: {len(audio) / 16000:.2f} s on {device}")

    starts = np.arange(args.note[0], args.note[1], args.slice)
    header = "".join(f"{s:7.1f}" for s in starts)
    for name in args.embedders:
        feats = embed(name, audio, device)
        anchor = F.normalize(feats[frames(args.anchor)].mean(0), dim=0)
        cosine = F.normalize(feats, dim=1) @ anchor
        scale = (1 - cosine[frames(args.others)]).mean().clamp_min(1e-6)
        slices = [(s, min(s + args.slice, args.note[1])) for s in starts]
        cos_row = "".join(f"{cosine[frames(s)].mean().item():7.3f}" for s in slices)
        rel_row = "".join(f"{((1 - cosine[frames(s)]).mean() / scale).item():7.2f}" for s in slices)
        print(f"\n{name} ({feats.shape[0]} frames), other syllables at cosine {1 - scale.item():.3f}")
        print(f"  {'start (s)':18}{header}")
        print(f"  {'cosine to onset':18}{cos_row}")
        print(f"  {'drift / syllable':18}{rel_row}")


if __name__ == "__main__":
    main()
