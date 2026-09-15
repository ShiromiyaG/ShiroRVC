"""Store the latent directions that turn the prior draw into bursts.

Training writes ``prior_noise_subspace`` into every RefineGAN2 checkpoint it
saves; this adds it to one written before that, or re-estimates it.  See
``rvc/train/prior_subspace.py``.

Usage::

    python tools/estimate_prior_subspace.py logs/pretrain-contentvec/G_262218.pth
    python tools/estimate_prior_subspace.py logs/weights/model.pth --model pretrain-contentvec
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)
_train_dir = os.path.join(now_dir, "rvc", "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from rvc.train.prior_subspace import KEY, estimate_prior_subspace, pick_clips  # noqa: E402
from rvc.train.setup import get_g_model  # noqa: E402
from rvc.train.utils import load_config_from_json  # noqa: E402
from tools._staged_pretrain import model_dir, resolve_vocoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="G_*.pth or an exported model .pth")
    parser.add_argument(
        "--model",
        default=None,
        help="run folder or name whose config and dataset to use "
        "(default: the checkpoint's folder)",
    )
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--rank", type=int, default=16, help="directions to store")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: overwrite the checkpoint")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"No such checkpoint: {args.checkpoint}")
    directory = model_dir(args.model) if args.model else args.checkpoint.parent
    if not (directory / "config.json").is_file() or not (directory / "filelist.txt").is_file():
        sys.exit(f"'{directory}' needs config.json and filelist.txt; pass --model.")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = blob["model"] if "model" in blob else blob["weight"]
    config = load_config_from_json(str(directory / "config.json"))
    config.model.spk_embed_dim = int(weights["emb_g.weight"].shape[0])
    vocoder = blob.get("vocoder_id") or resolve_vocoder(directory, None)
    sr, hop = int(config.data.sample_rate), int(config.data.hop_length)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_g = get_g_model(config, sr, vocoder, False)
    missing, _ = net_g.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.startswith("enc_q.")]
    if missing:
        sys.exit(f"The checkpoint does not fit this run's model; missing {missing[:5]}")
    net_g = net_g.to(device).float().eval()
    for parameter in net_g.parameters():
        parameter.requires_grad_(False)

    clips = pick_clips(str(directory / "filelist.txt"), args.clips, args.seed)
    if not clips:
        sys.exit("No clip in the filelist is long and voiced enough.")
    print(f"{args.checkpoint.name}: {vocoder}, {len(clips)} clips from {directory}")

    basis, captured = estimate_prior_subspace(net_g, clips, sr, hop, rank=args.rank)
    if basis is None:
        sys.exit("No clip had voiced frames to measure.")
    blob[KEY] = basis

    out = args.out or args.checkpoint
    tmp = out.with_name(out.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, out)
    print(f"Wrote {KEY} ({basis.shape[0]}x{basis.shape[1]}, {captured:.0%} of the valley sensitivity) to {out}")


if __name__ == "__main__":
    main()
