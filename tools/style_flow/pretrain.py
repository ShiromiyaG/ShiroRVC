"""Pretrain a style base.

    python tools/style_flow/pretrain.py --experiment logs/style_pretrain
    python tools/style_flow/pretrain.py --data logs/style_pretrain/style_data \\
        --out logs/style_base_contentvec

With ``--experiment`` (a preprocessed and extracted RVC experiment) the style
dataset is extracted into ``<experiment>/style_data`` when missing, with the
experiment's embedder, and the base is written to
``<experiment>/style_base.pt``.  The base is tied to that embedder: train one
per embedder.  ``--overfit 8`` trains on 8 clips without augmentation or
dropout, as a sanity check.  Rerunning resumes.
"""

import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

# Every batch has its own length, which fragments the allocator's cache; see
# rvc/train/train.py.  Linux-only, and set before torch touches CUDA.
if sys.platform.startswith("linux") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rvc.lib.style_flow.config import default_config_path, load_config
from rvc.lib.style_flow.data import MANIFEST
from rvc.lib.style_flow.extraction import build_dataset
from rvc.lib.style_flow.trainer import train


def experiment_embedder(exp_dir: str) -> str:
    path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("embedder_model", "contentvec")
    return "contentvec"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="RVC experiment directory (logs/<name>).")
    source.add_argument("--data", help="Style dataset directory.")
    parser.add_argument("--out", help="Run directory; required with --data.")
    parser.add_argument("--config", default=default_config_path("pretrain.yaml"))
    parser.add_argument("--embedder", help="Embedder for a new dataset; defaults to the experiment's.")
    parser.add_argument("--speech-speakers", help='Override data.speech_speakers, e.g. "0-4"; "" for none.')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, help="Override train.steps.")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size.")
    parser.add_argument("--val-every", type=int, help="Override train.val_every.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], help="Override train.precision.")
    parser.add_argument("--no-tf32", action="store_true", help="Keep FP32 matmuls in full FP32.")
    parser.add_argument("--overfit", type=int, default=0, help="Train and validate on this many clips.")
    parser.add_argument("--checkpointing", action="store_true", help="Recompute block activations in backward to save VRAM.")
    parser.add_argument("--recompute", action="store_true", help="Ignore the experiment's F0 and features.")
    parser.add_argument("--workers", type=int, default=4, help="Audio loading threads for extraction.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.precision:
        cfg["train"]["precision"] = args.precision
    if args.no_tf32:
        cfg["train"]["tf32"] = False
    if args.checkpointing:
        cfg["train"]["checkpointing"] = True
    for key, value in (("steps", args.steps), ("batch_size", args.batch_size), ("val_every", args.val_every)):
        if value is not None:
            cfg["train"][key] = value
    if args.speech_speakers is not None:
        cfg["data"]["speech_speakers"] = [s.strip() for s in args.speech_speakers.split(",") if s.strip()]
    if args.overfit:
        cfg["validation"]["sample_clips"] = min(cfg["validation"]["sample_clips"], args.overfit)

    if args.experiment:
        data_dir = os.path.join(args.experiment, "style_data")
        out_dir = args.out or os.path.join(args.experiment, "style_pretrain")
        export = os.path.abspath(os.path.join(args.experiment, "style_base.pt"))
        if not os.path.exists(os.path.join(data_dir, MANIFEST)):
            build_dataset(
                data_dir, cfg, experiment=args.experiment,
                embedder=args.embedder or experiment_embedder(args.experiment),
                recompute=args.recompute, device=args.device, workers=args.workers,
            )
    else:
        if not args.out:
            parser.error("--out is required with --data.")
        data_dir, out_dir, export = args.data, args.out, "style_base.pt"
    train(cfg, data_dir, out_dir, device=args.device, kind="base", export_name=export, overfit=args.overfit)


if __name__ == "__main__":
    main()
