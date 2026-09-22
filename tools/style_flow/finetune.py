"""Fine-tune a style base on one singer.

    python tools/style_flow/finetune.py --base style_base.pt --experiment logs/<name>
    python tools/style_flow/finetune.py --base style_base.pt --audio-dir path/to/singer --name <name>

The singer's style dataset (``logs/<name>/style_data``) is extracted with the
base's embedder, codebook and statistics when it does not exist yet.  The
result, ``logs/<name>/<name>_style.pt``, carries the singer's pooled
descriptors.  Rerunning resumes.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="style_base.pt to start from.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="RVC experiment directory (logs/<name>).")
    source.add_argument("--audio-dir", help="Folder of the singer's audio.")
    parser.add_argument("--name", help="Model name; defaults to the experiment's folder name.")
    parser.add_argument("--config", default=default_config_path("finetune.yaml"))
    parser.add_argument("--steps", type=int, help="Override train.steps.")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], help="Override train.precision.")
    parser.add_argument("--no-tf32", action="store_true", help="Keep FP32 matmuls in full FP32.")
    parser.add_argument("--checkpointing", action="store_true", help="Recompute block activations in backward to save VRAM.")
    parser.add_argument("--recompute", action="store_true", help="Ignore the experiment's F0 and features.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4, help="Audio loading threads for extraction.")
    args = parser.parse_args()

    name = args.name or (os.path.basename(os.path.normpath(args.experiment)) if args.experiment else None)
    if not name:
        parser.error("--name is required with --audio-dir.")
    model_dir = args.experiment or os.path.join(os.getcwd(), "logs", name)
    data_dir = os.path.join(model_dir, "style_data")

    cfg = load_config(args.config)
    if args.precision:
        cfg["train"]["precision"] = args.precision
    if args.no_tf32:
        cfg["train"]["tf32"] = False
    if args.checkpointing:
        cfg["train"]["checkpointing"] = True
    for key, value in (("steps", args.steps), ("batch_size", args.batch_size)):
        if value is not None:
            cfg["train"][key] = value

    if not os.path.exists(os.path.join(data_dir, MANIFEST)):
        build_dataset(
            data_dir, cfg, experiment=args.experiment, audio_dir=args.audio_dir, reference=args.base,
            recompute=args.recompute, device=args.device, workers=args.workers,
        )
    train(
        cfg, data_dir, os.path.join(model_dir, "style_train"), device=args.device, init=args.base,
        kind="singer", export_name=os.path.abspath(os.path.join(model_dir, f"{name}_style.pt")),
    )


if __name__ == "__main__":
    main()
