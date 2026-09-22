"""Build a style dataset from an RVC experiment or a folder of audio.

    python tools/style_flow/extract.py --experiment logs/style_pretrain \\
        --config rvc/configs/style_flow/pretrain.yaml --out logs/style_pretrain/style_data

    python tools/style_flow/extract.py --audio-dir path/to/singer \\
        --reference style_base.pt --out logs/singer/style_data

F0 is always RMVPE and features always come from the base's embedder
(``--embedder``, or the reference's).  An experiment's own ``f0_voiced/``
and ``extracted/`` are reused when they were made that way, and recomputed
otherwise.  ``--reference`` (a style dataset or a style checkpoint) fixes the
embedder, codebook and normalization statistics, which a fine-tune set must
share with its base.  Interrupted runs resume where they stopped.
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

from rvc.lib.style_flow.config import default_config_path, load_config
from rvc.lib.style_flow.extraction import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="RVC experiment directory (logs/<name>).")
    source.add_argument("--audio-dir", help="Folder of audio files, searched recursively.")
    parser.add_argument("--speaker", type=int, help="One speaker id for all of --audio-dir; by default <id>_<name> subfolders are speakers.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--config", default=default_config_path("base.yaml"))
    parser.add_argument("--reference", help="Style dataset or checkpoint whose embedder, codebook and statistics to use.")
    parser.add_argument("--embedder", default="contentvec", help="Embedder of a new base (ignored with --reference).")
    parser.add_argument("--recompute", action="store_true", help="Ignore the experiment's F0 and features.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4, help="Audio loading threads.")
    args = parser.parse_args()

    try:
        build_dataset(
            args.out, load_config(args.config), experiment=args.experiment, audio_dir=args.audio_dir,
            speaker=args.speaker, reference=args.reference, embedder=args.embedder,
            recompute=args.recompute, device=args.device, workers=args.workers,
        )
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
