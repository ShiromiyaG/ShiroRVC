"""Recompute a base's style dataset descriptors from the events stored in each
clip -- per clip, the normalization and each speaker's pooled values -- after
a descriptor's definition changed.  In place; no RMVPE or embedder.

    python tools/style_flow/redescribe.py --data logs/style_pretrain/style_data

Only for a base's dataset: a fine-tune set shares its base's normalization, so
extract it again against the new base instead.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.getcwd())

from rvc.lib.style_flow.data import (
    CLIP_DIR, clip_paths, compute_normalizer, load_clip, read_manifest, save_clip, speaker_descriptors, write_manifest,
)
from rvc.lib.style_flow.descriptors import descriptor_vector
from rvc.lib.terminal import success, track

#: Clips are written here first and moved over the originals, so an
#: interrupted run leaves every clip whole.
TMP_DIR = ".redescribe"


def _redo(path):
    clip = load_clip(path)
    tmp = os.path.join(os.path.dirname(path), TMP_DIR, os.path.basename(path))
    save_clip(
        tmp, f0=clip["f0"], coarse=clip["coarse"], residual=clip["residual"], vuv=clip["vuv"], units=clip["units"],
        descriptors=descriptor_vector(clip["events"]), events=clip["events"], speaker=clip["speaker"],
        loudness=clip.get("loudness"),
    )
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Style dataset to update in place.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    manifest = read_manifest(args.data)
    paths = clip_paths(args.data)
    tmp_dir = os.path.join(args.data, CLIP_DIR, TMP_DIR)
    os.makedirs(tmp_dir, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for _ in track(pool.map(_redo, paths, chunksize=64), total=len(paths), description="Recomputing descriptors"):
            pass
    os.rmdir(tmp_dir)

    manifest.pop("descriptor_names", None)
    manifest.update(
        normalizer=compute_normalizer(paths).to_dict(),
        speaker_descriptors={str(k): v for k, v in speaker_descriptors(paths).items()},
    )
    write_manifest(args.data, **manifest)
    success(f"{len(paths)} clips re-described in {args.data}.", tag="[STYLE]")


if __name__ == "__main__":
    main()
