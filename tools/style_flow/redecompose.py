"""Re-split a style dataset's F0 into coarse and residual with another
representation, reusing its F0, units and codebook (no RMVPE or embedder).

    python tools/style_flow/redecompose.py --data logs/style_pretrain/style_data \
        --out logs/style_pretrain/style_data_notes

Each clip is split on its own, so a note cut by a clip edge sees less context
than in a fresh extraction.  Only for a new base: a fine-tune set follows its
base's representation.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.getcwd())

from rvc.lib.style_flow.config import default_config_path, load_config
from rvc.lib.style_flow.data import (
    CLIP_DIR, CODEBOOK, clip_paths, compute_normalizer, load_clip, read_manifest, save_clip, speaker_descriptors,
    write_manifest,
)
from rvc.lib.style_flow.descriptors import DescriptorConfig, analyze, descriptor_vector
from rvc.lib.style_flow.extraction import DONE_FILE
from rvc.lib.style_flow.f0_repr import ReprConfig, decompose
from rvc.lib.terminal import success, track


def _redo(job):
    src, dst, rcfg, dcfg = job
    clip = load_clip(src)
    coarse, residual, vuv = decompose(clip["f0"], rcfg)
    events = analyze(clip["f0"], rcfg, dcfg, residual)
    save_clip(
        dst, f0=clip["f0"], coarse=coarse, residual=residual, vuv=vuv, units=clip["units"],
        descriptors=descriptor_vector(events), events=events, speaker=clip["speaker"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Style dataset to read.")
    parser.add_argument("--out", required=True, help="New style dataset; must not exist.")
    parser.add_argument("--config", default=default_config_path("pretrain.yaml"), help="Config with the new representation.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if os.path.exists(args.out):
        parser.error(f"{args.out} already exists.")

    manifest = read_manifest(args.data)
    rcfg = ReprConfig.from_dict(load_config(args.config).get("representation"))
    dcfg = DescriptorConfig.from_dict(manifest["descriptor_config"])
    os.makedirs(os.path.join(args.out, CLIP_DIR))
    for name in (manifest.get("codebook", CODEBOOK), DONE_FILE):
        if os.path.exists(os.path.join(args.data, name)):
            shutil.copy2(os.path.join(args.data, name), os.path.join(args.out, name))

    sources = clip_paths(args.data)
    jobs = [(p, os.path.join(args.out, CLIP_DIR, os.path.basename(p)), rcfg, dcfg) for p in sources]
    with ProcessPoolExecutor(args.workers) as pool:
        for _ in track(pool.map(_redo, jobs, chunksize=64), total=len(jobs), description="Re-splitting clips"):
            pass

    paths = clip_paths(args.out)
    manifest.pop("descriptor_names", None)
    manifest.update(
        representation=rcfg.to_dict(),
        normalizer=compute_normalizer(paths).to_dict(),
        speaker_descriptors={str(k): v for k, v in speaker_descriptors(paths).items()},
    )
    write_manifest(args.out, **manifest)
    success(f"{len(paths)} clips in {args.out}.", tag="[STYLE]")


if __name__ == "__main__":
    main()
