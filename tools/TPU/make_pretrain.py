"""Turn a TPU run's G/D pair into a pretrain for the CUDA trainer.

``convert_to_cuda.py`` keeps a checkpoint resumable; this makes it a starting
point instead.  The pair is first loaded strictly into the CUDA models, then
written the way ``tools/clean_pretrain.py`` writes a pretrain: ``model`` in fp32
plus the metadata the pretrain guards read, without optimizer state or
counters.  The TPU trainer already saves the EMA weights as G's ``model``.

Outputs are ``<prefix>_G.pth`` / ``<prefix>_D.pth``, with no step in the name
(a ``G_<step>.pth`` name would seed a finetune's step counter).  ``--install``
writes them where the app looks for the vocoder's default pretrain instead.

    python tools/TPU/make_pretrain.py --model-name pretrain
    python tools/TPU/make_pretrain.py --model-name pretrain --install --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "rvc", "train"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-name", required=True, help="Folder under logs/.")
    parser.add_argument("--vocoder", default="", help="hifi, hifi++ or refinegan2; defaults to model_info.json's vocoder_architecture.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; defaults to the latest G_/D_ pair.")
    parser.add_argument("--output-dir", default=None, help="Defaults to the model folder.")
    parser.add_argument("--prefix", default=None, help="Output base name; defaults to the model name.")
    parser.add_argument("--install", action="store_true", help="Write to rvc/models/pretraineds/<vocoder>/f0G<rate>k.pth / f0D<rate>k.pth.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs (installed ones are kept as .bak).")
    return parser.parse_args(argv)


def pretrain_payload(blob, keep_metadata):
    model = {
        key: value.float() if value.is_floating_point() else value
        for key, value in blob["model"].items()
    }
    payload = {"model": model}
    payload.update({key: blob[key] for key in keep_metadata if blob.get(key) is not None})
    return payload


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    from convert_to_cuda import build_models, fill_metadata, find_pair, load_state_strict, strip_prefixes, to_cpu
    from rvc.configs.vocoders import get_vocoder_spec
    from rvc.train.utils import (
        assert_decoder_layout_matches,
        assert_excitation_matches,
        assert_msd_matches,
        assert_periods_match,
        load_config_from_json,
    )
    from tools.clean_pretrain import KEEP_METADATA
    from train_tpu import resolve_vocoder

    directory = os.path.join(REPO_ROOT, "logs", args.model_name)
    config = load_config_from_json(os.path.join(directory, "config.json"))
    sample_rate = int(config.data.sample_rate)
    vocoder = resolve_vocoder(os.path.join(directory, "model_info.json"), args.vocoder, sample_rate)

    step = find_pair(directory, args.step)
    blobs = {}
    for role in ("G", "D"):
        blob = to_cpu(torch.load(os.path.join(directory, f"{role}_{step}.pth"), map_location="cpu", weights_only=True))
        blob["model"] = strip_prefixes(blob["model"])
        blobs[role] = blob

    net_g, net_d = build_models(config, blobs["G"]["model"]["emb_g.weight"].shape[0], vocoder)
    load_state_strict(net_g, blobs["G"]["model"], f"G_{step}.pth")
    load_state_strict(net_d, blobs["D"]["model"], f"D_{step}.pth")
    fill_metadata(blobs["G"], net_g)
    fill_metadata(blobs["D"], net_d)
    assert_excitation_matches(net_g, blobs["G"])
    assert_decoder_layout_matches(net_g, blobs["G"])
    assert_periods_match(net_d, blobs["D"])
    assert_msd_matches(net_d, blobs["D"])

    if args.install:
        out_dir = os.path.join("rvc", "models", "pretraineds", get_vocoder_spec(vocoder).get("pretrained_dir", vocoder))
        # The names ``rvc/lib/tools/pretrained_selector.py`` looks up.
        names = {role: f"f0{role}{str(sample_rate)[:2]}k.pth" for role in ("G", "D")}
    else:
        out_dir = args.output_dir or directory
        prefix = args.prefix or args.model_name
        names = {role: f"{prefix}_{role}.pth" for role in ("G", "D")}
    os.makedirs(out_dir, exist_ok=True)

    targets = {role: os.path.join(out_dir, name) for role, name in names.items()}
    existing = [path for path in targets.values() if os.path.exists(path)]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing} already exist; pass --overwrite to replace them.")

    for role in ("G", "D"):
        target = targets[role]
        if args.install and os.path.exists(target):
            shutil.copy2(target, target + ".bak")
        torch.save(pretrain_payload(blobs[role], KEEP_METADATA), target)
        size = os.path.getsize(target) / 1e6
        print(f"{role}_{step}.pth -> {target} ({size:,.0f} MB, fp32, no optimizer state)")

    print(
        f"OK: {vocoder} pretrain at {sample_rate} Hz from step {step}"
        + (" installed as the app's default." if args.install else ". Pass it as pretrainG / pretrainD.")
    )


if __name__ == "__main__":
    main()
