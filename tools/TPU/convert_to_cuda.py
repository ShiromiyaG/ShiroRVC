"""Make a TPU run's checkpoints safe to resume or use on CUDA.

The TPU trainer builds the same modules as the CUDA one, so the state-dict keys
already match; this script checks that instead of assuming it, and repairs what
can differ: tensors left on another device or dtype, ``module.``/``_orig_mod.``
prefixes, missing metadata keys, and the AdamW flags (``capturable`` on TPU,
``fused`` on CUDA).  Originals are kept as ``*.tpu.pth`` when rewritten in place.

    python tools/TPU/convert_to_cuda.py --model-name my-voice
    python tools/TPU/convert_to_cuda.py --model-name my-voice --export --subspace
    python tools/TPU/convert_to_cuda.py --model-name my-voice --check-export logs/my-voice/my-voice_200e_5000s.pth
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "rvc", "train"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

PREFIXES = ("module.", "_orig_mod.")
ADAMW_FLAGS = dict(
    amsgrad=False,
    maximize=False,
    foreach=None,
    capturable=False,
    differentiable=False,
    decoupled_weight_decay=True,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-name", required=True, help="Folder under logs/.")
    parser.add_argument("--vocoder", default="", help="hifi, hifi++ or refinegan2; defaults to model_info.json's vocoder_architecture.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; defaults to the latest G_/D_ pair.")
    parser.add_argument("--output-dir", default=None, help="Where to write; defaults to the model folder (in place, with backups).")
    parser.add_argument("--no-fused", action="store_true", help="Leave AdamW unfused, for resuming on CPU.")
    parser.add_argument("--export", action="store_true", help="Also write the inference .pth from G.")
    parser.add_argument("--subspace", action="store_true", help="Estimate prior_noise_subspace for the export (slow on CPU).")
    parser.add_argument("--check-export", default=None, help="Only check an exported inference .pth against the config.")
    return parser.parse_args(argv)


def strip_prefixes(state):
    cleaned = {}
    for key, value in state.items():
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def to_cpu(value):
    if torch.is_tensor(value):
        value = value.detach().to("cpu")
        if value.is_floating_point() and value.dtype != torch.float32:
            value = value.float()
        return value.contiguous()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def normalize_optimizer(state, fused):
    if not state:
        return state
    for entry in state.get("state", {}).values():
        if "step" in entry and torch.is_tensor(entry["step"]):
            entry["step"] = entry["step"].reshape(()).float()
    for group in state.get("param_groups", []):
        group.update(ADAMW_FLAGS)
        group["fused"] = True if fused else None
    return state


def load_state_strict(model, state, label):
    result = model.load_state_dict(state, strict=False)
    problems = []
    if result.missing_keys:
        problems.append(f"missing {result.missing_keys[:8]}")
    if result.unexpected_keys:
        problems.append(f"unexpected {result.unexpected_keys[:8]}")
    if problems:
        raise SystemExit(f"{label} does not match the CUDA model: " + "; ".join(problems))
    # A strict load also checks shapes.
    model.load_state_dict(state, strict=True)


def find_pair(directory, step):
    pattern = re.compile(r"^G_(\d+)\.pth$")
    steps = sorted(
        int(m.group(1))
        for m in map(pattern.match, os.listdir(directory))
        if m and os.path.isfile(os.path.join(directory, f"D_{m.group(1)}.pth"))
    )
    if not steps:
        raise SystemExit(f"No G_/D_ pair in {directory}.")
    if step is None:
        step = steps[-1]
    elif step not in steps:
        raise SystemExit(f"No G_{step}.pth / D_{step}.pth pair in {directory}; found {steps}.")
    return step


def build_models(config, speakers, vocoder):
    from rvc.train.setup import get_d_model, get_g_model

    config.model.spk_embed_dim = int(speakers)
    return get_g_model(config, int(config.data.sample_rate), vocoder, False), get_d_model(config, vocoder, False)


def fill_metadata(blob, model):
    from rvc.train.utils import (
        decoder_layout,
        discriminator_has_msd,
        discriminator_periods,
        excitation_source,
        optimizer_param_names,
    )

    for key, value in (
        ("architecture_id", getattr(model, "architecture_id", None)),
        ("excitation_source", excitation_source(model)),
        ("discriminator_periods", discriminator_periods(model)),
        ("discriminator_msd", discriminator_has_msd(model)),
        ("decoder_layout", decoder_layout(model)),
    ):
        if value is not None and blob.get(key) is None:
            blob[key] = value

    names = [name for name, _ in model.named_parameters()]
    saved = blob.get("optimizer_param_names")
    optimizer = blob.get("optimizer") or {}
    count = sum(len(group["params"]) for group in optimizer.get("param_groups", []))
    if saved is None and count == len(names):
        blob["optimizer_param_names"] = names
    elif saved is not None and any(name not in names for name in saved if name):
        raise SystemExit("The optimizer state names parameters this model does not have.")


def check_export(path, config, vocoder):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    weights = blob.get("weight")
    if weights is None:
        raise SystemExit(f"{path} is not an inference export (no 'weight' key).")
    speakers = weights["emb_g.weight"].shape[0]
    vocoder = blob.get("vocoder_id") or vocoder
    net_g, _ = build_models(config, speakers, vocoder)
    net_g.remove_training_modules()
    result = net_g.load_state_dict(to_cpu(weights), strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("enc_q.")]
    if missing or result.unexpected_keys:
        raise SystemExit(f"Export mismatch: missing {missing[:8]}, unexpected {result.unexpected_keys[:8]}.")
    print(f"OK: {os.path.basename(path)} loads into the CUDA {vocoder} synthesizer ({speakers} speaker(s)).")


def estimate_subspace(net_g, config, vocoder):
    if vocoder != "refinegan2":
        print("prior_noise_subspace is RefineGAN v2 only; exporting without it.")
        return None
    from rvc.train.prior_subspace import estimate_prior_subspace, pick_clips

    clips = pick_clips(config.data.training_files)
    if not clips:
        print("No clip qualifies for prior_noise_subspace; exporting without it.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g.to(device)
    try:
        basis, captured = estimate_prior_subspace(
            net_g, clips, int(config.data.sample_rate), int(config.data.hop_length), max_frames=400
        )
    finally:
        net_g.to("cpu")
    if basis is not None:
        print(f"prior_noise_subspace: rank {basis.shape[1]}, {captured:.1%} of the valley gradient.")
    return basis


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    from rvc.train.utils import load_config_from_json, small_model_naming

    directory = os.path.join(REPO_ROOT, "logs", args.model_name)
    config = load_config_from_json(os.path.join(directory, "config.json"))
    config.data.training_files = os.path.join(directory, "filelist.txt")

    from train_tpu import resolve_vocoder

    vocoder = resolve_vocoder(
        os.path.join(directory, "model_info.json"), args.vocoder, config.data.sample_rate
    )

    if args.check_export:
        check_export(args.check_export, config, vocoder)
        return

    step = find_pair(directory, args.step)
    g_path = os.path.join(directory, f"G_{step}.pth")
    d_path = os.path.join(directory, f"D_{step}.pth")
    g_blob = to_cpu(torch.load(g_path, map_location="cpu", weights_only=True))
    d_blob = to_cpu(torch.load(d_path, map_location="cpu", weights_only=True))
    g_blob["model"] = strip_prefixes(g_blob["model"])
    d_blob["model"] = strip_prefixes(d_blob["model"])

    net_g, net_d = build_models(config, g_blob["model"]["emb_g.weight"].shape[0], vocoder)
    load_state_strict(net_g, g_blob["model"], os.path.basename(g_path))
    load_state_strict(net_d, d_blob["model"], os.path.basename(d_path))

    from rvc.train.utils import assert_decoder_layout_matches, assert_periods_match

    fill_metadata(g_blob, net_g)
    fill_metadata(d_blob, net_d)
    assert_decoder_layout_matches(net_g, g_blob)
    assert_periods_match(net_d, d_blob)
    for blob in (g_blob, d_blob):
        blob["optimizer"] = normalize_optimizer(blob.get("optimizer"), fused=not args.no_fused)

    output_dir = args.output_dir or directory
    os.makedirs(output_dir, exist_ok=True)
    for blob, source in ((g_blob, g_path), (d_blob, d_path)):
        target = os.path.join(output_dir, os.path.basename(source))
        if os.path.abspath(target) == os.path.abspath(source):
            backup = source[: -len(".pth")] + ".tpu.pth"
            if not os.path.exists(backup):
                shutil.copy2(source, backup)
        torch.save(blob, target)
        print(f"Wrote {target}")

    if args.export:
        from rvc.train.process.extract_model import extract_model

        subspace = estimate_subspace(net_g, config, vocoder) if args.subspace else None
        epoch = int(g_blob.get("iteration", 0))
        export_path = os.path.join(directory, small_model_naming(args.model_name, epoch, step))
        extract_model(
            ckpt=g_blob["model"],
            sr=int(config.data.sample_rate),
            name=args.model_name,
            model_path=export_path,
            epoch=epoch,
            step=step,
            hps=config,
            vocoder=vocoder,
            architecture="RVC",
            weights_source="TPU checkpoint",
            prior_noise_subspace=subspace,
        )
        check_export(export_path, config, vocoder)

    print(f"OK: G_{step}/D_{step} load strictly into the CUDA {vocoder} models.")


if __name__ == "__main__":
    main()
