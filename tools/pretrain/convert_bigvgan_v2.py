"""Turn an official BigVGAN v2 checkpoint into a rectified vocoder export.

The released generators (``bigvgan_generator.pt`` of nvidia/bigvgan_v2_*,
MIT) take the mel alone. This repo's ``BigVGANv2`` also reads f0: its source
convs are written as zeros here, so the export renders exactly as the
original until ``adapt_pupu_f0.py --architecture bigvgan-v2`` teaches them.
The weight norm is folded and the filters, which are not weights, are left
out. The checkpoint's ``config.json`` is read from beside it.

    python tools/pretrain/convert_bigvgan_v2.py bigvgan_generator.pt \\
        rvc/models/vocoders/bigvgan_v2_44.1k_hop512_128bin_vocoder.pth

``--discriminator`` also writes the release's MPD and sub-band CQT
discriminator (``bigvgan_discriminator_optimizer.pt`` beside the generator)
as one file for ``pretrained_d``, which fits the ``bigvgan-v2`` recipe:

    python tools/pretrain/convert_bigvgan_v2.py bigvgan_generator.pt bigvgan_v2_vocoder.pth \\
        --discriminator rvc/models/pretraineds/custom/D_bigvgan_v2_44.1k.pth
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import error as print_error, info, install_rich_print, success
from rvc.rectified import bigvgan_v2
from rvc.rectified import singing_vocoders as sv

TAG = "[CONVERT]"
RECIPE = os.path.join(now_dir, "rvc", "configs", "rectified", "vocoders", "bigvgan-v2.json")
D_CHECKPOINT = "bigvgan_discriminator_optimizer.pt"
#: The release's settings this generator is: anything else is another network.
REQUIRED = {
    "resblock": "1", "activation": "snakebeta", "snake_logscale": True,
    "use_tanh_at_final": False, "use_bias_at_final": False,
}
#: Defaults of the release's code for keys older configs leave out.
RELEASE_DEFAULTS = {"use_tanh_at_final": True, "use_bias_at_final": True}
#: Recipe discriminator key -> the release's config key.
D_KEYS = {
    "periods": "mpd_reshapes", "cqt_hop_lengths": "cqtd_hop_lengths", "cqt_n_octaves": "cqtd_n_octaves",
    "cqt_bins_per_octave": "cqtd_bins_per_octaves", "cqt_filters": "cqtd_filters",
}
#: Buffers of the CQT heads the code computes from the settings, not weights.
FIXED_BUFFERS = (".cqt_transform.", ".resample.")


def check_release(h: dict) -> None:
    wrong = [key for key, value in REQUIRED.items() if h.get(key, RELEASE_DEFAULTS.get(key)) != value]
    if wrong:
        raise ValueError(f"Not a generator this repo has: {', '.join(f'{key}={h.get(key)!r}' for key in wrong)}.")


def folded(state: dict) -> dict:
    """``state`` with each ``weight_g``/``weight_v`` pair as one ``weight``,
    the upsamplers under this repo's names and without the filter buffers."""
    out = {}
    for key, value in state.items():
        if key.endswith(".filter") or key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            norm = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
            key, value = base, value * (state[base + "_g"] / norm)
        # The release keeps each upsampler in a list of its own.
        out[re.sub(r"^ups\.(\d+)\.0\.", r"ups.\1.", key)] = value.float().contiguous()
    return out


def convert(state: dict, h: dict):
    """The export of an official generator's ``state`` under config ``h`` and the lines of its report."""
    check_release(h)
    data = {
        "sample_rate": int(h["sampling_rate"]), "hop_length": int(h["hop_size"]), "n_fft": int(h["n_fft"]),
        "win_length": int(h["win_size"]), "n_mels": int(h["num_mels"]), "mel_fmin": float(h["fmin"]),
        "mel_fmax": float(h["fmax"] or h["sampling_rate"] / 2),
    }
    hparams = {
        "sample_rate": data["sample_rate"], "num_mels": data["n_mels"],
        "upsample_initial_channel": int(h["upsample_initial_channel"]),
        "upsample_rates": list(h["upsample_rates"]), "upsample_kernel_sizes": list(h["upsample_kernel_sizes"]),
        "resblock_kernel_sizes": list(h["resblock_kernel_sizes"]),
        "resblock_dilation_sizes": [list(d) for d in h["resblock_dilation_sizes"]], "harmonic_num": 0,
    }
    weights = folded(state)
    generator = bigvgan_v2.BigVGANv2(**hparams)
    # What the original lacks: the f0 source, silent until trained. Its sine
    # goes to the convs as it is.
    added = {}
    for key, value in generator.state_dict().items():
        if key not in weights:
            plain = key == "m_source.l_linear.weight"
            added[key] = torch.ones_like(value) if plain else torch.zeros_like(value)
    generator.load_state_dict({**weights, **added})
    size = sum(value.numel() for value in weights.values()) / 1e6
    lines = [
        f"generator: {hparams['upsample_initial_channel']} initial channels, x{generator.upp} at "
        f"{data['sample_rate']} Hz, {data['n_mels']} mels, {size:.1f}M weights",
        f"added:     {len(added)} f0 source tensors, the convs as zeros",
    ]
    export = {
        "kind": "rectified_vocoder", "architecture": bigvgan_v2.ARCHITECTURE, "recipe": sv.BIGVGAN_V2,
        "config": {"data": data, "vocoder": {"model": hparams}},
        "model": {key: value.detach().clone() for key, value in generator.state_dict().items()},
    }
    return export, lines


def convert_discriminators(checkpoint: dict, h: dict, sample_rate: int):
    """The release's MPD and CQTD (saved as ``mrd``) as one state for the
    recipe's discriminator, and the lines of its report. The names of the
    older weight norm stay: ``training_state`` reads them."""
    if not h.get("use_cqtd_instead_of_mrd", False):
        raise ValueError("This release trains against an MRD; only its CQTD has a counterpart here.")
    with open(RECIPE, encoding="utf-8") as handle:
        settings = json.load(handle)["discriminator"]
    differing = [key for key, release in D_KEYS.items() if settings[key] != h[release]]
    if h.get("cqtd_filters_scale", 1) != 1 or list(h.get("cqtd_dilations", [1, 2, 4])) != [1, 2, 4]:
        differing.append("cqtd_filters_scale/cqtd_dilations")
    if differing:
        raise ValueError(f"The recipe's discriminator differs from the release's in {', '.join(differing)}.")
    discriminator = sv.build_discriminators(sv.BIGVGAN_V2, settings, sample_rate)
    reference = discriminator.state_dict()
    weights = {f"mpd.{key}": value for key, value in checkpoint["mpd"].items()}
    weights.update({f"cqt.{key}": value for key, value in checkpoint["mrd"].items()})
    # The CQT kernels and the resampler's, as this nnAudio and torchaudio build them.
    weights = {key: value for key, value in weights.items() if not any(mark in key for mark in FIXED_BUFFERS)}
    weights.update({key: value for key, value in reference.items() if any(mark in key for mark in FIXED_BUFFERS)})
    discriminator.load_state_dict(sv.training_state(weights, reference))
    size = sum(value.numel() for key, value in weights.items() if not any(m in key for m in FIXED_BUFFERS)) / 1e6
    return {"model": weights}, [f"discriminator: mpd, cqt, {size:.1f}M weights"]


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(description="Convert an official BigVGAN v2 generator to an export.")
    parser.add_argument("checkpoint", help="The generator's bigvgan_generator.pt.")
    parser.add_argument("output", help="Where to write the export.")
    parser.add_argument("--config", help="The release's config.json; default: beside the checkpoint.")
    parser.add_argument("--discriminator", help=f"Also write {D_CHECKPOINT}, from beside the checkpoint, here.")
    parser.add_argument("--discriminator_checkpoint", help=f"Where {D_CHECKPOINT} is, if not beside it.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the outputs.")
    args = parser.parse_args()

    source, output = Path(args.checkpoint), Path(args.output)
    try:
        outputs = [output] + ([Path(args.discriminator)] if args.discriminator else [])
        for path in outputs:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace.")
        with open(args.config or source.with_name("config.json"), encoding="utf-8") as handle:
            h = json.load(handle)
        state = torch.load(source, map_location="cpu", weights_only=True)
        export, lines = convert(state.get("generator", state), h)
        payloads = [export]
        if args.discriminator:
            d_path = Path(args.discriminator_checkpoint or source.with_name(D_CHECKPOINT))
            d_state, more = convert_discriminators(
                torch.load(d_path, map_location="cpu", weights_only=True), h, export["config"]["data"]["sample_rate"]
            )
            payloads.append(d_state)
            lines += more
        info(f"{source} -> {', '.join(path.name for path in outputs)}", tag=TAG)
        for line in lines:
            info(f"  {line}", tag=TAG)
        for path, payload in zip(outputs, payloads):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            success(f"Written to {path}.", tag=TAG)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print_error(f"{source.name}: {exc}", tag=TAG)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
