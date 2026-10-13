"""Turn an official Pupu-Vocoder checkpoint into a rectified vocoder export.

The released generators (``model.safetensors`` under ``pupuvocoder`` and
``pupuvocoder_large`` of spellbrush/AliasingFreeNeuralAudioSynthesis, MIT)
take the mel alone. This repo's ``PupuVocoder`` also reads f0: its source
convs are written as zeros here, so the export renders exactly as the
original until a fine-tune teaches them. The weight norm is folded and the
filters, which are not weights, are left out.

    python tools/pretrain/convert_pupu_vocoder.py model.safetensors rvc/models/vocoders/pupu_small_vocoder.pth

The export also starts a fine-tune, as ``pretrained_g`` of a ``pupu-vocoder``
run whose ``upsample_initial_channel`` is the checkpoint's (512 small, 1536
large). ``--discriminator`` also writes the checkpoint's four discriminators
(``model_1`` to ``model_4.safetensors`` beside the generator) as one file for
``pretrained_d``, which fits the recipe's own discriminator:

    python tools/pretrain/convert_pupu_vocoder.py model.safetensors pupu_small_vocoder.pth \\
        --discriminator rvc/models/pretraineds/custom/D_pupu_small.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import error as print_error, info, install_rich_print, success
from rvc.rectified import pupu_vocoder
from rvc.rectified import singing_vocoders as sv

TAG = "[CONVERT]"
#: The mel the released checkpoints were trained on.
DATA = {
    "sample_rate": 44100, "hop_length": 512, "n_fft": 2048, "win_length": 2048,
    "n_mels": 128, "mel_fmin": 0.0, "mel_fmax": 22050.0,
}
UPSAMPLE_RATES = [8, 8, 2, 2, 2]
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
RECIPE = os.path.join(now_dir, "rvc", "configs", "rectified", "vocoders", "pupu-vocoder.json")
#: A tensor name only that family's file has, by the family's name here.
FAMILY_MARKS = {"msd": "weight_orig", "mbd": "band_convs", "cqt": "cqt_transform"}


def folded(state: dict) -> dict:
    """``state`` with each ``weight_g``/``weight_v`` pair as one ``weight``
    and without the filter buffers."""
    out = {}
    for key, value in state.items():
        if key.endswith(".filter") or key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            norm = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
            key, value = base, value * (state[base + "_g"] / norm)
        out[key] = value.float().contiguous()
    return out


def convert(state: dict):
    """The export of an official generator's ``state`` and the lines of its report."""
    weights = folded(state)
    channels = weights["conv_pre.weight"].shape[0]
    hparams = {
        "sample_rate": DATA["sample_rate"], "num_mels": DATA["n_mels"],
        "upsample_initial_channel": channels, "upsample_rates": UPSAMPLE_RATES,
        "resblock_kernel_sizes": RESBLOCK_KERNEL_SIZES,
        "resblock_dilation_sizes": RESBLOCK_DILATION_SIZES, "harmonic_num": 0,
    }
    generator = pupu_vocoder.PupuVocoder(**hparams)
    # What the original lacks: the f0 source, silent until trained. Its sine
    # goes to the convs as it is.
    added = {}
    for key, value in generator.state_dict().items():
        if key not in weights:
            plain = key == "m_source.l_linear.weight"
            added[key] = torch.ones_like(value) if plain else torch.zeros_like(value)
    generator.load_state_dict({**weights, **added})
    lines = [
        f"generator: {channels} initial channels, "
        f"{sum(value.numel() for value in weights.values()) / 1e6:.1f}M weights",
        f"added:     {len(added)} f0 source tensors, the convs as zeros",
    ]
    export = {
        "kind": "rectified_vocoder", "architecture": pupu_vocoder.ARCHITECTURE, "recipe": "pupu-vocoder",
        "config": {"data": dict(DATA), "vocoder": {"model": hparams}},
        "model": {key: value.detach().clone() for key, value in generator.state_dict().items()},
    }
    return export, lines


def family_of(state: dict) -> str:
    """Which discriminator family a file of the checkpoint holds; the
    multi-period one is the file with none of the others' marks."""
    for family, mark in FAMILY_MARKS.items():
        if any(mark in key for key in state):
            return family
    return "mpd"


def convert_discriminators(states: list):
    """The four discriminator files' states as one for the recipe's own
    discriminator, and the lines of its report. The names of the older
    weight norm stay: ``training_state`` reads them."""
    weights = {}
    for state in states:
        family = family_of(state)
        for key, value in state.items():
            if ".spec_fn." in key:
                continue
            # The original wraps the CQT heads' convs in a module of their own.
            weights[f"{family}.{key.replace('.conv.', '.')}"] = value
    with open(RECIPE, encoding="utf-8") as handle:
        settings = json.load(handle)["discriminator"]
    discriminator = sv.build_discriminators(sv.PUPU, settings, DATA["sample_rate"])
    discriminator.load_state_dict(sv.training_state(weights, discriminator.state_dict()))
    families = sorted({key.split(".")[0] for key in weights})
    size = sum(value.numel() for value in weights.values()) / 1e6
    return {"model": weights}, [f"discriminator: {', '.join(families)}, {size:.1f}M weights"]


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(description="Convert an official Pupu-Vocoder generator to an export.")
    parser.add_argument("checkpoint", help="The generator's model.safetensors.")
    parser.add_argument("output", help="Where to write the export.")
    parser.add_argument("--discriminator", help="Also write the discriminators beside the checkpoint here.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the outputs.")
    args = parser.parse_args()

    source, output = Path(args.checkpoint), Path(args.output)
    try:
        from safetensors.torch import load_file

        outputs = [output] + ([Path(args.discriminator)] if args.discriminator else [])
        for path in outputs:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace.")
        export, lines = convert(load_file(str(source)))
        payloads = [export]
        if args.discriminator:
            files = [source.with_name(f"{source.stem}_{number}.safetensors") for number in range(1, 5)]
            states, more = convert_discriminators([load_file(str(file)) for file in files])
            payloads.append(states)
            lines += more
        info(f"{source} -> {', '.join(path.name for path in outputs)}", tag=TAG)
        for line in lines:
            info(f"  {line}", tag=TAG)
        for path, payload in zip(outputs, payloads):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            success(f"Written to {path}.", tag=TAG)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError) as exc:
        print_error(f"{source.name}: {exc}", tag=TAG)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
