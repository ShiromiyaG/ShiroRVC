"""Make OpenVPI's PC-NSF-HiFiGAN render the rectified flow's mel, in two phases.

The release reads a 40-16000 Hz mel; the flow, Pupu-Vocoder and BigVGAN v2
read 0-22050 Hz, on the same STFT and the same 128 bins.

    python tools/pretrain/adapt_openvpi_mel.py convert
    python tools/pretrain/adapt_openvpi_mel.py discriminator --source pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt
    python tools/pretrain/adapt_openvpi_mel.py finetune --model_name my-data --epochs 1

``convert`` needs no training: it writes an export whose generator maps the
flow's mel to the release's (``openvpi.mel_map``) before its first conv, and
carries a second first conv over the flow's mel itself as zeros
(``mel_skip``). It renders as the release does, the band above 16 kHz still
made up by the generator, and loads anywhere the release's export does.

``discriminator`` writes the discriminator of the release's training
checkpoint (the ``.ckpt``, not its export) as a ``pretrained_d`` for the
recipe, checked against it. Its multi-scale and multi-period heads read the
waveform, not the mel, so they fit the adapted generator as they are.

``finetune`` is an ordinary run of the ``pc-nsf-hifigan`` recipe from that
export, everything trainable, at a fine-tune's learning rate behind a warmup,
on the experiment's dataset: ``mel_skip`` learns to use what the release's mel
lacks. It takes that discriminator when it has been written; without one the
recipe's starts from scratch and the generator's adversarial loss ramps in
over ``--adv_warmup_steps`` while the mel loss holds it. Run it again with
``--resume`` to continue. The experiment has to be preprocessed and extracted
at the flow's mel.

The release's weights are CC BY-NC-SA 4.0, and so is what is trained from them.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import error as print_error, info, install_rich_print, success
from rvc.rectified import openvpi
from rvc.rectified import singing_vocoders as sv
from rvc.rectified.common import DEFAULT_CONFIG, load_run_config, vocoder_run_dir

TAG = "[ADAPT]"
ARCHITECTURE = "pc-nsf-hifigan"
VOCODER_DIR = os.path.join("rvc", "models", "vocoders")
DEFAULT_SOURCE = os.path.join(VOCODER_DIR, "pc_nsf_hifigan_44.1k_hop512_128bin_vocoder.pth")
DEFAULT_OUTPUT = os.path.join(VOCODER_DIR, "pc_nsf_hifigan_mel22k_44.1k_hop512_128bin_vocoder.pth")
DEFAULT_D = os.path.join("rvc", "models", "pretraineds", "custom", "D_pc_nsf_hifigan_44.1k.pth")
RECIPE = os.path.join("rvc", "configs", "rectified", "vocoders", f"{ARCHITECTURE}.json")
#: What has to agree between the two mels for one to be mapped to the other.
SAME_GRID = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels")


def band(mel: dict) -> list:
    return [float(mel["mel_fmin"]), float(mel["mel_fmax"] or mel["sample_rate"] / 2)]


def mel_of(data: dict) -> dict:
    """``data``'s mel settings, with an unset top as the Nyquist."""
    mel = {key: data[key] for key in openvpi.DEFAULT_MEL}
    mel["mel_fmax"] = band(mel)[1]
    return mel


def target_mel(model_name: str) -> dict:
    """The mel to render: the experiment's, else the shipped rectified config's."""
    if model_name:
        return mel_of(load_run_config(model_name)["data"])
    with open(DEFAULT_CONFIG, encoding="utf-8") as handle:
        return mel_of(json.load(handle)["data"])


def read_source(path: str):
    """``(hparams, mel, weights)`` of an OpenVPI export or checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") == "rectified_vocoder":
        if checkpoint.get("architecture") not in (openvpi.ARCHITECTURE, openvpi.LEGACY_ARCHITECTURE):
            raise ValueError(f"{path} is not an OpenVPI NSF-HiFiGAN export.")
        return dict(checkpoint["config"]["vocoder"]["model"]), checkpoint["config"]["data"], checkpoint["model"]
    state = openvpi.generator_state(checkpoint)
    if state is None:
        raise ValueError(f"{path} is neither an OpenVPI checkpoint nor its export.")
    return openvpi.openvpi_spec(path, state)


def convert(args) -> None:
    hparams, native, weights = read_source(args.source)
    if hparams.get("mel_adapter"):
        raise ValueError(f"{args.source} is already adapted (to {hparams['mel_adapter']['input']} Hz).")
    target = target_mel(args.model_name)
    if differing := [key for key in SAME_GRID if float(native[key]) != float(target[key])]:
        raise ValueError(f"The mels differ in {', '.join(differing)}; only the band can be mapped.")
    if band(native) == band(target):
        raise ValueError(f"{args.source} already renders the {band(target)} Hz mel.")
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")
    adapter = {"n_fft": int(target["n_fft"]), "input": band(target), "native": band(native)}
    hparams["mel_adapter"] = adapter
    generator = openvpi.NSFHiFiGAN(**hparams)
    added = {key: value for key, value in generator.state_dict().items() if key not in weights}
    generator.load_state_dict({**weights, **added})

    # How much of each native filter the input filters can make: its share
    # outside their span is lost resolution.
    from librosa.filters import mel as librosa_mel_fn

    def basis(edges):
        return librosa_mel_fn(sr=int(target["sample_rate"]), n_fft=adapter["n_fft"], n_mels=int(target["n_mels"]),
                              fmin=edges[0], fmax=edges[1])

    want = basis(adapter["native"])
    made = generator.mel_map.numpy() @ basis(adapter["input"])
    error = np.linalg.norm(made - want, axis=1) / np.linalg.norm(want, axis=1)
    torch.save(
        {
            "kind": "rectified_vocoder", "architecture": openvpi.ARCHITECTURE, "recipe": ARCHITECTURE,
            "config": {"data": target, "vocoder": {"model": hparams}},
            "model": {key: value.detach().float().contiguous() for key, value in generator.state_dict().items()},
            "source": os.path.basename(args.source),
        },
        args.output,
    )
    info(f"{os.path.basename(args.source)}: {adapter['native']} Hz mel -> {adapter['input']} Hz mel.", tag=TAG)
    info(f"  native filters as mixes of the input ones: shape error median {np.median(error):.0%}, "
         f"worst {error.max():.0%} (each native band reads a slightly smoothed one)", tag=TAG)
    info(f"  added: {', '.join(sorted(added))} (zeros)", tag=TAG)
    success(f"Written to {args.output}.", tag=TAG)


def discriminator(args) -> None:
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")
    _, weights = sv.pretrained_states(args.source)
    if not weights:
        raise ValueError(f"{args.source} holds no discriminator; the training checkpoint (.ckpt) does.")
    with open(RECIPE, encoding="utf-8") as handle:
        settings = json.load(handle)["discriminator"]
    net_d = sv.build_discriminators(ARCHITECTURE, settings, openvpi.DEFAULT_MEL["sample_rate"])
    net_d.load_state_dict(sv.training_state(weights, net_d.state_dict()))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"model": weights}, args.output)
    families = sorted({key.split(".")[0] for key in weights})
    size = sum(value.numel() for value in weights.values()) / 1e6
    info(f"{os.path.basename(args.source)}: discriminator {', '.join(families)}, {size:.1f}M weights, "
         f"fits the recipe's ({len(settings['periods'])} periods).", tag=TAG)
    success(f"Written to {args.output}.", tag=TAG)


def finetune(args) -> int:
    name = args.model_name
    pretrained_d = args.pretrained_d
    if pretrained_d and not os.path.exists(pretrained_d):
        if pretrained_d != DEFAULT_D:
            raise FileNotFoundError(f"{pretrained_d} does not exist.")
        pretrained_d = ""
    checkpoint = torch.load(args.pretrained_g, map_location="cpu", weights_only=True, mmap=True)
    layout = checkpoint["config"]["vocoder"]["model"]
    if not layout.get("mel_adapter"):
        raise ValueError(f"{args.pretrained_g} has no mel adapter; run the convert phase first.")
    data, own = mel_of(load_run_config(name)["data"]), mel_of(checkpoint["config"]["data"])
    if differing := [key for key in openvpi.DEFAULT_MEL if float(own[key]) != float(data[key])]:
        raise ValueError(f"{name}'s mel differs from {os.path.basename(args.pretrained_g)}'s in "
                         f"{', '.join(differing)}; convert with --model_name {name}.")
    settings = sv.read_vocoder_settings(name, ARCHITECTURE)
    settings["model"].update({key: layout[key] for key in settings["model"] if key in layout})
    settings["model"]["mel_adapter"] = layout["mel_adapter"]
    adv_warmup = args.adv_warmup_steps
    if adv_warmup < 0:
        adv_warmup = 0 if pretrained_d else 5000
    settings.update(
        learning_rate=args.learning_rate, finetune_learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, adv_warmup_steps=adv_warmup,
    )
    if args.pc_aug_rate is not None:
        settings["pc_aug_rate"] = args.pc_aug_rate
    sv.write_vocoder_settings(name, ARCHITECTURE, settings)
    out_dir = vocoder_run_dir(name, ARCHITECTURE)
    os.makedirs(out_dir, exist_ok=True)
    spec = {
        "model_name": name,
        "architecture": ARCHITECTURE,
        "discriminator": "original",
        "total_epochs": args.epochs,
        "save_every": 1,
        "save_steps": args.save_steps,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "pretrained_g": os.path.abspath(args.pretrained_g),
        "pretrained_d": pretrained_d,
        "checkpoints": "latest",
        "fresh": not args.resume,
        "precision": args.precision,
        "mel_degradation": 0.0,
    }
    spec_path = os.path.join(out_dir, "run_spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=4)
    info(
        f"{name}: {ARCHITECTURE} from {os.path.basename(args.pretrained_g)}, D "
        + (f"from {os.path.basename(pretrained_d)}" if pretrained_d else "from scratch")
        + f", {args.epochs} epoch(s), lr {args.learning_rate:g} behind {args.warmup_steps} warmup steps"
        + (f", adversarial loss ramped over {adv_warmup} steps" if adv_warmup else "")
        + (f", pitch-controllable augmentation on {settings.get('pc_aug_rate', 0):g} of each batch"
           if settings.get("pc_aug_rate") else ", no pitch-controllable augmentation")
        + (", resuming." if args.resume else "."),
        tag=TAG,
    )
    return subprocess.run([sys.executable, os.path.join("rvc", "rectified", "train_vocoder.py"), spec_path]).returncode


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(description="Make PC-NSF-HiFiGAN render the rectified flow's mel.")
    phases = parser.add_subparsers(dest="phase", required=True)

    first = phases.add_parser("convert", help="Map the flow's mel to the release's; no training.")
    first.add_argument("--source", default=DEFAULT_SOURCE, help="The release's export or checkpoint.")
    first.add_argument("--output", default=DEFAULT_OUTPUT)
    first.add_argument("--model_name", default="", help="Take the mel from this experiment; default: the shipped one.")
    first.add_argument("--overwrite", action="store_true", help="Allow replacing the output.")

    disc = phases.add_parser("discriminator", help="Write the release checkpoint's discriminator for the recipe.")
    disc.add_argument("--source", required=True, help="The release's training checkpoint (.ckpt).")
    disc.add_argument("--output", default=DEFAULT_D)
    disc.add_argument("--overwrite", action="store_true", help="Allow replacing the output.")

    second = phases.add_parser("finetune", help="Train the converted export on the experiment's dataset.")
    second.add_argument("--model_name", required=True, help="Experiment with the dataset, preprocessed and extracted.")
    second.add_argument("--pretrained_g", default=DEFAULT_OUTPUT, help="The convert phase's export.")
    second.add_argument(
        "--pretrained_d", default=DEFAULT_D,
        help="Discriminator for the recipe; default: the discriminator phase's, from scratch without it.",
    )
    second.add_argument("--epochs", type=int, default=1)
    second.add_argument("--batch_size", type=int, default=8)
    second.add_argument("--learning_rate", type=float, default=1e-5)
    second.add_argument("--warmup_steps", type=int, default=1000)
    second.add_argument(
        "--pc_aug_rate", type=float, default=None,
        help="Share of each batch rendered pitch-shifted (0 turns it off); default: the recipe's.",
    )
    second.add_argument(
        "--adv_warmup_steps", type=int, default=-1,
        help="Steps the adversarial loss ramps in over; default 5000 without --pretrained_d, else 0.",
    )
    second.add_argument(
        "--save_steps", type=int, default=2000,
        help="Also save every this many steps; a resume from one repeats its epoch. 0 for epoch ends only.",
    )
    second.add_argument("--gpu", default="0")
    second.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    second.add_argument("--resume", action="store_true", help="Continue the run instead of starting it over.")

    args = parser.parse_args()
    try:
        if args.phase in ("convert", "discriminator"):
            {"convert": convert, "discriminator": discriminator}[args.phase](args)
            return 0
        return finetune(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print_error(str(exc), tag=TAG)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
