"""Teach a converted Pupu-Vocoder or BigVGAN v2 to use f0, in two phases.

The official weights take the mel alone, so a sustained note drifts off pitch
and its phase slips. ``convert_pupu_vocoder.py`` and ``convert_bigvgan_v2.py``
add the f0 source as zeros; this trains it.

    python tools/pretrain/adapt_pupu_f0.py source --model_name my-data
    python tools/pretrain/adapt_pupu_f0.py finish --model_name my-data --epochs 1

``--architecture bigvgan-v2`` in both phases does the same for BigVGAN v2.

``source`` trains the f0 source alone, the rest of the generator frozen, on
the recipe's spectral loss with no discriminator: a step costs a sixth of an
adversarial one, and nothing the official weights know can be undone. The mel
it reads is degraded, so the pitch has to come from f0, and a second loss
holds the harmonics to the sine's phase, which the spectral one cannot see
(``--lock_weight``, 0 for none). It writes an export into the experiment's
folder of that vocoder.

``finish`` is a short run of the ordinary trainer from that export and the
official discriminator, everything trainable, at a fine-tune's learning rate
behind a warmup. It keeps the period family alone, the one that sees a phase
slip in the waveform; ``--families`` names others. ``--lock_weight`` is the
weight of the loss that holds the harmonics to the f0 sine's phase, 0 for
none. Run it again with ``--resume`` to continue, with the same families. It saves every
``--save_steps`` steps as well as at each epoch's end.

For the large Pupu-Vocoder, name its files: ``--pretrained_g`` in ``source``
and ``--pretrained_d`` in ``finish``. The experiment's recipe takes the layout
of the generator it is given. BigVGAN v2's discriminator has the families mpd
and cqt.

The experiment has to be preprocessed and extracted at the recipe's rate.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import soundfile as sf
import torch
from torch.utils.data import DataLoader

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import error as print_error, info, install_rich_print, success
from rvc.rectified import singing_vocoders as sv
from rvc.rectified.common import (
    RectifiedDataset,
    amp_setup,
    collate_vocoder,
    read_filelist,
    vocoder_run_dir,
)
from rvc.rectified.mel import degrade_mel, denormalize_mel, normalize_mel
from rvc.rectified.pupu_vocoder import PhaseLockLoss
from rvc.rectified.train_vocoder import SpectralLoss, export_payload
from rvc.train.setup import apply_precision_policy

TAG = "[ADAPT]"
#: What ``source`` trains: the sine's mix and the convs that bring it to each stage.
SOURCE = ("source_convs.", "m_source.")
#: Tag of the exports ``source`` writes, which ``finish`` looks for.
SOURCE_EXPORT = "f0source"
LOG_EVERY = 50
VOCODER_DIR = os.path.join("rvc", "models", "vocoders")
D_DIR = os.path.join("rvc", "models", "pretraineds", "custom")
#: Per architecture: the converted generator and discriminator it starts from.
DEFAULTS = {
    sv.PUPU: (
        os.path.join(VOCODER_DIR, "pupu_small_44.1k_hop512_128bin_vocoder.pth"),
        os.path.join(D_DIR, "D_pupu_small_44.1k.pth"),
    ),
    sv.BIGVGAN_V2: (
        os.path.join(VOCODER_DIR, "bigvgan_v2_44.1k_hop512_128bin_vocoder.pth"),
        os.path.join(D_DIR, "D_bigvgan_v2_44.1k.pth"),
    ),
}
FAMILIES = {sv.PUPU: ("msd", "mpd", "mbd", "cqt"), sv.BIGVGAN_V2: ("mpd", "cqt")}


def source_norm(net_g) -> float:
    """Norm of the source convs' weights: 0 in the converted official weights."""
    return sum(conv.weight.detach().float().square().sum() for conv in net_g.source_convs).sqrt().item()


def export(net_g, architecture: str, config: dict, hparams: dict, path: str) -> None:
    weights = SimpleNamespace(
        cpu_state_dict=lambda: {key: value.detach().cpu() for key, value in net_g.state_dict().items()}
    )
    torch.save({"kind": "rectified_vocoder", **export_payload(architecture, config, hparams, weights)}, path)


def adopt_layout(name: str, architecture: str, path: str) -> None:
    """Give the experiment's recipe the layout of the export at ``path``: the
    small and the large Pupu-Vocoder differ in their channels."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("architecture") != sv.EXPORT_ARCHITECTURE[architecture]:
        raise ValueError(f"{os.path.basename(path)} is not a {architecture} export.")
    layout = checkpoint["config"]["vocoder"]["model"]
    settings = sv.read_vocoder_settings(name, architecture)
    wanted = {key: layout[key] for key in settings["model"] if key in layout}
    if any(settings["model"][key] != value for key, value in wanted.items()):
        settings["model"].update(wanted)
        sv.write_vocoder_settings(name, architecture, settings)
        info(f"{name}'s recipe now has the layout of {os.path.basename(path)}.", tag=TAG)


def train_source(args) -> None:
    name, architecture = args.model_name, args.architecture
    pretrained_g = args.pretrained_g or DEFAULTS[architecture][0]
    adopt_layout(name, architecture, pretrained_g)
    config = sv.load_recipe_config(name, architecture)
    data, settings = config["data"], config["vocoder"]
    out_dir = vocoder_run_dir(name, architecture)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    hparams = sv.generator_hparams(architecture, config)
    net_g = sv.build_generator(architecture, hparams).to(device)
    sv.load_pretrained_generator(net_g, sv.pretrained_states(pretrained_g)[0])
    trained = []
    for key, parameter in net_g.named_parameters():
        parameter.requires_grad_(key.startswith(SOURCE))
        if parameter.requires_grad:
            trained.append(parameter)
    optimizer = torch.optim.AdamW(trained, args.learning_rate, betas=tuple(settings["betas"]), weight_decay=0.0)
    amp_dtype, scaler = amp_setup(args.precision, device, TAG)
    apply_precision_policy(net_g, amp_dtype)
    spectral_loss = SpectralLoss(architecture, config, device)
    phase_lock = None
    if args.lock_weight > 0:
        phase_lock = PhaseLockLoss(
            data["sample_rate"], data["hop_length"], args.lock_harmonics, fmax=args.lock_fmax
        ).to(device)

    dataset = RectifiedDataset(
        read_filelist(name), config, "vocoder", settings["segment_size"] // data["hop_length"]
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_vocoder,
        drop_last=True, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    reference = dataset.reference()
    info(
        f"{name}: f0 source of {os.path.basename(pretrained_g)}, {sum(p.numel() for p in trained)} of "
        f"{sum(p.numel() for p in net_g.parameters())} weights, {args.steps} steps of batch {args.batch_size}, "
        f"lr {args.learning_rate:g}, {spectral_loss.label}"
        + (f", phase lock x{args.lock_weight:g} on {args.lock_harmonics} harmonics" if phase_lock is not None else "")
        + f", mel degradation {args.mel_degradation:g}.",
        tag=TAG,
    )

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None)

    def preview(step: int) -> None:
        """The reference clip from its real mel, beside the exports."""
        if reference is None:
            return
        with torch.no_grad():
            audio = net_g(reference.mel.to(device), reference.inputs.f0.to(device))
        path = os.path.join(out_dir, f"{SOURCE_EXPORT}_preview_{step}.wav")
        sf.write(path, audio[0, 0].float().cpu().numpy(), data["sample_rate"])

    def save(step: int) -> str:
        path = os.path.join(out_dir, f"{name}_vocoder_{SOURCE_EXPORT}_{step}s.pth")
        export(net_g, architecture, config, hparams, path)
        return path

    step, total, started = 0, 0.0, time.perf_counter()
    # The lock loss of the render and of the real audio, which f0's own error keeps off 0.
    locks = torch.zeros(2, device=device)
    preview(0)
    try:
        while step < args.steps:
            for mel, f0, y in loader:
                mel = normalize_mel(mel.to(device, non_blocking=True), data)
                if args.mel_degradation > 0:
                    mel = degrade_mel(mel, args.mel_degradation)
                y = y.to(device, non_blocking=True)
                f0 = f0.to(device, non_blocking=True)
                with autocast():
                    y_hat = net_g(denormalize_mel(mel, data), f0)
                    spectral = loss = spectral_loss(y, y_hat)
                if phase_lock is not None:
                    lock = phase_lock(y_hat, f0)
                    loss = loss + lock * args.lock_weight
                    with torch.no_grad():
                        locks += torch.stack((lock.detach(), phase_lock(y, f0)))
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                step += 1
                total += spectral.item()
                if step % LOG_EVERY == 0:
                    rate = (time.perf_counter() - started) / step
                    info(
                        f"step {step}/{args.steps}  loss {total / LOG_EVERY:.3f}  "
                        + (
                            "phase lock {:.3f} (real {:.3f})  ".format(*(locks / LOG_EVERY).tolist())
                            if phase_lock is not None else ""
                        )
                        + f"source norm {source_norm(net_g):.3f}  {1000 * rate:.0f} ms/step",
                        tag=TAG,
                    )
                    total = 0.0
                    locks.zero_()
                if step % args.save_every == 0:
                    preview(step)
                    info(f"Saved {os.path.basename(save(step))}.", tag=TAG)
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        info("Stopped; saving what was trained.", tag=TAG)
    preview(step)
    success(f"f0 source after {step} steps written to {save(step)}.", tag=TAG)


def newest_source_export(out_dir: str) -> str:
    paths = glob.glob(os.path.join(out_dir, f"*_vocoder_{SOURCE_EXPORT}_*s.pth"))
    if not paths:
        raise FileNotFoundError(f"No {SOURCE_EXPORT} export in {out_dir}; run the source phase first.")
    return max(paths, key=lambda path: int(path.rpartition("_")[2][:-5]))


def finish(args) -> int:
    name, architecture = args.model_name, args.architecture
    if unknown := set(args.families) - set(FAMILIES[architecture]):
        raise ValueError(f"{architecture} has the families {', '.join(FAMILIES[architecture])}, "
                         f"not {', '.join(sorted(unknown))}.")
    out_dir = vocoder_run_dir(name, architecture)
    os.makedirs(out_dir, exist_ok=True)
    pretrained_g = args.pretrained_g or newest_source_export(out_dir)
    pretrained_d = args.pretrained_d or DEFAULTS[architecture][1]
    adopt_layout(name, architecture, pretrained_g)
    # A fine-tune's start: a cold optimizer at the recipe's learning rate
    # throws the trained weights off before its statistics settle.
    settings = sv.read_vocoder_settings(name, architecture)
    settings.update(learning_rate=args.learning_rate, warmup_steps=args.warmup_steps)
    settings["discriminator"]["families"] = args.families
    settings["c_phase_lock"] = args.lock_weight
    settings["phase_lock_harmonics"] = args.lock_harmonics
    settings["phase_lock_fmax"] = args.lock_fmax
    settings["phase_lock_low"] = args.lock_low
    sv.write_vocoder_settings(name, architecture, settings)
    spec = {
        "model_name": name,
        "architecture": architecture,
        "discriminator": "original",
        "total_epochs": args.epochs,
        "save_every": 1,
        "save_steps": args.save_steps,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "pretrained_g": pretrained_g,
        "pretrained_d": pretrained_d,
        "checkpoints": "latest",
        "fresh": not args.resume,
        "precision": args.precision,
        "mel_degradation": args.mel_degradation,
    }
    spec_path = os.path.join(out_dir, "run_spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=4)
    info(
        f"{name}: adversarial finish from {os.path.basename(pretrained_g)} and "
        f"{os.path.basename(pretrained_d)} ({' + '.join(args.families)}), {args.epochs} epoch(s), "
        f"lr {args.learning_rate:g} behind {args.warmup_steps} warmup steps"
        + (f", phase lock x{args.lock_weight:g}" if args.lock_weight > 0 else "")
        + (", resuming." if args.resume else "."),
        tag=TAG,
    )
    return subprocess.run([sys.executable, os.path.join("rvc", "rectified", "train_vocoder.py"), spec_path]).returncode


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(description="Teach a converted Pupu-Vocoder or BigVGAN v2 to use f0.")
    phases = parser.add_subparsers(dest="phase", required=True)

    source = phases.add_parser("source", help="Train the f0 source alone, without a discriminator.")
    source.add_argument("--model_name", required=True, help="Experiment with the dataset, preprocessed and extracted.")
    source.add_argument("--architecture", choices=sv.PHASE_LOCKED, default=sv.PUPU)
    source.add_argument(
        "--pretrained_g", default="",
        help=f"Converted export to start from; default: the architecture's in {VOCODER_DIR}.",
    )
    source.add_argument("--steps", type=int, default=20000)
    source.add_argument("--batch_size", type=int, default=8)
    source.add_argument("--learning_rate", type=float, default=5e-4)
    source.add_argument("--mel_degradation", type=float, default=0.5, help="0 to 1; 0 leaves the mel as it is.")
    source.add_argument("--lock_weight", type=float, default=20.0, help="Weight of the phase lock loss; 0 for none.")
    source.add_argument("--lock_harmonics", type=int, default=4, help="Harmonics the phase lock loss holds.")
    source.add_argument(
        "--lock_fmax", type=float, default=0.0,
        help="Hold the harmonics under this frequency, at most --lock_harmonics of them; 0 for all of those.",
    )
    source.add_argument("--save_every", type=int, default=2000, help="Steps between exports and previews.")
    source.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")

    end = phases.add_parser("finish", help="A short adversarial run from the source phase's export.")
    end.add_argument("--model_name", required=True)
    end.add_argument("--architecture", choices=sv.PHASE_LOCKED, default=sv.PUPU)
    end.add_argument("--pretrained_g", default="", help="Default: the newest export of the source phase.")
    end.add_argument(
        "--pretrained_d", default="",
        help=f"The official discriminator, converted; default: the architecture's in {D_DIR}.",
    )
    end.add_argument(
        "--families", nargs="+", default=["mpd"], choices=("msd", "mpd", "mbd", "cqt"),
        help="Discriminator families to train against; BigVGAN v2 has mpd and cqt.",
    )
    end.add_argument("--lock_weight", type=float, default=5.0, help="Weight of the phase lock loss; 0 for none.")
    end.add_argument("--lock_harmonics", type=int, default=4, help="Harmonics the phase lock loss holds.")
    end.add_argument(
        "--lock_fmax", type=float, default=0.0,
        help="Hold the harmonics under this frequency, at most --lock_harmonics of them; 0 for all of those.",
    )
    end.add_argument(
        "--lock_low", type=int, default=4,
        help="First harmonics that are also a term of their own, when --lock_harmonics is more; 0 for none.",
    )
    end.add_argument("--epochs", type=int, default=1)
    end.add_argument(
        "--save_steps", type=int, default=2000,
        help="Also save every this many steps; a resume from one repeats its epoch. 0 for epoch ends only.",
    )
    end.add_argument("--batch_size", type=int, default=8)
    end.add_argument("--learning_rate", type=float, default=2e-5)
    end.add_argument("--warmup_steps", type=int, default=2000)
    end.add_argument("--mel_degradation", type=float, default=0.0)
    end.add_argument("--gpu", default="0")
    end.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    end.add_argument("--resume", action="store_true", help="Continue the run instead of starting it over.")

    args = parser.parse_args()
    try:
        if args.phase == "source":
            train_source(args)
            return 0
        return finish(args)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc), tag=TAG)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
