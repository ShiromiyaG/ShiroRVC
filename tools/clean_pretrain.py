"""Turn a training checkpoint into a pretrain, without touching precision.

A ``G_*.pth`` / ``D_*.pth`` pair written by the training loop is a *resume*
checkpoint: alongside the weights it carries the optimizer moments, the step
counter, the last learning rate and the AMP controller state.  Roughly three
quarters of the file is optimizer state that a finetune re-derives on its first
few steps anyway, and every counter in it belongs to the run that produced it,
not to the run that will consume it.

This strips a checkpoint down to what ``pretrainG`` / ``pretrainD`` actually
read, and nothing else:

  kept    ``model``, in **fp32** -- including ``enc_q``, which the inference
          export (``rvc/train/process/extract_model.py``) drops.  A pretrain
          loads with ``strict=True`` into a full training model, so a posterior
          encoder missing from the file fails the load outright; the 52 MB
          ``*_pre-overtrain_*.pth`` weights are inference models and cannot be
          used here.
  kept    ``architecture_id``, ``excitation_source``, ``decoder_layout``,
          ``discriminator_periods`` -- invisible in the weights, and the guards
          in ``load_models_and_optimizers`` read them to refuse a pretrain
          built for a different decoder, excitation or period set.  Dropping
          them turns a loud mismatch into a silent one.
  dropped ``optimizer``  the moments are shaped for the old run's parameter
          groups and LR; a finetune builds its own.
  dropped ``iteration``, ``learning_rate``, ``extra``  counters and controller
          state (grad scaler, AMP skip tally) belonging to the finished run.
  dropped ``ema``  folded into ``model`` by default (see ``--weights``) rather
          than carried, since a pretrain has no averaging history to continue.

Precision is left alone on purpose.  ``extract_model`` halves the weights
because an inference model is read once and never trained further; a pretrain
is the *initialisation* of another training run, and starting a finetune from
fp16-rounded weights bakes that rounding into everything downstream for no gain
-- the file is loaded to CPU and immediately copied into fp32 parameters.

    python tools/clean_pretrain.py --model-dir logs/pretrain
    python tools/clean_pretrain.py --model-dir logs/pretrain --weights live
    python tools/clean_pretrain.py --generator logs/pretrain/G_306194.pth \\
        --discriminator logs/pretrain/D_306194.pth --output-dir logs/pretrains

The output is named ``<prefix>_G.pth`` / ``<prefix>_D.pth``.  The step is
deliberately *not* put back in the name: ``checkpoint_step_from_path`` reads a
``G_<digits>.pth`` filename to seed ``global_step`` on a finetune, so a pretrain
called ``G_306194.pth`` would start its finetune 306k steps into the LR
schedule.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import (
    error as print_error,
    info,
    install_rich_print,
    success,
    warning,
)

#: Everything a pretrain load path reads.  ``model`` is mandatory; the rest are
#: additive metadata keys that only some checkpoints carry.
KEEP_METADATA = (
    "architecture_id",
    "excitation_source",
    "decoder_layout",
    "discriminator_periods",
)
#: Named so the report can say what went and why, instead of "everything else".
DROP_REASONS = {
    "optimizer": "optimizer moments (a finetune builds its own)",
    "iteration": "epoch counter of the finished run",
    "learning_rate": "last LR of the finished run",
    "extra": "AMP/grad-scaler controller state",
    "ema": "weight average (folded into 'model' or discarded)",
}


def latest_checkpoint(model_dir: Path, prefix: str) -> Path:
    """The ``<prefix>_<step>.pth`` in ``model_dir`` with the highest step."""

    pattern = re.compile(rf"^{prefix}_(\d+)\.pth$")
    found = []
    for entry in model_dir.iterdir():
        match = pattern.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    if not found:
        raise FileNotFoundError(
            f"No '{prefix}_<step>.pth' under {model_dir}. Point --{prefix.lower()}"
            f"enerator at the file directly if it is named differently."
            if prefix == "G"
            else f"No '{prefix}_<step>.pth' under {model_dir}."
        )
    return max(found)[1]


def tensor_bytes(state: dict) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if torch.is_tensor(value)
    )


def clean_checkpoint(path: Path, weights: str, role: str) -> tuple[dict, list[str]]:
    """Read one checkpoint and return the pretrain payload plus a report."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        raise ValueError(
            f"{path.name} has no 'model' key; it is not a training checkpoint."
        )

    state = checkpoint["model"]
    lines = []

    source = "live weights"
    if weights == "ema":
        ema = checkpoint.get("ema")
        if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
            shadow = ema["shadow"]
            missing = set(state) - set(shadow)
            if missing:
                # Partial shadows are not silently half-applied: mixing averaged
                # and live tensors gives weights that were never evaluated
                # together.
                warning(
                    f"{path.name}: EMA shadow is missing {len(missing)} of "
                    f"{len(state)} tensors; keeping live weights.",
                    tag="[CLEAN]",
                )
            else:
                state = shadow
                source = f"EMA shadow (decay {ema.get('decay')}, "
                source += f"{ema.get('updates')} updates)"
        elif role == "G":
            info(
                f"{path.name} carries no EMA; using live weights.", tag="[CLEAN]"
            )

    out = {"model": {key: value.float() for key, value in state.items()}}
    lines.append(f"weights: {source}, {len(out['model'])} tensors, fp32")

    for key in KEEP_METADATA:
        if key in checkpoint:
            out[key] = checkpoint[key]
            lines.append(f"kept:    {key} = {checkpoint[key]!r:.60}")

    for key in checkpoint:
        if key in out or key == "model":
            continue
        lines.append(f"dropped: {key} -- {DROP_REASONS.get(key, 'not read by a pretrain load')}")

    return out, lines


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(
        description="Strip a training checkpoint down to a finetune-ready pretrain.",
    )
    source = parser.add_argument_group("input")
    source.add_argument(
        "--model-dir",
        help="Run directory (e.g. logs/pretrain); uses its highest-step G/D pair.",
    )
    source.add_argument("--generator", help="Generator checkpoint, instead of --model-dir.")
    source.add_argument(
        "--discriminator",
        help="Discriminator checkpoint. Optional: a pretrain G alone is valid.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write (default: alongside the input).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output base name (default: the run directory's name).",
    )
    parser.add_argument(
        "--weights",
        choices=("ema", "live"),
        default="ema",
        help="Which generator weights to write. 'ema' (default) uses the "
        "averaged shadow when the checkpoint has one, which is what the "
        "holdout was scored on; 'live' takes the raw last step.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing existing outputs."
    )
    args = parser.parse_args()

    if not args.model_dir and not args.generator:
        parser.error("pass --model-dir, or --generator (and optionally --discriminator).")

    try:
        if args.model_dir:
            model_dir = Path(args.model_dir)
            g_path = latest_checkpoint(model_dir, "G")
            try:
                d_path = latest_checkpoint(model_dir, "D")
            except FileNotFoundError:
                d_path = None
                warning(f"No discriminator found in {model_dir}.", tag="[CLEAN]")
        else:
            g_path = Path(args.generator)
            d_path = Path(args.discriminator) if args.discriminator else None
            model_dir = g_path.parent

        out_dir = Path(args.output_dir) if args.output_dir else model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.prefix or model_dir.resolve().name

        jobs = [("G", g_path, out_dir / f"{prefix}_G.pth")]
        if d_path is not None:
            jobs.append(("D", d_path, out_dir / f"{prefix}_D.pth"))

        for _, _, out_path in jobs:
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"{out_path} exists; pass --overwrite to replace.")

        for role, in_path, out_path in jobs:
            info(f"{in_path.name} -> {out_path.name}", tag="[CLEAN]")
            payload, lines = clean_checkpoint(in_path, args.weights, role)
            for line in lines:
                info(f"  {line}", tag="[CLEAN]")
            torch.save(payload, out_path)
            before = in_path.stat().st_size / 1e6
            after = out_path.stat().st_size / 1e6
            success(
                f"  {before:,.0f} MB -> {after:,.0f} MB ({100 * (1 - after / before):.0f}% smaller)",
                tag="[CLEAN]",
            )

        success(
            f"Pretrain written to {out_dir}. Pass it as pretrainG/pretrainD "
            "(fp32; not an inference model).",
            tag="[CLEAN]",
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print_error(str(exc), tag="[CLEAN]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
