"""Grow a checkpoint's ``emb_g`` so a pretrain can carry its speakers forward.

Adding speakers to a dataset leaves no good path through the loader.  A resume
is out: with ``G_*.pth`` in the experiment folder, ``verify_spk_dim`` locks
``spk_embed_dim`` to the checkpoint's row count, and the table cannot grow.
The pretrained route works, but ``verify_spk_dim`` sees a row count that
disagrees with the dataset and -- correctly, for the case it was written for --
throws the whole table away, so a staged pretrain loses every speaker it had
already learned.

The rule it applies is the shape, because the shape is what carries the
question.  So this makes the shape tell the truth: it widens ``emb_g.weight``
to the new speaker count, keeping rows ``0..n-1`` exactly as trained and
appending fresh ones.  The counts then match, ``reset_pretrained`` stays false,
and the run starts with its old speakers intact and the new ones uninitialised
-- which is what staged pretraining wants.

This is only valid when the existing row indices still mean the same speakers.
Slices are keyed ``{sid}_{idx0}_{idx1}`` and ``sid`` comes from the ``N_name``
folder prefix, not from directory order, so appending new speakers as ``50_x``,
``51_y`` keeps rows ``0..49`` pointing where they did.  Renumbering existing
folders does not, and no check here can see that.

How the new rows are drawn matters more than it looks.  ``nn.Embedding``
initialises at ``N(0, 1)``, but that is the scale a table has when the decoder
is trained *with* it; here ``dec.cond`` is already trained and reads rows at
whatever scale the table actually drifted to, so ``--init normal`` can hand it
rows several times louder than any speaker it has seen.  The default instead
draws from the existing rows' per-dimension mean and standard deviation: the
new speakers land at the centroid of the trained ones -- no particular
identity -- spread as widely as real speakers are.

Usage::

    python tools/widen_speaker_table.py logs/pretrain/G_194326.pth --speakers 80
    python tools/widen_speaker_table.py logs/pretrain/G_194326.pth --from-log logs/pretrain-v2

The output keeps the ``G_<step>.pth`` name in a new directory, because the
fine-tuning branch reads the starting ``global_step`` off that filename
(``checkpoint_step_from_path``); renaming it to ``G_194326_80spk.pth`` silently
restarts the step counter at zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


EMB_KEY = "emb_g.weight"


def resolve_speaker_count(args: argparse.Namespace) -> int:
    """The target row count, from ``--speakers`` or a run's ``model_info.json``."""
    if args.speakers is not None:
        return int(args.speakers)
    info_path = Path(args.from_log) / "model_info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            return int(json.load(handle)["speakers_id"])
    except Exception as error:  # noqa: BLE001 -- the message is the whole point
        sys.exit(f"Could not read the speaker count from '{info_path}': {error}")


def new_rows(existing: torch.Tensor, count: int, mode: str, seed: int) -> torch.Tensor:
    """``count`` fresh embedding rows, shaped like ``existing``'s columns."""
    generator = torch.Generator().manual_seed(seed)
    shape = (count, existing.shape[1])

    if mode == "zeros":
        return torch.zeros(shape, dtype=existing.dtype)
    if mode == "normal":
        # What ``nn.Embedding`` would have produced.  See the module docstring
        # for why this is an option and not the default.
        return torch.randn(shape, generator=generator).to(existing.dtype)

    # "match": the trained table's own location and spread, per dimension.
    reference = existing.float()
    mean = reference.mean(dim=0, keepdim=True)
    # A single-row table has no spread to measure.
    std = (
        reference.std(dim=0, unbiased=False, keepdim=True)
        if reference.shape[0] > 1
        else torch.ones_like(mean)
    )
    drawn = torch.randn(shape, generator=generator) * std + mean
    return drawn.to(existing.dtype)


def mean_pairwise_distance(rows: torch.Tensor) -> float:
    """Average L2 distance between rows -- how separated these speakers are."""
    if rows.shape[0] < 2:
        return float("nan")
    return float(torch.pdist(rows.float()).mean())


def widen(state_dict: dict, speakers: int, mode: str, seed: int) -> torch.Tensor:
    """Replace ``emb_g.weight`` with a taller one; returns the appended rows."""
    existing = state_dict[EMB_KEY]
    added = new_rows(existing, speakers - existing.shape[0], mode, seed)
    state_dict[EMB_KEY] = torch.cat([existing, added], dim=0)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Widen a generator checkpoint's speaker table so a staged "
        "pretrain keeps its trained speakers when the dataset grows.",
    )
    parser.add_argument("checkpoint", type=Path, help="the G_*.pth to widen")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--speakers", type=int, help="target speaker count")
    target.add_argument(
        "--from-log",
        type=Path,
        help="read the count from this run folder's model_info.json",
    )
    parser.add_argument("--out", type=Path, help="output path")
    parser.add_argument(
        "--init",
        choices=("match", "normal", "zeros"),
        default="match",
        help="how to draw the new rows (default: match the trained rows' "
        "per-dimension mean and spread)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--strip-optimizer",
        action="store_true",
        help="drop the optimizer state, whose moments still have the old "
        "width; the fine-tuning branch never reads it, so this only shrinks "
        "the file -- but it makes the result unusable as a resume",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change and write nothing",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"No such checkpoint: {args.checkpoint}")

    speakers = resolve_speaker_count(args)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Both shapes are in circulation: a training checkpoint wraps the weights
    # under "model", an extracted one is the state dict itself.
    state_dict = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    if EMB_KEY not in state_dict:
        sys.exit(
            f"'{args.checkpoint.name}' has no {EMB_KEY}.  Discriminator "
            "checkpoints have no speaker table; only the generator needs this."
        )

    current, dim = state_dict[EMB_KEY].shape
    print(f"{args.checkpoint.name}: {current} speakers x {dim} channels -> {speakers}")

    if speakers == current:
        sys.exit(
            "Already the right width.  verify_spk_dim will keep this table as "
            "it is; nothing to do."
        )
    if speakers < current:
        sys.exit(
            f"Refusing to shrink {current} rows to {speakers}: the rows that "
            "would be dropped are trained speakers, and which ones to drop is "
            "not something the shape can tell.  Reorder the dataset folders so "
            "the speakers you keep hold the low indices, or let the loader "
            "reset the table by pointing the run at this checkpoint unchanged."
        )

    added = widen(state_dict, speakers, args.init, args.seed)

    # The EMA shadow is a parallel copy of every weight, so it has to grow with
    # the same rows -- ``WeightEMA.load_state_dict`` copies key by key and would
    # raise on the width mismatch.  Only a resume reads it, but a checkpoint
    # that is internally inconsistent is a trap for whoever finds it later.
    ema = blob.get("ema") if isinstance(blob, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        shadow = ema["shadow"]
        if EMB_KEY in shadow:
            shadow[EMB_KEY] = torch.cat([shadow[EMB_KEY], added.clone()], dim=0)
            print(f"EMA shadow widened to {shadow[EMB_KEY].shape[0]} speakers.")

    if args.strip_optimizer and isinstance(blob, dict):
        blob.pop("optimizer", None)
        print("Optimizer state dropped.")

    kept = state_dict[EMB_KEY][:current]
    print(
        f"Init '{args.init}': {added.shape[0]} new rows, "
        f"rms {float(added.float().pow(2).mean().sqrt()):.3f} against "
        f"{float(kept.float().pow(2).mean().sqrt()):.3f} for the trained ones."
    )
    print(
        "Mean pairwise distance: "
        f"trained {mean_pairwise_distance(kept):.1f}, "
        f"new {mean_pairwise_distance(added):.1f}."
    )

    if args.dry_run:
        print("Dry run: nothing written.")
        return

    out = args.out
    if out is None:
        # Same filename, new folder: the fine-tuning branch parses the starting
        # global_step out of "G_<step>.pth", so the name has to survive.
        out = args.checkpoint.parent / f"widened_{speakers}spk" / args.checkpoint.name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    print(f"Written: {out}")
    print(
        "Point the new run's pretrained G at this file (and its D at the "
        "matching D_*.pth, untouched).  verify_spk_dim will now see matching "
        "counts and keep the table."
    )


if __name__ == "__main__":
    main()
