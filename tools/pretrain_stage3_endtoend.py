"""Stage 3 of a staged pretrain: end-to-end, at a reduced decoder rate.

WHAT THIS RUNS
--------------
Everything (``freeze_mode="none"``), with ``dec`` on a fraction of the
frontend's learning rate (``--dec-lr-scale``, default 0.2) and ``c_kl`` back at
the config's full weight.

WHY THERE HAS TO BE A STAGE 3
-----------------------------
After stage 2 the decoder has only ever been handed ``z`` from the *posterior*
``enc_q(spec)``.  At inference it gets ``z`` from the prior, through the flow
run in reverse -- a distribution it has had no gradient about.  That difference
is the inference-time quality gap, and nothing in stages 1 or 2 can close it,
because in both of them one side of the pair is held.

So this stage lets both move.  The decoder is not frozen, but it is not free
either: it arrives trained, and at the frontend's full rate it would be pulled
apart in the first epochs while the latent is still settling.  A tenth to a
third of the rate lets it adapt without losing what stage 1 bought.

WHEN TO USE IT
--------------
After stage 2 has converged -- ``loss_kl`` flattened, ``diag/prior_gap_mel_l1``
stopped falling -- on the SAME ``logs/<model>/`` folder.  This is also the
longest of the three: stages 1 and 2 are setup, this is the run that produces
the model.

It refuses to start without a checkpoint pair in the folder, and warns when the
folder's last launch was not stage 2, because starting here from a fresh init
is just a normal training run with the decoder's LR needlessly cut.

``--total-epochs`` IS CUMULATIVE
--------------------------------
As in stage 2: the trainer counts epochs from the checkpoint it resumed.  If
stage 1 ran 40 and stage 2 ran to 70, a 200-epoch stage 3 is
``--total-epochs 270``.  The script refuses a target at or below the
checkpoint's epoch rather than letting the run finish immediately.

THE LEARNING RATE IS RE-ANCHORED BY DEFAULT
-------------------------------------------
Editing ``learning_rate_g``/``learning_rate_d`` in ``config.json`` and resuming
does nothing: the checkpoint's optimizer state carries the old rate, and it is
what gets loaded.  ``resume_lr`` is the only override that reaches a resumed
run, so this stage sets it for you, to the config's ``learning_rate_g`` -- a
checkpoint arriving here has decayed through two stages and would otherwise
start the longest run of the three with almost no rate left.

``--resume-lr`` names a different base; ``--keep-checkpoint-lr`` turns the
re-anchor off.  Each parameter group keeps its own ``lr_scale``, so the base
set here and ``--dec-lr-scale`` compose rather than override each other.

USAGE
-----
Everything this stage recommends is already on -- FP16, TF32, cuDNN
benchmarking, weight EMA, the overtrain detector, the decoder at 0.2x and the
LR re-anchor -- so the command is the folder and the epoch target:

    python tools/pretrain_stage3_endtoend.py --model vctk-vocoder \\
        --total-epochs 270

    # the decoder needs more room than the default 0.2
    python tools/pretrain_stage3_endtoend.py --model vctk-vocoder \\
        --total-epochs 270 --dec-lr-scale 0.3

    # an 8 GB card that still will not fit
    python tools/pretrain_stage3_endtoend.py --model vctk-vocoder \\
        --total-epochs 270 --batch-size 4 --checkpointing

AFTERWARDS
----------
This stage leaves ``G_<step>.pth`` / ``D_<step>.pth`` (resume checkpoints) and
``<model>_<epoch>e_<step>s.pth`` (fp16 inference exports with ``enc_q``
dropped).  Neither is a pretrain.  Run ``tools/clean_pretrain.py`` on the
``G``/``D`` pair to get one -- fp32, ``enc_q`` kept, counters and optimizer
state stripped, architecture guards preserved.

WHAT TO LOOK AT
---------------
``diag/prior_gap_mel_l1`` is the one that says whether the stage is doing its
job: it is the gap this stage exists to close. ``loss_spectral`` should improve
again (it was flat through stage 2, with the decoder held).  If the decoder's
render degrades early on, ``--dec-lr-scale`` is too high.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ``python tools/<script>.py`` puts ``tools/`` on ``sys.path``, not the repo
# root, so the shared helper has to be reachable as a package first.
now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

import torch

from tools._staged_pretrain import (
    StageError,
    add_common_arguments,
    build_and_launch,
    check_dataset,
    find_checkpoints,
    model_dir,
    read_config,
    resolve_vocoder,
)


def checkpoint_epoch(path) -> int:
    """The epoch stored in a ``G_*.pth``, or 0 if it carries none."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return 0
    return int(payload.get("iteration", 0) or 0)


def previous_freeze_mode(directory) -> str | None:
    """What the folder's last launch held, from its ``run_spec.json``."""
    spec_path = directory / "run_spec.json"
    if not spec_path.is_file():
        return None
    try:
        return json.loads(spec_path.read_text(encoding="utf-8")).get("freeze_mode")
    except (json.JSONDecodeError, OSError):
        return None


#: Where stage 3 differs from ``COMMON_DEFAULTS``.  This is the long run of
#: the three -- stages 1 and 2 are setup -- so checkpoints are less frequent.
STAGE_DEFAULTS = {
    "save_every": 10,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: end-to-end, with the decoder on a reduced LR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser, STAGE_DEFAULTS)
    parser.add_argument(
        "--dec-lr-scale",
        type=float,
        default=0.2,
        help=(
            "Decoder LR as a fraction of the base (default 0.2). 1.0 is a "
            "plain end-to-end run; below ~0.1 the decoder barely adapts."
        ),
    )
    parser.add_argument(
        "--vae-lr-scale",
        type=float,
        default=1.0,
        help="Frontend and emb_g LR as a fraction of the base (default 1.0).",
    )
    parser.add_argument(
        "--resume-lr",
        type=float,
        default=None,
        help=(
            "Base LR to re-anchor to after loading the checkpoint's optimizer "
            "state. Defaults to the config's learning_rate_g, because a "
            "checkpoint arriving here has already decayed through two stages "
            "and would start stage 3 with almost no rate left. The only "
            "override that reaches a resumed run."
        ),
    )
    parser.add_argument(
        "--keep-checkpoint-lr",
        action="store_true",
        help=(
            "Do not re-anchor: continue on whatever LR the checkpoint's "
            "optimizer state carries."
        ),
    )
    parser.add_argument(
        "--resume-lr-target",
        default="full",
        choices=("full", "g", "d"),
        help="Which optimizer --resume-lr applies to (default full).",
    )
    parser.add_argument(
        "--c-kl-scale",
        type=float,
        default=1.0,
        help="Multiplier on the config's c_kl (default 1.0, the full weight).",
    )
    parser.add_argument(
        "--overtrain-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Held-out overtrain detector (default: on for this stage). Unlike "
            "stages 1 and 2, this run IS optimising conversion, so the "
            "detector is measuring the objective."
        ),
    )
    parser.add_argument(
        "--stop-on-overtrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop the run when the detector fires, rather than only logging "
            "(default: off -- the detector's own reading is worth seeing "
            "before it is allowed to end a long run)."
        ),
    )
    parser.add_argument(
        "--allow-no-checkpoint",
        action="store_true",
        help=(
            "Start stage 3 with nothing in the folder. That is just a normal "
            "end-to-end run with the decoder's LR cut for no reason."
        ),
    )
    args = parser.parse_args()

    directory = model_dir(args.model)
    if not directory.is_dir():
        raise StageError(f"{directory} does not exist.")

    config = read_config(directory)
    dataset = check_dataset(directory)
    checkpoints = find_checkpoints(directory)
    vocoder = resolve_vocoder(directory, args.vocoder)

    if not checkpoints and not args.allow_no_checkpoint:
        raise StageError(
            f"No G_*.pth / D_*.pth pair in {directory}.\n"
            f"Stage 3 continues a staged pretrain -- it has nothing to "
            f"continue from. Run stages 1 and 2 on this folder first, or pass "
            f"--allow-no-checkpoint to start end-to-end from scratch (in "
            f"which case --dec-lr-scale is only slowing the decoder down)."
        )

    print("Stage 3 -- end-to-end (decoder on a reduced rate)")
    print(f"  dir       {directory}")
    print(f"  data      {dataset['slices']:,} slices, "
          f"{dataset['speakers']} speaker(s)")
    print(f"  rate      {config['data']['sample_rate']} Hz")
    print(f"  vocoder   {vocoder}")
    print("  training  everything; nothing held")

    previous = previous_freeze_mode(directory)
    if checkpoints and previous != "encoders":
        print(
            f"  warning   this folder's last launch was freeze_mode="
            f"{previous!r}, not 'encoders'. Stage 3 is meant to follow stage "
            f"2; starting it after stage 1 leaves the frontend untrained and "
            f"the reduced decoder LR working against you."
        )

    if checkpoints:
        latest = max(checkpoints)
        epoch = checkpoint_epoch(checkpoints[latest]["G"])
        print(f"  resume    G_{latest}.pth / D_{latest}.pth (epoch {epoch})")
        if epoch and args.total_epochs <= epoch:
            raise StageError(
                f"--total-epochs {args.total_epochs} is not above the "
                f"checkpoint's epoch {epoch}, so the run would finish "
                f"immediately. The target is cumulative: pass {epoch} + "
                f"however many epochs stage 3 should last."
            )

    # ``None`` here means "not asked for", which this stage answers with the
    # config's LR rather than with the checkpoint's decayed one.  Editing
    # config.json and resuming is otherwise a silent no-op: the saved
    # optimizer state carries the old rate and is what gets loaded.
    if args.keep_checkpoint_lr:
        resume_lr = None
        print("  lr        keeping the checkpoint's optimizer LR")
    else:
        resume_lr = (
            args.resume_lr
            if args.resume_lr is not None
            else float(config["train"]["learning_rate_g"])
        )

    return build_and_launch(
        args,
        directory,
        freeze_mode="none",
        c_kl_scale=args.c_kl_scale,
        dec_lr_scale=args.dec_lr_scale,
        vae_lr_scale=args.vae_lr_scale,
        resume_lr=resume_lr,
        resume_lr_target=args.resume_lr_target,
        overtrain_detector=args.overtrain_detector,
        stop_on_overtrain=args.stop_on_overtrain,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as error:
        print(f"\n[stage 3] {error}\n", file=sys.stderr)
        sys.exit(1)
