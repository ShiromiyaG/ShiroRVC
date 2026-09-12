"""Stage 2 of a staged pretrain: train the encoders against a held decoder.

WHAT THIS RUNS
--------------
``enc_p`` + ``flow`` + ``enc_q`` + ``emb_g`` + the discriminator.  ``dec`` is
held (``freeze_mode="encoders"``) at the weights stage 1 produced, and ``c_kl``
goes back to the config's full value.

A frozen module still passes gradient *through* itself, so holding ``dec`` does
not make the spectral and adversarial terms inert -- they keep training the
frontend, through the decoder, towards a ``z`` that this decoder renders well.
That is the whole point of the stage: the decoder stops chasing a moving latent
and the latent starts targeting a fixed decoder.

``enc_q`` is deliberately left trainable.  Holding it too would leave the KL as
the only live loss (nothing upstream of a frozen ``dec`` <- frozen ``enc_q``
path would be learning from the audio), and the discriminator would then be
training against a generator whose output cannot change.

WHEN TO USE IT
--------------
After stage 1, on the same ``logs/<model>/`` folder, once the decoder renders
cleanly from a real ``z``.  If stage 1 still shows the artefact you were
chasing, do not come here -- fix the decoder first, because this stage cannot
change it.

Run it in the SAME model directory as stage 1.  It resumes from the ``G_*.pth``
/ ``D_*.pth`` that stage 1 left, which is what carries the trained decoder
across.  ``load_checkpoint`` re-maps the optimizer state when the frozen set
changes, so the switch from stage 1's hold to this one is handled.

``--total-epochs`` IS CUMULATIVE
--------------------------------
The trainer counts epochs from the checkpoint it resumed, not from zero.  If
stage 1 ran 40 epochs and you want 30 more here, pass ``--total-epochs 70``.
Passing 30 would make the run finish immediately.  The script checks the
checkpoint's epoch and refuses a target below it.

WHAT COMES OUT -- AND WHAT IT IS NOT
------------------------------------
Stage 2 does not leave a usable pretrain behind.  Two separate reasons.

*Mechanically*, the files this writes are ``G_<step>.pth`` / ``D_<step>.pth``
(resume checkpoints: weights plus optimizer moments, step counter and AMP
state) and ``<model>_<epoch>e_<step>s.pth`` (an fp16 *inference* export with
``enc_q`` dropped).  A pretrain is neither: it loads with ``strict=True`` into
a full training model, so the inference export fails outright on the missing
posterior, and the resume checkpoint carries the finished run's counters into
whatever consumes it.  ``tools/clean_pretrain.py`` is what converts a
``G_*.pth``/``D_*.pth`` pair into a pretrain -- fp32, ``enc_q`` kept, counters
and optimizer state dropped, architecture guards preserved.

*Substantively*, the model is not finished.  The decoder has only ever seen
``z`` from the posterior; at inference it gets ``z`` from the prior through the
reversed flow, and it has had no gradient about that distribution.  That gap is
what stage 3 closes, and shipping a pretrain from here hands every downstream
finetune a decoder that has never been asked to render what it will actually
receive.

So: stage 2 -> stage 3 -> ``tools/clean_pretrain.py``.

WHEN TO MOVE ON
---------------
When ``loss_kl`` has flattened and ``diag/prior_gap_mel_l1`` has stopped
falling, run ``tools/pretrain_stage3_endtoend.py`` on the same folder.

USAGE
-----
    python tools/pretrain_stage2_encoders.py --model vctk-vocoder \\
        --total-epochs 70

    # look at what it would launch, without launching:
    python tools/pretrain_stage2_encoders.py --model vctk-vocoder \\
        --total-epochs 70 --dry-run

WHAT TO LOOK AT
---------------
``loss_kl`` is the objective here, and it should fall.  The diagnostic that
matters more is ``diag/prior_gap_mel_l1``: the difference between decoding the
posterior and decoding the prior, which is exactly the inference-time gap this
stage exists to close.  ``loss_spectral`` will be roughly flat -- the decoder
cannot improve -- and that is expected, not a stall.
"""

from __future__ import annotations

import argparse
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


#: Stage 2 takes ``COMMON_DEFAULTS`` unchanged.  Named anyway so the three
#: launchers read the same way and a future difference has somewhere to go.
STAGE_DEFAULTS: dict = {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: train enc_p + flow against a frozen decoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser, STAGE_DEFAULTS)
    parser.add_argument(
        "--c-kl-scale",
        type=float,
        default=1.0,
        help="Multiplier on the config's c_kl (default 1.0, the full weight).",
    )
    parser.add_argument(
        "--allow-no-checkpoint",
        action="store_true",
        help=(
            "Start stage 2 with no stage-1 checkpoint in the folder. Almost "
            "always a mistake: it freezes a randomly initialised decoder."
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
            f"Stage 2 freezes the decoder, so it needs the decoder stage 1 "
            f"trained. Run tools/pretrain_stage1_vocoder.py on this folder "
            f"first, or pass --allow-no-checkpoint if you really mean to "
            f"freeze a random decoder.\n"
            f"Note that the *_<epoch>e_<step>s.pth files are inference "
            f"exports with enc_q stripped; they cannot start a stage."
        )

    print("Stage 2 -- encoders (frontend against a held decoder)")
    print(f"  dir       {directory}")
    print(f"  data      {dataset['slices']:,} slices, "
          f"{dataset['speakers']} speaker(s)")
    print(f"  rate      {config['data']['sample_rate']} Hz")
    print(f"  vocoder   {vocoder}")
    print(f"  training  enc_p, flow, enc_q, emb_g, discriminator")
    print("  held      dec")

    if checkpoints:
        latest = max(checkpoints)
        epoch = checkpoint_epoch(checkpoints[latest]["G"])
        print(f"  resume    G_{latest}.pth / D_{latest}.pth (epoch {epoch})")
        if epoch and args.total_epochs <= epoch:
            raise StageError(
                f"--total-epochs {args.total_epochs} is not above the "
                f"checkpoint's epoch {epoch}, so the run would finish "
                f"immediately. The target is cumulative: pass "
                f"{epoch} + however many epochs stage 2 should last."
            )

    return build_and_launch(
        args,
        directory,
        freeze_mode="encoders",
        c_kl_scale=args.c_kl_scale,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as error:
        print(f"\n[stage 2] {error}\n", file=sys.stderr)
        sys.exit(1)
