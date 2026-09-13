"""Stage 1 of a staged pretrain: train the vocoder as a spectral autoencoder.

WHAT THIS RUNS
--------------
``enc_q`` + ``dec`` + ``emb_g`` + the discriminator.  ``enc_p`` and ``flow``
are held (``freeze_mode="vocoder"``), and ``c_kl`` is scaled down.

That is the RVC translation of "just the vocoder, mel to wav".  The decoder
here does not read a mel spectrogram -- it reads ``z``, which comes out of the
*posterior* ``enc_q(spec)`` (``rvc/lib/algorithm/synthesizers.py``).  So the
isolated vocoder is not ``dec`` on its own, it is the autoencoder
``spec -> enc_q -> z -> dec -> audio``.  Freezing ``enc_q`` as well (which is
what the older ``freeze_vae`` flag does) only makes sense when the checkpoint
already has a trained posterior; from a random init it hands the decoder noise
and the run measures nothing.

WHEN TO USE IT
--------------
Two situations.

1. *As a diagnostic.*  When a render shows artefacts -- mirrored partials,
   aliasing, a step at a band edge -- and you cannot tell whether the decoder
   is producing them or the frontend is feeding it something odd.  In this
   stage the target is exact reconstruction of the input audio, so a spectral
   defect that survives here is the decoder's, and no amount of encoder
   training will remove it.  A few thousand steps on clean multi-speaker
   material (VCTK and similar) is enough to see it.

2. *As the first half of a pretrain.*  A decoder that already renders cleanly
   from a real ``z`` is a much better starting point for the end-to-end run
   than a random one, and it means the adversarial game is not being played at
   the same time as the latent is being formed.

Do NOT use it to fine-tune a finished model onto new data -- that is
``freeze_vae`` / ``freeze_mode="frontend"``, the opposite hold.

WHY ``c_kl`` IS LOW AND NOT ZERO
--------------------------------
The point of this stage is reconstruction, so the KL should not be dragging
the latent around.  But at exactly 0 the posterior is unconstrained and its
scale drifts, which leaves stage 2 trying to match a target that moved.  The
default (0.1x the config's ``c_kl``) leaves just enough pressure to keep the
latent in a sane range.  Watch ``posterior_std_fast`` in TensorBoard: if it
climbs steadily, raise the scale.

USAGE
-----
The dataset must already be preprocessed and extracted -- this script does not
create ``logs/<model>/``, it reads the ``filelist.txt``, ``sliced_audios/``,
``extracted/`` and ``f0/`` that preprocess + extract leave behind.

``--vocoder`` is required on the first launch of a folder and is remembered
afterwards.  Nothing is defaulted: ``config.json`` records the VITS frontend's
shape and the upsample schedule, which both decoders share, so there is no
field to infer it from -- and picking one silently is how a folder ends up with
the wrong decoder loaded non-strictly against a trained discriminator.

    python tools/pretrain_stage1_vocoder.py --model vctk-vocoder \\
        --vocoder refinegan2 --total-epochs 40

    # diagnostic pass: short, and check the spectrum of a long render after
    python tools/pretrain_stage1_vocoder.py --model vctk-vocoder \\
        --vocoder refinegan2 --total-epochs 4

    # continue an interrupted stage 1: same command, it resumes from G_*.pth
    # start from an existing pretrain instead of scratch:
    python tools/pretrain_stage1_vocoder.py --model vctk-vocoder \\
        --total-epochs 40 --pretrain-g rvc/models/pretraineds/G.pth \\
        --pretrain-d rvc/models/pretraineds/D.pth

DEFAULTS
--------
FP16, TF32, cuDNN benchmarking and the weight EMA are on by default in all
three stages -- ``--no-fp16`` and friends turn any of them off.  FP16 buys no
throughput here (the step is dispatch-bound, not kernel-bound) but takes
18-43% off peak VRAM, which is what makes batch 8 fit an 8 GB card.  Gradient
checkpointing and ``torch.compile`` on the decoder stay off: both cost speed on
a step that is already CPU-bound.  The overtrain detector is off in this stage
and in stage 2 -- it scores held-out *conversion*, which neither optimises --
and on in stage 3.  See ``COMMON_DEFAULTS`` in ``tools/_staged_pretrain.py``.

WHAT TO LOOK AT
---------------
``loss_spectral`` is the objective here; ``loss_kl`` is a background term and
should stay small and flat.

The loop's previews are rendered through the posterior in this stage --
``enc_q(spec) -> dec``, unsliced -- so they show the reconstruction the stage
is actually judged on.  They used to come from ``infer``, which reads the
``enc_p`` and ``flow`` this stage *freezes*: from a scratch pretrain that is an
untrained prior's draw, and the mottle it leaves between the harmonics is the
prior's rather than the decoder's.  See ``eval_reconstruct`` in
``rvc/train/train.py``.  Stages 2 and 3 train the prior and keep the ``infer``
render, which is what conversion will run.

They decode ``m_q``.  Set ``preview_posterior_noise_scale`` in ``config.json``
to render the posterior *draw* instead: it is an independent sample per frame,
and this decoder puts it in the picture as broadband flutter, so a preview at
1.0 shows that on top of whatever the decoder is doing.

What a preview still cannot settle: anything that depends on *render length*
beyond the reference clip, and whether an inharmonic line is a fold.  Render a
long continuous file and look at that -- ``tools/render_constant_f0.py``,
``tools/probe_mirror_fold.py``, and ``tools/source_gain_ab.py`` when
``refinegan2_source_gain`` is on.

When it is done, go to ``tools/pretrain_stage2_encoders.py``.
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


#: Where stage 1 differs from ``COMMON_DEFAULTS``.  Checkpoints are frequent
#: because this stage is often run as a short diagnostic, where the point is
#: to render from it and look at the spectrum rather than to train it out.
STAGE_DEFAULTS = {
    "save_every": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: train enc_q + dec as a spectral autoencoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser, STAGE_DEFAULTS)
    parser.add_argument(
        "--c-kl-scale",
        type=float,
        default=0.1,
        help=(
            "Multiplier on the config's c_kl for this stage (default 0.1). "
            "Not 0: an unconstrained posterior drifts in scale."
        ),
    )
    parser.add_argument(
        "--pretrain-g",
        default="",
        help="Optional generator pretrain to start from instead of scratch.",
    )
    parser.add_argument("--pretrain-d", default="")
    args = parser.parse_args()

    directory = model_dir(args.model)
    if not directory.is_dir():
        raise StageError(
            f"{directory} does not exist. Run preprocess + extract for this "
            f"model first."
        )

    config = read_config(directory)
    dataset = check_dataset(directory)
    checkpoints = find_checkpoints(directory)
    # Resolved here as well as inside ``build_and_launch`` so an unknown or
    # conflicting vocoder is reported before the summary rather than after it.
    vocoder = resolve_vocoder(directory, args.vocoder)

    print("Stage 1 -- vocoder (spectral autoencoder)")
    print(f"  dir       {directory}")
    print(f"  data      {dataset['slices']:,} slices, "
          f"{dataset['speakers']} speaker(s)")
    print(f"  rate      {config['data']['sample_rate']} Hz")
    print(f"  vocoder   {vocoder}")
    print(f"  training  enc_q, dec, emb_g, discriminator")
    print(f"  held      enc_p, flow")

    if checkpoints:
        latest = max(checkpoints)
        print(f"  resume    G_{latest}.pth / D_{latest}.pth already here; the "
              f"trainer will continue from them")
        if args.pretrain_g or args.pretrain_d:
            print(
                "  note      --pretrain-* is ignored once a checkpoint pair "
                "exists in the folder; delete them to start over"
            )

    return build_and_launch(
        args,
        directory,
        freeze_mode="vocoder",
        c_kl_scale=args.c_kl_scale,
        pretrain_g=args.pretrain_g,
        pretrain_d=args.pretrain_d,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as error:
        print(f"\n[stage 1] {error}\n", file=sys.stderr)
        sys.exit(1)
