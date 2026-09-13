"""Retarget a trained generator from one embedder to another.

The embedder reaches the synthesizer through exactly one weight.
``TextEncoder.__init__`` builds ``self.emb_phone = nn.Linear(embedding_dim,
hidden_channels)`` and nothing else in the network is a function of
``embedding_dim`` -- not the transformer encoder above it, not the flow, not the
posterior encoder, not the decoder, and not any discriminator.  So a model
trained against ``spin_v1`` (256-wide) differs from a ``contentvec`` one
(768-wide) by a single ``[192, 256]`` vs ``[192, 768]`` tensor out of the whole
generator, and swapping embedders is a question of what to put in that tensor
rather than a question of retraining.

Putting a *random* tensor there works -- finetuning will eventually fit it --
but it throws away the one thing that is actually recoverable for free.  The
rest of the network learned to consume the specific 192-d vectors the old
``emb_phone`` produced; if the new embedder's features can predict those same
vectors, the network never has to move.  Since ``emb_phone`` is linear, "can
predict" is a least-squares question with a closed-form answer, so this solves
it directly instead of training:

    minimise over (W, b)   || X_tgt @ W.T + b  -  (X_src @ W_src.T + b_src) ||^2

on frames from real audio, both embedders reading the same 16 kHz file.  That
is a ridge regression, computed here from accumulated normal equations in
float64, which is a few minutes over hours of audio and needs no GPU memory
beyond one clip.

What this cannot do is invent information.  SPIN WavLM is 256-d, L2-normalised
and trained to *discard* speaker identity; ContentVec is 768-d, unnormalised,
and keeps some.  The fit recovers whatever the target embedder carries linearly
of what the source carried, and the reported R^2 is the honest measure of how
much that is.  Read it before trusting the output: a high R^2 means the
finetune starts near where the pretrain left off, and a low one means this is a
warm random init and should be treated as such.

The output is a **pretrain**, to be passed as ``pretrainG`` -- not a resume
checkpoint.  The optimizer moments are shaped for the old width and cannot be
carried across, so they are dropped rather than left in the file to fail a
resume later.

Run ``tools/clean_pretrain.py`` on the checkpoint *first*, and point this script
at its output rather than at a raw ``G_<step>.pth``.  The order is not
cosmetic.  The weights read here are ``checkpoint["model"]``, which in a
training checkpoint is the live last step; the EMA average lives under ``ema``
and is retargeted below only so a resume cannot load a stale width, never
promoted into ``model``.  Since the holdout is scored under the EMA
(``train.py``, ``with ema.applied(net_g)``), converting a raw checkpoint builds
the new pretrain on the one set of weights nothing ever measured.
``clean_pretrain`` folds the shadow into ``model``, so converting its output
retargets the averaged weights instead.  It also drops the step and epoch
counters, which this script keeps -- harmless, since the pretrain load path
reads neither, but they are the run's history and do not belong in its
successor.

Only the generator goes through here.  Nothing in the discriminator is a
function of ``embedding_dim``, so the cleaned ``*_D.pth`` is used as-is; there
is no ``--checkpoint`` form that takes one.

Name ``--output`` anything that does not match ``G_<digits>.pth``.
``checkpoint_step_from_path`` reads that pattern off the *filename* to seed
``global_step`` on a finetune, so a pretrain still carrying the old run's step
in its name starts its finetune that far into the LR schedule -- which, with a
horizon-derived decay, can mean the whole finetune runs pinned at the floor LR.

And re-extract the dataset with the target embedder before finetuning.  The
retargeted layer consumes the target's features; against a dataset still
extracted with the source embedder the widths do not even match.

Usage:

    python tools/clean_pretrain.py --model-dir logs/pretrain \\
        --output-dir logs/pretrains

    python tools/convert_embedder.py \\
        --checkpoint logs/pretrains/pretrain_G.pth \\
        --output logs/pretrains/pretrain_contentvec_G.pth \\
        --source-embedder spin_v1 \\
        --target-embedder contentvec \\
        --audio-dir logs/pretrain/sliced_audios
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.audio_io import load_audio_16k
from rvc.lib.terminal import (
    error as print_error,
    info,
    progress_task,
    success,
    warning,
)
from rvc.lib.utils import (
    EMBEDDER_FEATURE_DIMS,
    embedder_feature_dim,
    extract_features,
    load_embedder_model,
)


#: The one weight that changes width with the embedder.
EMB_PHONE_WEIGHT = "enc_p.emb_phone.weight"
EMB_PHONE_BIAS = "enc_p.emb_phone.bias"

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")

#: Clips shorter than this contribute almost no frames while costing a full
#: model load-and-forward, and very short ones are mostly silence padding.
MIN_SECONDS = 0.5


def collect_audio(audio_dir, limit, seed):
    """A deterministic random sample of the clips under ``audio_dir``."""

    files = []
    for root, _, names in os.walk(audio_dir):
        for name in sorted(names):
            if name.lower().endswith(AUDIO_SUFFIXES):
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No audio found under {audio_dir!r}.")
    # Sampled rather than truncated: the sliced audio is named by speaker, so
    # taking the first N files would fit the map to whoever sorts first.
    random.Random(seed).shuffle(files)
    return files[:limit] if limit > 0 else files


def build_embedder(name, device):
    model, do_normalize = load_embedder_model(name)
    model = model.to(device).float().eval()
    return model, do_normalize


@torch.inference_mode()
def frame_features(model, do_normalize, audio, device):
    """``[T, D]`` features for one clip, in the layout the extractor writes."""

    feats = torch.from_numpy(audio).to(device).float().view(1, -1)
    # fp32 throughout, unlike the extractor's autocast: this is a few thousand
    # clips, not the whole dataset, and the normal equations are solved in
    # float64 -- there is no reason to feed them fp16 rounding.
    out = extract_features(model, feats, "v2", do_normalize=do_normalize)
    return out.squeeze(0).float()


def solve_ridge(gram, moment, ridge):
    """``(A^T A + lambda I)^-1 A^T t``, with the bias column left unpenalised.

    Penalising the intercept would shrink the output toward zero rather than
    toward the target's mean, which is not what the regulariser is for.
    """

    size = gram.shape[0]
    penalty = torch.eye(size, dtype=gram.dtype) * ridge
    penalty[-1, -1] = 0.0
    return torch.linalg.solve(gram + penalty, moment)


def main():
    parser = argparse.ArgumentParser(
        description="Retarget a trained generator to a different embedder.",
    )
    parser.add_argument("--checkpoint", required=True, help="Generator .pth to convert.")
    parser.add_argument("--output", required=True, help="Where to write the retargeted .pth.")
    parser.add_argument(
        "--source-embedder",
        required=True,
        help=f"Embedder the checkpoint was trained with ({', '.join(EMBEDDER_FEATURE_DIMS)}).",
    )
    parser.add_argument("--target-embedder", required=True, help="Embedder to retarget onto.")
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Clips to fit on, resampled to 16 kHz as they are read; the run's 'sliced_audios' is the right input.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2000,
        help="Clips to sample (0 = all). 2000 slices is already ~1e6 frames.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty, relative to the feature scale. Raise it if the solve is ill-conditioned.",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.1,
        help="Fraction of clips scored but not fitted, so the reported R^2 is not the training fit.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output.")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print_error(f"{args.output} exists; pass --overwrite to replace it.", tag="[RETARGET]")
        sys.exit(1)

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Whatever is in ``model`` is what gets retargeted -- the live last step in a
    # raw training checkpoint, the EMA average in one that went through
    # ``clean_pretrain`` first.  The holdout is scored under the EMA, so the
    # cleaned file is the input worth converting; see the module docstring.
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if EMB_PHONE_WEIGHT not in state_dict:
        print_error(
            f"{args.checkpoint} has no {EMB_PHONE_WEIGHT!r}; this is not a generator checkpoint.",
            tag="[RETARGET]",
        )
        sys.exit(1)

    weight_src = state_dict[EMB_PHONE_WEIGHT].float()
    bias_src = state_dict[EMB_PHONE_BIAS].float()
    hidden_channels, dim_src_ckpt = weight_src.shape

    dim_src = embedder_feature_dim(args.source_embedder)
    dim_tgt = embedder_feature_dim(args.target_embedder)
    # The checkpoint is the authority on what it was trained with; --source is
    # what the user believes.  Disagreement means the fit would be built from
    # the wrong old projection, which produces a plausible file that is silently
    # wrong, so it stops here.
    if dim_src != dim_src_ckpt:
        print_error(
            f"{args.checkpoint} carries a {dim_src_ckpt}-wide emb_phone, but "
            f"--source-embedder {args.source_embedder!r} is {dim_src} wide.",
            tag="[RETARGET]",
        )
        sys.exit(1)
    if dim_src == dim_tgt and args.source_embedder == args.target_embedder:
        print_error("Source and target embedder are the same; nothing to do.", tag="[RETARGET]")
        sys.exit(1)

    info(
        f"{args.source_embedder} ({dim_src}) -> {args.target_embedder} ({dim_tgt}), "
        f"emb_phone [{hidden_channels}, {dim_src}] -> [{hidden_channels}, {dim_tgt}]",
        tag="[RETARGET]",
    )

    files = collect_audio(args.audio_dir, args.max_files, args.seed)
    split = int(len(files) * (1.0 - max(0.0, min(0.9, args.holdout))))
    fit_files, score_files = files[:split], files[split:]
    if not fit_files:
        print_error("No clips left to fit on; lower --holdout.", tag="[RETARGET]")
        sys.exit(1)
    info(f"{len(fit_files)} clips to fit, {len(score_files)} held out.", tag="[RETARGET]")

    model_src, norm_src = build_embedder(args.source_embedder, device)
    model_tgt, norm_tgt = build_embedder(args.target_embedder, device)
    weight_src_dev = weight_src.to(device)
    bias_src_dev = bias_src.to(device)

    def paired_frames(path):
        """``(features_tgt, target_192d)`` for one clip, or ``None`` if unusable."""

        audio = load_audio_16k(path)
        if audio.shape[0] < int(MIN_SECONDS * 16000):
            return None
        feats_src = frame_features(model_src, norm_src, audio, device)
        feats_tgt = frame_features(model_tgt, norm_tgt, audio, device)
        # Both embedders stride the same 16 kHz waveform at 20 ms, but a
        # convolutional frontend can differ by a frame on the tail; trimming to
        # the shorter is what the extractor's own length handling does.
        frames = min(feats_src.shape[0], feats_tgt.shape[0])
        if frames == 0:
            return None
        feats_src, feats_tgt = feats_src[:frames], feats_tgt[:frames]
        if not (torch.isfinite(feats_src).all() and torch.isfinite(feats_tgt).all()):
            return None
        # The regression target is the *old projection's output*, not the old
        # features: what the rest of the network consumes is these 192 numbers,
        # and reproducing them is the whole objective.
        target = feats_src @ weight_src_dev.T + bias_src_dev
        return feats_tgt, target

    # Normal equations rather than stacked design matrices: [769, 769] and
    # [769, 192] accumulators are a few megabytes regardless of how many hours
    # go in, and float64 keeps the Gram matrix conditioned.
    gram = torch.zeros(dim_tgt + 1, dim_tgt + 1, dtype=torch.float64)
    moment = torch.zeros(dim_tgt + 1, hidden_channels, dtype=torch.float64)
    frame_count = 0
    skipped = 0

    with progress_task(len(fit_files), "Fitting", leave=True) as (progress, task_id):
        for path in fit_files:
            pair = paired_frames(path)
            progress.advance(task_id, 1)
            if pair is None:
                skipped += 1
                continue
            feats_tgt, target = pair
            # Accumulated on the CPU in float64.  The per-clip product is a few
            # hundred frames, so it is not the bottleneck next to two embedder
            # forwards, and float64 is what keeps a [769, 769] Gram matrix
            # conditioned over a million frames.
            design = torch.cat(
                [feats_tgt, torch.ones(feats_tgt.shape[0], 1, device=device)], dim=1
            ).cpu().double()
            gram += design.T @ design
            moment += design.T @ target.cpu().double()
            frame_count += design.shape[0]

    if frame_count <= dim_tgt:
        print_error(
            f"Only {frame_count} frames for {dim_tgt + 1} unknowns; the fit would be "
            "underdetermined. Raise --max-files.",
            tag="[RETARGET]",
        )
        sys.exit(1)
    if skipped:
        warning(f"{skipped} clips skipped (too short, or non-finite features).", tag="[RETARGET]")
    info(f"Fitted on {frame_count} frames.", tag="[RETARGET]")

    # Scaled by the frame count so --ridge means the same thing whatever the
    # dataset size, and by the mean feature energy so it means the same thing
    # for an L2-normalised embedder and an unnormalised one.
    scale = float(torch.diagonal(gram)[:-1].mean())
    theta = solve_ridge(gram, moment, args.ridge * scale)
    weight_new = theta[:dim_tgt].T.contiguous().float()
    bias_new = theta[dim_tgt].contiguous().float()

    def score(paths):
        """1 - SSE/SST against the old projection's output, on ``paths``."""

        weight_dev = weight_new.to(device)
        bias_dev = bias_new.to(device)
        total, residual, mean_acc, count = 0.0, 0.0, None, 0
        squares = 0.0
        for path in paths:
            pair = paired_frames(path)
            if pair is None:
                continue
            feats_tgt, target = pair
            predicted = feats_tgt @ weight_dev.T + bias_dev
            residual += float(((predicted - target) ** 2).sum())
            squares += float((target ** 2).sum())
            summed = target.sum(dim=0).double()
            mean_acc = summed if mean_acc is None else mean_acc + summed
            count += target.shape[0]
        if count == 0:
            return None
        # SST about the target's own mean, so R^2 is measured against
        # "predict the average frame" rather than against predicting zero.
        total = squares - float((mean_acc ** 2).sum()) / count
        return 1.0 - residual / total if total > 0 else None

    r2 = score(score_files) if score_files else score(fit_files[: min(64, len(fit_files))])
    if r2 is None:
        warning("Could not score the fit; no usable held-out clips.", tag="[RETARGET]")
    else:
        report = f"R^2 on {'held-out' if score_files else 'fitted'} frames: {r2:.4f}"
        if r2 >= 0.9:
            success(f"{report} - the pretrain transfers well.", tag="[RETARGET]")
        elif r2 >= 0.6:
            info(f"{report} - usable, but budget a real finetune.", tag="[RETARGET]")
        else:
            warning(
                f"{report} - the target embedder does not linearly carry what the "
                "source did. Treat the output as a warm init, not a pretrain.",
                tag="[RETARGET]",
            )

    state_dict[EMB_PHONE_WEIGHT] = weight_new
    state_dict[EMB_PHONE_BIAS] = bias_new
    if "model" in checkpoint:
        checkpoint["model"] = state_dict
    else:
        checkpoint = state_dict

    # The EMA shadow is a second copy of every weight, and a resume seeds from
    # it.  Leaving the old width in there would fail the load with a message
    # pointing at the EMA rather than at this conversion.  Note this repairs the
    # shadow, it does not promote it: ``model`` keeps whatever weights arrived.
    ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        if EMB_PHONE_WEIGHT in ema["shadow"]:
            ema["shadow"][EMB_PHONE_WEIGHT] = weight_new.clone()
            ema["shadow"][EMB_PHONE_BIAS] = bias_new.clone()
            info("EMA shadow retargeted alongside the weights.", tag="[RETARGET]")

    # Dropped, not remapped: Adam's moments for emb_phone are shaped for the old
    # width, and the surrounding params' moments describe a gradient history
    # taken under the old frontend.  A finetune should start the optimizer fresh.
    if isinstance(checkpoint, dict) and checkpoint.pop("optimizer", None) is not None:
        info("Optimizer state dropped; use this as pretrainG, not as a resume.", tag="[RETARGET]")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(checkpoint, args.output)
    success(f"Wrote {args.output}", tag="[RETARGET]")
    info(
        f"Re-extract the dataset with {args.target_embedder} before finetuning, then pass "
        "this file as the pretrained G.",
        tag="[RETARGET]",
    )


if __name__ == "__main__":
    main()
