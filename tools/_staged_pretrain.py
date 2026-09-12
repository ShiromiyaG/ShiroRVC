"""Shared plumbing for the three staged-pretrain launchers.

``pretrain_stage1_vocoder.py``, ``pretrain_stage2_encoders.py`` and
``pretrain_stage3_endtoend.py`` differ only in which half of the model they
hold still, how the learning rate is split, and what they check before
starting.  Everything else -- finding the dataset, reading the config,
building the run spec, launching the trainer -- is the same, and lives here so
the stages cannot drift apart in the details that have to match (sample rate,
vocoder, batch size, the log directory all three share).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.vocoders import (  # noqa: E402
    get_vocoder_ids,
    get_vocoder_sample_rates,
    normalize_vocoder,
)
from rvc.train.run_spec import TrainRunSpec  # noqa: E402


#: ``G_<step>.pth`` / ``D_<step>.pth`` -- the resume checkpoints the loop
#: writes.  The ``*_<epoch>e_<step>s.pth`` weights next to them are *inference*
#: exports with ``enc_q`` stripped, and cannot start either stage.
CHECKPOINT_RE = re.compile(r"^([GD])_(\d+)\.pth$")


class StageError(RuntimeError):
    """A precondition the stage cannot start without."""


def model_dir(name_or_path: str) -> Path:
    """Accept either a model name or a path to its log directory."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    return Path(now_dir) / "logs" / name_or_path


def read_config(directory: Path) -> dict:
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise StageError(
            f"{config_path} is missing. The stage launchers do not write a "
            f"config -- run preprocess + extract for this model first (the "
            f"UI's Preprocess and Extract tabs, or core.py), which is what "
            f"creates config.json and filelist.txt."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def check_dataset(directory: Path) -> dict:
    """Verify the filelist and that the files it names are actually there.

    Checked here rather than left to the trainer because the failure mode
    otherwise is a dataloader that raises a few hundred steps in, after the
    models are built and the first checkpoint interval has been scheduled.
    Only the first and last entries are stat'd -- a full pass over 40k slices
    costs more than it catches, and a truncated extract shows up at the end.
    """
    filelist_path = directory / "filelist.txt"
    if not filelist_path.is_file():
        raise StageError(
            f"{filelist_path} is missing. Run preprocess + extract for this "
            f"model before either stage."
        )

    lines = [
        line
        for line in filelist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise StageError(f"{filelist_path} is empty.")

    speakers = set()
    for line in lines:
        parts = line.split("|")
        if len(parts) != 5:
            raise StageError(
                f"{filelist_path}: expected 5 pipe-separated fields "
                f"(audio|feature|f0|f0_voiced|speaker), got {len(parts)} in:\n"
                f"  {line}"
            )
        speakers.add(parts[4])

    for line in (lines[0], lines[-1]):
        for field, label in zip(
            line.split("|")[:4], ("slice", "feature", "f0", "f0_voiced")
        ):
            if not Path(field).is_file():
                raise StageError(
                    f"{filelist_path} names a {label} that does not exist:\n"
                    f"  {field}\n"
                    f"The extract step did not finish, or the folder was moved "
                    f"after it ran (the paths in the filelist are relative to "
                    f"the repository root)."
                )

    return {"slices": len(lines), "speakers": len(speakers)}


def find_checkpoints(directory: Path) -> dict:
    """``{step: {"G": path, "D": path}}`` for complete pairs only."""
    found: dict = {}
    for entry in directory.iterdir():
        match = CHECKPOINT_RE.match(entry.name)
        if match:
            found.setdefault(int(match.group(2)), {})[match.group(1)] = entry
    return {step: pair for step, pair in found.items() if set(pair) == {"G", "D"}}


def resolve_vocoder(directory: Path, requested: str | None) -> str:
    """Decide which vocoder this stage builds, and refuse to guess.

    ``config.json`` does not record it -- the ``model`` block is the VITS
    frontend's shape plus the upsample schedule, all of which both decoders
    share.  The vocoder is a per-*launch* choice that ``core.py`` writes into
    ``run_spec.json``, so that is where a folder's existing answer lives.

    Defaulting would be the failure this repository's vocoder registry exists
    to prevent: ``normalize_vocoder(None)`` resolves to HiFi-GAN, so a silent
    default would build the wrong decoder for a RefineGAN folder, load the
    checkpoint non-strictly, and train a fresh decoder against a trained
    discriminator with nothing said.  So: take the folder's previous answer,
    else make the caller state one.
    """
    previous = None
    spec_path = directory / "run_spec.json"
    if spec_path.is_file():
        try:
            previous = json.loads(spec_path.read_text(encoding="utf-8")).get(
                "vocoder"
            )
        except (json.JSONDecodeError, OSError):
            previous = None

    if requested and previous and normalize_vocoder(requested) != normalize_vocoder(
        previous
    ):
        raise StageError(
            f"--vocoder {requested!r} does not match the vocoder this folder "
            f"was last launched with ({previous!r}, from {spec_path}).\n"
            f"A folder's decoder is fixed once it has a checkpoint: the "
            f"weights of one decoder do not load into the other. Use a new "
            f"model directory for a different vocoder."
        )

    chosen = requested or previous
    if not chosen:
        raise StageError(
            f"No vocoder given and {spec_path} does not name one, so there is "
            f"nothing to infer it from -- config.json does not record it. "
            f"Pass --vocoder with one of {get_vocoder_ids()}."
        )
    return normalize_vocoder(chosen)


#: The recommended setting for every option all three stages share.  Stated
#: here rather than left to the caller so that "what should I run this with?"
#: has one answer in one place, and so a stage that wants something else has
#: to say so explicitly (see ``STAGE_DEFAULTS`` in each launcher).
#:
#: The two that are on and would not be obvious:
#:
#: ``fp16``   buys no throughput on its own -- the step is dispatch-bound, not
#:            kernel-bound -- but takes 18-43% off peak VRAM, which is what
#:            makes batch 8 fit an 8 GB card.  It was unsafe while the
#:            discriminator carried an R1 penalty, because an ``autograd.grad``
#:            probe taken outside the ``GradScaler`` produces NaNs under FP16
#:            (finite losses with a 100% skip rate is the tell).  No such probe
#:            exists in this build.
#: ``tf32``   free on Ampere and later, ignored on anything older.
#:
#: And the two that are off:
#:
#: ``checkpointing``    recomputes activations to save VRAM.  A real cost on a
#:                      step that is already CPU-bound; turn it on only when
#:                      the run will not otherwise fit.
#: ``compile_vocoder``  compiling the decoder alone measured no speedup here,
#:                      and Inductor has CPU-fallback traps that surface as
#:                      opaque build errors.  Opt-in.
COMMON_DEFAULTS = {
    "batch_size": 8,
    "save_every": 5,
    "gpu": "0",
    "optimizer": "AdamW",
    "lr_scheduler": "exp decay step",
    "fp16": True,
    "tf32": True,
    "benchmark": True,
    "ema": True,
    "checkpointing": False,
    "compile_vocoder": True,
}


def add_common_arguments(
    parser: argparse.ArgumentParser, defaults: dict | None = None
) -> None:
    """Add the options every stage shares, at this stage's recommended values.

    ``defaults`` overrides ``COMMON_DEFAULTS`` for the keys a stage disagrees
    about.  Every boolean is a ``--x`` / ``--no-x`` pair, so a default that is
    on can be turned off from the command line without the flag's name having
    to change meaning.
    """
    settings = dict(COMMON_DEFAULTS)
    settings.update(defaults or {})

    def flag(name: str, key: str, help_text: str) -> None:
        parser.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=settings[key],
            help=f"{help_text} (default: "
            f"{'on' if settings[key] else 'off'} for this stage).",
        )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name under logs/, or a path to its log directory.",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        required=True,
        help=(
            "CUMULATIVE epoch target, not this stage's length. The trainer "
            "counts epochs from the checkpoint it resumes, so stages 2 and 3 "
            "must ask for the previous stage's total plus their own."
        ),
    )
    parser.add_argument(
        "--vocoder",
        default=None,
        help=(
            "Which decoder to build. Taken from the folder's existing "
            "run_spec.json when there is one; required otherwise, because "
            "config.json does not record it and guessing builds the wrong "
            "decoder silently."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=settings["batch_size"])
    parser.add_argument(
        "--save-every",
        type=int,
        default=settings["save_every"],
        help=f"Epochs between checkpoints (default {settings['save_every']}).",
    )
    parser.add_argument("--gpu", default=settings["gpu"])
    parser.add_argument("--optimizer", default=settings["optimizer"])
    parser.add_argument("--lr-scheduler", default=settings["lr_scheduler"])
    parser.add_argument(
        "--custom-lr-g",
        type=float,
        default=None,
        help="Override the config's generator LR.",
    )
    parser.add_argument(
        "--custom-lr-d",
        type=float,
        default=None,
        help="Override the config's discriminator LR.",
    )
    flag("fp16", "fp16", "Mixed precision: ~18-43% less peak VRAM, no speedup")
    flag("tf32", "tf32", "TF32 matmuls; free on Ampere+, ignored before it")
    flag("benchmark", "benchmark", "cuDNN autotuning")
    flag("ema", "ema", "Weight EMA of the generator")
    flag(
        "checkpointing",
        "checkpointing",
        "Gradient checkpointing; trades speed for VRAM",
    )
    flag("compile-vocoder", "compile_vocoder", "torch.compile the decoder")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write run_spec.json and print the command without launching.",
    )


def build_and_launch(
    args,
    directory: Path,
    *,
    freeze_mode: str,
    c_kl_scale: float,
    pretrain_g: str = "",
    pretrain_d: str = "",
    dec_lr_scale: float | None = None,
    vae_lr_scale: float | None = None,
    resume_lr: float | None = None,
    resume_lr_target: str = "full",
    overtrain_detector: bool = False,
    stop_on_overtrain: bool = False,
) -> int:
    config = read_config(directory)
    sample_rate = int(config["data"]["sample_rate"])
    vocoder = resolve_vocoder(directory, getattr(args, "vocoder", None))
    supported = get_vocoder_sample_rates(vocoder)
    if sample_rate not in supported:
        raise StageError(
            f"{vocoder} has no configuration for {sample_rate} Hz (it "
            f"supports {supported}). The rate comes from {directory}/"
            f"config.json, so the dataset was preprocessed at a rate this "
            f"decoder does not ship -- re-run preprocess at a supported one."
        )

    spec = TrainRunSpec(
        model_name=directory.name,
        sample_rate=sample_rate,
        vocoder=vocoder,
        total_epoch_count=int(args.total_epochs),
        epoch_save_frequency=int(args.save_every),
        batch_size=int(args.batch_size),
        gpus=str(args.gpu),
        save_only_latest_net_models=False,
        save_weight_models=True,
        cleanup=False,
        pretrain_g=pretrain_g,
        pretrain_d=pretrain_d,
        optimizer_choice=str(args.optimizer),
        lr_scheduler=str(args.lr_scheduler),
        use_custom_lr=args.custom_lr_g is not None or args.custom_lr_d is not None,
        custom_lr_g=float(
            args.custom_lr_g
            if args.custom_lr_g is not None
            else config["train"]["learning_rate_g"]
        ),
        custom_lr_d=float(
            args.custom_lr_d
            if args.custom_lr_d is not None
            else config["train"]["learning_rate_d"]
        ),
        use_checkpointing=bool(args.checkpointing),
        use_tf32=bool(args.tf32),
        use_fp16=bool(args.fp16),
        use_benchmark=bool(args.benchmark),
        compile_vocoder=bool(args.compile_vocoder),
        # Off for stages 1 and 2, which are not optimising held-out
        # *conversion* -- stage 1 is judged on reconstruction, stage 2 on the
        # KL -- so the detector would stop them early on a metric that is not
        # the objective.  Stage 3 is the end-to-end run and turns it back on.
        overtrain_detector=bool(overtrain_detector),
        stop_on_overtrain=bool(stop_on_overtrain),
        use_ema=bool(args.ema),
        freeze_mode=freeze_mode,
        c_kl_scale=float(c_kl_scale),
        dec_lr_scale=dec_lr_scale,
        vae_lr_scale=vae_lr_scale,
        resume_lr=resume_lr,
        resume_lr_target=resume_lr_target,
    )
    spec_path = spec.save(directory / "run_spec.json")

    command = [
        sys.executable,
        os.path.join("rvc", "train", "train.py"),
        str(spec_path),
    ]
    print(f"\n  spec     {spec_path}")
    print(f"  stage    freeze_mode={freeze_mode}  c_kl_scale={c_kl_scale}")
    if dec_lr_scale is not None or vae_lr_scale is not None:
        print(f"  lr       dec x{dec_lr_scale}  frontend x{vae_lr_scale}")
    if resume_lr is not None:
        print(f"  lr       re-anchored to {resume_lr:.2e} ({resume_lr_target})")
    print(f"  command  {' '.join(command)}\n")

    if args.dry_run:
        print("Dry run: nothing launched.")
        return 0

    if platform.system() == "Windows":
        process = subprocess.Popen(
            command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        process = subprocess.Popen(command)
    return process.wait()
