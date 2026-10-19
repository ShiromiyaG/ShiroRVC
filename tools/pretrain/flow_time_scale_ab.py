"""Paired A/B of the flow backbone's time scale before a pretrain: whether
time scale 1, which mean flow needs from the pretrain on, costs the plain
flow anything against DiffSinger's 1000. (Mean flow itself trained stably at
scale 1 in logs/pretrain, 215k steps.)

Arms, each a fresh run on the same fixed subset of a source experiment, with
the same seed and ``stream_seed`` (identical batches, noise and times):
  ts1000     time scale 1000, plain flow (the current default)
  ts1        time scale 1, plain flow

  python tools/pretrain/flow_time_scale_ab.py prepare --source pretrain
  python tools/pretrain/flow_time_scale_ab.py run      (the arms one after the other)
  python tools/pretrain/flow_time_scale_ab.py report

Read the held-out flow loss from step 4000 on; under about 0.3% is noise.
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys

sys.path.append(os.getcwd())

from rvc.lib.paths import LOGS_DIR, ROOT

PREFIX = "ab_ts_"
ARMS = {
    "ts1000": {"time_scale": 1000.0, "mean_flow": False},
    "ts1": {"time_scale": 1.0, "mean_flow": False},
}
SEED = 1234
# The held-out loss is taken every this many steps; read from step 4000 on.
EVAL_INTERVAL = 1000


def arm_dir(arm: str) -> str:
    return os.path.join(LOGS_DIR, PREFIX + arm)


def preview_vocoder(source: str) -> str:
    """The source experiment's newest ChouwaGAN export, for the audio previews, or none."""
    paths = glob.glob(os.path.join(LOGS_DIR, source, "vocoder_chouwagan", "**", "*_vocoder_*.pth"), recursive=True)
    return os.path.relpath(max(paths, key=os.path.getmtime), ROOT) if paths else ""


def prepare(args):
    source = os.path.join(LOGS_DIR, args.source)
    with open(os.path.join(source, "filelist.txt"), encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    rows = sorted(random.Random(SEED).sample(rows, min(args.clips, len(rows))))
    with open(os.path.join(source, "rectified_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    flow = config["flow"]
    # Short runs: a 0.9999 average would still be the initialisation at step 5000.
    flow["ema_decay"] = 0.999
    flow["stream_seed"] = SEED
    # Every arm would otherwise build its own cache of augmented copies.
    flow["feature_cache"] = False
    flow["bucket_batches"] = False
    flow["eval_interval"] = EVAL_INTERVAL
    flow["preview_interval"] = 10**9
    vocoder = preview_vocoder(args.source)
    for arm, settings in ARMS.items():
        folder = arm_dir(arm)
        os.makedirs(os.path.join(folder, "flow"), exist_ok=True)
        with open(os.path.join(folder, "filelist.txt"), "w", encoding="utf-8") as handle:
            handle.writelines(rows)
        for name in ("model_info.json",):
            if os.path.exists(os.path.join(source, name)):
                shutil.copyfile(os.path.join(source, name), os.path.join(folder, name))
        arm_config = json.loads(json.dumps(config))
        arm_config["flow"]["model"].setdefault("backbone_args", {})["time_scale"] = settings["time_scale"]
        with open(os.path.join(folder, "rectified_config.json"), "w", encoding="utf-8") as handle:
            json.dump(arm_config, handle, indent=4)
        spec = {
            "model_name": PREFIX + arm, "total_epochs": args.epochs, "save_every": args.epochs,
            "batch_size": args.batch_size, "gpu": "0", "pretrained_flow": "", "vocoder": vocoder,
            "learning_rate": 0.0, "checkpoints": "latest", "fresh": True, "precision": args.precision,
            "compile": False, "torch_compile_mode": "default", "feature_cache": False,
            "mean_flow": settings["mean_flow"],
        }
        with open(os.path.join(folder, "flow", "run_spec.json"), "w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=4)
    # The holdout takes one clip per speaker, at least 32.
    steps = len(rows) // args.batch_size
    print(f"{len(rows)} clips of {args.source} in {', '.join(PREFIX + arm for arm in ARMS)}; about {steps} steps "
          f"an epoch, {args.epochs} epochs. Previews: {vocoder or 'none (no ChouwaGAN export found)'}.")


def run(args):
    for arm in args.arms or list(ARMS):
        spec = os.path.join(arm_dir(arm), "flow", "run_spec.json")
        if not os.path.exists(spec):
            raise SystemExit(f"{spec} is missing; run prepare first.")
        print(f"=== {arm}", flush=True)
        # The same seed for every arm: model initialisation and the loader's order.
        code = (f"import sys, torch; sys.path.insert(0, {str(ROOT)!r}); torch.manual_seed({SEED}); "
                f"from rvc.rectified.train_flow import main; main({str(spec)!r})")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def scalars(arm: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = {}
    for path in sorted(glob.glob(os.path.join(arm_dir(arm), "flow", "eval", "events.out.tfevents.*"))):
        events = EventAccumulator(path, size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                values.setdefault(tag, {})[event.step] = event.value
    return values


def report(_):
    found = {arm: scalars(arm) for arm in ARMS}
    base = found["ts1000"].get("holdout/flow", {})
    print("held-out flow loss (and vs ts1000)")
    for step in sorted(base):
        row = [f"{step:6d}"]
        for arm in ARMS:
            value = found[arm].get("holdout/flow", {}).get(step)
            row.append("        -" if value is None else f"{value:.4f} ({(value / base[step] - 1) * 100:+.2f}%)")
        print("  ".join(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--source", default="pretrain", help="Extracted experiment under logs/ to sample from.")
    setup.add_argument("--clips", type=int, default=16700)
    setup.add_argument("--epochs", type=int, default=5)
    setup.add_argument("--batch-size", type=int, default=16)
    setup.add_argument("--precision", default="bf16", choices=("fp32", "fp16", "bf16"))
    launch = commands.add_parser("run")
    launch.add_argument("arms", nargs="*", choices=list(ARMS))
    commands.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
