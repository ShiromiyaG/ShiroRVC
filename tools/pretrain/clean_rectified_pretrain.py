"""Strip a rectified flow or vocoder export down to what a fine-tune reads.

The flow's speaker table becomes one row, the mean of the trained speakers,
plus the null row: what ``resize_speakers`` starts every fine-tune from, so
the fine-tune is unchanged while the pretrain's voices and their number are
gone. Of the config only the mel settings and the network's own layout stay;
the training settings, the counters and the vocoder path are dropped.

    python tools/pretrain/clean_rectified_pretrain.py logs/run/flow/run_flow_10e_40200s.pth flow_pretrain.pth
    python tools/pretrain/clean_rectified_pretrain.py logs/run/vocoder/run_vocoder_27e_108864s.pth vocoder_pretrain.pth

The output's name is yours to choose: the export's carries the run, its epoch
and its step.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import error as print_error, info, install_rich_print, success

TAG = "[CLEAN]"
SPEAKER_TABLE = "encoder.speaker.weight"
#: Top-level fields a load path reads, besides ``config`` and ``model``.
KEEP = {
    "rectified_flow": ("kind", "embedder_model"),
    "rectified_vocoder": ("kind", "architecture", "recipe"),
}
#: The config section that holds each kind's network layout.
SECTION = {"rectified_flow": "flow", "rectified_vocoder": "vocoder"}


def anonymous_speakers(state: dict) -> dict:
    """``state`` with the speaker table cut to the trained speakers' mean and
    the null row."""
    table = state[SPEAKER_TABLE].float()
    state = dict(state)
    state[SPEAKER_TABLE] = torch.cat((table[:-1].mean(0, keepdim=True), table[-1:]), dim=0)
    return state


def clean_export(checkpoint: dict):
    """The cleaned export and the lines of its report."""
    kind = checkpoint.get("kind")
    if kind not in KEEP or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("not a rectified flow or vocoder export (a training checkpoint has no 'kind').")
    section = SECTION[kind]
    config = checkpoint["config"]
    out = {key: checkpoint[key] for key in KEEP[kind] if key in checkpoint}
    out["config"] = {"data": config["data"], section: {"model": config[section]["model"]}}
    out["model"] = dict(checkpoint["model"])
    lines = [f"kept:    {key} = {out[key]!r}" for key in KEEP[kind] if key in out]
    lines.append(f"kept:    config.data, config.{section}.model")
    if kind == "rectified_flow":
        speakers = out["model"][SPEAKER_TABLE].shape[0] - 1
        out["model"] = anonymous_speakers(out["model"])
        out["speaker_count"] = 1
        lines.append(f"speakers: {speakers} rows -> their mean, speaker_count = 1")
    dropped = [key for key in checkpoint if key not in out]
    dropped += [f"config.{key}" for key in config if key not in out["config"]]
    dropped += [f"config.{section}.{key}" for key in config[section] if key != "model"]
    lines.append("dropped: " + ", ".join(dropped))
    return out, lines


def main() -> int:
    install_rich_print()
    parser = argparse.ArgumentParser(description="Strip a rectified flow or vocoder export to a pretrain.")
    parser.add_argument("export", help="A *_flow_*.pth or *_vocoder_*.pth export.")
    parser.add_argument("output", help="Where to write the pretrain.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output.")
    args = parser.parse_args()

    source, output = Path(args.export), Path(args.output)
    try:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace.")
        payload, lines = clean_export(torch.load(source, map_location="cpu", weights_only=True))
        info(f"{source.name} -> {output.name}", tag=TAG)
        for line in lines:
            info(f"  {line}", tag=TAG)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        success(f"Pretrain written to {output}.", tag=TAG)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print_error(f"{source.name}: {exc}", tag=TAG)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
