"""Compare a converted output against its source and the target singer.

    python tools/style_flow/evaluate.py --source src.wav --output out.wav \\
        --target logs/singer/style_data --transpose 0 --json report.json

``--source``/``--output`` are files or folders (paired by file name); each
may be audio or a contour ``.npz`` with an ``f0`` key.  ``--target`` is a
style dataset, a folder of audio, or a folder of ``.npz``.  Descriptor
distances are in the normalized units of ``--stats`` (a style dataset),
which defaults to ``--target`` when that is one.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from rvc.lib.style_flow.config import default_config_path, load_config
from rvc.lib.style_flow.descriptors import DescriptorConfig
from rvc.lib.style_flow.evaluation import (
    ContourLoader,
    SpeakerEmbedder,
    Transcriber,
    character_error_rate,
    cosine,
    descriptor_distance,
    descriptor_rows,
    list_inputs,
    melody_error_cents,
    normalizer_from,
    pooled_descriptors,
)
from rvc.lib.style_flow.f0_repr import ReprConfig
from rvc.lib.terminal import get_console, warning


def pair(sources, outputs):
    if len(sources) == 1 and len(outputs) == 1:
        return [(sources[0], outputs[0])]
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in outputs}
    pairs = [(s, by_stem[k]) for s in sources if (k := os.path.splitext(os.path.basename(s))[0]) in by_stem]
    if not pairs:
        raise SystemExit("No source/output pairs share a file name.")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stats", help="Style dataset whose descriptor statistics scale the distances.")
    parser.add_argument("--transpose", type=float, default=0.0, help="Semitones applied at conversion.")
    parser.add_argument("--config", default=default_config_path("base.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cer", action="store_true", help="Whisper CER of output against source (audio only).")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--speaker-sim", action="store_true", help="Resemblyzer similarity to the target (audio only).")
    parser.add_argument("--json", help="Write the report here.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rcfg = ReprConfig.from_dict(cfg.get("representation"))
    dcfg = DescriptorConfig.from_dict(cfg.get("descriptors"))
    loader = ContourLoader(args.device)

    pairs = pair(list_inputs(args.source), list_inputs(args.output))
    targets = list_inputs(args.target)
    report = {"pairs": []}

    melody = []
    for src, out in pairs:
        err = melody_error_cents(loader(out), loader(src), rcfg, args.transpose)
        melody.append(err)
        report["pairs"].append({"source": src, "output": out, "melody_error_cents": err})

    columns = {
        "output": pooled_descriptors([o for _, o in pairs], loader, rcfg, dcfg),
        "target": pooled_descriptors(targets, loader, rcfg, dcfg),
        "source": pooled_descriptors([s for s, _ in pairs], loader, rcfg, dcfg),
    }
    report["descriptors"] = {k: [None if np.isnan(v) else float(v) for v in col] for k, col in columns.items()}

    normalizer = normalizer_from(args.stats or args.target)
    if normalizer is not None:
        report["distance_to_target"] = {
            "output": descriptor_distance(columns["output"], columns["target"], normalizer),
            "source": descriptor_distance(columns["source"], columns["target"], normalizer),
        }

    audio_pairs = [(s, o) for s, o in pairs if not s.endswith(".npz") and not o.endswith(".npz")]
    if args.cer:
        try:
            transcribe = Transcriber(args.whisper_model, args.device)
            cers = [character_error_rate(transcribe(s), transcribe(o)) for s, o in audio_pairs]
            report["cer"] = float(np.nanmean(cers)) if cers else None
        except ImportError:
            warning("openai-whisper is not installed; skipping CER.", tag="[STYLE]")
    if args.speaker_sim:
        target_audio = [t for t in targets if not t.endswith(".npz")]
        if not target_audio:
            warning("--speaker-sim needs target audio, not contours; skipping.", tag="[STYLE]")
        else:
            try:
                embed = SpeakerEmbedder()
                target_emb = embed(target_audio)
                report["speaker_similarity"] = {
                    "output": cosine(embed([o for _, o in audio_pairs]), target_emb),
                    "source": cosine(embed([s for s, _ in audio_pairs]), target_emb),
                }
            except ImportError:
                warning("resemblyzer is not installed; skipping speaker similarity.", tag="[STYLE]")

    from rich.table import Table

    console = get_console()
    means = [m["mean"] for m in melody if m["frames"]]
    console.print(f"Melody error (coarse, cents): mean {np.mean(means):.1f} over {len(means)} pair(s)" if means else "Melody error: no voiced overlap")
    table = Table(title="Style descriptors")
    table.add_column("descriptor")
    for name in columns:
        table.add_column(name, justify="right")
    for name, values in descriptor_rows(columns):
        table.add_row(name, *("-" if np.isnan(v) else f"{v:.3f}" for v in values))
    console.print(table)
    for key in ("distance_to_target", "cer", "speaker_similarity"):
        if key in report:
            console.print(f"{key}: {report[key]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)


if __name__ == "__main__":
    main()
