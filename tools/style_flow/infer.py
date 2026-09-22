"""Generate a styled F0 contour for an audio file, without RVC.

    python tools/style_flow/infer.py --model logs/singer/singer_style.pt \\
        --input song.wav --out song_styled.npz --transpose 0

The ``.npz`` has ``f0`` (styled), ``f0_source``, ``coarse`` and ``residual``,
and can be fed to evaluate.py and visualize.py like any contour.  For audio,
convert with the style model selected in the inference tab or with
``core.run_infer_script(..., style={...})``.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from rvc.lib.audio_io import load_audio_16k
from rvc.lib.style_flow.frontend import highpass
from rvc.lib.style_flow.infer import StyleEngine, StyleOptions, blend


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Style checkpoint.")
    parser.add_argument("--input", required=True, help="Audio file or folder.")
    parser.add_argument("--out", required=True, help="Output .npz, or a folder for folder input.")
    parser.add_argument("--transpose", type=float, default=0.0, help="Semitones.")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--intensity", type=float, default=1.0, help="0 keeps the source's descriptors (the dataset mean with --absolute), 1 the model's.")
    parser.add_argument("--absolute", action="store_true", help="Condition on the model's descriptors everywhere, not relative to the source.")
    parser.add_argument("--no-recenter", action="store_true", help="Don't centre each note on the source's pitch.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--descriptors", default="{}", help='JSON offsets, e.g. \'{"vibrato_extent_cents": 1.0}\'.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    options = StyleOptions(
        model=args.model, strength=args.strength, rate=args.rate, intensity=args.intensity,
        relative=not args.absolute, recenter=not args.no_recenter, steps=args.steps, cfg=args.cfg,
        descriptors=json.loads(args.descriptors), seed=args.seed,
    )
    engine = StyleEngine(args.model, args.device)

    if os.path.isdir(args.input):
        from rvc.lib.style_flow.evaluation import list_inputs

        inputs = [p for p in list_inputs(args.input) if not p.endswith(".npz")]
        os.makedirs(args.out, exist_ok=True)
        outputs = [os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".npz") for p in inputs]
    else:
        inputs, outputs = [args.input], [args.out]

    for src, dst in zip(inputs, outputs):
        audio = highpass(load_audio_16k(src))
        coarse, residual, vuv, plain = engine.residual(audio, args.transpose, options)
        source = engine.frontend.f0(audio)[: len(vuv)] * 2.0 ** (args.transpose / 12.0)
        f0 = np.where(vuv, blend(coarse, residual, plain, args.rate), 0.0).astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        np.savez(dst, f0=f0, f0_source=source.astype(np.float32), coarse=coarse, residual=residual)
        print(dst)


if __name__ == "__main__":
    main()
