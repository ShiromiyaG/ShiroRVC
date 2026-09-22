"""Plot source and generated F0 over the source's coarse melody, with example
contours of the target singer below.

    python tools/style_flow/visualize.py --source src.wav --generated out.wav \\
        --target logs/singer/style_data --out plot.png

Inputs are audio or contour ``.npz`` files (``f0`` key); ``--target`` is a
style dataset or a folder of either.
"""

import argparse
import os
import random
import sys

import numpy as np

sys.path.append(os.getcwd())

from rvc.lib.style_flow.config import default_config_path, load_config
from rvc.lib.style_flow.evaluation import ContourLoader, list_inputs
from rvc.lib.style_flow.f0_repr import ReprConfig, coarse_f0


def _masked(f0):
    f0 = np.asarray(f0, dtype=np.float64)
    return np.where(f0 > 0, f0, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--generated", action="append", default=[], help="Repeatable.")
    parser.add_argument("--target", help="Target singer examples.")
    parser.add_argument("--target-prefix", default="", help="Only target files whose name starts with this, e.g. '110_'.")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--start", type=float, default=0.0, help="Seconds.")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--transpose", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=default_config_path("base.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rcfg = ReprConfig.from_dict(load_config(args.config).get("representation"))
    fr = rcfg.frame_rate
    loader = ContourLoader(args.device)

    targets = list_inputs(args.target) if args.target else []
    targets = [p for p in targets if os.path.basename(p).startswith(args.target_prefix)]
    random.Random(args.seed).shuffle(targets)
    targets = targets[: args.examples]

    fig, axes = plt.subplots(1 + len(targets), 1, figsize=(14, 3.2 * (1 + len(targets))), squeeze=False)
    axes = axes[:, 0]

    lo, hi = int(args.start * fr), int((args.start + args.seconds) * fr)
    src = loader(args.source) * 2.0 ** (args.transpose / 12.0)
    t = np.arange(len(src))[lo:hi] / fr
    ax = axes[0]
    ax.plot(t, _masked(src[lo:hi]), color="0.55", lw=1.2, label="source")
    ax.plot(t, _masked(coarse_f0(src, rcfg) * (src > 0))[lo:hi], color="0.2", lw=1.0, ls="--", label="source coarse")
    for i, path in enumerate(args.generated):
        gen = loader(path)[lo:hi]
        ax.plot(np.arange(len(gen)) / fr + args.start, _masked(gen), lw=1.2, color=f"C{i}", label=os.path.basename(path))
    ax.set_title("source vs generated")

    for ax, path in zip(axes[1:], targets):
        f0 = loader(path)
        span = f0[: int(args.seconds * fr)]
        tt = np.arange(len(span)) / fr
        ax.plot(tt, _masked(span), color="C3", lw=1.2, label="target")
        ax.plot(tt, _masked(coarse_f0(f0, rcfg) * (f0 > 0))[: len(span)], color="0.2", lw=1.0, ls="--", label="coarse")
        ax.set_title(os.path.basename(path))

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("F0 (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)


if __name__ == "__main__":
    main()
