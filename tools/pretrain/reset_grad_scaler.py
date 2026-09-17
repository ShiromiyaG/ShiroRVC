"""Reset the GradScaler state saved in a D checkpoint.

A resume restores the scaler from ``extra["grad_scaler"]`` in ``D_*.pth``.
Removing it makes the next resume start at the trainer's ``init_scale``;
``--scale`` writes a fresh state at that scale instead.

Usage::

    python tools/pretrain/reset_grad_scaler.py logs/pretrain-contentvec
    python tools/pretrain/reset_grad_scaler.py logs/pretrain-contentvec/D_131112.pth --scale 1024
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


def resolve_checkpoint(target: Path) -> Path:
    """``target`` itself, or the highest-step ``D_*.pth`` in a model folder."""
    if target.is_file():
        return target
    candidates = [
        path for path in target.glob("D_*.pth") if path.stem.split("_")[-1].isdigit()
    ]
    if not candidates:
        sys.exit(f"No D_*.pth found in '{target}'.")
    return max(candidates, key=lambda path: int(path.stem.split("_")[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="Model folder or D_*.pth file.")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Write a fresh scaler state at this scale instead of removing it.",
    )
    parser.add_argument(
        "--reset-skipped",
        action="store_true",
        help="Also zero the amp_skipped_steps counter.",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip the .bak copy.")
    args = parser.parse_args()

    path = resolve_checkpoint(args.target)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    extra = dict(checkpoint.get("extra") or {})

    old_state = extra.get("grad_scaler")
    if old_state:
        print(f"{path.name}: saved scale {old_state.get('scale')}")
    else:
        print(f"{path.name}: no GradScaler state saved")

    if args.scale is None:
        extra.pop("grad_scaler", None)
    else:
        extra["grad_scaler"] = {
            "scale": float(args.scale),
            "growth_factor": float((old_state or {}).get("growth_factor", 2.0)),
            "backoff_factor": float((old_state or {}).get("backoff_factor", 0.5)),
            "growth_interval": int((old_state or {}).get("growth_interval", 2000)),
            "_growth_tracker": 0,
        }
    if args.reset_skipped:
        extra.pop("amp_skipped_steps", None)

    # ``save_checkpoint`` omits an empty ``extra``; match it.
    if extra:
        checkpoint["extra"] = extra
    else:
        checkpoint.pop("extra", None)

    if not args.no_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")

    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

    if args.scale is None:
        print("GradScaler state removed; the next resume starts at init_scale.")
    else:
        print(f"GradScaler state set to scale {args.scale}.")


if __name__ == "__main__":
    main()
