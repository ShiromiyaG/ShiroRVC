"""``branch_separation``, from ``rvc/train/diagnostics.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvc.train.diagnostics import branch_separation  # noqa: E402

__all__ = ["branch_separation"]
