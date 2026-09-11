"""``split_branch_outputs``, from ``rvc/train/diagnostics.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvc.train.diagnostics import split_branch_outputs  # noqa: E402

__all__ = ["split_branch_outputs"]
