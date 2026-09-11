"""``band_deficit_db`` and friends, from ``rvc/train/overtrain.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvc.train.overtrain import band_deficit_db, deficit_band_edges  # noqa: E402

__all__ = ["band_deficit_db", "deficit_band_edges"]
