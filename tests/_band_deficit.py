"""``_band_deficit_db`` and friends from ``rvc/train/train.py``, by source.

``train.py`` reads ``sys.argv[1]`` at import time, so the tests cannot import it
directly; this lifts the three functions out of the module by AST.
"""

import ast
from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parents[1] / "rvc" / "train" / "train.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_WANTED = {"_band_deficit_db", "_deficit_band_edges"}
_NODES = [
    node
    for node in _TREE.body
    if (isinstance(node, ast.FunctionDef) and node.name in _WANTED)
    or (
        isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "HOLDOUT_DEFICIT_BANDS"
            for t in node.targets
        )
    )
]
exec(compile(ast.Module(body=_NODES, type_ignores=[]), str(_SOURCE), "exec"))

band_deficit_db = _band_deficit_db
deficit_band_edges = _deficit_band_edges
