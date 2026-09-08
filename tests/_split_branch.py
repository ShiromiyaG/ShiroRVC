"""``_split_branch_outputs`` from ``rvc/train/train.py``, imported by source.

``train.py`` reads ``sys.argv[1]`` at import time, so the tests cannot import
it directly; this lifts the one function out of the module by AST.
"""

import ast
from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parents[1] / "rvc" / "train" / "train.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_FN = next(
    node
    for node in _TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_split_branch_outputs"
)
exec(compile(ast.Module(body=[_FN], type_ignores=[]), str(_SOURCE), "exec"))

split_branch_outputs = _split_branch_outputs
