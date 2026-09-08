"""``_branch_separation`` from ``rvc/train/train.py``, imported by source.

``train.py`` reads ``sys.argv[1]`` at import time, so the test cannot import
it directly; this lifts the one function out of the module by AST."""

import ast
from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parents[1] / "rvc" / "train" / "train.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_FN = next(
    node
    for node in _TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_branch_separation"
)
exec(compile(ast.Module(body=[_FN], type_ignores=[]), str(_SOURCE), "exec"))

branch_separation = _branch_separation
