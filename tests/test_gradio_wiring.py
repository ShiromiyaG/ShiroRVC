"""Gradio passes its inputs positionally, so the lists have to stay in step.

``.click(fn=..., inputs=[...])`` matches components to parameters by position
and nothing checks it: add a slider to the tab and forget the list, and every
argument after that point shifts by one.  Nothing raises -- the run just uses
the wrong numbers, and ``protect`` quietly becomes whatever the new control was.

This is not hypothetical.  ``silence_gate_db`` was first added next to
``volume_envelope`` in ``core.py``, which is where it belongs by meaning, and
that alone would have shifted five arguments in three tabs.  It now sits at the
end of those signatures for exactly this reason.

Read statically: importing the tabs builds a Gradio app.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABS = ("tabs/inference/inference.py", "tabs/tts/tts.py")


def _core_entry_points():
    import core

    return {
        "run_infer_script": core.run_infer_script,
        "run_batch_infer_script": core.run_batch_infer_script,
        "run_tts_script": core.run_tts_script,
    }


def _wirings():
    """Every ``.click(fn=<name>, inputs=[...])`` in the tabs, as (label, got, expected)."""
    entry_points = _core_entry_points()
    found = []
    for relative in TABS:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        local_functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "click"
            ):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            fn, inputs = keywords.get("fn"), keywords.get("inputs")
            if not isinstance(fn, ast.Name) or not isinstance(inputs, ast.List):
                continue
            if fn.id in entry_points:
                expected = len(inspect.signature(entry_points[fn.id]).parameters)
            elif fn.id in local_functions:
                arguments = local_functions[fn.id].args
                expected = len(arguments.args) + len(arguments.kwonlyargs)
            else:
                continue
            found.append((f"{relative}:{fn.id}", len(inputs.elts), expected))
    return found


def test_the_conversion_tabs_are_wired():
    """The three that matter are actually present, so an empty pass is not a pass."""
    labels = {label for label, _, _ in _wirings()}
    assert any(label.endswith("run_single_infer") for label in labels)
    assert any(label.endswith("run_batch_infer_script") for label in labels)
    assert any(label.endswith("run_tts_script") for label in labels)


@pytest.mark.parametrize("wiring", _wirings(), ids=lambda w: w[0])
def test_every_input_list_fits_its_function(wiring):
    label, got, expected = wiring
    assert got <= expected, (
        f"{label} passes {got} components to a function taking {expected}; "
        "a component was added to inputs without a parameter to receive it"
    )


def test_the_silence_gate_is_last_everywhere():
    """It is appended, not slotted in beside ``volume_envelope``.

    Meaning would put it next to the envelope blend.  Position is what the
    tabs actually rely on, and a new parameter in the middle of these
    signatures shifts every argument after it.
    """
    import core

    for entry in (
        core.run_infer_script,
        core.run_batch_infer_script,
        core.run_tts_script,
    ):
        parameters = list(inspect.signature(entry).parameters)
        assert parameters[-1] == "silence_gate_db", entry.__name__

    for relative in TABS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "click"
            ):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            fn, inputs = keywords.get("fn"), keywords.get("inputs")
            if not isinstance(fn, ast.Name) or not isinstance(inputs, ast.List):
                continue
            if fn.id not in {
                "run_single_infer",
                "run_batch_infer_script",
                "run_tts_script",
            }:
                continue
            last = inputs.elts[-1]
            assert isinstance(last, ast.Name) and last.id.startswith(
                "silence_gate_db"
            ), f"{relative}:{fn.id} does not end with the silence gate"
