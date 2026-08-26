"""The launcher and the trainer must agree on the run spec.

They used to agree by position: ``core.py`` built a list of 32 values and
``rvc/train/train.py`` read ``sys.argv[N]``.  Nothing checked the alignment, and
a wrong index was undetectable at runtime -- ``batch_size`` arriving where
``sample_rate`` was expected parses fine and simply trains the wrong thing.

``TrainRunSpec`` replaced that with one definition, and these tests pin the
three things that could still drift: the trainer reading a field the spec does
not define, the launcher failing to forward a UI parameter, and the on-disk
format not round-tripping.

``core.py`` and ``train.py`` are read with ``ast`` rather than imported, so this
runs without torch.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.train.run_spec import SPEC_VERSION, TrainRunSpec  # noqa: E402

SPEC_FIELDS = {f.name for f in fields(TrainRunSpec)}


def _parse(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _run_train_script() -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(_parse("core.py"))
        if isinstance(node, ast.FunctionDef) and node.name == "run_train_script"
    )


# -- the trainer side -------------------------------------------------------


def test_trainer_only_reads_fields_the_spec_defines():
    """Every ``spec.<name>`` in train.py resolves to a real field.

    This is the check that replaces the old argv index audit.  A typo or a
    renamed field fails here instead of at training time.
    """
    used = {
        node.attr
        for node in ast.walk(_parse("rvc/train/train.py"))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "spec"
    }
    # ``training_phase`` is a property, not a field.
    derived = {"training_phase"}
    unknown = used - SPEC_FIELDS - derived
    assert not unknown, f"train.py reads spec fields that do not exist: {sorted(unknown)}"


def test_trainer_takes_the_spec_path_from_argv_1():
    """The spec is ``sys.argv[1]``.

    Two different indexings meet here and are easy to confuse.  Inside the
    process ``sys.argv[0]`` is the script, so the spec is ``argv[1]``.  From the
    outside, ``core._find_trainer_processes`` reads the OS command line, which
    *does* include the interpreter -- there the script is ``cmdline[1]`` and the
    spec ``cmdline[2]``.  Reading ``argv[2]`` here raises IndexError on every
    launch; reading ``cmdline[2]`` there would break the stop button.
    """
    tree = _parse("rvc/train/train.py")
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "TrainRunSpec"
    ]
    assert len(loads) == 1, "expected exactly one TrainRunSpec.load in train.py"
    (arg,) = loads[0].args
    assert ast.unparse(arg) == "sys.argv[1]"


# -- the launcher side ------------------------------------------------------


def test_launcher_sets_every_spec_field():
    """``core.py`` must populate the whole spec, not just what it remembers."""
    call = next(
        node
        for node in ast.walk(_run_train_script())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainRunSpec"
    )
    supplied = {kw.arg for kw in call.keywords}
    missing = SPEC_FIELDS - supplied
    assert not missing, f"core.py leaves spec fields at their default: {sorted(missing)}"
    unknown = supplied - SPEC_FIELDS
    assert not unknown, f"core.py passes unknown spec fields: {sorted(unknown)}"


def test_launcher_passes_only_the_spec_path():
    """The command is [python, script, spec] -- no positional payload left."""
    assign = next(
        node
        for node in ast.walk(_run_train_script())
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "command" for t in node.targets)
    )
    assert isinstance(assign.value, ast.List)
    assert len(assign.value.elts) == 3, (
        "the training command grew a positional argument again; it should carry "
        "the run spec path only"
    )


# -- the format -------------------------------------------------------------


def test_round_trip_preserves_every_field(tmp_path):
    spec = TrainRunSpec(
        model_name="round-trip",
        sample_rate=44100,
        vocoder="refinegan",
        batch_size=12,
        gpus="0-1",
        pretrain_g="weights/G.pth",
        use_custom_lr=True,
        custom_lr_g=2.5e-5,
        use_ema=False,
    )
    assert TrainRunSpec.load(spec.save(tmp_path / "run_spec.json")) == spec


def test_training_phase_follows_the_pretrained_paths():
    base = {"model_name": "m", "sample_rate": 44100}
    assert TrainRunSpec(**base).training_phase == "pretrain"
    # "None" is what a UI with no selection stringifies to.
    assert TrainRunSpec(**base, pretrain_g="None").training_phase == "pretrain"
    assert TrainRunSpec(**base, pretrain_g="G.pth").training_phase == "finetune"
    assert TrainRunSpec(**base, pretrain_d="D.pth").training_phase == "finetune"


def test_a_stale_spec_version_is_refused(tmp_path):
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps({"spec_version": SPEC_VERSION + 1, "model_name": "m",
                                "sample_rate": 44100}), encoding="utf-8")
    with pytest.raises(ValueError, match="run spec version"):
        TrainRunSpec.load(path)


def test_an_unreadable_field_names_itself(tmp_path):
    path = tmp_path / "run_spec.json"
    payload = {"spec_version": SPEC_VERSION, "model_name": "m",
               "sample_rate": "not-a-number"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_rate"):
        TrainRunSpec.load(path)
