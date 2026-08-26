"""The training tab's advanced settings are grouped by what they decide.

They used to be two columns split by widget type -- checkboxes on the left,
radios on the right -- with a full-width pile underneath.  That put
``Pretrained`` six controls above ``Custom Pretrained``, the LR scheduler in a
different column from warmup and the custom learning rates, and the index
options among training flags although the button that reads them is outside the
accordion entirely.

Read with ``ast``, so this runs without a Gradio install.  What it pins is the
grouping and, more importantly, that reordering the *definitions* never
disturbs the positional ``inputs`` list they are wired into.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "tabs" / "train" / "train.py"


def _train_tab() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_tab"
    )


def _component_order() -> list[str]:
    """Every keyed component, in the order it is created."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(_train_tab()):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "key" and isinstance(keyword.value, ast.Constant):
                found.append((node.lineno, str(keyword.value.value)))
    return [name for _line, name in sorted(found)]


def _section_order() -> list[str]:
    """The Markdown headers, in order."""
    headers = []
    for node in ast.walk(_train_tab()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Markdown"
            and node.args
        ):
            text = ast.unparse(node.args[0])
            if "####" in text:
                headers.append((node.lineno, text))
    return [text for _line, text in sorted(headers)]


#: The tab does not call ``core.run_train_script`` directly: Gradio binds
#: ``inputs`` positionally, so a named wrapper keeps that coupling in one place
#: and lets settings-level flags (FP16) be filled in at launch.
LAUNCHER = "start_train_from_ui"


def _train_inputs() -> list[str]:
    for node in ast.walk(_train_tab()):
        if not isinstance(node, ast.Call):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        target = keywords.get("fn")
        if isinstance(target, ast.Name) and target.id == LAUNCHER:
            return [e.id for e in keywords["inputs"].elts if isinstance(e, ast.Name)]
    raise AssertionError(f"the {LAUNCHER} click went missing")


def _launcher() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == LAUNCHER
    )


# -- the wiring, which the reorder must not have touched --------------------


def test_every_input_is_a_component_that_exists():
    """Gradio passes ``inputs`` positionally; a stale name is a hard failure."""
    defined = set()
    for node in ast.walk(_train_tab()):
        if isinstance(node, ast.Assign):
            defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        if isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            defined.add(node.optional_vars.id)

    missing = [name for name in _train_inputs() if name not in defined]
    assert not missing, f"inputs reference undefined components: {missing}"


def test_the_input_list_fills_every_launcher_parameter():
    """One component per wrapper parameter, since Gradio binds by position."""
    inputs = _train_inputs()
    params = [arg.arg for arg in _launcher().args.args]
    assert len(inputs) == len(params), (
        f"{len(inputs)} inputs for {len(params)} launcher parameters: "
        f"{inputs} vs {params}"
    )


def test_the_launcher_calls_the_backend_by_keyword():
    """Positional forwarding is what made an inserted flag shift every later one.

    ``use_fp16`` is the reason this test exists: it sits between ``use_tf32``
    and ``use_benchmark`` in ``run_train_script``, so a positional call here
    would have sent ``use_benchmark`` as the FP16 flag.
    """
    import inspect

    core = pytest.importorskip(
        "core", reason="the CLI carries its own dependencies"
    )

    call = next(
        node for node in ast.walk(_launcher())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_train_script"
    )
    assert not call.args, "run_train_script is being called positionally again"

    sent = {keyword.arg for keyword in call.keywords}
    accepted = inspect.signature(core.run_train_script).parameters
    unknown = sent - set(accepted)
    assert not unknown, f"the launcher sends parameters core does not take: {unknown}"

    required = {
        name for name, param in accepted.items()
        if param.default is inspect.Parameter.empty
    }
    assert required <= sent, f"the launcher never sets: {sorted(required - sent)}"


def test_fp16_reaches_the_backend_from_the_settings_tab():
    """The FP16 flag has no control here; it is read at launch time."""
    call = next(
        node for node in ast.walk(_launcher())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_train_script"
    )
    fp16 = next(k for k in call.keywords if k.arg == "use_fp16")
    assert (
        isinstance(fp16.value, ast.Call)
        and isinstance(fp16.value.func, ast.Name)
        and fp16.value.func.id == "get_use_fp16"
    ), "use_fp16 no longer comes from the persisted precision setting"


def test_no_component_key_was_lost_or_duplicated():
    keys = _component_order()
    duplicates = {name for name in keys if keys.count(name) > 1}
    assert not duplicates, f"duplicate keys: {sorted(duplicates)}"
    # Everything the training run needs must still be keyed, or the preset
    # save/load silently drops it.
    for required in ("pretrained", "custom_pretrained", "cleanup", "optimizer_choice",
                     "lr_scheduler", "use_warmup", "use_custom_lr", "use_ema",
                     "overtrain_detector", "stop_on_overtrain", "use_tf32",
                     "index_algorithm", "index_metric"):
        assert required in keys, f"{required} lost its key"


# -- the grouping ------------------------------------------------------------


def test_the_sections_appear_in_a_deliberate_order():
    headers = " | ".join(_section_order())
    for expected in ("Starting point", "Optimisation", "Checkpoints and quality",
                     "Performance", "Hardware", "Index"):
        assert expected in headers, f"the {expected!r} section is gone"

    order = [
        next(index for index, text in enumerate(_section_order()) if name in text)
        for name in ("Starting point", "Optimisation", "Checkpoints and quality",
                     "Performance", "Hardware", "Index")
    ]
    assert order == sorted(order), f"sections are out of order: {_section_order()}"


@pytest.mark.parametrize(
    "first,second,why",
    [
        ("pretrained", "custom_pretrained",
         "one turns pretrained weights on, the other picks which"),
        ("custom_pretrained", "cleanup", "both decide what the run starts from"),
        ("lr_scheduler", "use_warmup", "warmup shapes the same learning rate"),
        ("use_warmup", "use_custom_lr", "both override the learning rate"),
        ("overtrain_detector", "stop_on_overtrain",
         "the second is meaningless without the first"),
    ],
)
def test_related_controls_stay_adjacent(first, second, why):
    """Not a style preference: these pairs are read together.

    ``Pretrained`` and ``Custom Pretrained`` were eleven controls apart.
    """
    keys = _component_order()
    distance = abs(keys.index(second) - keys.index(first))
    assert distance <= 4, (
        f"{first} and {second} are {distance} controls apart -- {why}"
    )


def test_the_index_options_come_last():
    """They are not training options; the button that reads them is elsewhere."""
    keys = _component_order()
    training_flags = ("use_tf32", "use_ema", "optimizer_choice", "use_checkpointing")
    for flag in training_flags:
        assert keys.index("index_algorithm") > keys.index(flag), (
            f"index_algorithm still precedes {flag}"
        )
