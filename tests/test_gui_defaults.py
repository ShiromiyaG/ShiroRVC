"""The two interfaces must agree on what an untouched form sends.

Covering the parameter *names* is not enough: the Qt training page had eight
controls whose default differed from the Gradio tab's, so the same "just press
start" run produced a different model depending on which window it was started
from -- 300 epochs against 500, ``post_rms`` normalisation against
``post_peak``, a checkpoint every 10 epochs against every 1.  None of that
raises anything; it just quietly trains something else.

Gradio is the reference. Its tabs are read with ``ast`` rather than imported,
which keeps this runnable without a Gradio install, and the Qt values come from
building the page and asking for the payload it would actually send.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

#: Supplied by the user or derived from the machine, so there is no shared
#: default to agree on.
NOT_A_DEFAULT = {
    "model_name", "dataset_path", "gpu", "training_gpu", "extract_gpu",
    "multiple_gpu", "g_pretrained_path", "d_pretrained_path", "sample_rate",
    "sampling_rate", "embedder_model_custom", "batch_size", "cpu_threads",
    "vocoder_arch", "custom_lr_g", "custom_lr_d",
    # Chosen from what the hardware reports.
    "use_tf32", "use_checkpointing",
    # The valid ids come from the selected model's extracted features, so there
    # is nothing for three interfaces to agree on until one is picked.  The
    # default that *is* shared -- ``index_single_speaker``, off -- is compared.
    "index_speaker",
}


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _UNRESOLVED


class _Unresolved:
    def __repr__(self):
        return "<computed at runtime>"


_UNRESOLVED = _Unresolved()


def gradio_defaults(relative_path: str) -> dict[str, object]:
    """``key=`` -> default, for every keyed component in a Gradio tab."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        key = _literal(keywords["key"]) if "key" in keywords else None
        if not isinstance(key, str):
            continue
        if "value" in keywords:
            found[key] = _literal(keywords["value"])
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "Slider"
            and len(node.args) >= 3
        ):
            # gr.Slider(minimum, maximum, value, ...)
            found[key] = _literal(node.args[2])
    return found


def equivalent(left, right) -> bool:
    """1 and 1.0 are the same default; True and 1 are not.

    The distinction matters: a Slider that was handed ``value=True`` is a
    Checkbox keyword left behind, not a deliberate 1.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) == isinstance(right, bool) and bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    return str(left) == str(right)


def test_training_defaults_match_gradio(app):
    """Every keyed control on the training tab, against the Qt payload."""
    from gui.views.training import TrainingPage

    reference = gradio_defaults("tabs/train/train.py")

    page = TrainingPage()
    try:
        # The preprocess and extract payloads are built inside their click
        # handlers, so the handlers are run with the backend call stubbed.
        page.dataset.set_path(str(ROOT))
        produced = dict(page.train_args())
        for handler in ("_preprocess", "_extract", "_build_index"):
            captured: dict[str, object] = {}
            page.run = lambda command, payload, **kwargs: captured.update(payload)
            page.require = lambda **kwargs: True
            page.notify = type("_S", (), {"emit": staticmethod(lambda *a: None)})()
            getattr(page, handler)()
            produced.update(captured)
    finally:
        page.deleteLater()

    mismatches = []
    compared = 0
    for key, expected in sorted(reference.items()):
        if key in NOT_A_DEFAULT or expected is _UNRESOLVED or key not in produced:
            continue
        compared += 1
        if not equivalent(expected, produced[key]):
            mismatches.append(f"  {key}: gradio={expected!r} qt={produced[key]!r}")

    assert compared > 30, f"only {compared} defaults compared; the join broke"
    assert not mismatches, (
        "the Qt training form disagrees with the Gradio tab:\n" + "\n".join(mismatches)
    )


#: ``--filter_radius`` is deliberately left out of the CLI check.  The single
#: inference tab sets it to 0.006 on a control it then hides
#: (``interactive=False, visible=False``), while batch and TTS expose it as an
#: integer 3 -- and ``Pipeline.get_f0`` ignores the argument entirely, hardcoding
#: 0.03 for rmvpe and 0.006 for fcpe.  Mirroring 0.006 would mean turning a
#: documented integer flag into a float to match a hidden control that changes
#: nothing.
CLI_EXEMPT = {"filter_radius"}


@pytest.mark.parametrize(
    "command, page_factory, payload",
    [
        ("infer", "inference", "settings"),
        ("batch_infer", "inference", "settings"),
        ("tts", "tts", "settings"),
    ],
)
def test_cli_conversion_defaults_match_the_gui(app, command, page_factory, payload):
    """Each conversion command's click defaults against the panel's.

    The three commands share one option list, so before this they shared one set
    of defaults while the tabs shipped three -- ``rvc tts`` blended the index at
    0.3 where the TTS tab uses 0.75.
    """
    core = pytest.importorskip(
        "core", reason="the CLI carries its own dependencies"
    )
    from gui.services import catalog

    context = {"infer": "single", "batch_infer": "batch", "tts": "tts"}[command]
    expected = catalog.INFERENCE_DEFAULTS[context]

    declared = {
        parameter.name: parameter.default
        for parameter in core.cli.commands[command].params
    }

    mismatches = [
        f"  {key}: gui={value!r} cli={declared[key]!r}"
        for key, value in sorted(expected.items())
        if key in declared
        and key not in CLI_EXEMPT
        and not equivalent(value, declared[key])
    ]
    assert not mismatches, (
        f"`{command}` disagrees with the {context} form:\n" + "\n".join(mismatches)
    )


def test_cli_training_defaults_match_gradio(app):
    """Training and preprocessing click defaults against the Gradio tab."""
    core = pytest.importorskip(
        "core", reason="the CLI carries its own dependencies"
    )

    reference = gradio_defaults("tabs/train/train.py")
    declared = {}
    for command in ("train", "preprocess", "extract", "index"):
        for parameter in core.cli.commands[command].params:
            # A required option has no default to agree on.
            if not parameter.required:
                declared.setdefault(parameter.name, parameter.default)

    mismatches = []
    compared = 0
    for key, expected in sorted(reference.items()):
        if key in NOT_A_DEFAULT or expected is _UNRESOLVED or key not in declared:
            continue
        compared += 1
        if not equivalent(expected, declared[key]):
            mismatches.append(f"  {key}: gradio={expected!r} cli={declared[key]!r}")

    assert compared > 20, f"only {compared} defaults compared; the join broke"
    assert not mismatches, (
        "the CLI disagrees with the Gradio training tab:\n" + "\n".join(mismatches)
    )


@pytest.mark.parametrize(
    "attribute, context",
    [("settings", "single"), ("settings", "tts")],
)
def test_conversion_defaults_match_catalog(app, attribute, context):
    """The shared settings panel must start from the catalogue's numbers.

    ``catalog.INFERENCE_DEFAULTS`` is the transcription of the Gradio tabs, and
    it is deliberately different per context -- single, batch and TTS each ship
    their own -- so this checks the panel actually applies the set it was given
    rather than one of the others.
    """
    from gui.services import catalog

    if context == "tts":
        from gui.views.tts import TtsPage as Page
    else:
        from gui.views.inference import InferencePage as Page

    page = Page()
    try:
        values = getattr(page, attribute).values()
    finally:
        page.deleteLater()

    expected = catalog.INFERENCE_DEFAULTS[context]
    mismatches = [
        f"  {key}: catalog={value!r} panel={values[key]!r}"
        for key, value in sorted(expected.items())
        if key in values and not equivalent(value, values[key])
    ]
    assert not mismatches, (
        f"the {context} settings panel drifted from the catalogue:\n"
        + "\n".join(mismatches)
    )
