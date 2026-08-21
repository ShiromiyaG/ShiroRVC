"""The GUI must expose every parameter the backend accepts.

A missing control is the quietest kind of bug: the call still succeeds, the
parameter silently keeps its default, and nothing anywhere says so.  That is
how ``torch_compile_mode`` went unnoticed.

The inverse also matters: a parameter that is no longer a per-run decision
belongs in the config JSON, not in three UIs.  ``rolling_loss_steps`` and
``lr_decay`` live there now, so they are deliberately absent here.

``core.py`` is read with ``ast`` rather than imported, so this runs without
torch -- the signature is the contract, and the source is enough to see it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError)

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def backend_parameters(function_name: str) -> set[str]:
    """Parameter names of a top-level function in core.py."""
    tree = ast.parse((ROOT / "core.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            arguments = node.args
            names = [arg.arg for arg in arguments.posonlyargs + arguments.args]
            names += [arg.arg for arg in arguments.kwonlyargs]
            return set(names)
    raise AssertionError(f"core.py has no function named {function_name}")


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


# -- training ---------------------------------------------------------------


def test_training_form_covers_run_train_script(app):
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        produced = set(page.train_args())
    finally:
        page.deleteLater()

    expected = backend_parameters("run_train_script")
    missing = expected - produced
    unknown = produced - expected

    assert not missing, (
        "run_train_script accepts parameters the training form never sets, so "
        f"they silently keep their defaults: {sorted(missing)}"
    )
    assert not unknown, (
        f"the training form sends arguments run_train_script does not take: {sorted(unknown)}"
    )


def test_torch_compile_mode_is_offered(app):
    """The mode has to be a real choice, not a hardcoded string."""
    from gui.services import catalog
    from gui.views.training import TrainingPage

    page = TrainingPage()
    try:
        offered = [
            page.torch_compile_mode.combo.itemText(index)
            for index in range(page.torch_compile_mode.combo.count())
        ]
        # Revealed only once compilation is switched on, as in the Gradio tab.
        # isHidden() rather than isVisibleTo(): the control also sits inside a
        # collapsed section, and this assertion is about the toggle, not the
        # accordion.
        assert page.torch_compile_mode_field.isHidden()
        page.compile_vocoder.setChecked(True)
        assert not page.torch_compile_mode_field.isHidden()
        page.train_advanced.set_expanded(True)
        assert page.torch_compile_mode_field.isVisibleTo(page)
    finally:
        page.deleteLater()

    assert offered == catalog.TORCH_COMPILE_MODES


# -- inference --------------------------------------------------------------


def test_inference_form_covers_run_infer_script(app):
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        produced = set(page.settings.values()) | set(page.selector.values())
        # Supplied by the page rather than the settings block.
        produced |= {"input_path", "output_path"}
    finally:
        page.deleteLater()

    expected = backend_parameters("run_infer_script")
    missing = expected - produced
    assert not missing, (
        f"run_infer_script accepts parameters the inference form never sets: {sorted(missing)}"
    )


def test_batch_form_covers_run_batch_infer_script(app):
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        produced = set(page.batch_settings.values()) | set(page.batch_selector.values())
        produced |= {"input_folder", "output_folder"}
        # The batch entry point has no bundle sub-model parameter.
        produced.discard("bundle_submodel")
    finally:
        page.deleteLater()

    expected = backend_parameters("run_batch_infer_script")
    missing = expected - produced
    assert not missing, (
        f"run_batch_infer_script accepts parameters the batch form never sets: {sorted(missing)}"
    )


# -- advanced sections ------------------------------------------------------


def test_advanced_sections_start_collapsed(app):
    """A disclosure that opens itself is not hiding anything."""
    from gui.views.inference import InferencePage

    page = InferencePage()
    try:
        assert not page.settings.advanced.is_expanded()
        assert not page.batch_settings.advanced.is_expanded()
    finally:
        page.deleteLater()
