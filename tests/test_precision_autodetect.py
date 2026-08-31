"""Which precision a fresh install starts in.

``assets/config.json`` shipped ``"use_fp16": false``, so every install began in
FP32 and stayed there until someone found the settings tab.  That was the right
default when AMP bought nothing; it is the wrong one now.  Measured on the full
training step -- decoder plus discriminator, forward and backward, batch 8 over
0.4 s -- FP16 takes RefineGAN from 5.12 to 3.78 GiB and ChouwaGAN from 4.27 to
2.61 GiB, and both get *faster* rather than slower.  On an 8 GB card that is the
difference between batch 8 fitting and not.

So the key ships absent, which means *undecided*: the machine is asked instead.
The answer is not written back -- the file records only what a person chose, so
a read-only install works and a config carried between machines re-asks rather
than importing the other one's answer.  "Supports FP16" is a capability check, not
a CUDA-presence check -- every card back to Kepler can multiply half-precision
numbers, but without tensor cores (below compute 7.0) autocast costs casts and
buys only memory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.configs import config as config_module  # noqa: E402


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    """Point the persisted preference at a scratch file."""

    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "_APP_CONFIG_PATH", str(path))
    return path


def _pretend(monkeypatch, *, cuda, capability=(8, 6), hip=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)


# --------------------------------------------------------------------------
# the capability check
# --------------------------------------------------------------------------


def test_no_cuda_means_no_fp16(monkeypatch):
    _pretend(monkeypatch, cuda=False)
    assert config_module.fp16_is_supported() is False


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((6, 1), False),  # Pascal: half precision at 1/64 rate, no tensor cores
        ((7, 0), True),  # Volta: the first with FP16 tensor cores
        ((7, 5), True),
        ((8, 6), True),
        ((12, 0), True),
    ],
)
def test_the_cutoff_is_tensor_cores_not_arithmetic(monkeypatch, capability, expected):
    _pretend(monkeypatch, cuda=True, capability=capability)
    assert config_module.fp16_is_supported() is expected


def test_rocm_is_judged_by_its_presence_not_its_gfx_number(monkeypatch):
    """ROCm reports a gfx architecture through the same call and its numbers do
    not mean CUDA capabilities -- gfx1030 comes back as (10, 3), which would
    pass the check for the wrong reason and gfx900 as (9, 0), likewise."""

    _pretend(monkeypatch, cuda=True, capability=(9, 0), hip="6.2.0")
    assert config_module.fp16_is_supported() is True


def test_a_driver_that_refuses_to_answer_falls_back_to_fp32(monkeypatch):
    _pretend(monkeypatch, cuda=True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("no CUDA-capable device is detected")

    monkeypatch.setattr(torch.cuda, "get_device_capability", explode)
    assert config_module.fp16_is_supported() is False


# --------------------------------------------------------------------------
# the persisted decision
# --------------------------------------------------------------------------


def test_an_undecided_config_asks_the_machine(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    app_config.write_text(json.dumps({"model_author": "None"}))

    assert config_module.get_use_fp16() is True
    # And leaves the file alone: only ``set_use_fp16`` writes the key, so what
    # is on disk stays a record of what a person chose.
    assert json.loads(app_config.read_text()) == {"model_author": "None"}


def test_a_machine_without_the_hardware_answers_fp32(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=False)
    app_config.write_text("{}")

    assert config_module.get_use_fp16() is False
    assert json.loads(app_config.read_text()) == {}


def test_a_decision_already_made_is_never_second_guessed(app_config, monkeypatch):
    """The point of persisting: someone who turned FP16 off must not have it
    turned back on by the next launch."""

    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    app_config.write_text(json.dumps({"use_fp16": False}))

    assert config_module.get_use_fp16() is False
    assert json.loads(app_config.read_text())["use_fp16"] is False


def test_a_missing_config_file_still_answers(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    assert not app_config.exists()
    assert config_module.get_use_fp16() is True


def test_an_unwritable_install_directory_does_not_stop_the_app(
    app_config, monkeypatch
):
    """Nothing is written on the read path, so this works by construction --
    pinned because reintroducing a write-back would break it silently."""

    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    monkeypatch.setattr(config_module, "_APP_CONFIG_PATH", str(app_config / "nope"))
    assert config_module.get_use_fp16() is True


def test_a_hand_broken_config_is_survivable(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=False)
    app_config.write_text("{ not json")
    assert config_module.get_use_fp16() is False


def test_the_shipped_config_leaves_the_decision_open():
    """If the key ships present, no install ever auto-detects."""

    shipped = json.loads((ROOT / "assets" / "config.json").read_text())
    assert "use_fp16" not in shipped


# --------------------------------------------------------------------------
# the Qt toggle
# --------------------------------------------------------------------------


def test_the_qt_toggle_follows_the_same_capability_cutoff():
    pytest.importorskip(
        "PySide6.QtWidgets", reason="the Qt interface is optional", exc_type=ImportError
    )
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.views.training import TrainingPage

    def _device(capability):
        return {
            "index": 0,
            "name": "Test GPU",
            "total_vram": 8 * 2**30,
            "capability": capability,
        }

    app = QApplication.instance() or QApplication([])
    page = TrainingPage()

    # Pre-Volta: no FP16 tensor cores, so it is slower -- but it still halves
    # activation memory, which is a real trade and not a dead setting.  The box
    # stays available; only the default changes.
    page.populate_gpus([_device("6.1")])
    assert page.fp16.isEnabled() is True
    assert page.fp16.isChecked() is False
    assert "no FP16 tensor cores" in page.fp16.toolTip()

    page.populate_gpus([_device("7.0")])
    assert page.fp16.isEnabled() is True
    assert page.fp16.isChecked() is True

    # TF32 needs 8.0, FP16 only 7.0 -- the two must not share a cutoff.
    assert page.tf32.isEnabled() is False

    # No device at all is a different answer from "this device is too old", and
    # must not be told it has a GPU without tensor cores.
    page.populate_gpus([])
    assert page.fp16.isEnabled() is False
    assert page.fp16.isChecked() is False
    assert "needs a CUDA GPU" in page.fp16.toolTip()
    assert "tensor cores" not in page.fp16.toolTip()

    del page
    del app
