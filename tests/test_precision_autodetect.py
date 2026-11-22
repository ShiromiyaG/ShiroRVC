"""Which precision a run trains in.

``assets/config.json`` ships without a ``precision`` key, meaning *auto*: the
machine is asked -- BF16 with native kernels (8.0+), else FP16 with tensor
cores (7.0+), else FP32.  The answer is never written back, so the file only
records what a person chose and a config carried between machines re-detects.
An explicit choice the hardware cannot run (autocast without CUDA, BF16
without native kernels) falls back to the best supported precision.
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
# auto-detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cuda,capability,expected",
    [
        (False, (8, 6), "fp32"),
        (True, (6, 1), "fp32"),  # Pascal: no tensor cores
        (True, (7, 5), "fp16"),  # Turing: FP16 tensor cores, no BF16
        (True, (8, 0), "bf16"),  # Ampere: native BF16
        (True, (12, 0), "bf16"),
    ],
)
def test_auto_picks_the_best_supported(monkeypatch, cuda, capability, expected):
    _pretend(monkeypatch, cuda=cuda, capability=capability)
    assert config_module.detect_training_precision() == expected
    assert config_module.resolve_training_precision("auto") == expected
    assert config_module.resolve_training_precision(None) == expected


# --------------------------------------------------------------------------
# explicit choices and their fallback
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cuda,capability,requested,expected",
    [
        (True, (8, 6), "fp32", "fp32"),  # FP32 is always honoured
        (True, (8, 6), "fp16", "fp16"),  # a supported downgrade is honoured
        (True, (7, 5), "bf16", "fp16"),  # no native BF16 -> best supported
        (True, (6, 1), "bf16", "fp32"),
        (True, (6, 1), "fp16", "fp16"),  # still halves memory: kept
        (False, (8, 6), "bf16", "fp32"),  # autocast without CUDA does nothing
        (False, (8, 6), "fp16", "fp32"),
    ],
)
def test_an_unsupported_choice_falls_back(monkeypatch, cuda, capability, requested, expected):
    _pretend(monkeypatch, cuda=cuda, capability=capability)
    assert config_module.resolve_training_precision(requested) == expected


def test_an_unknown_precision_is_rejected(monkeypatch):
    _pretend(monkeypatch, cuda=True)
    with pytest.raises(ValueError):
        config_module.resolve_training_precision("int8")


# --------------------------------------------------------------------------
# the persisted preference
# --------------------------------------------------------------------------


def test_an_undecided_config_is_auto_and_left_alone(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    app_config.write_text(json.dumps({"model_author": "None"}))

    assert config_module.get_precision_preference() == "auto"
    assert config_module.get_training_precision() == "bf16"
    assert json.loads(app_config.read_text()) == {"model_author": "None"}


def test_a_decision_already_made_is_never_second_guessed(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    app_config.write_text(json.dumps({"precision": "fp32"}))
    assert config_module.get_training_precision() == "fp32"


def test_the_legacy_use_fp16_key_is_still_read(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(8, 9))
    app_config.write_text(json.dumps({"use_fp16": True}))
    assert config_module.get_precision_preference() == "fp16"


def test_a_missing_or_broken_config_still_answers(app_config, monkeypatch):
    _pretend(monkeypatch, cuda=True, capability=(7, 5))
    assert not app_config.exists()
    assert config_module.get_training_precision() == "fp16"
    app_config.write_text("{ not json")
    assert config_module.get_training_precision() == "fp16"


def test_the_shipped_config_leaves_the_decision_open():
    """If the key ships present, no install ever auto-detects."""

    shipped = json.loads((ROOT / "assets" / "config.json").read_text())
    assert "precision" not in shipped
    assert "use_fp16" not in shipped
