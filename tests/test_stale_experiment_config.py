"""An experiment config must not outlive the architecture it was written for.

``generate_config`` deliberately keeps an existing ``logs/<model>/config.json``
so per-run tuning survives re-extraction.  Combined with ``train.py`` -- which
forces ``architecture_id`` to the current build and otherwise fills in
only *absent* keys -- that meant a stale config kept every shape-defining value
it happened to name while advertising the new architecture id.

Concretely, restarting the 44.1 kHz pretrain in place across the v2 -> v3 change
would have built a 64-wide fast latent, labelled it v3, and reported nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.train.extract.preparing_files import generate_config

SHIPPED = ROOT / "rvc" / "configs" / "refinegan" / "44100.json"


@pytest.fixture
def shipped():
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _in_app_root(monkeypatch):
    """Run from the app root, and keep the rich console out of it.

    ``generate_config`` resolves the shipped config relative to ``cwd``, and
    its ``info`` banner renders through a module-level rich Console that other
    tests in the suite leave pointed at a replaced stdout.
    """
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        "rvc.train.extract.preparing_files.info", lambda *a, **k: None
    )


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_config_from_an_older_architecture_is_replaced(tmp_path, shipped):
    stale = json.loads(json.dumps(shipped))
    stale["model"]["architecture_id"] = "shiro_vits_svae_v2"
    # A shape-defining key the stale config disagrees on: the point of the
    # guard is that it does not survive the replacement.
    stale["model"]["refinegan_harmonics"] = [64, 64, 64]
    _write(tmp_path / "config.json", stale)

    generate_config(44100, str(tmp_path), "refinegan")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["model"]["architecture_id"] == (
        shipped["model"]["architecture_id"]
    )
    assert written["model"]["refinegan_harmonics"] == (
        shipped["model"]["refinegan_harmonics"]
    )


def test_the_replaced_config_is_kept(tmp_path, shipped):
    """Never a silent overwrite -- the tuning in it may still be wanted."""
    stale = json.loads(json.dumps(shipped))
    stale["model"]["architecture_id"] = "shiro_vits_svae_v2"
    stale["train"]["learning_rate_d"] = 0.0009
    _write(tmp_path / "config.json", stale)

    generate_config(44100, str(tmp_path), "refinegan")

    backup = tmp_path / "config.json.shiro_vits_svae_v2.bak"
    assert backup.exists()
    assert json.loads(backup.read_text(encoding="utf-8"))["train"][
        "learning_rate_d"
    ] == pytest.approx(0.0009)


def test_a_config_that_names_no_architecture_at_all_is_replaced(tmp_path, shipped):
    """The case the guard was blind to, and the one that motivates it.

    An architecture change that renames its option keys leaves every config
    written before it naming no architecture id under the current name.
    Requiring both sides to name one would have kept such a config and
    silently dropped every renamed option -- ``train.py`` fills in only
    *absent* keys, so the old entries would have sat there unread while their
    renamed counterparts took defaults.
    """
    stale = json.loads(json.dumps(shipped))
    del stale["model"]["architecture_id"]
    stale["model"]["refinegan_use_frame_energy"] = True
    _write(tmp_path / "config.json", stale)

    generate_config(44100, str(tmp_path), "refinegan")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["model"]["architecture_id"] == (
        shipped["model"]["architecture_id"]
    )
    assert "refinegan_use_frame_energy" not in written["model"]
    assert (tmp_path / "config.json.unknown.bak").exists()


def test_a_hifi_config_is_untouched_by_the_architecture_guard(tmp_path):
    """HiFi-GAN names no architecture id on either side, so nothing fires."""
    shipped_hifi = json.loads(
        (ROOT / "rvc" / "configs" / "hifi" / "48000.json").read_text(encoding="utf-8")
    )
    tuned = json.loads(json.dumps(shipped_hifi))
    tuned["train"]["learning_rate_d"] = 0.0009
    _write(tmp_path / "config.json", tuned)

    generate_config(48000, str(tmp_path), "hifi")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["train"]["learning_rate_d"] == pytest.approx(0.0009)
    assert not list(tmp_path.glob("*.bak"))


def test_a_matching_config_is_left_exactly_alone(tmp_path, shipped):
    """Per-run tuning is the reason the file is kept; it has to survive."""
    tuned = json.loads(json.dumps(shipped))
    tuned["train"]["learning_rate_d"] = 0.0009
    _write(tmp_path / "config.json", tuned)

    generate_config(44100, str(tmp_path), "refinegan")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["train"]["learning_rate_d"] == pytest.approx(0.0009)
    assert not list(tmp_path.glob("*.bak"))


def test_a_missing_config_is_created(tmp_path, shipped):

    generate_config(44100, str(tmp_path), "refinegan")

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["model"]["architecture_id"] == (
        shipped["model"]["architecture_id"]
    )
