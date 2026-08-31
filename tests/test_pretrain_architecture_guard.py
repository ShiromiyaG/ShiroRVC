"""A pretrain that carries no ``architecture_id`` is legacy, not wrong.

HiFi-GAN fine-tuning could not start at all:

    [ ] Loading pretrained (G) 'rvc\\models\\pretraineds\\hifi-gan\\f0G32k.pth'
    ValueError: Pretrained generator architecture mismatch: expected
    'hifi_gan_nsf_v1'.

The guard read ``checkpoint.get("architecture_id")`` and compared it straight
against the synthesizer's.  Every stock RVC v2 checkpoint predates that key --
the bundled ``hifi-gan/f0G{32,40,48}k.pth`` carry ``iteration`` and
``learning_rate`` and nothing else -- so ``None != "hifi_gan_nsf_v1"`` rejected
the shipped pretrains for the shipped vocoder.

Two things hid it.  The resume guard treats an absent id as a mismatch *on
purpose*, and correctly: a ``G_*.pth`` in the experiment folder that predates
the key is one of this fork's own old runs, and those layouts have since
changed.  A pretrain is the opposite case -- it comes from upstream, where the
key never existed.  And HiFi-GAN is the only architecture the check can reach:
both VITS-latent vocoders report ``vits_gaussian_v1``, whose opt-out exists for
exactly this reason (an Applio checkpoint carries no id either).  The oldest
path was the one left out of the exemption written for it.

What this file pins: absent means unchecked on the pretrained path, present and
disagreeing is still refused, and the resume path keeps its stricter rule.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
VOCODERS = ROOT / "rvc" / "configs" / "vocoders.json"
PRETRAIN_DIR = ROOT / "rvc" / "models" / "pretraineds" / "hifi-gan"


def _guard_source() -> str:
    """The pretrained-path architecture check, as source text."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    marker = "Pretrained generator architecture mismatch"
    assert marker in source, "train.py no longer guards the pretrained generator"
    start = source.rindex("if (", 0, source.index(marker))
    return source[start : source.index(marker)]


def test_hifigan_is_the_only_architecture_the_guard_can_reach():
    """The premise.  If the VITS-latent vocoders stopped opting out, the guard
    would need a different fix than this one."""
    registry = json.loads(VOCODERS.read_text())

    def architecture(name):
        spec = registry[name]
        return spec.get("gaussian_architecture_id") or spec["architecture_id"]

    assert architecture("hifi") == "hifi_gan_nsf_v1"
    assert architecture("chouwagan") == "vits_gaussian_v1"
    assert architecture("refinegan") == "vits_gaussian_v1"


def test_an_absent_id_is_not_a_mismatch_on_the_pretrained_path():
    guard = _guard_source()
    assert "checkpoint_architecture is not None" in guard, (
        "an id-less pretrain is rejected again; stock RVC v2 checkpoints carry "
        "no architecture_id and this blocks every HiFi-GAN fine-tune"
    )
    assert "checkpoint_architecture != expected_architecture" in guard, (
        "a checkpoint that names a different architecture must still be refused"
    )


def test_the_resume_path_keeps_the_stricter_rule():
    """Deliberately not relaxed: a ``G_*.pth`` predating the key is one of this
    fork's own old runs, which is exactly the layout that has changed."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    assert "Cannot resume from" in source
    resume = source[source.index("Cannot resume from") - 900 : source.index("Cannot resume from")]
    assert "if found == expected:" in resume, (
        "the resume guard now accepts an id it did not verify"
    )


@pytest.mark.skipif(
    not (PRETRAIN_DIR / "f0G32k.pth").exists(),
    reason="the bundled pretrains are downloaded, not committed",
)
@pytest.mark.parametrize("name", ["f0G32k.pth", "f0G40k.pth", "f0G48k.pth"])
def test_the_bundled_hifigan_pretrains_really_do_lack_the_key(name):
    """The fact the guard has to accommodate, read off the actual artifacts."""
    torch = pytest.importorskip("torch", exc_type=ImportError)

    checkpoint = torch.load(PRETRAIN_DIR / name, map_location="cpu", weights_only=True)
    assert "model" in checkpoint
    assert checkpoint.get("architecture_id") is None
    assert sorted(k for k in checkpoint if k != "model") == [
        "iteration",
        "learning_rate",
    ]
