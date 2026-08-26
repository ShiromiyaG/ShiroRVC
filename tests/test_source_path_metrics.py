"""The excitation-path series, and why the decoder needs one.

Dissecting the stock RVC v2 HiFi-GAN pretrains turned up one property that
separated them from every short reproduction: the strength of the NSF source
injection at the *high-rate* stages.  Their ``noise_convs`` sit at 1.4-2.0x of
initialisation on the first, lowest-rate stage and at 0.4-8% of it on the other
three -- the harmonic source enters once, filtered by 23 dB, and the network
synthesises the rest of the harmonic structure itself.  A fresh model injects a
full-band comb at every stage, and the last of those convolutions is a
single-tap ``stride=1, kernel=1`` with no filter at all.  That is where RVC's
mirroring comes from, and the decay to 0.4% is passive: nothing in a mel loss
sees a spectral image.

ChouwaGAN cannot alias that way -- every stage renders the source band-limited
to its own rate -- but *how much* excitation each stage leans on is the same
maturity signal, and nothing was reading it.  The gates existed and were never
logged.

This pins the collector to the two decoder layouts, which express the same
quantity differently: the U-Net path concatenates an excitation skip and lets
``fusion_proj`` weigh it, the legacy path adds a gated projection.  A ratio that
silently reads the wrong half of a concatenated weight would look like a
plausible number forever.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the collector reads real weights", exc_type=ImportError)

from rvc.lib.algorithm.generators.chouwagan import ChouwaGANGenerator  # noqa: E402

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
RATES = (3, 3, 7, 7)
CHANNELS = (256, 160, 80, 40)


@pytest.fixture(scope="module")
def collect():
    """Lift the collector out of ``train.py``, whose module body reads argv."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "collect_source_path_metrics":
            namespace: dict = {"torch": torch}
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
            return namespace["collect_source_path_metrics"]
    pytest.fail("train.py no longer defines collect_source_path_metrics")


def _generator(excitation_unet: bool) -> ChouwaGANGenerator:
    torch.manual_seed(0)
    return ChouwaGANGenerator(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=RATES,
        upsample_initial_channel=320,
        chouwagan_channels=CHANNELS,
        chouwagan_block_kernels=((3, 7), (3, 7), (7,), (7,)),
        chouwagan_expansion=(2, 2, 1, 1),
        chouwagan_excitation_unet=excitation_unet,
    )


class _Model(torch.nn.Module):
    """Stands in for the Synthesizer: the collector only reaches ``.dec``."""

    def __init__(self, decoder):
        super().__init__()
        self.dec = decoder


class _Wrapped:
    """Stands in for DDP, whose real module hides one attribute deeper."""

    def __init__(self, model):
        self.module = model


@pytest.mark.parametrize("excitation_unet", [True, False])
def test_every_stage_reports_a_ratio(collect, excitation_unet):
    metrics = collect(_Model(_generator(excitation_unet)))

    for stage in range(len(RATES)):
        key = f"Source/inject_ratio_stage_{stage}"
        assert key in metrics, f"{key} missing on excitation_unet={excitation_unet}"
        assert metrics[key] > 0.0, "a live excitation path cannot read as zero"
    # The output-rate stage is the one whose excitation share lands in the top
    # octave, so it gets a flat alias rather than an index the reader must know.
    assert metrics["Source/inject_ratio_output_stage"] == pytest.approx(
        metrics[f"Source/inject_ratio_stage_{len(RATES) - 1}"]
    )


def test_gates_are_reported_per_layout(collect):
    """Each layout exposes the gate it actually has, and not the other one."""
    unet = collect(_Model(_generator(True)))
    legacy = collect(_Model(_generator(False)))

    # sigmoid(log(0.25/0.75)) is the shared initialisation of both gate kinds.
    assert unet["Source/bottleneck_gate"] == pytest.approx(0.25, abs=1e-6)
    assert not any(key.startswith("Source/gate_stage_") for key in unet)

    assert "Source/bottleneck_gate" not in legacy
    for stage in range(len(RATES)):
        assert legacy[f"Source/gate_stage_{stage}"] == pytest.approx(0.25, abs=1e-6)


def test_the_gate_moves_the_legacy_ratio(collect):
    """The gate is what scales the injection, so closing it has to show up."""
    decoder = _generator(False)
    before = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    with torch.no_grad():
        decoder.source_gates.fill_(-8.0)  # sigmoid -> ~0.0003
    after = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    assert after < before / 100, "a shut gate must read as a collapsed injection"


def test_unet_ratio_splits_the_fusion_weight_the_way_the_decoder_does(collect):
    """``forward`` concatenates ``(x, skip)``; reading the halves backwards
    would report the main path as the excitation and never look wrong."""
    decoder = _generator(True)
    stage = 0
    channels = CHANNELS[stage]

    with torch.no_grad():
        # Zero the excitation columns only.  The ratio has to go to zero; if the
        # halves were swapped it would go to infinity instead.  Writing through
        # the parametrisation, because ``.weight`` under ``weight_norm`` is a
        # recomputed view and assigning to it is a no-op.
        decoder.fusion_proj[stage].parametrizations.weight.original1[
            :, channels:
        ] = 0.0
    metrics = collect(_Model(decoder))

    assert metrics[f"Source/inject_ratio_stage_{stage}"] == pytest.approx(0.0, abs=1e-6)
    assert metrics[f"Source/inject_ratio_stage_{stage + 1}"] > 0.0


def test_ddp_wrapper_is_unwrapped(collect):
    model = _Model(_generator(True))
    assert collect(_Wrapped(model)) == collect(model)


def test_a_non_chouwagan_decoder_reports_nothing(collect):
    """The call site guards on the vocoder, but the collector is also asked
    about HiFi-GAN by anything that reuses it, and must not raise."""

    class _HiFiish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ups = torch.nn.ModuleList([torch.nn.Conv1d(4, 4, 1)])

    assert collect(_Model(_HiFiish())) == {}
    assert collect(_Model(torch.nn.Identity())) == {}
