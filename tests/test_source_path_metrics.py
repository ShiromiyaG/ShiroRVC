"""The excitation-path series, and why the decoder needs one.

Dissecting the stock RVC v2 HiFi-GAN pretrains turned up one property that
separated them from every short reproduction: the strength of the source
injection at the *high-rate* stages.  Their ``noise_convs`` sit at 1.4-2.0x of
initialisation on the first, lowest-rate stage and at 0.4-8% of it on the other
three -- the harmonic source enters once, filtered by 23 dB, and the network
synthesises the rest of the harmonic structure itself.  A fresh model injects a
full-band comb at every stage, and the last of those convolutions is a
single-tap ``stride=1, kernel=1`` with no filter at all.  That is where RVC's
mirroring comes from, and the decay to 0.4% is passive: nothing in a mel loss
sees a spectral image.

RefineGAN makes the same quantity directly readable.  Each decoder stage opens
by concatenating its upsampled activation with the encoder skip taken at that
resolution, and the encoder sees nothing but the pitch template -- so the skip
*is* the excitation, and ``ParallelResBlock.input_conv`` is the 1x1 that weighs
it.  A ratio that silently read the wrong half of that concatenated weight would
report the main path as the excitation and look like a plausible number forever,
which is what these tests exist to prevent.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the collector reads real weights", exc_type=ImportError)

from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator  # noqa: E402

TRAIN_PY = ROOT / "rvc" / "train" / "train.py"
RATES = (3, 3, 7, 7)


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


def _generator(**overrides) -> RefineGANGenerator:
    torch.manual_seed(0)
    kwargs = dict(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=RATES,
    )
    kwargs.update(overrides)
    return RefineGANGenerator(**kwargs)


#: The shipped schedule, which deliberately does not halve.
WIDE = dict(
    refinegan_decoder_channels=[256, 128, 96, 64],
    refinegan_resblock_kernel_sizes=[[3, 7, 11], [3, 7, 11], [11], [11]],
    refinegan_resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3], [1, 3]],
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


def test_every_stage_reports_a_ratio(collect):
    metrics = collect(_Model(_generator()))

    for stage in range(len(RATES)):
        key = f"Source/inject_ratio_stage_{stage}"
        assert key in metrics, f"{key} missing"
        assert metrics[key] > 0.0, "a live excitation path cannot read as zero"
    # The output-rate stage is the one whose excitation share lands in the top
    # octave, so it gets a flat alias rather than an index the reader must know.
    assert metrics["Source/inject_ratio_output_stage"] == pytest.approx(
        metrics[f"Source/inject_ratio_stage_{len(RATES) - 1}"]
    )


def test_the_ratio_splits_the_input_conv_the_way_the_decoder_does(collect):
    """``forward_core`` concatenates ``(upsampled, skip)``; reading the halves
    backwards would report the main path as the excitation and never look
    wrong."""
    decoder = _generator(**WIDE)
    stage = 0
    conv = decoder.upsample_conv_blocks[stage].input_conv
    skip_channels = decoder.skip_channels[stage]

    with torch.no_grad():
        # Zero the excitation columns only.  The ratio has to go to zero; if the
        # halves were swapped it would go to infinity instead.  Writing through
        # the parametrisation, because ``.weight`` under ``weight_norm`` is a
        # recomputed view and assigning to it is a no-op.
        conv.parametrizations.weight.original1[:, -skip_channels:] = 0.0
    metrics = collect(_Model(decoder))

    assert metrics[f"Source/inject_ratio_stage_{stage}"] == pytest.approx(0.0, abs=1e-6)
    assert metrics[f"Source/inject_ratio_stage_{stage + 1}"] > 0.0


@pytest.mark.parametrize("overrides", [{}, WIDE], ids=["paper", "shipped"])
def test_the_split_is_read_from_the_decoder_not_inferred(collect, overrides):
    """Inferring it as ``in_channels // 3`` is only right under the halving.

    The shipped schedule stops halving at the top, so at stages 2 and 3 that
    inference would hand 10 and 10 channels of the *main* path to the
    excitation and report a ratio that looks plausible and is wrong.
    """
    decoder = _generator(**overrides)

    assert len(decoder.skip_channels) == len(RATES)
    for index, (upsample, block) in enumerate(
        zip(decoder.upsample_blocks, decoder.upsample_conv_blocks)
    ):
        skip = decoder.skip_channels[index]
        assert block.input_conv.weight.shape[1] == upsample.out_channels + skip


def test_zeroing_the_skip_zeroes_the_ratio_on_the_shipped_schedule(collect):
    """The end-to-end guard that the split lands on the right columns."""
    decoder = _generator(**WIDE)
    for stage in range(len(RATES)):
        conv = decoder.upsample_conv_blocks[stage].input_conv
        skip = decoder.skip_channels[stage]
        with torch.no_grad():
            conv.parametrizations.weight.original1[:, -skip:] = 0.0
    metrics = collect(_Model(decoder))

    for stage in range(len(RATES)):
        assert metrics[f"Source/inject_ratio_stage_{stage}"] == pytest.approx(0.0, abs=1e-6)


def test_a_shrinking_excitation_column_lowers_the_ratio(collect):
    """The series has to be monotone in the thing it claims to measure."""
    decoder = _generator(**WIDE)
    before = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    conv = decoder.upsample_conv_blocks[0].input_conv
    skip_channels = decoder.skip_channels[0]
    with torch.no_grad():
        conv.parametrizations.weight.original1[:, -skip_channels:] *= 0.01
    after = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    assert after < before / 10


def test_ddp_wrapper_is_unwrapped(collect):
    model = _Model(_generator())
    assert collect(_Wrapped(model)) == collect(model)


def test_a_non_refinegan_decoder_reports_nothing(collect):
    """The call site guards on the vocoder, but the collector is also asked
    about HiFi-GAN by anything that reuses it, and must not raise."""

    class _HiFiish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ups = torch.nn.ModuleList([torch.nn.Conv1d(4, 4, 1)])
            self.skip_channels = []

    assert collect(_Model(_HiFiish())) == {}
    assert collect(_Model(torch.nn.Identity())) == {}
