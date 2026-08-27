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

ChouwaGAN's excitation U-Net makes the same quantity directly readable.  Each
decoder stage opens by concatenating its upsampled activation with the
excitation skip taken at that resolution, and the excitation encoder sees
nothing but the NSF source -- so the skip *is* the excitation, and
``fusion_proj`` is the 1x1 that weighs it.  A ratio that silently read the wrong
half of that concatenated weight would report the main path as the excitation
and look like a plausible number forever, which is what these tests exist to
prevent.  So would reading the skip widths in build order: the encoder produces
them highest-rate first and the decoder consumes them reversed, and the two
orders only disagree numerically, never structurally.
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


#: ``collect_source_path_metrics`` delegates to the Wavehax reader when the
#: decoder has no stages to walk, so both have to come across together.
_LIFTED = ("collect_source_path_metrics", "_wavehax_source_path_metrics")


@pytest.fixture(scope="module")
def collect():
    """Lift the collector out of ``train.py``, whose module body reads argv."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    namespace: dict = {"torch": torch}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _LIFTED:
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
    missing = [name for name in _LIFTED if name not in namespace]
    if missing:
        pytest.fail(f"train.py no longer defines {missing}")
    return namespace["collect_source_path_metrics"]


def _generator(**overrides) -> ChouwaGANGenerator:
    torch.manual_seed(0)
    kwargs = dict(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=RATES,
        chouwagan_excitation_unet=True,
    )
    kwargs.update(overrides)
    return ChouwaGANGenerator(**kwargs)


def _skip_width(decoder, stage: int) -> int:
    """The skip width stage ``stage`` concatenates, in the decoder's own order."""
    return int(decoder.exc_skip_channels[len(decoder.channels) - 1 - stage])


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


def test_the_ratio_splits_the_fusion_conv_the_way_the_decoder_does(collect):
    """The decoder concatenates ``(upsampled, skip)``; reading the halves
    backwards would report the main path as the excitation and never look
    wrong."""
    decoder = _generator()
    stage = 0
    conv = decoder.fusion_proj[stage]
    skip_channels = _skip_width(decoder, stage)

    with torch.no_grad():
        # Zero the excitation columns only.  The ratio has to go to zero; if the
        # halves were swapped it would go to infinity instead.  Writing through
        # the parametrisation, because ``.weight`` under ``weight_norm`` is a
        # recomputed view and assigning to it is a no-op.
        conv.parametrizations.weight.original1[:, -skip_channels:] = 0.0
    metrics = collect(_Model(decoder))

    assert metrics[f"Source/inject_ratio_stage_{stage}"] == pytest.approx(0.0, abs=1e-6)
    assert metrics[f"Source/inject_ratio_stage_{stage + 1}"] > 0.0


def test_the_skip_widths_are_read_in_the_decoders_order(collect):
    """``exc_skip_channels`` is in *encoder* order -- highest rate first.

    Stage ``i`` concatenates ``exc_skip_channels[n - 1 - i]``, so walking the
    tuple forwards would take the wrong number of columns at every stage but the
    middle one, and still produce a finite, plausible ratio.
    """
    decoder = _generator()

    assert len(decoder.exc_skip_channels) == len(RATES)
    assert decoder.exc_skip_channels != tuple(reversed(decoder.exc_skip_channels))
    for stage, conv in enumerate(decoder.fusion_proj):
        expected = decoder.channels[stage] + _skip_width(decoder, stage)
        assert conv.weight.shape[1] == expected


def test_zeroing_every_skip_zeroes_every_ratio(collect):
    """The end-to-end guard that the split lands on the right columns."""
    decoder = _generator()
    for stage in range(len(RATES)):
        conv = decoder.fusion_proj[stage]
        with torch.no_grad():
            conv.parametrizations.weight.original1[
                :, -_skip_width(decoder, stage) :
            ] = 0.0
    metrics = collect(_Model(decoder))

    for stage in range(len(RATES)):
        assert metrics[f"Source/inject_ratio_stage_{stage}"] == pytest.approx(0.0, abs=1e-6)


def test_a_shrinking_excitation_column_lowers_the_ratio(collect):
    """The series has to be monotone in the thing it claims to measure."""
    decoder = _generator()
    before = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    conv = decoder.fusion_proj[0]
    skip_channels = _skip_width(decoder, 0)
    with torch.no_grad():
        conv.parametrizations.weight.original1[:, -skip_channels:] *= 0.01
    after = collect(_Model(decoder))["Source/inject_ratio_stage_0"]

    assert after < before / 10


def test_ddp_wrapper_is_unwrapped(collect):
    model = _Model(_generator())
    assert collect(_Wrapped(model)) == collect(model)


def test_the_additive_excitation_path_reports_nothing(collect):
    """With the U-Net off the source is added through ``source_gates``, not
    concatenated, so there is no mixing weight to split -- and the collector
    must say so rather than inventing a ratio out of the upsampling conv."""
    decoder = _generator(chouwagan_excitation_unet=False)

    assert decoder.exc_skip_channels == ()
    assert collect(_Model(decoder)) == {}


def test_a_decoder_without_an_excitation_path_reports_nothing(collect):
    """The call site guards on the vocoder, but the collector is also asked
    about HiFi-GAN by anything that reuses it, and must not raise."""

    class _HiFiish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ups = torch.nn.ModuleList([torch.nn.Conv1d(4, 4, 1)])
            self.skip_channels = []

    assert collect(_Model(_HiFiish())) == {}
    assert collect(_Model(torch.nn.Identity())) == {}
