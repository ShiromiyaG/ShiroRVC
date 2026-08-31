"""The anti-image filters on the feature upsamplers of both anti-aliased vocoders.

Measured on a ChouwaGAN pretrain at step 27.5k: every harmonic of the generated
audio carried a symmetric pair of ghost partials at +-100 Hz, 17-27 dB under the
harmonic against 30-58 dB in the reference, and the 5-10 kHz envelope was
modulated at exactly 100 Hz, 39 dB over its own floor (the reference: 1.0 dB).
100 Hz is ``sample_rate / hop_length`` -- the frame grid, not anything musical.

The cause is imaging.  Upsampling replicates the input band, and the replica is
*mirrored*, which is what the ghost pair is.  Both decoders' anti-image filters
were too weak to suppress it: RefineGAN interpolated linearly, and ChouwaGAN's
windowed sinc ran at ``rolloff = 0.95``, leaving a 5% transition band that 25
taps cannot realise.  Neither shows up in a loss curve -- the ghosts are 20 dB
down and in-band -- so it is pinned here instead.

The threshold is on the filters themselves rather than on a decoder's output,
because that is where the property lives: it is linear, deterministic, and
independent of what the trunk was trained to do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip(
    "torch", reason="the resamplers need torch", exc_type=ImportError
)

from torch import nn  # noqa: E402

from rvc.lib.algorithm.generators.chouwagan import ChouwaGANGenerator  # noqa: E402
from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator  # noqa: E402
from rvc.lib.algorithm.resampling import (  # noqa: E402
    AntiAliasedUpsample1d,
    filter_schedule,
)

RATES = (3, 3, 7, 7)
CHOUWAGAN_CONFIG = ROOT / "rvc" / "configs" / "chouwagan" / "44100.json"
REFINEGAN_CONFIG = ROOT / "rvc" / "configs" / "refinegan" / "44100.json"


def _image_energy(module, factor: int, length: int = 4096) -> float:
    """Energy above the input's Nyquist after upsampling a full-band input, in dB.

    White noise occupies the input band right up to its Nyquist, which is what
    the latent entering stage 0 actually does -- ``z`` is drawn per frame, so it
    is white across the frame axis.  Everything the upsampler puts above that
    band is an image it failed to reject.
    """

    torch.manual_seed(0)
    x = torch.randn(1, 1, length)
    with torch.no_grad():
        y = module(x)[0, 0]
    spectrum = torch.fft.rfft(y).abs().square()
    frequency = torch.fft.rfftfreq(y.shape[-1])
    baseband = frequency <= 0.5 / factor
    ratio = spectrum[~baseband].sum() / spectrum[baseband].sum()
    return float(10 * torch.log10(ratio))


# --------------------------------------------------------------------------
# the filters themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [2, 3, 7])
def test_linear_interpolation_barely_rejects_an_image(factor):
    """Documents the baseline both decoders were on, so the number below has a
    scale.  A triangular kernel leaves the images ~12 dB down, which for a
    full-band frame signal is not a filter at all."""

    leak = _image_energy(nn.Upsample(scale_factor=factor, mode="linear"), factor)
    assert -16.0 < leak < -8.0, leak


@pytest.mark.parametrize("factor", [2, 3, 7])
def test_the_shipped_stage_0_design_rejects_images_by_at_least_35_dB(factor):
    """``width = 12, rolloff = 0.88, beta = 6.0`` -- what the low-rate stages of
    both decoders now use.  The old ``width = 4, rolloff = 0.95, beta = 1.5``
    reaches only -23 dB here, and that 16 dB is the whole defect."""

    good = AntiAliasedUpsample1d(factor, filter_width=12, rolloff=0.88, filter_beta=6.0)
    weak = AntiAliasedUpsample1d(factor, filter_width=4, rolloff=0.95, filter_beta=1.5)
    assert _image_energy(good, factor) < -35.0
    assert _image_energy(good, factor) < _image_energy(weak, factor) - 12.0


def test_the_rolloff_is_what_costs_the_bandwidth_not_the_width():
    """A longer kernel at the same rolloff cannot buy the rejection, because the
    transition band is what is missing.  This is the reason the schedule moves
    all three knobs together rather than only widening the filter."""

    long_at_095 = AntiAliasedUpsample1d(
        3, filter_width=12, rolloff=0.95, filter_beta=6.0
    )
    assert _image_energy(long_at_095, 3) > -32.0


def test_an_upsampler_preserves_the_exact_stage_length():
    """Both decoders concatenate the upsampled tensor with a skip taken at that
    stage's rate, so an off-by-one here is a shape error at every call site."""

    x = torch.randn(2, 5, 37)
    for factor in (2, 3, 7):
        module = AntiAliasedUpsample1d(
            factor, filter_width=12, rolloff=0.88, filter_beta=6.0
        )
        assert module(x).shape[-1] == 37 * factor


# --------------------------------------------------------------------------
# the per-stage schedule
# --------------------------------------------------------------------------


def test_filter_schedule_accepts_a_scalar_or_one_value_per_stage():
    assert filter_schedule(0.9, 4, "rolloff") == (0.9, 0.9, 0.9, 0.9)
    assert filter_schedule([1, 2, 3, 4], 4, "width") == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        filter_schedule([0.9, 0.9], 4, "rolloff")
    with pytest.raises(ValueError):
        filter_schedule([4, 4, 0, 4], 4, "width", 1)


def test_a_scalar_rolloff_and_beta_still_mean_the_same_everywhere():
    """The per-stage form is additive.  Bit-identical, not merely close: the
    resamplers are fixed buffers, so any difference here is a silently different
    model on every existing config."""

    def build(**overrides):
        torch.manual_seed(0)
        return ChouwaGANGenerator(
            initial_channel=192,
            gin_channels=256,
            sr=44100,
            upsample_rates=RATES,
            chouwagan_excitation_unet=True,
            **overrides,
        ).eval()

    scalar = build(chouwagan_rolloff=0.95, chouwagan_filter_beta=1.5)
    listed = build(
        chouwagan_rolloff=[0.95] * 4, chouwagan_filter_beta=[1.5] * 4
    )
    listed.load_state_dict(scalar.state_dict())

    latent = torch.randn(2, 192, 16) * 0.5
    f0 = torch.full((2, 16), 180.0)
    g = torch.randn(2, 256, 1)
    with torch.no_grad():
        torch.manual_seed(0)
        a = scalar(latent, f0, g)
        torch.manual_seed(0)
        b = listed(latent, f0, g)
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chouwagan_rolloff": [0.9, 0.9]},
        {"chouwagan_rolloff": [0.9, 0.9, 0.9, 1.4]},
        {"chouwagan_filter_beta": [1.5, 1.5, 1.5]},
        {"chouwagan_filter_beta": [1.5, 1.5, 1.5, -1.0]},
    ],
)
def test_a_schedule_that_does_not_match_the_stages_is_refused(overrides):
    with pytest.raises(ValueError):
        ChouwaGANGenerator(
            initial_channel=192,
            gin_channels=256,
            sr=44100,
            upsample_rates=RATES,
            chouwagan_excitation_unet=True,
            **overrides,
        )


def test_the_shipped_chouwagan_config_protects_the_two_low_rate_stages():
    """Where the +-100 Hz images are born, and the cheap end of the decoder.
    The output stage keeps ``rolloff = 0.95`` on purpose: its pair runs at
    88.2 kHz, so 0.88 there would spend real audio bandwidth."""

    model = json.loads(CHOUWAGAN_CONFIG.read_text())["model"]
    assert model["chouwagan_filter_width"] == [12, 12, 4, 4]
    assert model["chouwagan_rolloff"] == [0.88, 0.88, 0.95, 0.95]
    assert model["chouwagan_filter_beta"] == [6.0, 6.0, 1.5, 1.5]

    decoder = ChouwaGANGenerator(
        initial_channel=192,
        gin_channels=256,
        sr=44100,
        upsample_rates=RATES,
        upsample_initial_channel=model["upsample_initial_channel"],
        **{k: v for k, v in model.items() if k.startswith("chouwagan")},
    )
    # width 12 at factor 3 is 2 * 12 * 3 + 1 taps; width 4 at factor 7 is 57.
    assert [block[0].kernel.shape[-1] for block in decoder.ups] == [73, 73, 57, 57]
    # The latent mixer sits before stage 0 and must follow stage 0's design.
    assert decoder.latent_mixer[0].activation.upsample.kernel.shape[-1] == 49


# --------------------------------------------------------------------------
# RefineGAN
# --------------------------------------------------------------------------


def test_refinegan_upsamples_with_a_windowed_sinc_not_linear_interpolation():
    model = json.loads(REFINEGAN_CONFIG.read_text())["model"]
    generator = RefineGANGenerator(
        sample_rate=44100,
        upsample_rates=model["upsample_rates"],
        num_mels=model["inter_channels"],
        gin_channels=model["gin_channels"],
        upsample_initial_channel=model["upsample_initial_channel"],
    ).eval()

    assert all(
        isinstance(block, AntiAliasedUpsample1d) for block in generator.upsample_blocks
    )
    for block, factor in zip(generator.upsample_blocks, model["upsample_rates"]):
        assert _image_energy(block, factor) < -35.0

    # The upsamplers hold only non-persistent buffers, so this moves no
    # checkpoint key -- which is why it needs a test rather than a load failure
    # to catch a regression back to ``nn.Upsample``.
    assert not any(
        key.startswith("upsample_blocks") for key in generator.state_dict()
    )

    mel = torch.randn(1, model["inter_channels"], 8)
    f0 = torch.full((1, 8), 200.0)
    g = torch.randn(1, 256, 1)
    with torch.no_grad():
        assert generator(mel, f0, g).shape == (1, 1, 8 * 441)
