"""The ChouwaGAN decoder, pinned against its excitation and against 44.1 kHz.

Most of what makes this decoder different from the other two lives in the NSF
source rather than in the trunk, and none of it is visible in a loss curve:

* the harmonic bank is masked as it approaches Nyquist, so the excitation never
  aliases -- get this wrong and the decoder spends capacity cancelling tones it
  was handed, which looks exactly like a decoder that is merely undertrained;
* it is normalised by that mask, so it leaves ``render`` at a scale that does
  not move with f0 or with the harmonic count;
* ``deterministic`` inference draws *seeded* phase offsets and noise rather than
  zeroing them, because zero is not a draw from the training distribution -- it
  is its extreme.

The rest pins the shapes the synthesizer and the exporter rely on.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the decoder needs torch", exc_type=ImportError)

from rvc.lib.algorithm.generators.chouwagan import (  # noqa: E402
    DETERMINISTIC_PHASE_SEED,
    BandLimitedNSFSource,
    ChouwaGANGenerator,
    soft_clip,
)

CONFIG = ROOT / "rvc" / "configs" / "chouwagan" / "44100.json"
RATES = (3, 3, 7, 7)
UPSAMPLE = math.prod(RATES)


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


def _inputs(batch=2, frames=16, channels=192):
    latent = torch.randn(batch, channels, frames) * 0.5
    f0 = torch.full((batch, frames), 180.0)
    g = torch.randn(batch, 256, 1)
    return latent, f0, g


# --------------------------------------------------------------------------
# Shapes and the call site


def test_it_upsamples_by_exactly_the_product_of_its_rates():
    decoder = _generator().eval()
    latent, f0, g = _inputs()

    with torch.no_grad():
        audio = decoder(latent, f0, g)

    assert audio.shape == (latent.shape[0], 1, latent.shape[-1] * UPSAMPLE)


def test_a_non_44100_configuration_is_refused():
    """Every rate here -- the Nyquist mask, the crossfade, the anti-aliasing
    kernels -- is derived from 44.1 kHz, so a 48 kHz run would train happily and
    be wrong about all three."""
    with pytest.raises(ValueError):
        _generator(sr=48000)


def test_the_channel_and_upsample_schedules_have_to_agree():
    with pytest.raises(ValueError):
        _generator(chouwagan_channels=(256, 160, 80))
    with pytest.raises(ValueError):
        _generator(chouwagan_expansion=(2, 2, 1))


def test_the_template_amplitude_the_synthesizer_passes_is_accepted_and_inert():
    """``Synthesizer`` hands every VITS-latent decoder the measured frame
    energy, because RefineGAN's pulse template scaled its impulses by it.  This
    excitation is normalised by construction, so the argument has to be accepted
    -- the call site is shared -- and has to change nothing."""
    decoder = _generator().eval()
    latent, f0, g = _inputs()
    amplitude = torch.rand(latent.shape[0], 1, latent.shape[-1])

    with torch.no_grad():
        torch.manual_seed(1)
        without = decoder(latent, f0, g)
        torch.manual_seed(1)
        with_amplitude = decoder(latent, f0, g, template_amplitude=amplitude)

    assert torch.equal(without, with_amplitude)


# --------------------------------------------------------------------------
# The output head


def test_the_output_stays_inside_the_ceiling():
    decoder = _generator().eval()
    latent, f0, g = _inputs()

    with torch.no_grad():
        audio = decoder(latent * 40.0, f0, g)

    assert float(audio.abs().max()) <= decoder.output_head_ceiling + 1e-4


def test_the_limiter_is_linear_below_its_threshold():
    """A plain ``tanh`` reaches a given loudness only from large
    pre-activations, which parks the operating point where its own gradient
    vanishes.  Staying exactly linear below the threshold is the whole point of
    the soft clip, so it is asserted rather than assumed."""
    quiet = torch.linspace(-0.8, 0.8, 33)

    assert torch.allclose(soft_clip(quiet, 0.85, 1.0), quiet)
    # ...and bounded above it, however hard it is driven.
    assert float(soft_clip(torch.tensor([50.0]), 0.85, 1.0)) <= 1.0
    assert float(soft_clip(torch.tensor([0.9]), 0.85, 1.0)) < 0.9


def test_the_dc_blocker_removes_the_mean_without_rescaling_healthy_peaks():
    decoder = _generator().eval()
    latent, f0, g = _inputs()

    with torch.no_grad():
        audio = decoder(latent, f0, g)

    assert float(audio.mean(dim=-1).abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert float(audio.abs().max()) > 1e-4


# --------------------------------------------------------------------------
# The excitation


def _source(**overrides) -> BandLimitedNSFSource:
    kwargs = dict(sample_rate=44100, harmonic_count=64, noise_std=0.01)
    kwargs.update(overrides)
    return BandLimitedNSFSource(**kwargs)


def test_harmonics_past_nyquist_contribute_nothing():
    """At f0=600 the 37th harmonic is already at Nyquist, so harmonics 38 and up
    exist only to alias back into the top of the band as inharmonic tones the
    decoder would then have to learn to cancel.  The mask is what stops them,
    and the test for it is that adding them changes nothing: 128 harmonics must
    render the same excitation as 64.

    Seeded, because the phase offsets are drawn per call and the leading 64 only
    match across the two banks if both draws start from the same generator.
    """
    f0 = torch.full((1, 8), 600.0)
    excitations = []
    for count in (64, 128):
        source = _source(harmonic_count=count).eval()
        source.deterministic = True
        excitations.append(source(f0, 4410, 44100.0))

    assert torch.allclose(excitations[0], excitations[1], atol=2e-3)


def test_the_top_harmonic_keeps_its_phase_once_the_accumulator_is_large():
    """``prepare`` used to accumulate phase as unbounded float32 radians and
    then multiply by the harmonic index, so the top harmonic's phase reached
    4.8e6 rad over a 30 s span -- where a float32 ULP is 0.5 rad -- and the
    error grew along the span.  Training reads 0.4 s segments, where it is
    inaudible, so only previews and inference ever met it: a defect the decoder
    cannot be taught to cancel because it never sees it.

    Driven by f0 rather than by length, to reach the same accumulator without
    allocating a 30 s harmonic bank.  1 s at 6 kHz is 6000 cycles and the 128th
    harmonic of that is the same 4.8e6 rad.  The bank ``prepare`` returns is
    unmasked, so the harmonic is there whatever Nyquist says about it.

    The deterministic offsets are reproduced from the same seed, which is what
    lets this compare against float64 term by term rather than in aggregate.
    """
    sample_rate = 44100.0
    length = int(sample_rate)
    harmonic_count = 128
    f0 = torch.full((1, 16), 6000.0)

    source = _source(harmonic_count=harmonic_count).eval()
    source.deterministic = True
    resampled, _, _, harmonic_wave, _, _ = source.prepare(f0, length)

    generator = torch.Generator()
    generator.manual_seed(DETERMINISTIC_PHASE_SEED)
    offsets = torch.zeros(1, 1, harmonic_count)
    offsets[..., 1:] = torch.rand(
        (1, 1, harmonic_count - 1), generator=generator
    ) * (2.0 * math.pi)

    phase = torch.cumsum(2.0 * math.pi * resampled.double() / sample_rate, dim=-1)
    for harmonic in (1, harmonic_count):
        reference = torch.sin(
            phase * harmonic + float(offsets[0, 0, harmonic - 1])
        ).float()
        error = harmonic_wave[..., harmonic - 1] - reference
        level = 20.0 * math.log10(float(error.pow(2).mean().sqrt()) + 1e-12)
        assert level < -80.0, f"harmonic {harmonic} is {level:.1f} dB off"


def test_the_excitation_scale_does_not_move_with_the_harmonic_count():
    """It is divided by its own masked normaliser, which is what lets the
    decoder's ``source_gates`` and fusion weights mean the same thing across
    configurations."""
    f0 = torch.full((1, 8), 180.0)
    levels = [
        float(_source(harmonic_count=count).eval()(f0, 4410, 44100.0).std())
        for count in (8, 32, 128)
    ]

    assert max(levels) < 1.6 * min(levels)


def test_unvoiced_frames_keep_their_noise_under_deterministic_inference():
    """Zeroing it there is not "reproducible", it is silence: the noise term is
    the *entire* excitation on an unvoiced frame, and the decoder never trained
    on a silent one.  The symptom is a hard cut at the end of every utterance."""
    source = _source().eval()
    source.deterministic = True
    f0 = torch.zeros(1, 8)

    excitation = source(f0, 4410, 44100.0)

    assert float(excitation.abs().max()) > 0.0
    # ...and reproducible, which is what ``deterministic`` was asked for.
    assert torch.equal(excitation, source(f0, 4410, 44100.0))


def test_deterministic_phase_offsets_are_drawn_rather_than_aligned():
    """Aligning every harmonic maximises the crest factor -- the one excitation
    shape training never produced."""
    source = _source(harmonic_count=32).eval()
    source.deterministic = True
    f0 = torch.full((1, 8), 200.0)

    _f0, _voiced, _envelope, harmonic_wave, _noise, _harmonicity = source.prepare(
        f0, 4410
    )
    offsets_at_zero = harmonic_wave[:, 0, 1:]

    assert float(offsets_at_zero.abs().max()) > 1e-3


def test_the_voiced_boundary_is_crossfaded_rather_than_stepped():
    """A 0/1 amplitude mask splices a discontinuity into the excitation, which
    is a broadband click on every voiced/unvoiced edge."""
    source = _source().eval()
    f0 = torch.cat((torch.full((1, 4), 180.0), torch.zeros(1, 4)), dim=-1)

    excitation = source(f0, 4410, 44100.0).squeeze()
    steps = excitation.diff().abs()
    boundary = 4410 // 2

    # Read against the voiced region's own sample-to-sample movement: a 64
    # harmonic bank is pulse-like and jumps by order 1 all through it, so an
    # absolute threshold would say nothing.  What must not happen is a *larger*
    # jump at the edge than anywhere inside.
    inside = float(steps[: boundary - 200].max())
    at_edge = float(steps[boundary - 100 : boundary + 100].max())

    assert at_edge <= inside


# --------------------------------------------------------------------------
# The excitation U-Net


def test_every_stage_fuses_the_skip_taken_at_its_own_resolution():
    """Skips leave the encoder highest-rate first and are consumed reversed;
    walking them forwards would concatenate the wrong width at every stage but
    the middle one."""
    decoder = _generator()

    assert len(decoder.exc_skip_channels) == len(RATES)
    for stage, conv in enumerate(decoder.fusion_proj):
        skip = decoder.exc_skip_channels[len(RATES) - 1 - stage]
        assert conv.weight.shape[1] == decoder.channels[stage] + skip


def test_the_additive_path_is_the_other_design_not_a_degraded_one():
    """With the U-Net off the source is re-rendered per stage and gated in, so
    the encoder disappears entirely and ``source_gates`` appears in its place.
    Both have to build and run; the config chooses."""
    decoder = _generator(chouwagan_excitation_unet=False).eval()
    latent, f0, g = _inputs()

    assert decoder.exc_encoder is None
    assert decoder.source_gates.shape == (len(RATES),)
    with torch.no_grad():
        audio = decoder(latent, f0, g)
    assert audio.shape[-1] == latent.shape[-1] * UPSAMPLE


# --------------------------------------------------------------------------
# Export


def test_remove_weight_norm_preserves_the_output():
    decoder = _generator().eval()
    # The excitation is redrawn on every call unless this is set, so without it
    # the comparison below would be measuring the RNG.
    decoder.source.deterministic = True
    latent, f0, g = _inputs()

    with torch.no_grad():
        before = decoder(latent, f0, g)
        decoder.remove_weight_norm()
        after = decoder(latent, f0, g)

    assert torch.allclose(before, after, atol=1e-4)


def test_the_shipped_config_builds_the_decoder_it_describes():
    model = json.loads(CONFIG.read_text(encoding="utf-8"))["model"]
    decoder = ChouwaGANGenerator(
        initial_channel=model["inter_channels"],
        gin_channels=model["gin_channels"],
        sr=44100,
        **{
            key: value
            for key, value in model.items()
            if key.startswith("chouwagan_") or key == "upsample_rates"
        },
        upsample_initial_channel=model["upsample_initial_channel"],
    )

    assert decoder.total_upsample == 441, "one frame per hop at 44.1 kHz"
    assert decoder.channels == tuple(model["chouwagan_channels"])
    assert decoder.excitation_unet is model["chouwagan_excitation_unet"]
    assert decoder.source.harmonic_count == model["chouwagan_harmonics"]


def test_the_filter_width_key_reaches_the_activation_pair():
    """It was hardcoded to 4 in ``AntiAliasedSnakeBeta`` while threaded elsewhere.

    So the one config key you would reach for to make the 21 activation
    resamplers cheaper controlled the excitation encoder and the upsamplers and
    nothing else.  Setting it measured as a 2% change -- noise -- because it was
    not connected to what costs the time.
    """
    from rvc.lib.algorithm.generators.chouwagan import AntiAliasedSnakeBeta

    taps = {}
    for width in (2, 4):
        act = AntiAliasedSnakeBeta(8, rolloff=0.95, filter_width=width)
        taps[width] = act.downsample.kernel.shape[-1]
    assert taps[2] < taps[4], taps
    assert taps == {2: 9, 4: 17}, taps


def test_skipping_the_antialias_pair_changes_no_parameter():
    """Which stages get the pair has to be free to choose, and it is.

    ``alpha`` and ``beta`` are the only weights in the activation; the
    resamplers are fixed windowed-sinc buffers registered non-persistently.  So
    ``chouwagan_antialias_stages`` never moves a checkpoint key.
    """
    from rvc.lib.algorithm.generators.chouwagan import AntiAliasedSnakeBeta

    with_pair = AntiAliasedSnakeBeta(8, antialias=True)
    without = AntiAliasedSnakeBeta(8, antialias=False)
    assert set(with_pair.state_dict()) == set(without.state_dict())
    assert sum(p.numel() for p in with_pair.parameters()) == sum(
        p.numel() for p in without.parameters()
    )
    x = torch.randn(2, 8, 64)
    assert torch.isfinite(without(x)).all()
    assert without(x).shape == with_pair(x).shape


# --------------------------------------------------------------------------
# Residual unit style


def test_the_default_unit_style_is_still_the_separable_one():
    """A config predating the key has to build the previous model exactly."""
    from rvc.lib.algorithm.generators.chouwagan import DepthwiseSeparableUnit

    decoder = _generator()
    assert decoder.unit_style == "separable"
    assert isinstance(decoder.blocks[0].branches[0].units[0], DepthwiseSeparableUnit)
    assert isinstance(decoder.latent_mixer[0], DepthwiseSeparableUnit)


def test_the_dense_unit_runs_one_convolution_where_the_separable_runs_three():
    """The whole point of the style: dispatches, not arithmetic.

    The training step is CPU-bound end to end, and the
    ``conv1d -> convolution -> _convolution -> cudnn_convolution`` chain costs
    ~63 us of CPU per call.  Counting the convolutions is therefore counting the
    cost, which is why this pins the count and not a wall clock.
    """
    from rvc.lib.algorithm.generators.chouwagan import DenseUnit

    separable = _generator().blocks[0].branches[0].units[0]
    dense = _generator(chouwagan_unit_style="dense").blocks[0].branches[0].units[0]

    assert isinstance(dense, DenseUnit)
    n = lambda unit: sum(isinstance(m, torch.nn.Conv1d) for m in unit.modules())
    assert n(separable) == 3
    assert n(dense) == 1


def test_the_dense_unit_keeps_the_decoder_contract():
    decoder = _generator(chouwagan_unit_style="dense").eval()
    latent, f0, g = _inputs()

    with torch.no_grad():
        audio = decoder(latent, f0, g)

    assert audio.shape == (latent.shape[0], 1, latent.shape[-1] * UPSAMPLE)
    assert torch.isfinite(audio).all()


def test_the_dense_unit_keeps_the_receptive_field_of_the_one_it_replaces():
    """Same kernel, same dilation: only the bottleneck goes away."""
    separable = _generator().blocks[0].branches[0].units[0]
    dense = _generator(chouwagan_unit_style="dense").blocks[0].branches[0].units[0]

    assert dense.conv.kernel_size == separable.depthwise.kernel_size
    assert dense.conv.dilation == separable.depthwise.dilation
    assert dense.conv.padding == separable.depthwise.padding


def test_the_two_styles_do_not_share_a_state_dict():
    """Which is why the architecture id moves with this option.

    ``net_g`` loads non-strictly, so without the guard a ``v2`` checkpoint would
    load into a dense trunk with every block silently left at its init.
    """
    separable = set(_generator().state_dict())
    dense = set(_generator(chouwagan_unit_style="dense").state_dict())
    assert separable != dense


def test_an_unknown_unit_style_fails_at_construction():
    with pytest.raises(ValueError):
        _generator(chouwagan_unit_style="depthwise")
