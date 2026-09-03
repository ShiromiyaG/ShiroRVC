"""The decoder settings that leave no trace in the weights.

A reorder keeps all 271 tensors' keys *and* shapes; an ``AntiAliasedActivation``
registers its kernels non-persistently and adds no key.  So a checkpoint loads
cleanly into a decoder whose signal path it was never trained for, and
``decoder_layout`` is the only thing that can say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

import torch.nn.functional as F  # noqa: E402

from rvc.lib.algorithm.generators.refinegan2 import (  # noqa: E402
    ParallelResBlock,
    ANTIALIAS_MODES,
    ResBlock,
    RefineGAN2Generator,
)
from rvc.lib.algorithm.resampling import AntiAliasedActivation  # noqa: E402
from rvc.train.utils import (  # noqa: E402
    LEGACY_ANTIALIAS_FILTER,
    LEGACY_UPSAMPLE_FILTER,
    LEGACY_UPSAMPLE_RATES,
    assert_decoder_layout_matches,
    decoder_layout,
)

CONFIGS = {
    32000: ROOT / "rvc" / "configs" / "refinegan2" / "32000.json",
}

#: ``upsample_conv_blocks[2]``'s rate with the shipped ``[5, 4, 4, 4]``, and
#: the block whose fold lands in the audible band.
STAGE_RATE = 8000


def _inharmonic_db(signal, f0, rate=STAGE_RATE):
    """Energy that is not within 8 Hz of a harmonic of ``f0``, in dB of total."""

    window = torch.hann_window(signal.numel(), dtype=signal.dtype)
    spectrum = torch.fft.rfft(signal * window).abs().square()
    freqs = torch.fft.rfftfreq(signal.numel(), 1.0 / rate)
    audible = freqs > 1.0
    harmonic = ((freqs - (freqs / f0).round() * f0).abs() < 8.0) & audible
    total = spectrum[audible].sum()
    return 10.0 * torch.log10(spectrum[audible & ~harmonic].sum() / total).item()


def _harmonic_stack(f0, amplitude=1.0, partials=6, seconds=4, rate=STAGE_RATE):
    """Upper partials near Nyquist, which is what the trunk's activations see
    and where the single-tone result stops holding."""

    t = torch.arange(int(rate * seconds), dtype=torch.float64) / rate
    wave = sum(
        torch.sin(2 * torch.pi * f0 * k * t + k) / k for k in range(1, partials + 1)
    )
    wave = wave / wave.square().mean().sqrt()
    return (amplitude * wave).float().view(1, 1, -1)


# --------------------------------------------------------------------------
# why it is oversampling and not a nicer curve
# --------------------------------------------------------------------------


def test_oversampling_beats_the_plain_activation_on_a_realistic_signal():
    """The claim the module exists on, as a measurement."""

    x = _harmonic_stack(440.0)
    plain = _inharmonic_db(F.leaky_relu(x, 0.2).squeeze(), 440.0)
    with torch.no_grad():
        aliased = AntiAliasedActivation()(x).squeeze()
    assert _inharmonic_db(aliased, 440.0) < plain - 12.0


def test_the_round_trip_is_not_also_a_lowpass():
    """The other half of the design, and the one that was wrong until 2026-09-03.

    The round trip's filter cuts at ``rolloff`` of the stage's own Nyquist, so
    this module band-limits as well as anti-aliases.  Gain at each fraction of
    Nyquist, relative to the flat ``(1 + slope) / 2`` a half-wave rectifier
    gives a single tone:

        fraction of Nyquist   0.50   0.70   0.80   0.90   0.95
        w6  r0.90 (old)       0.00  -1.00  -4.35 -11.93 -17.11
        w16 r0.99 (now)       0.00   0.00  +0.04  -0.42  -3.89

    12 dB down at 0.9 of Nyquist is not a transition band, it is a tone
    control, and on the 8 kHz down-loop site it was discarding 3.6-4 kHz of
    the excitation.  A regression here reads as dullness, not as aliasing,
    which is why it needs its own test.
    """

    rate, n = 8000, 1 << 14
    act = AntiAliasedActivation()
    t = torch.arange(n, dtype=torch.float64) / rate
    window = torch.hann_window(n, dtype=torch.float64)

    def gain_db(fraction):
        freq = fraction * rate / 2
        x = torch.sin(2 * torch.pi * freq * t).float().view(1, 1, -1)
        with torch.no_grad():
            y = act(x).view(-1).double()
        bin_ = int(round(freq / rate * n))
        num = torch.fft.rfft(y * window).abs()[bin_]
        den = torch.fft.rfft(x.view(-1).double() * window).abs()[bin_]
        return 20 * torch.log10(num / den).item()

    reference = gain_db(0.5)
    for fraction in (0.7, 0.8, 0.9):
        assert abs(gain_db(fraction) - reference) < 1.5, (
            f"the round trip is {reference - gain_db(fraction):.1f} dB down at "
            f"{fraction:.0%} of Nyquist"
        )


def test_the_activation_is_length_preserving_and_stateless():
    """It sits inside a residual add, so a lost sample is a shape error -- and
    the empty state dict is what makes ``decoder_layout`` necessary."""

    act = AntiAliasedActivation()
    for length in (1, 7, 64, 441, 1024):
        assert act(torch.randn(2, 3, length)).shape == (2, 3, length)
    assert act.state_dict() == {}


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------


def test_half_takes_the_first_activation_of_each_pair():
    """The second activation of a pair reads a signal the first has already
    band-limited.  Full step at 32 kHz, batch 8, eager: 458.7 ms / 3314 MiB for
    ``"none"``, 529.2 / 3906 for ``"half"``, 576.8 / 4503 for ``"full"``."""

    counts = {}
    for mode in ANTIALIAS_MODES:
        block = ResBlock(8, dilation=(1, 3, 5), antialias=mode)
        counts[mode] = len(block.activations)
        assert block(torch.randn(2, 8, 128)).shape == (2, 8, 128)
    # ``"adain"`` wraps the activations around this block, not inside it.
    assert counts == {"none": 0, "adain": 0, "half": 3, "full": 6}


@pytest.mark.parametrize("stages", [None, [], [3], [2, 3]])
def test_only_the_named_stages_are_anti_aliased(stages):
    generator = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        num_mels=192,
        gin_channels=256,
        upsample_initial_channel=512,
        antialias_stages=stages,
    )
    wanted = tuple(sorted(stages or ()))
    assert generator.antialias_stages == wanted
    for index, block in enumerate(generator.upsample_conv_blocks):
        expected = "half" if index in wanted else "none"
        assert block.antialias == expected


def test_an_out_of_range_stage_is_refused():
    with pytest.raises(ValueError, match="antialias_stages"):
        RefineGAN2Generator(upsample_rates=(3, 3, 7, 7), antialias_stages=[4])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="antialias"):
        ResBlock(8, antialias="sometimes")


# --------------------------------------------------------------------------
# what the weights cannot say
# --------------------------------------------------------------------------


def _generator(
    rates, stages=None, rate=32000, antialias_rates=None, upsample_filter=None
):
    return RefineGAN2Generator(
        sample_rate=rate,
        upsample_rates=tuple(rates),
        num_mels=192,
        gin_channels=256,
        upsample_initial_channel=512,
        antialias_stages=stages,
        antialias_rates=antialias_rates,
        **(
            {}
            if upsample_filter is None
            else dict(
                filter_width=upsample_filter[0],
                rolloff=upsample_filter[1],
                filter_beta=upsample_filter[2],
            )
        ),
    )


def test_neither_setting_appears_in_the_state_dict():
    """The premise of the guard.  If this ever fails, ``load_state_dict``
    catches the mismatch on its own and the checkpoint key is redundant."""

    plain = _generator((3, 3, 7, 7))
    reordered = _generator((5, 4, 4, 4), stages=[3])
    a, b = plain.state_dict(), reordered.state_dict()
    assert a.keys() == b.keys()
    assert {k: v.shape for k, v in a.items()} == {k: v.shape for k, v in b.items()}
    result = reordered.load_state_dict(a, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    # ...and it stays wrong, silently: the trunk now runs at 700/4900/14700 Hz
    # instead of 300/900/6300 and carries nine extra activation modules, and a
    # *strict* load raised nothing.
    assert reordered.upsample_rates != plain.upsample_rates
    assert reordered.antialias_stages and not plain.antialias_stages


def test_the_layout_round_trips_through_a_checkpoint():
    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    layout = decoder_layout(model)
    assert layout == {
        "upsample_rates": [5, 4, 4, 4],
        "antialias_stages": [3],
        "antialias": "half",
        "source_gain": False,
        "source_bands": 0,
        "antialias_rates": [],
        "antialias_filter": [2, 16, 0.99, 6.0],
        "antialias_adain": True,
        # Heavy interpolation filter on the last stage only: it is the one
        # whose image reaches the output (path gain -19.9 dB against -49.5
        # and -59.3 for stages 2 and 1) and the one whose input is long
        # enough that a 385-tap kernel costs no edge.
        "upsample_filter": [
            [12, 12, 12, 48],
            [0.9, 0.9, 0.9, 0.99],
            [6.0, 6.0, 6.0, 9.0],
        ],
    }
    assert_decoder_layout_matches(model, {"decoder_layout": layout})


@pytest.mark.parametrize(
    "checkpoint_layout",
    [
        {"upsample_rates": [3, 3, 7, 7], "antialias_stages": [3], "antialias": "half"},
        {"upsample_rates": [5, 4, 4, 4], "antialias_stages": [], "antialias": "none"},
        {"upsample_rates": [5, 4, 4, 4], "antialias_stages": [3], "antialias": "full"},
    ],
)
def test_a_mismatch_is_refused_and_names_the_keys(checkpoint_layout):
    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    with pytest.raises(ValueError, match="Decoder layout mismatch") as excinfo:
        assert_decoder_layout_matches(model, {"decoder_layout": checkpoint_layout})
    message = str(excinfo.value)
    assert "upsample_rates" in message and "refinegan2_antialias" in message


@pytest.mark.parametrize("sample_rate", sorted(LEGACY_UPSAMPLE_RATES))
def test_a_checkpoint_without_the_key_is_read_as_the_old_layout(sample_rate):
    """"Absent" is not "nothing to check": every checkpoint predating the key
    is one of the runs this change re-arranges, which is the case the guard
    exists for.  A run that kept the old layout still resumes untouched."""

    new = torch.nn.Module()
    new.dec = _generator(
        (5, 4, 4, 4) if sample_rate == 32000 else (5, 4, 4, 4),
        stages=[3],
        rate=sample_rate,
    )
    new.sr = sample_rate
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(new, {})

    # The old layout is the old *filter* too: the interpolation design is as
    # invisible to ``load_state_dict`` as the ordering is, so a run that kept
    # one and changed the other is not "the old layout".
    old = torch.nn.Module()
    old.dec = _generator(
        LEGACY_UPSAMPLE_RATES[sample_rate],
        rate=sample_rate,
        upsample_filter=LEGACY_UPSAMPLE_FILTER,
    )
    old.sr = sample_rate
    assert_decoder_layout_matches(old, {})


def test_a_model_with_no_decoder_is_not_checked():
    """The guard must not invent a mismatch for a stack it does not describe."""

    assert decoder_layout(torch.nn.Module()) is None
    assert_decoder_layout_matches(torch.nn.Module(), {"decoder_layout": {}})


def test_the_trainer_guards_both_doors():
    """``load_checkpoint`` covers the resume; the pretrain path loads with a
    bare ``load_state_dict`` and needs the call spelled out.  Both matter."""

    import ast

    trees = [
        ast.parse((ROOT / "rvc" / "train" / name).read_text(encoding="utf-8"))
        for name in ("train.py", "utils.py")
    ]
    called = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "assert_decoder_layout_matches"
    ]
    assert len(called) == 3


# --------------------------------------------------------------------------
# the shipped configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_puts_the_big_stage_first(sample_rate):
    """A stage's anti-image filter keeps ``rolloff`` of the rate it reads, so
    the final residual block synthesises everything above
    ``rolloff * rate[-2] / 2`` from scratch.  Descending order maximises that
    ceiling; it is also what every HiFi-GAN variant does."""

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    rates = model["upsample_rates"]

    import math

    assert math.prod(rates) == sample_rate // 100
    assert rates == sorted(rates, reverse=True)
    assert sorted(rates) == sorted(LEGACY_UPSAMPLE_RATES[sample_rate])
    # The kernel sizes travel with the rates, even though RefineGAN does not
    # read them -- a config whose two halves disagree is a trap for the next
    # decoder that does.
    assert model["upsample_kernel_sizes"] == [2 * r for r in rates]


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_anti_aliases_the_adain_and_nothing_else(sample_rate):
    """Where the artefact was, settled by A/B renders off ``G_8833``.

    Same weights, same reference input, only the anti-aliasing changed; the
    lines read off a spectrogram rather than off a statistic, because every
    statistic tried on them measured something else (band-difference energy
    said imaging dominated; inter-harmonic energy on a moving f0 said nothing
    was aliasing at all -- its baseline is harmonic smear inside the window):

        rates[8k], the two loop activations       lines plainly there
        + the 18 conv activations of resblock[2]  lines plainly there
        + the 6 AdaIN of stage 2                  faint residue
        rates[2k,8k], every site, "full"          clean
        rates[2k,8k], every site, "half"          faint residue
        rates[2k,8k], loops + conv, no AdaIN      lines back
        the 6 AdaIN of stages 1 and 2, alone      clean

    So 20 of the 24 sites a stage could pay for are doing nothing the 6 AdaIN
    are not already doing, which is why ``antialias`` is ``"adain"`` and not
    ``"full"``.  Decoder fwd+bwd, batch 8, 0.4 s:

        nothing                                    97.8 ms   1832 MiB
        stages[1,2] "adain"                       128.1      2001
        the same plus rates[2k,8k]                137.8      1944   <- ships
        stages[1,2] "full" plus rates[2k,8k]      234.3      2813

    The loop rates went out on that last render and came back.  Every render
    above is off ``G_8833``, which was trained with no anti-aliasing at all,
    and a trunk that learns *with* the AdaIN protected from step zero fits a
    different signal path -- so the ladder bounds what helps a decoder that has
    already converged without it, not what a fresh run needs.  The lines
    returned in a real run with the loops raw.  10 ms.
    """

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    from rvc.lib.algorithm.generators.refinegan2 import loop_rates

    down, up = loop_rates(sample_rate, model["upsample_rates"])
    assert model["refinegan2_antialias"] == "adain"
    count = len(model["upsample_rates"])
    assert model["refinegan2_antialias_stages"] == [count - 3, count - 2]
    for rate in model["refinegan2_antialias_rates"]:
        assert rate in down and rate in up
    # The protected rates are the *output* rates of the protected stages, so
    # the loop activations either side of each anti-aliased AdaIN are covered.
    assert sorted(model["refinegan2_antialias_rates"]) == sorted(
        up[stage] * model["upsample_rates"][stage]
        for stage in model["refinegan2_antialias_stages"]
    )
    # the stages named are the ones running at 2 and 8 kHz
    for stage in model["refinegan2_antialias_stages"]:
        assert up[stage] * model["upsample_rates"][stage] in (2000, 8000)


def test_an_antialias_rate_no_activation_runs_at_is_refused():
    """A rate that matches nothing is a typo, and silently protecting no
    activation is exactly the failure this whole option exists to end."""

    with pytest.raises(ValueError, match="match no activation rate"):
        _generator((5, 4, 4, 4), stages=[3], antialias_rates=[7000])


def test_the_protected_rates_reach_both_loops():
    """The down loop is the half ``antialias_stages`` could not address."""

    from rvc.lib.algorithm.resampling import AntiAliasedActivation

    decoder = _generator((5, 4, 4, 4), stages=[3], antialias_rates=[2000, 8000])
    wrapped_down = [
        rate
        for rate, act in zip(decoder.down_rates, decoder.down_activations)
        if isinstance(act, AntiAliasedActivation)
    ]
    wrapped_up = [
        rate
        for rate, act in zip(decoder.up_rates, decoder.up_activations)
        if isinstance(act, AntiAliasedActivation)
    ]
    assert wrapped_down == [8000, 2000]
    assert wrapped_up == [2000, 8000]


def test_a_redesigned_activation_filter_is_refused():
    """Widening the filter or moving the rolloff changes what the decoder
    computes by 19 dB of round-trip error and adds no state-dict key, so it is
    the same class of silent mismatch as a reorder."""

    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), antialias_rates=[2000, 8000])
    model.sr = 32000
    layout = decoder_layout(model)
    assert layout["antialias_filter"] == [2, 16, 0.99, 6.0]

    stale = dict(layout, antialias_filter=LEGACY_ANTIALIAS_FILTER)
    with pytest.raises(ValueError, match="antialias_filter") as excinfo:
        assert_decoder_layout_matches(model, {"decoder_layout": stale})
    assert "constructor default" in str(excinfo.value)

    # and a checkpoint from before the key existed is read as the old design,
    # not as whatever this run happens to build
    keyless = {k: v for k, v in layout.items() if k != "antialias_filter"}
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(model, {"decoder_layout": keyless})


def test_a_decoder_with_no_anti_aliasing_carries_no_filter_design():
    """A run with raw activations has no design to record, and inventing one
    would make every legacy checkpoint mismatch over a field neither uses."""

    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4))
    model.sr = 32000
    layout = decoder_layout(model)
    assert layout["antialias_filter"] is None
    assert_decoder_layout_matches(model, {"decoder_layout": layout})
    # and the pre-key form of the same layout resumes untouched: absent means
    # the legacy design only where something was actually anti-aliased
    keyless = {k: v for k, v in layout.items() if k != "antialias_filter"}
    assert_decoder_layout_matches(model, {"decoder_layout": keyless})


def test_the_protected_rates_add_no_state_dict_key():
    """Same contract as the stages: the resamplers' kernels are non-persistent,
    so a checkpoint cannot tell a protected decoder from a raw one and
    ``decoder_layout`` has to carry it."""

    plain = _generator((5, 4, 4, 4), stages=[3])
    protected = _generator((5, 4, 4, 4), stages=[3], antialias_rates=[2000, 8000])
    assert set(plain.state_dict()) == set(protected.state_dict())


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_an_exported_model_rebuilds_the_same_decoder(sample_rate):
    """If either key stopped travelling, inference would rebuild a decoder
    without the anti-aliasing and render differently from the run that trained
    it, with nothing raised.  ``upsample_rates`` travels separately, in the
    positional ``config`` list."""

    from rvc.lib.algorithm.synthesizers import (
        Synthesizer,
        vocoder_config_from_model,
    )

    config = json.loads(CONFIGS[sample_rate].read_text())
    data, model, train = config["data"], config["model"], config["train"]
    frames = train["segment_size"] // data["hop_length"]
    spec_channels = data["filter_length"] // 2 + 1

    trained = Synthesizer(
        spec_channels=spec_channels,
        segment_size=frames,
        sr=sample_rate,
        use_f0=True,
        vocoder="RefineGAN",
        **{k: v for k, v in model.items() if k != "architecture_id"},
    )
    exported = vocoder_config_from_model(dict(model))
    assert "refinegan2_antialias_stages" in exported
    assert "refinegan2_antialias" in exported

    rebuilt = Synthesizer(
        spec_channels=spec_channels,
        segment_size=frames,
        inter_channels=model["inter_channels"],
        hidden_channels=model["hidden_channels"],
        filter_channels=model["filter_channels"],
        n_heads=model["n_heads"],
        n_layers=model["n_layers"],
        kernel_size=model["kernel_size"],
        p_dropout=model["p_dropout"],
        resblock=model["resblock"],
        resblock_kernel_sizes=model["resblock_kernel_sizes"],
        resblock_dilation_sizes=model["resblock_dilation_sizes"],
        upsample_rates=model["upsample_rates"],
        upsample_initial_channel=model["upsample_initial_channel"],
        upsample_kernel_sizes=model["upsample_kernel_sizes"],
        spk_embed_dim=model["spk_embed_dim"],
        gin_channels=model["gin_channels"],
        sr=sample_rate,
        use_f0=True,
        vocoder="RefineGAN",
        vocoder_config=exported,
    )
    assert decoder_layout(rebuilt) == decoder_layout(trained)


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_pays_for_it_out_of_the_discriminator(sample_rate):
    """The corrections are budget-neutral only because the discriminator got
    cheaper in the same breath: UnivHD off and ``v4``'s four periods."""

    from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    assert model["d_version"] == "v4"
    # UnivHD is *on*.  The probe that settled it, 300 steps / 3 seeds on the
    # defects each side should see: swapping v3's longest period for UnivHD
    # takes the ">9 kHz shelf loss" detection from 67.9% +- 10.6 to 100% +- 0
    # and f0 jitter from 88.7% +- 8.0 to 100% +- 0, while on a 330 ms dynamics
    # defect -- the one the dropped period's 344 ms window exists for -- the
    # two tie at 73.3% against 72.9%.  The branch it replaces does not defend
    # itself even on its own ground.
    assert model["d_use_univhd"] is True
    # The set itself is derived from the version and the rate rather than
    # written out; what matters here is that it is one branch lighter than v3.
    built = MPD_MSD_Combined(False, version="v4", sample_rate=sample_rate)
    full = MPD_MSD_Combined(False, version="v3", sample_rate=sample_rate)
    assert len(built.periods) == len(full.periods) - 1 == 4


# --------------------------------------------------------------------------
# the source gain
# --------------------------------------------------------------------------


def test_the_gain_is_the_identity_at_initialisation():
    """Zero weights and a bias of ``softplus^-1(1)``: a run that switches this
    on starts from exactly the excitation it had, and a fine-tune cannot have
    its source silently rescaled on step zero."""

    generator = _generator((5, 4, 4, 4), stages=[3])
    assert not generator.has_source_gain
    with_gain = RefineGAN2Generator(
        sample_rate=32000, upsample_rates=(5, 4, 4, 4), num_mels=192,
        gin_channels=256, upsample_initial_channel=512, source_gain=True)
    z = torch.randn(2, 192, 20)
    gain = F.softplus(with_gain.source_gain(z))
    assert torch.allclose(gain, torch.ones_like(gain), atol=1e-6)


def test_the_gain_reaches_the_excitation_before_the_skips():
    """It has to multiply ``har_source`` ahead of ``pre_conv``, or the decimated
    copies handed to every stage carry the unscaled excitation."""

    import inspect
    from rvc.lib.algorithm.generators import refinegan2 as module

    source = inspect.getsource(module.RefineGAN2Generator.forward)
    gain_at = source.index("_apply_source_gain")
    pre_at = source.index("self.pre_conv(")
    down_at = source.index("downs = []")
    assert gain_at < pre_at < down_at


def test_the_gain_upsample_is_filtered_not_interpolated():
    """The gain multiplies the excitation, so a residual image in it lands as a
    sideband on every harmonic.  A smooth envelope makes ``F.interpolate`` good
    enough; this gain is learned, and on a frame-rate-white one the two differ
    by ~32 dB."""

    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

    frames, hop = 400, 320
    length = frames * hop
    rough = torch.rand(1, 1, frames) * 0.7 + 0.3

    linear = F.interpolate(rough, size=length, mode="linear", align_corners=False)
    filtered = rough
    for factor in (5, 4, 4, 4):
        filtered = AntiAliasedUpsample1d(
            factor, filter_width=12, rolloff=0.90, filter_beta=6.0)(filtered)
    filtered = filtered[..., :length]

    def energy_above(signal, cutoff_hz, sr=32000):
        spectrum = torch.fft.rfft(signal.squeeze() * torch.hann_window(length)).abs()
        freqs = torch.fft.rfftfreq(length, 1 / sr)
        return spectrum[freqs > cutoff_hz].square().sum().item()

    # everything above 60 Hz in the upsampled gain is image, since the frame
    # grid cannot carry more than 50 Hz of envelope.
    assert energy_above(filtered, 60.0) < energy_above(linear, 60.0) / 100


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_turns_the_excitation_gain_on(sample_rate):
    """A large win for this source: held-out multi-scale mel over three seeds,
    on a fixed trunk, 1.9714 -> 1.7418 with the sine.  It was *negative* on the
    harmonic bank (1.7357 -> 1.8057), which already carried what the gain buys
    -- they were alternatives, not a stack -- and the bank is gone, so the
    pairing question goes with it.  193 parameters.
    """

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    assert model["refinegan2_source_gain"] is True


def test_the_upsampler_does_not_delay_what_it_interpolates():
    """``AntiAliasedUpsample1d`` has to put input sample ``n`` on output
    ``n * factor``, because the trunk it feeds is concatenated with the
    ``downs[]`` skips, which never went through it.

    ``pad_left`` cropped ``(kernel_size - factor) // 2`` until 2026-09-03 where
    the kernel's group delay is ``(kernel_size - 1) // 2``, so every instance
    ran ``factor // 2`` output samples late -- +1 at x2/x3, +2 at x4/x5, +3 at
    x7, +4 at x8.  Summed over ``[5, 4, 4, 4]`` at 32 kHz that is 5.31 ms of
    trunk-against-skip misalignment, and it is invisible in the state dict: the
    kernels are non-persistent buffers, so nothing about a checkpoint says
    which convention built it.

    An impulse is the whole test.  A symmetric kernel that is correctly
    centred puts its peak on the input's position scaled by the factor, and
    a delay of even one sample moves it.
    """

    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

    position = 40
    for factor in (2, 3, 4, 5, 7, 8):
        for width in (4, 6, 12):
            module = AntiAliasedUpsample1d(
                factor, filter_width=width, rolloff=0.90, filter_beta=6.0
            )
            impulse = torch.zeros(1, 1, 80)
            impulse[0, 0, position] = 1.0
            out = module(impulse)

            assert out.shape[-1] == 80 * factor, (
                f"x{factor} width {width} returned {out.shape[-1]} samples, "
                f"not {80 * factor}; the block that owns this adds its output "
                f"to a residual and needs the length exactly."
            )
            peak = int(out[0, 0].argmax())
            assert peak == position * factor, (
                f"x{factor} width {width} put the impulse at {peak}, "
                f"{peak - position * factor:+d} from {position * factor}."
            )


# --------------------------------------------------------------------------
# imaging: the defect no antialias option touches
# --------------------------------------------------------------------------


def test_the_last_stage_upsampler_rejects_its_image():
    """Imaging is not aliasing, and it was the larger of the two here.

    Zero-stuffing by ``factor`` copies the input spectrum to every multiple of
    the input rate; what the interpolation filter leaves is an image at
    ``|k*R_in +- f|``.  No ``antialias_*`` option touches it -- and an image at
    ``k*R_in - j*f0`` moves against f0 exactly like a fold, so a spectrogram
    cannot tell them apart.

    The flat ``w12 r0.90`` this decoder shipped with left the worst image at
    -37.1 dB, which after the -19.9 dB path from stage 3 to the output puts it
    at -57 dB there -- above every fold measured anywhere in this decoder.
    The partial that makes it was also being attenuated 13.0 dB, the same
    rolloff trap as ``AntiAliasedActivation``.
    """

    import math

    from rvc.lib.algorithm.generators.refinegan2 import (
        DEFAULT_UPSAMPLE_BETA,
        DEFAULT_UPSAMPLE_ROLLOFF,
        DEFAULT_UPSAMPLE_WIDTH,
    )
    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

    rate_in, factor, n = 8000, 4, 1 << 13
    frac = 0.95  # just under the input Nyquist, where the image is worst
    freq = frac * rate_in / 2
    t = torch.arange(n, dtype=torch.float64) / rate_in
    x = torch.sin(2 * torch.pi * freq * t).view(1, 1, -1).float()

    def worst_image(width, rolloff, beta):
        up = AntiAliasedUpsample1d(factor, width, rolloff, beta)
        with torch.no_grad():
            y = up(x).view(-1).double()
        m = y.numel()
        spectrum = torch.fft.rfft(y * torch.hann_window(m, dtype=torch.float64)).abs()
        freqs = torch.fft.rfftfreq(m, 1 / (rate_in * factor))

        def level(target):
            j = int(torch.argmin((freqs - target).abs()))
            return spectrum[max(0, j - 3) : j + 4].max().item()

        images = [
            level(k * rate_in + sign * freq)
            for k in range(1, factor)
            for sign in (-1, 1)
            if 20 < k * rate_in + sign * freq < rate_in * factor / 2 - 20
            and abs(k * rate_in + sign * freq - freq) > 30
        ]
        return 20 * math.log10(max(images) / level(freq) + 1e-30)

    assert worst_image(12, 0.90, 6.0) > -45.0, "the flat design was not this good"
    assert (
        worst_image(
            DEFAULT_UPSAMPLE_WIDTH[-1],
            DEFAULT_UPSAMPLE_ROLLOFF[-1],
            DEFAULT_UPSAMPLE_BETA[-1],
        )
        < -85.0
    )


def test_the_upsample_schedule_is_heavy_only_where_the_input_is_long():
    """The two measurements that picked this schedule, as a shape constraint.

    A long kernel on a short input is worse than a short one: the upsampler
    replicate-pads, so the extra taps read an invented continuation.  Stage 3
    is both the stage whose image reaches the output and the stage with the
    most input samples, which is why it is the only one lengthened.
    """

    from rvc.lib.algorithm.generators.refinegan2 import (
        DEFAULT_UPSAMPLE_BETA,
        DEFAULT_UPSAMPLE_ROLLOFF,
        DEFAULT_UPSAMPLE_WIDTH,
    )

    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_ROLLOFF)
    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_BETA)
    # every stage but the last keeps the design it always had
    assert set(DEFAULT_UPSAMPLE_WIDTH[:-1]) == {12}
    assert set(DEFAULT_UPSAMPLE_ROLLOFF[:-1]) == {0.90}
    assert DEFAULT_UPSAMPLE_WIDTH[-1] > 2 * DEFAULT_UPSAMPLE_WIDTH[0]
    assert DEFAULT_UPSAMPLE_ROLLOFF[-1] > DEFAULT_UPSAMPLE_ROLLOFF[0]


def test_a_wrong_length_filter_schedule_is_refused():
    """``filter_schedule`` is the only thing standing between a mistyped list
    and a decoder that silently filters the wrong stage."""

    with pytest.raises(ValueError, match="filter_width"):
        RefineGAN2Generator(
            sample_rate=32000, upsample_rates=(5, 4, 4, 4), num_mels=192,
            gin_channels=256, upsample_initial_channel=512,
            filter_width=(12, 48),
        )
    with pytest.raises(ValueError, match="rolloff"):
        RefineGAN2Generator(
            sample_rate=32000, upsample_rates=(5, 4, 4, 4), num_mels=192,
            gin_channels=256, upsample_initial_channel=512,
            rolloff=1.4,
        )


def test_antialias_reaches_the_adain_activations():
    """The six per stage that no option could reach until 2026-09-03.

    ``ParallelResBlock`` is ``Sequential(AdaIN, ResBlock, AdaIN)`` and
    ``antialias`` went to the ``ResBlock`` only, so a stage with the option on
    still ran six raw nonlinearities at its own rate.  A/B renders off
    ``G_8833`` put those six, not the eighteen next to them, at the centre of
    the artefact: with them the lines go, without them they come back at the
    same rate coverage.
    """

    from rvc.lib.algorithm.generators.refinegan2 import AdaIN

    for mode, wanted in (("none", False), ("half", True), ("full", True)):
        block = ParallelResBlock(
            in_channels=16, out_channels=8, kernel_sizes=(3, 7),
            dilation=(1, 3), antialias=mode,
        )
        adains = [m for m in block.modules() if isinstance(m, AdaIN)]
        assert len(adains) == 4  # two per kernel size
        assert all(m.antialias is wanted for m in adains)
        assert all(
            isinstance(m.activation, AntiAliasedActivation) is wanted for m in adains
        )
        assert block(torch.randn(2, 16, 96)).shape == (2, 8, 96)


def test_the_adain_coverage_is_in_the_layout():
    """It adds no state-dict key, exactly like the rest of this file's subject:
    a checkpoint from before the flag existed loads silently into a decoder
    that now anti-aliases six more sites per stage."""

    covered = torch.nn.Module()
    covered.dec = _generator((5, 4, 4, 4), stages=[1, 2])
    covered.sr = 32000
    raw = torch.nn.Module()
    raw.dec = _generator((5, 4, 4, 4), stages=[])
    raw.sr = 32000

    assert set(covered.dec.state_dict()) == set(raw.dec.state_dict())
    layout = decoder_layout(covered)
    assert layout["antialias_adain"] is True
    assert decoder_layout(raw)["antialias_adain"] is False

    stale = dict(layout, antialias_adain=False)
    with pytest.raises(ValueError, match="antialias_adain"):
        assert_decoder_layout_matches(covered, {"decoder_layout": stale})
    keyless = {k: v for k, v in layout.items() if k != "antialias_adain"}
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(covered, {"decoder_layout": keyless})


def test_the_polyphase_upsample_matches_the_transposed_form():
    """The optimisation, as an equivalence.

    ``AntiAliasedUpsample1d`` computed a grouped ``conv_transpose1d``, which is
    cuDNN's slow path: at the 8 kHz res-block shape it was 2.79 ms of a 3.12 ms
    module, against 0.35 ms for the strided ``conv1d`` on the way back down
    with the same tap count.  A stride-``F`` transposed convolution *is* ``F``
    convolutions of ``K/F`` taps interleaved, so the polyphase form does the
    same multiplies through the fast path -- 5-14x faster, and the whole module
    4.2x at that shape.

    Pinned as an equivalence rather than a timing, because the failure mode is
    silent: the phases have unequal length whenever ``K`` is not a multiple of
    ``F``, and left-aligning a short one instead of right-aligning it shifts
    that phase alone by a sample.  That barely moves a spectrum.
    """

    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

    def transposed(up, x):
        kernel = up.kernel.to(x).expand(x.shape[1], -1, -1).contiguous()
        padded = F.pad(x, (up.pad, up.pad + 1), mode="replicate")
        out = F.conv_transpose1d(
            padded, kernel, stride=up.factor, padding=up.pad_left,
            groups=x.shape[1],
        )
        return out[..., : x.shape[-1] * up.factor]

    torch.manual_seed(0)
    for factor in (2, 3, 4, 5):
        for width in (6, 12, 16, 48):
            up = AntiAliasedUpsample1d(factor, width, 0.99, 6.0)
            x = torch.randn(2, 5, 137)
            with torch.no_grad():
                got, want = up(x), transposed(up, x)
            assert got.shape == (2, 5, 137 * factor)
            scale = want.abs().max()
            assert (got - want).abs().max() < 1e-5 * scale, (
                f"factor {factor} width {width}: "
                f"{((got - want).abs().max() / scale).item():.2e} relative"
            )


def test_the_round_trip_activation_is_in_place_on_its_own_tensor():
    """The 2x tensor is the module's largest intermediate, and the upsampler's
    output is not referenced anywhere else, so the activation can overwrite it.
    Guarded because an in-place op on a tensor someone else holds is a silent
    wrong-gradient bug, not a crash."""

    act = AntiAliasedActivation()
    x = torch.randn(2, 4, 128, requires_grad=True)
    y = act(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # a non-LeakyReLU activation must still take the ordinary path
    other = AntiAliasedActivation(torch.nn.SiLU())
    z = torch.randn(2, 4, 128, requires_grad=True)
    other(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_adain_mode_covers_only_the_wrappers():
    """``"adain"`` has to be a mode, not a second boolean.

    The two are one decision -- which activations at this stage's rate get
    oversampled -- and splitting it into ``antialias`` plus an
    ``antialias_adain`` flag makes ``"full" + adain=False`` expressible, which
    is precisely the configuration that shipped for months and left the
    dominant site raw.  Ordering them by coverage makes that unsayable.
    """

    from rvc.lib.algorithm.generators.refinegan2 import AdaIN, ParallelResBlock

    wanted = {"none": (0, 0), "adain": (6, 0), "half": (6, 9), "full": (6, 18)}
    assert set(wanted) == set(ANTIALIAS_MODES)
    for mode, (adain_sites, pair_sites) in wanted.items():
        block = ParallelResBlock(
            in_channels=16, out_channels=8, kernel_sizes=(3, 7, 11),
            dilation=(1, 3, 5), antialias=mode,
        )
        wrapped = [
            m for m in block.modules() if isinstance(m, AntiAliasedActivation)
        ]
        in_adain = sum(
            isinstance(m.activation, AntiAliasedActivation)
            for m in block.modules()
            if isinstance(m, AdaIN)
        )
        assert in_adain == adain_sites, mode
        assert len(wrapped) - in_adain == pair_sites, mode
        assert block(torch.randn(2, 16, 96)).shape == (2, 8, 96)

    # and the mode list is ordered by coverage, which is what makes a
    # half-covered stage impossible to ask for
    counts = [sum(wanted[m]) for m in ANTIALIAS_MODES]
    assert counts == sorted(counts)
