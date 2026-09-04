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
    rates,
    stages=None,
    rate=32000,
    antialias_rates=None,
    upsample_filter=None,
    linear_down_path=False,
    up_activation_after_upsample=False,
):
    return RefineGAN2Generator(
        sample_rate=rate,
        upsample_rates=tuple(rates),
        num_mels=192,
        gin_channels=256,
        upsample_initial_channel=512,
        antialias_stages=stages,
        antialias_rates=antialias_rates,
        linear_down_path=linear_down_path,
        up_activation_after_upsample=up_activation_after_upsample,
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
        # ``None`` rather than ``"post"``: this decoder has no gain, so it has
        # no softplus to place, and a run without one must not mismatch a run
        # without one over where a module neither of them has would go.
        "source_gain_order": None,
        "source_bands": 0,
        "antialias_rates": [],
        "antialias_filter": [2, 16, 0.99, 6.0],
        # The oversampling factor per site group.  ``antialias_filter`` reports
        # only the first instance in module order, so with the two groups at
        # different factors it names one of them and these two name both.
        "antialias_factor": 2,
        "antialias_rate_factor": 2,
        # Per stage: the taps are free where the round trip is launch-bound and
        # about 45% of it at the output-rate shapes.
        "antialias_width": [16, 16, 16, 16],
        "antialias_rate_width": 16,
        "antialias_adain": True,
        # No rate is protected here, so the activation before ``conv_post`` is
        # raw too: it follows ``sample_rate in antialias_rates``.
        "antialias_output": False,
        # Same contract as ``antialias_output``: implied by the rates, but only
        # for a checkpoint written after the wrapper existed.
        "antialias_tanh": False,
        # Structural, and off: every activation the down loop ever had, each
        # up activation still ahead of its upsampler.
        "linear_down_path": False,
        "up_activation_after_upsample": False,
        # Interpolation filter that grows with the stage's input length:
        # stage 0 (40 samples in a training segment) keeps the short kernel,
        # stages 1-3 get a deeper stopband, since their -37 dB image is not
        # aliasing and no ``antialias_*`` option removes it downstream.
        "upsample_filter": [
            [12, 24, 32, 48],
            [0.9, 0.95, 0.97, 0.99],
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
        the same plus rates[2k,8k]                137.8      1944
        stages[1,2] "full" plus rates[2k,8k]      234.3      2813

    What ships is wider than the last row of that ladder on the cheap axis and
    narrower on the expensive one -- every loop rate and the output activation,
    every stage's AdaIN but the last:

        stages[1,2] + rates[2k,8k]                145.7 ms   1848 MiB
        + every loop rate + conv_post             156.1      1956
        + stage 0                                 162.9      2059   <- ships
        + stage 3                                 215.5      2489
        the same with "half"                      346.0      3631

    The loop rates went out on that last render and came back.  Every render
    above is off ``G_8833``, which was trained with no anti-aliasing at all,
    and a trunk that learns *with* the AdaIN protected from step zero fits a
    different signal path -- so the ladder bounds what helps a decoder that has
    already converged without it, not what a fresh run needs.  The lines
    returned in a real run with the loops raw.  10 ms.
    """

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    from rvc.lib.algorithm.generators.refinegan2 import loop_rates

    down, up = loop_rates(
        sample_rate,
        model["upsample_rates"],
        linear_down_path=model["refinegan2_linear_down_path"],
        up_activation_after_upsample=model[
            "refinegan2_up_activation_after_upsample"
        ],
    )
    assert model["refinegan2_antialias"] == "adain"
    # The ladder above was measured with twelve loop sites, all of them
    # protected.  The two structural flags deleted or moved every site that
    # folded into the audible band, so what is left to protect is the output
    # rate: ``downs[0]``, which rectifies a full-band sine and has no headroom
    # at all, the activation before ``conv_post``, and the ``tanh``.  Naming a
    # rate that no longer exists is refused at construction, which is what
    # keeps this list honest rather than decorative.
    assert model["refinegan2_antialias_rates"] == [sample_rate]
    assert sample_rate in set(down) | set(up)
    # ...and every rate the loops still run at is either the protected one or
    # one the layout already gave headroom to.
    assert set(down) == {sample_rate}
    # Every stage but the last.  Stage ``count - 1`` runs at the output rate,
    # where its six AdaIN are 4x the tensor of the 8 kHz stage's: 145.7 ms for
    # the shipped config against 215.5 with it, batch 8 over 0.4 s, where the
    # whole rest of the coverage is 17 ms.  It is the one place cost and
    # coverage disagree by enough to leave a site raw on purpose.
    count = len(model["upsample_rates"])
    assert model["refinegan2_antialias_stages"] == list(range(count - 1))
    # The stages named are the ones running at 500, 2000 and 8000 Hz.  A
    # stage's residual blocks run at its *output* rate, which is what ``up``
    # now reports directly -- the up activation moved to the same side of the
    # upsampler as they are on.
    assert [
        up[stage] for stage in model["refinegan2_antialias_stages"]
    ] == [500, 2000, 8000]


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
    # The export is derived from ``Synthesizer``'s signature, so these travel
    # for free -- but "for free" is exactly the kind of thing that stops being
    # true silently, and a factor that does not travel renders the exported
    # model at 2 whatever it trained at.
    assert "refinegan2_antialias_factor" in exported
    assert "refinegan2_antialias_rate_factor" in exported

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
    has the most input samples and takes the longest kernel; stage 0 has the
    fewest and keeps the shortest.
    """

    from rvc.lib.algorithm.generators.refinegan2 import (
        DEFAULT_UPSAMPLE_BETA,
        DEFAULT_UPSAMPLE_ROLLOFF,
        DEFAULT_UPSAMPLE_WIDTH,
    )

    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_ROLLOFF)
    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_BETA)
    # stage 0 has 40 input samples in a training segment and keeps the
    # short kernel; every later stage has more input and a longer filter,
    # so the schedule is monotone in width and in rolloff
    assert DEFAULT_UPSAMPLE_WIDTH[0] == 12
    assert DEFAULT_UPSAMPLE_ROLLOFF[0] == 0.90
    assert list(DEFAULT_UPSAMPLE_WIDTH) == sorted(DEFAULT_UPSAMPLE_WIDTH)
    assert list(DEFAULT_UPSAMPLE_ROLLOFF) == sorted(DEFAULT_UPSAMPLE_ROLLOFF)
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


def test_the_output_rate_reaches_the_activation_before_conv_post():
    """The site with the shortest path to the render, and the one nothing
    could address: it is in neither loop's ``ModuleList`` and in no residual
    block, so ``antialias_stages`` and ``antialias_rates`` both missed it."""

    raw = _generator((5, 4, 4, 4), stages=[1, 2], antialias_rates=[2000, 8000])
    covered = _generator(
        (5, 4, 4, 4), stages=[1, 2], antialias_rates=[2000, 8000, 32000]
    )
    assert isinstance(raw.out_activation, torch.nn.Identity)
    assert isinstance(covered.out_activation, AntiAliasedActivation)
    # ...and the same rate covers ``downs[0]``, which is the other site at the
    # output rate and is concatenated straight into the last stage.
    assert isinstance(covered.down_activations[0], AntiAliasedActivation)

    # same contract as everything else in this file: no state-dict key
    assert set(raw.state_dict()) == set(covered.state_dict())

    model = torch.nn.Module()
    model.dec = covered
    model.sr = 32000
    layout = decoder_layout(model)
    assert layout["antialias_output"] is True
    assert decoder_layout_of(raw)["antialias_output"] is False

    # a checkpoint from before the activation existed names the same protected
    # rates and had a raw site there, so absent has to read as False
    keyless = {k: v for k, v in layout.items() if k != "antialias_output"}
    with pytest.raises(ValueError, match="antialias_output"):
        assert_decoder_layout_matches(model, {"decoder_layout": keyless})


def decoder_layout_of(decoder):
    model = torch.nn.Module()
    model.dec = decoder
    model.sr = 32000
    return decoder_layout(model)


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


def _factors(decoder):
    """Every anti-aliased activation's factor, split by site group."""

    loops = [
        m.design[0]
        for group in (decoder.down_activations, decoder.up_activations)
        for m in group
        if isinstance(m, AntiAliasedActivation)
    ]
    if isinstance(decoder.out_activation, AntiAliasedActivation):
        loops.append(decoder.out_activation.design[0])
    inside = set()
    for block in decoder.upsample_conv_blocks:
        inside.update(
            m.design[0]
            for m in block.modules()
            if isinstance(m, AntiAliasedActivation)
        )
    return sorted(set(loops)), sorted(inside)


def test_the_two_site_groups_take_their_own_factor():
    """The factor is what sets the alias floor, and the groups differ in price.

    The nine loop and output sites are a handful against the stages' dozens,
    so a factor that is unaffordable inside the res-blocks is affordable there
    -- which is the whole reason these are two knobs rather than one.
    """

    decoder = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        upsample_initial_channel=512,
        num_mels=192,
        start_channels=16,
        gin_channels=256,
        antialias_stages=(0, 1, 2),
        antialias="adain",
        antialias_rates=(100, 500, 2000, 8000, 32000),
        antialias_factor=2,
        antialias_rate_factor=3,
    )
    loops, inside = _factors(decoder)
    assert loops == [3]
    assert inside == [2]
    assert len(
        [
            m
            for m in list(decoder.down_activations)
            + list(decoder.up_activations)
            + [decoder.out_activation]
            if isinstance(m, AntiAliasedActivation)
        ]
    ) == 9


def test_the_rate_factor_follows_the_stage_one_when_unset():
    """A config that names only one of them cannot get the two disagreeing."""

    decoder = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        upsample_initial_channel=512,
        num_mels=192,
        start_channels=16,
        gin_channels=256,
        antialias_stages=(0,),
        antialias="adain",
        antialias_rates=(8000,),
        antialias_factor=4,
    )
    assert decoder.antialias_rate_factor == 4
    loops, inside = _factors(decoder)
    assert loops == [4] and inside == [4]


@pytest.mark.parametrize("field", ["antialias_factor", "antialias_rate_factor"])
def test_a_factor_below_two_is_refused(field):
    """At 1 the round trip oversamples nothing and is a bare lowpass.

    ``AntiAliasedUpsample1d`` returns its input untouched at factor 1 while
    ``FixedLowPass1d`` still filters, so the module would quietly become a tone
    control -- the same trap ``rolloff`` was at 0.90.
    """

    with pytest.raises(ValueError, match="at least 2"):
        RefineGAN2Generator(
            sample_rate=32000,
            upsample_rates=(5, 4, 4, 4),
            upsample_initial_channel=512,
            num_mels=192,
            start_channels=16,
            gin_channels=256,
            antialias_stages=(0,),
            antialias="adain",
            **{field: 1},
        )


def test_a_checkpoint_from_before_the_factor_key_reads_as_two():
    """Absent means 2: it was the constructor default and nothing reached it.

    A checkpoint written before these fields existed has to keep loading, and
    it can only have been trained at 2.
    """

    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    layout = decoder_layout(model)
    without = {k: v for k, v in layout.items() if not k.endswith("_factor")}
    assert_decoder_layout_matches(model, {"decoder_layout": without})


def test_a_changed_factor_is_refused():
    """It leaves no trace in the weights, like everything else in the layout."""

    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    stale = dict(decoder_layout(model))
    stale["antialias_rate_factor"] = 3
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(model, {"decoder_layout": stale})


def _widths(decoder):
    """Every anti-aliased activation's filter width, by stage and by loop."""

    loops = {
        m.design[1]
        for group in (decoder.down_activations, decoder.up_activations)
        for m in group
        if isinstance(m, AntiAliasedActivation)
    }
    if isinstance(decoder.out_activation, AntiAliasedActivation):
        loops.add(decoder.out_activation.design[1])
    stages = [
        sorted({
            m.design[1]
            for m in block.modules()
            if isinstance(m, AntiAliasedActivation)
        })
        for block in decoder.upsample_conv_blocks
    ]
    return sorted(loops), stages


def test_the_width_can_differ_per_stage():
    """The taps are free where the round trip is launch-bound.

    At the small shapes the AA pair is dominated by launch overhead and a wider
    filter costs nothing; at the output-rate shapes the taps are about 45% of
    it.  So the schedule exists to keep the better filter exactly where it is
    free -- which a single scalar cannot express.
    """

    decoder = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        upsample_initial_channel=512,
        num_mels=192,
        start_channels=16,
        gin_channels=256,
        antialias_stages=(0, 1, 2),
        antialias="full",
        antialias_rates=(8000,),
        antialias_width=[16, 16, 8, 8],
        antialias_rate_width=8,
    )
    loops, stages = _widths(decoder)
    assert loops == [8]
    assert stages[0] == [16] and stages[1] == [16] and stages[2] == [8]
    # Stage 3 is not in ``antialias_stages``, so it has no wrapped activation
    # to carry a width at all.
    assert stages[3] == []


def test_a_scalar_width_covers_every_stage():
    decoder = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        upsample_initial_channel=512,
        num_mels=192,
        start_channels=16,
        gin_channels=256,
        antialias_stages=(0, 1, 2),
        antialias="adain",
        antialias_rates=(8000,),
        antialias_width=8,
    )
    assert decoder.antialias_width == (8, 8, 8, 8)
    # Unset, the loops take the widest stage value rather than a constant, so
    # a config that narrows everything does not leave them behind.
    assert decoder.antialias_rate_width == 8


def test_a_width_schedule_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="antialias_width"):
        RefineGAN2Generator(
            sample_rate=32000,
            upsample_rates=(5, 4, 4, 4),
            upsample_initial_channel=512,
            num_mels=192,
            start_channels=16,
            gin_channels=256,
            antialias_stages=(0,),
            antialias="adain",
            antialias_width=[16, 8],
        )


def test_a_checkpoint_from_before_the_width_key_reads_as_sixteen():
    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    layout = decoder_layout(model)
    without = {
        k: v for k, v in layout.items() if k not in
        ("antialias_width", "antialias_rate_width")
    }
    assert_decoder_layout_matches(model, {"decoder_layout": without})


def test_a_changed_width_is_refused():
    """Non-persistent kernels again: the weights cannot tell 8 from 16."""

    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4), stages=[3])
    model.sr = 32000
    stale = dict(decoder_layout(model))
    stale["antialias_width"] = [16, 16, 8, 8]
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(model, {"decoder_layout": stale})


def test_the_output_tanh_follows_the_output_rate():
    """It is a pointwise nonlinearity with nothing after it.

    Not the decoder's problem -- against a 32x reference of itself it adds
    -61.4 dB at ``|max| = 0.9``, well under the activation floor, because a
    smooth curve's harmonic series decays fast where ``leaky_relu``'s corner
    makes products of every order.  Wrapped anyway because ``conv_post`` emits
    one channel, so it costs almost nothing.
    """

    raw = _generator((5, 4, 4, 4), stages=[0], antialias_rates=[])
    protected = _generator((5, 4, 4, 4), stages=[0], antialias_rates=[32000])
    assert isinstance(raw.out_tanh, torch.nn.Tanh)
    assert isinstance(protected.out_tanh, AntiAliasedActivation)
    # It carries the loop group's design, like the other output-rate site.
    assert protected.out_tanh.design[0] == protected.antialias_rate_factor
    # And it adds no state-dict key, which is why the layout has to say so.
    assert set(raw.state_dict()) == set(protected.state_dict())
    assert decoder_layout(_wrap(raw))["antialias_tanh"] is False
    assert decoder_layout(_wrap(protected))["antialias_tanh"] is True


def _wrap(decoder):
    model = torch.nn.Module()
    model.dec = decoder
    model.sr = 32000
    return model


def test_a_checkpoint_from_before_the_tanh_wrapper_reads_as_raw():
    model = _wrap(_generator((5, 4, 4, 4), stages=[3], antialias_rates=[32000]))
    layout = decoder_layout(model)
    assert layout["antialias_tanh"] is True
    without = {k: v for k, v in layout.items() if k != "antialias_tanh"}
    # A checkpoint that protected the output rate but predates the wrapper had
    # a raw tanh, so it must not silently match one that has it.
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(model, {"decoder_layout": without})


# --------------------------------------------------------------------------
# not making the fold, rather than filtering it
# --------------------------------------------------------------------------
#
# Anti-aliasing is the generic answer for a nonlinearity that has to exist, and
# it is never free: the round trip is an upsample, an activation and a
# downsample where there was one activation.  At the two 8 kHz sites that make
# the 4 kHz fold, whether the nonlinearity has to exist at all is a question
# with an answer.
#
#     down loop, past the first stage   it does not have to exist    -> deleted
#     up loop, before the upsampler     it does, but not at 8 kHz    -> moved
#
# Both change what the decoder computes at identical weights, so both are
# ``decoder_layout`` fields and neither is a patch on a checkpoint.


def _down_skips(decoder):
    """``(pre-activation output, skip the trunk receives)`` per down stage.

    Read off the running decoder rather than recomputed: ``downs[]`` is a local
    in ``forward``, but every entry after the first is
    ``downsample_blocks[i - 1]``'s output with or without an activation on it,
    and it arrives at the trunk as the *last* channels of
    ``upsample_conv_blocks[count - 1 - i]``'s input.  So a hook at each end
    brackets exactly the one operation these flags remove.
    """

    count = len(decoder.upsample_rates)
    outputs, skips, handles = {}, {}, []

    for index, block in enumerate(decoder.downsample_blocks):
        handles.append(
            block.register_forward_hook(
                lambda _m, _i, out, index=index: outputs.__setitem__(index, out)
            )
        )
    for index, block in enumerate(decoder.upsample_conv_blocks):
        handles.append(
            block.register_forward_pre_hook(
                lambda _m, inputs, index=index: skips.__setitem__(index, inputs[0])
            )
        )

    with torch.no_grad():
        decoder.eval()(*_inputs())

    for handle in handles:
        handle.remove()

    pairs = []
    for source in range(count - 1):
        y = outputs[source]
        # ``downs[source + 1]`` goes to the up stage that runs at its rate
        skip = skips[count - 1 - (source + 1)][:, -y.shape[1] :]
        pairs.append((y, skip))
    return pairs


def _inputs(frames=40, batch=1, seed=0):
    torch.manual_seed(seed)
    mel = torch.randn(batch, 192, frames) * 0.5
    f0 = torch.full((batch, frames), 220.0)
    g = torch.randn(batch, 256, 1) * 0.1
    return mel, f0, g


def test_the_linear_down_path_deletes_the_sites_rather_than_filtering_them():
    """``downs[]`` hands the trunk band-limited copies of the excitation.  The
    only activation there doing work the trunk needs is the first, at the
    output rate, which rectifies the sine and creates the harmonics; the ones
    after it run on a signal that is already rectified and already band-limited
    to half their own rate, so their second-order products land straight above
    Nyquist.  That is the site that bisected at +38.4 dB of excess at
    ``8000 - k*f0``, and it is deleted here rather than filtered."""

    plain = _generator((5, 4, 4, 4))
    linear = _generator((5, 4, 4, 4), linear_down_path=True)
    linear.load_state_dict(plain.state_dict(), strict=True)

    for y, skip in _down_skips(plain):
        assert torch.allclose(skip, F.leaky_relu(y, 0.2), atol=1e-6)
    for y, skip in _down_skips(linear):
        # the conv's output reaches the trunk untouched
        assert torch.allclose(skip, y, atol=1e-6)

    # one site left, at the output rate, and the rest of the loop is conv,
    # decimate, conv
    assert plain.down_rates == (32000, 8000, 2000, 500)
    assert linear.down_rates == (32000,)
    assert len(linear.down_activations) == 1

    # and it costs nothing in the weights, which is why the layout carries it
    assert set(plain.state_dict()) == set(linear.state_dict())


def test_the_up_activation_can_run_after_the_upsampler():
    """Same nonlinearity, same parameters, four times the rate.  Stage 3's runs
    at 32 kHz instead of 8 kHz, so it folds about 16 kHz rather than about
    4 kHz -- which is what an anti-aliased activation buys, at the stage's own
    factor and without the trip back, because the signal is going to that rate
    anyway."""

    before = _generator((5, 4, 4, 4))
    after = _generator((5, 4, 4, 4), up_activation_after_upsample=True)
    after.load_state_dict(before.state_dict(), strict=True)

    def trunk(decoder):
        """``(upsampler input, what the res block reads)`` per up stage."""
        seen, handles = {}, []
        for index, block in enumerate(decoder.upsample_blocks):
            handles.append(
                block.register_forward_pre_hook(
                    lambda _m, inputs, index=index: seen.__setitem__(
                        ("in", index), inputs[0]
                    )
                )
            )
        for index, block in enumerate(decoder.upsample_conv_blocks):
            handles.append(
                block.register_forward_pre_hook(
                    lambda _m, inputs, index=index: seen.__setitem__(
                        ("out", index), inputs[0]
                    )
                )
            )
        with torch.no_grad():
            decoder.eval()(*_inputs())
        for handle in handles:
            handle.remove()
        return seen

    # The upsampler's own input is the observable: with the activation ahead
    # of it that input is already rectified and the res block reads ``ups(x)``
    # unchanged, and with the activation moved the res block reads
    # ``leaky_relu(ups(x))``.  Both directions are asserted, so neither case
    # can pass by the two expressions happening to agree.
    for decoder, moved in ((before, False), (after, True)):
        seen = trunk(decoder)
        for index, ups in enumerate(decoder.upsample_blocks):
            with torch.no_grad():
                interpolated = ups(seen[("in", index)])
            activated = F.leaky_relu(interpolated, 0.2)
            # the skip is concatenated after the trunk, so the trunk is the
            # leading channels
            got = seen[("out", index)][:, : interpolated.shape[1]]
            expected, other = (
                (activated, interpolated) if moved else (interpolated, activated)
            )
            assert torch.allclose(got, expected, atol=1e-6)
            assert not torch.allclose(got, other, atol=1e-6)

    # the site rates move with it, which is what ``antialias_rates`` indexes
    assert before.up_rates == (100, 500, 2000, 8000)
    assert after.up_rates == (500, 2000, 8000, 32000)
    assert set(before.state_dict()) == set(after.state_dict())


def test_the_flags_move_which_rates_can_be_protected():
    """``antialias_rates`` names a rate, and these two decide which rates
    exist.  A config that keeps naming 100 Hz after moving the up sites is
    protecting nothing, which is the failure that option exists to end."""

    from rvc.lib.algorithm.generators.refinegan2 import loop_rates

    assert loop_rates(32000, (5, 4, 4, 4)) == (
        (32000, 8000, 2000, 500),
        (100, 500, 2000, 8000),
    )
    assert loop_rates(32000, (5, 4, 4, 4), linear_down_path=True) == (
        (32000,),
        (100, 500, 2000, 8000),
    )
    assert loop_rates(32000, (5, 4, 4, 4), up_activation_after_upsample=True) == (
        (32000, 8000, 2000, 500),
        (500, 2000, 8000, 32000),
    )
    # With both on the decoder has four activation rates, not five, and 100 Hz
    # is not one of them.
    assert loop_rates(
        32000, (5, 4, 4, 4), linear_down_path=True, up_activation_after_upsample=True
    ) == ((32000,), (500, 2000, 8000, 32000))

    with pytest.raises(ValueError, match="match no activation rate") as excinfo:
        _generator(
            (5, 4, 4, 4),
            antialias_rates=[100, 500, 2000, 8000, 32000],
            linear_down_path=True,
            up_activation_after_upsample=True,
        )
    message = str(excinfo.value)
    assert "linear_down_path=True" in message
    assert "up_activation_after_upsample=True" in message

    # ...and the rates that survive still reach both loops
    decoder = _generator(
        (5, 4, 4, 4),
        antialias_rates=[500, 2000, 8000, 32000],
        linear_down_path=True,
        up_activation_after_upsample=True,
    )
    assert all(
        isinstance(act, AntiAliasedActivation) for act in decoder.up_activations
    )
    assert isinstance(decoder.down_activations[0], AntiAliasedActivation)


@pytest.mark.parametrize(
    "flags",
    [
        {"linear_down_path": True},
        {"up_activation_after_upsample": True},
        {"linear_down_path": True, "up_activation_after_upsample": True},
    ],
)
def test_the_structural_flags_are_in_the_layout(flags):
    """Neither adds, removes or reshapes a tensor, so a checkpoint trained
    without them loads perfectly into a decoder that has them -- and computes
    something else."""

    plain = _wrap(_generator((5, 4, 4, 4)))
    moved = _wrap(_generator((5, 4, 4, 4), **flags))
    assert set(plain.dec.state_dict()) == set(moved.dec.state_dict())

    layout = decoder_layout(moved)
    for key, value in flags.items():
        assert layout[key] is value
    assert_decoder_layout_matches(moved, {"decoder_layout": layout})

    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(moved, {"decoder_layout": decoder_layout(plain)})

    # absent means off: every checkpoint written before the flags existed ran
    # the full down loop and activated ahead of every upsampler
    without = {k: v for k, v in layout.items() if k not in flags}
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(moved, {"decoder_layout": without})
    assert_decoder_layout_matches(
        plain, {"decoder_layout": {k: v for k, v in decoder_layout(plain).items()
                                   if k not in flags}}
    )


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_turns_both_structural_flags_on(sample_rate):
    """Not filtering the fold, not making it.  Both change what the decoder
    computes at fixed weights, so they are a fresh pretrain rather than
    something a running experiment picks up -- and the keys ship written out
    so the choice is in the config instead of buried in a default."""

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    assert model["refinegan2_linear_down_path"] is True
    assert model["refinegan2_up_activation_after_upsample"] is True

    # The config has to *build*: ``antialias_rates`` names rates, the flags
    # decide which rates exist, and a rate that matches no site is refused at
    # construction.  So this is the assertion that the three keys agree.
    decoder = RefineGAN2Generator(
        sample_rate=sample_rate,
        upsample_rates=tuple(model["upsample_rates"]),
        num_mels=model["inter_channels"],
        gin_channels=model["gin_channels"],
        upsample_initial_channel=model["upsample_initial_channel"],
        antialias_stages=model["refinegan2_antialias_stages"],
        antialias=model["refinegan2_antialias"],
        antialias_rates=model["refinegan2_antialias_rates"],
        source_gain=model["refinegan2_source_gain"],
        linear_down_path=model["refinegan2_linear_down_path"],
        up_activation_after_upsample=model[
            "refinegan2_up_activation_after_upsample"
        ],
    )
    assert len(decoder.down_activations) == 1
    assert decoder.up_rates[-1] == sample_rate


def test_the_synthesizer_passes_both_flags_through():
    from rvc.lib.algorithm.synthesizers import Synthesizer

    import inspect

    source = inspect.getsource(Synthesizer.__init__)
    assert "refinegan2_linear_down_path" in source
    assert "refinegan2_up_activation_after_upsample" in source


def test_the_gain_is_positive_at_every_sample():
    """The softplus runs after the interpolators, and that is what makes the
    envelope positive *where it multiplies the excitation*.

    Run at the frame rate it guarantees positivity to a sinc that then rings
    through zero, and a negative gain is not a small error: it is the
    excitation changing sign inside a frame, every harmonic phase-flipped.
    Nothing downstream can undo it, because nothing was aliased -- it is the
    envelope the trunk was handed.
    """

    decoder = _generator((5, 4, 4, 4))
    decoder.has_source_gain = True
    decoder.source_gain = torch.nn.Conv1d(192, 1, 1)
    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d

    decoder.source_gain_ups = torch.nn.ModuleList(
        [
            AntiAliasedUpsample1d(
                rate,
                filter_width=decoder.filter_width[stage],
                rolloff=decoder.rolloff[stage],
                filter_beta=decoder.filter_beta[stage],
            )
            for stage, rate in enumerate((5, 4, 4, 4))
        ]
    )

    # A projection with no reason to be smooth, which is the case the sinc
    # interpolators were put here for in the first place.
    torch.manual_seed(0)
    with torch.no_grad():
        decoder.source_gain.weight.normal_(0.0, 0.3)
        decoder.source_gain.bias.zero_()
        mel = torch.randn(1, 192, 120)
        har_source = torch.ones(1, 1, 120 * 320)
        gain = decoder._apply_source_gain(har_source, mel)

        # what the other order would have produced, on the same weights
        rough = F.softplus(decoder.source_gain(mel))
        for ups in decoder.source_gain_ups:
            rough = ups(rough)
        rough = rough[..., : har_source.shape[-1]]

    assert gain.min() > 0.0
    # ...and the point of the move: the frame-rate order does not clear zero
    assert rough.min() < 0.0
    # not a rounding artefact at one sample either
    assert (rough < 0).float().mean() > 0.01


def test_the_gain_is_still_the_identity_at_initialisation():
    """The interpolators preserve DC, so moving the softplus past them leaves
    ``softplus(0.5413) = 1`` exactly where it was -- which is what lets a run
    switch the gain on without rescaling its source on step zero."""

    decoder = RefineGAN2Generator(
        sample_rate=32000,
        upsample_rates=(5, 4, 4, 4),
        num_mels=192,
        gin_channels=256,
        upsample_initial_channel=512,
        source_gain=True,
    )
    with torch.no_grad():
        har_source = torch.ones(1, 1, 40 * 320)
        gain = decoder._apply_source_gain(har_source, torch.randn(1, 192, 40))
    # The residual is the interpolators' own DC error at the segment edge, and
    # it is *smaller* this way round: 7.3e-5 against 2.1e-4 with the softplus
    # at the frame rate.
    assert torch.allclose(gain, torch.ones_like(gain), atol=2e-4)


def test_the_softplus_placement_is_in_the_layout():
    """It is not a setting -- there is one order -- but the two orders are
    different signal paths at identical weights, so a checkpoint from before
    the move has to be refused rather than rendered through the wrong one."""

    with_gain = _wrap(
        RefineGAN2Generator(
            sample_rate=32000,
            upsample_rates=(5, 4, 4, 4),
            num_mels=192,
            gin_channels=256,
            upsample_initial_channel=512,
            source_gain=True,
        )
    )
    layout = decoder_layout(with_gain)
    assert layout["source_gain_order"] == "post"
    assert_decoder_layout_matches(with_gain, {"decoder_layout": layout})

    # absent means the frame-rate order, which is every checkpoint before it
    without = {k: v for k, v in layout.items() if k != "source_gain_order"}
    with pytest.raises(ValueError, match="source_gain_order"):
        assert_decoder_layout_matches(with_gain, {"decoder_layout": without})

    # a decoder with no gain has no softplus to place, and must not mismatch
    # another one over it
    plain = _wrap(_generator((5, 4, 4, 4)))
    assert decoder_layout(plain)["source_gain_order"] is None
    assert_decoder_layout_matches(
        plain,
        {
            "decoder_layout": {
                k: v
                for k, v in decoder_layout(plain).items()
                if k != "source_gain_order"
            }
        },
    )
