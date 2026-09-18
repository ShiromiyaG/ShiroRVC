"""The decoder settings that leave no trace in the weights.

A reorder keeps all 271 tensors' keys *and* shapes, and the upsamplers register
their interpolation kernels non-persistently, so redesigning one adds no key.
A checkpoint therefore loads cleanly into a decoder whose signal path it was
never trained for, and ``decoder_layout`` is the only thing that can say so.

The anti-aliased activations these settings used to select are gone from this
decoder; what remains from ``resampling`` is the upsamplers' interpolation
filter and the lowpass inside ``_decimate``, which are imaging and decimation
rather than activation fold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

import torch.nn.functional as F  # noqa: E402

from rvc.lib.algorithm.generators.refinegan2 import (  # noqa: E402
    ParallelResBlock,
    ResBlock,
    RefineGAN2Generator,
)
from rvc.train.utils import (  # noqa: E402
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
# what the weights cannot say
# --------------------------------------------------------------------------


def _generator(rates, rate=32000, upsample_filter=None, **design):
    return RefineGAN2Generator(
        sample_rate=rate,
        upsample_rates=tuple(rates),
        num_mels=192,
        gin_channels=256,
        upsample_initial_channel=512,
        **design,
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
    reordered = _generator((5, 4, 4, 4))
    a, b = plain.state_dict(), reordered.state_dict()
    assert a.keys() == b.keys()
    assert {k: v.shape for k, v in a.items()} == {k: v.shape for k, v in b.items()}
    result = reordered.load_state_dict(a, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    # ...and it stays wrong, silently: the trunk now runs at 700/4900/14700 Hz
    # instead of 300/900/6300, and a *strict* load raised nothing.
    assert reordered.upsample_rates != plain.upsample_rates


def test_the_layout_round_trips_through_a_checkpoint():
    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4))
    model.sr = 32000
    layout = decoder_layout(model)
    assert layout == {
        "upsample_rates": [5, 4, 4, 4],
        "source_gain": False,
        "source_bands": 0,
        # The excitation's harmonic slope.  The count is a state-dict shape,
        # but the tilt is a non-persistent buffer that scales every partial,
        # so it travels here for the same reason ``upsample_filter`` does.
        # Defaults are the one-partial source every run before 2026-09-09 had.
        "source_harmonics": 0,
        "source_tilt": 1.0,
        # The per-stage interpolation schedule, ascending with the rate: the
        # last stage's image is the one that reaches the output (path gain
        # -19.9 dB against -49.5 and -59.3 for stages 2 and 1) and it is the
        # only stage whose input is long enough that a 385-tap kernel costs no
        # edge.  Stage 0 stays short for the opposite reason.
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
        {"upsample_rates": [3, 3, 7, 7]},
        {"upsample_rates": [5, 4, 4, 4], "source_gain": True},
        # A checkpoint from before the anti-aliased activations were removed.
        # Nothing in the current layout can see the difference, and its weights
        # were fitted through those activations, so it is refused by name.
        {"upsample_rates": [5, 4, 4, 4], "antialias_rates": [8000, 32000]},
    ],
)
def test_a_mismatch_is_refused_and_names_the_keys(checkpoint_layout):
    model = torch.nn.Module()
    model.dec = _generator((5, 4, 4, 4))
    model.sr = 32000
    layout = dict(checkpoint_layout)
    with pytest.raises(ValueError, match="Decoder layout mismatch") as excinfo:
        assert_decoder_layout_matches(model, {"decoder_layout": layout})
    message = str(excinfo.value)
    assert "upsample_rates" in message
    if "antialias_rates" in checkpoint_layout:
        assert "anti-aliased activations this decoder no longer has" in message


@pytest.mark.parametrize("sample_rate", sorted(LEGACY_UPSAMPLE_RATES))
def test_a_checkpoint_without_the_key_is_read_as_the_old_layout(sample_rate):
    """"Absent" is not "nothing to check": every checkpoint predating the key
    is one of the runs this change re-arranges, which is the case the guard
    exists for.  A run that kept the old layout still resumes untouched."""

    new = torch.nn.Module()
    new.dec = _generator((5, 4, 4, 4), rate=sample_rate)
    new.sr = sample_rate
    with pytest.raises(ValueError, match="Decoder layout mismatch"):
        assert_decoder_layout_matches(new, {})

    # The old layout is the old *filter* too: the interpolation design is as
    # invisible to ``load_state_dict`` as the ordering is, so a run that kept
    # one and changed the other is not "the old layout".  The excitation no
    # longer needs saying here: with the sine there is nothing to normalise,
    # so a keyless checkpoint and a decoder built today agree on that field by
    # construction rather than by matching two deliberately opposed defaults.
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
        for name in ("train.py", "setup.py", "checkpoints.py")
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
def test_the_shipped_config_factorises_the_hop_the_way_it_always_did(sample_rate):
    """The stage rates, and the boundary they put inside the band.

    A stage's anti-image filter keeps ``rolloff`` of the rate it *reads*, and
    the stage mirrors around half of it.  What is pinned here is where that
    mirror lands, because it is the one thing a factorisation decides: the
    last factor alone sets it, so every arrangement of ``{4, 4, 4, 5}`` puts it
    at 3960 or 3168 Hz, and ``[10, 8, 2, 2]`` puts it at 7920.

    ``[10, 8, 2, 2]`` was tried twice and reverted twice.  The first time
    (2026-09-09) it was to raise that mirror, on the theory that the ceiling
    under it was why renders lost their harmonics above ~6 kHz; an overfit
    probe refuted it -- both layouts reproduce a target's harmonic contrast to
    within 0.7 dB up to 13 kHz.  The second time (2026-09-10) it was for
    inharmonic content instead, which contrast cannot see, after renders showed
    three lines between 3 and 5 kHz around the 3960 Hz mirror.  That one was
    reverted on cost: the decoder measured ~2x for a mechanism the layout only
    half addresses, since a residual block folds around half the rate it runs
    at and a block runs at 8000 Hz under either factorisation.  See
    ``RefineGAN2Generator``; this test pins the outcome, not the reasoning.
    """

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    rates = model["upsample_rates"]

    import math

    assert math.prod(rates) == sample_rate // 100
    assert sorted(rates) == sorted(LEGACY_UPSAMPLE_RATES[sample_rate])
    assert rates == sorted(rates, reverse=True)
    # The kernel sizes travel with the rates, even though RefineGAN does not
    # read them -- a config whose two halves disagree is a trap for the next
    # decoder that does.
    assert model["upsample_kernel_sizes"] == [2 * r for r in rates]


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_the_shipped_config_leaves_the_source_at_one_partial(sample_rate):
    """The knob exists; 0 is the shipped value, and deliberately.

    ``source_harmonics`` was added while chasing renders whose harmonics
    stopped near 6 kHz, and shipped at 31 for a day.  It is at 0 because the
    premise did not survive: overfitting one clip, this decoder at 0 partials
    reproduces a target's harmonic contrast to within 0.7 dB up to 13 kHz --
    so the source is not what a trained model's missing harmonics are made of.
    See ``RefineGAN2Generator`` for the table.

    There is a second reason not to raise it casually.  A source that hands
    the trunk its harmonics for free is the case ``source_gain``'s comment
    already reasons about: the trunk stops consulting ``z``, and the KL falls
    because the decoder needs less rather than because the prior caught up.
    Raising this costs a pretrain, so it wants a measurement first.
    """

    model = json.loads(CONFIGS[sample_rate].read_text())["model"]
    assert model["refinegan2_source_harmonics"] == 0
    # The tilt only bites at harmonics > 0, but it travels in the config so a
    # run that raises the count does not silently get a flat source.
    assert model["refinegan2_source_tilt"] > 0.0


@pytest.mark.parametrize("sample_rate", sorted(CONFIGS))
def test_an_exported_model_rebuilds_the_same_decoder(sample_rate):
    """If a decoder key stopped travelling, inference would rebuild a
    different signal path from the run that trained it, with nothing raised.
    ``upsample_rates`` travels separately, in the positional ``config``
    list."""

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
    assert "refinegan2_source_gain" in exported
    assert "refinegan2_start_channels" in exported

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


# --------------------------------------------------------------------------
# the source gain
# --------------------------------------------------------------------------


def test_the_gain_is_the_identity_at_initialisation():
    """Zero weights and a bias of ``softplus^-1(1)``: a run that switches this
    on starts from exactly the excitation it had, and a fine-tune cannot have
    its source silently rescaled on step zero."""

    generator = _generator((5, 4, 4, 4))
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
    replicate-pads, so the extra taps read an invented continuation.  The
    fraction of a 0.4 s segment that padding corrupts falls from 21% at stage 0
    to 0.3% at stage 3, while the path gain from a stage's image to the output
    *rises* (-59.3, -49.5, -19.9 dB for stages 1 to 3).  Both arguments point
    the same way, so the schedule ascends monotonically rather than lengthening
    one stage: stage 0 stays shortest because it is the one place a long kernel
    costs more than the image it removes.
    """

    from rvc.lib.algorithm.generators.refinegan2 import (
        DEFAULT_UPSAMPLE_BETA,
        DEFAULT_UPSAMPLE_ROLLOFF,
        DEFAULT_UPSAMPLE_WIDTH,
    )

    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_ROLLOFF)
    assert len(DEFAULT_UPSAMPLE_WIDTH) == len(DEFAULT_UPSAMPLE_BETA)
    # Ascending with the rate, and strictly so at both ends -- a flat schedule
    # is what this replaced, and it left the worst image at -37.1 dB.
    widths, rolloffs = DEFAULT_UPSAMPLE_WIDTH, DEFAULT_UPSAMPLE_ROLLOFF
    assert list(widths) == sorted(widths)
    assert list(rolloffs) == sorted(rolloffs)
    assert widths[-1] > 2 * widths[0]
    assert rolloffs[-1] > rolloffs[0]
    # Width, rolloff and beta are one design: a wider transition band is what
    # buys the stopband at a given length, so the short stage cannot hold the
    # long stage's rolloff.
    assert widths[0] == 12 and rolloffs[0] == 0.90


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


def decoder_layout_of(decoder):
    holder = torch.nn.Module()
    holder.dec = decoder
    holder.sr = 32000
    return decoder_layout(holder)
