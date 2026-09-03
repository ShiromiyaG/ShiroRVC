"""RefineGAN2's discriminator and spectral loss.

The decoder started as Applio's RefineGAN, and what it inherited is pinned
here against reference values taken from ``.tmp/Applio`` (a read-only
checkout): ``v3``'s branch layout, the multi-scale mel loss reducing to
Applio's, the decimation filter, the ``AdaIN`` noise being training-only.

Where the fork has diverged, this file pins the divergence instead -- ``v4``
against ``v3``, a single-scale mel loss, and the discriminator's per-family
switches.  Neither kind of difference is visible in a checkpoint or a loss
curve, only in what a run converges to.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.configs.vocoders import get_discriminator_id  # noqa: E402
from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402
from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (  # noqa: E402
    DISCRIMINATOR_VERSIONS,
    DiscriminatorP,
    DiscriminatorR,
    DiscriminatorS,
)
from rvc.lib.algorithm.generators import refinegan2 as R  # noqa: E402

CONFIG = ROOT / "rvc" / "configs" / "refinegan2" / "32000.json"
CONFIG_32K = CONFIG  # kept as a name; RefineGAN ships at 32 kHz only


def _generator(**overrides):
    model = json.loads(CONFIG.read_text())["model"]
    torch.manual_seed(0)
    kwargs = dict(
        sample_rate=32000,
        upsample_rates=model["upsample_rates"],
        num_mels=model["inter_channels"],
        gin_channels=model["gin_channels"],
        upsample_initial_channel=model["upsample_initial_channel"],
    )
    kwargs.update(overrides)
    return R.RefineGAN2Generator(**kwargs)


def _inputs(batch=1, frames=8):
    model = json.loads(CONFIG.read_text())["model"]
    return (
        torch.randn(batch, model["inter_channels"], frames),
        torch.full((batch, frames), 200.0),
        torch.randn(batch, 256, 1),
    )


# --------------------------------------------------------------------------
# discriminator
# --------------------------------------------------------------------------


def test_v3_builds_the_branches_applio_builds():
    model = MPD_MSD_Combined(False, version="v3")
    kinds = [type(d) for d in model.discriminators]
    assert kinds == [DiscriminatorS] + [DiscriminatorP] * 5 + [DiscriminatorR] * 3
    # v2 happens to hold the same *number* of tensors, so the count alone
    # would not catch a wrong branch -- the parameter total, verified against
    # Applio's own v3 layout, does.
    assert sum(p.numel() for p in model.parameters()) == 47_028_034


def test_the_default_version_leaves_every_other_vocoder_where_it_was():
    """v2 is what ``mpd_msd`` meant before this parameter existed."""

    default = MPD_MSD_Combined(False)
    explicit = MPD_MSD_Combined(False, version="v2")
    assert default.version == "v2"
    assert [type(d) for d in default.discriminators] == [
        type(d) for d in explicit.discriminators
    ]
    assert [type(d) for d in default.discriminators] == (
        [DiscriminatorS] + [DiscriminatorP] * 8
    )
    assert not any(isinstance(d, DiscriminatorR) for d in default.discriminators)


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError):
        MPD_MSD_Combined(False, version="v9")


def test_the_v3_branches_return_the_shapes_the_loop_expects():
    model = MPD_MSD_Combined(False, version="v3").eval()
    real = torch.randn(2, 1, 17640) * 0.1
    fake = torch.randn(2, 1, 17640) * 0.1
    with torch.no_grad():
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(real, fake)
    assert len(y_d_rs) == len(y_d_gs) == len(fmap_rs) == len(fmap_gs) == 9
    for logits_r, logits_g in zip(y_d_rs, y_d_gs):
        assert logits_r.shape == logits_g.shape
        assert logits_r.ndim == 2 and logits_r.shape[0] == 2
    for maps_r, maps_g in zip(fmap_rs, fmap_gs):
        assert [t.shape for t in maps_r] == [t.shape for t in maps_g]


def test_the_resolution_branch_reads_a_spectrogram():
    """The point of v3: a defect that is one line in frequency and stationary in
    time is invisible to a period branch and obvious here."""

    branch = DiscriminatorR([512, 50, 240]).eval()
    with torch.no_grad():
        magnitude = branch.spectrogram(torch.randn(2, 1, 8820))
    assert magnitude.shape[0] == 2
    assert magnitude.shape[1] == 512 // 2 + 1
    assert (magnitude >= 0).all()


# --------------------------------------------------------------------------
# spectral loss
# --------------------------------------------------------------------------


def test_refinegan2_trains_against_a_single_scale_mel_loss():
    """Applio picks the multi-scale mel loss for RefineGAN; this config does
    not.  The two are not interchangeable at the same ``c_mel``: the
    multi-scale branch divides by three and the single-scale one does not, so
    the effective weight is 45 here against 15 there.  Pinned in the config
    because that is where the choice is made -- the fork selects the loss by
    key rather than by vocoder name."""

    train = json.loads(CONFIG.read_text())["train"]
    assert train["spectral_loss"] == "L1 Mel Loss"
    assert train["c_mel"] == 45


def test_the_multi_scale_mel_loss_is_applios_when_the_chouwagan_stack_is_off():
    """``safe_log`` and a weighted distance are this fork's additions, and both
    are gated on the ChouwaGAN stack.  With it off -- which is RefineGAN -- the
    module must reduce to Applio's: ``log10`` of a clamped magnitude, plain L1,
    summed over the seven resolutions."""

    from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss

    loss = MultiScaleMelSpectrogramLoss(
        sample_rate=32000, safe_log=False, loss_fn=torch.nn.L1Loss()
    )
    assert loss.stft_params == [
        (5, 32), (10, 64), (20, 128), (40, 256),
        (80, 512), (160, 1024), (320, 2048),
    ]

    torch.manual_seed(7)
    real = torch.randn(2, 1, 17640) * 0.1
    fake = torch.randn(2, 1, 17640) * 0.1
    # Recomputed here the way Applio writes it, rather than trusted.
    expected = 0.0
    log_base = torch.log(torch.tensor(10.0))
    for n_mels, window in loss.stft_params:
        real_mels = loss.mel_spectrogram(real, n_mels, window)
        fake_mels = loss.mel_spectrogram(fake, n_mels, window)
        expected = expected + torch.nn.functional.l1_loss(
            torch.log(real_mels.clamp(min=1e-5)) / log_base,
            torch.log(fake_mels.clamp(min=1e-5)) / log_base,
        )
    assert torch.equal(loss(real, fake), expected)


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------


def test_adain_noise_is_a_training_only_cost():
    """Six draws per stage, at the output rate on the last one: 25% of the
    forward, measured.  It is a regulariser, so eval does not need it."""

    generator = _generator()
    mel, f0, g = _inputs()

    generator.eval()
    with torch.no_grad():
        for module in generator.modules():
            if isinstance(module, R.AdaIN):
                x = torch.randn(1, module.weight.shape[0], 32)
                assert torch.equal(module(x), module.activation(x))

    generator.train()
    for module in generator.modules():
        if isinstance(module, R.AdaIN):
            with torch.no_grad():
                module.weight.fill_(1.0)
                x = torch.randn(1, module.weight.shape[0], 32)
                assert not torch.equal(module(x), module(x))
            break


def test_remove_weight_norm_runs_and_preserves_the_output():
    """It could not run at all before: the old call raised on the first layer,
    and nothing caught it because ``Synthesizer`` walks the decoder itself."""

    generator = _generator().eval()
    mel, f0, g = _inputs()
    with torch.no_grad():
        torch.manual_seed(0)
        before = generator(mel, f0, g)
    generator.remove_weight_norm()
    with torch.no_grad():
        torch.manual_seed(0)
        after = generator(mel, f0, g)
    assert torch.allclose(before, after, atol=1e-4)
    assert not any(
        hasattr(m, "parametrizations") and hasattr(m.parametrizations, "weight")
        for m in generator.modules()
    )


# --------------------------------------------------------------------------
# torch.compile
# --------------------------------------------------------------------------


def test_nothing_in_the_decoder_forces_a_cpu_kernel():
    """``enable_decoder_compile`` used to fall back to eager on every RefineGAN
    step because Inductor emitted a *CPU* C++ kernel (needing MSVC on
    Windows), from three independent sources: ``self.upp`` being an
    ``np.int64`` (Dynamo wraps a traced numpy scalar as a CPU tensor),
    ``SineGenerator.forward``'s cumsum lowering to a codegen path that raises,
    and ``torchaudio.functional.resample`` rebuilding its sinc kernel from
    Python ints on every call. Compiling on CUDA is too slow and
    machine-dependent for a unit test, so what is pinned here is the three
    underlying properties instead.
    """

    generator = _generator()
    assert type(generator.upp) is int
    assert getattr(R.SineGenerator.forward, "_torchdynamo_disable", False)
    assert getattr(R.RefineGAN2Generator._decimate, "_torchdynamo_disable", False)


def test_the_decimation_filter_is_unchanged():
    """``_decimate`` exists to keep torchaudio's resample out of the graph, not
    to replace it -- its stopband attenuation is well beyond what a
    ``FixedLowPass1d`` of the shape the upsamplers use provides, so swapping
    it would be a numerical change hidden inside a build fix."""

    import inspect

    source = inspect.getsource(R.RefineGAN2Generator._decimate)
    assert "torchaudio.functional.resample" in source
    assert "lowpass_filter_width=64" in source
    assert "beta=14.769656459379492" in source


def test_the_discriminator_version_is_overridable_from_the_config():
    """v3 is the default because it is Applio's choice for RefineGAN and the
    better discriminator -- its spectrogram branches are the only ones that see
    a narrow, stationary frequency defect.  It is also 1.8 GiB more at batch 8,
    which does not fit an 8 GB card, so the config has to be able to say v2.
    """

    model = json.loads(CONFIG.read_text())["model"]
    # ``v4`` is v3's branches with a cheaper frequency stride; ``v3`` restores
    # exact Applio parity.  Either way the layout is v3's, not v2's.
    assert model["d_version"] in ("v3", "v4")

    def resolve(config_model):
        # Mirrors ``_build_d_model``: the registry chooses, ``d_version`` wins.
        version = "v3" if get_discriminator_id("refinegan2") == "mpd_msd_v3" else "v2"
        return str(getattr(config_model, "d_version", None) or version)

    assert resolve(types.SimpleNamespace()) == "v3"
    assert resolve(types.SimpleNamespace(d_version="v2")) == "v2"
    assert resolve(types.SimpleNamespace(d_version=None)) == "v3"


def test_the_upsampler_folds_its_gain_into_the_kernel():
    """``factor * conv_transpose(...)`` allocated a second tensor at the
    *upsampled* rate -- 36 MiB at the output stage, batch 8 over 0.4 s.  Folding
    the gain into the kernel is 1 ULP different (2.5e-7 relative, and exact at
    factor 2) and 5% off the decoder forward."""

    import torch.nn.functional as F

    from rvc.lib.algorithm.resampling import AntiAliasedUpsample1d, lowpass_kernel

    for factor in (2, 3, 7):
        module = AntiAliasedUpsample1d(
            factor, filter_width=12, rolloff=0.90, filter_beta=6.0
        )
        plain = lowpass_kernel(factor, 12, 0.90, 6.0)
        assert torch.allclose(module.kernel, plain * factor)

        torch.manual_seed(0)
        x = torch.randn(2, 8, 120)
        kernel = plain.expand(8, -1, -1).contiguous()
        padded = F.pad(x, (module.pad, module.pad + 1), mode="replicate")
        before = factor * F.conv_transpose1d(
            padded, kernel, stride=factor, padding=module.pad_left, groups=8
        )
        before = before[..., : x.shape[-1] * factor]
        assert torch.allclose(module(x), before, atol=1e-5, rtol=1e-5)


def test_v4_is_v3_minus_its_longest_period():
    """The fork's variant: ``v3`` with the widest period branch dropped and
    nothing else touched.

    The spectrogram branches keep full frequency resolution on purpose. They
    are the only part of this discriminator with any resolution above 10 kHz,
    and the deficit being chased there is a 17-19 dB hole in exactly that band,
    so cheapening the frequency axis is the one saving that works against the
    thing the change is for. The longest period is the safe branch to lose
    instead: a longer period folds at a lower rate, so it says the least about
    the top octave.
    """

    periods_v3, resolutions_v3, strides_v3 = DISCRIMINATOR_VERSIONS["v3"]
    periods_v4, resolutions_v4, strides_v4 = DISCRIMINATOR_VERSIONS["v4"]
    assert periods_v4 == periods_v3[:-1]
    assert resolutions_v4 == resolutions_v3
    assert strides_v3 == strides_v4 == (1, 1, 1)

    model = MPD_MSD_Combined(False, version="v4")
    branch = [d for d in model.discriminators if isinstance(d, DiscriminatorR)][0]
    assert [c.stride for c in branch.convs] == [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)]

    # Dropping a branch *does* change the parameter count, which is the whole
    # difference from the frequency-strided schedule this name used to mean --
    # that one was checkpoint-compatible with v3 and this one is not. The
    # ``discriminator_periods`` key is what turns that into an error message.
    assert sum(p.numel() for p in model.parameters()) < sum(
        p.numel() for p in MPD_MSD_Combined(False, version="v3").parameters()
    )


def test_the_frequency_strided_schedule_is_still_reachable():
    """It stopped being a version, not an option: the probe that justified
    ``(1, 2, 2)`` still stands, and ``d_frequency_strides`` still names it."""

    model = MPD_MSD_Combined(False, version="v3", frequency_strides=(1, 2, 2))
    branch = [d for d in model.discriminators if isinstance(d, DiscriminatorR)][0]
    assert [c.stride for c in branch.convs] == [(1, 1), (1, 2), (2, 2), (2, 2), (1, 1)]
    # A stride schedule changes the grid, never the parameter count.
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in MPD_MSD_Combined(False, version="v3").parameters()
    )


def test_the_fine_hop_branch_is_what_catches_the_frame_rate_defect():
    """Pinned because two plausible "optimisations" both destroy detection of
    the frame-rate AM defect: reusing a coarser spectrogram branch, or
    trading the fine-hop branch for a larger FFT, both measured at chance.
    100 Hz is a 10 ms period, and only the ~1 ms hop resolves it -- a coarser
    hop aliases the modulation to DC. The branch reads mirroring in *time*."""

    _, resolutions, _ = DISCRIMINATOR_VERSIONS["v4"]
    hops = sorted(hop for _n_fft, hop, _win in resolutions)
    assert hops[0] == 50
    assert 50 / 44100 * 1000 < 10.0 / 2  # under half the 10 ms period

    with pytest.raises(ValueError):
        DiscriminatorR([512, 50, 240], frequency_strides=(1, 2))


# --------------------------------------------------------------------------
# 32 kHz
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [CONFIG_32K, CONFIG])
def test_the_upsample_schedule_multiplies_to_the_hop(path):
    """``self.upp`` is the product of the rates and the decoder writes
    ``frames * upp`` samples, so a schedule that does not multiply to the hop
    silently produces audio of the wrong length rather than failing."""

    config = json.loads(path.read_text())
    hop = config["data"]["hop_length"]
    assert math.prod(config["model"]["upsample_rates"]) == hop
    assert config["train"]["segment_size"] % hop == 0


def test_the_config_keeps_a_forty_frame_segment():
    """The segment is what the discriminator branches and the mel losses see,
    and it is a *frame* count -- 40 frames, 0.4 s, whatever the rate.  It used
    to be checked as an agreement between the 32 and 44.1 kHz configs; with
    44.1 kHz gone the invariant is the number itself."""

    at_32k = json.loads(CONFIG_32K.read_text())

    def frames(config):
        return config["train"]["segment_size"] // config["data"]["hop_length"]

    assert frames(at_32k) == 40
    assert at_32k["data"]["sample_rate"] == 32000
    assert at_32k["train"]["segment_size"] == 40 * at_32k["data"]["hop_length"]


def test_the_registry_ships_only_32k():
    """44.1 kHz was removed from RefineGAN.  Pinned so the config directory and
    the registry cannot drift apart -- a rate listed here with no config is a
    crash at model construction, and a config with no listing is unreachable."""

    from rvc.configs.vocoders import get_vocoder_sample_rates

    assert get_vocoder_sample_rates("refinegan2") == [32000]
    shipped = sorted(
        int(p.stem) for p in (ROOT / "rvc" / "configs" / "refinegan2").glob("*.json")
    )
    assert shipped == get_vocoder_sample_rates("refinegan2")


@pytest.mark.parametrize("rate", [32000])
def test_the_decoder_writes_the_right_length(rate):
    from rvc.lib.algorithm.synthesizers import Synthesizer

    config = json.loads(
        (ROOT / "rvc" / "configs" / "refinegan2" / ("%d.json" % rate)).read_text()
    )
    model = dict(config["model"])
    model.pop("use_spectral_norm", None)
    hop = config["data"]["hop_length"]
    frames = config["train"]["segment_size"] // hop
    torch.manual_seed(0)
    net = Synthesizer(
        spec_channels=config["data"]["filter_length"] // 2 + 1,
        segment_size=frames,
        use_f0=True,
        sr=rate,
        vocoder="refinegan2",
        checkpointing=False,
        **model
    ).eval()
    with torch.no_grad():
        out = net.dec(
            torch.randn(1, model["inter_channels"], frames),
            torch.full((1, frames), 200.0),
            torch.randn(1, 256, 1),
        )
    assert out.shape == (1, 1, frames * hop)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_an_unsupported_rate_is_still_refused(rate):
    """The hardcoded 44.1 kHz check became a registry lookup; it must not have
    become nothing at all -- and 44.1 kHz is now one of the rates it refuses,
    which is the cheapest possible proof that the removal reached the code
    rather than only the config directory."""

    from rvc.lib.algorithm.synthesizers import Synthesizer

    config = json.loads(CONFIG.read_text())
    model = dict(config["model"])
    model.pop("use_spectral_norm", None)
    with pytest.raises(ValueError, match="supports"):
        Synthesizer(
            spec_channels=config["data"]["filter_length"] // 2 + 1,
            segment_size=40,
            use_f0=True,
            sr=rate,
            vocoder="refinegan2",
            checkpointing=False,
            **model
        )


# --------------------------------------------------------------------------
# branch-by-branch configuration
# --------------------------------------------------------------------------


def _build(model):
    """Mirrors what ``_build_d_model`` reads on the mpd_msd path."""

    def setting(name, default=None):
        value = getattr(model, name, None)
        return default if value is None else value

    return MPD_MSD_Combined(
        model.use_spectral_norm,
        use_checkpointing=False,
        version=str(getattr(model, "d_version", None) or "v3"),
        periods=[] if not setting("d_use_periods", True) else setting("d_periods"),
        resolutions=(
            [] if not setting("d_use_resolutions", True) else setting("d_resolutions")
        ),
        frequency_strides=setting("d_frequency_strides"),
        use_msd=bool(setting("d_use_msd", True)),
        sample_rate=int(getattr(model, "sample_rate", 44100)),
        use_univhd=bool(setting("d_use_univhd", False)),
    )


@pytest.mark.parametrize("path", [CONFIG_32K, CONFIG])
def test_the_branch_knobs_are_present_and_inert_by_default(path):
    """``None`` means "whatever ``d_version`` says". The content knobs are
    written out so they are discoverable from the config, not just the code;
    the ``d_use_*`` switches are deliberately left absent when true, since
    they only ever mean anything as ``false``. Either way, the config has to
    build the preset its ``d_version`` names."""

    model = json.loads(path.read_text())["model"]
    # The three content knobs are derivable from ``d_version`` and are no
    # longer written out at all; the ``d_use_*`` switches are left absent when
    # true, since they only ever mean anything as ``false``.
    for key in ("d_periods", "d_resolutions", "d_frequency_strides"):
        assert key not in model, f"{key} is derivable from d_version"
    for key in ("d_use_msd", "d_use_periods", "d_use_resolutions"):
        assert model.get(key, True) is True, f"{key} would change the preset"

    rate = json.loads(path.read_text())["data"]["sample_rate"]
    built = _build(types.SimpleNamespace(
        **dict(model, d_use_univhd=False, sample_rate=rate)))
    preset = MPD_MSD_Combined(model["use_spectral_norm"],
                              version=model["d_version"], sample_rate=rate)
    assert [type(a) for a in built.discriminators] == [
        type(b) for b in preset.discriminators
    ]
    assert built.periods == preset.periods


def test_each_family_can_be_replaced_or_turned_off():
    config = json.loads(CONFIG.read_text())
    model = dict(config["model"], sample_rate=config["data"]["sample_rate"])
    n_periods = len(
        MPD_MSD_Combined(False, version=model["d_version"],
                         sample_rate=model["sample_rate"]).periods
    )

    def variant(**overrides):
        merged = dict(model, d_use_univhd=False)
        merged.update(overrides)
        return _build(types.SimpleNamespace(**merged))

    assert [type(d) for d in variant(d_use_msd=False).discriminators] == (
        [DiscriminatorP] * n_periods + [DiscriminatorR] * 3
    )
    assert [type(d) for d in variant(d_periods=[2, 3]).discriminators] == (
        [DiscriminatorS] + [DiscriminatorP] * 2 + [DiscriminatorR] * 3
    )
    # An empty list means "none of this family" -- the distinction a falsy
    # check loses, and the reason ``setting`` tests against ``None``.
    assert [type(d) for d in variant(d_resolutions=[]).discriminators] == (
        [DiscriminatorS] + [DiscriminatorP] * n_periods
    )
    assert variant(d_frequency_strides=[1, 1, 1]).frequency_strides == (1, 1, 1)
    assert variant(d_resolutions=[[512, 50, 240]]).resolutions == ((512, 50, 240),)


def test_a_discriminator_with_no_branches_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one branch"):
        MPD_MSD_Combined(False, use_msd=False, periods=[], resolutions=[])


def test_a_malformed_resolution_is_refused():
    with pytest.raises(ValueError, match="n_fft"):
        MPD_MSD_Combined(False, resolutions=[[512, 50]])


def test_the_branches_still_run_when_the_families_are_edited():
    model = MPD_MSD_Combined(
        False, version="v4", periods=[2], resolutions=[[512, 50, 240]], use_msd=False
    ).eval()
    real = torch.randn(2, 1, 17640) * 0.1
    fake = torch.randn(2, 1, 17640) * 0.1
    with torch.no_grad():
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(real, fake)
    assert len(y_d_rs) == len(y_d_gs) == len(fmap_rs) == len(fmap_gs) == 2


# --------------------------------------------------------------------------
# FP16
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
@pytest.mark.parametrize(
    "vocoder,rate", [("refinegan2", 32000)]
)
def test_a_step_survives_fp16_autocast(vocoder, rate):
    """``train.py`` has driven ``use_fp16`` through ``autocast`` and a
    ``GradScaler`` for a while, but this decoder had not been run under it.
    The risk is not speed -- it is a branch that quietly refuses half
    precision (``torch.stft``, the resampling kernels) and takes the loss to
    NaN on step one."""

    from torch.amp import autocast

    from rvc.lib.algorithm.synthesizers import Synthesizer

    config = json.loads(
        (ROOT / "rvc" / "configs" / vocoder / ("%d.json" % rate)).read_text()
    )
    model = dict(config["model"])
    model.pop("use_spectral_norm", None)
    hop = config["data"]["hop_length"]
    frames = 8
    torch.manual_seed(0)
    net_g = (
        Synthesizer(
            spec_channels=config["data"]["filter_length"] // 2 + 1,
            segment_size=frames,
            use_f0=True,
            sr=rate,
            vocoder=vocoder,
            checkpointing=False,
            **model
        )
        .cuda()
        .train()
    )
    net_d = MPD_MSD_Combined(False, version=model.get("d_version", "v4"))
    net_d = net_d.cuda().train()

    decoder = net_g.dec
    latent = torch.randn(1, model["inter_channels"], frames, device="cuda")
    f0 = torch.full((1, frames), 200.0, device="cuda")
    speaker = torch.randn(1, 256, 1, device="cuda")
    wave = torch.randn(1, 1, frames * hop, device="cuda") * 0.5
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10)
    optim = torch.optim.AdamW(
        list(decoder.parameters()) + list(net_d.parameters()), 1e-4
    )

    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            generated = decoder(latent, f0, speaker)
            assert generated.dtype is torch.float16
            outputs = net_d(wave, generated)
            loss = sum(t.float().pow(2).mean() for t in outputs[0]) + sum(
                (1 - t.float()).pow(2).mean() for t in outputs[1]
            )
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    # A scaler that has backed off means a branch overflows every step, which is
    # the failure this test exists to catch.
    assert scaler.get_scale() >= 2.0**10


def test_one_boolean_per_family_turns_it_off():
    """The switch layer.  ``d_periods: []`` says the same thing, but answering
    "off" should not require writing out an empty list -- and ``d_use_msd`` was
    already a boolean, so the other two being lists was simply inconsistent."""

    config = json.loads(CONFIG.read_text())
    model = dict(config["model"], sample_rate=config["data"]["sample_rate"])
    n_periods = len(
        MPD_MSD_Combined(False, version=model["d_version"],
                         sample_rate=model["sample_rate"]).periods
    )

    def kinds(**overrides):
        # UnivHD is not one of the families this is about, and it appends a
        # branch that would show up in every expectation below; the config it
        # ships in may have it on.
        merged = dict(model, d_use_univhd=False)
        merged.update(overrides)
        return [type(d) for d in _build(types.SimpleNamespace(**merged)).discriminators]

    assert kinds(d_use_periods=False) == [DiscriminatorS] + [DiscriminatorR] * 3
    assert kinds(d_use_resolutions=False) == (
        [DiscriminatorS] + [DiscriminatorP] * n_periods
    )
    assert kinds(d_use_msd=False) == (
        [DiscriminatorP] * n_periods + [DiscriminatorR] * 3
    )
    # Off wins over the content key: the two answer different questions, and a
    # leftover list must not resurrect a family someone switched off.
    assert kinds(d_use_periods=False, d_periods=[2, 3]) == (
        [DiscriminatorS] + [DiscriminatorR] * 3
    )


def test_no_grad_real_leaves_the_generator_gradient_untouched():
    """The generator update's real pass is a target, not a term to train on.

    ``no_grad_real`` skips its graph.  What must not move is the only gradient
    the generator update actually consumes -- the one that reaches ``y_hat``
    through the fake branch and through the feature matching loss's fake side.
    The real feature maps enter that loss as constants either way, so the two
    paths are the same function of ``y_hat``; this pins that they are also the
    same number.
    """

    from rvc.train.losses import feature_loss, generator_loss

    torch.manual_seed(0)
    model = MPD_MSD_Combined(
        False, version="v3", periods=[5, 7], resolutions=[[512, 50, 240]]
    )
    real = torch.randn(2, 1, 4096)

    def wave_gradient(no_grad_real):
        fake = torch.randn(2, 1, 4096, generator=torch.Generator().manual_seed(1))
        fake.requires_grad_(True)
        _, fake_logits, real_features, fake_features = model(
            real, fake, no_grad_real=no_grad_real
        )
        loss = generator_loss(fake_logits) + 2.0 * feature_loss(
            real_features, fake_features
        )
        loss = loss[0] if isinstance(loss, tuple) else loss
        (wave_grad,) = torch.autograd.grad(loss, fake)
        parameter_grads = [
            p.grad is not None for p in model.parameters() if p.requires_grad
        ]
        return wave_grad, any(parameter_grads)

    joint, _ = wave_gradient(False)
    detached, _ = wave_gradient(True)
    assert torch.allclose(joint, detached, atol=0, rtol=0)


def test_no_grad_real_builds_no_real_side_graph():
    """The point of the flag: the real logits come back as constants."""

    torch.manual_seed(0)
    model = MPD_MSD_Combined(False, version="v3", periods=[5], resolutions=[])
    real = torch.randn(2, 1, 4096)
    fake = torch.randn(2, 1, 4096, requires_grad=True)

    real_logits, fake_logits, real_features, _ = model(real, fake, no_grad_real=True)
    assert all(not logits.requires_grad for logits in real_logits)
    assert all(not f.requires_grad for branch in real_features for f in branch)
    # The fake side is untouched -- it is what carries the generator's gradient.
    assert all(logits.requires_grad for logits in fake_logits)


def test_no_grad_real_under_checkpointing_skips_only_the_real_branch():
    """Checkpointing and ``no_grad_real`` are one three-way choice.

    The real branch has no graph to recompute, so it must run plain rather
    than through ``checkpoint``; the fake branch must still be checkpointed,
    since it is the one carrying the generator's gradient.
    """

    torch.manual_seed(0)
    model = MPD_MSD_Combined(
        False, version="v3", periods=[5], resolutions=[], use_checkpointing=True
    )
    model.train()
    real = torch.randn(2, 1, 4096)
    fake = torch.randn(2, 1, 4096, requires_grad=True)

    real_logits, fake_logits, real_features, _ = model(real, fake, no_grad_real=True)
    assert all(not logits.requires_grad for logits in real_logits)
    assert all(not f.requires_grad for branch in real_features for f in branch)
    assert all(logits.requires_grad for logits in fake_logits)


def test_forward_still_differentiates_the_real_side_by_default():
    """The discriminator update needs exactly what ``no_grad_real`` removes."""

    torch.manual_seed(0)
    model = MPD_MSD_Combined(False, version="v3", periods=[5], resolutions=[])
    real = torch.randn(2, 1, 4096)
    real_logits, _, _, _ = model(real, torch.randn(2, 1, 4096))
    assert all(logits.requires_grad for logits in real_logits)


# --------------------------------------------------------------------------
# the light period branch
# --------------------------------------------------------------------------


def test_fast_mpd_is_off_unless_asked_for():
    """The *constructor* default is off, so an absent key builds the branches
    every existing discriminator was trained with.

    What the shipped config chooses is a separate question and is deliberately
    not asserted here -- it is a tuning decision that moves, while "absent
    means the old behaviour" is the contract that must not.
    """

    from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (
        FastDiscriminatorP,
    )

    default = MPD_MSD_Combined(False, version="v4", sample_rate=32000)
    assert default.use_fast_mpd is False
    assert not any(isinstance(b, FastDiscriminatorP) for b in default.discriminators)

    # And the trainer's own default agrees with the constructor's.
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    assert 'setting("d_use_fast_mpd", False)' in source


def test_fast_mpd_swaps_the_period_family_and_nothing_else():
    """A capacity swap, not a layout change: same branches in the same order,
    same six feature maps per period branch, so the feature-matching loss keeps
    its shape and only its width changes."""

    from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (
        FastDiscriminatorP,
    )

    common = dict(version="v4", sample_rate=32000, use_univhd=True)
    stock = MPD_MSD_Combined(False, use_fast_mpd=False, **common)
    fast = MPD_MSD_Combined(False, use_fast_mpd=True, **common)

    def layout(model):
        """Branch families in order, with both period classes as one family."""
        return [
            "period" if isinstance(b, (DiscriminatorP, FastDiscriminatorP))
            else type(b).__name__
            for b in model.discriminators
        ]

    assert layout(stock) == layout(fast)
    assert [type(b) for b in fast.discriminators
            if isinstance(b, (DiscriminatorP, FastDiscriminatorP))] == (
        [FastDiscriminatorP] * len(fast.periods)
    )
    assert [type(b) for b in stock.discriminators
            if isinstance(b, (DiscriminatorP, FastDiscriminatorP))] == (
        [DiscriminatorP] * len(stock.periods)
    )

    audio = torch.randn(2, 1, 12800)
    other = torch.randn(2, 1, 12800)
    with torch.no_grad():
        _, _, fmap_stock, _ = stock(audio, other)
        _, _, fmap_fast, _ = fast(audio, other)
    assert [len(f) for f in fmap_stock] == [len(f) for f in fmap_fast]


def test_fast_mpd_is_where_the_parameters_are():
    """The period family is 32.88 M of a v4 discriminator's 38.81 M, which is
    why this is the lever that matters for memory.  Measured end to end at
    32 kHz, batch 8, eager: 631.7 ms / 4528 MiB becomes 623.5 / 3769."""

    common = dict(version="v4", sample_rate=32000, use_univhd=True)
    stock = sum(p.numel() for p in
                MPD_MSD_Combined(False, use_fast_mpd=False, **common).parameters())
    fast = sum(p.numel() for p in
               MPD_MSD_Combined(False, use_fast_mpd=True, **common).parameters())
    assert stock / fast > 4.0
    assert fast < 10e6


def test_the_swap_cannot_load_the_wrong_checkpoint():
    """No separate guard for this one: every period branch changes shape, so a
    strict load raises on its own.  Pinned so that stays true -- the periods
    needed an explicit key precisely because they *do not* change any shape."""

    common = dict(version="v4", sample_rate=32000)
    stock = MPD_MSD_Combined(False, use_fast_mpd=False, **common)
    fast = MPD_MSD_Combined(False, use_fast_mpd=True, **common)
    with pytest.raises(RuntimeError):
        fast.load_state_dict(stock.state_dict(), strict=True)


def test_the_trainer_reads_the_flag():
    import ast

    tree = ast.parse((ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8"))
    names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "setting"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert "d_use_fast_mpd" in names
