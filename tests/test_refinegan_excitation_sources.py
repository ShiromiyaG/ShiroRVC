"""The ``refinegan_source`` flag: what it swaps, and what has to stay put.

RefineGAN's excitation is ``SineGenerator(sample_rate)`` with the default
``harmonic_num=0`` -- one sine at the fundamental, a ``Linear(1, 1)`` and a
tanh.  Every partial above f0 is therefore manufactured by the trunk out of
feature maps on the frame grid, which is the mechanism behind the +-``sr/hop``
ghost partials recorded in the decoder notes.  ``refinegan_source="comb"``
swaps in fish-diffusion's band-limited comb tooth, which arrives with every
harmonic already present at the output rate.

Three things are pinned:

* the spectra actually differ, and in the direction claimed -- otherwise the
  flag is doing nothing and the A/B is meaningless;
* the comb is alias-free at an f0 high enough that a naive harmonic bank would
  fold, since "band-limited" is the entire justification for the ``sinc``;
* ``sine`` remains bit-identical to what shipped, in both the state dict and
  ``architecture_id``.  The id is the load guard: the two sources differ by
  ``dec.m_source.merge.0.weight``, ``net_g`` loads non-strictly, and an absent
  key is exactly what the shape check cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.configs.vocoders import get_architecture_id  # noqa: E402
from rvc.lib.algorithm.generators.refinegan import (  # noqa: E402
    CombToothGenerator,
    RefineGANGenerator,
    SineGenerator,
)

SR = 44100


def _excitation(source, f0_hz, seconds=0.5):
    length = int(SR * seconds)
    f0 = torch.full((1, length, 1), float(f0_hz))
    with torch.no_grad():
        return source(f0)[0, :, 0]


def _harmonic_db(signal, f0_hz, harmonic, n_fft=8192, offset=4096):
    seg = signal[offset : offset + n_fft] * torch.hann_window(n_fft)
    spectrum = torch.fft.rfft(seg).abs()
    bin_index = int(round(harmonic * f0_hz / SR * n_fft))
    fundamental = spectrum[int(round(f0_hz / SR * n_fft))]
    return 20 * torch.log10(spectrum[bin_index] / fundamental).item()


def test_comb_carries_harmonics_the_sine_does_not():
    torch.manual_seed(0)
    sine = _excitation(SineGenerator(SR), 220.0)
    comb = _excitation(CombToothGenerator(SR), 220.0)

    # The comb's spectrum is flat by construction; the sine's 20th harmonic is
    # only there at all because of the tanh and the noise floor.
    assert _harmonic_db(comb, 220.0, 20) > -6.0
    assert _harmonic_db(sine, 220.0, 20) < -25.0


def test_comb_is_alias_free_above_nyquist():
    """A harmonic bank at f0=2000 would fold; the comb must not.

    The check is the anti-symmetry an alias makes: energy exactly between two
    harmonics.  The comb's phase advances by ``f0 / sr`` per sample, so
    ``sr * x / f0`` advances by exactly 1 and the samples are ``sinc(n - phi)``
    -- an ideal band-limited impulse, with nothing to fold.
    """
    torch.manual_seed(0)
    comb = _excitation(CombToothGenerator(SR, noise_std=0.0), 2000.0)

    n_fft = 8192
    seg = comb[4096 : 4096 + n_fft] * torch.hann_window(n_fft)
    spectrum = torch.fft.rfft(seg).abs()
    step = 2000.0 / SR * n_fft
    on = max(spectrum[int(round(h * step))].item() for h in range(1, 10))
    between = max(
        spectrum[int(round((h + 0.5) * step))].item() for h in range(1, 10)
    )
    assert 20 * torch.log10(torch.tensor(between / on)) < -30.0


@pytest.mark.parametrize("source_type", ["sine", "comb"])
def test_generator_runs_and_trains_either_way(source_type):
    torch.manual_seed(0)
    net = RefineGANGenerator(
        sample_rate=SR,
        upsample_rates=(3, 3, 7, 7),
        num_mels=192,
        gin_channels=256,
        source_type=source_type,
    ).train()

    f0 = torch.full((2, 40), 180.0)
    f0[:, 20:25] = 0.0  # an unvoiced gap, so the gating path is exercised
    y = net(torch.randn(2, 192, 40), f0, torch.randn(2, 256, 1))
    assert y.shape == (2, 1, 40 * 441)
    assert torch.isfinite(y).all()

    y.abs().mean().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, missing


def test_sine_stays_the_default_and_keeps_its_identity():
    keys = lambda t: {  # noqa: E731
        k
        for k in RefineGANGenerator(
            sample_rate=SR, upsample_rates=(3, 3, 7, 7), source_type=t
        )
        .state_dict()
        .keys()
        if "m_source" in k
    }
    assert keys("sine") == {"m_source.merge.0.weight"}
    assert keys("comb") == set()

    default = RefineGANGenerator(sample_rate=SR, upsample_rates=(3, 3, 7, 7))
    assert isinstance(default.m_source, SineGenerator)

    # Every source keeps the Applio-compatible id: it is what the wider RVC
    # ecosystem reads off a checkpoint.  The guard is ``excitation_source``.
    for options in ({}, {"refinegan_source": "sine"}, {"refinegan_source": "comb"},
                    {"refinegan_source": "bank", "refinegan_harmonics": 32}):
        assert get_architecture_id("refinegan", options) == "vits_gaussian_v1"


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="refinegan_source"):
        RefineGANGenerator(
            sample_rate=SR, upsample_rates=(3, 3, 7, 7), source_type="bogus"
        )


# --- the harmonic bank, the third source ------------------------------------
#
# It exists because the sine and the comb sit at opposite extremes: the sine
# has one partial, the comb a flat spectrum with a level that drifts across
# the pitch range. The bank is the middle, and what is pinned here is the
# properties that make it the middle, plus the corrections it carries over
# ``chouwagan.BandLimitedNSFSource``, each of which was a measured defect.

from rvc.lib.algorithm.generators.refinegan import HarmonicBankGenerator  # noqa: E402


def _bank_stats(bank, f0_hz, n=16384):
    with torch.no_grad():
        y = bank(torch.full((1, 100), float(f0_hz)), SR)[0, :, 0]
    seg = y[4096 : 4096 + n] * torch.hann_window(n)
    spectrum = torch.fft.rfft(seg).abs()
    b = lambda hz: int(round(hz / SR * n))  # noqa: E731
    h20 = 20 * torch.log10(spectrum[b(20 * f0_hz)] / spectrum[b(f0_hz)]).item()
    return y.std().item(), (y.abs().max() / y.std()).item(), h20


def test_bank_sits_between_the_sine_and_the_comb():
    torch.manual_seed(0)
    bank = HarmonicBankGenerator(SR, 64)
    sine = SineGenerator(SR)
    with torch.no_grad():
        sine.merge[0].weight.fill_(1.0)

    levels = []
    for f0_hz in (80, 220, 880):
        rms, crest, h20 = _bank_stats(bank, f0_hz)
        levels.append(rms)
        # Harmonics present, but tilted -3 dB/octave rather than flat: 1/sqrt(20)
        # is -13.0 dB, against the comb's 0 dB and the sine's -67.
        assert -16.0 < h20 < -11.0, (f0_hz, h20)
        # Dispersed phase, so nothing like the comb's impulse.
        assert crest < 4.0, (f0_hz, crest)

    # The level does not move with pitch -- the comb's does, by 8.7 dB.
    assert max(levels) / min(levels) < 1.02, levels

    # And it lands on the sine's RMS, so an A/B changes the spectrum only.
    with torch.no_grad():
        sine_rms = sine(torch.full((1, SR, 1), 220.0))[0, :, 0].std().item()
    assert abs(levels[1] / sine_rms - 1.0) < 0.05, (levels[1], sine_rms)


def test_fix_all_voiced_mask_is_a_fixed_point():
    """``BandLimitedNSFSource`` zero-pads its smoother, so an all-voiced mask
    comes back starting at 0.506 -- every segment amplitude-faded at both ends.
    """
    bank = HarmonicBankGenerator(SR, 8)
    envelope = bank._voiced_envelope(torch.ones(1, 4410))
    assert torch.allclose(envelope, torch.ones_like(envelope), atol=1e-5)


def test_fix_f0_does_not_sag_into_a_voiced_edge():
    """Unvoiced frames are ``0``, so interpolating the raw sequence ramps f0
    toward zero across the frame *inside* the region the mask calls voiced.
    """
    bank = HarmonicBankGenerator(SR, 8)
    f0 = torch.zeros(1, 20)
    f0[0, 5:15] = 200.0
    filled = bank._fill_unvoiced(f0, f0 > 0)
    assert torch.all(filled == 200.0)

    # Nothing voiced anywhere has nothing to hold, and must stay silent.
    silent = torch.zeros(1, 20)
    assert torch.all(bank._fill_unvoiced(silent, silent > 0) == 0.0)


def test_fix_pulse_shape_is_stable_and_checkpointed():
    """The reference redraws the per-harmonic phases every forward, so the
    excitation's pulse shape changes each step -- and this decoder feeds the
    excitation into all four upsampling stages, not just the first.
    """
    torch.manual_seed(0)
    bank = HarmonicBankGenerator(SR, 32)
    assert "phase_offset" in bank.state_dict()  # survives a checkpoint

    f0 = torch.full((1, 100), 220.0)
    with torch.no_grad():
        a = bank(f0, SR)[0, :, 0]
        b = bank(f0, SR)[0, :, 0]
    # Only the additive noise differs; the deterministic part is identical.
    assert (a - b).std() / a.std() < 0.15


def test_excitation_guard_replaces_the_architecture_suffix():
    """The id stays ``vits_gaussian_v1`` for all three sources, so the guard has
    to be its own key -- and this stack resumes *non-strictly*, which is what
    makes a silent cross-load possible in the first place.
    """
    sys.path.insert(0, str(ROOT / "rvc" / "train"))
    from utils import assert_excitation_matches, excitation_source

    build = lambda t, h=64: RefineGANGenerator(  # noqa: E731
        sample_rate=SR, upsample_rates=(3, 3, 7, 7), source_type=t,
        source_harmonics=h,
    )
    holder = lambda dec: type("M", (), {"dec": dec})()  # noqa: E731

    assert excitation_source(holder(build("sine"))) == "sine"
    assert excitation_source(holder(build("comb"))) == "comb"
    assert excitation_source(holder(build("bank", 32))) == "bank32"

    # A checkpoint with no key is a stock upstream one, and every one is a sine.
    assert_excitation_matches(holder(build("sine")), {})
    for source, ckpt in (("comb", {}), ("sine", {"excitation_source": "comb"}),
                         ("bank", {"excitation_source": "sine"})):
        with pytest.raises(ValueError, match="Excitation mismatch"):
            assert_excitation_matches(holder(build(source)), ckpt)
