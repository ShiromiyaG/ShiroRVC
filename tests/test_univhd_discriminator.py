"""UnivHD, the harmonic discriminator branch (arXiv 2512.03486).

Pinned here rather than in the RefineGAN parity file because it is *not*
parity: Applio does not have this branch, and the point of these tests is that
turning it on is the only thing that changes anything.  The properties that
matter are the ones a plausible refactor would quietly break -- the Nyquist
criterion on the filter centers, the one-sided ``gamma`` constraint, and the
fact that the frequency axis is the only one ever decimated.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402
from rvc.lib.algorithm.discriminators.single import (  # noqa: E402
    HarmonicFilterBank,
    UnivHDDiscriminator,
)
from rvc.lib.algorithm.discriminators.single.univhd import (  # noqa: E402
    ERB_OFFSET,
    ERB_SLOPE,
    center_frequencies,
    harmonic_orders,
)

CONFIGS = [
    ROOT / "rvc" / "configs" / "refinegan" / "32000.json",
    ROOT / "rvc" / "configs" / "refinegan" / "44100.json",
]


# --------------------------------------------------------------------------
# the filterbank
# --------------------------------------------------------------------------


def test_the_orders_are_the_paper_s_ten_plus_a_half():
    assert harmonic_orders(10) == [0.5] + [float(h) for h in range(1, 11)]
    assert harmonic_orders(10, half_harmonic=False)[0] == 1.0


@pytest.mark.parametrize("sample_rate", [32000, 44100])
def test_no_filter_center_crosses_nyquist(sample_rate):
    """``f_max = fs / (2H)`` is a statement about the *highest order*, not the
    highest center.  Reading it as a cap on ``fc`` alone -- the natural
    misreading, since that is the variable it is written on -- would put the
    tenth harmonic of the top center at 5x Nyquist and fold the whole bank."""

    bank = HarmonicFilterBank(sample_rate=sample_rate, n_fft=2048)
    top = bank.orders.max().item() * bank.centers.max().item()
    assert top <= sample_rate / 2.0
    # And it is *tight*: the criterion places the top order on Nyquist, and the
    # only slack is that the log grid's last step lands short of ``f_max``, so
    # the gap is under one bin.
    assert top > (sample_rate / 2.0) * 2 ** (-1 / bank.bins_per_octave)


def test_centers_are_log_spaced_at_the_requested_resolution():
    centers = center_frequencies(80.0, 2205.0, 24)
    ratios = [b / a for a, b in zip(centers, centers[1:])]
    assert ratios == pytest.approx([2 ** (1 / 24)] * len(ratios))
    assert centers[0] == pytest.approx(80.0)


def test_too_many_harmonics_for_the_rate_is_refused_not_silently_empty():
    with pytest.raises(ValueError, match="f_max > f_min"):
        HarmonicFilterBank(sample_rate=16000, n_fft=2048, harmonics=200, f_min=80.0)


def test_the_bandwidth_follows_the_erb_law_at_the_shifted_center():
    """The bandwidth is ERB of ``h * fc``, not of ``fc``.  The paper's notation
    (``fbw_h``) is indexed by the order, and the psychoacoustics only mean
    anything at the frequency the filter actually sits at -- reading it as
    ERB(fc) would give the tenth harmonic a 33 Hz window at 800 Hz."""

    bank = HarmonicFilterBank(sample_rate=44100, n_fft=8192)
    filters = bank.filters()
    hz = bank.bin_hz
    for order_index in (1, 5, 10):
        for filter_index in (0, bank.n_filters // 2, bank.n_filters - 1):
            row = filters[order_index, filter_index]
            center = (
                bank.orders[order_index] * bank.centers[filter_index]
            ).item()
            expected = (ERB_SLOPE * center + ERB_OFFSET) / bank.gamma().item()
            if center + expected / 2 >= hz[-1]:  # truncated by Nyquist
                continue
            support = hz[row > 0]
            if support.numel() < 3:  # narrower than the FFT resolution
                continue
            # A triangle of half-width ``fbw / 2`` has full support ``fbw``.
            width = (support.max() - support.min()).item()
            assert width == pytest.approx(expected, rel=0.25)
            # The peak sits on the center.
            assert hz[row.argmax()].item() == pytest.approx(center, abs=expected / 2)


def test_gamma_starts_at_one_and_can_only_sharpen():
    """One-sided on purpose: gamma *divides* the bandwidth, so the bank may
    narrow away from the ERB law but never widen past it.  An unconstrained
    parameter would let it collapse into a few wide bands, which is the
    degenerate solution that makes the harmonic indexing pointless."""

    bank = HarmonicFilterBank(sample_rate=44100, n_fft=2048)
    assert bank.gamma().item() == pytest.approx(1.0, abs=1e-2)
    with torch.no_grad():
        bank.gamma_raw.fill_(-50.0)
    assert bank.gamma().item() >= 1.0
    with torch.no_grad():
        bank.gamma_raw.fill_(20.0)
    assert bank.gamma().item() > 5.0


def test_gamma_receives_gradient():
    """The bank is rebuilt every forward precisely so this holds; caching the
    filters is the obvious optimisation and it silently freezes the bank."""

    branch = UnivHDDiscriminator()
    score, _ = branch(torch.randn(1, 1, 8192))
    score.sum().backward()
    assert branch.bank.gamma_raw.grad is not None
    assert torch.isfinite(branch.bank.gamma_raw.grad).all()
    assert branch.bank.gamma_raw.grad.abs().item() > 0.0


def test_the_bank_reindexes_a_harmonic_series_onto_one_column():
    """The claim the whole branch rests on: a voice at f0 lights up the column
    at ``fc = f0`` across *every* order row, so a harmonic series is an axis of
    the input rather than a pattern to be assembled."""

    sample_rate, f0 = 44100, 200.0
    t = torch.arange(int(sample_rate * 0.5)) / sample_rate
    wave = sum(torch.sin(2 * torch.pi * f0 * h * t) / h for h in range(1, 11))
    branch = UnivHDDiscriminator(sample_rate=sample_rate, n_fft=8192, hop_length=512)
    energy = branch.bank(branch.spectrogram(wave[None, None])).mean(-1)[0]
    # Sum over orders, and the f0 column has to win.
    column = energy.sum(0)
    best = int(column.argmax())
    assert branch.bank.centers[best].item() == pytest.approx(f0, rel=0.03)


# --------------------------------------------------------------------------
# the branch
# --------------------------------------------------------------------------


def test_the_branch_returns_what_the_loop_expects():
    branch = UnivHDDiscriminator().eval()
    score, fmap = branch(torch.randn(2, 1, 17640))
    assert score.ndim == 2 and score.shape[0] == 2
    # HCB + one per MDC + the score map, which is what the paper takes feature
    # matching from.
    assert len(fmap) == 5
    assert all(torch.isfinite(f).all() for f in fmap)


def test_only_the_frequency_axis_is_ever_decimated():
    """The time axis stays at the STFT's frame rate all the way to the score.
    Striding it is the trap that costs the 4096-point resolution branch its
    detection of the frame-rate defect, and it would be an easy thing to add
    here while "matching" the paper's 5x5 kernels."""

    branch = UnivHDDiscriminator().eval()
    audio = torch.randn(1, 1, 17640)
    frames = branch.spectrogram(audio).shape[-1]
    score, fmap = branch(audio)
    assert score.shape[-1] == frames
    assert [f.shape[-1] for f in fmap] == [frames] * len(fmap)
    rows = [f.shape[-2] for f in fmap]
    assert rows == sorted(rows, reverse=True) and rows[-1] == 1


def test_the_branch_is_cheap_enough_to_add_rather_than_trade():
    """0.31M in the paper.  This is the number that decides the branch is
    additive: a period branch is ~11M, so UnivHD is under 3% of one."""

    params = sum(p.numel() for p in UnivHDDiscriminator().parameters())
    assert 0.25e6 < params < 0.40e6


@pytest.mark.parametrize("sample_rate", [32000, 44100])
def test_the_branch_builds_and_runs_at_both_shipped_rates(sample_rate):
    branch = UnivHDDiscriminator(sample_rate=sample_rate).eval()
    score, _ = branch(torch.randn(1, 1, int(sample_rate * 0.4)))
    assert torch.isfinite(score).all()


def test_a_bank_that_collapses_under_the_stride_schedule_is_refused():
    with pytest.raises(ValueError, match="collapses"):
        UnivHDDiscriminator(bins_per_octave=1, f_min=1500.0)


# --------------------------------------------------------------------------
# the flag
# --------------------------------------------------------------------------


def _build(model):
    """Mirrors what ``_build_d_model`` reads on the mpd_msd path."""

    def setting(name, default=None):
        value = getattr(model, name, None)
        return default if value is None else value

    return MPD_MSD_Combined(
        model.use_spectral_norm,
        version=str(getattr(model, "d_version", None) or "v3"),
        sample_rate=int(getattr(model, "sample_rate", 44100)),
        use_univhd=bool(setting("d_use_univhd", False)),
    )


@pytest.mark.parametrize("path", CONFIGS)
def test_the_flag_is_written_out_and_is_what_gets_built(path):
    """Written out explicitly rather than left absent, unlike the ``d_use_*``
    switches that default to on: an off-by-default knob that is not in the file
    is a knob nobody finds.  The value itself is the experiment's business --
    what is pinned is that the config's answer is the one the builder gives."""

    model = json.loads(path.read_text())["model"]
    assert isinstance(model["d_use_univhd"], bool)
    built = any(
        isinstance(d, UnivHDDiscriminator)
        for d in _build(types.SimpleNamespace(**model)).discriminators
    )
    assert built is model["d_use_univhd"]


def test_the_default_is_off_when_nothing_says_otherwise():
    """The code-side default, independent of what any config happens to say."""

    assert not any(
        isinstance(d, UnivHDDiscriminator)
        for d in MPD_MSD_Combined(False, version="v3").discriminators
    )


def test_the_flag_appends_and_removes_nothing():
    """Additive is the paper's own configuration -- MS-STFT *and* UnivHD -- and
    it is also what keeps the flag from being a quality regression by omission.
    It appends, so every other branch keeps its index."""

    model = json.loads(CONFIGS[-1].read_text())["model"]
    off = _build(types.SimpleNamespace(**{**model, "d_use_univhd": False}))
    on = _build(types.SimpleNamespace(**{**model, "d_use_univhd": True}))
    assert [type(d) for d in on.discriminators][:-1] == [
        type(d) for d in off.discriminators
    ]
    assert isinstance(on.discriminators[-1], UnivHDDiscriminator)


def test_the_flag_carries_the_sample_rate_into_the_bank():
    """The one branch here that is not rate-agnostic.  A default 44100 leaking
    into a 32 kHz run would put the top harmonic order 38% past Nyquist and
    fold it back, which no shape check would catch."""

    built = MPD_MSD_Combined(False, version="v3", sample_rate=32000, use_univhd=True)
    branch = built.discriminators[-1]
    assert branch.bank.sample_rate == 32000
    assert branch.bank.f_max == pytest.approx(32000 / 20.0)


def test_the_trainer_s_call_site_matches_the_constructor():
    """``_build`` above is a *mirror* of ``_build_d_model``, and ``train.py``
    cannot be imported without a run spec, so nothing else checks that the
    trainer passes names the constructor has.  A typo there is a TypeError at
    step 0 of a real run and nowhere earlier."""

    import ast
    import inspect

    signature = set(inspect.signature(MPD_MSD_Combined.__init__).parameters)
    tree = ast.parse((ROOT / "rvc" / "train" / "train.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "MPD_MSD_Combined"
    ]
    assert len(calls) == 1
    keywords = [kw.arg for kw in calls[0].keywords]
    assert not [kw for kw in keywords if kw not in signature]
    assert "use_univhd" in keywords and "sample_rate" in keywords


def test_the_whole_discriminator_still_runs_with_the_branch_on():
    model = MPD_MSD_Combined(False, version="v3", use_univhd=True).eval()
    real, fake = torch.randn(1, 1, 8192), torch.randn(1, 1, 8192)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(real, fake)
    assert len(y_d_rs) == len(y_d_gs) == len(fmap_rs) == len(fmap_gs) == 10
    assert all(torch.isfinite(s).all() for s in y_d_rs + y_d_gs)
