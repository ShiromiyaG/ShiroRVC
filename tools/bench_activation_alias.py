"""Where a decoder's inharmonic fold is created, and what actually removes it.

Needs no trained weights, no checkpoint and no GPU: what a pointwise
nonlinearity does to a band-limited signal is decided by the activation, the
bandwidth it acts on, and how many of them run in series.  Those are the only
three variables a fold has, which is why this harness measures exactly them.

Three tables:

``fidelity``  reproduces the table in ``rvc/lib/algorithm/resampling.py`` --
    the anti-aliasing designs against a reference built by running
    ``leaky_relu`` at 32x and band-limiting back down.  It exists to show the
    harness agrees with a result measured independently of it.

``headroom``  how much aliasing a *raw* activation makes as a function of the
    fraction of the band its input occupies.  This is the claim that a
    nonlinearity placed immediately after an upsampler is nearly free, and the
    same nonlinearity placed after the band has filled is not.

``cascade``   how aliasing grows with the number of nonlinearities in series.
    A ``leaky_relu`` composed with itself is *still* a ``leaky_relu`` -- the
    cascade collapses -- so each step here is a fixed random FIR followed by
    the activation, which is what a res-block actually is.  A linear filter
    cannot widen a spectrum, so every bit of band-filling in the chain comes
    from the activations, exactly as in the decoder.

The alias metric for the last two needs no reference and has no arguable split:
a tone at an FFT bin that does not divide the frame length puts its harmonics
on multiples of that bin and every *folded* product on a bin that is not one,
so the two separate by position alone.

Usage::

    python tools/bench_activation_alias.py
    python tools/bench_activation_alias.py --table cascade --depth 8
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.resampling import AntiAliasedActivation  # noqa: E402

LENGTH = 4096


def design(slope: float, factor: int, width: int, rolloff: float, beta: float = 6.0):
    return AntiAliasedActivation(
        leaky_relu_slope=slope,
        factor=factor,
        filter_width=width,
        rolloff=rolloff,
        filter_beta=beta,
    )


def designs(slope: float):
    return [
        ("plain leaky_relu", lambda x: F.leaky_relu(x, slope)),
        ("f2 w6  r0.90", design(slope, 2, 6, 0.90)),
        ("f2 w16 r0.99  <- ships", design(slope, 2, 16, 0.99)),
        ("f4 w16 r0.99", design(slope, 4, 16, 0.99)),
    ]


def tone(bin_index: int, length: int = LENGTH) -> torch.Tensor:
    t = np.arange(length) / length
    return torch.from_numpy(np.cos(2 * math.pi * bin_index * t)).view(1, 1, -1).float()


def alias_db(rendered: np.ndarray, bin_index: int) -> float:
    """Alias energy against harmonic energy, in dB.

    ``bin_index`` must not divide the frame length: then the harmonics sit on
    its multiples and anything folded does not, so no reference render is
    needed.

    The harmonic bins are excluded exactly, with no guard band.  A guard of one
    bin either side -- which this had first -- is not conservative, it hides
    results: a tone at bin 241 folds onto bin 240, one bin from its own
    fundamental, so the guard swallowed that fold whole and reported the tone
    as *less* aliased than one an octave below it.  There is nothing to guard
    against, since a tone at an integer bin leaks nothing.
    """

    length = rendered.size
    spectrum = np.abs(np.fft.rfft(rendered)) ** 2
    mask = np.ones(spectrum.size, dtype=bool)
    mask[0] = False
    for order in range(1, length // 2 // bin_index + 1):
        mask[order * bin_index] = False
    return 10 * math.log10(spectrum[mask].sum() / (spectrum[~mask].sum() + spectrum[0]))


def render(module, signal: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return module(signal).view(-1).double().numpy()


# ---------------------------------------------------------------- fidelity


def harmonic_series(length: int, f0_bin: int, partials: int, seed: int):
    rng = np.random.default_rng(seed)
    orders = np.arange(1, partials + 1)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=partials)

    def draw(samples: int) -> np.ndarray:
        t = np.arange(samples, dtype=np.float64) / samples
        angles = 2.0 * math.pi * np.outer(orders * f0_bin, t) + phases[:, None]
        return ((1.0 / orders)[:, None] * np.cos(angles)).sum(axis=0)

    return draw, orders * f0_bin


def reference(draw, length: int, oversample: int, slope: float) -> np.ndarray:
    """``leaky_relu`` at ``oversample`` x, band-limited and decimated exactly.

    Truncating the spectrum is an exact band-limit because the signal is
    periodic in the frame; ``length / total`` is the decimation gain that
    ``irfft``'s own ``1/n`` does not carry.
    """

    total = length * oversample
    wide = draw(total)
    wide = np.where(wide >= 0.0, wide, slope * wide)
    return np.fft.irfft(np.fft.rfft(wide)[: length // 2 + 1], n=length) * (
        length / total
    )


def error_split(candidate, truth, partial_bins):
    """``(alias, passband, total)`` as dB of the reference's own energy.

    ``passband`` is the error at the bins carrying the input's own partials,
    which those partials dominate -- so error there is the design's linear
    response, a lowpass or a tilt.  ``alias`` is everything else.  The two are
    not interchangeable: a fixed lowpass is something the trunk can learn
    around and an inharmonic fold is not.
    """

    length = truth.size
    bins = np.arange(length // 2 + 1)
    got, want = np.fft.rfft(candidate), np.fft.rfft(truth)
    # rfft holds one of each conjugate pair, so every bin but DC and Nyquist
    # stands for twice its energy.
    weight = np.full(bins.size, 2.0)
    weight[0] = weight[-1] = 1.0
    error = np.abs(got - want) ** 2 * weight
    total = float((np.abs(want) ** 2 * weight).sum())
    mask = np.zeros(bins.size, dtype=bool)
    mask[partial_bins[partial_bins < bins.size]] = True

    def db(value: float) -> float:
        return 10.0 * math.log10(max(value, 1e-300) / total)

    alias = db(float(error[~mask].sum()))
    passband = db(float(error[mask].sum()))
    return alias, passband, db(float(error.sum()))


def table_fidelity(args) -> None:
    partials = int(args.fill * LENGTH / 2) // args.f0_bin
    draw, partial_bins = harmonic_series(LENGTH, args.f0_bin, partials, args.seed)
    base = draw(LENGTH)
    scale = 1.0 / np.abs(base).max()
    truth = reference(lambda n: draw(n) * scale, LENGTH, args.oversample, args.slope)
    signal = torch.from_numpy(base * scale).view(1, 1, -1).float()

    occupancy = 100 * partials * args.f0_bin / (LENGTH / 2)
    print(
        f"{partials} partials at {args.f0_bin} bins, filling {occupancy:.0f}% of "
        f"the band; reference at {args.oversample}x.\n"
    )
    print(f"{'design':<24}{'alias':>9}{'passband':>11}{'total':>9}")
    for label, module in designs(args.slope):
        alias, passband, total = error_split(
            render(module, signal), truth, partial_bins
        )
        print(f"{label:<24}{alias:>9.1f}{passband:>11.1f}{total:>9.1f}")
    print("\ndB relative to the reference's total energy; lower is better.")


# ---------------------------------------------------------------- headroom

# Bins that do not divide 4096, spanning the whole range of occupancies a
# stage can present.  The decoder's upsample factors are 5 and 4, so 1/5 and
# 1/4 are what a nonlinearity sees *immediately* after an upsampler and 1/2
# and 1/1 are what one sees after the res-block has filled the band.
TONE_BINS = (31, 61, 121, 241, 401, 449, 701)


def table_headroom(args) -> None:
    print(
        "Alias made by one activation against the fraction of the band its\n"
        "input occupies.  A stage's upsampler leaves its output band-limited\n"
        "to 1/factor of the new Nyquist, so the 1/4 column is what a\n"
        "nonlinearity placed immediately after a x4 upsample sees, and the\n"
        "right-hand columns are what the same nonlinearity sees once the\n"
        "res-block has filled the band.\n"
    )
    header = "".join(f"{'1/' + str(LENGTH // 2 // b):>9}" for b in TONE_BINS)
    print(f"{'design':<24}{header}")
    for label, module in designs(args.slope):
        row = "".join(
            f"{alias_db(render(module, tone(b)), b):>9.1f}" for b in TONE_BINS
        )
        print(f"{label:<24}{row}")
    print(
        "\nalias energy in dB against harmonic energy; lower is better.\n"
        "columns are the occupied fraction of the band."
    )


# ----------------------------------------------------------------- cascade


def mixer(length: int, seed: int) -> torch.Tensor:
    """A fixed random *allpass*, standing in for a res-block's convolution.

    Something has to sit between the activations: ``leaky_relu`` composed with
    itself is just a ``leaky_relu`` of squared slope, so an unmixed cascade
    collapses and creates nothing new.  A real res-block mixes neighbours
    between activations, moving where the signal changes sign, and that is what
    makes the second nonlinearity see a different waveform than the first.

    Allpass rather than a random FIR, which is what this had first and what
    made the measurement meaningless.  A random FIR reshapes the magnitude
    spectrum, and applying it ``depth`` times reshapes it ``depth`` times over,
    so the alias-to-harmonic ratio moved with depth for reasons that had
    nothing to do with aliasing -- visible as the single-activation control
    degrading as fast as the full cascade, which a linear filter cannot cause.
    A unit-magnitude random phase leaves every bin's magnitude untouched, so
    the whole change with depth is the activations'.

    The frame holds an integer number of the tone's cycles, so circular
    convolution is exact here and there is no edge to worry about.
    """

    generator = torch.Generator().manual_seed(seed)
    phase = torch.rand(length // 2 + 1, generator=generator) * 2.0 * math.pi
    # Real output needs a real DC and Nyquist bin.
    phase[0] = 0.0
    phase[-1] = 0.0
    return torch.polar(torch.ones_like(phase), phase)


def chain(module, signal, depth: int, filt, linear_after: bool) -> np.ndarray:
    x = signal
    for step in range(depth):
        x = torch.fft.irfft(torch.fft.rfft(x) * filt, n=x.shape[-1])
        # leaky_relu is homogeneous, so renormalising between steps cannot move
        # any of these numbers -- it only keeps the chain on scale.
        x = x / x.std().clamp_min(1e-8)
        if step == 0 or not linear_after:
            with torch.no_grad():
                x = module(x)
    return x.view(-1).double().numpy()


def table_cascade(args) -> None:
    filt = mixer(LENGTH, args.seed)
    signal = tone(args.tone)
    print(
        f"A tone at bin {args.tone} ({100 * args.tone / (LENGTH / 2):.0f}% of "
        f"Nyquist) through\nrepetitions of [random allpass -> activation].\n"
    )
    header = "".join(f"{d:>9}" for d in range(1, args.depth + 1))
    print(f"{'arrangement':<24}{header}")
    rows = [(label, module, False) for label, module in designs(args.slope)]
    rows.append(("one act, then linear", designs(args.slope)[0][1], True))
    for label, module, linear_after in rows:
        row = "".join(
            f"{alias_db(chain(module, signal, d, filt, linear_after), args.tone):>9.1f}"
            for d in range(1, args.depth + 1)
        )
        print(f"{label:<24}{row}")
    print(
        "\nalias energy in dB against harmonic energy; lower is better.\n"
        "columns are the number of [allpass -> activation] repetitions."
    )


# ---------------------------------------------------------------- coverage


def table_coverage(args) -> None:
    """Does covering *some* of a stage's activations buy anything?

    The decision the config actually offers.  ``"adain"`` anti-aliases 6 of the
    24 activations a stage runs, ``"half"`` 15 of them, ``"full"`` all 24 --
    and the three cost very different amounts, so whether the benefit is
    proportional to the coverage or only arrives at the end is the whole
    question.

    ``cascade`` answers it only by inference: six raw activations in series
    alias about as much as one, so one raw site left uncovered should hold the
    floor up on its own.  That is a deduction, and this measures it instead.
    """

    filt = mixer(LENGTH, args.seed)
    signal = tone(args.tone)
    protected = design(args.slope, 2, 16, 0.99)
    raw = designs(args.slope)[0][1]
    depth = args.depth

    patterns = [("none of them", ())]
    patterns.append(("the first", (0,)))
    patterns.append(("every other, from 1st", tuple(range(0, depth, 2))))
    patterns.append(("every other, from 2nd", tuple(range(1, depth, 2))))
    patterns.append(("all but the last", tuple(range(depth - 1))))
    patterns.append(("all but the first", tuple(range(1, depth))))
    patterns.append(("all of them", tuple(range(depth))))

    print(
        f"A tone at bin {args.tone} through {depth} repetitions of\n"
        "[random allpass -> activation], with only some of the activations\n"
        "anti-aliased (f2 w16 r0.99) and the rest left raw.\n"
    )
    print(f"{'anti-aliased':<24}{'covered':>9}{'alias':>9}")
    for label, covered in patterns:
        x = signal
        for step in range(depth):
            x = torch.fft.irfft(torch.fft.rfft(x) * filt, n=x.shape[-1])
            x = x / x.std().clamp_min(1e-8)
            with torch.no_grad():
                x = (protected if step in covered else raw)(x)
        value = alias_db(x.view(-1).double().numpy(), args.tone)
        print(f"{label:<24}{f'{len(covered)}/{depth}':>9}{value:>9.1f}")
    print("\nalias energy in dB against harmonic energy; lower is better.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        choices=("fidelity", "headroom", "cascade", "coverage", "all"),
        default="all",
    )
    parser.add_argument("--slope", type=float, default=0.2)
    parser.add_argument("--fill", type=float, default=0.9)
    parser.add_argument("--f0-bin", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--tone", type=int, default=61)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tables = {
        "fidelity": table_fidelity,
        "headroom": table_headroom,
        "cascade": table_cascade,
        "coverage": table_coverage,
    }
    for name in tables if args.table == "all" else (args.table,):
        print(f"=== {name}\n")
        tables[name](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
