"""Is an inharmonic line a *fold*, and which stage boundary made it?

A harmonic sits at ``j*f0`` and rises when the note rises.  A fold sits at
``R - j*f0`` for the rate ``R`` of the stage that made it, so it *descends*
when the note rises.  That sign is the whole diagnosis, and it is what this
probe reads -- it does not look for lines, it asks whether energy is where a
fold would have to be, frame by frame, at the f0 that frame actually has.

Why a spectrogram cannot settle it: a source with 32 partials puts real energy
across the band, so a fold can stop being *visible* while losing nothing.  What
is measured here is the prediction against a control the same distance from the
harmonic grid on the *other* side of it -- same frame, same band, same skirt of
the same partials, and no fold can live there.  Louder harmonics move both.

    python tools/probe_mirror_fold.py render.wav
    python tools/probe_mirror_fold.py a.wav b.wav --band 3000 5000
    python tools/probe_mirror_fold.py render.wav --rate 8000 --rate 32000

``--rate`` is the *stage rate*, not the mirror: a stage running at 8000 Hz
mirrors around 4000.  At ``[5, 4, 4, 4]`` and 32 kHz the trunk's stages read
100 / 500 / 2000 / 8000 Hz and the last block runs at 32000, so 8000 is the
only boundary anywhere near 3-5 kHz, and 32000 is the one above it.

Blind frames.  ``R - j*f0`` is congruent to ``R`` modulo f0, so for a given f0
every predicted fold sits at one fixed offset from the harmonic grid.  When f0
nearly divides R that offset is ~0, the prediction lands on the harmonics, and
no measurement can separate the two.  Those frames are counted and skipped
rather than averaged in -- an unblind run needs a note that does not divide the
rate, which most singing does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path, help="rendered audio")
    parser.add_argument(
        "--rate",
        type=float,
        action="append",
        dest="rates",
        help="stage rate in Hz; mirrors around rate/2. Repeatable (default 8000).",
    )
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=(3000.0, 5000.0),
        metavar=("LOW", "HIGH"),
        help="band to report, in Hz (default 3000 5000)",
    )
    parser.add_argument("--f0-min", type=float, default=70.0)
    parser.add_argument("--f0-max", type=float, default=700.0)
    parser.add_argument(
        "--n-fft",
        type=int,
        default=4096,
        help="7.8 Hz per bin at 32 kHz; must resolve the harmonic spacing",
    )
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument(
        "--min-offset",
        type=float,
        default=0.25,
        help="how far, in units of f0, a prediction must sit from the harmonic "
        "grid for the frame to count (default 0.25)",
    )
    parser.add_argument("--csv", type=Path, help="write the per-frame table here")
    return parser.parse_args(argv)


def search_window(target, f0, bin_hz):
    """How far around ``target`` to look, in Hz.

    Proportional to the target, because what is uncertain is f0 *relatively*:
    0.3% of error is a third of a bin on the fundamental and four bins on the
    twentieth partial, so a fixed window measures the estimator on the harmonics
    it matters most for.  Floored at two bins and capped well short of the
    neighbouring harmonic, or a window would reach the line it is meant to be
    compared against.
    """

    return float(np.clip(0.004 * target, 2.0 * bin_hz, 0.3 * f0))


def peak_near(magnitude, freqs, target, f0, bin_hz):
    """The largest bin near ``target``, in linear magnitude, or ``None``."""

    tolerance = search_window(target, f0, bin_hz)
    low = np.searchsorted(freqs, target - tolerance)
    high = np.searchsorted(freqs, target + tolerance)
    if high <= low:
        return None
    return float(magnitude[low:high].max())


def refine_f0(magnitude, freqs, f0, bin_hz, span=0.01, steps=41):
    """f0 that best lines the harmonic grid up with this frame's spectrum.

    pyin is estimating a pitch; this probe needs a *grid*, and the two differ by
    exactly the error that ruins a high partial.  A 1% search maximising the
    energy the grid lands on costs one pass over a few dozen bins and removes
    the failure mode entirely -- without it the harmonic reference reads a few
    Hz off the line at ``j`` around 20 and reports a floor above its own peaks.
    """

    top = min(float(freqs[-1]), 12000.0)
    best, best_score = f0, -1.0
    for candidate in np.linspace(f0 * (1.0 - span), f0 * (1.0 + span), steps):
        orders = np.arange(2, int(top / candidate) + 1)
        if orders.size == 0:
            continue
        score = 0.0
        for order in orders:
            value = peak_near(magnitude, freqs, order * candidate, candidate, bin_hz)
            if value is not None:
                score += value
        if score > best_score:
            best, best_score = float(candidate), score
    return best


#: Offsets from the harmonic grid, in units of f0, used as the null.  The fold
#: prediction is one offset among these; if it is not an outlier against them,
#: there is nothing there.  Kept away from 0 and 1, where every offset measures
#: the harmonic itself.
NULL_OFFSETS = np.linspace(0.12, 0.88, 33)


def offset_level(magnitude, freqs, f0, offset, band, bin_hz):
    """Median magnitude at ``j*f0 + offset*f0`` across the band, or ``None``.

    The whole probe is this function read at one offset against the same
    function read at every other offset.
    """

    low, high = band
    values = []
    for j in range(1, int(freqs[-1] / f0) + 1):
        target = (j + offset) * f0
        if not low <= target <= high:
            continue
        value = peak_near(magnitude, freqs, target, f0, bin_hz)
        if value is not None:
            values.append(value)
    return float(np.median(values)) if values else None


def signature(magnitude, freqs, f0, offset, sub_band, bin_hz):
    """The offset's level, the null it sits in, and its rank among it.

    ``sub_band`` matters as much as the offset.  A fold of a stage at ``R``
    lands *below* ``R/2`` and nowhere else, so measuring it over a band that
    reaches above the mirror mixes in frequencies where no fold can be and
    drags the median onto them -- which reported a synthetic file with an
    injected fold as clean until this was split.
    """

    low, high = sub_band
    if high <= low:
        return None
    level = offset_level(magnitude, freqs, f0, offset, (low, high), bin_hz)
    if level is None:
        return None
    null = [
        value
        for phi in NULL_OFFSETS
        if abs(phi - offset) > 0.06
        for value in [offset_level(magnitude, freqs, f0, phi, (low, high), bin_hz)]
        if value is not None
    ]
    if len(null) < 8:
        return None
    harmonic = offset_level(magnitude, freqs, f0, 0.0, (low, high), bin_hz)
    rank = float(np.mean([level > value for value in null]))
    return level, float(np.median(null)), rank, harmonic


def frame_report(magnitude, freqs, f0, rate, band, min_offset, bin_hz):
    """One frame, both mechanisms that live at a stage boundary.

    ``fold``: the stage's own nonlinearity mirrors ``j*f0`` above ``R/2`` down
    to ``R - j*f0``, below the mirror.  Nothing filters it -- it is created
    after the stage's input filter.

    ``image``: zero-stuffing copies the spectrum to ``R +- f``, so content just
    under the mirror reappears just above it, and only the upsampler's
    interpolation filter attenuates it.

    Both are congruent to ``R`` modulo f0, so one offset tests both; which side
    of the mirror the energy is on says which mechanism made it.

    Returns ``(fold, image)``, each ``None`` or a ``signature`` tuple, or
    ``"blind"``.
    """

    if not np.isfinite(f0) or f0 <= 0:
        return None

    f0 = refine_f0(magnitude, freqs, f0, bin_hz)
    offset = (rate % f0) / f0
    if min(offset, 1.0 - offset) < min_offset:
        return "blind"

    low, high = band
    mirror = rate / 2.0
    fold = signature(magnitude, freqs, f0, offset, (low, min(high, mirror)), bin_hz)
    image = signature(
        magnitude, freqs, f0, offset, (max(low, mirror), min(high, rate)), bin_hz
    )
    if fold is None and image is None:
        return None
    return fold, image


def probe(path, rates, band, args):
    audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=args.f0_min,
        fmax=args.f0_max,
        sr=sample_rate,
        frame_length=args.n_fft,
        hop_length=args.hop,
    )
    spectrum = np.abs(
        librosa.stft(
            audio, n_fft=args.n_fft, hop_length=args.hop, window="hann", center=True
        )
    )
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=args.n_fft)
    bin_hz = sample_rate / args.n_fft

    frames = min(spectrum.shape[1], len(f0))
    results = {}
    for rate in rates:
        if rate / 2.0 <= band[0]:
            print(
                f"  note: a stage at {rate:g} Hz mirrors around {rate / 2:g} Hz, "
                f"below the reported band -- nothing of it can land there.",
                file=sys.stderr,
            )
        rows, blind, unvoiced = [], 0, 0
        for index in range(frames):
            if not voiced[index]:
                unvoiced += 1
                continue
            report = frame_report(
                spectrum[:, index],
                freqs,
                float(f0[index]),
                rate,
                band,
                args.min_offset,
                bin_hz,
            )
            if report is None:
                unvoiced += 1
            elif report == "blind":
                blind += 1
            else:
                rows.append((index, float(f0[index]), *report))
        results[rate] = (rows, blind, unvoiced)
    return results, sample_rate, bin_hz


def to_db(value):
    return 20.0 * np.log10(max(value, 1e-12))


def summarise(label, rows, column, sub_band):
    """One mechanism, over every frame that could measure it."""

    usable = [row[column] for row in rows if row[column] is not None]
    if not usable:
        return None
    level = np.array([entry[0] for entry in usable])
    null = np.array([entry[1] for entry in usable])
    rank = np.array([entry[2] for entry in usable])
    harmonic = np.array(
        [entry[3] if entry[3] is not None else np.nan for entry in usable], dtype=float
    )
    excess = to_db(float(np.median(level))) - to_db(float(np.median(null)))
    median_rank = float(np.median(rank))
    # The rank carries the verdict, not the level: it asks whether the
    # prediction is unusual *within its own frame*, so a render whose whole
    # between-harmonic floor is high cannot pass on loudness alone.  0.5 is
    # "one offset among many", which is what real audio reads.
    if median_rank >= 0.90 and excess >= 2.0:
        verdict = f"{label} present"
    elif median_rank <= 0.75:
        verdict = f"no {label} here -- an ordinary offset"
    else:
        verdict = "ambiguous -- needs more voiced material, or notes further from dividing the rate"
    floor = to_db(float(np.median(null))) - to_db(float(np.nanmedian(harmonic)))
    return (
        f"    {label:<5} {sub_band[0]:.0f}-{sub_band[1]:.0f} Hz: "
        f"{excess:+.1f} dB over the null, rank {median_rank:.2f}, "
        f"floor {floor:+.1f} dB under the harmonics  [{len(usable)} frames]\n"
        f"          -> {verdict}"
    )


def main(argv=None):
    args = parse_args(argv)
    rates = args.rates or [8000.0]
    band = (float(args.band[0]), float(args.band[1]))

    csv_rows = []
    for path in args.files:
        if not path.exists():
            print(f"{path}: not found", file=sys.stderr)
            continue
        results, sample_rate, bin_hz = probe(path, rates, band, args)
        print(f"\n{path}  ({sample_rate} Hz, {bin_hz:.1f} Hz per bin)")
        print(f"  band {band[0]:.0f}-{band[1]:.0f} Hz")
        for rate, (rows, blind, unvoiced) in results.items():
            mirror = rate / 2.0
            print(
                f"  stage {rate:g} Hz, mirror {mirror:g} Hz  "
                f"[{len(rows)} frames, {blind} blind, {unvoiced} unvoiced]"
            )
            if not rows:
                print("    nothing usable")
                continue
            for label, column, sub in (
                ("fold", 2, (band[0], min(band[1], mirror))),
                ("image", 3, (max(band[0], mirror), min(band[1], rate))),
            ):
                if sub[1] <= sub[0]:
                    continue
                line = summarise(label, rows, column, sub)
                print(line if line else f"    {label:<5}: no usable frames")
            for index, f0_value, fold, image in rows:
                for label, entry in (("fold", fold), ("image", image)):
                    if entry is None:
                        continue
                    csv_rows.append(
                        (
                            str(path),
                            rate,
                            label,
                            index,
                            f"{f0_value:.2f}",
                            f"{to_db(entry[0]):.2f}",
                            f"{to_db(entry[1]):.2f}",
                            f"{entry[2]:.3f}",
                        )
                    )

    if args.csv and csv_rows:
        with open(args.csv, "w", encoding="utf-8") as handle:
            handle.write("file,rate,mechanism,frame,f0,level_db,null_db,rank\n")
            for row in csv_rows:
                handle.write(",".join(str(value) for value in row) + "\n")
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
