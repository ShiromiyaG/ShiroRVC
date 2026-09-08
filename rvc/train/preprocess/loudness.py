"""ITU-R BS.1770-4 integrated loudness, for level-matching dataset slices.

Why this exists rather than the RMS normaliser it replaces.  ``post_rms``
measures the RMS of the samples above a *fixed* -40 dBFS threshold, and a fixed
threshold applied to a relative measurement is scale-dependent: on a slice 10 dB
below the reference the measurement is already 30 dB out, and once the slice
falls far enough that no sample clears the gate, the code takes the ``gain =
1.0`` branch and normalises nothing at all, silently.  Measured on six 1 s
slices spread over 21.5 dB, which is an ordinary spread for mic distance and
phrase dynamics:

    input gain    post_rms error    this
      0 dB            0.00 dB      0.00 dB
    -10 dB          -30.50          0.00
    -20 dB          -40.50          0.00
    -30 dB            silent no-op  (both give up: below the -70 LUFS gate)

The second difference is perceptual weighting.  RMS counts a rumble and a
sibilant equally; BS.1770's K-weighting (a +4 dB shelf above ~1.5 kHz and a
~38 Hz high-pass) is the standard model of how they actually compare.  After
normalisation the residual spread of measured loudness across those slices was
8.43 dB untouched, 2.13 dB with ``post_peak``, 2.78 dB with ``post_rms``.

Evaluating a loudness normaliser by measuring loudness is circular, so the
number that matters above is how far the others sit from zero, not that this one
reaches it.

Two modes are built on this.  ``pre_loudness`` is the one that ships: one gain
per *source recording*, computed by pooling its slices' block powers and gating
once, so recordings match each other while a whisper and a belted line inside
one recording keep their relative levels.

The name says what it is *for*, not when it runs -- it runs after slicing, like
every mode here.  Measuring the recording before slicing, which is what the name
literally describes, was tried and is worse: the gaps between phrases pull the
gated mean down by ~0.7 dB even when they are digitally silent (blocks
straddling an onset are half-filled and still clear the relative gate), by ~1 dB
with room tone on a synthetic, and by 4.7 dB on real singing with 3 s gaps and a
-35 dBFS floor.  Worse, the size of that error
depends on how much a recording pauses and how loud its room is -- which is
exactly the between-recording variation the mode exists to remove.  Pooling the
*kept* slices measures the speech and nothing else.

``post_loudness`` levels every *slice* to the same loudness.  That is the
contract every mode here had before, and it flattens the dynamics between
phrases; it stays reachable for a config that names it but is no longer
offered.  ``block_powers`` and ``loudness_from_blocks`` are split apart so
``pre_loudness`` is possible at all: the gates are a property of the whole
measurement and cannot be recovered from per-slice loudness values.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

#: Block length and hop of the sliding loudness measurement (BS.1770-4 §2.1):
#: 400 ms with 75% overlap.
BLOCK_SECONDS = 0.400
OVERLAP = 0.75

#: The two gates.  The absolute one drops digital silence; the relative one --
#: 10 LU below the ungated level -- is what stops the quiet parts of a
#: recording from dragging the measurement down, and is the reason a loudness
#: match survives room tone that an RMS average does not.
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0

#: The offset in ``-0.691 + 10 log10(power)``, which calibrates a 0 dBFS
#: 1 kHz sine to 0 LUFS after K-weighting.
LOUDNESS_OFFSET = -0.691


def k_weighting(sample_rate: int):
    """The two K-weighting biquads at ``sample_rate``.

    The spec tabulates coefficients at 48 kHz only; these come from the same
    analogue prototypes, so they are correct at any rate rather than being the
    48 kHz numbers reused.
    """

    # Stage 1: high shelf, +4 dB above ~1.5 kHz.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    denom = 1.0 + k / q + k * k
    shelf_b = np.array([
        (vh + vb * k / q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / q + k * k) / denom,
    ])
    shelf_a = np.array([1.0, 2.0 * (k * k - 1.0) / denom, (1.0 - k / q + k * k) / denom])

    # Stage 2: high pass at ~38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    denom = 1.0 + k / q + k * k
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([1.0, 2.0 * (k * k - 1.0) / denom, (1.0 - k / q + k * k) / denom])
    return (shelf_b, shelf_a), (hp_b, hp_a)


def block_powers(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mean-square power of each K-weighted 400 ms block.

    Exposed separately from :func:`integrated_lufs` because the gates are a
    property of the *whole* measurement, not of a block: pooling the blocks of
    several files and gating once is how one gain gets computed for a whole
    source recording, and it cannot be done from per-file loudness values.
    Ten floats per second, so pooling costs nothing.
    """

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=tuple(range(1, audio.ndim)))

    shelf, high_pass = k_weighting(sample_rate)
    weighted = signal.lfilter(*shelf, audio)
    weighted = signal.lfilter(*high_pass, weighted)

    block = int(round(BLOCK_SECONDS * sample_rate))
    hop = max(1, int(round(block * (1.0 - OVERLAP))))
    if block <= 0 or len(weighted) < block:
        # Shorter than one block: measure what there is rather than refusing.
        # A slicer minimum below 400 ms is unusual but not forbidden, and the
        # alternative is a slice that silently skips normalisation.
        if len(weighted) == 0:
            return np.empty(0, dtype=np.float64)
        power = np.array([np.mean(weighted ** 2)])
    else:
        count = 1 + (len(weighted) - block) // hop
        index = np.arange(block)[None, :] + hop * np.arange(count)[:, None]
        power = (weighted[index] ** 2).mean(axis=1)

    return power


def loudness_from_blocks(power: np.ndarray) -> float:
    """Apply BS.1770's two gates to pooled block powers, in LUFS."""

    power = np.asarray(power, dtype=np.float64)
    if power.size == 0:
        return -math.inf
    with np.errstate(divide="ignore"):
        block_lufs = LOUDNESS_OFFSET + 10.0 * np.log10(power + 1e-30)

    keep = block_lufs > ABSOLUTE_GATE_LUFS
    if not keep.any():
        return -math.inf
    ungated = LOUDNESS_OFFSET + 10.0 * np.log10(power[keep].mean() + 1e-30)
    keep &= block_lufs > (ungated + RELATIVE_GATE_LU)
    if not keep.any():
        return -math.inf
    return float(LOUDNESS_OFFSET + 10.0 * np.log10(power[keep].mean() + 1e-30))


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness in LUFS, or ``-inf`` for silence.

    ``-inf`` rather than a fallback value: "this has no measurable loudness"
    and "this is at 0 LUFS" must not be the same answer, and the caller decides
    what to do about it.  ``post_rms``'s ``gain = 1.0`` branch is what that
    decision looks like when it is made silently.
    """

    return loudness_from_blocks(block_powers(audio, sample_rate))


def loudness_gain(audio: np.ndarray, sample_rate: int, target_lufs: float) -> float:
    """The linear gain that puts ``audio`` at ``target_lufs``; 1.0 if unmeasurable."""

    measured = integrated_lufs(audio, sample_rate)
    if not math.isfinite(measured):
        return 1.0
    return float(10.0 ** ((target_lufs - measured) / 20.0))


#: Peak ceiling for a levelled slice, in dBFS.
#:
#: -1.0 dBFS is the broadcast convention, and it is margin for two things that
#: do not happen here: inter-sample peaks that appear when a signal is
#: reconstructed or resampled, and lossy codecs that overshoot on encode.  The
#: files this writes are float32 WAV read straight by the trainer.  What the
#: margin is still for is the FLAC path, which clips to [-1, 1] before writing
#: PCM_24 -- a ceiling at exactly 0 would clip on rounding -- so this keeps
#: enough for that and hands the rest back as level.
CEILING_DB = -0.3


def apply_gain_with_ceiling(audio: np.ndarray, gain: float,
                            ceiling_db: float = CEILING_DB):
    """Scale by ``gain``, then pull back if the peak would exceed ``ceiling_db``.

    Returns ``(audio, limited_by_db)``; ``limited_by_db`` is 0.0 when the
    target was reached and positive when the ceiling cost loudness, which is
    the number the dry run reports so a dataset that cannot reach its target is
    a message rather than a surprise.
    """

    ceiling = 10.0 ** (ceiling_db / 20.0)
    scaled = np.asarray(audio, dtype=np.float64) * gain
    peak = float(np.abs(scaled).max()) if scaled.size else 0.0
    if peak > ceiling and peak > 0.0:
        scaled = scaled * (ceiling / peak)
        return scaled, 20.0 * math.log10(peak / ceiling)
    return scaled, 0.0


#: Half-length of the limiter's gain window, in milliseconds.  Both the attack
#: and the release, since the window is symmetric.  1.5 ms is chosen against
#: what actually overshoots here: at -18 LUFS on a 116 h speech set only 0.034%
#: of samples clear a -1 dBFS ceiling, and they arrive as isolated transients
#: rather than sustained passages, so a short symmetric window ducks the
#: plosive and leaves the syllable around it alone.  A broadcast-style long
#: release would pull down the whole word to hold back one click.
LIMITER_WINDOW_MS = 1.5


def limit_peaks(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_db: float = CEILING_DB,
    window_ms: float = LIMITER_WINDOW_MS,
) -> np.ndarray:
    """Hold ``audio`` under ``ceiling_db`` by ducking peaks, not by rescaling.

    The alternative -- :func:`apply_gain_with_ceiling` -- divides the *whole*
    signal by its own worst peak, so one transient decides the level of
    everything around it.  Across a dataset that compounds: matching every
    recording to the one with the worst crest factor cost 22 dB on a real
    116 h set.  Ducking 0.03% of the samples instead costs nothing measurable
    and lets every recording sit at its target.

    The envelope is a sliding minimum of the per-sample required gain, then a
    Hann smoothing of the same length.  That pairing is what makes the result
    provably under the ceiling rather than approximately under it: for an
    offset ``k`` inside the Hann support, the minimum taken at ``n - k`` spans
    a window that still contains ``n``, so every term being averaged is at most
    the gain sample ``n`` itself requires -- and a weighted mean of terms that
    are each ``<= required[n]``, with weights summing to one, cannot exceed it.

    A dtype note: the gain envelope is built in float64 and applied before the
    caller casts back, because a float32 ``ceiling / peak`` can round *up* by
    an ulp and put the result a hair over a ceiling the caller then asserts on.
    """

    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return audio

    ceiling = 10.0 ** (ceiling_db / 20.0)
    magnitude = np.abs(audio)
    peak = float(magnitude.max())
    if peak <= ceiling:
        return audio

    # Half-window in samples, forced odd so the window is centred; at least one
    # sample either side, or the "min window contains n" argument above fails.
    half = max(1, int(round(sample_rate * window_ms / 1000.0)))
    length = 2 * half + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        required = np.where(magnitude > ceiling, ceiling / magnitude, 1.0)

    # Sliding minimum.  The padding goes on ``required`` -- twice the half
    # window, so the envelope is defined for the ``half`` positions either side
    # of the signal that the smoothing then reads -- and never on the envelope
    # itself.  Padding the envelope was the first version of this and it breaks
    # the guarantee above at the edges: the smoothing there averages in ones
    # that no ``required`` sample justifies, and a full-scale sine came out at
    # -0.78 dBFS against a -1.0 ceiling.
    #
    # Ones rather than a reflection: a slice is a fragment of a longer
    # recording, and inventing a mirrored transient just outside it would duck
    # audio that never needed it.
    pad = np.ones(2 * half, dtype=np.float64)
    padded = np.concatenate((pad, required, pad))
    envelope = np.lib.stride_tricks.sliding_window_view(padded, length).min(axis=1)

    window = np.hanning(length + 2)[1:-1]
    window /= window.sum()
    smoothed = np.convolve(envelope, window, mode="valid")

    return audio * smoothed
