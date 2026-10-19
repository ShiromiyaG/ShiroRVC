"""Integrated loudness (ITU-R BS.1770-4) and a look-ahead peak limiter, for
bringing a rendered voice to a set loudness without its peaks clipping."""

import math

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

#: BS.1770's gating: 400 ms blocks every 100 ms, an absolute gate at -70
#: LUFS and a relative one 10 LU under the absolutely gated loudness.
BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
#: The largest boost ``loudness_gain`` gives, so a near-silent render is not
#: blown up into its noise.
MAX_BOOST_DB = 24.0
#: The limiter holds its reduction this long around a peak and ramps it in
#: and out over the second: down before the peak, and never per cycle.
LIMIT_HOLD_SECONDS = 0.05
LIMIT_RAMP_SECONDS = 0.02


def k_weighting(sample_rate: int) -> np.ndarray:
    """BS.1770's K-weighting at ``sample_rate``, as second-order sections:
    its high shelf, then its high pass."""
    gain, q, frequency = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    k = math.tan(math.pi * frequency / sample_rate)
    high, low = 10.0 ** (gain / 20.0), (10.0 ** (gain / 20.0)) ** 0.499666774155
    a0 = 1.0 + k / q + k * k
    shelf = [(high + low * k / q + k * k) / a0, 2.0 * (k * k - high) / a0, (high - low * k / q + k * k) / a0,
             1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    q, frequency = 0.5003270373253953, 38.13547087613982
    k = math.tan(math.pi * frequency / sample_rate)
    a0 = 1.0 + k / q + k * k
    high_pass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([shelf, high_pass])


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Loudness of ``audio`` ([samples] or [channels, samples]) in LUFS;
    -inf when it is shorter than a block or silent."""
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    block, step = int(round(BLOCK_SECONDS * sample_rate)), int(round(STEP_SECONDS * sample_rate))
    if audio.shape[-1] < block:
        return -math.inf
    power = signal.sosfilt(k_weighting(sample_rate), audio, axis=-1) ** 2
    total = np.concatenate((np.zeros((power.shape[0], 1)), np.cumsum(power, axis=-1)), axis=-1)
    starts = np.arange(0, power.shape[-1] - block + 1, step)
    blocks = ((total[:, starts + block] - total[:, starts]) / block).sum(0)
    loudness = -0.691 + 10.0 * np.log10(blocks + 1e-20)
    kept = loudness > ABSOLUTE_GATE
    if not kept.any():
        return -math.inf
    kept &= loudness > -0.691 + 10.0 * math.log10(blocks[kept].mean()) + RELATIVE_GATE
    return -0.691 + 10.0 * math.log10(blocks[kept].mean())


def loudness_gain(audio: np.ndarray, sample_rate: int, target: float) -> float:
    """The gain that brings ``audio`` to ``target`` LUFS, at most
    ``MAX_BOOST_DB`` up; 1 for a silent one."""
    measured = integrated_loudness(audio, sample_rate)
    if not math.isfinite(measured):
        return 1.0
    return 10.0 ** (min(target - measured, MAX_BOOST_DB) / 20.0)


def limit_peaks(audio: np.ndarray, sample_rate: int, ceiling_db: float = -1.0) -> np.ndarray:
    """``audio`` ([samples] or [channels, samples]) with every sample under
    ``ceiling_db`` dBFS: each peak's reduction is held ``LIMIT_HOLD_SECONDS``
    around it and ramped over ``LIMIT_RAMP_SECONDS``, half the hold, so the
    gain is down by the time the peak arrives."""
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = np.abs(audio) if audio.ndim == 1 else np.abs(audio).max(0)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if needed.min() >= 1.0:
        return audio
    gain = minimum_filter1d(needed, size=max(1, int(LIMIT_HOLD_SECONDS * sample_rate)) | 1, mode="nearest")
    gain = uniform_filter1d(gain, size=max(1, int(LIMIT_RAMP_SECONDS * sample_rate)), mode="nearest")
    return (audio * gain).astype(audio.dtype)
