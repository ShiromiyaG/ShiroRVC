import numpy as np

# Vendored from librosa.filters, for use without a full librosa dependency.


def mel(
        *,
        sr,
        n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm="slaney",
        dtype=np.float32,
):
    """Build a Mel filter-bank matrix projecting FFT bins onto Mel bands."""

    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Area-normalize: divide by mel-band width for approx constant energy per channel.
        enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        print(
            "   [WARN] UserWarning:"
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


def fft_frequencies(*, sr=22050, n_fft=2048):
    """FFT bin center frequencies, ``(0, sr/n_fft, ..., sr/2)``."""
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0, htk=False):
    """Mel-scale-spaced frequencies between fmin and fmax.

    ``htk=False`` (default) uses the Slaney/Auditory-Toolbox formula (linear
    below 1 kHz, logarithmic above); ``htk=True`` uses the HTK formula
    ``mel = 2595 * log10(1 + f / 700)``.
    """
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def hz_to_mel(frequencies, *, htk=False):
    """Convert Hz to Mels."""
    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, *, htk=False):
    """Convert mel bin numbers to Hz."""
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def normalize(S, *, norm=np.inf, axis=0, threshold=None, fill=None):
    """Normalize an array along ``axis`` so that ``norm(S, axis=axis) == 1``.

    Slices whose norm falls below ``threshold`` are left as-is (``fill=None``),
    zeroed (``fill=False``), or filled to unit norm (``fill=True``).
    """

    if threshold is None:
        threshold = tiny(S)
    elif threshold <= 0:
        raise ValueError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ValueError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    mag = np.abs(S).astype(float)
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise NotImplementedError("Unsupported norm: {}".format(repr(norm)))

    small_idx = length < threshold

    S_norm = np.empty_like(S)
    if fill is None:
        length[small_idx] = 1.0
        S_norm[:] = S / length

    elif fill:
        # Locate small-norm entries via a nan-divide, then patch them.
        length[small_idx] = np.nan
        S_norm[:] = S / length
        S_norm[np.isnan(S_norm)] = fill_norm
    else:
        # inf-divide is safe here (IEEE-754) since S is already checked finite.
        length[small_idx] = np.inf
        S_norm[:] = S / length

    return S_norm


def tiny(x):
    """Smallest usable positive number for ``x.dtype`` (float32 for integer types)."""
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
            x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


if __name__ == '__main__':
    # Sanity check against librosa.filters.mel, if librosa is installed.
    try:
        import librosa
    except ImportError:
        print('  [UNIT_TEST] torchfcpe.mel_tools.mel_fn_librosa: librosa not installed,'
              ' if you want check this file with librosa, please install it first.')
        exit(1)
    from librosa.filters import mel as librosa_mel_fn

    raw_fn = librosa_mel_fn(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000)
    self_fn = mel(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000)
    print("  [UNIT_TEST] torchfcpe.mel_tools.mel_fn_librosa: raw_fn.shape", raw_fn.shape)
    print("  [UNIT_TEST] torchfcpe.mel_tools.mel_fn_librosa: self_fn.shape", self_fn.shape)
    check = np.allclose(raw_fn, self_fn)
    print("  [UNIT_TEST] torchfcpe.mel_tools.mel_fn_librosa: np.allclose(raw_fn, self_fn) is same:", check)
