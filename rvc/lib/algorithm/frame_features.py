"""Per-frame conditioning signals derived from a magnitude spectrogram.

Two properties of the target that the prior is otherwise never told, both
measurable from the audio and both available at inference from the *source*:

``periodicity``
    How much of the frame's energy sits on the harmonic comb rather than
    between its teeth.  The NSF source decides harmonic-versus-noise with a
    binary voicing flag and two fixed constants, but measured on this dataset
    the harmonic-to-noise ratio varies by 13.8 dB *within a single file* --
    breath, creak and mixed excitation are a per-frame property, and a
    two-position switch cannot represent them.

``energy``
    The loudness envelope.  A linear probe on ContentVec + f0 leaves 8.7 dB of
    the within-file envelope unexplained, and the model carries two losses
    (``envelope`` and ``rms``) that ask it to reproduce exactly that.

Both are computed **here and only here** so the training and inference paths
cannot drift apart.  Two properties make that safe:

* Both are band-limited to ``BAND_LIMIT_HZ``.  Training measures a 44.1 kHz
  spectrogram and inference measures the 16 kHz source the embedder already
  consumes; without a common band the harmonic share would differ
  systematically, because 8-22 kHz is mostly noise and only one side has it.
* Both are gain invariant -- periodicity is a ratio, and the energy is
  expressed relative to the utterance's own mean.  A file that arrives 6 dB
  hotter produces the same conditioning, so no normalisation difference
  between the trainer and the inference pipeline can silently shift the
  operating point.  The absolute level stays where it already is, in
  ``change_rms``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

BAND_LIMIT_HZ = 8000.0

# Width of the soft window placed on each harmonic, as a fraction of f0.  Wide
# enough that a slightly mistracked f0 still lands inside it, narrow enough that
# the windows do not merge at the lowest pitches this dataset contains.
HARMONIC_SIGMA_RATIO = 0.12

# The energy envelope in log units divided by this lands close to unit variance
# for speech, which keeps it on the same scale as the other prior inputs.
ENERGY_LOG_SCALE = 3.0


def _band_mask(bins: int, sample_rate: float, n_fft: int, device, dtype) -> Tensor:
    frequencies = torch.arange(bins, device=device, dtype=dtype) * (
        float(sample_rate) / float(n_fft)
    )
    return frequencies, (frequencies <= BAND_LIMIT_HZ)


def frame_periodicity(
    spec: Tensor,
    f0: Tensor,
    sample_rate: float,
    n_fft: int,
) -> Tensor:
    """Excess harmonic energy share per frame, in ``[0, 1]``.

    ``spec`` is a magnitude spectrogram ``(batch, bins, frames)`` and ``f0`` is
    ``(batch, frames)`` in Hz with zeros on unvoiced frames.

    The raw share of energy falling under the harmonic windows is not itself
    the answer: even white noise scores whatever fraction of the band those
    windows happen to cover, and that fraction grows as f0 falls.  Subtracting
    the flat-spectrum expectation and renormalising makes 0 mean "indis-
    tinguishable from noise" and 1 mean "all energy on the comb" at every pitch.
    """
    spec = spec.float()
    f0 = f0.float()
    batch, bins, frames = spec.shape
    if f0.shape[-1] != frames:
        f0 = torch.nn.functional.interpolate(
            f0.unsqueeze(1), size=frames, mode="nearest"
        ).squeeze(1)

    frequencies, band = _band_mask(bins, sample_rate, n_fft, spec.device, spec.dtype)
    power = spec.square() * band.view(1, -1, 1)

    safe_f0 = f0.clamp_min(1e-3).unsqueeze(1)                    # (b, 1, t)
    grid = frequencies.view(1, -1, 1)                            # (1, f, 1)
    harmonic = torch.round(grid / safe_f0)
    distance = (grid - harmonic * safe_f0).abs()
    sigma = (HARMONIC_SIGMA_RATIO * safe_f0).clamp_min(
        float(sample_rate) / float(n_fft) * 0.5
    )
    window = torch.exp(-0.5 * (distance / sigma).square()) * band.view(1, -1, 1)

    total = power.sum(dim=1).clamp_min(1e-12)
    share = (power * window).sum(dim=1) / total
    # What that share would be if the frame were flat noise.
    flat = window.sum(dim=1) / band.sum().clamp_min(1).to(window.dtype)
    periodicity = (share - flat) / (1.0 - flat).clamp_min(1e-6)
    voiced = (f0 > 0).to(periodicity.dtype)
    return (periodicity.clamp(0.0, 1.0) * voiced).unsqueeze(1)


def frame_energy(spec: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Log frame energy relative to the utterance mean, ``(batch, 1, frames)``.

    Relative, not absolute: the between-file spread is 13.7 dB on this dataset
    and is genuinely unknowable from content, but it is also the part that a
    differently normalised input file would corrupt.  The within-file envelope
    is 19.8 dB, is what the envelope losses actually ask for, and survives any
    gain change.
    """
    spec = spec.float()
    power = spec.square().mean(dim=1).clamp_min(1e-12)           # (b, t)
    log_power = power.log()
    if mask is None:
        reference = log_power.mean(dim=-1, keepdim=True)
    else:
        valid = mask.reshape(log_power.shape[0], -1).float()
        if valid.shape[-1] != log_power.shape[-1]:
            valid = torch.nn.functional.interpolate(
                valid.unsqueeze(1), size=log_power.shape[-1], mode="nearest"
            ).squeeze(1)
        reference = (log_power * valid).sum(-1, keepdim=True) / valid.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
    return ((log_power - reference) / ENERGY_LOG_SCALE).unsqueeze(1)


def frame_conditioning(
    spec: Optional[Tensor],
    f0: Optional[Tensor],
    sample_rate: float,
    n_fft: int,
    use_periodicity: bool,
    use_energy: bool,
    mask: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """Stack the enabled signals into ``(batch, channels, frames)``.

    Returns ``None`` when nothing is enabled or when ``spec`` is unavailable, so
    every consumer degrades to its previous behaviour rather than to zeros --
    a caller that forgets to supply the spectrogram gets the old model, not a
    silently miscalibrated one.
    """
    if spec is None or not (use_periodicity or use_energy):
        return None
    channels = []
    if use_periodicity:
        if f0 is None:
            return None
        channels.append(frame_periodicity(spec, f0, sample_rate, n_fft))
    if use_energy:
        channels.append(frame_energy(spec, mask))
    return torch.cat(channels, dim=1)


def template_amplitude(energy: Tensor, wave_amp: float = 0.1) -> Tensor:
    """Turn relative log frame energy into RefineGAN's intensity envelope.

    The paper drives the pitch template's amplitude from a frame-level intensity
    measurement, which is what gives the vocoder its intensity response: the
    excitation is already at the right loudness before the refinement network
    sees it.  ``frame_energy`` is a *log power* relative to the utterance mean,
    so amplitude is ``exp(energy * scale / 2)``, and ``wave_amp`` places the
    utterance mean at the template's nominal level.

    Computed in FP32: the exponential of a +/- 3 sigma log energy overflows the
    FP16 range for loud frames well before the clamp could catch it.
    """
    energy = energy.float()
    return (float(wave_amp) * torch.exp(0.5 * ENERGY_LOG_SCALE * energy)).clamp(0.0, 1.0)


def conditioning_channels(use_periodicity: bool, use_energy: bool) -> int:
    return int(bool(use_periodicity)) + int(bool(use_energy))


# The training spectrogram these features were defined against.
TRAIN_SAMPLE_RATE = 44100
TRAIN_N_FFT = 2048


def matched_n_fft(sample_rate: float) -> int:
    """FFT size giving the same Hz-per-bin as training, at another rate.

    Everything in ``frame_periodicity`` is expressed in Hz through
    ``sample_rate / n_fft``, so holding that ratio fixed makes the measurement
    identical across rates: 2048 at 44.1 kHz and 744 at 16 kHz are both 21.5 Hz
    per bin.  The inference pipeline runs at 16 kHz, and without this the same
    audio would score a different periodicity there than in training.
    """
    size = int(round(TRAIN_N_FFT * float(sample_rate) / TRAIN_SAMPLE_RATE))
    return max(64, size + (size % 2))


def spectrogram_for_features(
    audio: Tensor, n_fft: int, hop_length: int
) -> Tensor:
    """Magnitude spectrogram on the convention the trainer uses (no centring).

    ``center=False`` matches ``rvc.train.mel_processing.spectrogram_torch``; a
    centred transform would shift every frame by half a window and quietly
    misalign the envelope against the f0 it is paired with.
    """
    audio = audio.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(int(n_fft), device=audio.device, dtype=audio.dtype)
    stft = torch.stft(
        audio,
        int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=False,
        return_complex=True,
    )
    return stft.abs()
