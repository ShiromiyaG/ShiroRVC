"""Aperiodic share of each frame's energy and the tilt of its harmonics: the
flow's breathiness and tension inputs.

Measured between and on the harmonics of the audio's own pitch track, in Hz rather
than bins, so the 44.1 kHz training audio and the 16 kHz inference input read
alike. A ratio, so the input gain does not move it.
"""

import torch
from torch.nn import functional as F

#: Frames per second, the pitch track's.
FRAME_RATE = 100
#: Three periods of an 80 Hz note, so its harmonics are resolved.
WINDOW_SECONDS = 0.04
#: The band the inference input (16 kHz) has.
MIN_HZ = 50.0
MAX_HZ = 8000.0
#: Spread of the band around each harmonic counted as periodic.
HARMONIC_SIGMA_HZ = 25.0
#: Everything more periodic reads as fully voiced.
FLOOR_DB = -30.0


def _harmonics(audio: torch.Tensor, sample_rate: int, f0: torch.Tensor, frames: int):
    """Power per bin [batch, bins, frames] in the band, each bin's weight as
    part of a harmonic of ``f0``, and which bins sit on the fundamental."""
    hop = int(sample_rate) // FRAME_RATE
    window = int(round(WINDOW_SECONDS * sample_rate))
    # Frame t centred on sample t * hop, like the pitch track.
    short = max(0, frames * hop - audio.shape[-1])
    audio = F.pad(audio.float(), (window // 2, window // 2 + short))
    power = torch.stft(
        audio, window, hop_length=hop, window=torch.hann_window(window, device=audio.device),
        center=False, return_complex=True,
    ).abs().square()[..., :frames]
    freqs = torch.fft.rfftfreq(window, 1.0 / sample_rate).to(audio.device)
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    power, freqs = power[:, band], freqs[band]

    f0 = F.pad(f0.float(), (0, max(0, frames - f0.shape[-1])))[:, :frames]
    ratio = freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
    nearest = ratio.round()
    distance = (ratio - nearest).abs() * f0[:, None, :]
    periodic = torch.exp(-0.5 * (distance / HARMONIC_SIGMA_HZ).square())
    periodic = periodic * ((nearest >= 1) & (f0[:, None, :] > 0))
    return power, periodic, nearest == 1


def aperiodicity(audio: torch.Tensor, sample_rate: int, f0: torch.Tensor, frames: int) -> torch.Tensor:
    """Aperiodic share of the energy per 10 ms frame, [batch, frames], mapped
    from [FLOOR_DB, 0] dB to [-1, 1].

    ``audio`` is [batch, samples] at ``sample_rate``; ``f0`` [batch, >= frames]
    is its own pitch in Hz at ``FRAME_RATE``, 0 where unvoiced. Unvoiced and
    silent frames read as fully aperiodic.
    """
    power, periodic, _ = _harmonics(audio, sample_rate, f0, frames)
    between = 1.0 - periodic
    # The power between the harmonics, spread over the whole band.
    noise = (power * between).sum(1) / between.sum(1).clamp_min(1e-3) * power.shape[1]
    total = power.sum(1)
    share = torch.where(total > 1e-8, noise / total.clamp_min(1e-10), torch.ones_like(total))
    share = share.clamp(1e-10, 1.0)
    db = (10.0 * torch.log10(share)).clamp(FLOOR_DB, 0.0)
    return db / (-FLOOR_DB / 2.0) + 1.0


def tension(audio: torch.Tensor, sample_rate: int, f0: torch.Tensor, frames: int) -> torch.Tensor:
    """How much of the harmonic amplitude lies above the fundamental per 10 ms
    frame, [batch, frames]: DiffSinger's tension, the logit of that share
    scaled by 0.1, from the harmonics in the band instead of a WORLD
    resynthesis. Arguments as ``aperiodicity``'s.

    Taken from the median of each input's voiced frames, so it reads how the
    voice departs from its own usual tilt and not the tilt itself, which is
    timbre. Unvoiced and silent frames read 0.
    """
    power, periodic, fundamental = _harmonics(audio, sample_rate, f0, frames)
    harmonic = (power * periodic).sum(1)
    above = (power * periodic * ~fundamental).sum(1)
    share = (above / harmonic.clamp_min(1e-10)).sqrt().clamp(1e-4, 1.0 - 1e-4)
    voiced = harmonic > 1e-8
    value = torch.logit(share) * 0.1
    median = value.masked_fill(~voiced, float("nan")).nanmedian(dim=1, keepdim=True).values
    return torch.where(voiced, value - median.nan_to_num(0.0), torch.zeros_like(value))
