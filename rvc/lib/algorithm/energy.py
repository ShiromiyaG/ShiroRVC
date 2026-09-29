"""Frame energy for the content encoder's energy conditioning.

Computed from the audio on the fly, the same way in training and at inference,
so nothing is extracted to disk.  Log domain rather than fish-diffusion's
linear RMS: loudness is perceived in dB, and a linear RMS puts almost every
quiet frame at the same value.
"""

import torch
from torch.nn import functional as F

#: Frames per second of the content features (10 ms hops).
FRAME_RATE = 100
#: Analysis window; long enough to span a low sung period.
WINDOW_SECONDS = 0.032
#: Everything quieter reads as silence.
FLOOR_DB = -70.0


def frame_energy(audio: torch.Tensor, sample_rate: int, frames: int) -> torch.Tensor:
    """Log RMS per 10 ms frame, [batch, frames], mapped from [FLOOR_DB, 0] dBFS
    to [-1, 1].

    ``audio`` is [batch, samples] or [batch, 1, samples] at ``sample_rate``.
    Frame ``t`` is centred on sample ``t * hop``, like the pitch track.
    """
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    hop = int(sample_rate) // FRAME_RATE
    window = int(round(WINDOW_SECONDS * sample_rate)) | 1
    power = F.avg_pool1d(
        audio.float().pow(2),
        kernel_size=window,
        stride=hop,
        padding=window // 2,
        count_include_pad=False,
    )
    if power.shape[-1] < frames:
        power = F.pad(power, (0, frames - power.shape[-1]), mode="replicate")
    db = (10.0 * torch.log10(power[..., :frames].clamp_min(1e-10))).clamp(FLOOR_DB, 0.0)
    return (db / (-FLOOR_DB / 2.0) + 1.0).squeeze(1)
