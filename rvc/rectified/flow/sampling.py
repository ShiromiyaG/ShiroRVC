"""The sampler's time grids and guidance."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rvc.rectified.flow.conditioning import Conditioning

SAMPLERS = ("euler", "heun")
#: Step spacing over the sampled time range: even; sway (F5-TTS), denser near
#: the start; logit-normal, denser in the middle.
SCHEDULES = ("uniform", "sway", "logit-normal")
#: Where guidance rescale measures the output's spread.
RESCALE_MODES = ("global", "frame")


def blur_content(content: torch.Tensor) -> torch.Tensor:
    """``content`` [B, T, C] blurred to a quarter of its frame rate: what
    content guidance pushes away from."""
    frames = content.shape[1]
    blurred = F.interpolate(content.transpose(1, 2), size=max(1, frames // 4), mode="linear")
    return F.interpolate(blurred, size=frames, mode="linear").transpose(1, 2)


def time_grid(schedule: str, steps: int, start: float, device) -> torch.Tensor:
    """``steps + 1`` times from ``start`` to 1 under ``schedule``."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}.")
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "sway":
        # F5-TTS's sway with coefficient -1.
        g = 1.0 - torch.cos(0.5 * math.pi * u)
    elif schedule == "logit-normal":
        g = torch.sigmoid(math.sqrt(2.0) * torch.erfinv((2.0 * u - 1.0).clamp(-1.0, 1.0)))
    else:
        g = u
    g[0], g[-1] = 0.0, 1.0
    return start + (1.0 - start) * g


@dataclass(frozen=True)
class Guidance:
    """``RectifiedFlow.sample``'s guidance options, and what they make of the
    passes' velocities."""

    cfg_scale: float = 1.0
    content_guidance: float = 0.0
    rescale: float = 0.0
    rescale_mode: str = "global"
    interval: tuple = (0.0, 1.0)

    def variants(self, inputs: Conditioning, null_speaker: int) -> list:
        """(content, speaker) of each pass: the plain one, then the null
        speaker's and the blurred content's when their guidance is on."""
        variants = [(inputs.content, inputs.speaker)]
        if self.cfg_scale != 1.0:
            variants.append((inputs.content, torch.full_like(inputs.speaker, null_speaker)))
        if self.content_guidance > 0:
            variants.append((blur_content(inputs.content), inputs.speaker))
        return variants

    def active(self, now: float) -> bool:
        """Whether the guidances apply at the flow time ``now``."""
        start, until = self.interval
        # An interval reaching 1 includes it, where Heun's last evaluation lands.
        return start <= now and (now < until or until >= 1.0)

    def combine(self, passes, mask: torch.Tensor) -> torch.Tensor:
        """The guided velocity, from the passes' in ``variants``' order."""
        plain = passes[0]
        guided, index = plain, 1
        if self.cfg_scale != 1.0:
            guided = guided + (self.cfg_scale - 1.0) * (plain - passes[index])
            index += 1
        if self.content_guidance > 0:
            guided = guided + self.content_guidance * (plain - passes[index])
        if self.rescale > 0:
            rescaled = guided * self._spread(plain, mask) / self._spread(guided, mask).clamp_min(1e-6)
            guided = self.rescale * rescaled + (1.0 - self.rescale) * guided
        return guided

    def _spread(self, velocity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """RMS of ``velocity`` [B, n_mels, T], per frame or over each item."""
        if self.rescale_mode == "frame":
            return velocity.square().mean(1, keepdim=True).sqrt()
        count = mask.sum((1, 2)).clamp_min(1.0) * velocity.shape[1]
        return ((velocity.square() * mask).sum((1, 2)) / count).sqrt()[:, None, None]
