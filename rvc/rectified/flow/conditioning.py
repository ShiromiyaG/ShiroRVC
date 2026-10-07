"""What the flow is conditioned on, and the encoder that turns it into
per-frame conditioning."""

import math
from typing import NamedTuple, Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn

from rvc.rectified.flow.layers import ConvNeXtBlock

#: Centre and spread of log f0, so the normalised pitch sits roughly in [-2, 2].
LOG_F0_CENTER = math.log(200.0)
LOG_F0_SCALE = 0.7


class Conditioning(NamedTuple):
    """What the flow is conditioned on: ``content`` [B, T, C]; ``f0`` in Hz,
    ``energy``, ``breathiness`` and ``tension`` [B, T]; ``speaker``,
    ``key_shift`` and ``speed`` [B]; ``mask`` [B, 1, T].

    ``key_shift`` is the formant shift in semitones: training shifts the mel's
    whole spectrum with the pitch, so 0 keeps the voice's own formants at any
    pitch. ``speed`` is the time stretch training applied, 1 at inference.
    Missing ``breathiness``, ``tension``, ``key_shift`` and ``speed`` read as
    fully aperiodic, 0, 0 and 1.
    """

    content: torch.Tensor
    f0: torch.Tensor
    energy: torch.Tensor
    speaker: torch.Tensor
    mask: torch.Tensor
    breathiness: Optional[torch.Tensor] = None
    key_shift: Optional[torch.Tensor] = None
    speed: Optional[torch.Tensor] = None
    tension: Optional[torch.Tensor] = None

    def map(self, function) -> "Conditioning":
        """``function`` of every input that is there."""
        return Conditioning(*(None if value is None else function(value) for value in self))

    def to(self, device, non_blocking: bool = False) -> "Conditioning":
        return self.map(lambda value: value.to(device, non_blocking=non_blocking))

    def crop(self, start: int, stop: int) -> "Conditioning":
        """Frames ``start`` to ``stop`` of the per-frame inputs."""

        def cut(value):
            return None if value is None else value[:, start:stop]

        return self._replace(
            content=cut(self.content), f0=cut(self.f0), energy=cut(self.energy),
            mask=self.mask[..., start:stop],
            breathiness=cut(self.breathiness), tension=cut(self.tension),
        )


def pitch_features(f0: torch.Tensor, fourier: int = 0) -> torch.Tensor:
    """``f0`` [batch, frames] in Hz -> [batch, 2 + 2 * fourier, frames]:
    normalised log f0 and the voiced flag, then sines and cosines of the log
    f0 at ``fourier`` octave-spaced frequencies, all 0 where unvoiced. The
    finest of 6 resolves under a semitone, which a linear map of one scalar
    does not."""
    voiced = (f0 > 0).float()
    log_f0 = (torch.log(f0.clamp_min(1.0)) - LOG_F0_CENTER) / LOG_F0_SCALE
    features = [log_f0 * voiced, voiced]
    for index in range(fourier):
        angle = (2.0**index * math.pi) * log_f0
        features += [torch.sin(angle) * voiced, torch.cos(angle) * voiced]
    return torch.stack(features, dim=1)


class HarmonicPrior(nn.Module):
    """Where f0's harmonics fall in the mel, [batch, n_mels, frames].

    Each bin is the share of its filter covered by a flat harmonic comb, so a
    resolved harmonic reads near 1 and the gaps near 0, while bins wider than
    the harmonic spacing read the comb's mean density. 0 where unvoiced.
    """

    def __init__(self, sample_rate, n_fft, n_mels, fmin, fmax):
        super().__init__()
        basis = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        ).float()
        basis = basis / basis.sum(1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "freqs", torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).float(), persistent=False
        )
        # One STFT bin: half the Hann window's main lobe.
        self.sigma = sample_rate / n_fft

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        f0 = f0.float()
        ratio = self.freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
        nearest = ratio.round()
        distance = (ratio - nearest).abs() * f0[:, None, :]
        comb = torch.exp(-0.5 * (distance / self.sigma).square()) * (nearest >= 1)
        with torch.autocast(f0.device.type, enabled=False):
            prior = torch.matmul(self.basis, comb)
        return prior * (f0 > 0).float()[:, None, :]


class ConditionEncoder(nn.Module):
    """Content, pitch, loudness, breathiness, tension, key shift and speaker
    -> per-frame conditioning.

    Speaker row ``speaker_count`` is the null speaker used for classifier-free
    guidance. The boolean arguments say which of ``Conditioning``'s optional
    inputs the model takes; the others are ignored.
    """

    def __init__(
        self,
        content_channels: int,
        hidden_channels: int,
        speaker_count: int,
        speaker_channels: int,
        layers: int,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[HarmonicPrior] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        speed: bool = False,
        tension: bool = False,
    ):
        super().__init__()
        self.speaker_count = int(speaker_count)
        self.content = nn.Linear(content_channels, hidden_channels)
        self.pitch_fourier = int(pitch_fourier)
        self.pitch = nn.Conv1d(2 + 2 * self.pitch_fourier, hidden_channels, 3, padding=1)
        self.harmonic_prior = harmonic_prior
        if harmonic_prior is not None:
            self.harmonics = nn.Conv1d(harmonic_prior.basis.shape[0], hidden_channels, 1)
        self.energy = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.breathiness = nn.Conv1d(1, hidden_channels, 3, padding=1) if breathiness else None
        self.tension = None
        if tension:
            # From zero, so a checkpoint without the input starts unchanged.
            self.tension = nn.Conv1d(1, hidden_channels, 3, padding=1)
            nn.init.zeros_(self.tension.weight)
            nn.init.zeros_(self.tension.bias)
        self.key_shift = nn.Linear(1, hidden_channels) if key_shift else None
        self.speed = nn.Linear(1, hidden_channels) if speed else None
        self.speaker = nn.Embedding(self.speaker_count + 1, speaker_channels)
        self.speaker_proj = nn.Linear(speaker_channels, hidden_channels)
        self.blocks = nn.ModuleList([ConvNeXtBlock(hidden_channels) for _ in range(layers)])

    def voice(self, speaker: torch.Tensor) -> torch.Tensor:
        """The speaker's embedding, [B, hidden]; the null row for guidance."""
        return self.speaker_proj(self.speaker(speaker))

    def forward(self, inputs: Conditioning) -> torch.Tensor:
        """The per-frame conditioning, [B, hidden, T]."""
        content, f0, energy, speaker, mask, breathiness, key_shift, speed, tension = inputs
        x = self.content(content).transpose(1, 2)
        x = x + self.pitch(pitch_features(f0, self.pitch_fourier))
        if self.harmonic_prior is not None:
            x = x + self.harmonics(self.harmonic_prior(f0))
        x = x + self.energy(energy.unsqueeze(1))
        if self.breathiness is not None:
            if breathiness is None:
                breathiness = torch.ones_like(energy)
            x = x + self.breathiness(breathiness.unsqueeze(1))
        if self.tension is not None:
            if tension is None:
                tension = torch.zeros_like(energy)
            x = x + self.tension(tension.unsqueeze(1))
        if self.key_shift is not None:
            if key_shift is None:
                key_shift = torch.zeros(content.shape[0], device=content.device)
            x = x + self.key_shift(key_shift.float().view(-1, 1) / 12.0).unsqueeze(-1)
        if self.speed is not None:
            if speed is None:
                speed = torch.ones(content.shape[0], device=content.device)
            x = x + self.speed(speed.float().view(-1, 1)).unsqueeze(-1)
        x = x + self.voice(speaker).unsqueeze(-1)
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        return x
