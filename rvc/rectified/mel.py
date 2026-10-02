import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F


class LogMel(nn.Module):
    """Log mel spectrogram at one frame per ``hop_length`` samples.

    The same mel as SingingVocoders' ``PitchAdjustableMelSpectrogram``, so its
    vocoders render this model's mels. Its own basis buffer: ``mel_processing``
    caches the basis by ``fmax`` alone, which collides with a second mel size.
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        basis = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    @classmethod
    def from_config(cls, data: dict) -> "LogMel":
        return cls(
            data["sample_rate"],
            data["n_fft"],
            data["win_length"],
            data["hop_length"],
            data["n_mels"],
            data["mel_fmin"],
            data["mel_fmax"],
        )

    def forward(self, audio: torch.Tensor, key_shift: float = 0.0, hop_length=None) -> torch.Tensor:
        """``audio`` [batch, samples] -> [batch, n_mels, samples // hop].

        ``key_shift`` semitones scale every frequency, pitch and formants, by
        ``2 ** (key_shift / 12)``: the STFT is taken that much longer and read
        with the original bins, as SingingVocoders' ``key_shift`` does. A
        ``hop_length`` other than the configured one stretches time instead.
        """
        hop_length = int(hop_length or self.hop_length)
        factor = 2.0 ** (key_shift / 12.0)
        n_fft = int(round(self.n_fft * factor))
        win_length = int(round(self.win_length * factor))
        window = self.window
        if win_length != self.win_length:
            window = torch.hann_window(win_length, device=audio.device)
        pad = win_length - hop_length
        # Reflection needs more samples than it pads: a shorter clip gets
        # trailing silence, and its extra frames are the caller's to drop.
        short = (pad + 1) // 2 + 1 - audio.shape[-1]
        if short > 0:
            audio = F.pad(audio, (0, short))
        audio = F.pad(
            audio.float().unsqueeze(1), (pad // 2, (pad + 1) // 2), mode="reflect"
        ).squeeze(1)
        spec = torch.stft(
            audio,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        if n_fft != self.n_fft:
            bins = self.n_fft // 2 + 1
            spec = F.pad(spec, (0, 0, 0, max(0, bins - spec.shape[1])))[:, :bins]
            spec = spec * (self.win_length / win_length)
        return torch.log(torch.clamp(self.basis @ spec, min=1e-5))


def normalize_mel(mel: torch.Tensor, data: dict) -> torch.Tensor:
    return (mel - data["mel_mean"]) / data["mel_std"]


def denormalize_mel(mel: torch.Tensor, data: dict) -> torch.Tensor:
    return mel * data["mel_std"] + data["mel_mean"]


#: At full strength: how far the top band drops and the noise's spread, in
#: normalised mel units.
DEGRADE_HIGH_BAND = 0.5
DEGRADE_NOISE = 0.3


def degrade_mel(mel: torch.Tensor, strength: float) -> torch.Tensor:
    """Normalised ``mel`` [batch, n_mels, frames] with the kinds of error a
    generated mel carries, so a vocoder trained on it tolerates them: blur
    along time and along frequency, a duller top band and noise. Each is drawn
    per item up to ``strength`` (0 to 1) of its full amount; half the items
    stay clean, which keeps the vocoder exact on a real mel.
    """
    batch, bands = mel.shape[0], mel.shape[1]

    def amount():
        return torch.rand(batch, 1, 1, device=mel.device) * strength

    out = torch.lerp(mel, F.avg_pool1d(F.pad(mel, (1, 1), mode="replicate"), 3, 1), amount())
    across = F.avg_pool1d(F.pad(out.transpose(1, 2), (1, 1), mode="replicate"), 3, 1).transpose(1, 2)
    out = torch.lerp(out, across, amount())
    # A ramp down from a band drawn in the upper two thirds to the top.
    start = 0.3 + 0.6 * torch.rand(batch, 1, 1, device=mel.device)
    position = torch.linspace(0.0, 1.0, bands, device=mel.device).view(1, -1, 1)
    out = out - DEGRADE_HIGH_BAND * amount() * ((position - start) / (1.0 - start)).clamp(0.0, 1.0)
    out = out + DEGRADE_NOISE * amount() * torch.randn_like(out)
    clean = torch.rand(batch, 1, 1, device=mel.device) < 0.5
    return torch.where(clean, mel, out)
