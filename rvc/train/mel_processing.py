import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    return spec


def spec_to_mel_torch(
    spec,
    n_fft,
    num_mels,
    sample_rate,
    fmin,
    fmax,
    log_compression=True,
):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    if log_compression:
        melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y,
    n_fft,
    num_mels,
    sample_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    log_compression=True,
):
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    melspec = spec_to_mel_torch(
        spec,
        n_fft,
        num_mels,
        sample_rate,
        fmin,
        fmax,
        log_compression=log_compression,
    )

    return melspec


def compute_window_length(n_mels: int, sample_rate: int):
    f_min = 0
    f_max = sample_rate / 2
    window_length_seconds = 8 * n_mels / (f_max - f_min)
    window_length = int(window_length_seconds * sample_rate)
    return 2 ** (window_length.bit_length() - 1)


class MultiScaleMelSpectrogramLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: list[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: list[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn=None,
        safe_log: bool = False,
        log_scale: float = 1000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        # Defaulted here, not in the signature: a module instance as a default
        # argument would be shared by every caller that omits it.
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.L1Loss()
        self.safe_log = bool(safe_log)
        self.log_scale = float(log_scale)
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params: list[tuple] = []
        self.hann_window: dict[int, torch.Tensor] = {}
        self.mel_banks: dict[int, torch.Tensor] = {}

        self.stft_params = [
            (mel, win) for mel, win in zip(n_mels, window_lengths)
        ]

    def mel_spectrogram(
        self,
        wav: torch.Tensor,
        n_mels: int,
        window_length: int,
    ):
        dtype_device = str(wav.dtype) + "_" + str(wav.device)
        win_dtype_device = str(window_length) + "_" + dtype_device
        mel_dtype_device = str(n_mels) + "_" + dtype_device
        if win_dtype_device not in self.hann_window:
            self.hann_window[win_dtype_device] = torch.hann_window(
                window_length, device=wav.device, dtype=torch.float32
            )

        wav = wav.squeeze(1)

        stft = torch.stft(
            wav.float(),
            n_fft=window_length,
            hop_length=window_length//4,
            window=self.hann_window[win_dtype_device],
            return_complex=True,
        )

        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

        if mel_dtype_device not in self.mel_banks:
            self.mel_banks[mel_dtype_device] = torch.from_numpy(
                librosa_mel_fn(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None,
                )
            ).to(device=wav.device, dtype=torch.float32)

        mel_spectrogram = torch.matmul(
            self.mel_banks[mel_dtype_device], magnitude
        )
        return mel_spectrogram

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        loss = 0.0
        for p in self.stft_params:
            real_mels = self.mel_spectrogram(real, *p)
            fake_mels = self.mel_spectrogram(fake, *p)
            if self.safe_log:
                real_logmels = torch.log1p(real_mels * self.log_scale)
                fake_logmels = torch.log1p(fake_mels * self.log_scale)
            else:
                real_logmels = torch.log(real_mels.clamp(min=1e-5)) / self.log_base
                fake_logmels = torch.log(fake_mels.clamp(min=1e-5)) / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)
        return loss
