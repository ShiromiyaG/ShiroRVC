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
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        # Defaulted here, not in the signature: a module instance as a default
        # argument would be shared by every caller that omits it.
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.L1Loss()
        self.safe_log = bool(safe_log)
        self.log_scale = float(log_scale)
        #: Applied to the summed loss, so a scale set carries its own
        #: normalisation instead of leaving every call site to remember a
        #: divisor.  1.0 is Applio's behaviour -- the sum, undivided -- which
        #: is what the parity test constructs.
        self.output_scale = float(output_scale)
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
        return loss * self.output_scale


# Which resolutions the multi-scale mel loss actually runs at, and the divisor
# that goes with them.  Applio's seven (5/32 .. 320/2048) are the class
# defaults and stay there, because the parity test pins them; what follows
# is what this fork builds instead, via ``build_ms_mel_loss``.
#
# Both ends of the list were chosen by probing one defect at a time and
# comparing the penalty against the single-scale L1 at c_mel=45.  Every figure
# below is on speech at -16 LUFS: at a low level the mel clamp floor pins a
# large share of the high-frequency bins in the coarse scales -- 52% of them
# above 6.5 kHz at 40 LUFS down -- and a pinned bin charges nothing, which
# makes those scales look worse than they are.
#
# Per scale, each normalised to parity with the L1 on a 1-3 kHz comb so the
# numbers compare shape rather than weight:
#
#   mels/win    ms   noise  jitter1ms  smear  detune  comb>6.5k
#     20/128     4    0.66       3.09   0.54    0.92       0.74
#     40/256     8    0.83       2.22   0.74    1.04       0.87
#    160/1024   32    1.01       1.33   0.98    1.14       0.99
#    320/2048   64    1.05       1.34   1.09    1.13       1.02
#    640/4096  128    1.07       1.69   1.23    1.22       1.05
#
# Short windows buy timing and cost spectral detail; long windows do the
# reverse, and a set wants both ends.  The coarse end (32/64/128) still goes,
# because it charges almost nothing for a defect above 6.5 kHz while charging
# as much as the fine scales at 1-3 kHz -- summing that and dividing by three
# tilts the loss toward low frequency.  Destroying the harmonic comb one band
# at a time, each set normalised to parity at 1-3 kHz:
#
#   config                1-3k   3-6k   6-9k  9-12k  12-15k   span
#   seven, /3.25          1.00   0.97   0.98   0.92    0.73   0.27
#   256..2048, /1.81      1.00   0.98   0.96   0.96    0.92   0.08
#   256..4096, /2.20      1.00   0.98   0.97   0.96    0.92   0.08
#
# And what the whole set does against the L1, by defect:
#
#   config             noise  jitter1ms  jitter3ms  smear  detune  comb>6.5k
#   seven, /3.0         0.82       ----       ----   0.77    1.10       0.91
#   256..2048, /1.79    0.95       1.64       1.48   0.92    1.11       0.96
#   256..4096, /2.20    0.97       1.65       1.40   0.97    1.13       0.97
#
# 4096 is free: 13 frames per 400 ms segment against 26 for 2048, so the whole
# loss measured 25 ms per call at batch 16 against 27 without it.  Nothing here
# beats the L1 on every axis at once -- that is the trade between the two ends
# of the list, not something a scale set escapes -- but this one is well ahead
# on timing, ahead on pitch, and within 3% everywhere else.
MS_MEL_N_MELS = [40, 80, 160, 320, 640]
MS_MEL_WINDOWS = [256, 512, 1024, 2048, 4096]
#: Chosen for parity with the single-scale L1 on a 1-3 kHz comb, which is where
#: the original divisor already sat; what these constants change is the shape
#: across frequency and time, not the overall weight.  It rides inside the
#: module as ``output_scale`` rather than being applied by callers: the
#: divisor is a property of *this scale set*, and a call site that forgot it
#: would train at more than twice the intended mel weight with nothing to
#: show for it but a number that looks plausible.
MS_MEL_DIVISOR = 2.20


def build_ms_mel_loss(sample_rate: int, loss_fn=None):
    """This fork's multi-scale mel loss: the scale set above, already normalised.

    The divisor is folded in, so this is weighted with the same ``c_mel`` the
    single-scale L1 uses and needs no adjustment at the call site.
    """

    return MultiScaleMelSpectrogramLoss(
        sample_rate=sample_rate,
        n_mels=MS_MEL_N_MELS,
        window_lengths=MS_MEL_WINDOWS,
        safe_log=False,
        loss_fn=loss_fn,
        output_scale=1.0 / MS_MEL_DIVISOR,
    )
