"""XLA-friendly replacements for the parts of vocoder training (RefineGAN v2,
HiFi-GAN, HiFi-GAN++) and RMVPE extraction that TPUs handle badly.

Each patch keeps the fork's math and changes only the form: complex STFTs become
two real matmuls, strided in-place writes become stack/reshape, and
``F.interpolate`` becomes fixed-index gathers.  No parameter or buffer is added
or renamed, so checkpoints stay interchangeable with the CUDA trainer.

Call :func:`apply` (training) or :func:`apply_extract` (extraction) in every
process before the first forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_DFT_BASES: dict = {}
_RESIZE_INDEX: dict = {}


def _dft_basis(n_fft: int, device) -> torch.Tensor:
    """``[cos; -sin]`` rows of the one-sided DFT, ``(2 * (n_fft // 2 + 1), n_fft)``."""
    key = (int(n_fft), str(device))
    if key not in _DFT_BASES:
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        angle = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)], dim=0)
        _DFT_BASES[key] = basis.float().to(device)
    return _DFT_BASES[key]


def stft_parts(x, n_fft, hop_length, win_length, window, center, pad_mode="reflect"):
    """Real and imaginary parts of ``torch.stft(x, ...)``, each ``(B, n_fft // 2 + 1, T)``.

    ``x`` is ``(B, samples)``.  Always FP32, whatever autocast is active.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        if center:
            x = F.pad(x.unsqueeze(1), (n_fft // 2, n_fft // 2), mode=pad_mode).squeeze(1)
        window = window.to(device=x.device, dtype=torch.float32)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = F.pad(window, (left, n_fft - win_length - left))
        frames = x.unfold(-1, n_fft, hop_length) * window
        spectrum = torch.matmul(frames, _dft_basis(n_fft, x.device).t())
        real, imag = spectrum.transpose(1, 2).chunk(2, dim=1)
        return real, imag


def _magnitude(real, imag):
    # Same backward as ``torch.norm(view_as_real(...))``: zero at a zero bin.
    return torch.stack((real, imag), dim=-1).norm(p=2, dim=-1)


def _mrd_spectrogram(self, x):
    n_fft, hop_length, win_length = self.resolution
    pad = int((n_fft - hop_length) / 2)
    x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
    real, imag = stft_parts(x, n_fft, hop_length, win_length, self.window, center=False)
    return _magnitude(real, imag)


def _univhd_spectrogram(self, x):
    real, imag = stft_parts(
        x.squeeze(1), self.n_fft, self.hop_length, self.n_fft, self.window, center=True
    )
    return _magnitude(real, imag)


def _ms_mel_spectrogram(self, wav, n_mels, window_length):
    from librosa.filters import mel as librosa_mel_fn

    window_key = f"{window_length}_{wav.device}"
    mel_key = f"{n_mels}_{window_length}_{wav.device}"
    if window_key not in self.hann_window:
        self.hann_window[window_key] = torch.hann_window(
            window_length, dtype=torch.float32
        ).to(wav.device)
    if mel_key not in self.mel_banks:
        bank = librosa_mel_fn(
            sr=self.sample_rate, n_mels=n_mels, n_fft=window_length, fmin=0, fmax=None
        )
        self.mel_banks[mel_key] = torch.from_numpy(bank).float().to(wav.device)

    real, imag = stft_parts(
        wav.squeeze(1),
        window_length,
        window_length // 4,
        window_length,
        self.hann_window[window_key],
        center=True,
    )
    with torch.autocast(device_type=wav.device.type, enabled=False):
        magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-6)
        return torch.matmul(self.mel_banks[mel_key], magnitude)


_ORIGINAL_POLYPHASE = None
_ORIGINAL_SINC_KERNEL = None


def _polyphase(self, x):
    """``AntiAliasedUpsample1d._polyphase`` built on the CPU and uploaded.

    Built on the device, its construction ops land in step 1's graph only, and
    step 2 then compiles the whole step a second time.
    """
    key = (int(x.shape[1]), x.dtype, x.device)
    if self.__dict__.get("_xla_poly_key") != key:
        probe = torch.empty(1, key[0], 0, dtype=x.dtype)
        self.__dict__["_xla_poly"] = _ORIGINAL_POLYPHASE(self, probe).to(x.device)
        self.__dict__["_xla_poly_key"] = key
    return self.__dict__["_xla_poly"]


def _get_sinc_resample_kernel(*args, device=None, dtype=None, **kwargs):
    """torchaudio's kernel built on the CPU and uploaded, for the same reason as ``_polyphase``."""
    kernel, width = _ORIGINAL_SINC_KERNEL(*args, device=torch.device("cpu"), dtype=dtype, **kwargs)
    return kernel.to(device), width


def _upsample_plan(self):
    """``(left, right, first_phase, count)`` per run of phases sharing a ``start``."""
    plan = self.__dict__.get("_xla_upsample_plan")
    if plan is None:
        plan = []
        for phase, start in enumerate(self.starts):
            left = self.pad + self.extra_left - start
            right = self.taps - 1 - left
            if left < 0 or right < 0:
                raise ValueError(f"AntiAliasedUpsample1d x{self.factor}: phase {phase} needs a crop.")
            if plan and plan[-1][:2] == (left, right):
                plan[-1] = (left, right, plan[-1][2], plan[-1][3] + 1)
            else:
                plan.append((left, right, phase, 1))
        self.__dict__["_xla_upsample_plan"] = plan
    return plan


def _upsample_forward(self, x):
    """``AntiAliasedUpsample1d.forward`` with each phase's offset moved into its input pad.

    Cropping the conv output at ``start`` is what the TPU compiler folds into a
    negative window pad and rejects (``fusion_emitter.cc``); padding the input
    ``start`` samples less gives the same samples with no crop.
    """
    if self.factor == 1:
        return x
    batch, channels, length = x.shape[0], x.shape[1], x.shape[-1]
    weight = self._polyphase(x).view(channels, self.factor, 1, self.taps)
    parts = []
    for left, right, first, count in _upsample_plan(self):
        padded = F.pad(x, (left, right), mode="replicate")
        kernel = weight.narrow(1, first, count).reshape(channels * count, 1, self.taps)
        out = F.conv1d(padded, kernel, groups=channels)
        parts.append(out.view(batch, channels, count, length))
    phases = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
    return phases.transpose(2, 3).reshape(batch, channels, length * self.factor)


def _f02sine(self, f0):
    """``SineGenerator._f02sine`` without in-place writes into fresh tensors."""
    rad_values = (f0 / self.sampling_rate) % 1

    rand_ini = torch.rand(f0.shape[0], 1, self.dim, device=f0.device)
    rand_ini = rand_ini * (self.harmonic_order > 1).to(rand_ini.dtype)

    tmp_over_one = torch.cumsum(rad_values, 1) % 1
    wrapped = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
    cumsum_shift = F.pad(-wrapped.to(rad_values.dtype), (0, 0, 1, 0))
    phase = torch.cumsum(rad_values + cumsum_shift, dim=1)

    phase = phase * self.harmonic_order + rand_ini
    return torch.sin(phase * (2.0 * math.pi))


def _resize_index(in_len: int, out_len: int, mode: str, device):
    """``F.interpolate``'s source indices (align_corners=False), built in FP32 on the CPU like PyTorch's."""
    key = (in_len, out_len, mode, str(device))
    if key not in _RESIZE_INDEX:
        scale = torch.tensor(in_len / out_len, dtype=torch.float32)
        dst = torch.arange(out_len, dtype=torch.float32)
        if mode == "nearest":
            index = torch.floor(dst * scale).long().clamp_max(in_len - 1)
            _RESIZE_INDEX[key] = (index.to(device),)
        else:
            src = ((dst + 0.5) * scale - 0.5).clamp_min(0.0)
            low = src.long().clamp_max(in_len - 1)
            high = torch.where(low < in_len - 1, low + 1, low)
            weight = src - low.float()
            _RESIZE_INDEX[key] = (low.to(device), high.to(device), weight.to(device))
    return _RESIZE_INDEX[key]


def _resize(x, length: int, mode: str):
    index = _resize_index(int(x.shape[-1]), int(length), mode, x.device)
    if mode == "nearest":
        return x.index_select(-1, index[0])
    low, high, weight = index
    weight = weight.to(x.dtype)
    return x.index_select(-1, low) * (1.0 - weight) + x.index_select(-1, high) * weight


def expand_f0(f0: torch.Tensor, length: int) -> torch.Tensor:
    """``commons.expand_f0`` with the interpolations written as gathers."""
    voiced = (f0 > 0).to(f0.dtype)
    log_f0 = torch.log(f0.clamp_min(1.0)) * voiced
    weight = _resize(voiced, length, "linear")
    log_f0 = _resize(log_f0, length, "linear")
    log_f0 = log_f0 / weight.clamp_min(1e-6)
    gate = _resize(voiced, length, "nearest")
    return torch.exp(log_f0) * gate


def _make_spectrogram_torch(original):
    """``mel_processing.spectrogram_torch``; CPU tensors (the data loader) keep the original."""
    windows = {}

    def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
        if y.device.type == "cpu":
            return original(y, n_fft, hop_size, win_size, center)
        key = (int(win_size), str(y.device))
        if key not in windows:
            windows[key] = torch.hann_window(win_size).to(y.device)
        pad = int((n_fft - hop_size) / 2)
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        real, imag = stft_parts(y, n_fft, hop_size, win_size, windows[key], center)
        with torch.autocast(device_type=y.device.type, enabled=False):
            return torch.sqrt(real.pow(2) + imag.pow(2) + 1e-6)

    spectrogram_torch._xla_original = original
    return spectrogram_torch


def _nsf_sine_forward(self, f0, upsampling_factor):
    """``hifigan_nsf.SineGenerator.forward`` with the voiced mask upsampled by a gather."""
    with torch.autocast(device_type=f0.device.type, enabled=False), torch.no_grad():
        f0 = f0.float().unsqueeze(-1)
        sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
        voiced_mask = self._compute_voiced_unvoiced(f0).transpose(2, 1)
        voiced_mask = _resize(
            voiced_mask, voiced_mask.shape[-1] * int(upsampling_factor), "nearest"
        ).transpose(2, 1)
        noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
            self.sine_amplitude / 3
        )
        noise = noise_amplitude * torch.randn_like(sine_waves)
        sine_waveforms = sine_waves * voiced_mask + noise
    return sine_waveforms, voiced_mask, noise


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels: int):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width):
    """torchaudio's, with the right pad trimmed so the strided conv needs no crop.

    The TPU compiler folds that crop into the conv as a negative pad and aborts
    (``fusion_emitter.cc: window.pad_low[output_dim] == 0``).  The windows the
    shorter pad drops only covered padding zeros, so the output is identical.
    """
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    length = int(waveform.shape[-1])
    waveform = F.pad(waveform, (width, width + orig_freq - 1))
    resampled = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(waveform.shape[0], -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    if resampled.shape[-1] != target_length:
        resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def _rmvpe_mel_forward(self, audio, keyshift=0, speed=1, center=True):
    """``RMVPE.MelSpectrogram.forward`` for the unshifted extraction path."""
    if keyshift != 0 or speed != 1:
        raise NotImplementedError("The TPU mel only covers keyshift=0, speed=1.")
    key = f"0_{audio.device}"
    if key not in self.hann_window:
        self.hann_window[key] = torch.hann_window(self.win_length).to(audio.device)
    real, imag = stft_parts(
        audio, self.n_fft, self.hop_length, self.win_length, self.hann_window[key], center
    )
    with torch.autocast(device_type=audio.device.type, enabled=False):
        magnitude = torch.sqrt(real.pow(2) + imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


def apply_extract() -> None:
    from rvc.lib.predictors import RMVPE

    RMVPE.MelSpectrogram.forward = _rmvpe_mel_forward


def apply() -> None:
    global _ORIGINAL_POLYPHASE, _ORIGINAL_SINC_KERNEL
    from rvc.lib.algorithm import resampling, wavenet
    from rvc.lib.algorithm.discriminators.multi import mpd_msd_combined
    from rvc.lib.algorithm.discriminators.single import univhd
    from rvc.lib.algorithm.generators import hifigan_nsf, refinegan2
    from rvc.train import mel_processing

    mpd_msd_combined.DiscriminatorR.spectrogram = _mrd_spectrogram
    univhd.UnivHDDiscriminator.spectrogram = _univhd_spectrogram
    modules = [mel_processing]
    # The trainer's modules also import it script-relative, as a second module.
    try:
        import mel_processing as script_mel_processing

        modules.append(script_mel_processing)
    except ImportError:
        pass
    for module in modules:
        module.MultiScaleMelSpectrogramLoss.mel_spectrogram = _ms_mel_spectrogram
        if not hasattr(module.spectrogram_torch, "_xla_original"):
            module.spectrogram_torch = _make_spectrogram_torch(module.spectrogram_torch)
    hifigan_nsf.SineGenerator.forward = _nsf_sine_forward
    resampling.AntiAliasedUpsample1d.forward = _upsample_forward
    if _ORIGINAL_POLYPHASE is None:
        _ORIGINAL_POLYPHASE = resampling.AntiAliasedUpsample1d._polyphase
        _ORIGINAL_SINC_KERNEL = refinegan2._get_sinc_resample_kernel
    resampling.AntiAliasedUpsample1d._polyphase = _polyphase
    refinegan2._get_sinc_resample_kernel = _get_sinc_resample_kernel
    refinegan2.SineGenerator._f02sine = _f02sine
    refinegan2.expand_f0 = expand_f0
    refinegan2._apply_sinc_resample_kernel = _apply_sinc_resample_kernel
    wavenet.fused_add_tanh_sigmoid_multiply = fused_add_tanh_sigmoid_multiply
