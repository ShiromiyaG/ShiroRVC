import math
from typing import Optional

import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint
import numpy as np
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock


class SineGenerator(torch.nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super(SineGenerator, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1  # fundamental + harmonics

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        uv_mask = (f0 > self.voiced_threshold).float()
        return uv_mask

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):

        batch_size, length, _ = f0.shape

        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )

        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += torch.nn.functional.pad(
            cumulative_phase, (0, 0, 1, 0), mode="constant"
        )

        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        # Both of these are provably the identity when there are no overtones,
        # which is the only way this module is ever built: ``SourceModuleHnNSF``
        # is constructed with ``harmonic_num=0``, so ``waveform_dim`` is 1,
        # ``harmonic_scale`` is exactly ``[1.0]`` and ``random_phase`` is
        # exactly ``[0.0]`` -- its one element is the fundamental, which the
        # next line zeroes.  Skipping them drops two full-tensor elementwise
        # kernels over the output-rate tensor plus an RNG draw, per forward.
        # The deterministic part of the output is bit-identical -- verified
        # against the previous revision with the noise neutralised.  The output
        # as a whole is not, and cannot be: dropping the ``torch.rand`` draw
        # shifts the RNG stream, so the additive noise differs by ~1e-4.  That
        # is a reseed, not a change of behaviour.  Kept behind the branch rather
        # than deleted so raising ``harmonic_num`` still works.
        if self.waveform_dim > 1:
            harmonic_scale = torch.arange(
                1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
            ).reshape(1, 1, -1)
            phase_increments *= harmonic_scale

            random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
            random_phase[..., 0] = 0  # fundamental has no random offset
            phase_increments += random_phase

        sine_waves = torch.sin(2 * np.pi * phase_increments)
        return sine_waves

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        # Phase accumulation must not run in a reduced precision.  Under
        # ``autocast(fp16)`` this block happens to stay FP32 today only because
        # none of ``cumsum``/``fmod``/``sin``/``interpolate`` are on autocast's
        # op list and ``f0`` arrives as FP32 -- an incidental property, not a
        # guarantee.  Adding any autocast-listed op (a conv, a matmul) anywhere
        # in here would silently drop the running phase to an 11-bit mantissa,
        # and the failure mode is a detuned harmonic source rather than a NaN,
        # so neither the GradScaler nor any loss would flag it.  The fence and
        # the explicit cast make the FP32 guarantee structural; RefineGAN's
        # ``BandLimitedNSFSource.prepare`` states it the same way with its own
        # ``f0.float()``.
        with torch.autocast(device_type=f0.device.type, enabled=False), torch.no_grad():
            f0 = f0.float().unsqueeze(-1)

            sine_waves = (
                self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            )

            voiced_mask = self._compute_voiced_unvoiced(f0)

            voiced_mask = torch.nn.functional.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)

            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
                self.sine_amplitude / 3
            )

            noise = noise_amplitude * torch.randn_like(sine_waves)

            sine_waveforms = sine_waves * voiced_mask + noise

        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(torch.nn.Module):
    """Harmonic-plus-noise excitation source used by neural vocoders."""

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class HiFiGANNSFGenerator(torch.nn.Module):
    """NSF-HiFiGAN: filters a harmonic+noise excitation source through upsampling/residual blocks."""

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        super(HiFiGANNSFGenerator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        # ``f0_upsamp`` used to live here.  It was never called: ``SineGenerator``
        # takes the upsampling factor as an argument and expands f0 itself.
        # ``nn.Upsample`` holds no parameters and no buffers, so dropping it
        # leaves the state dict byte-identical and every RVC v2 pretrain loads
        # exactly as before.
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            # Slice the stage's blocks out once.  This used to walk all
            # ``num_upsamples * num_kernels`` blocks per stage and filter with
            # ``j in range(...)``, so 48 Python iterations did the work of 12.
            # Same blocks, same order.
            stage_resblocks = self.resblocks[
                i * self.num_kernels : (i + 1) * self.num_kernels
            ]
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = x + noise_convs(har_source)
                xs = sum([
                    checkpoint(resblock, x, use_reentrant=False)
                    for resblock in stage_resblocks])
            else:
                x = ups(x)
                x = x + noise_convs(har_source)
                xs = sum([resblock(x) for resblock in stage_resblocks])
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        return self
