"""Lightweight anti-aliased NSF generator used by ChouwaGAN."""

import math
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint


def _safe_pad(x: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return x
    mode = "reflect" if x.shape[-1] > padding else "replicate"
    return F.pad(x, (padding, padding), mode=mode)


def _lowpass_kernel(factor: int, width: int, rolloff: float) -> Tensor:
    half = max(1, int(width) * max(1, int(factor)))
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = 0.5 * float(rolloff) / max(1, int(factor))
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)
    kernel = kernel * torch.kaiser_window(
        kernel.numel(), periodic=False, beta=14.0, dtype=kernel.dtype
    )
    return (kernel / kernel.sum()).view(1, 1, -1)


class FixedLowPass1d(nn.Module):
    def __init__(
        self,
        factor: int,
        width: int = 4,
        rolloff: float = 0.95,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = int(stride)
        self.register_buffer(
            "kernel", _lowpass_kernel(factor, width, rolloff), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        kernel = kernel.expand(x.shape[1], -1, -1)
        padding = (kernel.shape[-1] - 1) // 2
        return F.conv1d(
            _safe_pad(x, padding),
            kernel,
            stride=self.stride,
            groups=x.shape[1],
        )


class AntiAliasedUpsample1d(nn.Module):
    def __init__(self, factor: int, filter_width: int, rolloff: float):
        super().__init__()
        self.factor = int(factor)
        kernel = _lowpass_kernel(self.factor, filter_width, rolloff)
        self.register_buffer("kernel", kernel, persistent=False)

        kernel_size = int(kernel.shape[-1])
        self.pad = kernel_size // self.factor - 1
        self.pad_left = (
            self.pad * self.factor + (kernel_size - self.factor) // 2
        )
        self.pad_right = (
            self.pad * self.factor + (kernel_size - self.factor + 1) // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        channels = x.shape[1]
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        kernel = kernel.expand(channels, -1, -1)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.factor * F.conv_transpose1d(
            x,
            kernel,
            stride=self.factor,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class AntiAliasedSnakeBeta(nn.Module):
    """SnakeBeta surrounded by a 2x low-pass resampling pair."""

    def __init__(self, channels: int, rolloff: float = 0.95):
        super().__init__()
        initial = math.log(math.expm1(1.0))
        self.alpha = nn.Parameter(torch.full((channels,), initial))
        self.beta = nn.Parameter(torch.full((channels,), initial))
        self.upsample = AntiAliasedUpsample1d(2, filter_width=4, rolloff=rolloff)
        self.downsample = FixedLowPass1d(
            2, width=4, rolloff=rolloff, stride=2
        )

    def forward(self, x: Tensor) -> Tensor:
        original_length = x.shape[-1]
        x = self.upsample(x)

        alpha = F.softplus(self.alpha).view(1, -1, 1) + 1e-4
        beta = F.softplus(self.beta).view(1, -1, 1) + 1e-4
        x = x + torch.sin(alpha * x).square() / beta

        x = self.downsample(x)
        if x.shape[-1] > original_length:
            x = x[..., :original_length]
        elif x.shape[-1] < original_length:
            x = F.pad(x, (0, original_length - x.shape[-1]), mode="replicate")
        return x


class BandLimitedNSFSource(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_count: int = 7,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.harmonic_count = int(harmonic_count)
        self.noise_std = float(noise_std)

    def prepare(self, f0: Tensor, length: int):
        if f0 is None:
            raise ValueError("ChouwaGAN requires an F0 sequence.")
        if f0.ndim == 3:
            f0 = f0.squeeze(-1)
        if f0.ndim != 2:
            raise ValueError("F0 must have shape [batch, frames].")

        f0 = f0.float().clamp_min(0.0)
        voiced_frames = (f0 > 0).float()
        f0 = F.interpolate(
            f0.unsqueeze(1), size=length, mode="linear", align_corners=False
        ).squeeze(1)
        voiced = F.interpolate(
            voiced_frames.unsqueeze(1), size=length, mode="nearest"
        ).squeeze(1)
        f0 = f0 * voiced

        phase = torch.cumsum(
            2.0 * math.pi * f0 / self.sample_rate,
            dim=-1,
        )
        harmonics = torch.arange(
            1,
            self.harmonic_count + 1,
            device=f0.device,
            dtype=f0.dtype,
        )
        phase_offset = torch.zeros(
            f0.shape[0],
            1,
            self.harmonic_count,
            device=f0.device,
            dtype=f0.dtype,
        )
        if self.training and self.harmonic_count > 1:
            phase_offset[..., 1:] = (
                torch.rand_like(phase_offset[..., 1:]) * (2.0 * math.pi)
            )
        harmonic_wave = torch.sin(
            phase.unsqueeze(-1) * harmonics.view(1, 1, -1) + phase_offset
        )
        noise = torch.randn_like(f0)
        return f0, voiced, harmonic_wave, noise

    def render(
        self,
        state,
        length: int,
        sample_rate: float,
    ) -> Tensor:
        f0, voiced, harmonic_wave, noise = state
        full_length = int(f0.shape[-1])
        if full_length % int(length):
            raise ValueError

        factor = full_length // int(length)
        offset = (factor - 1) // 2
        indices = torch.arange(
            int(length), device=f0.device, dtype=torch.long
        ) * factor + offset
        f0 = f0.index_select(-1, indices)
        voiced = voiced.index_select(-1, indices)
        harmonic_wave = harmonic_wave.index_select(1, indices)
        noise = noise.index_select(-1, indices)

        harmonics = torch.arange(
            1,
            self.harmonic_count + 1,
            device=f0.device,
            dtype=f0.dtype,
        )
        harmonic_frequency = f0.unsqueeze(-1) * harmonics.view(1, 1, -1)
        transition = max(1.0, float(sample_rate) * 0.02)
        nyquist_mask = torch.sigmoid(
            (float(sample_rate) * 0.48 - harmonic_frequency) / transition
        )
        amplitude = harmonics.rsqrt().view(1, 1, -1)
        harmonic_wave = (harmonic_wave * amplitude * nyquist_mask).sum(dim=-1)
        normalizer = (amplitude.square() * nyquist_mask.square()).sum(dim=-1).sqrt()
        harmonic_wave = harmonic_wave / normalizer.clamp_min(1e-4)

        noise_level = self.noise_std * (0.25 * voiced + 1.0 - voiced)
        source = harmonic_wave * voiced + noise * noise_level
        return source.unsqueeze(1)

    def forward(
        self,
        f0: Tensor,
        length: int,
        sample_rate: float,
    ) -> Tensor:
        state = self.prepare(f0, length)
        return self.render(state, length, sample_rate)


class DepthwiseSeparableUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int,
        rolloff: float,
    ):
        super().__init__()
        hidden = int(channels * expansion)
        self.activation = AntiAliasedSnakeBeta(channels, rolloff)
        self.expand = weight_norm(nn.Conv1d(channels, hidden, 1))
        self.depthwise = weight_norm(
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size,
                padding=((kernel_size - 1) * dilation) // 2,
                dilation=dilation,
                groups=hidden,
            )
        )
        self.project = weight_norm(nn.Conv1d(hidden, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.activation(x)
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return residual + x


class LiteAMPBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int,
        dilations: Sequence[int],
        expansion: int,
        rolloff: float,
    ):
        super().__init__()
        self.units = nn.ModuleList(
            [
                DepthwiseSeparableUnit(
                    channels,
                    int(kernel),
                    int(dilation),
                    expansion,
                    rolloff,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.units:
            x = unit(x)
        return x


class LiteAMPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernels: Iterable[int],
        dilations: Sequence[int],
        expansion: int,
        rolloff: float,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                LiteAMPBranch(
                    channels,
                    int(kernel),
                    dilations,
                    expansion,
                    rolloff,
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.branches[0](x)
        for branch in self.branches[1:]:
            output = output + branch(x)
        return output / len(self.branches)


class ChouwaGANGenerator(nn.Module):
    """Compact full-band anti-aliased NSF generator for 44.1 kHz audio."""

    def __init__(
        self,
        initial_channel: int,
        gin_channels: int,
        sr: int,
        upsample_rates: Sequence[int],
        upsample_initial_channel: int = 320,
        checkpointing: bool = False,
        chouwagan_channels: Sequence[int] = (256, 160, 80, 40),
        chouwagan_block_kernels: Sequence[Sequence[int]] = (
            (3, 7),
            (3, 7),
            (7,),
            (7,),
        ),
        chouwagan_dilations: Sequence[int] = (1, 3, 5),
        chouwagan_expansion: int | Sequence[int] = (2, 2, 1, 1),
        chouwagan_harmonics: int = 7,
        chouwagan_noise_std: float = 0.01,
        chouwagan_filter_width: int = 4,
        chouwagan_rolloff: float = 0.95,
        chouwagan_latent_mixer_kernel: int = 7,
        chouwagan_latent_mixer_dilations: Sequence[int] = (1, 3),
        chouwagan_latent_mixer_expansion: int = 2,
        **_: object,
    ):
        super().__init__()
        if int(sr) != 44100:
            raise ValueError("ChouwaGAN only supports 44.1 kHz configurations.")

        self.sr = int(sr)
        self.checkpointing = bool(checkpointing)
        self.upsample_rates = tuple(int(value) for value in upsample_rates)
        self.total_upsample = math.prod(self.upsample_rates)
        self.channels = tuple(int(value) for value in chouwagan_channels)
        if len(self.channels) != len(self.upsample_rates):
            raise ValueError("ChouwaGAN channel and upsample schedules must match.")
        if isinstance(chouwagan_expansion, int):
            stage_expansions = (int(chouwagan_expansion),) * len(self.channels)
        else:
            stage_expansions = tuple(int(value) for value in chouwagan_expansion)
        if len(stage_expansions) != len(self.channels):
            raise ValueError("ChouwaGAN expansion and upsample schedules must match.")
        if any(value < 1 for value in stage_expansions):
            raise ValueError("ChouwaGAN expansion values must be positive.")

        self.source = BandLimitedNSFSource(
            sample_rate=self.sr,
            harmonic_count=chouwagan_harmonics,
            noise_std=chouwagan_noise_std,
        )
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, int(upsample_initial_channel), 5, padding=2)
        )
        self.cond = (
            weight_norm(nn.Conv1d(gin_channels, int(upsample_initial_channel), 1))
            if gin_channels
            else None
        )
        self.latent_mixer = nn.Sequential(
            *(
                DepthwiseSeparableUnit(
                    int(upsample_initial_channel),
                    int(chouwagan_latent_mixer_kernel),
                    int(dilation),
                    int(chouwagan_latent_mixer_expansion),
                    chouwagan_rolloff,
                )
                for dilation in chouwagan_latent_mixer_dilations
            )
        )

        self.ups = nn.ModuleList()
        self.source_projections = nn.ModuleList()
        source_gate = math.log(0.25 / 0.75)
        self.source_gates = nn.Parameter(
            torch.full((len(self.channels),), source_gate)
        )
        self.blocks = nn.ModuleList()
        current_channels = int(upsample_initial_channel)
        for index, (factor, channels, expansion) in enumerate(
            zip(
                self.upsample_rates,
                self.channels,
                stage_expansions,
                strict=True,
            )
        ):
            self.ups.append(
                nn.Sequential(
                    AntiAliasedUpsample1d(
                        factor,
                        filter_width=chouwagan_filter_width,
                        rolloff=chouwagan_rolloff,
                    ),
                    weight_norm(nn.Conv1d(current_channels, channels, 1)),
                )
            )
            self.source_projections.append(
                weight_norm(nn.Conv1d(1, channels, 1, bias=False))
            )
            self.blocks.append(
                LiteAMPBlock(
                    channels,
                    chouwagan_block_kernels[index],
                    chouwagan_dilations,
                    expansion,
                    chouwagan_rolloff,
                )
            )
            current_channels = channels

        self.post_activation = AntiAliasedSnakeBeta(current_channels, chouwagan_rolloff)
        self.conv_post = weight_norm(nn.Conv1d(current_channels, 1, 7, padding=3))
        self.output_upsample = AntiAliasedUpsample1d(
            2, filter_width=4, rolloff=chouwagan_rolloff
        )
        self.output_downsample = FixedLowPass1d(
            2, width=4, rolloff=chouwagan_rolloff, stride=2
        )

    def _source_for_stage(self, state, stage: int, length: int) -> Tensor:
        prefix = math.prod(self.upsample_rates[: stage + 1])
        remaining = self.total_upsample // prefix
        stage_rate = self.sr / remaining
        return self.source.render(state, length, stage_rate)

    def forward(
        self,
        x: Tensor,
        f0: Tensor,
        g: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.conv_pre(x)
        if self.cond is not None and g is not None:
            x = x + self.cond(g)
        x = self.latent_mixer(x)

        output_length = x.shape[-1] * self.total_upsample
        with torch.no_grad():
            source_state = self.source.prepare(f0, output_length)

        source_gates = torch.sigmoid(self.source_gates)
        for index, (upsample, source_projection, block) in enumerate(
            zip(self.ups, self.source_projections, self.blocks, strict=True)
        ):
            x = upsample(x)
            with torch.no_grad():
                source = self._source_for_stage(source_state, index, x.shape[-1])
            source = source_projection(source.to(dtype=x.dtype))
            x = x + source_gates[index].to(dtype=x.dtype) * source
            if self.training and self.checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.post_activation(x)
        x = self.conv_post(x)

        x = self.output_upsample(x)
        x = torch.tanh(x)
        return self.output_downsample(x)

    def remove_weight_norm(self):
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self
