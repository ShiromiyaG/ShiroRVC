"""RefineGAN generator, following the paper rather than the Applio port.

Reference: Xu et al., *RefineGAN: Universally Generating Waveform Better than
Ground Truth with Highly Accurate Pitch and Intensity Responses* (2021).

The paper's generator is a U-Net over a **pitch template**, not a sine-excited
upsampler.  Three things follow from that, and all three are where the Applio
port departs from it:

1. **The template is a pulse train, not a harmonic sine bank.**  Section 2.2
   builds the template directly from F0 and a frame-level intensity envelope:
   a single-sample impulse at every pitch period in voiced regions, and
   *uniform* noise in unvoiced ones, both scaled by the intensity.  A one-sample
   impulse already carries every harmonic at equal amplitude, which is the whole
   reason the paper needs no harmonic count and no learned merge layer.  Applio
   substitutes NSF's ``SineGenerator`` -- a band-limited sum of ``harmonic_num``
   sines collapsed by a learned ``Linear`` -- so the excitation arrives with the
   harmonic structure already chosen for it, and the refinement network spends
   its capacity undoing that choice.

2. **The encoder downsamples with strided convolutions and ResBlocks.**  Each
   stage in the paper is ``Conv1d(stride=rate)`` followed by a ResBlock, and the
   skip taken into the decoder is the *pre-downsample* activation.  Applio
   downsamples with ``torchaudio.functional.resample`` -- a fixed sinc
   filter -- followed by a stride-1 convolution, so the encoder has no residual
   depth at any resolution and the skips carry a resampled template rather than
   learned features.

3. **The decoder upsamples with transposed convolutions.**  Applio uses
   ``nn.Upsample(mode="linear")``, whose fixed linear interpolation cannot
   synthesise anything above the previous stage's band; the learned kernels that
   the paper relies on to *generate* high-frequency detail are simply absent.

Everything else here is the paper: ``ParallelResBlock`` averages multi-kernel
ResBlocks wrapped in the noise-injection layer the paper calls AdaIN, and the
output head is a plain ``tanh``.

FP16 notes.  Nothing below changes the architecture for mixed precision, but the
operations that FP16 cannot represent are pinned to FP32 and cast back: the
phase accumulator (a cumulative sum over ``T * hop_length`` samples), the
intensity envelope's exponential, and the F0 interpolation.  The convolutional
stack itself runs in whatever dtype the caller's autocast selects.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _remove_weight_norm(module: nn.Module) -> None:
    """Idempotent ``remove_weight_norm``; export paths may run twice."""
    if hasattr(module, "parametrizations") and hasattr(
        module.parametrizations, "weight"
    ):
        remove_parametrizations(module, "weight", leave_parametrized=True)


def _resample_padding(rate: int) -> int:
    """Padding that makes a ``kernel=2*rate, stride=rate`` conv exactly ``1/rate``.

    The published implementations use ``rate // 2``, which is only exact for even
    rates.  This fork runs at 44.1 kHz with ``hop_length=441 = 3*3*7*7``, so every
    stage rate here is odd and ``rate // 2`` would drop a frame per stage --
    silently, since the decoder would still produce *a* waveform, just not one
    whose length matches the target.  ``(rate + 1) // 2`` satisfies the exactness
    condition ``rate/2 <= padding < rate`` for odd rates and reduces to ``rate/2``
    for even ones, so it is the same filter wherever the original applies.
    """
    return (int(rate) + 1) // 2


class ResBlock(nn.Module):
    """Dilated residual stack, kernel ``k``, dilations ``(1, 3, 5)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: Sequence[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        self.in_channels if index == 0 else self.out_channels,
                        self.out_channels,
                        kernel_size,
                        stride=1,
                        dilation=value,
                        padding=get_padding(kernel_size, value),
                    )
                )
                for index, value in enumerate(dilation)
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        self.out_channels,
                        self.out_channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs1.apply(self._init_weights)
        self.convs2.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        for index, (first, second) in enumerate(zip(self.convs1, self.convs2)):
            value = F.leaky_relu(x, self.leaky_relu_slope)
            value = first(value)
            value = F.leaky_relu(value, self.leaky_relu_slope)
            value = second(value)
            # The first sub-block is the one allowed to change width, so it is
            # the only one that cannot close its residual.
            x = value + x if index != 0 or self.in_channels == self.out_channels else value
        return x

    def remove_weight_norm(self) -> None:
        for first, second in zip(self.convs1, self.convs2):
            _remove_weight_norm(first)
            _remove_weight_norm(second)


class AdaIN(nn.Module):
    """Per-channel Gaussian noise injection, then a leaky ReLU.

    The paper calls this AdaIN; the operation it describes is noise injection
    with a learned per-channel gain, not adaptive instance normalization.  The
    gain starts small so the deterministic refinement path dominates early
    training without the stochastic branch being unreachable.
    """

    def __init__(self, channels: int, leaky_relu_slope: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(channels)) * 1e-4)
        self.activation = nn.LeakyReLU(float(leaky_relu_slope))

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x) * self.weight[None, :, None]
        return self.activation(x + noise)


class ParallelResBlock(nn.Module):
    """Average of ResBlocks at several kernel sizes, each wrapped in AdaIN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (3, 7, 11),
        dilation: Sequence[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.input_conv = weight_norm(
            nn.Conv1d(int(in_channels), int(out_channels), 7, stride=1, padding=3)
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(out_channels, leaky_relu_slope),
                    ResBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(out_channels, leaky_relu_slope),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_conv(x)
        total = None
        for block in self.blocks:
            value = block(x)
            total = value if total is None else total + value
        return total / len(self.blocks)

    def remove_weight_norm(self) -> None:
        _remove_weight_norm(self.input_conv)
        for block in self.blocks:
            block[1].remove_weight_norm()


class PulseTemplate(nn.Module):
    """The speech template of section 2.2: pitch pulses plus unvoiced noise.

    Voiced regions carry a one-sample impulse at every pitch period, scaled by
    the frame intensity.  Unvoiced regions carry uniform noise -- uniform, not
    Gaussian, which is what the paper specifies -- at a third of that amplitude.

    The module has no parameters.  That is the point: the excitation is exactly
    determined by F0 and intensity, so the pitch response of the vocoder is a
    property of the template rather than something the network has to learn to
    respect.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        wave_amp: float = 0.1,
        noise_scale: float = 1.0 / 3.0,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.wave_amp = float(wave_amp)
        self.noise_scale = float(noise_scale)
        self.voiced_threshold = float(voiced_threshold)

    @torch.no_grad()
    def forward(self, f0: Tensor, amplitude: Optional[Tensor] = None) -> Tensor:
        """``f0``: ``(batch, 1, samples)``.  Returns ``(batch, 1, samples)``.

        Runs in FP32 regardless of autocast: the phase accumulator is a
        ``cumsum`` over the full waveform length, and in FP16 its increments
        (``f0 / sample_rate``, order 1e-3) stop registering against the
        accumulated value within a few hundred samples, which turns the pulse
        train into a slowly drifting one.
        """
        output_dtype = f0.dtype
        f0 = f0.float().clamp_min(0.0)
        voiced = f0 > self.voiced_threshold

        if amplitude is None:
            amplitude = torch.full_like(f0, self.wave_amp)
        else:
            amplitude = F.interpolate(
                amplitude.float(),
                size=f0.shape[-1],
                mode="linear",
                align_corners=False,
            ).clamp_min(0.0)

        # A pulse lands wherever the accumulated phase crosses an integer.
        # Re-zeroing the phase at each voiced onset keeps a phrase from opening
        # mid-period with whatever fraction the previous voiced region left.
        previous_voiced = torch.zeros_like(voiced)
        previous_voiced[..., 1:] = voiced[..., :-1]
        onset = voiced & ~previous_voiced
        increment = torch.where(voiced, f0 / self.sample_rate, torch.zeros_like(f0))
        phase = torch.cumsum(increment, dim=-1)
        onset_phase = torch.where(onset, phase, torch.zeros_like(phase))
        phase = phase - torch.cummax(onset_phase, dim=-1).values
        previous_phase = F.pad(phase[..., :-1], (1, 0))
        wrapped = torch.floor(phase) > torch.floor(previous_phase)
        pulses = (wrapped | onset).to(amplitude.dtype) * amplitude

        noise = (torch.rand_like(pulses) * 2.0 - 1.0) * amplitude * self.noise_scale
        return torch.where(voiced, pulses, noise).to(output_dtype)


def _per_stage(value, stages: int, what: str) -> Tuple[Tuple[int, ...], ...]:
    """Normalise a kernel or dilation setting into one set per decoder stage.

    A flat sequence -- ``(3, 7, 11)`` -- applies to every stage, which is the
    paper's decoder and stays the default.  A nested one gives each stage its
    own set, because neither a kernel nor a dilation means the same thing at
    every stage.

    What they buy is *time span*, and each stage runs at a different rate.  With
    ``hop_length=441`` and dilations up to 5, one ResBlock branch reaches:

    ==========  =========  ==========  ==========  ==========
    stage       rate       ``k=3``     ``k=7``     ``k=11``
    ==========  =========  ==========  ==========  ==========
    0           300 Hz     36.6 ms     103 ms      170 ms
    1           900 Hz     12.2 ms     34 ms       57 ms
    2           6.3 kHz    1.75 ms     4.9 ms      8.1 ms
    3           44.1 kHz   0.25 ms     0.70 ms     1.16 ms
    ==========  =========  ==========  ==========  ==========

    At stage 0 the three kernels cover phoneme, syllable and phrase scale and are
    genuinely doing different jobs.  At stage 3 all three are sub-millisecond and
    differ only in how much fine texture they see, while that stage carries 56%
    of the decoder's activation memory and 34% of its time.
    """
    values = list(value)
    if values and isinstance(values[0], (list, tuple)):
        schedule = tuple(tuple(int(k) for k in stage) for stage in values)
        if len(schedule) != stages:
            raise ValueError(
                f"RefineGAN {what} schedule has {len(schedule)} entries for "
                f"{stages} decoder stages"
            )
    else:
        schedule = (tuple(int(k) for k in values),) * stages
    if any(not stage for stage in schedule):
        raise ValueError(f"Every RefineGAN decoder stage needs at least one {what}")
    return schedule


def _kernel_schedule(value, stages: int) -> Tuple[Tuple[int, ...], ...]:
    return _per_stage(value, stages, "kernel")


def _decoder_channels(value, bottleneck: int, stages: int) -> Tuple[int, ...]:
    """Channel width per decoder stage; ``None`` reproduces the paper's halving.

    The halving is what makes RefineGAN slow on this hardware, and the reason is
    measurable rather than aesthetic.  A ``conv1d`` reaches 26% of an RTX 5060's
    TF32 throughput at 32 channels and 79% at 256, because a 32x32 matrix per
    output position is too small a GEMM to fill the card -- so the final stage,
    which the halving leaves at 32 channels over 17640 samples, is simultaneously
    the most expensive stage and the least efficient one.

    Spending the same budget on *width* instead of on three parallel narrow
    branches measured, at batch 8: 33% more arithmetic, 24% less wall time, 17%
    less activation memory and 1.76x the achieved throughput.  Hence the shipped
    schedule stops halving at the top; see ``rvc/configs/refinegan/44100.json``.
    """
    if value is None:
        return tuple(bottleneck // (2 ** (index + 1)) for index in range(stages))
    channels = tuple(int(width) for width in value)
    if len(channels) != stages:
        raise ValueError(
            f"RefineGAN decoder channels have {len(channels)} entries for "
            f"{stages} decoder stages"
        )
    if any(width < 1 for width in channels):
        raise ValueError("RefineGAN decoder channels must be positive")
    return channels


class RefineGANGenerator(nn.Module):
    """U-Net refinement of a pitch template, conditioned on the latent sequence.

    ``initial_channel`` takes the place of the paper's ``num_mels``: this fork
    conditions on the frame-rate latent produced by the VITS frontend rather
    than on a mel spectrogram.  Nothing else about the conditioning path
    changes -- it is still a single ``Conv1d`` into the U-Net bottleneck.
    """

    def __init__(
        self,
        initial_channel: int,
        gin_channels: int,
        sr: int,
        upsample_rates: Sequence[int],
        checkpointing: bool = False,
        refinegan_start_channels: int = 16,
        refinegan_leaky_relu_slope: float = 0.2,
        refinegan_resblock_kernel_sizes: Sequence = (3, 7, 11),
        refinegan_resblock_dilations: Sequence = (1, 3, 5),
        refinegan_decoder_channels: Optional[Sequence[int]] = None,
        refinegan_encoder_kernel_size: int = 7,
        refinegan_wave_amp: float = 0.1,
        **_: object,
    ):
        super().__init__()

        self.sr = int(sr)
        self.checkpointing = bool(checkpointing)
        self.leaky_relu_slope = float(refinegan_leaky_relu_slope)
        self.upsample_rates = tuple(int(value) for value in upsample_rates)
        # The encoder walks the same rates in reverse, so stage *i* of the
        # decoder always meets the skip taken at its own resolution.
        self.downsample_rates = tuple(reversed(self.upsample_rates))
        self.hop_length = math.prod(self.upsample_rates)
        self.initial_channel = int(initial_channel)

        self.template_gen = PulseTemplate(
            sample_rate=self.sr, wave_amp=float(refinegan_wave_amp)
        )
        self.template_conv = weight_norm(
            nn.Conv1d(1, int(refinegan_start_channels), 7, stride=1, padding=3)
        )

        stages = len(self.upsample_rates)
        self.kernel_schedule = _kernel_schedule(refinegan_resblock_kernel_sizes, stages)
        self.dilation_schedule = _per_stage(
            refinegan_resblock_dilations, stages, "dilation"
        )

        channels = int(refinegan_start_channels)
        self.downsample_blocks = nn.ModuleList()
        # Width of the pre-downsample activation each encoder stage hands to the
        # decoder.  Recorded rather than re-derived, because with an explicit
        # decoder schedule the skip is no longer half the stage width.
        skip_channels = []
        for rate in self.downsample_rates:
            skip_channels.append(channels)
            new_channels = channels * 2
            self.downsample_blocks.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            new_channels,
                            kernel_size=rate * 2,
                            stride=rate,
                            padding=_resample_padding(rate),
                        )
                    ),
                    ResBlock(
                        new_channels,
                        new_channels,
                        kernel_size=int(refinegan_encoder_kernel_size),
                        dilation=self.dilation_schedule[0],
                        leaky_relu_slope=self.leaky_relu_slope,
                    ),
                )
            )
            channels = new_channels

        self.conditioning_conv = weight_norm(
            nn.Conv1d(self.initial_channel, channels, 7, stride=1, padding=3)
        )
        self.cond = (
            weight_norm(nn.Conv1d(int(gin_channels), channels, 1))
            if gin_channels
            else None
        )
        # Bottleneck holds the encoder output and the conditioning side by side.
        channels *= 2

        self.upsample_blocks = nn.ModuleList()
        self.upsample_conv_blocks = nn.ModuleList()
        self.decoder_channels = _decoder_channels(
            refinegan_decoder_channels, channels, stages
        )
        self.skip_channels: list[int] = []
        for index, rate in enumerate(self.upsample_rates):
            new_channels = self.decoder_channels[index]
            # Stage *i* meets the skip taken at its own resolution, which the
            # encoder produced in reverse order.
            skip = skip_channels[stages - 1 - index]
            # Published so readers of ``input_conv`` do not have to infer the
            # concatenation split.  It is only ``in_channels // 3`` under the
            # paper's halving; with an explicit schedule it is neither that nor
            # constant across stages.
            self.skip_channels.append(skip)
            self.upsample_blocks.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        new_channels,
                        kernel_size=rate * 2,
                        stride=rate,
                        padding=_resample_padding(rate),
                        # Odd rates leave the transpose one sample short of the
                        # exact factor; see ``_resample_padding``.
                        output_padding=rate % 2,
                    )
                )
            )
            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    new_channels + skip,
                    new_channels,
                    kernel_sizes=self.kernel_schedule[index],
                    dilation=self.dilation_schedule[index],
                    leaky_relu_slope=self.leaky_relu_slope,
                )
            )
            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, stride=1, padding=3, bias=False)
        )

    def forward_core(self, conditioning: Tensor, template: Tensor, g: Optional[Tensor]) -> Tensor:
        """The convolutional U-Net, kept free of reductions so it can compile."""
        x = self.template_conv(template)

        skips = []
        for block in self.downsample_blocks:
            x = F.leaky_relu(x, self.leaky_relu_slope)
            # The skip is the pre-downsample activation, so the decoder stage
            # that consumes it is working at the resolution that produced it.
            skips.append(x)
            if self.training and self.checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        conditioning = self.conditioning_conv(conditioning)
        if self.cond is not None and g is not None:
            conditioning = conditioning + self.cond(g)
        x = torch.cat((x, conditioning), dim=1)

        for upsample, refine, skip in zip(
            self.upsample_blocks, self.upsample_conv_blocks, reversed(skips)
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            if self.training and self.checkpointing:
                x = checkpoint(upsample, x, use_reentrant=False)
                x = torch.cat((x, skip), dim=1)
                x = checkpoint(refine, x, use_reentrant=False)
            else:
                x = upsample(x)
                x = torch.cat((x, skip), dim=1)
                x = refine(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        # A single non-finite activation anywhere in an FP16 step would reach
        # every loss through ``tanh``'s saturated tail as a NaN gradient. There
        # is no useful signal in one, so it is dropped rather than propagated.
        return torch.tanh(torch.nan_to_num(x))

    def forward(
        self,
        x: Tensor,
        f0: Tensor,
        g: Optional[Tensor] = None,
        template_amplitude: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        """``x``: ``(batch, initial_channel, frames)``, ``f0``: ``(batch, frames)``."""
        target_length = x.shape[-1] * self.hop_length

        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        # FP32: ``linear`` interpolation of a pitch contour in FP16 quantises
        # the contour to steps the template then turns into audible jitter.
        with torch.autocast(device_type=f0.device.type, enabled=False):
            f0 = F.interpolate(
                f0.float(), size=target_length, mode="linear", align_corners=False
            )
            if template_amplitude is not None:
                template_amplitude = template_amplitude.float().clamp(0.0, 1.0)
            template = self.template_gen(f0, amplitude=template_amplitude)
        template = template.to(x.dtype)

        return self.forward_core(x, template, g)

    def remove_weight_norm(self) -> None:
        _remove_weight_norm(self.template_conv)
        _remove_weight_norm(self.conditioning_conv)
        _remove_weight_norm(self.conv_post)
        if self.cond is not None:
            _remove_weight_norm(self.cond)
        for block in self.downsample_blocks:
            _remove_weight_norm(block[0])
            block[1].remove_weight_norm()
        for upsample, refine in zip(self.upsample_blocks, self.upsample_conv_blocks):
            _remove_weight_norm(upsample)
            refine.remove_weight_norm()

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self
