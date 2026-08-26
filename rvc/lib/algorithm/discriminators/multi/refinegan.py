"""RefineGAN's discriminator: HiFi-GAN's MPD plus a multi-resolution branch.

The paper judges the waveform on two axes and nothing else -- periodicity, via
the multi-period discriminator it inherits from HiFi-GAN, and spectral detail,
via a multi-resolution discriminator over STFT magnitudes.  There is no SAN
projection and no learned sub-band or CQT front end here; those belonged to the
previous decoder and are gone with it.

The trainer's per-branch machinery -- the R1 strength controller, the branch
loss series, the paired real/fake forward -- is written against a small contract
rather than against a specific discriminator, so this module implements it:
``num_branches``, ``branch_names``, ``prepare_spectrograms``, ``forward_branch``,
``r1_penalty`` and ``enable_compile``.

Mixed precision.  Every branch drops non-finite activations between layers
instead of propagating them: a single overflow in FP16 otherwise reaches the
generator as a NaN gradient through feature matching, and a non-finite
activation carries no adversarial signal worth preserving.  The STFT and its
magnitude stay in FP32 because cuFFT's FP16 path underflows on quiet frames.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

PERIODS = (2, 3, 5, 7, 11)
#: ``(n_fft, hop_length, win_length)`` per resolution.
RESOLUTIONS = (
    (1024, 120, 600),
    (2048, 240, 1200),
    (512, 50, 240),
)
LRELU_SLOPE = 0.2


class PeriodDiscriminator(nn.Module):
    """Reshape the waveform to ``(period, time/period)`` and judge it in 2D."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        channels: Sequence[int] = (1, 64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.period = int(period)
        norm = spectral_norm if use_spectral_norm else weight_norm
        channels = tuple(int(value) for value in channels)
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )
        self.conv_post = norm(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        batch, channels, length = x.shape
        if length % self.period:
            padding = self.period - (length % self.period)
            x = F.pad(x, (0, padding), "reflect")
            length = length + padding
        x = x.view(batch, channels, length // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            x = torch.nan_to_num(x)
            fmap.append(x)
        x = torch.nan_to_num(self.conv_post(x))
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class ResolutionDiscriminator(nn.Module):
    """Judge one STFT magnitude resolution.

    ``spectrogram`` is deliberately separable from the convolutional head. The
    trainer computes each real resolution once and reuses it across the paired
    forward, and the R1 penalty differentiates the head with the transform held
    outside the graph.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 120,
        win_length: int = 600,
        use_spectral_norm: bool = False,
        channels: int = 32,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        norm = spectral_norm if use_spectral_norm else weight_norm
        # A rectangular window, as in the UnivNet/RefineGAN lineage: the branch
        # is a learned detector over magnitudes, not an analysis filter bank,
        # and a taper would only narrow what it can see at the frame edges.
        self.register_buffer("window", torch.ones(self.win_length), persistent=False)

        self.convs = nn.ModuleList(
            [
                norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x: Tensor) -> Tensor:
        """``(batch, 1, samples)`` -> ``(batch, bins, frames)`` magnitude."""
        device_type = x.device.type
        autocast_enabled = torch.is_autocast_enabled(device_type=device_type)
        output_dtype = (
            torch.get_autocast_dtype(device_type) if autocast_enabled else x.dtype
        )
        with torch.autocast(device_type=device_type, enabled=False):
            padding = int((self.n_fft - self.hop_length) / 2)
            value = F.pad(x.float(), (padding, padding), mode="reflect").squeeze(1)
            value = torch.stft(
                value,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=False,
                return_complex=True,
            )
            magnitude = value.abs()
        if autocast_enabled and output_dtype in (torch.float16, torch.bfloat16):
            if output_dtype == torch.float16:
                # The magnitude is already exact in FP32; saturating only the
                # values FP16 cannot hold keeps the cast itself from creating
                # infinities out of loud frames.
                magnitude = magnitude.clamp(max=torch.finfo(output_dtype).max)
            return magnitude.to(output_dtype)
        return magnitude

    def forward_spectrogram(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            x = torch.nan_to_num(x)
            fmap.append(x)
        x = torch.nan_to_num(self.conv_post(x))
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self.forward_spectrogram(self.spectrogram(x))


class RefineGANDiscriminator(nn.Module):
    """MPD + MRD, exposing the per-branch contract the trainer is built on."""

    uses_branchwise_r1 = True

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 44100,
        periods: Sequence[int] = PERIODS,
        resolutions: Sequence[Sequence[int]] = RESOLUTIONS,
        resolution_channels: int = 32,
        **_: object,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.use_checkpointing = bool(use_checkpointing)

        period_branches = [
            PeriodDiscriminator(period, use_spectral_norm=use_spectral_norm)
            for period in periods
        ]
        resolution_branches = [
            ResolutionDiscriminator(
                int(n_fft),
                int(hop_length),
                int(win_length),
                use_spectral_norm=use_spectral_norm,
                channels=int(resolution_channels),
            )
            for n_fft, hop_length, win_length in resolutions
        ]
        self.period_count = len(period_branches)
        self.resolution_count = len(resolution_branches)
        self.discriminators = nn.ModuleList(period_branches + resolution_branches)
        # Built from what was constructed rather than from a literal list, so a
        # changed ``periods`` or ``resolutions`` cannot mislabel a metric series.
        self.branch_names = tuple(
            [f"period_{branch.period}" for branch in period_branches]
            + [f"stft_{branch.n_fft}" for branch in resolution_branches]
        )

    @property
    def num_branches(self) -> int:
        return len(self.discriminators)

    def _spectrogram_index(self, index: int) -> Optional[int]:
        """Map a branch index onto its slot in a precomputed spectrogram list."""
        offset = int(index) - self.period_count
        return offset if 0 <= offset < self.resolution_count else None

    def _forward_one(
        self,
        discriminator: nn.Module,
        audio: Tensor,
        spectrogram: Optional[Tensor] = None,
    ):
        forward = (
            discriminator.forward_spectrogram
            if spectrogram is not None
            else discriminator
        )
        value = spectrogram if spectrogram is not None else audio
        if self.training and self.use_checkpointing:
            return checkpoint(forward, value, use_reentrant=False)
        return forward(value)

    def prepare_spectrograms(self, audio: Tensor) -> list[Tensor]:
        return [
            discriminator.spectrogram(audio)
            for discriminator in self.discriminators[self.period_count :]
        ]

    def forward_branch(self, audio: Tensor, branch_index: int):
        return self._forward_one(self.discriminators[int(branch_index)], audio)

    def _forward_audio(self, audio: Tensor, spectrograms=None):
        logits, feature_maps = [], []
        for index, discriminator in enumerate(self.discriminators):
            spectrogram = None
            if spectrograms is not None:
                slot = self._spectrogram_index(index)
                if slot is not None:
                    spectrogram = spectrograms[slot]
            logit, fmap = self._forward_one(discriminator, audio, spectrogram)
            logits.append(logit)
            feature_maps.append(fmap)
        return logits, feature_maps

    def forward(
        self,
        y: Tensor,
        y_hat: Optional[Tensor] = None,
        real_spectrograms=None,
        fake_spectrograms=None,
        branch_index: Optional[int] = None,
        pair_batches: bool = False,
        san_training: bool = False,
    ):
        # ``san_training`` is part of the trainer's call signature and is
        # accepted so the loop needs no discriminator-specific branch. RefineGAN
        # has no SAN projection, so every head returns a plain logit and
        # ``discriminator_loss`` takes its LSGAN path, which is the objective
        # the paper specifies.
        del san_training

        if branch_index is not None:
            return self.forward_branch(y, branch_index)

        if y_hat is None:
            return self._forward_audio(y, real_spectrograms)

        if not pair_batches:
            real_logits, real_features = self._forward_audio(y, real_spectrograms)
            fake_logits, fake_features = self._forward_audio(y_hat, fake_spectrograms)
            return real_logits, fake_logits, real_features, fake_features

        # One batched forward over ``cat(real, fake)``: half the kernel launches
        # of two passes, and every branch sees both halves under identical
        # weights.
        if real_spectrograms is None:
            real_spectrograms = self.prepare_spectrograms(y)
        if fake_spectrograms is None:
            fake_spectrograms = self.prepare_spectrograms(y_hat)

        real_batch_size = y.shape[0]
        paired_audio = torch.cat((y, y_hat), dim=0)
        paired_spectrograms = [
            torch.cat((real, fake), dim=0)
            for real, fake in zip(real_spectrograms, fake_spectrograms, strict=True)
        ]
        paired_logits, paired_features = self._forward_audio(
            paired_audio, paired_spectrograms
        )
        real_logits = [value[:real_batch_size] for value in paired_logits]
        fake_logits = [value[real_batch_size:] for value in paired_logits]
        real_features = [
            [value[:real_batch_size] for value in branch] for branch in paired_features
        ]
        fake_features = [
            [value[real_batch_size:] for value in branch] for branch in paired_features
        ]
        return real_logits, fake_logits, real_features, fake_features

    def enable_compile(self, mode: str = "default") -> bool:
        """Compile the paired real/fake path without wrapping the module.

        Only :meth:`forward` is compiled. The R1 penalty reaches its branch
        through :meth:`forward_branch` and differentiates it twice with
        ``create_graph=True``; double-backward through a compiled region is the
        part of this loop most likely to break, and it runs on one branch every
        ``r1_interval`` steps, so there is nothing to win by including it.
        """
        if getattr(self, "_compile_enabled", False):
            return True

        eager_forward = self.forward
        try:
            compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
        except Exception as error:
            from rvc.train.messages import DISCRIMINATOR_COMPILE_ENABLE_FAILED

            print(DISCRIMINATOR_COMPILE_ENABLE_FAILED, str(error))
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            # ``branch_index`` is the R1 path; keep it on the eager callable.
            if (
                not self.training
                or compile_failed
                or kwargs.get("branch_index") is not None
            ):
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                from rvc.train.messages import DISCRIMINATOR_COMPILE_RUNTIME_FAILED

                print(DISCRIMINATOR_COMPILE_RUNTIME_FAILED, str(error))
                return eager_forward(*args, **kwargs)

        self.forward = training_forward
        self._compile_enabled = True
        self._compile_mode = mode
        return True

    def r1_penalty(self, real_audio: Tensor, branch_index: int) -> Tensor:
        """Gradient penalty on the input the branch actually judges.

        For a period branch that is the waveform.  For a resolution branch it is
        the **magnitude spectrogram**, with ``torch.stft`` held outside the
        graph.  R1 exists to keep the *discriminator* from becoming too sharp,
        and a branch can do nothing about the conditioning of a fixed, unlearned
        front end except shrink its own weights toward zero.  The units change
        -- per bin rather than per sample -- which is harmless because the
        per-branch controller normalises against each branch's own
        discriminative gradient and never compares absolute scales across
        branches.  It is also cheaper: the double backward no longer runs
        through the FFT.
        """
        index = int(branch_index)
        slot = self._spectrogram_index(index)
        discriminator = self.discriminators[index]
        if slot is not None:
            source = discriminator.spectrogram(real_audio).detach().requires_grad_(True)
            real_logits, _ = self._forward_one(
                discriminator, real_audio, spectrogram=source
            )
        else:
            source = real_audio.detach().requires_grad_(True)
            real_logits, _ = self._forward_one(discriminator, source)
        score = real_logits.float().mean()
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=source,
            create_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().flatten(1).sum(dim=1).mean()

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
