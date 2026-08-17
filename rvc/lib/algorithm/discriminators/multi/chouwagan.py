import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding


def _norm_layer(layer):
    return weight_norm(layer)


class ChouwaPeriodDiscriminator(nn.Module):
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        norm = torch.nn.utils.parametrizations.spectral_norm if use_spectral_norm else _norm_layer
        in_channels = [1, 16, 64, 128, 192]
        out_channels = [16, 64, 128, 192, 256]
        strides = [3, 3, 3, 3, 1]
        self.period = int(period)
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        (5, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                )
                for in_ch, out_ch, stride in zip(
                    in_channels, out_channels, strides, strict=True
                )
            ]
        )
        self.conv_post = norm(nn.Conv2d(256, 1, (3, 1), padding=(1, 0)))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor):
        fmap = []
        batch, channels, length = x.shape
        if length % self.period:
            x = F.pad(x, (0, self.period - length % self.period), mode="reflect")
        x = x.view(batch, channels, -1, self.period)
        for conv in self.convs:
            x = self.activation(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class ChouwaSpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm = torch.nn.utils.parametrizations.spectral_norm if use_spectral_norm else _norm_layer
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        channels = [3, 24, 48, 96, 128]
        strides = [(1, 2), (2, 2), (2, 2), (2, 2)]
        self.convs = nn.ModuleList(
            [
                ChouwaSeparableConv2d(
                    in_ch,
                    out_ch,
                    stride,
                    norm,
                )
                for in_ch, out_ch, stride in zip(
                    channels[:-1], channels[1:], strides, strict=True
                )
            ]
        )
        self.conv_post = norm(nn.Conv2d(128, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: Tensor):
        return self.forward_spectrogram(self.spectrogram(x))

    def spectrogram(self, x: Tensor) -> Tensor:
        waveform = x.squeeze(1).float()
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = spec.abs().clamp_min(1e-5)
        complex_scale = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        x = torch.stack(
            (
                torch.log1p(magnitude),
                spec.real / complex_scale,
                spec.imag / complex_scale,
            ),
            dim=1,
        )

        return x

    def forward_spectrogram(self, x: Tensor):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class ChouwaSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride, norm):
        super().__init__()
        self.depthwise = norm(
            nn.Conv2d(
                in_channels,
                in_channels,
                (3, 5),
                stride,
                padding=(1, 2),
                groups=in_channels,
            )
        )
        self.pointwise = norm(nn.Conv2d(in_channels, out_channels, 1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.pointwise(self.activation(self.depthwise(x))))


class ChouwaGANDiscriminator(nn.Module):
    uses_branchwise_r1 = True

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 44100,
        **_: object,
    ):
        super().__init__()
        if int(sample_rate) != 44100:
            raise ValueError("ChouwaGAN only supports 44.1 kHz configurations.")
        self.use_checkpointing = bool(use_checkpointing)
        self.period_count = 5
        self.discriminators = nn.ModuleList(
            [
                ChouwaPeriodDiscriminator(
                    period,
                    use_spectral_norm=use_spectral_norm,
                )
                for period in (2, 3, 5, 7, 11)
            ]
            + [
                ChouwaSpectrogramDiscriminator(
                    n_fft,
                    hop_length,
                    n_fft,
                    use_spectral_norm=use_spectral_norm,
                )
                for n_fft, hop_length in (
                    (512, 128),
                    (1024, 256),
                    (2048, 512),
                )
            ]
        )

    @property
    def num_branches(self) -> int:
        return len(self.discriminators)

    def _forward_one(
        self,
        discriminator: nn.Module,
        audio: Tensor,
        spectrogram: Tensor | None = None,
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

    def prepare_spectrograms(self, audio: Tensor):
        return [
            discriminator.spectrogram(audio)
            for discriminator in self.discriminators[self.period_count :]
        ]

    def forward_branch(self, audio: Tensor, branch_index: int):
        discriminator = self.discriminators[int(branch_index)]
        return self._forward_one(discriminator, audio)

    def _forward_audio(self, audio: Tensor, spectrograms=None):
        logits, feature_maps = [], []
        for index, discriminator in enumerate(self.discriminators):
            spectrogram = None
            if spectrograms is not None and index >= self.period_count:
                spectrogram = spectrograms[index - self.period_count]
            logit, fmap = self._forward_one(discriminator, audio, spectrogram)
            logits.append(logit)
            feature_maps.append(fmap)
        return logits, feature_maps

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor | None = None,
        real_spectrograms=None,
        fake_spectrograms=None,
        branch_index: int | None = None,
        pair_batches: bool = False,
    ):
        if branch_index is not None:
            return self.forward_branch(y, branch_index)

        if y_hat is None:
            return self._forward_audio(y, real_spectrograms)

        if not pair_batches:
            real_logits, real_features = self._forward_audio(y, real_spectrograms)
            fake_logits, fake_features = self._forward_audio(
                y_hat,
                fake_spectrograms,
            )
            return real_logits, fake_logits, real_features, fake_features

        if real_spectrograms is None:
            real_spectrograms = self.prepare_spectrograms(y)
        if fake_spectrograms is None:
            fake_spectrograms = self.prepare_spectrograms(y_hat)

        real_batch_size = y.shape[0]
        paired_audio = torch.cat((y, y_hat), dim=0)
        paired_spectrograms = [
            torch.cat((real, fake), dim=0)
            for real, fake in zip(
                real_spectrograms,
                fake_spectrograms,
                strict=True,
            )
        ]
        paired_logits, paired_features = self._forward_audio(
            paired_audio,
            paired_spectrograms,
        )
        real_logits = [value[:real_batch_size] for value in paired_logits]
        fake_logits = [value[real_batch_size:] for value in paired_logits]
        real_features = [
            [value[:real_batch_size] for value in branch]
            for branch in paired_features
        ]
        fake_features = [
            [value[real_batch_size:] for value in branch]
            for branch in paired_features
        ]
        return real_logits, fake_logits, real_features, fake_features

    def r1_penalty(self, real_audio: Tensor, branch_index: int) -> Tensor:
        real_audio = real_audio.detach().requires_grad_(True)
        real_logits, _ = self.forward_branch(real_audio, branch_index)
        score = real_logits.float().mean()
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=real_audio,
            create_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().flatten(1).sum(dim=1).mean()

    def remove_weight_norm(self):
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
