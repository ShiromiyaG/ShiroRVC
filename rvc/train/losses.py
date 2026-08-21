import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

def feature_loss(fmap_r, fmap_g, normalize=False):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    terms = [
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    ]
    if not terms:
        return torch.zeros((), device=fmap_r[0][0].device)
    loss = sum(terms)
    return loss / len(terms) if normalize else loss


def discriminator_loss(
    disc_real_outputs,
    disc_generated_outputs,
    san_direction_weight=1.0,
    normalize=False,
):
    """
    Compute the discriminator loss for real and generated outputs.

    Returns:
        Tuple of (total_loss, real_loss_sum, fake_loss_sum) aggregated across
        all MPD/MSD discriminator heads.
    """
    loss = 0
    loss_real = 0
    loss_fake = 0
    branch_count = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        branch_count += 1
        if isinstance(dr, (list, tuple)):
            dr_fun, dr_dir = dr
            dg_fun, dg_dir = dg
            # SAN splits every head into a *function* output (trains the scale
            # and the trunk) and a *direction* output (trains the unit-norm
            # projection only).  Both must be driven by the same one-sided,
            # bounded surrogate: real up, fake down.
            #
            # The fake-direction term used to be written as
            # ``- w * softplus(1 - dg_dir) ** 2``.  That has the right sign but
            # is unbounded below, and its gradient *grows* like ``1 - dg_dir``
            # instead of saturating, so once the direction output on fakes goes
            # negative that single term outweighs everything else and the
            # discriminator can lower its loss without discriminating at all.
            # Mirroring the function term keeps it bounded below by zero and
            # makes it saturate, exactly like the real-side term.
            r_loss = (
                torch.mean(F.softplus(1 - dr_fun.float()) ** 2)
                + float(san_direction_weight) * torch.mean(F.softplus(1 - dr_dir.float()) ** 2)
            )
            g_loss = (
                torch.mean(F.softplus(dg_fun.float()) ** 2)
                + float(san_direction_weight) * torch.mean(F.softplus(dg_dir.float()) ** 2)
            )
        else:
            r_loss = torch.mean((1 - dr.float()) ** 2)
            g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss
        loss_real += r_loss
        loss_fake += g_loss

    if normalize and branch_count:
        loss = loss / branch_count
        loss_real = loss_real / branch_count
        loss_fake = loss_fake / branch_count
    return loss, loss_real, loss_fake


def generator_loss(
    disc_outputs,
    normalize=False,
    san_direction_weight=1.0,
    use_softplus=False,
):
    """
    Generator loss with LSGAN as the default and optional SAN softplus loss.
    """
    losses = []
    for dg in disc_outputs:
        if isinstance(dg, (list, tuple)):
            if use_softplus:
                dg = dg[0]
            else:
                l = torch.mean((1 - dg[0].float()) ** 2)
                if len(dg) > 1:
                    l = l + float(san_direction_weight) * torch.mean(
                        (1 - dg[1].float()) ** 2
                    )
                losses.append(l)
                continue
        if use_softplus:
            l = torch.mean(F.softplus(1.0 - dg.float()).square())
        else:
            l = torch.mean((1 - dg.float()) ** 2)
        losses.append(l)

    if not losses:
        return torch.zeros(())
    loss = sum(losses)
    return loss / len(losses) if normalize else loss


def _compress_envelope(value: Tensor, floor: float) -> Tensor:
    """Signed log companding, normalised so |x| <= 1 maps into [-1, 1].

    ``log1p(|x|/floor)`` turns a ratio into a difference, so each decade of
    amplitude occupies a roughly equal span.  ``floor`` sets where the curve
    stops resolving detail and should sit just under the quietest material the
    loss is meant to constrain (1e-3 ~= -60 dBFS).
    """
    scale = math.log1p(1.0 / floor)
    return torch.sign(value) * torch.log1p(value.abs() / floor) / scale


def envelope_loss(
    y, y_hat, kernel_size: int = 5, stride: int = 3, floor: float = 1e-3
):
    """Positive and negative max-pool envelope MAE on companded amplitude.

    ``kernel_size``/``stride`` set the timescale the envelope is measured on.
    The 5/3 default is roughly a 100 us window at 44.1 kHz, which is tight
    enough to act like a waveform loss.  ChouwaGAN randomises the phase of the
    NSF overtones every step, so a sample-exact waveform is not a reachable
    target: pass a millisecond-scale window (e.g. 100/50) to constrain the
    amplitude envelope without fighting the excitation's phase.

    The envelopes are companded before the MAE because a linear-amplitude L1 is
    dominated by whatever is loudest: at -40 dBFS the largest error the term can
    report is 0.01, so silencing a decay tail outright costs about as much as a
    1% error on a peak, and the generator is free to gate quiet material away.
    Companding makes the penalty depend on the ratio, so a tail that is 20 dB
    too quiet is scored like any other 20 dB error.  ``floor`` bounds that
    sensitivity so dither and dropout noise cannot dominate the loss.
    """
    # stride < kernel_size ensures overlapping coverage so no spikes are missed
    m = torch.nn.MaxPool1d(kernel_size=int(kernel_size), stride=int(stride))

    # Positive envelope  (peaks )
    y_env = m(y)
    y_hat_env = m(y_hat)

    # Negative envelope ( troughs )
    y_rev_env = m(-y)
    y_hat_rev_env = m(-y_hat)

    floor = max(1e-8, float(floor))
    return F.l1_loss(
        _compress_envelope(y_hat_env, floor), _compress_envelope(y_env, floor)
    ) + F.l1_loss(
        _compress_envelope(y_hat_rev_env, floor),
        _compress_envelope(y_rev_env, floor),
    )


def local_log_rms_loss(
    target: Tensor,
    generated: Tensor,
    window_size: int = 1024,
    hop_size: int = 256,
) -> Tensor:
    """Smooth-L1 on frame-wise log AC-RMS.

    The local mean is removed only inside this loss, so a slowly moving
    baseline cannot satisfy the loudness objective without changing either
    waveform before the reconstruction or adversarial losses see it.  Being a
    ratio in log space it carries no absolute level assumption, so it is safe
    to enable regardless of how the dataset was normalised.
    """

    common_length = min(target.shape[-1], generated.shape[-1])
    target = target[..., :common_length].float()
    generated = generated[..., :common_length].float()

    def local_ac_rms(audio: Tensor) -> Tensor:
        audio = F.pad(audio, (window_size // 2, window_size // 2), mode="reflect")
        local_mean = F.avg_pool1d(audio, kernel_size=window_size, stride=hop_size)
        mean_square = F.avg_pool1d(
            audio.square(), kernel_size=window_size, stride=hop_size
        )
        return (mean_square - local_mean.square()).clamp_min(1.0e-8).sqrt()

    target_rms = local_ac_rms(target)
    generated_rms = local_ac_rms(generated)
    frames = min(target_rms.shape[-1], generated_rms.shape[-1])

    return F.smooth_l1_loss(
        torch.log(generated_rms[..., :frames] + 1.0e-4),
        torch.log(target_rms[..., :frames] + 1.0e-4),
    )


def peak_headroom_loss(generated: Tensor, threshold: float = 0.85) -> Tensor:
    """One-sided L1 penalty for samples emitted above the headroom threshold.

    A bounded output head lets a generator chasing the loudest targets pin its
    peaks against the ceiling, where the activation gradient vanishes.  This
    pushes the waveform back below the threshold without touching signals that
    already respect it.

    One-sided and absolute: it assumes the targets themselves stay below
    ``threshold``.  Leave it at weight 0 unless the dataset is peak-normalised,
    otherwise it fights the spectral loss on legitimately loud material.
    """

    if threshold <= 0:
        raise ValueError("peak_headroom_loss threshold must be positive")
    return generated.float().abs().sub(float(threshold)).clamp_min(0.0).mean()


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()

    return loss


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for audio reconstruction.

    Computes spectral convergence and log magnitude loss
    at multiple STFT resolutions.
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_sizes: Tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        """Compute STFT magnitude."""

        # [B, C, T] -> [B, T]
        x = x.squeeze(1) 

        # Pad to avoid edge effects
        x = F.pad(x, (win_size // 2, win_size // 2), mode='reflect')

        window = torch.hann_window(win_size, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True, center=False
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale STFT loss.

        Per-sample spectral convergence with silence masking —
        mute samples (||X||_F ≈ 0) are excluded since SC is undefined for zero-energy.

        Args:
            pred: (B, T) predicted audio
            target: (B, T) target audio
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)      # [B, F, T]
            target_mag = self._stft(target, fft_size, hop_size, win_size)  # [B, F, T]

            # Per-sample Frobenius norms
            flat_target = target_mag.reshape(target_mag.size(0), -1)           # [B, F*T]
            flat_diff = (target_mag - pred_mag).reshape(target_mag.size(0), -1)
            target_nrg = torch.norm(flat_target, p=2, dim=1)                # [B]
            diff_nrg = torch.norm(flat_diff, p=2, dim=1)                    # [B]

            # Mask out silent samples (SC is undefined for zero-energy)
            mask = target_nrg > 1e-4
            if mask.any():
                sc_loss += (diff_nrg[mask] / target_nrg[mask]).mean()

            # Log magnitude loss — safe for all samples (clamp avoids -inf)
            mag_loss += F.l1_loss(
                torch.log(pred_mag.clamp(min=1e-5)),
                torch.log(target_mag.clamp(min=1e-5)),
            )

        sc_loss = sc_loss / len(self.fft_sizes) if sc_loss != 0.0 else 0.0
        mag_loss = mag_loss / len(self.fft_sizes)
        return sc_loss + mag_loss
