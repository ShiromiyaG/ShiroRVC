import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Returns:
        Tuple of (total_loss, real_loss_sum, fake_loss_sum) aggregated across
        all MPD/MSD discriminator heads.
    """
    loss = 0
    loss_real = 0
    loss_fake = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss
        loss_real += r_loss
        loss_fake += g_loss

    return loss, loss_real, loss_fake


def generator_loss(disc_outputs):
    """
    LSGAN Generator Loss:
    """
    loss = 0
    #gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss #, gen_losses


def envelope_loss(y, y_hat):
    # stride < kernel_size ensures overlapping coverage so no spikes are missed
    m = torch.nn.MaxPool1d(kernel_size=5, stride=3)

    # Positive envelope  (peaks )
    y_env = m(y)
    y_hat_env = m(y_hat)

    # Negative envelope ( troughs )
    y_rev_env = m(-y)
    y_hat_rev_env = m(-y_hat)

    return F.l1_loss(y_env, y_hat_env) + F.l1_loss(y_rev_env, y_hat_rev_env)


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask, z_p2=None):
    """
    Compute the Kullback-Leibler divergence loss.

    Supports 2-sample estimation when z_p2 is provided (variance-reduced
    Monte-Carlo estimate; no free-bits or clamping).

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
        z_p2 (torch.Tensor, optional): Second independent sample through the flow.
    """
    def _term(zp):
        return logs_p - logs_q - 0.5 + 0.5 * ((zp - m_p) ** 2) * torch.exp(-2 * logs_p)

    if z_p2 is not None:
        kl = (_term(z_p) + _term(z_p2)) * 0.5
    else:
        kl = _term(z_p)

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
