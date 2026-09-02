import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

def feature_loss(fmap_r, fmap_g, normalize=False):
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
    per_branch=False,
):
    """Discriminator loss, aggregated across all MPD/MSD heads.

    With ``per_branch``, a fourth element is appended: a detached
    ``(heads, 2)`` tensor of each head's ``(real, fake)`` contribution before
    the ``normalize`` division -- the aggregate alone hides which head (e.g.
    a period vs. a spectrogram branch) is collapsing.
    """
    loss = 0
    loss_real = 0
    loss_fake = 0
    branch_losses = [] if per_branch else None
    branch_count = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        branch_count += 1
        if isinstance(dr, (list, tuple)):
            dr_fun, dr_dir = dr
            dg_fun, dg_dir = dg
            # SAN splits every head into a *function* output (trains scale and
            # trunk) and a *direction* output (trains the unit-norm projection
            # only); both need the same one-sided, bounded surrogate. Mirroring
            # the function term here (rather than `-w * softplus(1-dg_dir)**2`)
            # keeps the fake-direction term bounded below and saturating --
            # the unbounded form let the discriminator win by pushing the
            # direction output on fakes negative without discriminating at all.
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
        if branch_losses is not None:
            branch_losses.append(
                torch.stack((r_loss.detach(), g_loss.detach()))
            )

    if normalize and branch_count:
        loss = loss / branch_count
        loss_real = loss_real / branch_count
        loss_fake = loss_fake / branch_count
    if branch_losses is not None:
        return loss, loss_real, loss_fake, torch.stack(branch_losses)
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
    enough to act like a waveform loss.  RefineGAN randomises the phase of the
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


def mel_low_frequency_weights(
    num_mels: int,
    sample_rate: int,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
    emphasis: float = 1.0,
    cutoff_hz: float = 1000.0,
    taper_octaves: float = 1.0,
) -> Tensor:
    """Per-mel-bin weights that stop the low bins being outvoted.

    The mel distance reduces with a mean, so every bin pulls with the same
    authority, and past the Huber knee the pull no longer even scales with how
    wrong the bin is.  A region can therefore be several times more wrong than
    the rest of the spectrum and still be a minority of the vote, so it never
    gets fixed -- the many nearly-right bins above 1 kHz outnumber it.

    Weighting by frequency restores the proportionality the reduction threw
    away.  The weights are normalised to a mean of 1, which matters more than it
    looks: the mel term feeds the adaptive adversarial balance through
    ``adv_to_rec_ratio``, so a reweighting that also changed the loss *scale*
    would move the GAN's operating point as a side effect.

    ``taper_octaves`` fades the emphasis out over an octave above ``cutoff_hz``
    rather than stepping it, so no bin sits on a discontinuity in the objective.
    """

    if num_mels <= 0:
        raise ValueError("mel_low_frequency_weights needs at least one bin")
    if emphasis <= 0.0:
        raise ValueError("mel_low_frequency_weights emphasis must be positive")
    if cutoff_hz <= 0.0:
        raise ValueError("mel_low_frequency_weights cutoff must be positive")

    top = float(sample_rate) / 2.0 if mel_fmax is None else float(mel_fmax)
    # ``htk=False`` matches ``librosa.filters.mel``'s default, which is the
    # basis the mels being weighted were actually built with.
    centres = librosa.mel_frequencies(
        n_mels=int(num_mels) + 2, fmin=float(mel_fmin), fmax=top, htk=False
    )[1:-1]

    weights = np.ones(int(num_mels), dtype=np.float64)
    if emphasis != 1.0:
        end = float(cutoff_hz) * (2.0 ** float(taper_octaves))
        # Cosine ramp in log frequency: octaves are the axis the ear and the
        # mel scale both use, so a ramp that is linear in Hz would spend most
        # of its length on the top half of the taper.
        with np.errstate(divide="ignore"):
            position = np.log2(np.maximum(centres, 1e-6) / float(cutoff_hz))
        position = np.clip(position / max(float(taper_octaves), 1e-6), 0.0, 1.0)
        fade = 0.5 * (1.0 + np.cos(np.pi * position))
        weights = 1.0 + (float(emphasis) - 1.0) * fade
        weights[centres <= float(cutoff_hz)] = float(emphasis)
        weights[centres >= end] = 1.0

    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


class BandWeightedSpectralLoss(nn.Module):
    """A mel distance whose per-bin reduction is weighted, not uniform.

    Wraps an unreduced elementwise distance so the weighting is orthogonal to
    the choice of Huber/L1/L2 -- the two knobs answer different questions, and
    entangling them would mean you cannot change the robustness of the distance
    without also changing which frequencies it cares about.
    """

    def __init__(self, base: nn.Module, weights: Tensor, weight_factory=None):
        super().__init__()
        if getattr(base, "reduction", "none") != "none":
            raise ValueError("BandWeightedSpectralLoss needs an unreduced base")
        self.base = base
        self.register_buffer("weights", weights.detach().reshape(1, -1, 1))
        #: ``callable(num_mels) -> Tensor``, for callers that legitimately hand
        #: this several bin counts.  Absent -- the default -- a mismatch stays a
        #: hard error, because for a single-resolution mel it means the config
        #: and the weights disagree, and silently reweighting the wrong bands is
        #: worse than stopping.
        self._weight_factory = weight_factory
        self._cache: dict[int, Tensor] = {}

    def _weights_for(self, bins: int, like: Tensor) -> Tensor:
        if bins == self.weights.shape[-2]:
            return self.weights.to(like.dtype)
        if self._weight_factory is None:
            raise ValueError(
                f"Band weights cover {self.weights.shape[-2]} mel bins but the "
                f"loss was handed {bins}."
            )
        cached = self._cache.get(bins)
        if cached is None or cached.device != like.device:
            cached = (
                self._weight_factory(bins).detach().reshape(1, -1, 1).to(like.device)
            )
            self._cache[bins] = cached
        return cached.to(like.dtype)

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        elementwise = self.base(target, prediction)
        weights = self._weights_for(elementwise.shape[-2], elementwise)
        return (elementwise * weights).mean()


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask, return_terms: bool = False):
    """KL divergence between posterior q and prior p, masked mean over valid frames.

    ``return_terms`` additionally hands back the *detached* per-element
    divergence, before any masking or reduction.  The per-dimension KL
    diagnostics want exactly the tensor this function has already formed, and
    forming it a second time at the call site is a full extra elementwise pass
    -- a square and an ``exp`` over ``(batch, channels, frames)`` -- every
    step, for a number that is only read once per logging interval.
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    loss = (kl * z_mask).sum() / z_mask.sum()

    if return_terms:
        return loss, kl.detach()
    return loss


class MultiScaleSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude loss at multiple STFT resolutions."""

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_sizes: Tuple[int, ...] = (512, 1024, 2048),
        log_scale: float = 1000.0,
        spectral_convergence: bool = False,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        #: Compression knee, matching ``wave_to_mel(for_loss=True)``.  See
        #: :meth:`forward` for why the compression is ``log1p`` and not ``log``.
        self.log_scale = float(log_scale)
        #: Spectral convergence, off by default. ``||X - X̂||_F / ||X||_F``'s
        #: Frobenius norm is dominated by the loudest bins (measured: top 1%
        #: above 7.1 vs. median 0.013), which duplicates what the low-frequency
        #: mel term already covers and works against MS-STFT's actual value:
        #: linear frequency resolution at the top of the band.
        self.spectral_convergence = bool(spectral_convergence)

    def _stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        x = x.squeeze(1)
        x = F.pad(x, (win_size // 2, win_size // 2), mode='reflect')

        window = torch.hann_window(win_size, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True, center=False
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale STFT loss over ``(B, T)`` audio.

        Uses ``log1p(mag * log_scale)``, matching ``wave_to_mel(for_loss=True)``,
        instead of ``log(mag.clamp(1e-5))``: on this dataset 8% of bins are
        real digital silence, and the clamp's fixed floor gives those bins a
        gradient ~13x larger than audible ones, pushing silence toward more
        silence. ``log1p`` agrees with ``log`` at audible levels but turns
        linear below ``1 / log_scale``, so near-zero bins are scored on
        distance from zero instead of a ratio between two inaudible numbers.
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)
            target_mag = self._stft(target, fft_size, hop_size, win_size)

            if self.spectral_convergence:
                flat_target = target_mag.reshape(target_mag.size(0), -1)
                flat_diff = (target_mag - pred_mag).reshape(target_mag.size(0), -1)
                target_nrg = torch.norm(flat_target, p=2, dim=1)
                diff_nrg = torch.norm(flat_diff, p=2, dim=1)

                # SC is undefined for zero-energy targets
                mask = target_nrg > 1e-4
                if mask.any():
                    sc_loss += (diff_nrg[mask] / target_nrg[mask]).mean()

            mag_loss += F.l1_loss(
                torch.log1p(pred_mag * self.log_scale),
                torch.log1p(target_mag * self.log_scale),
            )

        if self.spectral_convergence and sc_loss != 0.0:
            sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        return sc_loss + mag_loss
