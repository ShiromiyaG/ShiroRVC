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
    The 5/3 default is roughly a 160 us window at 32 kHz, which is tight
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


def mel_frequency_tilt_weights(
    num_mels: int,
    sample_rate: int,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
    tilt: float = 0.0,
    max_ratio: float = 8.0,
) -> Tensor:
    """Per-mel-bin weights that undo part of the mel scale's own bin density.

    An L1 over log-mel gives every bin a gradient of the same magnitude, so the
    share of the objective a frequency region receives is its share of the
    *bins*, not of the spectrum, and not of how wrong it is.  The mel scale
    puts those bins at the bottom: over 0-16 kHz, 0-2 kHz is 12.5% of the
    spectrum and 45% of the bins, while 10-16 kHz is 37.5% of the spectrum and
    12% of them.  That split is identical at 40, 80, 160, 320 and 640 bins --
    it is a property of the warping, not of the resolution -- which is why no
    choice of scale set in the multi-scale loss moves it.

    ``tilt`` interpolates between the two ways of counting.  Each bin is
    weighted by its own bandwidth raised to ``tilt``: at 0.0 the weights are
    flat and this is exactly the unweighted loss, at 1.0 every *hertz* pulls
    equally and the mel warping is cancelled outright.  In between is a
    deliberate compromise -- the warping is a perceptual statement worth
    keeping some of, and the top bands are also where a vocoder's errors are
    least audible per dB.

    What it buys, measured on a held-out validation clip at 32 kHz with the
    multi-scale set (40/256 .. 640/4096): the penalty the loss charges for one
    and the same -6 dB shelf, by where the shelf is, each column normalised to
    its own 1-3 kHz value --

        shelf at      tilt 0    tilt 0.35   tilt 0.5   tilt 1.0
        1-3 kHz        1.000      1.000       1.000      1.000
        3-6 kHz        0.633      0.865       0.986      1.512
        6-10 kHz       0.458      0.773       0.964      1.849
        10-13 kHz      0.232      0.447       0.591      0.993
        13-16 kHz      0.155      0.324       0.443      0.666

    Unweighted, an identical defect costs 4.3x less at 10-13 kHz than at
    1-3 kHz and 6.5x less at 13-16 kHz.  0.5 roughly triples what the top two
    bands charge without reordering them; 1.0 overshoots -- 6-10 kHz ends up
    charging more than 1-3 kHz -- because a band's weight there follows its
    width in hertz and those bands are wide.  The loss value on a real
    generated/reference pair moves +10% between tilt 0 and 0.5, which is the
    part that reaches ``adv_to_rec_ratio``.

    ``max_ratio`` caps how far the extremes may separate.  Without it the
    bottom bins, which are a few tens of hertz wide, take weights near zero at
    high tilt and the loss stops constraining the fundamental at all.

    Normalised to a mean of 1, like ``mel_low_frequency_weights`` and for the
    same reason: the mel term feeds the adaptive adversarial balance through
    ``adv_to_rec_ratio``, so the weighting must change *which* bins are heard
    and not the scale of the term.  Note that this holds the weights' mean, not
    the loss value -- on an error spectrum that is itself tilted, a tilted
    weighting does move the number, which is the point of it.
    """

    if num_mels <= 0:
        raise ValueError("mel_frequency_tilt_weights needs at least one bin")
    if tilt < 0.0:
        raise ValueError("mel_frequency_tilt_weights tilt must not be negative")
    if max_ratio < 1.0:
        raise ValueError("mel_frequency_tilt_weights max_ratio must be >= 1")

    if tilt == 0.0:
        return torch.ones(int(num_mels), dtype=torch.float32)

    top = float(sample_rate) / 2.0 if mel_fmax is None else float(mel_fmax)
    # ``+2`` and the trim are how ``librosa.filters.mel`` places centres; the
    # untrimmed array is also what gives the first and last bin a neighbour to
    # measure a bandwidth against.
    edges = librosa.mel_frequencies(
        n_mels=int(num_mels) + 2, fmin=float(mel_fmin), fmax=top, htk=False
    )
    # A triangular mel filter spans its two neighbouring centres, so this is
    # the filter's own width rather than a finite difference standing in for
    # one.
    bandwidth = edges[2:] - edges[:-2]
    bandwidth = np.maximum(bandwidth, 1e-6)

    weights = bandwidth ** float(tilt)
    # Cap around the geometric mean so the clip is symmetric in the log domain
    # the weights live in; clipping around the arithmetic mean would tighten
    # one end harder than the other.
    centre = float(np.exp(np.log(weights).mean()))
    limit = float(max_ratio) ** 0.5
    weights = np.clip(weights, centre / limit, centre * limit)

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


class HighFrequencyFloorNegative(nn.Module):
    """Real audio with its high-frequency *noise floor* pulled down.

    A synthetic negative for the discriminator, and it exists because of what
    the discriminator was measured doing without one.  What a RefineGAN2
    generator is short of above 10 kHz is not the harmonic comb -- its comb is
    *sharper* than the reference's, 6.6 dB of peak-to-valley contrast against
    2.6 -- but the stochastic floor between the harmonics, which came out
    11.1 dB down where the peaks were 7.1 down.  That component is by
    definition the unpredictable part of the signal, so every reconstruction
    loss is minimised by producing less of it: the multi-scale mel does charge
    for the defect (13% of its own total, 26% with the frequency tilt) and
    still cannot ask for it, because charging more does not move where its
    optimum sits.

    Restoring a stochastic component is the adversarial term's job.  Measured
    on a pretrain at 87k steps, it was doing the opposite -- separation of
    -0.13 on the spectrogram head at 512 points and -0.82 on UnivHD, both
    meaning the head scored the *quieter* floor as the more real of the two.
    Nothing corrects that, because telling real audio from a generator's output
    never required an opinion about it, and a direction the task does not
    constrain is free to point anywhere.

    Making it part of the task is what this is.  The same head, trained on this
    negative from a fresh initialisation, reaches 100% held-out accuracy in 300
    steps -- with the linear-magnitude input the branch already has, which was
    checked and is not the obstacle it looks like.

    The transform is a power law on the magnitude above ``cutoff``, normalised
    to the band's own peak per frame, so the peaks stay and the valleys fall:
    at ``gamma`` 1.6 that measured -3.8 dB on the peaks against -11.2 on the
    valleys, which is the shape of the real defect.  Phase is kept, and the
    round trip is transparent -- 130 dB below the signal, and a branch trained
    to separate real audio from the same audio through the round trip alone
    scores 45.3%, i.e. chance.  So the negative teaches the floor and not the
    resynthesis, which is the way this fails when it fails.

    ``gamma`` and ``cutoff`` are drawn per item per step.  Fixed, a
    discriminator can learn one filter's signature instead of the thing the
    filter is standing in for.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        gamma_range: Tuple[float, float] = (1.3, 2.0),
        cutoff_range: Tuple[float, float] = (6000.0, 11000.0),
    ):
        super().__init__()
        if not 0.0 < float(gamma_range[0]) <= float(gamma_range[1]):
            raise ValueError(
                f"gamma_range must be a positive, ordered pair, not {gamma_range!r}."
            )
        if not 0.0 <= float(cutoff_range[0]) <= float(cutoff_range[1]):
            raise ValueError(
                f"cutoff_range must be an ordered pair, not {cutoff_range!r}."
            )
        if float(cutoff_range[1]) >= float(sample_rate) / 2.0:
            raise ValueError(
                f"cutoff_range tops out at {cutoff_range[1]} Hz, which is at or "
                f"above the {sample_rate} Hz Nyquist; there would be no band to "
                f"act on."
            )
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.cutoff_range = (float(cutoff_range[0]), float(cutoff_range[1]))
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(self.n_fft, 1.0 / float(sample_rate)),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, wave: Tensor) -> Tensor:
        """``wave`` in, ``(B, 1, T)``, the same shape back.

        ``no_grad`` on the method rather than at the call site: there is no
        gradient path from here to the generator -- the negative is built from
        the *reference*, not from anything the generator produced -- and a
        graph that reaches nothing is the kind of waste that survives review.
        """

        # ``torch.stft`` has no half-precision path on every backend, and the
        # discriminator update runs under autocast.
        with torch.autocast(device_type=wave.device.type, enabled=False):
            source = wave.float()
            batch = source.shape[0]
            spectrum = torch.stft(
                source.squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(source.device),
                center=True,
                return_complex=True,
            )
            magnitude = spectrum.abs()

            low, high = self.gamma_range
            gamma = torch.empty(batch, 1, 1, device=source.device).uniform_(low, high)
            low, high = self.cutoff_range
            cutoff = torch.empty(batch, 1, 1, device=source.device).uniform_(low, high)
            band = self.freqs.to(source.device).view(1, -1, 1) >= cutoff

            # Per frame, per item: the band's own peak is what the power law is
            # normalised to, so it is the *shape* inside the band that changes
            # and not the band's ceiling.
            peak = magnitude.masked_fill(~band, 0.0).amax(dim=1, keepdim=True)
            peak = peak.clamp_min(1e-9)
            expanded = peak * (magnitude / peak).clamp(0.0, 1.0) ** gamma
            magnitude = torch.where(band, expanded, magnitude)

            spectrum = torch.polar(magnitude, spectrum.angle())
            return torch.istft(
                spectrum,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(source.device),
                center=True,
                length=source.shape[-1],
            ).unsqueeze(1)


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
