"""ChouwaGAN: an iSTFT vocoder whose network predicts only coarse gains.

A frame-rate ConvNeXt reads the mel and f0 and predicts, per band, the gain of
two analytic sources: a band-limited, phase-aligned pulse train (all harmonics
of f0 up to Nyquist, faded out before it) under a Kaiser window, and white
noise, modulated by the glottal cycle, under a shorter Hann window. The
harmonic gain is mixed phase: a minimum-phase envelope for the tract and a
coarser maximum-phase one for the glottis, both linear maps of the band gains.
The harmonics' place and phase come from the source, a real signal, so the
spectrogram is consistent by construction and nothing but the noise branch
fills the valleys between harmonics. An optional bounded complex residual per
band refines the harmonic branch. With ``noise_curve_hz``, voiced noise is
the harmonic envelope times a curve through those knots: the mel cannot tell
harmonics from noise above a few kHz, and free per-band noise gains guessed
it band by band, in holes. With ``aspiration_regions_hz``, the voiced noise
follows the glottal cycle as a pulse of predicted depth and sharpness per
region instead of one cosine. With ``contrast_band_edges``, the network also
reads each band's harmonic contrast, measured from the audio (``measure_contrast``)
or absent (zeros). With ``max_phase_spread``, each tract band's harmonics get
a random phase per synthesis step of predicted spread: a phase-aligned pulse
train is coherent across every glottal cycle up to Nyquist, which a voice is
not above a few kHz, and gains alone cannot undo that. With ``phase_head_hz``,
the network also turns the phase of every harmonic-branch bin above it, per
harmonic sub-step: no band-wide or random phase made the top sound like a
voice, while the recording's own phase per bin did. With ``phase_noise_mix``
it also blends, per bin, the phase of a white noise's spectrogram into that
turn, as Wavehax's prior carries noise: the network alone is deterministic and
cannot draw the top's randomness. With ``source_phase_hz``, every harmonic of
the harmonic branch from there up keeps a fixed random phase of its own, set
per bin from the harmonic of f0 nearest to it, so the top is not one click per
cycle and the gains train on that. Both branches are
synthesised at ``synthesis_hop`` by overlap-add over blocks of frames, which
add up to the whole transform exactly and bound the memory of long inputs. No
upsampling layer and no activation run at the audio rate.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.discriminators.san import SANConv2d

#: ``architecture`` of a ChouwaGAN rectified vocoder export, and of its runs.
ARCHITECTURE = "chouwagan"
SOURCE_F0_FLOOR = 50.0
LOG_GAIN_RANGE = (-12.0, 6.0)
#: The band contrast's input scale, from nepers to about [0, 1.5].
CONTRAST_SCALE = 4.0
PHASES = ("mixed", "minimum", "zero")
INTERPOLATIONS = ("linear", "cubic")
#: Initial logit of the phase spread: about 2% of ``max_phase_spread``, nearly coherent.
SPREAD_INIT_BIAS = -4.0
#: Below ``phase_head_hz`` the phase head's turn fades in over this much.
PHASE_HEAD_RAMP_HZ = 1000.0
#: Initial logit of the noise phase's share: about 0.25%, nearly none.
NOISE_MIX_INIT_BIAS = -6.0
#: Floor under a compressed magnitude, as a share of a full-scale sine's:
#: bounds the slope at silence.
COMPRESSION_FLOOR = 1e-4
#: Of the phase heads' unit-RMS band: where a bin's phase starts to weigh.
PHASE_FLOOR = 0.2
#: Below ``source_phase_hz`` the harmonics' phases fade in over this much.
SOURCE_PHASE_RAMP_HZ = 1000.0
SOURCE_PHASE_SEED = 0


def continuous_f0(f0):
    """``f0`` [B, T] with its unvoiced zeros filled linearly between voiced
    frames and held past the first and last; all-unvoiced rows stay zero."""
    frames = f0.shape[-1]
    voiced = f0 > 0
    index = torch.arange(frames, device=f0.device).expand_as(f0)
    last = torch.where(voiced, index, -1).cummax(-1).values
    following = torch.where(voiced, index, frames).flip(-1).cummin(-1).values.flip(-1)
    f_last = f0.gather(-1, last.clamp(min=0))
    f_next = f0.gather(-1, following.clamp(max=frames - 1))
    weight = (index - last) / (following - last).clamp(min=1)
    filled = torch.where(
        (last >= 0) & (following < frames), f_last + (f_next - f_last) * weight,
        torch.where(last >= 0, f_last, torch.where(following < frames, f_next, torch.zeros_like(f0))),
    )
    return torch.where(voiced, f0, filled)


def band_centers(sample_rate, min_hz, ratio):
    """Band centres from 0 to Nyquist, ``min_hz`` apart at the bottom and a
    ``ratio`` of their frequency apart above, which keeps every gain's impulse
    response short."""
    nyquist = sample_rate / 2
    centers = [0.0]
    while centers[-1] < nyquist:
        centers.append(centers[-1] + max(min_hz, ratio * centers[-1]))
    centers[-1] = nyquist
    return torch.tensor(centers, dtype=torch.float64)


def band_matrix(centers, n_fft, sample_rate):
    """[bands, n_fft // 2 + 1]: linear interpolation of band values to bins."""
    return interpolation_matrix(centers, torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64))


def interpolation_weights(centers, freqs):
    """For each of ``freqs``, the indices of the ``centers`` either side and
    the weight of the upper one, held flat past both ends."""
    freqs = freqs.clamp(centers[0], centers[-1])
    upper = torch.searchsorted(centers, freqs, right=True).clamp(1, len(centers) - 1)
    lower = upper - 1
    return lower, upper, (freqs - centers[lower]) / (centers[upper] - centers[lower])


def interpolation_matrix(centers, freqs):
    """[len(centers), len(freqs)]: linear interpolation of values at the
    ``centers`` to ``freqs``, held flat past both ends."""
    lower, upper, t = interpolation_weights(centers, freqs)
    matrix = torch.zeros(len(centers), len(freqs), dtype=torch.float64)
    columns = torch.arange(len(freqs))
    matrix[lower, columns] = 1 - t
    matrix[upper, columns] += t
    return matrix


def minimum_phase_basis(matrix, n_fft):
    """Each row of a log-magnitude basis [bands, bins] as the complex log
    spectrum of its minimum-phase version. Folding the real cepstrum is
    linear, so the same weighted sum of these rows is the minimum-phase log
    spectrum of the weighted sum of log magnitudes; its conjugate is the
    maximum-phase one."""
    cepstrum = torch.fft.irfft(matrix, n=n_fft, dim=-1)
    fold = torch.zeros(n_fft, dtype=cepstrum.dtype)
    fold[0] = 1.0
    fold[1:n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    return torch.fft.rfft(cepstrum * fold, n=n_fft, dim=-1)


def to_steps(x, steps, step, frame, offset, mode="linear"):
    """``x`` [B, C, N] at the centres ``n * frame + offset``, interpolated to
    ``steps`` points ``s * step`` apart, [B, C, steps]: linearly, or with
    ``mode`` "cubic" by Catmull-Rom, whose slope has no kink at the centres
    (a linear one's kinks modulate the gains at the frame rate)."""
    if mode not in INTERPOLATIONS:
        raise ValueError(f"mode must be one of {INTERPOLATIONS}, not {mode!r}.")
    count = x.shape[-1]
    position = ((torch.arange(steps, device=x.device) * step - offset) / frame).clamp(0, count - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=count - 1)
    weight = (position - lower).to(x.dtype)
    if mode == "linear":
        return x[..., lower] * (1 - weight) + x[..., upper] * weight
    before, after = x[..., (lower - 1).clamp(min=0)], x[..., (lower + 2).clamp(max=count - 1)]
    a, b = x[..., lower], x[..., upper]
    return a + 0.5 * weight * (
        (b - before) + weight * ((2 * before - 5 * a + 4 * b - after) + weight * (3 * (a - b) + after - before))
    )


def cosine_sum(x, phase, first, last):
    """``sum(cos(k * x + phase) for k in first..last)`` elementwise, in closed
    form; zero where ``last < first``."""
    count = (last - first + 1).clamp(min=0)
    x = torch.remainder(x + math.pi, 2 * math.pi) - math.pi
    half = torch.sin(x / 2)
    small = half.abs() < 1e-9
    kernel = torch.where(small, count, torch.sin(count * x / 2) / torch.where(small, torch.ones_like(half), half))
    return kernel * torch.cos((first + last) * x / 2 + phase)



class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, kernel_size, layer_scale):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale)))

    def forward(self, x):
        y = self.dwconv(x).transpose(1, 2)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).transpose(1, 2)


class ChouwaGAN(nn.Module):
    """Normalised log mel [B, n_mels, T], f0 [B, T] (0 unvoiced) and, with
    ``contrast_band_edges``, optionally ``measure_contrast``'s [B, bands + 1, T]
    -> [B, 1, T * hop_length]."""

    def __init__(self, sample_rate, num_mels, hop_length, dim=256, intermediate_dim=768, num_blocks=8,
                 kernel_size=7, band_min_hz=100.0, band_ratio=0.03, glottal_band_min_hz=400.0,
                 glottal_band_ratio=0.15, glottal_max_hz=None, harmonic_window=1024, harmonic_fft=2048, kaiser_beta=8.0,
                 noise_window=512, noise_fft=1024, synthesis_hop=128, harmonic_substeps=2, harmonic_amp=0.01,
                 taper_hz=2000.0, max_jitter=0.02, max_shimmer=0.1, max_f0_correction=20.0, phase="mixed",
                 residual_bound=0.1, block_steps=4096, noise_curve_hz=None, aspiration_regions_hz=None,
                 max_aspiration_sharpness=16.0, contrast_band_edges=None, max_phase_spread=0.0, phase_head_hz=None,
                 phase_noise_mix=False, control_interpolation="linear", source_phase_hz=None):
        super().__init__()
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, not {phase!r}.")
        if control_interpolation not in INTERPOLATIONS:
            raise ValueError(f"control_interpolation must be one of {INTERPOLATIONS}, not {control_interpolation!r}.")
        # How the head's gains go from frames to synthesis steps.
        self.control_interpolation = control_interpolation
        if hop_length % synthesis_hop or hop_length % (synthesis_hop * harmonic_substeps):
            raise ValueError(
                f"synthesis_hop {synthesis_hop} x harmonic_substeps {harmonic_substeps} must divide the mel hop "
                f"{hop_length}."
            )
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.synthesis_hop = int(synthesis_hop)
        self.noise_substeps = self.hop_length // self.synthesis_hop
        self.harmonic_substeps = int(harmonic_substeps)
        self.harmonic_fft = int(harmonic_fft)
        self.noise_fft = int(noise_fft)
        self.harmonic_amp = float(harmonic_amp)
        self.taper_hz = float(taper_hz)
        self.max_jitter = float(max_jitter)
        self.max_shimmer = float(max_shimmer)
        self.max_f0_correction = float(max_f0_correction) / 1200.0
        self.residual_bound = float(residual_bound)
        self.block_steps = int(block_steps)
        self.max_phase_spread = float(max_phase_spread)
        self.source_phase_hz = None if source_phase_hz is None else float(source_phase_hz)
        if self.source_phase_hz is not None:
            generator = torch.Generator().manual_seed(SOURCE_PHASE_SEED)
            phases = math.pi * (2 * torch.rand(int(sample_rate / 2 / SOURCE_F0_FLOOR) + 2, generator=generator) - 1)
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1).view(1, -1, 1)
            ramp = ((bins - self.source_phase_hz) / SOURCE_PHASE_RAMP_HZ + 1).clamp(0, 1)
            self.register_buffer("source_phases", phases, persistent=False)
            self.register_buffer("source_bins", bins, persistent=False)
            self.register_buffer("source_ramp", ramp, persistent=False)

        band_hz = band_centers(sample_rate, band_min_hz, band_ratio)
        tract = band_matrix(band_hz, self.harmonic_fft, sample_rate)
        self.bands = tract.shape[0]
        self.glottal_bands = 0
        if phase == "zero":
            tract_log = tract.to(torch.complex128)
        else:
            tract_log = minimum_phase_basis(tract, self.harmonic_fft)
        self.register_buffer("tract_log", tract_log.to(torch.complex64), persistent=False)
        self.register_buffer("tract_bands", tract.float(), persistent=False)
        if self.max_phase_spread > 0:
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1, dtype=torch.float64)
            lower, upper, t = interpolation_weights(band_hz, bins)
            self.register_buffer("spread_lower", lower, persistent=False)
            self.register_buffer("spread_upper", upper, persistent=False)
            self.register_buffer("spread_weight", t.float().view(1, -1, 1), persistent=False)
            # Overlap-add sums random phases in power: sum(w^4) / sum(w^2)^2 of the
            # frames over each sample, as a share of the coherent sum.
            hop = self.synthesis_hop
            squares = torch.kaiser_window(int(harmonic_window), periodic=True, beta=kaiser_beta, dtype=torch.float64)
            squares = F.pad(squares.square(), (0, -int(harmonic_window) % hop)).view(-1, hop)
            self.spread_rho = float((squares.square().sum(0) / squares.sum(0).square()).mean())
        self.phase_bins = 0
        if phase_head_hz is not None:
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1)
            ramp = ((bins - float(phase_head_hz)) / PHASE_HEAD_RAMP_HZ + 1).clamp(0, 1)
            self.phase_first_bin = int((ramp > 0).nonzero()[0])
            self.phase_bins = len(bins) - self.phase_first_bin
            self.register_buffer("phase_ramp", ramp[self.phase_first_bin:].view(1, -1, 1), persistent=False)
        self.phase_noise_mix = bool(phase_noise_mix) and self.phase_bins > 0
        if phase == "mixed":
            centers = band_centers(sample_rate, glottal_band_min_hz, glottal_band_ratio)
            glottal = band_matrix(centers, self.harmonic_fft, sample_rate)
            if glottal_max_hz is not None:
                # Above it the glottal gain fades to neutral, and the tract's
                # narrow bands alone set how harmonic the top is: the wide
                # glottal ones were cutting it in blocks.
                glottal = glottal[centers <= float(glottal_max_hz)]
            self.glottal_bands = glottal.shape[0]
            glottal_log = minimum_phase_basis(glottal, self.harmonic_fft).conj()
            self.register_buffer("glottal_log", glottal_log.to(torch.complex64), persistent=False)
            bins = torch.linspace(0, sample_rate / 2, self.harmonic_fft // 2 + 1, dtype=torch.float64)
            nearest = torch.searchsorted(bins, band_hz).clamp(max=len(bins) - 1)
            self.register_buffer("glottal_at_bands", glottal[:, nearest].float(), persistent=False)
        self.curve_knots = 0
        if noise_curve_hz:
            knots = torch.tensor(sorted(float(hz) for hz in noise_curve_hz), dtype=torch.float64)
            self.curve_knots = len(knots)
            self.register_buffer("curve_bands", interpolation_matrix(knots, band_hz).float(), persistent=False)
        noise = band_matrix(band_centers(sample_rate, band_min_hz, band_ratio), self.noise_fft, sample_rate)
        self.register_buffer("noise_bands", noise.float(), persistent=False)
        self.aspiration_regions = 0
        if aspiration_regions_hz:
            knots = torch.tensor(sorted(float(hz) for hz in aspiration_regions_hz), dtype=torch.float64)
            self.aspiration_regions = len(knots)
            self.max_aspiration_sharpness = float(max_aspiration_sharpness)
            bins = torch.linspace(0, sample_rate / 2, self.noise_fft // 2 + 1, dtype=torch.float64)
            # Each region draws its own noise, so amplitudes whose squares sum to one.
            self.register_buffer("aspiration_masks", interpolation_matrix(knots, bins).sqrt().float(), persistent=False)
        self.contrast = BandContrast(sample_rate, hop_length, contrast_band_edges) if contrast_band_edges else None
        contrast_inputs = self.contrast.bands + 1 if self.contrast is not None else 0
        self.register_buffer(
            "harmonic_window", torch.kaiser_window(int(harmonic_window), periodic=True, beta=kaiser_beta),
            persistent=False,
        )
        self.register_buffer("noise_window", torch.hann_window(int(noise_window)), persistent=False)
        # Where in the glottal cycle the aspiration peaks; the pulse is at 0.
        self.aspiration_phase = nn.Parameter(
            torch.full((self.aspiration_regions,), math.pi) if self.aspiration_regions else torch.tensor(math.pi)
        )

        self.embed = nn.Conv1d(num_mels + 2 + contrast_inputs, dim, 7, padding=3)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, kernel_size, 1 / num_blocks) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        # Tract gains per harmonic sub-step, glottal gains, noise gains per
        # synthesis step, the residual's real and imaginary parts, then
        # the voiced noise curve's knots, then aspiration depth (and
        # sharpness, per region), jitter, shimmer and the f0 correction, then
        # the phase spread per tract band (last, so older exports load).
        self.splits = [
            self.bands * self.harmonic_substeps, self.glottal_bands, self.bands * self.noise_substeps,
            2 * self.bands if self.residual_bound > 0 else 0, self.curve_knots,
            2 * self.aspiration_regions + 3 if self.aspiration_regions else 4,
            self.bands if self.max_phase_spread > 0 else 0,
        ]
        self.head = nn.Linear(dim, sum(self.splits))
        # Each turned bin's (cos, sin), and with the noise mix its logit, per harmonic sub-step.
        self.phase_outputs = 3 if self.phase_noise_mix else 2
        self.phase_head = (
            nn.Linear(dim, self.phase_outputs * self.phase_bins * self.harmonic_substeps) if self.phase_bins else None
        )
        self.apply(self._init_weights)
        with torch.no_grad():
            noise_start = self.splits[0] + self.splits[1]
            # A harmonic at ``harmonic_amp`` and the noise ~30 dB under it at the start.
            self.head.bias[noise_start:noise_start + self.splits[2]] = -4.0
            if self.splits[3]:
                residual_start = noise_start + self.splits[2]
                self.head.weight[residual_start:residual_start + self.splits[3]] = 0.0
            curve_start = sum(self.splits[:4])
            self.head.bias[curve_start:curve_start + self.splits[4]] = -4.0
            if self.aspiration_regions:
                # Sharpness 1 at the start, close to the cosine.
                sharpness_start = curve_start + self.splits[4] + self.aspiration_regions
                self.head.bias[sharpness_start:sharpness_start + self.aspiration_regions] = -math.log(
                    self.max_aspiration_sharpness - 1
                )
            if self.splits[6]:
                self.head.bias[-self.splits[6]:] = SPREAD_INIT_BIAS
            if self.phase_head is not None:
                # No turn at the start: cos 1, sin 0.
                self.phase_head.weight.zero_()
                bias = self.phase_head.bias.view(self.phase_outputs, -1)
                bias[0], bias[1] = 1.0, 0.0
                if self.phase_noise_mix:
                    bias[2] = NOISE_MIX_INIT_BIAS

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # An export from before the phase spread has no rows for it: they
        # start nearly coherent, as a new model's do.
        weight, bias = state_dict.get(prefix + "head.weight"), state_dict.get(prefix + "head.bias")
        missing = self.splits[6]
        if missing and weight is not None and weight.shape[0] == sum(self.splits) - missing:
            state_dict[prefix + "head.weight"] = torch.cat((weight, weight.new_zeros(missing, weight.shape[1])))
            state_dict[prefix + "head.bias"] = torch.cat((bias, bias.new_full((missing,), SPREAD_INIT_BIAS)))
        # Nor a phase head: it starts turning nothing, as a new model's does;
        # one from before the noise mix gets its rows at the start's share.
        if self.phase_head is not None and weight is not None:
            head_weight = state_dict.get(prefix + "phase_head.weight")
            if head_weight is None:
                state_dict[prefix + "phase_head.weight"] = self.phase_head.weight.detach().clone()
                state_dict[prefix + "phase_head.bias"] = self.phase_head.bias.detach().clone()
            elif self.phase_noise_mix and head_weight.shape[0] == 2 * self.phase_head.weight.shape[0] // 3:
                rows = self.phase_head.weight.shape[0] // 3
                head_bias = state_dict[prefix + "phase_head.bias"]
                head_weight = torch.cat((head_weight, head_weight.new_zeros(rows, head_weight.shape[1])))
                head_bias = torch.cat((head_bias, head_bias.new_full((rows,), NOISE_MIX_INIT_BIAS)))
                state_dict[prefix + "phase_head.weight"] = head_weight
                state_dict[prefix + "phase_head.bias"] = head_bias
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def compile_blocks(self) -> None:
        for block in self.blocks:
            block.compile()

    def measure_contrast(self, audio, f0):
        """The ``contrast`` input from the audio itself ([B, 1, T] or [B, T])."""
        return known_contrast(self.contrast(audio, f0) / CONTRAST_SCALE)

    def _controls(self, mel, cf0, voiced, contrast):
        inputs = [mel, torch.log(cf0.clamp(min=SOURCE_F0_FLOOR) / 200.0).unsqueeze(1),
                  voiced.unsqueeze(1).to(mel.dtype)]
        if self.contrast is not None:
            if contrast is None:
                contrast = mel.new_zeros(mel.shape[0], self.contrast.bands + 1, mel.shape[-1])
            inputs.append(contrast.to(mel.dtype))
        x = self.embed(torch.cat(inputs, dim=1))
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x.transpose(1, 2))
        turns = self.phase_head(x).transpose(1, 2).float() if self.phase_head is not None else None
        return self.head(x).transpose(1, 2).float(), turns

    def _per_step(self, x, substeps, steps):
        """[B, C * substeps, frames] -> [B, C, steps] at the synthesis steps."""
        batch, channels, frames = x.shape
        channels //= substeps
        x = x.reshape(batch, channels, substeps, frames).transpose(2, 3).reshape(batch, channels, frames * substeps)
        size = self.hop_length // substeps
        return to_steps(x, steps, self.synthesis_hop, size, size // 2, self.control_interpolation)

    def _pulse(self, f0, voiced, shimmer):
        """The phase-aligned pulse train with every harmonic of ``f0`` below
        Nyquist, those in the top ``taper_hz`` faded by a raised cosine; the
        fade is itself a cosine in the harmonic number, so the faded band is
        three shifted closed-form sums. Everything at the sample rate."""
        nyquist = self.sample_rate / 2
        start = nyquist - self.taper_hz
        f0 = f0.double().clamp(min=SOURCE_F0_FLOOR)
        theta = 2 * math.pi * (torch.cumsum(f0 / self.sample_rate, dim=-1) % 1.0)
        flat = torch.floor(start / f0)
        top = torch.ceil(nyquist / f0) - 1
        slope = math.pi * f0 / self.taper_hz
        offset = -math.pi * start / self.taper_hz
        faded = (
            0.5 * cosine_sum(theta, 0.0, flat + 1, top)
            + 0.25 * cosine_sum(theta + slope, offset, flat + 1, top)
            + 0.25 * cosine_sum(theta - slope, -offset, flat + 1, top)
        )
        pulse = (cosine_sum(theta, 0.0, torch.ones_like(flat), flat) + faded).float() * self.harmonic_amp
        return pulse * voiced * (1 + shimmer), theta.float()

    def _phase_spread(self, spread):
        """Unit-mean-power random phases [B, bins, steps] for ``spread``
        [B, bands, steps] in radians: drawn per band in +-spread and
        interpolated across the bins between band centres, so each harmonic
        turns as a whole; scaled by the overlap-added power they keep."""
        lower, upper, t = self.spread_lower, self.spread_upper, self.spread_weight
        drawn = spread * (2 * torch.rand_like(spread) - 1)
        phase = drawn[:, lower] * (1 - t) + drawn[:, upper] * t
        coherent = torch.sinc(spread[:, lower] * (1 - t) / math.pi) * torch.sinc(spread[:, upper] * t / math.pi)
        power = coherent.square() + (1 - coherent.square()) * self.spread_rho
        return torch.polar(power.rsqrt(), phase)

    def _source_phase(self, f0):
        """Unit phasors [B, bins, steps] for ``f0`` [B, 1, steps]: each bin of
        the harmonic branch turns by the fixed phase of the harmonic of f0
        nearest to it, so every harmonic turns whole and keeps its phase as it
        moves; faded in below ``source_phase_hz``."""
        harmonic = torch.round(self.source_bins / f0).long().clamp(max=len(self.source_phases) - 1)
        phase = self.source_phases[harmonic] * self.source_ramp
        return torch.polar(torch.ones_like(phase), phase)

    def _noise_phase(self, noise):
        """``(start, end) -> [B, phase_bins, end - start]``: the unit phasors of
        ``noise``'s spectrogram over the harmonic branch's frames, from the
        phase head's first bin. A real signal's, so they stay coherent from
        one overlapping frame to the next, as independent draws would not."""
        n_fft, hop = self.harmonic_fft, self.synthesis_hop
        padded = F.pad(noise.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect").squeeze(1)
        left = (n_fft - self.harmonic_window.shape[0]) // 2
        window = F.pad(self.harmonic_window, (left, n_fft - self.harmonic_window.shape[0] - left))

        def phase(start, end):
            chunk = padded[:, start * hop:(end - 1) * hop + n_fft]
            spectrum = torch.stft(chunk, n_fft, hop, window=window, center=False, return_complex=True)
            spectrum = spectrum[:, self.phase_first_bin:]
            return spectrum / (spectrum.abs() + 1e-8)

        return phase

    def _render(self, signal, n_fft, window, gain, steps, samples):
        """The iSTFT, centred as ``torch.istft``'s, of ``signal``'s STFT times
        ``gain(start, end)`` ([B, bins, end - start]), a block of frames at a
        time: overlap-add sums over frames, so the blocks add up exactly."""
        hop = self.synthesis_hop
        padded = F.pad(signal.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect").squeeze(1)
        left = (n_fft - window.shape[0]) // 2
        window = F.pad(window, (left, n_fft - window.shape[0] - left))
        length = (steps - 1) * hop + n_fft
        out = signal.new_zeros(signal.shape[0], length)
        for start in range(0, steps, self.block_steps):
            end = min(steps, start + self.block_steps)
            chunk = padded[:, start * hop:(end - 1) * hop + n_fft]
            spectrum = torch.stft(chunk, n_fft, hop, window=window, center=False, return_complex=True)
            frames = torch.fft.irfft(spectrum * gain(start, end), n=n_fft, dim=1) * window.view(1, -1, 1)
            span = (end - start - 1) * hop + n_fft
            added = F.fold(frames, (1, span), (1, n_fft), stride=(1, hop)).flatten(1)
            out = out.index_add(1, torch.arange(start * hop, start * hop + span, device=out.device), added)
        envelope = F.fold(
            window.square().view(1, -1, 1).expand(1, -1, steps).contiguous(), (1, length), (1, n_fft), stride=(1, hop)
        ).flatten()
        return (out / envelope.clamp(min=1e-8))[:, n_fft // 2:n_fft // 2 + samples]

    def forward(self, mel, f0, contrast=None):
        f0 = f0.float()
        frames = f0.shape[-1]
        samples = frames * self.hop_length
        steps = samples // self.synthesis_hop + 1
        voiced = f0 > 0
        cf0 = continuous_f0(f0)
        controls, turns = self._controls(mel, cf0, voiced, contrast)

        with torch.autocast(mel.device.type, enabled=False):
            tract, glottal, noise, residual, curve, scalars, spread = controls.split(self.splits, dim=1)
            log_tract = self._per_step(tract, self.harmonic_substeps, steps).clamp(*LOG_GAIN_RANGE)
            log_glottal = self._per_step(glottal, 1, steps).clamp(*LOG_GAIN_RANGE) if self.glottal_bands else None
            log_noise = self._per_step(noise, self.noise_substeps, steps)
            if self.curve_knots:
                harmonic_level = log_tract
                if log_glottal is not None:
                    harmonic_level = harmonic_level + torch.einsum("bcs,cf->bfs", log_glottal, self.glottal_at_bands)
                curve = torch.einsum("bks,kf->bfs", self._per_step(curve, 1, steps), self.curve_bands)
                voicing = to_steps(voiced.float().unsqueeze(1), steps, self.synthesis_hop, self.hop_length,
                                   self.hop_length // 2)
                log_noise = voicing * (harmonic_level + curve) + (1 - voicing) * log_noise
            log_noise = log_noise.clamp(*LOG_GAIN_RANGE)
            residual = torch.tanh(self._per_step(residual, 1, steps)) if self.splits[3] else None
            if self.splits[6]:
                spread = self._per_step(torch.sigmoid(spread) * self.max_phase_spread, 1, steps)
            if turns is not None:
                parts = self._per_step(turns, self.harmonic_substeps, steps).chunk(self.phase_outputs, dim=1)
                turns = torch.polar(torch.ones_like(parts[0]), torch.atan2(parts[1], parts[0]) * self.phase_ramp)
                if self.phase_noise_mix:
                    mix = torch.sigmoid(parts[2]) * self.phase_ramp
                    noise_phase = self._noise_phase(torch.randn(f0.shape[0], samples, device=f0.device))
            frame_center = self.hop_length // 2
            aspiration, scalars = scalars.split([scalars.shape[1] - 3, 3], dim=1)
            jitter, shimmer = torch.sigmoid(scalars[:, :2]).unbind(1)
            correction = torch.tanh(scalars[:, 2]) * self.max_f0_correction
            gates = torch.cat((torch.sigmoid(aspiration), torch.stack(
                (jitter * self.max_jitter, shimmer * self.max_shimmer, correction, voiced.float(), cf0), dim=1
            )), dim=1)
            gates = to_steps(gates, samples, 1, self.hop_length, frame_center)
            aspiration = gates[:, :-5]
            jitter, shimmer, correction, gate, pitch = gates[:, -5:].unbind(1)

            # Jitter and shimmer as random walks smooth over a synthesis step.
            wobble = to_steps(torch.randn(f0.shape[0], 2, steps, device=f0.device), samples, 1, self.synthesis_hop, 0)
            pitch = pitch * torch.exp2(correction) * (1 + jitter * wobble[:, 0])
            pulse, theta = self._pulse(pitch, gate, shimmer * wobble[:, 1])
            if self.source_phase_hz is not None:
                # The pulse's own f0 at each frame centre.
                centres = (torch.arange(steps, device=pitch.device) * self.synthesis_hop).clamp(max=samples - 1)
                step_f0 = pitch.detach()[:, centres].clamp(min=SOURCE_F0_FLOOR).unsqueeze(1)

            def harmonic_gain(start, end):
                log_gain = torch.einsum("bcs,cf->bfs", log_tract[..., start:end].to(torch.complex64), self.tract_log)
                if log_glottal is not None:
                    log_gain = log_gain + torch.einsum(
                        "bcs,cf->bfs", log_glottal[..., start:end].to(torch.complex64), self.glottal_log
                    )
                gain = torch.exp(log_gain)
                if residual is not None:
                    real, imag = residual[..., start:end].chunk(2, dim=1)
                    real = torch.einsum("bcs,cf->bfs", real, self.tract_bands)
                    imag = torch.einsum("bcs,cf->bfs", imag, self.tract_bands)
                    gain = gain * (1 + self.residual_bound * torch.complex(real, imag))
                if self.splits[6]:
                    gain = gain * self._phase_spread(spread[..., start:end])
                if turns is not None:
                    first = self.phase_first_bin
                    turn = turns[..., start:end]
                    if self.phase_noise_mix:
                        # Unit mean power whatever the share, the two being independent.
                        share = mix[..., start:end]
                        blend = (1 - share) * turn + share * noise_phase(start, end)
                        turn = blend * ((1 - share).square() + share.square()).rsqrt()
                    gain = torch.cat((gain[:, :first], gain[:, first:] * turn), dim=1)
                if self.source_phase_hz is not None:
                    gain = gain * self._source_phase(step_f0[..., start:end])
                return gain

            def noise_gain(start, end):
                return torch.exp(torch.einsum("bcs,cf->bfs", log_noise[..., start:end], self.noise_bands))

            voice = self._render(pulse, self.harmonic_fft, self.harmonic_window, harmonic_gain, steps, samples)
            # Aspiration follows the glottal cycle where voiced; plain noise elsewhere.
            if not self.aspiration_regions:
                envelope = 1 + aspiration[:, 0] * gate * torch.cos(theta - self.aspiration_phase)
                breath = self._render(
                    torch.randn_like(pulse) * envelope, self.noise_fft, self.noise_window, noise_gain, steps, samples
                )
                return (voice + breath).unsqueeze(1)

            regions = self.aspiration_regions
            depth = aspiration[:, :regions] * gate.unsqueeze(1)
            sharpness = aspiration[:, regions:] * self.max_aspiration_sharpness
            # A von Mises pulse of mean 1: aspiration comes as a burst, which a
            # cosine cannot be, and it is more so at the top.
            cycle = torch.cos(theta.unsqueeze(1) - self.aspiration_phase.view(1, -1, 1)) - 1
            envelope = 1 + depth * (torch.exp(sharpness * cycle) / torch.special.i0e(sharpness) - 1)

            def region_gain(start, end):
                return (noise_gain(start, end).unsqueeze(1) * self.aspiration_masks.unsqueeze(-1)).flatten(0, 1)

            breath = self._render(
                (torch.randn_like(envelope) * envelope).flatten(0, 1), self.noise_fft, self.noise_window, region_gain,
                steps, samples,
            )
            return (voice + breath.view(-1, regions, samples).sum(1)).unsqueeze(1)


class HarmonicLoss(nn.Module):
    """The log magnitude at each harmonic of f0 (the peak within a quarter of
    f0 of it) and halfway between harmonics, each averaged over the harmonics
    of a band and the voiced frames of a clip; their difference (contrast) and
    mean (level), generated against target: the per-band harmonic-to-noise
    structure a mel loss cannot resolve. Frames are centred on the mel frames'
    centres. ``band_edges`` (Hz) split the harmonics. ``fft_size`` defaults
    to eight frames, so the window lasts about as long at every rate."""

    def __init__(
        self,
        sample_rate,
        hop_length,
        fft_size=None,
        nyquist_margin=0.95,
        band_edges=(2000, 4000, *range(5000, 21000, 1000)),
    ):
        super().__init__()
        fft_size = int(fft_size or 8 * hop_length)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.top = nyquist_margin * sample_rate / 2
        self.bin_hz = sample_rate / fft_size
        self.bands = len(band_edges) + 1
        self.register_buffer("window", torch.hann_window(fft_size), persistent=False)
        self.register_buffer("band_edges", torch.tensor(band_edges, dtype=torch.float32), persistent=False)

    def _log_spectrum(self, x, frames, hop=None):
        hop = hop or self.hop_length
        pad = self.fft_size // 2 - hop // 2
        x = F.pad(x, (pad, pad))
        spectrum = torch.stft(
            x, self.fft_size, hop, self.fft_size, self.window, center=False, return_complex=True
        ).abs()
        return torch.log(spectrum[..., :frames] + 1e-5).transpose(1, 2)

    def _harmonics(self, f0, spectra, scale=1.0):
        """For each log spectrum [B, T, bins] of ``spectra``, the peak at each
        harmonic of ``f0`` and the valley above it [B, T, H]; which harmonics
        are voiced and below the top, and their bands [B, T, H, bands]. The
        top and the band edges are divided by ``scale``."""
        voiced = f0 > 0
        top = self.top / scale
        low = f0[voiced].min().clamp(min=SOURCE_F0_FLOOR / scale)
        harmonics = torch.arange(1, int(top / low) + 1, device=f0.device, dtype=f0.dtype)
        peaks = f0.unsqueeze(-1) * harmonics
        valid = voiced.unsqueeze(-1) & (peaks + 0.5 * f0.unsqueeze(-1) < top)
        size = spectra[0].shape[-1]
        width = int(math.ceil(float(f0.max()) / 4 / self.bin_hz))
        offsets = torch.arange(-width, width + 1, device=f0.device)
        centre = torch.round(peaks / self.bin_hz).long()
        index = (centre.unsqueeze(-1) + offsets).clamp(0, size - 1)
        inside = offsets.abs() <= (f0 / 4 / self.bin_hz).unsqueeze(-1).unsqueeze(-1)
        position = ((peaks + 0.5 * f0.unsqueeze(-1)) / self.bin_hz).clamp(0, size - 1)
        lower = position.floor().long().clamp(max=size - 2)
        weight = position - lower

        def peak(spectrum):
            values = spectrum.gather(-1, index.flatten(2)).view(index.shape)
            return values.masked_fill(~inside, -1e4).amax(-1)

        def valley(spectrum):
            return spectrum.gather(-1, lower) * (1 - weight) + spectrum.gather(-1, lower + 1) * weight

        members = F.one_hot(torch.bucketize(peaks, self.band_edges / scale), self.bands) * valid.unsqueeze(-1)
        return [(peak(s), valley(s)) for s in spectra], valid, members.to(f0.dtype)

    def forward(self, y_hat, y, f0):
        """``y_hat`` and ``y`` [B, 1, T], ``f0`` [B, T // hop] (0 unvoiced)."""
        with torch.autocast(y.device.type, enabled=False):
            f0 = f0.float()
            if not (f0 > 0).any():
                return y.new_zeros(())
            frames = f0.shape[-1]
            spectra = (self._log_spectrum(y_hat.squeeze(1).float(), frames),
                       self._log_spectrum(y.squeeze(1).float(), frames))
            points, valid, members = self._harmonics(f0, spectra)
            # Single valleys are noise in both signals, so a point-wise match
            # floors at ~0.7; means over a clip's frames keep the level and
            # contrast only, and leave enough points for 1 kHz bands, which a
            # hole between better neighbours cannot hide in.
            count = members.sum((1, 2))

            def band_mean(values):
                return torch.einsum("bth,bthn->bn", values.masked_fill(~valid, 0), members) / count.clamp(min=1)

            def contrast_and_level(peaks, valleys):
                peaks_mean, valleys_mean = band_mean(peaks), band_mean(valleys)
                return peaks_mean - valleys_mean, (peaks_mean + valleys_mean) / 2

            # Matching peaks and valleys separately lets added noise raise the
            # peaks (a max over noise), so the contrast is matched as such.
            contrast_g, level_g = contrast_and_level(*points[0])
            contrast_t, level_t = contrast_and_level(*points[1])
            difference = (contrast_g - contrast_t).abs() + (level_g - level_t).abs()
            return difference[count > 0].mean()


class BandContrast(HarmonicLoss):
    """Per frame and band, the mean log contrast of the audio's harmonics over
    the points halfway between them; 0 where a band has none. Under half the
    loss's window by default, which vibrato smears less at the top."""

    def __init__(self, sample_rate, hop_length, band_edges, fft_size=None):
        super().__init__(sample_rate, hop_length, fft_size or 4 * hop_length, band_edges=tuple(band_edges))

    @torch.no_grad()
    def forward(self, audio, f0, hop=None, scale=1.0):
        """``audio`` [B, 1, T] or [B, T], ``f0`` [B, frames] -> [B, bands, frames],
        frames ``hop`` samples apart (the mel's by default). ``scale`` reads the
        bands as they would be with the audio's pitch and formants shifted up
        by that factor."""
        hop = hop or self.hop_length
        with torch.autocast(audio.device.type, enabled=False):
            f0 = f0.float()
            frames = f0.shape[-1]
            if not (f0 > 0).any():
                return f0.new_zeros(f0.shape[0], self.bands, frames)
            audio = audio.reshape(audio.shape[0], -1).float()
            audio = F.pad(audio, (0, max(0, frames * hop - audio.shape[-1])))
            ((peaks, valleys),), valid, members = self._harmonics(
                f0, (self._log_spectrum(audio, frames, hop),), scale
            )
            contrast = torch.einsum("bth,bthn->bnt", (peaks - valleys).masked_fill(~valid, 0), members)
            return contrast / members.sum(2).transpose(1, 2).clamp(min=1)


class EnvelopeLoss(nn.Module):
    """How fast each band's Hilbert envelope moves, generated against target:
    the log power spectrum, up to ``max_hz``, of the envelope over its mean,
    averaged over the clip. It sees the burst of each glottal cycle at the top
    and any ripple at the frame rate, which magnitude losses average away.
    ``band_edges`` (Hz) split the bands."""

    def __init__(self, sample_rate, band_edges=(3000, 6000, 10000, 16000), fft_size=2048, hop=512, max_hz=1500.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.edges = [min(float(hz), sample_rate / 2) for hz in band_edges]
        self.fft_size, self.hop = int(fft_size), int(hop)
        self.bins = int(max_hz * fft_size / sample_rate) + 1
        window = torch.hann_window(self.fft_size)
        self.register_buffer("window", window / window.sum(), persistent=False)

    def _envelopes(self, x):
        """[N, T] -> each band's envelope over its mean [N, bands, T], and that mean."""
        length = x.shape[-1]
        spectrum = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(length, 1 / self.sample_rate, device=x.device)
        masks = torch.stack([(freqs >= lo) & (freqs < hi) for lo, hi in zip(self.edges, self.edges[1:])])
        one_sided = spectrum.unsqueeze(1) * masks
        analytic = torch.fft.ifft(F.pad(one_sided, (0, length - one_sided.shape[-1])), n=length) * 2
        envelope = (analytic.real.square() + analytic.imag.square() + 1e-12).sqrt()
        mean = envelope.mean(-1, keepdim=True)
        return envelope / mean.clamp(min=1e-5), mean

    def forward(self, y_hat, y):
        """``y_hat`` and ``y`` [B, 1, T]."""
        with torch.autocast(y.device.type, enabled=False):
            batch = y.shape[0]
            envelopes, means = self._envelopes(torch.cat((y_hat, y)).squeeze(1).float())
            bands = envelopes.shape[1]
            spectrum = torch.stft(
                (envelopes - 1).flatten(0, 1), self.fft_size, self.hop, window=self.window, return_complex=True
            )[:, :self.bins]
            power = torch.log(spectrum.abs().square().mean(-1) + 1e-6).view(2, batch, bands, -1)
            # Bands silent in the target have no envelope to match.
            audible = (means[batch:] > 1e-4).to(power.dtype)
            difference = (power[0] - power[1]).abs().mean(-1, keepdim=True) * audible
            return difference.sum() / audible.sum().clamp(min=1)


def known_contrast(contrast):
    """The vocoder's ``contrast`` input from band contrasts [B, bands, T]
    already over ``CONTRAST_SCALE``: a channel of ones after them says they
    are known. Zeros, the input's absence, read as unknown."""
    return torch.cat((contrast, torch.ones_like(contrast[:, :1])), dim=1)


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN's period discriminator, as Wavehax sizes it (ported from
    https://github.com/chomeyama/wavehax, MIT)."""

    def __init__(self, period, channels=32, kernel_sizes=(5, 3), downsample_scales=(3, 3, 3, 3, 1),
                 max_channels=1024, use_san=False):
        super().__init__()
        self.period = period
        self.use_san = bool(use_san)
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, channels
        for scale in downsample_scales:
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_sizes[0], 1), (scale, 1), ((kernel_sizes[0] - 1) // 2, 0)
            )))
            in_channels, out_channels = out_channels, min(out_channels * 4, max_channels)
        output = dict(kernel_size=(kernel_sizes[1], 1), padding=((kernel_sizes[1] - 1) // 2, 0))
        if self.use_san:
            self.output_conv = SANConv2d(out_channels, 1, **output)
        else:
            self.output_conv = weight_norm(nn.Conv2d(out_channels, 1, **output))

    def forward(self, x, san_training=False):
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for index, conv in enumerate(self.convs):
            if index:
                x = conv(x)
            else:
                # On the waveform, out of autocast: BF16 would keep 8 bits of each sample.
                with torch.autocast(x.device.type, enabled=False):
                    x = conv(x.float())
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        return _output(self, x, san_training, flatten=True), fmap


class SpectralDiscriminator(nn.Module):
    """UnivNet's spectrogram discriminator, as Wavehax sizes it.

    ``compression`` under 1 raises the magnitude to that power, which brings
    the quiet high band up to the scale of the low one: the kernels are shared
    along frequency and cannot weigh it up themselves. ``floor`` is added
    first and bounds the slope on quiet bins. ``first_channels`` narrows the
    input and the first conv, which runs at the full resolution and costs
    most of the head. With ``phase``, from bin ``min_bin`` up, the input is
    how the phase moves: see ``_phase_relations``."""

    KERNEL_SIZES = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3), (3, 3))
    STRIDES = ((2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1))

    def __init__(self, fft_size, hop_size, win_length, channels=32, compression=1.0, floor=COMPRESSION_FLOOR,
                 use_san=False, first_channels=None, phase=False, min_bin=0):
        super().__init__()
        self.use_san = bool(use_san)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compression = float(compression)
        self.floor = float(floor)
        self.phase = bool(phase)
        self.min_bin = int(min_bin)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        first = int(first_channels or channels)
        self.input_conv = weight_norm(nn.Conv2d(5 if self.phase else 1, first, 1))
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(
                first if index == 0 else channels, channels, kernel,
                padding=(kernel[0] // 2, kernel[1] // 2), stride=stride,
            ))
            for index, (kernel, stride) in enumerate(zip(self.KERNEL_SIZES, self.STRIDES))
        )
        self.output_conv = SANConv2d(channels, 1, 1) if self.use_san else weight_norm(nn.Conv2d(channels, 1, 1))

    def forward(self, x, san_training=False):
        x = torch.stft(
            x.squeeze(1).float(), self.fft_size, self.hop_size, self.win_length, self.window,
            center=True, return_complex=True,
        )[:, self.min_bin:]
        if self.phase:
            x = self._phase_relations(x)
        else:
            x = x.abs().unsqueeze(1)
            if self.compression != 1.0:
                x = (x / (0.5 * self.win_length) + self.floor) ** self.compression
        # On the magnitude, out of autocast: unnormalised, FP16's range is
        # short for its gradient.
        with torch.autocast(x.device.type, enabled=False):
            x = F.leaky_relu(self.input_conv(x.float()), 0.1)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return _output(self, x, san_training), fmap

    def _phase_relations(self, spec):
        """[B, 5, bins, frames] from a [B, bins, frames] STFT: the compressed
        magnitude, and as phasors it weights, the phase advance over a hop
        less the bin's own and the phase step from the bin below. Raw real
        and imaginary parts turn by tens of radians a frame up here, which
        convolutions barely read; these relations they do. The band is first
        brought to unit RMS, its level being the magnitude heads' to judge."""
        spec = spec / (spec.abs().square().mean((1, 2), keepdim=True).sqrt() + 1e-8)
        magnitude = (spec.abs() + PHASE_FLOOR) ** self.compression - PHASE_FLOOR ** self.compression
        bins = torch.arange(self.min_bin, self.min_bin + spec.shape[1], device=spec.device)
        advance_angle = -2 * math.pi * bins * self.hop_size / self.fft_size
        own = torch.polar(torch.ones(bins.shape, device=spec.device), advance_angle)
        advance = spec[..., 1:] * spec[..., :-1].conj() * own.view(1, -1, 1)
        step = spec[:, 1:] * spec[:, :-1].conj()
        # Over a floor, so a quiet bin's phasor shrinks and its slope stays bounded.
        advance = advance / (advance.abs() + PHASE_FLOOR ** 2) * magnitude[..., 1:]
        step = step / (step.abs() + PHASE_FLOOR ** 2) * magnitude[:, 1:]
        return torch.cat((
            magnitude.unsqueeze(1),
            F.pad(torch.view_as_real(advance).permute(0, 3, 1, 2), (1, 0)),
            F.pad(torch.view_as_real(step).permute(0, 3, 1, 2), (0, 0, 1, 0)),
        ), 1)


def _output(head, x, san_training, flatten=False):
    """The head's last conv; a SAN head in the discriminator update gives the
    (function, direction) pair the SAN losses take."""
    shape = (lambda y: torch.flatten(y, 1, -1)) if flatten else (lambda y: y)
    if not head.use_san:
        return shape(head.output_conv(x))
    out = head.output_conv(x, san_training=san_training)
    return tuple(shape(y) for y in out) if san_training else shape(out)


class Heads(nn.Module):
    """A family of discriminators: ``(outputs, feature maps)``, one per head."""

    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, san_training=False):
        """With ``san_training``, a SAN head's output is its (function, direction) pair."""
        outputs, fmaps = [], []
        for discriminator in self.discriminators:
            output, fmap = discriminator(y, san_training)
            outputs.append(output)
            fmaps.append(fmap)
        return outputs, fmaps


class EnvelopeDiscriminator(nn.Module):
    """How the loudness of each band between ``band_edges_hz`` moves within
    and across glottal cycles: its Hilbert envelope in log, less its mean
    over the clip (the level is the spectral losses'), at 1 / ``downsample``
    of the rate, through narrow strided convs. Magnitude only: on the raw
    waveform the top is ~40 dB under the low band, and a multi-band view with
    phase was seen to kill the generator."""

    def __init__(self, sample_rate, band_edges_hz, downsample=4, channels=(32, 64, 128, 128, 128),
                 strides=(2, 2, 2, 2, 1), kernel_size=9, ramp_hz=500.0, use_san=False):
        super().__init__()
        self.use_san = bool(use_san)
        self.sample_rate = float(sample_rate)
        self.band_edges = sorted(float(hz) for hz in band_edges_hz)
        self.downsample = int(downsample)
        self.ramp_hz = float(ramp_hz)
        self.convs = nn.ModuleList()
        in_channels = len(self.band_edges) - 1
        for out_channels, stride in zip(channels, strides):
            self.convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0)
            )))
            in_channels = out_channels
        output = dict(kernel_size=(3, 1), padding=(1, 0))
        if self.use_san:
            self.output_conv = SANConv2d(in_channels, 1, **output)
        else:
            self.output_conv = weight_norm(nn.Conv2d(in_channels, 1, **output))

    def _masks(self, freqs):
        """[bands, bins]: each band with linear ramps ``ramp_hz`` wide at its
        edges, none at Nyquist."""
        nyquist = self.sample_rate / 2
        masks = []
        for low, high in zip(self.band_edges[:-1], self.band_edges[1:]):
            mask = ((freqs - low) / self.ramp_hz + 0.5).clamp(0, 1)
            if high < nyquist:
                mask = mask * ((high - freqs) / self.ramp_hz + 0.5).clamp(0, 1)
            masks.append(mask)
        return torch.stack(masks)

    def forward(self, x, san_training=False):
        # The transforms and the first conv in FP32, as the period heads' first.
        with torch.autocast(x.device.type, enabled=False):
            x = x.squeeze(1).float()
            samples = x.shape[-1]
            freqs = torch.fft.rfftfreq(samples, 1.0 / self.sample_rate, device=x.device)
            # The analytic signal: positive frequencies doubled, the rest zero.
            spectrum = torch.fft.rfft(x).unsqueeze(1) * (2 * self._masks(freqs))
            analytic = torch.fft.ifft(spectrum, n=samples)
            envelope = (analytic.real.square() + analytic.imag.square() + 1e-12).sqrt()
            envelope = F.avg_pool1d(envelope, self.downsample)
            log = torch.log(envelope + 1e-3 * envelope.mean(-1, keepdim=True) + 1e-8)
            x = F.leaky_relu(self.convs[0]((log - log.mean(-1, keepdim=True)).unsqueeze(-1)), 0.1)
        fmap = [x]
        for conv in self.convs[1:]:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        return _output(self, x, san_training, flatten=True), fmap


def build_families(settings: dict, sample_rate: int) -> dict:
    """ChouwaGAN's discriminator: compressed multi-resolution spectrograms,
    narrowed at the input by ``mrd_first_channels``; if ``periods`` names
    any, narrow period heads; with ``envelope_bands_hz``, the envelope head;
    with ``phase_fft_sizes``, phase heads above ``phase_min_hz``;
    all ending in SAN projections with ``use_san``. ``mrd_floor`` bounds the
    compression's slope, which otherwise sends most of the MRD's gradient to
    quiet bins."""
    san = bool(settings.get("use_san", False))
    spectral = dict(
        compression=settings.get("mrd_compression", 0.3),
        floor=settings.get("mrd_floor", COMPRESSION_FLOOR),
        first_channels=settings.get("mrd_first_channels"),
        use_san=san,
    )
    families = {"mrd": Heads(
        SpectralDiscriminator(*sizes, **spectral)
        for sizes in zip(settings["mrd_fft_sizes"], settings["mrd_hop_sizes"], settings["mrd_win_lengths"])
    )}
    if settings.get("periods"):
        families["mpd"] = Heads(
            PeriodDiscriminator(p, max_channels=int(settings.get("mpd_max_channels", 256)), use_san=san)
            for p in settings["periods"]
        )
    if settings.get("envelope_bands_hz"):
        families["env"] = Heads([EnvelopeDiscriminator(
            sample_rate, settings["envelope_bands_hz"], int(settings.get("envelope_downsample", 4)), use_san=san
        )])
    if settings.get("phase_fft_sizes"):
        # The magnitude views cannot see how the high harmonics line up in time.
        low = float(settings.get("phase_min_hz", 0.0))
        families["phase"] = Heads(
            SpectralDiscriminator(*sizes, phase=True, min_bin=math.ceil(low * sizes[0] / sample_rate), **spectral)
            for sizes in zip(settings["phase_fft_sizes"], settings["phase_hop_sizes"], settings["phase_win_lengths"])
        )
    return families
