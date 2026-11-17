"""Nagi: a time-domain vocoder whose audio path is linear in the audio.

Band-limited sources (a pulse train with every harmonic of f0 in phase, one
with a scattered, slowly drifting phase per harmonic, and white noise, plain
and modulated by the glottal cycle) go through time-varying FIR filters, applied
by convolution in blocks crossfaded under a Hann window. A network at the
frame rate, then at ``control_ratio`` times it, predicts the filters' band
gains and phases; the taps are designed from them at that rate. No
nonlinearity touches an audio signal and nothing is resampled at the audio
rate, so nothing folds or mirrors, whatever the weights.

With ``free_branch``, a second network reads the filtered sources' own band
envelopes (the same bank of windowed sinusoids, as an analysis) and writes
complex band envelopes that the bank turns into audio, for what filtered
sources cannot make. It is bounded by the mel's level in each band, and only
what moves in an envelope is rendered: a steady one is a tone at the band's
centre.
"""

import math

import torch
from librosa import mel_frequencies
from torch import nn
from torch.nn import functional as F

from rvc.rectified.chouwagan import (
    CONTRAST_SCALE,
    SOURCE_F0_FLOOR,
    BandContrast,
    ConvNeXtBlock,
    band_centers,
    band_matrix,
    continuous_f0,
    cosine_sum,
    interpolation_matrix,
    known_contrast,
    minimum_phase_basis,
    to_steps,
)

try:
    from rvc.rectified import nagi_triton
except ImportError:
    nagi_triton = None

#: ``architecture`` of a Nagi rectified vocoder export, and of its runs.
ARCHITECTURE = "nagi"
#: Range of the filters' log gains.
LOG_GAIN_RANGE = (-14.0, 5.0)
#: Where the log gains start over the mel's value at their frequency: roughly
#: a harmonic's amplitude, and a white noise's, over the mel they give.
PULSE_GAIN_BIAS = -1.5
NOISE_GAIN_BIAS = -1.0
#: The scattered pulse takes over from the aligned one across this range at init.
#: A logit the sigmoid has not saturated at: from 4 the share never moved.
SCATTER_HZ = (3000.0, 4000.0)
SCATTER_INIT_LOGIT = 1.0
SCATTER_SEED = 0
#: Knots per second of the scattered phases' drift.
SCATTER_DRIFT_HZ = 8.0
#: The crossfade window spans this many control hops.
OVERLAP = 4
#: Frames each side over which the free branch's envelopes lose their mean.
FREE_DETREND_FRAMES = 1
#: Nepers the pulses start under the mel outside voiced frames: the network
#: can bring them back where the pitch tracker missed a voiced one.
UNVOICED_DROP = 6.0
#: Seconds of the window the lock reads the fundamental over: its first null
#: is at 50 Hz, under any f0, so neither DC nor the second harmonic leaks in.
LOCK_SECONDS = 0.04


def locked_turns(turns, audio, sample_rate):
    """``turns`` [B, T] (cycles, in double) turned so that their cosine is in
    phase with the fundamental of ``audio`` [B, T], and that fundamental's
    amplitude [B, T]. The audio is brought down by ``turns`` and smoothed over
    ``LOCK_SECONDS``: what is left is how far its fundamental has run from
    them, which also takes in a pitch error of a few Hz."""
    angle = 2 * math.pi * turns
    length = int(LOCK_SECONDS * sample_rate) | 1
    window = torch.hann_window(length, periodic=False, dtype=turns.dtype, device=turns.device)
    audio = audio.to(turns.dtype)
    mixed = torch.stack((audio * torch.cos(angle), -audio * torch.sin(angle)), dim=1)
    low = F.conv1d(mixed, (window / window.sum()).expand(2, 1, -1), padding=length // 2, groups=2)
    offset = torch.atan2(low[:, 1], low[:, 0]) / (2 * math.pi)
    return torch.remainder(turns + offset, 1.0), 2 * torch.hypot(low[:, 0], low[:, 1]).float()


def class_pulse(turns, f0, phases, start, taper):
    """``sum_k w_k cos(k theta + phases[k % classes])`` over the harmonics
    of ``f0`` [B, T] under ``start + taper`` Hz, faded by a raised cosine over
    the last ``taper``; ``turns`` [B, T] is theta in cycles, in double, and
    ``phases`` is [classes], or [B, classes, T] when they move. Each phase
    class is an arithmetic progression of harmonics, so it has
    ``cosine_sum``'s closed form, and so has its faded part."""
    if phases.dim() == 1:
        phases = phases.view(1, -1, 1)
    classes = phases.shape[1]
    index = torch.arange(classes, device=f0.device).view(1, -1, 1)
    turns = turns.unsqueeze(1)
    x = (2 * math.pi * torch.remainder(classes * turns, 1.0)).float()
    phase = (2 * math.pi * torch.remainder(index * turns, 1.0)).float() + phases
    f0, index = f0.unsqueeze(1), index.float()
    # Class 0 starts at its second member: the first is DC.
    first = (index == 0).float()
    flat = torch.floor((start / f0 - index) / classes)
    top = torch.ceil(((start + taper) / f0 - index) / classes) - 1
    slope = math.pi * classes * f0 / taper
    offset = math.pi * (index * f0 - start) / taper
    fade_first = torch.maximum(flat + 1, first)
    faded = (
        0.5 * cosine_sum(x, phase, fade_first, top)
        + 0.25 * cosine_sum(x + slope, phase + offset, fade_first, top)
        + 0.25 * cosine_sum(x - slope, phase - offset, fade_first, top)
    )
    return (cosine_sum(x, phase, first.expand_as(flat), flat) + faded).sum(1)


def free_band_centers(min_hz, ratio, max_hz, top):
    """The free branch's band centres up to ``top``: as ``band_centers``, but
    none at 0 Hz and never more than ``max_hz`` apart."""
    centers = [float(min_hz)]
    while True:
        following = centers[-1] + min(max(min_hz, ratio * centers[-1]), max_hz)
        if following > top:
            return torch.tensor(centers, dtype=torch.float64)
        centers.append(following)


def free_atoms(centers, sample_rate, hop, min_hz, ratio, max_hz):
    """[2 * bands, 1, taps]: each band's Hann-windowed cosine, then its sine,
    the window two band spacings wide in frequency and at least ``OVERLAP``
    hops long; scaled so that hop-spaced copies add up to the sinusoid."""
    lengths = []
    for hz in centers.tolist():
        spacing = min(max(min_hz, ratio * hz), max_hz)
        lengths.append(max(int(2 * sample_rate / spacing), OVERLAP * hop) | 1)
    taps = max(lengths)
    atoms = torch.zeros(2, len(lengths), taps, dtype=torch.float64)
    for band, (hz, length) in enumerate(zip(centers.tolist(), lengths)):
        position = torch.arange(length, dtype=torch.float64) - (length - 1) / 2
        window = torch.hann_window(length, periodic=False, dtype=torch.float64) * (2 * hop / length)
        angle = 2 * math.pi * hz * position / sample_rate
        begin = (taps - length) // 2
        atoms[0, band, begin:begin + length] = window * torch.cos(angle)
        atoms[1, band, begin:begin + length] = window * torch.sin(angle)
    return atoms.view(-1, 1, taps).float()


class Nagi(nn.Module):
    """Normalised log mel [B, n_mels, T], f0 [B, T] (0 unvoiced) and, with
    ``contrast_band_edges``, optionally ``measure_contrast``'s [B, bands + 1, T]
    -> [B, 1, T * hop_length].

    The pulses' filter has a gain per band ``band_min_hz`` apart (a
    ``band_ratio`` of their frequency above), the mel's value there plus the
    network's correction, with the minimum phase of that envelope plus a
    predicted turn per band; a predicted share of it goes to the scattered
    pulse, whose phases wander ``scatter_drift`` radians (0 holds them). The
    noise has coarser bands, and per band a gain for each of its
    ``noise_harmonics`` modulations by the glottal cycle; its filters change
    ``noise_control_ratio`` times per control step. ``pulse_taps`` and
    ``noise_taps`` are the filters' lengths. No harmonic is made in the top
    ``top_margin_hz``, which the crossfade's spread needs, and those in the
    ``taper_hz`` under it fade out.

    The free branch has ``free_blocks`` blocks of ``free_dim`` channels at the
    control rate. ``free_bound`` scales its ceiling. ``free_keep`` [B, 1],
    when set, mutes the branch for the items it marks false. The mel's
    ``mel_fmin``, ``mel_fmax`` and normalisation are the trainer's to fill in.
    """

    def __init__(self, sample_rate, num_mels, hop_length, dim=256, intermediate_dim=768, num_blocks=6,
                 kernel_size=7, control_ratio=8, control_dim=128, control_blocks=3, band_min_hz=100.0,
                 band_ratio=0.03, noise_band_min_hz=300.0, noise_band_ratio=0.05, pulse_taps=1024, noise_taps=256,
                 noise_harmonics=6, noise_control_ratio=2, top_margin_hz=1000.0, taper_hz=1000.0, phase_classes=32,
                 scatter_drift=1.0, free_branch=True, free_dim=512, free_blocks=8, free_bound=1.0,
                 contrast_band_edges=None,
                 mel_fmin=0.0, mel_fmax=None, mel_mean=0.0, mel_std=1.0):
        super().__init__()
        if hop_length % (control_ratio * noise_control_ratio):
            raise ValueError(
                f"control_ratio {control_ratio} times noise_control_ratio {noise_control_ratio} "
                f"must divide the hop {hop_length}."
            )
        if pulse_taps % 2 or noise_taps % 2:
            raise ValueError("pulse_taps and noise_taps must be even.")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.control_hop = self.hop_length // int(control_ratio)
        self.noise_ratio = int(noise_control_ratio)
        self.noise_hop = self.control_hop // self.noise_ratio
        self.scatter_drift = float(scatter_drift)
        self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
        self.pulse_taps, self.noise_taps = int(pulse_taps), int(noise_taps)
        self.pulse_fft, self.noise_fft = 2 * self.pulse_taps, 2 * self.noise_taps
        # The minimum-phase response is causal; the predicted turn may lead it.
        self.pulse_pre = self.pulse_taps // 4
        self.noise_harmonics = int(noise_harmonics)
        self.taper_hz = float(taper_hz)
        self.pulse_top = self.sample_rate / 2 - float(top_margin_hz)
        self.free_keep = None
        self.lock_level = self.lock_turns = None

        self.contrast = BandContrast(sample_rate, hop_length, contrast_band_edges) if contrast_band_edges else None
        contrast_inputs = self.contrast.bands + 1 if self.contrast is not None else 0

        centers = band_centers(sample_rate, band_min_hz, band_ratio)
        noise_centers = band_centers(sample_rate, noise_band_min_hz, noise_band_ratio)
        self.bands, self.noise_bands = len(centers), len(noise_centers)
        mel_centers = torch.from_numpy(
            mel_frequencies(n_mels=num_mels + 2, fmin=mel_fmin, fmax=mel_fmax or sample_rate / 2)[1:-1]
        ).double()
        to_bins = band_matrix(centers, self.pulse_fft, sample_rate)
        self.register_buffer("band_bins", to_bins.float(), persistent=False)
        self.register_buffer("phase_bins", minimum_phase_basis(to_bins, self.pulse_fft).imag.float(),
                             persistent=False)
        self.register_buffer("noise_bins", band_matrix(noise_centers, self.noise_fft, sample_rate).float(),
                             persistent=False)
        shortcuts = [interpolation_matrix(mel_centers, centers), interpolation_matrix(mel_centers, noise_centers)]

        post = self.pulse_taps - self.pulse_pre
        self.register_buffer("pulse_fade", torch.cat((
            torch.hann_window(2 * self.pulse_pre + 1, periodic=False)[:self.pulse_pre],
            torch.hann_window(2 * post + 1, periodic=False)[post:2 * post],
        )), persistent=False)
        self.register_buffer("noise_fade", torch.hann_window(self.noise_taps), persistent=False)
        # Hop-spaced copies add up to one.
        self.register_buffer("window", torch.hann_window(OVERLAP * self.control_hop) * 2 / OVERLAP,
                             persistent=False)
        self.register_buffer("noise_window", torch.hann_window(OVERLAP * self.noise_hop) * 2 / OVERLAP,
                             persistent=False)
        generator = torch.Generator().manual_seed(SCATTER_SEED)
        self.register_buffer("scatter_phases", 2 * math.pi * torch.rand(int(phase_classes), generator=generator),
                             persistent=False)
        self.register_buffer("aligned_phases", torch.zeros(1), persistent=False)

        self.embed = nn.Conv1d(num_mels + 2 + contrast_inputs, dim, kernel_size, padding=kernel_size // 2)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, kernel_size, 1 / num_blocks) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.project = nn.Linear(dim, control_dim)
        self.control_blocks = nn.ModuleList(
            ConvNeXtBlock(control_dim, 3 * control_dim, kernel_size, 1 / control_blocks)
            for _ in range(control_blocks)
        )
        self.control_norm = nn.LayerNorm(control_dim, eps=1e-6)
        # Per pulse band a gain, a scattered share and a turn; per noise band a gain and its
        # modulations', each once per noise step.
        noise_outputs = self.noise_ratio * self.noise_bands
        self.splits = [self.bands] * 3 + [noise_outputs, 2 * self.noise_harmonics * noise_outputs]
        self.head = nn.Linear(control_dim, sum(self.splits))

        # Control steps computed past each end: the crossfade needs one, the free branch's atoms half their length.
        self.margin = 1
        self.free_head = None
        if free_branch:
            max_hz = self.sample_rate / (2 * self.control_hop)
            free_centers = free_band_centers(band_min_hz, band_ratio, max_hz, self.pulse_top)
            atoms = free_atoms(free_centers, sample_rate, self.control_hop, band_min_hz, band_ratio, max_hz)
            self.free_bands = len(free_centers)
            self.free_log_bound = math.log(float(free_bound)) + PULSE_GAIN_BIAS
            self.margin = math.ceil((atoms.shape[-1] - 1) / 2 / self.control_hop) + 1
            self.free_detrend = 2 * FREE_DETREND_FRAMES * int(control_ratio) + 1
            self.register_buffer("atoms", atoms, persistent=False)
            self.register_buffer("atom_norm", atoms.square().sum(-1).sqrt().clamp(min=1e-8).view(1, -1, 1),
                                 persistent=False)
            self.register_buffer("free_centers", free_centers, persistent=False)
            shortcuts.append(interpolation_matrix(mel_centers, free_centers))
            # The noise's envelopes and the filtered sources'.
            self.free_input = nn.Conv1d(4 * self.free_bands, free_dim, 1)
            self.free_project = nn.Linear(control_dim, free_dim)
            self.free_blocks = nn.ModuleList(
                ConvNeXtBlock(free_dim, 3 * free_dim, kernel_size, 1 / free_blocks) for _ in range(free_blocks)
            )
            self.free_norm = nn.LayerNorm(free_dim, eps=1e-6)
            self.free_head = nn.Linear(free_dim, 2 * self.free_bands)
        self.register_buffer("mel_shortcut", torch.cat(shortcuts, dim=1).float(), persistent=False)

        self.apply(self._init_weights)
        with torch.no_grad():
            tract, scatter, _, noise, _ = self.head.bias.split(self.splits)
            tract.fill_(PULSE_GAIN_BIAS)
            ramp = ((centers - SCATTER_HZ[0]) / (SCATTER_HZ[1] - SCATTER_HZ[0])).clamp(0, 1).float()
            scatter.copy_(SCATTER_INIT_LOGIT * (2 * ramp - 1))
            noise.fill_(NOISE_GAIN_BIAS)
            if self.free_head is not None:
                # Small, not zero: a branch that starts silent has no gradient to leave it.
                self.free_head.weight.mul_(0.1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def compile_blocks(self) -> None:
        free = self.free_blocks if self.free_head is not None else ()
        for block in (*self.blocks, *self.control_blocks, *free):
            block.compile()

    def measure_contrast(self, audio, f0):
        """The ``contrast`` input from the audio itself ([B, 1, T] or [B, T])."""
        return known_contrast(self.contrast(audio, f0) / CONTRAST_SCALE)

    def _sources(self, cf0, samples, target=None):
        """The pulses [B, 2, T] and the noises [B, 1 + 2 * noise_harmonics, T]:
        white, then its products with the cosine and sine of each harmonic of
        the glottal cycle. With ``target`` [B, T], the cycle is locked to that
        audio's fundamental, whose amplitude is kept in ``lock_level``, and
        kept in ``lock_turns`` [B, T]."""
        f0 = F.interpolate(cf0.unsqueeze(1), size=samples, mode="linear", align_corners=False).squeeze(1)
        f0 = f0.clamp(min=SOURCE_F0_FLOOR)
        increment = f0.double() / self.sample_rate
        if self.training:
            increment[:, :1] += torch.rand(f0.shape[0], 1, device=f0.device, dtype=torch.float64)
        turns = torch.remainder(torch.cumsum(increment, dim=-1), 1.0)
        self.lock_level = self.lock_turns = None
        if target is not None:
            turns, self.lock_level = locked_turns(turns, target, self.sample_rate)
            self.lock_turns = turns
        start = self.pulse_top - self.taper_hz
        scatter = self.scatter_phases
        if self.scatter_drift > 0:
            # Held, the scattered phases are one frozen pattern; the voice's wander up there.
            knots = math.ceil(samples * SCATTER_DRIFT_HZ / self.sample_rate) + 2
            drift = torch.randn(f0.shape[0], len(scatter), knots, device=f0.device)
            drift = F.interpolate(drift, size=samples, mode="linear", align_corners=True)
            scatter = scatter.view(1, -1, 1) + self.scatter_drift * drift
        pulses = torch.stack((
            class_pulse(turns, f0, self.aligned_phases, start, self.taper_hz),
            class_pulse(turns, f0, scatter, start, self.taper_hz),
        ), dim=1)
        white = torch.randn_like(f0)
        noises = [white]
        for harmonic in range(1, self.noise_harmonics + 1):
            theta = (2 * math.pi * torch.remainder(harmonic * turns, 1.0)).float()
            noises += [white * torch.cos(theta), white * torch.sin(theta)]
        return pulses, torch.stack(noises, dim=1)

    def _analysis(self, audio):
        """``audio`` [B, T] on the free branch's atoms at every control step,
        [B, 2 * free_bands, steps]."""
        pad = self.margin * self.control_hop + (self.atoms.shape[-1] - 1) // 2
        padded = F.pad(audio.unsqueeze(1), (pad, pad + 1))
        return F.conv1d(padded, self.atoms, stride=self.control_hop)

    def _carriers(self, steps):
        """The cosine and sine [free_bands, steps] of each band's carrier,
        run from sample 0, at the control steps' centres."""
        centre = (torch.arange(steps, device=self.atoms.device, dtype=torch.float64) - self.margin) * self.control_hop
        turn = 2 * math.pi * torch.remainder(torch.outer(self.free_centers.double(), centre) / self.sample_rate, 1.0)
        return torch.cos(turn).float(), torch.sin(turn).float()

    def _heard(self, audio, log_bound, carriers):
        """What the filtered sources ``audio`` [B, T] put in each band, as the
        envelopes ``_free`` writes: over the band's bound, and compressed so
        that none passes one."""
        cos, sin = carriers
        # A sinusoid at a band's centre reads half a hop of its amplitude.
        projected = self._analysis(audio) * (2 / self.control_hop)
        a, b = projected.reshape(audio.shape[0], 2, self.free_bands, -1).unbind(1)
        envelope = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=1) * torch.exp(-log_bound).unsqueeze(1)
        return (envelope / (1 + envelope.square().sum(1, keepdim=True).sqrt())).flatten(1, 2)

    def _controls(self, mel, cf0, voiced, contrast, steps):
        """The head's outputs [B, sum(splits), steps], in FP32, and the
        features [B, control_dim, steps] they are read from."""
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
        x = self.project(self.output_norm(x.transpose(1, 2))).transpose(1, 2)
        # Interpolated, not repeated: steps per frame would modulate the gains at the frame rate.
        x = self._to_steps(x, steps, "cubic")
        for block in self.control_blocks:
            x = block(x)
        return self.head(self.control_norm(x.transpose(1, 2))).transpose(1, 2).float(), x

    def _free_controls(self, features, analysis):
        """The free branch's envelopes [B, 2 * free_bands, steps], in FP32,
        from the control features and ``analysis`` [B, 4 * free_bands, steps]."""
        y = self.free_project(features.transpose(1, 2)).transpose(1, 2) + self.free_input(analysis.to(features.dtype))
        for block in self.free_blocks:
            y = block(y)
        return self.free_head(self.free_norm(y.transpose(1, 2))).transpose(1, 2).float()

    def _to_steps(self, x, steps, mode="linear", hop=None):
        """Frame values [B, C, frames] at ``steps`` points ``hop`` apart (the
        control hop unless given), the first ``margin`` control hops before
        sample 0."""
        offset = self.hop_length // 2 + self.margin * self.control_hop
        return to_steps(x, steps, hop or self.control_hop, self.hop_length, offset, mode)

    def _noise_steps(self, x):
        """The head's [B, noise_ratio * C, steps], a value per noise step
        inside each control step -> [B, C, noise_ratio * steps]."""
        batch, _, steps = x.shape
        x = x.reshape(batch, self.noise_ratio, -1, steps).permute(0, 2, 3, 1)
        return x.reshape(batch, -1, steps * self.noise_ratio)

    def _pulse_kernels(self, log_gain, scatter, turn):
        """Band values [B, bands, L] -> the aligned and the scattered pulse's
        taps [B, L, 2, pulse_taps], from lag ``-pulse_pre`` on."""
        magnitude = torch.exp(torch.einsum("brl,rf->blf", log_gain, self.band_bins))
        phase = torch.einsum("brl,rf->blf", log_gain, self.phase_bins)
        # An angle, so the two shares' powers add up to one.
        angle = 0.5 * math.pi * torch.einsum("brl,rf->blf", torch.sigmoid(scatter), self.band_bins)
        turn = torch.einsum("brl,rf->blf", turn, self.band_bins)
        response = torch.stack((
            torch.polar(magnitude * torch.cos(angle), phase + turn),
            torch.polar(magnitude * torch.sin(angle), phase + turn),
        ), dim=2)
        response = torch.fft.irfft(response, n=self.pulse_fft)
        taps = torch.cat((response[..., -self.pulse_pre:], response[..., :self.pulse_taps - self.pulse_pre]), dim=-1)
        return taps * self.pulse_fade

    def _noise_kernels(self, log_gain, modulation):
        """Band gains [B, noise_bands, L] and the modulations' [B, 2 *
        noise_harmonics, noise_bands, L] -> zero-phase taps [B, L, 1 + 2 *
        noise_harmonics, noise_taps], centred."""
        level = torch.exp(torch.einsum("brl,rf->blf", log_gain, self.noise_bins)).unsqueeze(2)
        depth = torch.einsum("bnrl,rf->blnf", torch.tanh(modulation), self.noise_bins)
        response = torch.fft.irfft(torch.cat((level, level * depth), dim=2), n=self.noise_fft)
        half = self.noise_taps // 2
        taps = torch.cat((response[..., -half:], response[..., :self.noise_taps - half]), dim=-1)
        return taps * self.noise_fade

    def _filter(self, sources, kernels, pre, hop):
        """``sources`` [B, C, T] through ``kernels`` [B, L, C, taps], whose
        first tap is lag ``-pre``: block l is the ``OVERLAP * hop`` samples
        centred on sample ``(l - 1) * hop``, summed over C -> [B, L, window].
        On CUDA with Triton it runs as ``nagi_triton``'s block FIR, which
        computes the same."""
        batch, blocks, channels, taps = kernels.shape
        window = OVERLAP * hop
        reach = (OVERLAP // 2 + 1) * hop
        padded = F.pad(sources, (reach + taps - pre - 1, reach + pre))
        if nagi_triton is not None and padded.is_cuda:
            return nagi_triton.BlockFIR.apply(padded, kernels, hop, window)
        segments = padded.unfold(-1, window + taps - 1, hop).transpose(1, 2)
        groups = batch * blocks * channels
        # One group per block and source: each has its own kernel.
        out = F.conv1d(segments.reshape(1, groups, -1), kernels.flip(-1).reshape(groups, 1, taps), groups=groups)
        return out.view(batch, blocks, channels, window).sum(2)

    @staticmethod
    def _overlap_add(rendered, window, hop, samples):
        """Blocks [B, L, OVERLAP * hop] crossfaded under ``window`` -> [B, samples]."""
        rendered = (rendered * window).transpose(1, 2)
        length = (rendered.shape[-1] + OVERLAP - 1) * hop
        audio = F.fold(rendered, (1, length), (1, OVERLAP * hop), stride=(1, hop))
        start = (OVERLAP // 2 + 1) * hop
        return audio[:, 0, 0, start:start + samples]

    def _free(self, raw, log_bound, carriers, samples):
        """The head's [B, 2 * free_bands, steps] as envelopes of the bands'
        carriers, each under ``exp(log_bound)`` [B, free_bands, steps] -> [B, T]."""
        steps = raw.shape[-1]
        envelope = torch.tanh(raw).reshape(raw.shape[0], 2, self.free_bands, steps) * torch.exp(log_bound).unsqueeze(1)
        # Less its local mean: held steady it would draw a line at every band's centre,
        # the cheapest way to match a mel.
        flat = envelope.flatten(1, 2)
        mean = F.avg_pool1d(flat, self.free_detrend, 1, self.free_detrend // 2, count_include_pad=False)
        envelope = (flat - mean).reshape(envelope.shape)
        # The atoms' carriers start at each atom's centre: turned to the carrier that runs from sample 0.
        cos, sin = carriers
        coefficients = torch.cat((
            envelope[:, 0] * cos + envelope[:, 1] * sin,
            envelope[:, 1] * cos - envelope[:, 0] * sin,
        ), dim=1)
        audio = F.conv_transpose1d(coefficients, self.atoms, stride=self.control_hop)
        start = self.margin * self.control_hop + (self.atoms.shape[-1] - 1) // 2
        return audio[:, 0, start:start + samples]

    def forward(self, mel, f0, contrast=None, target=None, return_harmonic=False):
        """``target`` [B, 1, T], in training, is the audio the pulses' cycle
        is locked to: the output then lines up with it sample by sample, which
        a loss on the waveform needs. ``return_harmonic`` also returns the
        filtered pulses alone [B, 1, T]."""
        f0 = f0.float()
        samples = f0.shape[-1] * self.hop_length
        hop, margin = self.control_hop, self.margin
        steps = samples // hop + 2 * margin + 1
        voiced = f0 > 0
        cf0 = continuous_f0(f0)
        device = mel.device.type
        with torch.no_grad(), torch.autocast(device, enabled=False):
            pulses, noises = self._sources(cf0, samples, None if target is None else target.reshape(len(f0), -1))
        controls, features = self._controls(mel, cf0, voiced, contrast, steps)
        # The filters in FP32: their taps span more than BF16 resolves.
        with torch.autocast(device, enabled=False):
            level = mel.float() * self.mel_std + self.mel_mean
            frames = torch.cat((torch.einsum("bmt,mr->brt", level, self.mel_shortcut),
                                voiced.unsqueeze(1).float()), dim=1)
            shortcut = self._to_steps(frames, steps)
            tract, scatter, turn, noise, modulation = controls.split(self.splits, dim=1)
            log_pulse = (
                shortcut[:, :self.bands] + tract - UNVOICED_DROP * (1 - shortcut[:, -1:])
            ).clamp(*LOG_GAIN_RANGE)
            # The crossfade's blocks: one step past each end.
            blocks = slice(margin - 1, steps - margin + 1)
            pulse_kernels = self._pulse_kernels(log_pulse[..., blocks], scatter[..., blocks], turn[..., blocks])
            ratio, noise_hop = self.noise_ratio, self.noise_hop
            noise_level = self._to_steps(frames[:, self.bands:self.bands + self.noise_bands], steps * ratio,
                                         hop=noise_hop)
            log_noise = (noise_level + self._noise_steps(noise)).clamp(*LOG_GAIN_RANGE)
            modulation = self._noise_steps(modulation).reshape(
                mel.shape[0], 2 * self.noise_harmonics, self.noise_bands, steps * ratio
            )
            noise_blocks = slice(margin * ratio - 1, margin * ratio + samples // noise_hop + 2)
            noise_kernels = self._noise_kernels(log_noise[..., noise_blocks], modulation[..., noise_blocks])
            harmonic = self._overlap_add(
                self._filter(pulses, pulse_kernels, self.pulse_pre, hop), self.window, hop, samples
            )
            audio = harmonic + self._overlap_add(
                self._filter(noises, noise_kernels, self.noise_taps // 2, noise_hop),
                self.noise_window, noise_hop, samples,
            )
        if self.free_head is not None:
            with torch.no_grad(), torch.autocast(device, enabled=False):
                log_bound = shortcut[:, self.bands + self.noise_bands:-1] + self.free_log_bound
                carriers = self._carriers(steps)
                # Not learned through: the filtered sources answer to their own losses.
                analysis = torch.cat((
                    self._analysis(noises[:, 0]) / self.atom_norm, self._heard(audio, log_bound, carriers),
                ), dim=1)
            free = self._free_controls(features, analysis)
            with torch.autocast(device, enabled=False):
                extra = self._free(free, log_bound, carriers, samples)
                if self.free_keep is not None:
                    extra = extra * self.free_keep.to(extra.dtype)
                audio = audio + extra
        if return_harmonic:
            return audio.unsqueeze(1), harmonic.unsqueeze(1)
        return audio.unsqueeze(1)
