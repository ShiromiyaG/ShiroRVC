"""Overtrain detection: the held-out set, its metrics, and the monitor.

Training loss cannot detect overtraining by construction -- it keeps falling
while generalisation rots -- so everything that answers "is this run still
improving?" lives here rather than in the training loop: carving a split the
model never sees, materialising it into fixed-length excerpts, scoring those
down the *inference* path, and deciding from that curve which weights to keep
and when the run has stopped improving.

Nothing in here reads the run spec or the training globals; the loop passes in
what it has (config, device, AMP settings) and gets numbers back.
"""

import copy
import math

from collections import deque
from random import Random

import torch
import torch.nn.functional as F

from torch.amp import autocast

from rvc.lib.terminal import info, warning

class HoldoutSet:
    """Fixed-length excerpts, kept in RAM and batched on demand.

    Cropped to equal length rather than padded: padding silence into the mel
    would put a batch-dependent constant into the metric. Held as excerpts
    rather than pre-formed batches so :meth:`shrink` can re-batch them.
    """

    def __init__(self, items, batch_size, label="holdout"):
        self.items = list(items)
        self.batch_size = max(1, int(batch_size))
        self.label = label
        # Ground truth does not depend on the weights, so its mel is computed
        # once for the life of the run rather than once per evaluation.  Keyed
        # by length as well as by batch, so a decoder whose output length moves
        # cannot silently be compared against the wrong excerpt.
        self._target_mels = {}

    def __len__(self):
        return len(self.items)

    @property
    def frames(self):
        return self.items[0][0].shape[0] if self.items else 0

    def seconds(self, config):
        return self.frames * config.data.hop_length / config.data.sample_rate

    def batches(self):
        for start in range(0, len(self.items), self.batch_size):
            chunk = self.items[start : start + self.batch_size]
            yield start // self.batch_size, self._collate(chunk)

    @staticmethod
    def _collate(chunk):
        phone = torch.stack([item[0] for item in chunk])
        pitch = torch.stack([item[1] for item in chunk])
        pitchf = torch.stack([item[2] for item in chunk])
        spectrogram = torch.stack([item[3] for item in chunk])
        wave = torch.stack([item[4] for item in chunk])
        sid = torch.cat([item[5] for item in chunk])
        # Every excerpt is the same length by construction, which is the whole
        # point: no padding, so no per-item lengths to carry.
        frames = torch.full((len(chunk),), phone.shape[1], dtype=torch.long)
        samples = torch.full((len(chunk),), wave.shape[-1], dtype=torch.long)
        return (
            phone,
            frames,
            pitch,
            pitchf,
            spectrogram,
            frames,
            wave,
            samples,
            sid,
        )

    def target_mel(self, index, length, factory):
        cached = self._target_mels.get(index)
        if cached is None or cached[0] != length:
            cached = (length, factory())
            self._target_mels[index] = cached
        return cached[1]

    def shrink(self):
        """Halve the batch.  False once there is nothing left to halve."""
        if self.batch_size <= 1:
            return False
        self.batch_size = max(1, self.batch_size // 2)
        self._target_mels.clear()
        return True


def uniform_excerpts(
    dataset,
    indices,
    crop_frames,
    config,
    batch_size,
    label="holdout",
    fixed=False,
    limit=None,
):
    """Load rows and crop them all to one shared length.

    ``crop_frames`` is a ceiling rather than a demand: unless ``fixed``, the
    crop lands at the lower quartile of what the rows actually carry, so three
    excerpts in four survive it.  Taking the ceiling literally would leave the
    set to whichever recording happened to be longest.

    ``fixed`` is for the training probe, which has to be cropped exactly like
    the holdout or the two numbers are not comparable.
    """
    hop = config.data.hop_length
    rows = []
    for index in indices:
        spectrogram, wave, phone, pitch, pitchf, sid = dataset[index]
        frames = min(spectrogram.shape[-1], phone.shape[0], wave.shape[-1] // hop)
        if frames > 0:
            rows.append((frames, spectrogram, wave, phone, pitch, pitchf, sid))
    if not rows:
        return None

    if fixed:
        crop = max(1, int(crop_frames))
    else:
        ordered = sorted(row[0] for row in rows)
        crop = max(1, min(int(crop_frames), ordered[len(ordered) // 4]))

    items = []
    for frames, spectrogram, wave, phone, pitch, pitchf, sid in rows:
        if frames < crop:
            continue
        # Cloned: the crops are views onto whole files, and keeping views would
        # keep every one of those files resident for the life of the run.
        items.append(
            (
                phone[:crop, :].clone(),
                pitch[:crop].clone(),
                pitchf[:crop].clone(),
                spectrogram[:, :crop].clone(),
                wave[:, : crop * hop].clone(),
                sid.clone(),
            )
        )
        if limit is not None and len(items) >= int(limit):
            break
    if not items:
        return None
    return HoldoutSet(items, batch_size, label=label)


def carve_holdout(config, train_dataset, rank, enabled=True):
    """Take a held-out split out of ``train_dataset`` before anything sees it.

    Returns the held-out dataset (a shallow copy carrying only the held-out
    rows) or ``None``, and shrinks ``train_dataset`` in place to what is left.
    """
    if not enabled:
        return None

    # Script-relative, like every other ``data_utils`` import in the trainer:
    # that module itself imports ``mel_processing``/``utils`` the same way, so
    # it only resolves with ``rvc/train`` on the path -- which is how
    # ``train.py`` is launched.
    from data_utils import holdout_split_indices

    # The evaluation cost is seconds of audio to synthesise, so the budget
    # is set in seconds.  ``lengths`` is in frames (see ``_filter``).
    max_seconds = float(getattr(config.train, "holdout_max_seconds", 120.0))
    train_indices, holdout_indices = holdout_split_indices(
        train_dataset.audiopaths_and_text,
        fraction=float(getattr(config.train, "holdout_fraction", 0.02)),
        minimum=int(getattr(config.train, "holdout_min_slices", 16)),
        maximum=int(getattr(config.train, "holdout_max_slices", 96)),
        seed=int(getattr(config.train, "seed", 1234)),
        lengths=train_dataset.lengths,
        max_frames=(
            max_seconds * config.data.sample_rate / config.data.hop_length
            if max_seconds > 0
            else None
        ),
    )
    if not holdout_indices:
        if rank == 0:
            warning(
                "Dataset too small to hold one out; overtrain detection is off.",
                tag="[HOLDOUT]",
            )
        return None

    rows = train_dataset.audiopaths_and_text
    lengths = train_dataset.lengths
    holdout_dataset = copy.copy(train_dataset)
    holdout_dataset.audiopaths_and_text = [rows[i] for i in holdout_indices]
    holdout_dataset.lengths = [lengths[i] for i in holdout_indices]
    train_dataset.audiopaths_and_text = [rows[i] for i in train_indices]
    train_dataset.lengths = [lengths[i] for i in train_indices]
    if rank == 0:
        sources = len({r[0].rsplit("_", 1)[0] for r in holdout_dataset.audiopaths_and_text})
        info(
            f"{len(holdout_indices)} slices from {sources} source "
            f"recordings held out of {len(rows)} - never trained on.",
            tag="[HOLDOUT]",
        )
    return holdout_dataset


def materialise_holdout(config, train_dataset, holdout_dataset):
    """Load the held-out rows (and the training probe) into RAM, once.

    Materialised once and kept: the point of a holdout is that every
    evaluation scores the exact same audio, and re-reading it through a loader
    each time would also cost more than the forward pass.  Returns
    ``(holdout_set, probe_set)``, either of which may be ``None``.
    """
    crop_frames = max(
        1,
        int(
            float(getattr(config.train, "holdout_crop_seconds", 3.0))
            * config.data.sample_rate
            / config.data.hop_length
        ),
    )
    eval_batch = int(getattr(config.train, "holdout_batch_size", 4))
    holdout_set = uniform_excerpts(
        holdout_dataset,
        range(len(holdout_dataset.audiopaths_and_text)),
        crop_frames,
        config,
        eval_batch,
        label="holdout",
    )
    if holdout_set is None:
        warning(
            "Held-out rows were all too short to score; detection is off.",
            tag="[HOLDOUT]",
        )
        return None, None

    info(
        f"Scoring {len(holdout_set)} excerpts of "
        f"{holdout_set.seconds(config):.1f}s each, batched "
        f"{holdout_set.batch_size} at a time.",
        tag="[HOLDOUT]",
    )

    # The same measurement on slices the model *has* been trained on.  The
    # held-out curve alone mixes two movements -- the model is still
    # learning, and it is starting to memorise -- and only their difference
    # is overtraining.  Subtracting the probe removes the shared trend, so
    # the turn shows up earlier and more cleanly than in the absolute
    # number.  Same crop and same count as the holdout, or the two are not
    # comparable; sampled deterministically, so a resume scores the same
    # slices rather than leaking a fresh draw into the comparison.
    probe_set = None
    if bool(getattr(config.train, "holdout_train_probe", True)):
        pool = list(range(len(train_dataset.audiopaths_and_text)))
        sampled = Random(int(getattr(config.train, "seed", 1234))).sample(
            pool, min(len(pool), 2 * len(holdout_set))
        )
        probe_set = uniform_excerpts(
            train_dataset,
            sampled,
            holdout_set.frames,
            config,
            eval_batch,
            label="train probe",
            fixed=True,
            limit=len(holdout_set),
        )
    return holdout_set, probe_set


#: Bands the held-out spectral deficit is reported over, in Hz.  Split where a
#: vocoder's error changes character rather than evenly: below 4 kHz is where
#: the mel scale already spends most of its bins, and the two top bands are the
#: ones a mel L1 charges 4-6x less for than the same defect at 1-3 kHz.  A band
#: starting at or above Nyquist is dropped and the last one is clipped to it, so
#: the same list serves every shipped sample rate.
HOLDOUT_DEFICIT_BANDS = (
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 8000),
    (8000, 10000),
    (10000, 13000),
    (13000, 16000),
)


def deficit_band_edges(sample_rate):
    """``(low, high, label)`` per band that fits under this rate's Nyquist."""

    nyquist = float(sample_rate) / 2.0
    edges = []
    for low, high in HOLDOUT_DEFICIT_BANDS:
        if low >= nyquist:
            break
        edges.append((float(low), min(float(high), nyquist), f"{low // 1000}k"))
    return edges


def band_deficit_db(generated, target, sample_rate):
    """How much energy the generator is missing per band, in dB, per band label.

    Negative is the generator below the reference, which is the direction a
    vocoder fails in.  ``mel_l1`` cannot answer this: it is one number over a
    warped axis, so a 14 dB hole above 10 kHz and a 2 dB error at 1 kHz reach
    it as comparable contributions -- which is the whole reason the band
    weighting exists.  Averaged in dB per item rather than over pooled energy,
    or one loud excerpt would decide the figure for the set.
    """

    generated = generated.float().flatten(0, -2) if generated.ndim > 2 else generated.float()
    target = target.float().flatten(0, -2) if target.ndim > 2 else target.float()
    spectrum_g = torch.fft.rfft(generated, dim=-1).abs().pow(2)
    spectrum_t = torch.fft.rfft(target, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(
        generated.shape[-1], 1.0 / float(sample_rate), device=generated.device
    )

    deficits = {}
    for low, high, label in deficit_band_edges(sample_rate):
        band = (freqs >= low) & (freqs < high)
        if not bool(band.any()):
            continue
        # The floor is what keeps a silent excerpt from reporting -inf and
        # taking the whole average with it.
        energy_g = spectrum_g[..., band].sum(-1).clamp_min(1e-12)
        energy_t = spectrum_t[..., band].sum(-1).clamp_min(1e-12)
        deficits[label] = float(
            (10.0 * torch.log10(energy_g / energy_t)).mean()
        )
    return deficits


def holdout_metrics(
    net_g,
    excerpts,
    config,
    device,
    want_latent=True,
    noise_scale=0.0,
    use_amp=False,
    amp_dtype=None,
):
    """Score held-out audio down the *inference* path, in a single pass.

    ``mel_l1`` scores the prior path, not the training forward: the training
    path samples a posterior that has seen the target spectrogram, so a
    memorising model would keep scoring well there after it stops
    generalising.  Prior, posterior and their gap come from one pass over one
    set of weights, so the numbers stay comparable.

    ``noise_scale=0`` decodes the prior mean instead of ``infer``'s default
    0.66666 draw, making the metric a pure function of the weights.  Plain L1
    on the log-mel: the adversarial/feature-matching/KL terms are scored
    against a discriminator and schedule that keep moving independently of
    the generator.
    """
    # Imported here rather than at module scope: ``utils`` pulls matplotlib and
    # librosa in, and the excerpt/monitor half of this module is used without
    # either.
    from rvc.train.utils import wave_to_mel

    model = net_g.module if hasattr(net_g, "module") else net_g
    # ``flow`` is what turns a prior draw into something the decoder can use;
    # without it there is no inference path to rebuild by hand and ``infer`` is
    # the only way in.  ``enc_q`` is dropped for export, which is the one state
    # in which the posterior half cannot be measured at all.
    manual_prior = getattr(model, "flow", None) is not None
    want_latent = (
        want_latent and manual_prior and getattr(model, "enc_q", None) is not None
    )
    was_training = model.training
    model.eval()
    totals = {"mel_l1": 0.0}
    for _, _, label in deficit_band_edges(config.data.sample_rate):
        totals[f"band_deficit_{label}"] = 0.0
    if want_latent:
        totals["latent_gap"] = 0.0
        totals["latent_posterior"] = 0.0
    count = 0

    # The metric has to be a pure function of the weights, and the prior draw
    # is noise: at ``noise_scale`` 0 there is none, and above it the seed is
    # pinned so every evaluation sees the same draw.  Restoring the state
    # afterwards keeps a variable number of draws from shifting the training
    # stream underneath the run.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(0x5EED)
    if cuda_rng_state is not None:
        torch.cuda.manual_seed_all(0x5EED)

    try:
        # The same autocast the training step runs under, so the metric
        # measures the model as it is actually being trained, and
        # ``inference_mode`` rather than ``no_grad`` because nothing here is
        # ever differentiated.
        with torch.inference_mode(), autocast(
            device_type="cuda", enabled=use_amp, dtype=amp_dtype
        ):
            for index, batch in excerpts.batches():
                # Not named ``spec``: ``test_run_spec`` audits every
                # ``spec.<field>`` in this file against the run spec, and a
                # local of that name collides with the check.
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    holdout_spec,
                    holdout_spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = batch
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
                pitch = pitch.to(device, non_blocking=True)
                pitchf = pitchf.to(device, non_blocking=True)
                sid = sid.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)

                g = model.emb_g(sid).unsqueeze(-1)
                if manual_prior:
                    m_p, logs_p, x_mask = model.enc_p(
                        phone=phone, pitch=pitch, lengths=phone_lengths
                    )
                    z_p = m_p
                    if noise_scale:
                        z_p = (
                            m_p
                            + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale
                        )
                    z_prior = model.flow(z_p * x_mask, x_mask, g=g, reverse=True)
                    z_prior = z_prior * x_mask
                    frames = min(z_prior.shape[-1], pitchf.shape[-1])
                    prior_wave = model.dec(
                        z_prior[..., :frames], pitchf[..., :frames], g=g
                    )
                else:
                    prior_wave, *_ = model.infer(
                        phone, phone_lengths, pitch, pitchf, sid, 0
                    )
                    frames = pitchf.shape[-1]

                posterior_wave = None
                if want_latent:
                    holdout_spec = holdout_spec.to(device, non_blocking=True)
                    holdout_spec_lengths = holdout_spec_lengths.to(
                        device, non_blocking=True
                    )
                    z_q, _, _, spec_mask = model.enc_q(
                        holdout_spec, holdout_spec_lengths, g=g
                    )
                    posterior_wave = model.dec(
                        (z_q * spec_mask)[..., :frames], pitchf[..., :frames], g=g
                    )

                # The decoder rebuilds the waveform from frame-rate features,
                # so its length lands within a hop of the excerpt rather than
                # on it.
                length = min(prior_wave.shape[-1], wave.shape[-1])
                if posterior_wave is not None:
                    length = min(length, posterior_wave.shape[-1])
                if length <= config.data.filter_length:
                    continue
                target_mel = excerpts.target_mel(
                    index,
                    length,
                    lambda: wave_to_mel(config, wave[..., :length], num_mels=None),
                )
                prior_mel = wave_to_mel(config, prior_wave[..., :length], num_mels=None)
                totals["mel_l1"] += float(F.l1_loss(prior_mel, target_mel))
                for label, deficit in band_deficit_db(
                    prior_wave[..., :length],
                    wave[..., :length],
                    config.data.sample_rate,
                ).items():
                    totals[f"band_deficit_{label}"] += deficit
                if posterior_wave is not None:
                    posterior_mel = wave_to_mel(
                        config, posterior_wave[..., :length], num_mels=None
                    )
                    totals["latent_gap"] += float(F.l1_loss(prior_mel, posterior_mel))
                    totals["latent_posterior"] += float(
                        F.l1_loss(posterior_mel, target_mel)
                    )
                count += 1
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            model.train()

    if not count:
        return {"mel_l1": float("nan")}
    return {key: value / count for key, value in totals.items()}


def holdout_metrics_resilient(net_g, excerpts, config, device, **kwargs):
    """:func:`holdout_metrics`, but an oversized batch is halved, not fatal.

    The evaluation decodes seconds of audio per item where a training step
    decodes a fraction of one, so its batch can be the largest allocation in
    the run despite storing no gradients -- and it would be a poor trade to
    lose a training run to a diagnostic.
    """
    while True:
        try:
            return holdout_metrics(net_g, excerpts, config, device, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not excerpts.shrink():
                raise
            warning(
                f"{excerpts.label} evaluation ran out of memory; retrying at "
                f"batch {excerpts.batch_size}.",
                tag="[HOLDOUT]",
            )


def cpu_state_dict(source):
    """Detached CPU copy of a model's or a ``WeightEMA``'s weights."""
    if hasattr(source, "cpu_state_dict"):
        return source.cpu_state_dict()
    module = source.module if hasattr(source, "module") else source
    return {
        key: tensor.detach().to("cpu", copy=True)
        for key, tensor in module.state_dict().items()
    }


class OvertrainMonitor:
    """Find the last point where held-out quality was still improving.

    Overtraining isn't visible in the training loss, so this scores audio the
    model has never trained on and watches for the minimum. Two things are
    judged against a *noise band* rather than a fixed threshold (which means
    something different in every run): the score is median-filtered over
    ``smoothing`` evaluations before anything is decided on it, and
    ``min_delta`` is a floor under a band computed as the residual spread of
    recent scores about a trend line, so a still-improving run gets a narrow
    band and a flat noisy one gets a wide one.

    Weights to keep: lowest smoothed score; ties inside the band favour the
    earlier step. Has the run stopped improving: the patience counter ignores
    anything inside the band, so noise can't reset it forever.
    """

    def __init__(self, patience=8, min_delta=0.001, smoothing=3, noise_window=6):
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.smoothing = max(1, int(smoothing))
        self.noise_window = max(4, int(noise_window))
        #: Best *smoothed* score, and the step whose weights produced it.
        self.best = float("inf")
        self.best_step: int | None = None
        self.state_dict: dict | None = None
        #: Best single score ever seen.  Not a selection criterion; it is what
        #: decides whether an evaluation is close enough to the running best to
        #: be worth cloning weights for.
        self.best_raw = float("inf")
        # The last score good enough to count as progress.  Separate from
        # ``best`` because it moves on a coarser ratchet.
        self.patience_reference = float("inf")
        self.since_progress = 0
        self.history: list[tuple[int, float]] = []
        self.smoothed = float("nan")
        self.sigma = 0.0
        # Once the run has been called, every further evaluation is paying for
        # a number nothing acts on; see :meth:`backoff`.
        self.interval_scale = 1
        self._window = deque(maxlen=self.smoothing)

    def _band(self, reference):
        """How large a change has to be before it is evidence rather than noise."""
        if not math.isfinite(reference):
            return 0.0
        return max(self.sigma, self.min_delta * abs(reference))

    def _noise_sigma(self) -> float:
        """Residual spread of recent scores about a trend line (not a mean: a
        genuinely improving run has a large spread about its mean but a small
        one about its slope). 0 until there are enough points to fit.
        """
        recent = [value for _, value in self.history[-self.noise_window :]]
        count = len(recent)
        if count < 4:
            return 0.0
        mean_x = (count - 1) / 2.0
        mean_y = sum(recent) / count
        variance_x = sum((index - mean_x) ** 2 for index in range(count))
        covariance = sum(
            (index - mean_x) * (value - mean_y) for index, value in enumerate(recent)
        )
        slope = covariance / variance_x if variance_x else 0.0
        intercept = mean_y - slope * mean_x
        residuals = [
            value - (slope * index + intercept) for index, value in enumerate(recent)
        ]
        return math.sqrt(sum(r * r for r in residuals) / max(1, count - 2))

    def update(self, source, value: float, step: int) -> bool:
        """``source``: a model or a ``WeightEMA``. Returns whether this moved
        the best. With a centred filter, the weights kept are the middle of
        the window, not the newest evaluation -- otherwise a snapshot of the
        live model would be credited with a score it never earned.
        """
        if not math.isfinite(value):
            return False
        self.history.append((int(step), float(value)))
        self.sigma = self._noise_sigma()

        # Only clone weights for evaluations close enough to the running best
        # to still win one; a window entry without weights can never be
        # selected.
        competitive = (
            not math.isfinite(self.best_raw)
            or value <= self.best_raw + 2.0 * self._band(self.best_raw)
        )
        self._window.append(
            (int(step), float(value), cpu_state_dict(source) if competitive else None)
        )
        self.best_raw = min(self.best_raw, float(value))

        if len(self._window) < self.smoothing:
            # Until the window fills, behave unsmoothed rather than report a
            # half-formed median.
            self.smoothed = float(value)
            center_step, _center_value, center_state = self._window[-1]
        else:
            scores = sorted(entry[1] for entry in self._window)
            self.smoothed = scores[len(scores) // 2]
            center_step, _center_value, center_state = self._window[
                len(self._window) // 2
            ]

        improved = False
        if center_state is not None and self.smoothed < self.best - self._band(
            self.best
        ):
            improved = True
            self.best = float(self.smoothed)
            self.best_step = int(center_step)
            self.state_dict = center_state

        if self.smoothed < self.patience_reference - self._band(
            self.patience_reference
        ):
            self.patience_reference = float(self.smoothed)
            self.since_progress = 0
        else:
            self.since_progress += 1
        return improved

    def backoff(self, factor: int = 4) -> int:
        """Evaluate less often once overtraining has already been flagged
        (with ``stop_on_overtrain`` off, the run keeps going and a run can
        still come back, so evaluation continues, just less often).
        """
        self.interval_scale = max(1, int(self.interval_scale * max(1, int(factor))))
        return self.interval_scale

    @property
    def overtrained(self) -> bool:
        return self.state_dict is not None and self.since_progress >= self.patience


def deliverable_weights(overtrain_monitor, ema, model_g, use_holdout=True):
    """The weights this run would hand you if it stopped right now: holdout
    best, then EMA, then live weights, in order of how much each source
    knows. Returns ``(state_dict, label, source_step)``.

    ``use_holdout=False`` skips the holdout snapshot and returns the current
    weights (EMA or live). The periodic exports use that, so they keep
    tracking the run; the holdout best goes only into the separate
    ``_pre-overtrain`` export.

    ``source_step`` is the step the weights are actually *from*, which is not
    the step the export is named after. The holdout branch only replaces its
    snapshot when the metric improves, so once it plateaus every later export
    returns the same tensors while the filename and the ``epoch``/``step``
    fields keep counting up -- six consecutive exports came back bit-identical
    that way. ``None`` means "as of now"; the caller writes it into the file so
    the staleness is readable after the run's console has scrolled away.
    """
    if (
        use_holdout
        and overtrain_monitor is not None
        and overtrain_monitor.state_dict is not None
    ):
        return (
            overtrain_monitor.state_dict,
            f"holdout best @ {overtrain_monitor.best_step}",
            overtrain_monitor.best_step,
        )
    if ema is not None:
        # The shadow is a running average ending at the current step, so it is
        # current even though it is not any single step's weights.
        return ema.cpu_state_dict(), f"EMA ({ema.updates} updates)", None
    # A copy, not ``model_g.state_dict()`` itself: that hands back live
    # parameter tensors, so anything mutating the weights before the export is
    # written -- a schedule-free optimizer returning to its training iterate,
    # for one -- would change what has already been chosen.  The other two
    # branches already return CPU copies.
    return cpu_state_dict(model_g), "live weights", None
