"""Validation previews: the mel comparison figure and the audio, written to
TensorBoard and as loose files under the run's log directory."""

import os
from contextlib import contextmanager

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.tensorboard import SummaryWriter

from rvc.lib.algorithm.commons import upsample_content
from rvc.lib.terminal import info, warning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from rvc.train.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_AUDIO_TAG,
    TENSORBOARD_VALIDATION_AXIS_X,
    TENSORBOARD_VALIDATION_AXIS_Y,
    TENSORBOARD_VALIDATION_DB_LABEL,
    TENSORBOARD_VALIDATION_DIFFERENCE_LABEL,
    TENSORBOARD_VALIDATION_FOOTER,
    TENSORBOARD_VALIDATION_MEL_TAG,
    TENSORBOARD_VALIDATION_MEL_TITLES,
    TENSORBOARD_VALIDATION_PANEL_TITLE,
    TENSORBOARD_VALIDATION_PREVIEW_DIR,
    TENSORBOARD_VALIDATION_PROFILE_TITLE,
    TENSORBOARD_VALIDATION_SOURCE_TAG,
)

MATPLOTLIB_FLAG = False
#: Defaults for the validation preview figure.  Overridable per model through
#: ``validation_preview_dpi`` / ``_width`` / ``_height`` in the config JSON.
#:
#: The two knobs are not interchangeable.  ``figsize`` is in inches and dpi
#: converts to pixels, so raising **dpi** scales the whole figure -- panels and
#: text together -- and is what "make it sharper" means.  Raising **figsize**
#: at a fixed dpi gives the panels more room while the text stays the same
#: physical size, so labels shrink relative to the plot.  67 x 24in lands on
#: 1608px wide, which is the historical output.
VALIDATION_PREVIEW_DPI = 67
VALIDATION_PREVIEW_FIGSIZE = (24.0, 5.8)

#: Candidate ticks for the frequency axis, in Hz.  A mel axis is close to
#: logarithmic, so the evenly spaced ticks a linear axis wants would crowd the
#: bottom octaves into a few pixels and label almost nothing where the voice
#: actually lives.  Whatever falls outside the filterbank's range is dropped.
VALIDATION_PREVIEW_FREQUENCY_TICKS = (
    100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 20000.0,
)
#: A tick closer than this share of the axis to the one under it is dropped.
VALIDATION_PREVIEW_TICK_SPACING = 0.05

#: The mels are the natural log of a magnitude, as both trainers make them.
NATURAL_LOG_TO_DB = 20.0 / np.log(10.0)
#: Shown below the original's peak; fixed, so previews compare across steps.
VALIDATION_PREVIEW_RANGE_DB = 80.0
VALIDATION_PREVIEW_DIFFERENCE_DB = 10.0
#: Where both mels are this far under the peak, the difference is inaudible
#: and only noise; it is greyed out.
VALIDATION_PREVIEW_QUIET_DB = 60.0
VALIDATION_PREVIEW_BACKGROUND = "#10161f"
VALIDATION_PREVIEW_QUIET_COLOR = "#2a313c"

#: The second row, from the audio: a linear axis to Nyquist, where the mel's
#: top bins (~700 Hz wide at 44.1 kHz) merge the harmonics. Its height in
#: inches is added to the figure's.
VALIDATION_PREVIEW_STFT_FFT = 2048
VALIDATION_PREVIEW_STFT_HEIGHT = 12.0
VALIDATION_PREVIEW_STFT_TICK_HZ = 2000.0
#: Its difference compares power pooled over this much (a bin is ~20 Hz), so a
#: harmonic a few Hz off reads as the same level, not as a red and a blue line.
VALIDATION_PREVIEW_STFT_POOL_HZ = 200.0
#: Figure margins and the gap between the rows, in inches.
VALIDATION_PREVIEW_MARGINS = (0.8, 1.0, 1.0)


#: Previews kept for TensorBoard, which by default shows 10 images and 10
#: clips per tag; anything older only made the event file bigger.
TENSORBOARD_MEDIA_KEEP = 10


class MediaLog:
    """TensorBoard run for the validation media: one event file per preview,
    pruned to the newest ``keep``.

    Event files are append-only, so previews written beside the scalars kept
    every one of them (~1 MB each) while TensorBoard displays only 10.
    """

    def __init__(self, log_dir, resume_step=0, keep=TENSORBOARD_MEDIA_KEEP):
        self.log_dir = log_dir
        self.keep = keep
        os.makedirs(log_dir, exist_ok=True)
        # Written by a previous run past the checkpoint this one resumes from.
        for path, step in self._files():
            if step > resume_step:
                os.remove(path)

    def _files(self):
        """``(path, step)`` of every preview file, oldest step first."""
        found = []
        for name in os.listdir(self.log_dir):
            stem, _, step = name.rpartition(".step")
            if "tfevents" in stem and step.isdigit():
                found.append((os.path.join(self.log_dir, name), int(step)))
        return sorted(found, key=lambda item: item[1])

    @contextmanager
    def writer(self, global_step):
        """A writer for one preview; the file is closed and pruned on exit."""
        writer = SummaryWriter(
            self.log_dir, filename_suffix=f".step{int(global_step)}"
        )
        try:
            yield writer
        finally:
            writer.close()
            for path, _ in self._files()[: -self.keep]:
                os.remove(path)


def limit_audio_peak(audio, max_peak=0.98):
    """Return a finite mono preview whose peak stays inside TensorBoard's range."""
    if torch.is_tensor(audio):
        preview = audio.detach().float().cpu()
        preview = torch.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        preview = preview.reshape(-1)
        if preview.numel() == 0:
            return preview
        peak = preview.abs().amax()
        if torch.isfinite(peak) and peak.item() > max_peak:
            preview = preview * (max_peak / peak)
        return preview.clamp(-max_peak, max_peak)

    preview = np.asarray(audio, dtype=np.float32)
    preview = np.nan_to_num(
        preview,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=True,
    ).reshape(-1)
    if preview.size == 0:
        return preview
    peak = float(np.max(np.abs(preview)))
    if np.isfinite(peak) and peak > max_peak:
        preview *= max_peak / peak
    return np.clip(preview, -max_peak, max_peak).astype(np.float32, copy=False)


def _mel_to_numpy(mel):
    if torch.is_tensor(mel):
        mel = mel.detach().float().cpu().numpy()
    mel = np.nan_to_num(np.asarray(mel, dtype=np.float32), copy=True)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError("Validation mel spectrograms must be two-dimensional.")
    return mel


def _mel_axis_positions(frequencies, mel_bins, mel_fmin, mel_fmax):
    """Where ``frequencies`` land on a mel filterbank's bin-index axis.

    The filterbank's ``mel_bins + 2`` band edges are evenly spaced *in mel*, so
    a frequency's position in bin-index space is linear in ``hz_to_mel``.  Bin 0
    is centred on edge 1 and the last bin on edge ``mel_bins``, which is where
    the ``mel_bins + 1`` scaling and the -1 offset come from.

    ``htk=False`` is not a preference: it is librosa's default, and so it is
    what ``librosa.filters.mel`` used to build the basis in ``mel_processing``.
    Reading the axis off a different mel scale than the data was binned with
    would swap one wrong answer for another.
    """
    low = librosa.hz_to_mel(float(mel_fmin), htk=False)
    high = librosa.hz_to_mel(float(mel_fmax), htk=False)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("The mel axis needs a positive frequency range.")
    mels = librosa.hz_to_mel(np.asarray(frequencies, dtype=float), htk=False)
    return (mels - low) / (high - low) * (mel_bins + 1) - 1.0


def _format_frequency(frequency):
    return f"{frequency:g}" if frequency < 1000.0 else f"{frequency / 1000:g}k"


def _wave_to_numpy(wave):
    if torch.is_tensor(wave):
        wave = wave.detach().float().cpu().numpy()
    return np.nan_to_num(np.asarray(wave, dtype=np.float32), copy=True).reshape(-1)


def _stft_db(wave, n_fft, hop_length):
    """[n_fft // 2 + 1, frames] magnitude in dB of a full-scale sine."""
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(
        torch.from_numpy(wave), n_fft, hop_length, window=window, center=True, return_complex=True
    ).abs()
    return (20.0 * torch.log10(torch.clamp(spectrum / (window.sum() / 2), min=1e-5))).numpy()


def _band_profile(difference):
    """The masked ``difference``'s mean over time per bin; NaN where a bin is
    never audible."""
    sums = difference.filled(0.0).sum(axis=1)
    counts = (~np.ma.getmaskarray(difference)).sum(axis=1).astype(float)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)


def _pool_db(values, width):
    """dB images [bins, frames] as their power averaged over ``width``
    neighbouring bins."""
    if width <= 1:
        return values
    return 10.0 * np.log10(uniform_filter1d(np.power(10.0, values / 10.0), int(width), axis=0, mode="nearest"))


def _mel_ticks(mel_bins, mel_fmin, mel_fmax):
    """Tick positions and labels on a mel bin axis, too-close ones dropped."""
    positions = _mel_axis_positions(VALIDATION_PREVIEW_FREQUENCY_TICKS, mel_bins, mel_fmin, mel_fmax)
    ticks, labels = [], []
    for frequency, position in zip(VALIDATION_PREVIEW_FREQUENCY_TICKS, positions):
        if not -0.5 <= position <= mel_bins - 0.5:
            continue
        if ticks and position - ticks[-1] < VALIDATION_PREVIEW_TICK_SPACING * mel_bins:
            continue
        ticks.append(float(position))
        labels.append(_format_frequency(frequency))
    return ticks, labels


def plot_validation_preview_to_figure(
    predicted_mel,
    target_mel,
    epoch,
    global_step,
    sample_index=0,
    sample_rate=22050,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
    predicted_wave=None,
    target_wave=None,
):
    """Create a dark TensorBoard-style validation report: generated, original
    and their difference with its mean per frequency, for the mel and, with
    both waves, for their STFT on a linear axis to Nyquist.

    ``hop_length`` sets the time axis and must match the config the mels were
    produced with -- the default is a fallback, not a good guess.  The same goes
    for ``mel_fmin`` / ``mel_fmax``, which set the frequency axis; ``mel_fmax``
    of ``None`` means Nyquist, matching the mel extraction. ``figsize`` sizes
    the mel row; the STFT row adds ``VALIDATION_PREVIEW_STFT_HEIGHT`` inches.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    target_mel = _mel_to_numpy(target_mel)
    predicted_mel = _mel_to_numpy(predicted_mel)
    mel_bins = min(target_mel.shape[0], predicted_mel.shape[0])
    frames = min(target_mel.shape[1], predicted_mel.shape[1])
    if mel_bins <= 0 or frames <= 0:
        raise ValueError("Validation mel spectrograms must not be empty.")

    # The image stays in bin space, mel bins not being evenly spaced in Hz;
    # the ticks move to where their frequencies fall instead.
    top_frequency = float(sample_rate) / 2.0 if mel_fmax is None else float(mel_fmax)
    duration = frames * float(hop_length) / float(sample_rate)
    rows = [(
        "mel",
        predicted_mel[:mel_bins, :frames] * NATURAL_LOG_TO_DB,
        target_mel[:mel_bins, :frames] * NATURAL_LOG_TO_DB,
        np.arange(mel_bins, dtype=float),
        *_mel_ticks(mel_bins, mel_fmin, top_frequency),
        1,
    )]
    if predicted_wave is not None and target_wave is not None:
        predicted_wave, target_wave = _wave_to_numpy(predicted_wave), _wave_to_numpy(target_wave)
        length = min(predicted_wave.size, target_wave.size)
        if length > VALIDATION_PREVIEW_STFT_FFT:
            n_fft = VALIDATION_PREVIEW_STFT_FFT
            step = float(sample_rate) / n_fft
            ticks = np.arange(0.0, float(sample_rate) / 2.0 + 1.0, VALIDATION_PREVIEW_STFT_TICK_HZ)
            rows.append((
                "STFT",
                _stft_db(predicted_wave[:length], n_fft, int(hop_length)),
                _stft_db(target_wave[:length], n_fft, int(hop_length)),
                np.arange(n_fft // 2 + 1) * step,
                list(ticks),
                [_format_frequency(frequency) for frequency in ticks],
                max(1, round(VALIDATION_PREVIEW_STFT_POOL_HZ / step)),
            ))

    width, height = tuple(figsize) if figsize else VALIDATION_PREVIEW_FIGSIZE
    top_margin, bottom_margin, gap = VALIDATION_PREVIEW_MARGINS
    heights = [max(1.0, float(height) - top_margin - bottom_margin)]
    heights += [VALIDATION_PREVIEW_STFT_HEIGHT] * (len(rows) - 1)
    total = top_margin + bottom_margin + sum(heights) + gap * (len(rows) - 1)
    background, text, edge = VALIDATION_PREVIEW_BACKGROUND, "#f4f4f4", "#c9cdd3"
    figure = plt.figure(
        figsize=(float(width), total),
        dpi=float(dpi) if dpi else VALIDATION_PREVIEW_DPI,
        facecolor=background,
    )
    # Generated | original | their colourbar | gap | difference | mean by bin | its colourbar.
    grid = figure.add_gridspec(
        len(rows), 7, width_ratios=(1.0, 1.0, 0.025, 0.16, 1.0, 0.16, 0.025), height_ratios=heights,
        left=0.04, right=0.965, bottom=bottom_margin / total, top=1.0 - top_margin / total,
        wspace=0.05, hspace=gap / (sum(heights) / len(heights)),
    )
    difference_cmap = matplotlib.colormaps["RdBu_r"].copy()
    difference_cmap.set_bad(VALIDATION_PREVIEW_QUIET_COLOR)
    limit = VALIDATION_PREVIEW_DIFFERENCE_DB
    first = None

    for index, (row, predicted, target, bins, ticks, labels, pool) in enumerate(rows):
        frames = min(predicted.shape[1], target.shape[1])
        predicted, target = predicted[:, :frames], target[:, :frames]
        top = float(np.ceil(target.max() / 5.0) * 5.0)
        pooled_predicted, pooled_target = _pool_db(predicted, pool), _pool_db(target, pool)
        quiet = np.maximum(pooled_target, pooled_predicted) < top - VALIDATION_PREVIEW_QUIET_DB
        difference = np.ma.masked_array(pooled_predicted - pooled_target, mask=quiet)
        # The mean over the audible frames of each bin: a band's bias at a glance.
        profile = _band_profile(difference)
        biases = np.abs(profile[np.isfinite(profile)])
        profile_limit = max(3.0, float(np.ceil(biases.max() * 1.1))) if biases.size else 3.0
        # Half a bin past the first and last centres, so a bin's value sits on its centre.
        half = (bins[1] - bins[0]) / 2.0 if bins.size > 1 else 0.5
        extent = (0.0, frames * float(hop_length) / float(sample_rate), bins[0] - half, bins[-1] + half)

        generated_axis = figure.add_subplot(grid[index, 0], sharex=first)
        if first is None:
            first = generated_axis
        original_axis = figure.add_subplot(grid[index, 1], sharex=first, sharey=generated_axis)
        difference_axis = figure.add_subplot(grid[index, 4], sharex=first, sharey=generated_axis)
        profile_axis = figure.add_subplot(grid[index, 5], sharey=generated_axis)
        level_bar, difference_bar = figure.add_subplot(grid[index, 2]), figure.add_subplot(grid[index, 6])

        panels = (
            (generated_axis, predicted, "inferno", top - VALIDATION_PREVIEW_RANGE_DB, top),
            (original_axis, target, "inferno", top - VALIDATION_PREVIEW_RANGE_DB, top),
            (difference_axis, difference, difference_cmap, -limit, limit),
        )
        images = []
        for (axis, values, cmap, vmin, vmax), title in zip(panels, TENSORBOARD_VALIDATION_MEL_TITLES):
            images.append(axis.imshow(
                values, origin="lower", aspect="auto", interpolation="antialiased",
                cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
            ))
            axis.set_title(
                TENSORBOARD_VALIDATION_PANEL_TITLE.format(title=title, row=row),
                color=text, fontsize=15, fontweight="bold", pad=10,
            )

        profile_axis.axvline(0.0, color="#8a93a0", linewidth=0.8)
        profile_axis.plot(profile, bins, color=text, linewidth=1.2)
        profile_axis.set_xlim(-profile_limit, profile_limit)
        profile_axis.set_xticks((-profile_limit, 0.0, profile_limit))
        profile_axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        profile_axis.set_title(TENSORBOARD_VALIDATION_PROFILE_TITLE, color=text, fontsize=12, pad=10)
        profile_axis.grid(axis="x", color=VALIDATION_PREVIEW_QUIET_COLOR, linewidth=0.8)

        generated_axis.set_ylim(extent[2], extent[3])
        generated_axis.set_yticks(ticks, labels)
        generated_axis.set_ylabel(TENSORBOARD_VALIDATION_AXIS_Y, color=text, fontsize=11)
        original_axis.tick_params(labelleft=False)
        profile_axis.tick_params(labelleft=False)
        for axis in (generated_axis, original_axis, difference_axis, profile_axis):
            axis.set_facecolor(background)
            axis.tick_params(colors=text, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color(edge)
        if index == len(rows) - 1:
            for axis in (generated_axis, original_axis, difference_axis):
                axis.set_xlabel(TENSORBOARD_VALIDATION_AXIS_X, color=text, fontsize=11)
            profile_axis.set_xlabel(TENSORBOARD_VALIDATION_DB_LABEL, color=text, fontsize=11)

        for bar, image, label, extend in (
            (level_bar, images[0], TENSORBOARD_VALIDATION_DB_LABEL, "neither"),
            (difference_bar, images[2], TENSORBOARD_VALIDATION_DIFFERENCE_LABEL, "both"),
        ):
            colorbar = figure.colorbar(image, cax=bar, extend=extend)
            colorbar.set_label(label, color="#d7dbe1", fontsize=9)
            colorbar.ax.tick_params(colors="#d7dbe1", labelsize=8)
            colorbar.outline.set_edgecolor(edge)

    first.set_xlim(0.0, duration)
    figure.text(
        0.5,
        0.35 / total,
        TENSORBOARD_VALIDATION_FOOTER.format(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            n_mels=int(mel_bins),
            epoch=int(epoch),
            step=int(global_step),
            quiet=VALIDATION_PREVIEW_QUIET_DB,
        ),
        color="#bfc5ce",
        fontsize=10,
        ha="center",
    )
    return figure


def _write_atomically(path, write):
    """Write ``path`` through a temporary file and a rename.

    A reader -- the GUI polls these files -- or an interrupted run then sees
    the old file or the new one, never a WAV whose header was not finalised.
    """
    temporary = f"{path}.tmp"
    write(temporary)
    try:
        os.replace(temporary, path)
    except OSError:
        # Windows refuses to replace a file another process has open, such as
        # the GUI playing the previous preview; fall back to writing in place.
        os.remove(temporary)
        write(path)


def log_validation_preview(
    writer,
    experiment_dir,
    epoch,
    sample_index,
    global_step,
    sample_rate,
    predicted_mel,
    target_mel,
    predicted_wave=None,
    target_wave=None,
    source=None,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
):
    """Save and log one organized validation preview with mel difference."""
    sample_stem = f"sample_{int(sample_index):02d}"
    preview_dir = os.path.join(
        str(experiment_dir),
        TENSORBOARD_VALIDATION_PREVIEW_DIR,
        f"epoch_{int(epoch):04d}",
    )
    mel_dir = os.path.join(preview_dir, "mel")
    audio_dir = os.path.join(preview_dir, "audio")
    os.makedirs(mel_dir, exist_ok=True)

    figure = plot_validation_preview_to_figure(
        predicted_mel=predicted_mel,
        target_mel=target_mel,
        epoch=epoch,
        global_step=global_step,
        sample_index=sample_index,
        sample_rate=sample_rate,
        hop_length=hop_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        dpi=dpi,
        figsize=figsize,
        predicted_wave=predicted_wave,
        target_wave=target_wave,
    )
    image_path = os.path.join(mel_dir, f"{sample_stem}.png")
    try:
        figure.canvas.draw()
        composite = np.asarray(
            figure.canvas.buffer_rgba(),
            dtype=np.uint8,
        )[..., :3].copy()
        # ``figure.dpi`` rather than the module default: the figure may have
        # been built at an overridden dpi, and saving at a different one would
        # write a PNG that does not match what TensorBoard received.
        _write_atomically(
            image_path,
            lambda target: figure.savefig(target, dpi=figure.dpi, format="png"),
        )

        if writer is not None:
            writer.add_image(
                TENSORBOARD_VALIDATION_MEL_TAG.format(sample=sample_stem),
                torch.from_numpy(composite).permute(2, 0, 1),
                global_step,
                dataformats="CHW",
            )
            if source:
                writer.add_text(
                    TENSORBOARD_VALIDATION_SOURCE_TAG.format(sample=sample_stem),
                    str(source),
                    global_step=global_step,
                )
    finally:
        plt.close(figure)

    if predicted_wave is None or target_wave is None:
        return image_path

    os.makedirs(audio_dir, exist_ok=True)
    predicted_wave = limit_audio_peak(predicted_wave)
    target_wave = limit_audio_peak(target_wave)
    audio_length = min(predicted_wave.numel(), target_wave.numel())
    if audio_length <= 0:
        return image_path
    predicted_wave = predicted_wave[:audio_length]
    target_wave = target_wave[:audio_length]

    generated_path = os.path.join(audio_dir, f"{sample_stem}_generated.wav")
    original_path = os.path.join(audio_dir, f"{sample_stem}_original.wav")
    for path, wave in ((generated_path, predicted_wave), (original_path, target_wave)):
        _write_atomically(
            path,
            lambda target, wave=wave: sf.write(
                target, wave.numpy(), int(sample_rate), subtype="PCM_16", format="WAV"
            ),
        )

    if writer is not None:
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"],
            ),
            predicted_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["original"],
            ),
            target_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )

    return image_path


def log_tensorboard_media(
    writer,
    namespace,
    global_step,
    sample_rate,
    figure=None,
    audio=None,
    text=None,
):
    """Write one sample's text, mel figure, and audio under a single namespace."""
    if writer is None:
        return

    for name, value in (text or {}).items():
        writer.add_text(
            f"{namespace}/{name}",
            str(value),
            global_step=global_step,
        )
    if figure is not None:
        writer.add_figure(
            f"{namespace}/mel",
            figure,
            global_step=global_step,
        )
    for name, value in (audio or {}).items():
        writer.add_audio(
            f"{namespace}/audio/{name}",
            limit_audio_peak(value),
            global_step=global_step,
            sample_rate=sample_rate,
        )


def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    # The reference is embedder-specific, and nothing about the filename says
    # which embedder wrote it.  Handing 768-wide features to a 256-wide text
    # encoder used to reach ``F.linear`` and die there -- "mat1 and mat2 shapes
    # cannot be multiplied" -- a few thousand steps into a run, at the first
    # preview rather than at startup.  Checked here instead, and the run
    # continues on a reference taken from the dataset, which is right by
    # construction.
    if use_custom_ref:
        expected_dim = int(getattr(config.model, "text_enc_hidden_dim", 768))
        features = np.load(os.path.join(reference_path, "ref_feats.npy"))
        if features.ndim != 2 or features.shape[1] != expected_dim:
            found = "x".join(str(size) for size in features.shape)
            warning(
                f"logs/reference/ref_feats.npy is {found} but this model's text "
                f"encoder takes {expected_dim}-wide features; it was made with a "
                f"different embedder. Falling back to a reference from the "
                f"dataset. To use your own, rebuild it with the embedder this "
                f"model trains on: python tools/make_reference.py <audio> "
                f"--embedder <name>",
                tag="[REFERENCE]",
            )
            use_custom_ref = False

    if use_custom_ref:
        info("Using custom reference input from 'logs/reference/'.", tag="[REFERENCE]")
        reference_audio = None
        reference_source = reference_path

        phone = upsample_content(
            torch.FloatTensor(features),
            getattr(config.model, "content_interpolation", "nearest"),
        ).unsqueeze(0).to(device)
        pitch = torch.LongTensor(np.load(os.path.join(reference_path, "ref_f0c.npy"))).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(np.load(os.path.join(reference_path, "ref_f0f.npy"))).unsqueeze(0).to(device)

        # Measure lengths
        lengths = [phone.shape[1], pitch.shape[1], pitchf.shape[1]]
        min_len = min(lengths)

        # Trim to min length
        phone = phone[:, :min_len, :]
        pitch = pitch[:, :min_len]
        pitchf = pitchf[:, :min_len]
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)
        sid = torch.LongTensor([0]).to(device)

        # Optional ground truth for the preview; without it the preview
        # degrades to the generated waveform alone. Resampled on load, so it
        # can be at any rate -- one f0 frame is one hop at every configured
        # sample rate, both being 10 ms.
        audio_path = os.path.join(reference_path, "ref_audio.wav")
        if os.path.isfile(audio_path):
            from rvc.lib.utils import load_audio

            wave = load_audio(audio_path, config.data.sample_rate)
            wanted = min_len * config.data.hop_length
            if wave.shape[0] < wanted:
                # Short is survivable -- the figure crops both mels to the
                # frames they share -- but silently comparing less than the
                # reference renders is not, so it is said out loud.
                warning(
                    "ref_audio.wav is "
                    f"{wave.shape[0] / config.data.sample_rate:.2f}s, short of the "
                    f"{wanted / config.data.sample_rate:.2f}s the features render; "
                    "the preview will compare only the overlap.",
                    tag="[REFERENCE]",
                )
            reference_audio = (
                torch.FloatTensor(wave[:wanted]).view(1, 1, -1).to(device)
            )
        else:
            warning(
                "No ref_audio.wav; the preview will show the "
                "generated audio without the mel comparison.",
                tag="[REFERENCE]",
            )

    else:
        info("No custom reference found; fetching from train_loader.", tag="[REFERENCE]")
        batch = next(iter(train_loader))
        # Unpack everything from the loader
        phone, phone_lengths, pitch, pitchf, _, _, reference_audio, _, sid = batch

        # Move only the first sample of the batch to device
        phone = phone[0:1].to(device)
        phone_lengths = phone_lengths[0:1].to(device)
        pitch = pitch[0:1].to(device)
        pitchf = pitchf[0:1].to(device)
        reference_audio = reference_audio[0:1].to(device)
        sid = sid[0:1].to(device)

        batch_indices = []
        for batch in train_loader.batch_sampler:
            batch_indices = batch
            break

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            file_paths = train_loader.dataset.dataset.get_file_paths(batch_indices)
        else:
            file_paths = train_loader.dataset.get_file_paths(batch_indices)

        file_name = os.path.basename(file_paths[0])
        info(f"Origin of the ref: {file_name}", tag="[REFERENCE]")
        reference_source = file_name

    return (
        (phone, phone_lengths, pitch, pitchf, sid, config.train.seed),
        reference_audio,
        reference_source,
    )
