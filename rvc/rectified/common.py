import glob
import json
import math
import os
import random
import re
import shutil

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rvc.lib.algorithm.commons import upsample_content
from rvc.lib.algorithm.energy import frame_energy
from rvc.lib.paths import LOGS_DIR, MODELS_DIR, ROOT
from rvc.rectified.aperiodicity import aperiodicity
from rvc.rectified.mel import LogMel

DEFAULT_CONFIG = os.path.join(ROOT, "rvc", "configs", "rectified", "44100.json")
#: Frame rate of the extracted pitch, of the content once ``upsample_content``
#: has doubled it, and of ``frame_energy``.
FEATURE_RATE = 100
#: Where downloaded rectified pretrains (flow and vocoder exports) go.
PRETRAINED_DIR = os.path.join(MODELS_DIR, "pretraineds", "rectified")


#: The trainers' folders inside an experiment, one per model.
PARTS = ("vocoder", "flow")


def run_dir(model_name: str, part: str) -> str:
    """``logs/<model_name>/<part>``: checkpoints, exports, events and previews."""
    return os.path.join(LOGS_DIR, model_name, part)


def load_run_config(model_name: str) -> dict:
    """The experiment's rectified config, copied from the shipped one on first
    use. Not ``config.json``, which is the RVC config extraction writes."""
    path = os.path.join(LOGS_DIR, model_name, "rectified_config.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG, path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_filelist(model_name: str):
    """``filelist.txt`` rows as [audio, content, f0, f0_voiced, sid], paths
    resolved against the application root.

    Not ``rvc.train.utils.load_filepaths_and_text``: that module only imports
    with ``rvc/train`` on ``sys.path``, which the interface does not have.
    """
    path = os.path.join(LOGS_DIR, model_name, "filelist.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist; preprocess and extract the dataset first."
        )
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("|")
            if len(fields) < 5:
                continue
            for column in range(4):
                fields[column] = os.path.normpath(os.path.join(ROOT, fields[column]))
            rows.append(fields)
    return rows


def speaker_count(entries) -> int:
    return max(int(entry[4]) for entry in entries) + 1


#: Width of the sine window the loudness and breathiness curves are smoothed
#: with, as DiffSinger smooths its variance curves.
SMOOTH_SECONDS = 0.06


def smooth_curve(curve: torch.Tensor) -> torch.Tensor:
    """``curve`` [batch, time] at ``FEATURE_RATE``, smoothed over ``SMOOTH_SECONDS``."""
    width = int(round(SMOOTH_SECONDS * FEATURE_RATE))
    kernel = torch.sin(torch.linspace(0, 1, width + 2, device=curve.device)[1:-1] * math.pi)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = F.pad(curve.unsqueeze(1), ((width - 1) // 2, width // 2), mode="replicate")
    return F.conv1d(padded, kernel.to(curve.dtype)).squeeze(1)


def split_holdout(entries, count: int, seed: int = 1234):
    """``(train, holdout)``: ``count`` non-mute clips drawn with ``seed``, kept
    out of training for the validation loss."""
    candidates = [i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])]
    held = set(random.Random(seed).sample(candidates, min(count, len(candidates) // 10)))
    train = [entry for i, entry in enumerate(entries) if i not in held]
    return train, [entries[i] for i in sorted(held)]


def mel_frames(feature_frames: int, sample_rate: int, hop: int) -> int:
    """Mel frames covered by ``feature_frames`` frames at ``FEATURE_RATE``."""
    return int(feature_frames * sample_rate / (hop * FEATURE_RATE))


def _positions(length: int, frames: int, sample_rate: int, hop: int, device):
    """Neighbouring ``FEATURE_RATE`` indices and weight for each mel frame."""
    position = torch.arange(frames, device=device, dtype=torch.float64)
    position = (position * hop * FEATURE_RATE / sample_rate).clamp(max=length - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=length - 1)
    return left, right, (position - left).float()


def to_mel_rate(features: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    """``[..., time, channels]`` at ``FEATURE_RATE`` -> ``frames`` mel frames, linearly."""
    left, right, weight = _positions(features.shape[-2], frames, sample_rate, hop, features.device)
    weight = weight.unsqueeze(-1)
    return features[..., left, :] * (1 - weight) + features[..., right, :] * weight


#: Half-width, in frames, of the median an octave error is judged against.
OCTAVE_RADIUS = 25


def _voiced_median(log_f0: np.ndarray, voiced: np.ndarray, radius: int) -> np.ndarray:
    """Median of the voiced ``log_f0`` within ``radius`` frames of each voiced
    frame; unvoiced frames are left as they are."""
    values = np.where(voiced, log_f0, np.nan)
    padded = np.pad(values, radius, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    out = log_f0.copy()
    out[voiced] = np.nanmedian(windows[voiced], axis=-1)
    return out


def clean_f0(f0: np.ndarray, median_radius: int = 0, fix_octaves: bool = False) -> np.ndarray:
    """Pitch at ``FEATURE_RATE`` with octave jumps folded back onto the local
    median and a median filter of ``median_radius`` frames over voiced frames;
    unvoiced frames stay 0."""
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = f0 > 0
    if not voiced.any() or (median_radius <= 0 and not fix_octaves):
        return f0
    log_f0 = np.log2(np.where(voiced, f0, 1.0))
    if fix_octaves:
        jump = log_f0 - _voiced_median(log_f0, voiced, OCTAVE_RADIUS)
        log_f0 = np.where(voiced & (np.abs(jump) > 0.75), log_f0 - np.round(jump), log_f0)
    if median_radius > 0:
        log_f0 = _voiced_median(log_f0, voiced, int(median_radius))
    return np.where(voiced, np.exp2(log_f0), 0.0).astype(np.float32)


def f0_to_mel_rate(f0: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    """``[..., time]`` pitch at ``FEATURE_RATE`` -> ``frames`` mel frames.

    Interpolated only between two voiced frames; elsewhere the nearer frame, so
    no frame gets a pitch halfway to zero.
    """
    left, right, weight = _positions(f0.shape[-1], frames, sample_rate, hop, f0.device)
    a, b = f0[..., left], f0[..., right]
    nearest = torch.where(weight < 0.5, a, b)
    return torch.where((a > 0) & (b > 0), a * (1 - weight) + b * weight, nearest)


def embedder_of(model_name: str) -> str:
    from rvc.lib.catalog import experiment_embedder

    return experiment_embedder(model_name, LOGS_DIR)


def latest_checkpoint(directory: str, prefix: str):
    """Newest ``<prefix>_<step>.pth`` in ``directory``, or None."""
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.pth$")
    found = []
    for path in glob.glob(os.path.join(directory, f"{prefix}_*.pth")):
        match = pattern.search(os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


#: What a run keeps of its training checkpoints (``<prefix>_<step>.pth``):
#: the latest, all of them, or none, when only the exports are written and a
#: stopped run cannot resume. Exports are always kept.
CHECKPOINT_MODES = ("latest", "all", "none")


def remove_older(directory: str, prefix: str, keep: str | None = None) -> None:
    """Delete ``<prefix>_*.pth`` in ``directory`` but ``keep``; all without it."""
    for path in glob.glob(os.path.join(directory, f"{prefix}_*.pth")):
        if keep is None or os.path.abspath(path) != os.path.abspath(keep):
            os.remove(path)


def pretrained_weights(path: str) -> dict:
    """Weights from a training checkpoint (its EMA when present) or an export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture"):
        raise ValueError(
            f"{path} is a {checkpoint['architecture']} vocoder; it renders previews "
            f"and audio but cannot start a rectified training run."
        )
    ema = checkpoint.get("ema")
    return ema["shadow"] if ema else checkpoint["model"]


def check_pretrain_embedder(path: str, embedder: str) -> None:
    """Raise when the flow export at ``path`` was trained on another content
    embedder's features than ``embedder``. A checkpoint records none, so it
    passes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    trained_on = checkpoint.get("embedder_model")
    if trained_on and trained_on != embedder:
        raise ValueError(
            f"{os.path.basename(path)} was trained on {trained_on} features, and this "
            f"experiment was extracted with {embedder}. Use a {embedder} pretrain, or "
            f"train from scratch."
        )


def read_audio(path: str, sample_rate: int) -> torch.Tensor:
    """Mono ``path``, which must already be at ``sample_rate``."""
    data, rate = sf.read(path, dtype="float32")
    audio = torch.from_numpy(data)
    if rate != sample_rate:
        raise ValueError(
            f"{path} is {rate} Hz; the rectified models train at "
            f"{sample_rate} Hz. Preprocess the dataset at that rate."
        )
    return audio.mean(-1) if audio.dim() == 2 else audio


def clip_curves(audio: torch.Tensor, f0: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Smoothed loudness and aperiodic share of ``audio`` [samples], whose
    pitch ``f0`` is at ``FEATURE_RATE``; [time, 2] at ``FEATURE_RATE``."""
    feature_frames = audio.shape[-1] // (sample_rate // FEATURE_RATE)
    audio = audio.unsqueeze(0)
    energy = frame_energy(audio, sample_rate, feature_frames)
    share = aperiodicity(audio, sample_rate, f0.unsqueeze(0), feature_frames)
    return smooth_curve(torch.cat([energy, share])).T.contiguous()


#: Bump when ``clip_curves`` changes, so stored curves are measured again.
CURVES_VERSION = 1


def curves_path(model_name: str) -> str:
    return os.path.join(LOGS_DIR, model_name, "rectified_curves.npz")


def _curve_clips(entries):
    """Each distinct (audio, pitch) pair of ``entries``, keyed by the audio
    path relative to the root, with both files' mtimes."""
    clips = {}
    for entry in entries:
        key = os.path.relpath(entry[0], ROOT)
        if key not in clips:
            clips[key] = (entry[0], entry[3], os.stat(entry[0]).st_mtime_ns, os.stat(entry[3]).st_mtime_ns)
    return clips


class _CurveSource(Dataset):
    def __init__(self, clips, sample_rate: int):
        self.clips = clips
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        audio_path, f0_path = self.clips[index]
        f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
        return clip_curves(read_audio(audio_path, self.sample_rate), f0, self.sample_rate)


def measure_clip_curves(model_name: str, entries, sample_rate: int, workers: int) -> None:
    """Write ``curves_path`` with every clip's ``clip_curves``, unless it is
    already there for these clips and their audio and pitch are unchanged."""
    from rvc.lib.terminal import info, progress_task

    path = curves_path(model_name)
    clips = _curve_clips(entries)
    try:
        with np.load(path, allow_pickle=False) as stored:
            if int(stored["version"]) == CURVES_VERSION:
                known = dict(zip(stored["keys"].tolist(), map(tuple, stored["stamps"].tolist())))
                if all(known.get(key) == clip[2:] for key, clip in clips.items()):
                    return
    except (OSError, KeyError, ValueError):
        pass

    info(f"Measuring loudness and breathiness of {len(clips)} clips.", tag="[CURVES]")
    source = _CurveSource([clip[:2] for clip in clips.values()], sample_rate)
    curves = []
    with progress_task(len(source), "Loudness and breathiness") as (progress, task):
        for item in DataLoader(source, batch_size=None, num_workers=workers):
            curves.append(item.numpy())
            progress.update(task, advance=1)
    offsets = np.cumsum([0] + [len(item) for item in curves])
    partial = f"{path}.tmp"
    with open(partial, "wb") as handle:
        np.savez(
            handle, version=CURVES_VERSION, keys=np.array(list(clips)),
            stamps=np.array([clip[2:] for clip in clips.values()], dtype=np.int64).reshape(-1, 2),
            offsets=offsets, curves=np.concatenate(curves).astype(np.float32),
        )
    os.replace(partial, path)


class ClipCurves:
    """The curves ``measure_clip_curves`` wrote, looked up by audio path."""

    def __init__(self, model_name: str):
        with np.load(curves_path(model_name), allow_pickle=False) as stored:
            self.curves = torch.from_numpy(stored["curves"])
            offsets = stored["offsets"].tolist()
            self.spans = {key: (offsets[i], offsets[i + 1]) for i, key in enumerate(stored["keys"].tolist())}

    def get(self, audio_path: str) -> torch.Tensor:
        start, stop = self.spans[os.path.relpath(audio_path, ROOT)]
        return self.curves[start:stop]


class RectifiedDataset(Dataset):
    """Clips from an extracted RVC experiment.

    ``vocoder`` items are fixed-length (mel, f0, audio) crops; ``flow`` items
    are (mel, content, f0, energy, breathiness, key_shift, speed, speaker)
    crops of up to ``segment_frames``. The mel is taken over the whole clip
    before cropping, as at inference. Flow items are pitch-shifted and
    time-stretched at random with the config's probabilities; without
    ``augment``, they are neither, nor randomly cropped. Flow items read
    loudness and breathiness from ``curves`` when given, a ``ClipCurves``.
    """

    def __init__(self, entries, config: dict, mode: str, segment_frames: int, augment: bool = True,
                 curves=None):
        if mode not in ("vocoder", "flow"):
            raise ValueError(f"mode must be 'vocoder' or 'flow', not {mode!r}.")
        self.entries = entries
        self.data = config["data"]
        self.mode = mode
        self.segment_frames = int(segment_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.content_channels = int(config["flow"]["model"]["content_channels"])
        self.key_shift_range = float(config["flow"].get("key_shift_range", 0.0))
        self.key_shift_prob = float(config["flow"].get("key_shift_prob", 0.0))
        self.stretch_range = tuple(config["flow"].get("time_stretch_range", (1.0, 1.0)))
        self.stretch_prob = float(config["flow"].get("time_stretch_prob", 0.0))
        self.augment = augment
        self.curves = curves

    def __len__(self):
        return len(self.entries)

    def _audio(self, path):
        return read_audio(path, self.sample_rate)

    def __getitem__(self, index):
        wav_path, content_path, _, f0_path, sid = self.entries[index]
        audio = self._audio(wav_path)
        source_f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
        if self.mode == "vocoder":
            return self._vocoder_item(audio, source_f0)
        content = upsample_content(
            torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
            self.data["content_interpolation"],
        )
        if self.curves is not None:
            curves = self.curves.get(wav_path)
        else:
            curves = clip_curves(audio, source_f0, self.sample_rate)
        return self._flow_item(audio, source_f0, content, curves, int(sid))

    def _vocoder_item(self, audio, f0):
        frames = min(audio.shape[0] // self.hop, mel_frames(f0.shape[0], self.sample_rate, self.hop))
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        if frames < self.segment_frames:
            audio = F.pad(audio, (0, self.segment_frames * self.hop - audio.shape[0]))
            f0 = F.pad(f0, (0, self.segment_frames - f0.shape[0]))
            frames = self.segment_frames
        audio = audio[: frames * self.hop]
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[0, :, :frames]
        start = random.randint(0, frames - self.segment_frames)
        stop = start + self.segment_frames
        return mel[:, start:stop], f0[start:stop], audio[start * self.hop : stop * self.hop]

    def _flow_item(self, audio, source_f0, content, curves, sid):
        """A crop shifted by ``key_shift`` semitones (pitch and formants) and
        stretched by ``speed`` (a longer hop reads the clip faster), both drawn
        when augmenting. ``curves`` are the clip's ``clip_curves``."""
        key_shift, hop = 0.0, self.hop
        if self.augment and self.key_shift_range > 0 and random.random() < self.key_shift_prob:
            key_shift = random.uniform(-self.key_shift_range, self.key_shift_range)
        if self.augment and self.stretch_prob > 0 and random.random() < self.stretch_prob:
            low, high = self.stretch_range
            hop = int(round(self.hop * low * (high / low) ** random.random()))
        speed = hop / self.hop

        frames = min(
            audio.shape[0] // hop,
            mel_frames(source_f0.shape[0], self.sample_rate, hop),
            mel_frames(content.shape[0], self.sample_rate, hop),
        )
        audio = audio[: frames * hop]
        content = to_mel_rate(content, frames, self.sample_rate, hop)
        f0 = f0_to_mel_rate(source_f0, frames, self.sample_rate, hop) * 2.0 ** (key_shift / 12.0)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), key_shift, hop)[0, :, :frames]
        energy, breathiness = to_mel_rate(curves, frames, self.sample_rate, hop).unbind(-1)

        length = min(frames, self.segment_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        return (
            mel[:, start:stop], content[start:stop], f0[start:stop], energy[start:stop],
            breathiness[start:stop], key_shift, speed, sid,
        )

    def _reference_item(self, audio, content, f0, sid, path, max_frames=None):
        """``content`` and ``f0`` at ``FEATURE_RATE``."""
        frames = min(
            audio.shape[0] // self.hop,
            mel_frames(min(f0.shape[0], content.shape[0]), self.sample_rate, self.hop),
        )
        if max_frames is not None:
            frames = min(frames, max_frames)
        audio = audio[: frames * self.hop]
        content = to_mel_rate(content, frames, self.sample_rate, self.hop)
        curves = to_mel_rate(clip_curves(audio, f0, self.sample_rate), frames, self.sample_rate, self.hop)
        energy, breathiness = curves.T.unsqueeze(1)
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
        return (
            mel, content.unsqueeze(0), f0.unsqueeze(0), energy, breathiness,
            audio.unsqueeze(0), int(sid), path,
        )

    def _custom_reference(self):
        """``logs/reference`` as the RVC trainer reads it, speaker 0, or None.

        Needs ``ref_audio.wav`` as well as the features: the vocoder renders
        from its mel and the flow reads its loudness.
        """
        from rvc.lib.audio_io import load_audio
        from rvc.lib.terminal import info, warning

        folder = os.path.join(LOGS_DIR, "reference")
        feats_path = os.path.join(folder, "ref_feats.npy")
        f0_path = os.path.join(folder, "ref_f0f.npy")
        audio_path = os.path.join(folder, "ref_audio.wav")
        if not (os.path.isfile(feats_path) and os.path.isfile(f0_path)):
            return None
        if not os.path.isfile(audio_path):
            warning("logs/reference has no ref_audio.wav; previewing a dataset clip instead.", tag="[REFERENCE]")
            return None
        features = np.load(feats_path)
        if features.ndim != 2 or features.shape[1] != self.content_channels:
            warning(
                f"logs/reference/ref_feats.npy is {'x'.join(map(str, features.shape))} but this "
                f"model takes {self.content_channels}-wide features; it was made with a different "
                f"embedder. Previewing a dataset clip instead.",
                tag="[REFERENCE]",
            )
            return None
        content = upsample_content(
            torch.from_numpy(features.astype(np.float32)), self.data["content_interpolation"]
        )
        f0 = torch.from_numpy(np.load(f0_path).astype(np.float32))
        audio = torch.from_numpy(load_audio(audio_path, self.sample_rate).astype(np.float32))
        info("Using custom reference input from 'logs/reference/'.", tag="[REFERENCE]")
        return self._reference_item(audio, content, f0, 0, folder)

    def reference(self, max_seconds: float = 10.0):
        """The preview clip, (mel, content, f0, energy, breathiness, audio, sid,
        path) with a leading batch axis, or None.

        ``logs/reference`` when it is usable, as in the RVC trainer; otherwise
        the first non-mute dataset clip of at least two seconds, in path order,
        cut to ``max_seconds``.
        """
        custom = self._custom_reference()
        if custom is not None:
            return custom
        ordered = sorted(range(len(self.entries)), key=lambda i: self.entries[i][0])
        for index in ordered:
            wav_path, content_path, _, f0_path, sid = self.entries[index]
            if "mute" in os.path.basename(wav_path):
                continue
            audio = self._audio(wav_path)
            if audio.shape[0] < 2 * self.sample_rate:
                continue
            f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
            content = upsample_content(
                torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
                self.data["content_interpolation"],
            )
            max_frames = int(max_seconds * self.sample_rate) // self.hop
            return self._reference_item(audio, content, f0, sid, wav_path, max_frames)
        return None


def collate_vocoder(batch):
    mel, f0, audio = zip(*batch)
    return torch.stack(mel), torch.stack(f0), torch.stack(audio).unsqueeze(1)


def collate_flow(batch, frames=None):
    """Pads to ``frames``, or to the longest item; returns a [B, 1, T] mask.
    A fixed length keeps every batch one shape for ``torch.compile``."""
    frames = frames or max(item[0].shape[-1] for item in batch)
    size = len(batch)
    mel = torch.zeros(size, batch[0][0].shape[0], frames)
    content = torch.zeros(size, frames, batch[0][1].shape[-1])
    f0 = torch.zeros(size, frames)
    energy = torch.full((size, frames), -1.0)
    breathiness = torch.ones(size, frames)
    key_shift = torch.zeros(size)
    speed = torch.ones(size)
    mask = torch.zeros(size, 1, frames)
    speaker = torch.zeros(size, dtype=torch.long)
    for i, (m, c, p, e, b, k, v, s) in enumerate(batch):
        n = m.shape[-1]
        mel[i, :, :n] = m
        content[i, :n] = c
        f0[i, :n] = p
        energy[i, :n] = e
        breathiness[i, :n] = b
        key_shift[i] = k
        speed[i] = v
        mask[i, :, :n] = 1.0
        speaker[i] = s
    return mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask


def _pretrained_exports(kind: str) -> list:
    """``kind`` exports in ``PRETRAINED_DIR``: ``*_<kind>_<suffix>.pth``, or a
    download named plainly ``*_<kind>.pth``."""
    return (glob.glob(os.path.join(PRETRAINED_DIR, f"*_{kind}_*.pth"))
            + glob.glob(os.path.join(PRETRAINED_DIR, f"*_{kind}.pth")))


def list_exports(kind: str) -> list:
    """Exported ``kind`` models (``flow`` or ``vocoder``) under
    ``logs/*/<kind>`` and ``PRETRAINED_DIR``; for flows, also the bundles
    holding one; for vocoders, the OpenVPI checkpoints (``*.ckpt``)."""
    from rvc.lib.catalog import list_bundles, relative, sort_key
    from rvc.lib.model_bundle import RECTIFIED_KIND

    paths = glob.glob(os.path.join(LOGS_DIR, "*", kind, f"*_{kind}_*.pth"))
    paths += _pretrained_exports(kind)
    if kind == "vocoder":
        paths += glob.glob(os.path.join(PRETRAINED_DIR, "*.ckpt"))
    found = sorted((relative(path) for path in paths), key=sort_key)
    if kind == "flow":
        found += list_bundles(LOGS_DIR, RECTIFIED_KIND)
    return found


def resolve_vocoder(reference: str) -> str:
    """The vocoder a flow names, as a path here, or "". A bundle made on
    another machine names a path that may not exist, so the file name is
    looked up among this machine's vocoders too."""
    from rvc.lib.catalog import relative

    if not reference:
        return ""
    if os.path.exists(reference):
        return relative(reference)
    name = os.path.basename(reference.replace("\\", "/"))
    return next((path for path in list_exports("vocoder") if os.path.basename(path) == name), "")


def describe_flow(path: str, submodel: str | None = None) -> dict:
    """``submodels`` (the flows in a bundle, else []), ``submodel`` (the one
    described), its ``speakers`` and its resolved ``vocoder``."""
    from rvc.lib.model_bundle import RECTIFIED_KIND, bundle_model_info, bundle_model_names, is_model_bundle

    if not path or not os.path.isfile(path):
        return {"submodels": [], "submodel": "", "speakers": [0], "vocoder": ""}
    if is_model_bundle(path):
        names = bundle_model_names(path, RECTIFIED_KIND)
        name = submodel if submodel in names else (names[0] if names else "")
        info = bundle_model_info(path, name) if name else {}
        count, reference = info.get("speakers_id"), info.get("vocoder") or ""
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        names, name = [], ""
        count, reference = checkpoint.get("speaker_count"), checkpoint.get("vocoder") or ""
    return {
        "submodels": names,
        "submodel": name,
        "speakers": list(range(int(count or 1))),
        "vocoder": resolve_vocoder(reference),
    }


def list_pretrained(kind: str) -> list:
    """Starting points for a run: ``vocoder_g``, ``vocoder_d`` or ``flow``.

    Other runs' training checkpoints and exports, plus the custom pretrained
    folder.
    """
    from rvc.lib.catalog import list_custom_pretraineds, relative

    patterns = {
        "vocoder_g": ["G_*.pth", "*_vocoder_*.pth", "*_vocoder.pth"],
        "vocoder_d": ["D_*.pth"],
        "flow": ["F_*.pth", "*_flow_*.pth", "*_flow.pth"],
    }[kind]
    folder = "flow" if kind == "flow" else "vocoder"
    found = []
    for pattern in patterns:
        found += glob.glob(os.path.join(LOGS_DIR, "*", folder, pattern))
        if not pattern.endswith(("G_*.pth", "D_*.pth", "F_*.pth")):
            found += glob.glob(os.path.join(PRETRAINED_DIR, pattern))
    custom = list_custom_pretraineds({"vocoder_g": "G", "vocoder_d": "D", "flow": "F"}[kind])
    return sorted(relative(path) for path in found) + custom


def amp_setup(precision: str, device: torch.device, tag: str):
    """Autocast dtype and GradScaler for the run's precision.

    FP16 needs the scaler; BF16 has FP32's exponent range and does not. A
    precision the GPU cannot run falls back to FP32.
    """
    from rvc.lib.terminal import warning

    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return None, None
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        warning("BF16 is not supported on this GPU; training in FP32.", tag=tag)
        return None, None
    if precision == "fp16":
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10, growth_interval=2000)
        return torch.float16, scaler
    return torch.bfloat16, None


def precision_label(amp_dtype) -> str:
    tf32 = torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    if amp_dtype is None:
        return "FP32 (TF32 matmul/conv)" if tf32 else "FP32"
    if amp_dtype == torch.float16:
        return "FP16 autocast + GradScaler"
    return "BF16 autocast"


def default_pretrained(kind: str):
    """The newest ``kind`` export (``flow`` or ``vocoder``) in ``PRETRAINED_DIR``,
    else for vocoders the newest OpenVPI checkpoint there, or None."""
    from rvc.lib.catalog import sort_key

    paths = _pretrained_exports(kind)
    if not paths and kind == "vocoder":
        paths = glob.glob(os.path.join(PRETRAINED_DIR, "*.ckpt"))
    return max(paths, key=sort_key) if paths else None
