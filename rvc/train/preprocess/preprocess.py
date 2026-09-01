import os
import sys
import time
import subprocess
from datetime import datetime
from scipy import signal
from scipy.io import wavfile
import numpy as np
import concurrent.futures
import json
from distutils.util import strtobool
import librosa
import multiprocessing
import shutil
import soundfile as sf
import io
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.terminal import (
    DEFAULT_CPU_THREADS,
    configure_logging,
    install_rich_print,
    progress_task,
    success,
    track,
)

install_rich_print()
configure_logging(tag="[PREPROCESS]")

from rvc.lib.audio_io import load_audio, load_audio_ffmpeg
from rvc.train.preprocess.slicer import Slicer

import logging
logger = logging.getLogger(__name__)

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
# Shortest tail worth emitting as its own slice in Simple cutting. Training
# samples a `segment_size` window out of each slice, so anything below this is
# not useful material.
MIN_TAIL_SECONDS = 1.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"
FLAC_COMPRESSION_LEVEL = 5 / 8 # FLAC level 5, libsndfile uses 0.0 - 1.0 range for compression level

def secs_to_samples(secs, sr):
    """Return an *exact* integer number of samples for `secs` seconds at `sr` Hz.
       Raises if the result is not an integer (prevents float drift)."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator

def save_audio(path: str, name: str, sample_rate: int, format: str, audio: np.ndarray):
    if format.lower() == "flac":
        memory_file = io.BytesIO()
        sf.write(
            memory_file,
            audio,
            sample_rate,
            format="FLAC",
            subtype="PCM_24",
            compression_level=FLAC_COMPRESSION_LEVEL
        )
        memory_file.seek(0)
        with open(os.path.join(path, f"{name}.flac"), "wb") as f:
            f.write(memory_file.read())
    else:
        wavfile.write(
            os.path.join(path, f"{name}.wav"),
            sample_rate,
            audio.astype(np.float32),
        )

class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        # Second-order sections rather than transfer-function coefficients:
        # at 48 Hz / 44.1 kHz the normalized cutoff is 0.0022, which pushes the
        # poles to |p| = 0.9979 and makes the `ba` form badly conditioned. It
        # still holds up in float64, but `sos` is the form scipy recommends here
        # and it lets the filter run natively in float32.
        self.hp_sos = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr, output="sos"
        )
        self.hp_zi = signal.sosfilt_zi(self.hp_sos)
        self.exp_dir = exp_dir

        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)


    def high_pass(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset and subsonic rumble.

        The filter state is primed from the first sample instead of starting
        from zero. A zero-state IIR treats a file that begins mid-waveform as a
        step input, and with poles this close to the unit circle (tau = 10.7 ms)
        that step rings for ~45 ms at up to 80% of the signal's peak -- which
        lands squarely inside the first slice of every file.
        """
        if audio.size == 0:
            return audio
        filtered, _ = signal.sosfilt(
            self.hp_sos, audio, zi=self.hp_zi * float(audio[0])
        )
        # sosfilt promotes to float64; the rest of the pipeline is float32 and
        # the extra precision is discarded on save anyway.
        return filtered.astype(np.float32, copy=False)

    def process_audio_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
        dataset_format: str
    ):
        save_audio(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}", self.sr, dataset_format, audio)

        if loading_resampling == "librosa":
            chunk_16k = librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        else:
            chunk_16k = load_audio_ffmpeg(
                audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
            )

        save_audio(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}", SAMPLE_RATE_16K, dataset_format, chunk_16k)


    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        overlap_smpl = secs_to_samples(overlap_len, self.sr)
        stride = chunk_len_smpl - overlap_smpl
        if stride <= 0:
            # A non-positive stride makes the cursor stand still (or walk
            # backwards) and the loop writes slices until the disk fills.
            raise ValueError(
                f"Simple cutting needs overlap_len < chunk_len, "
                f"got chunk_len={chunk_len}s overlap_len={overlap_len}s."
            )

        total = len(audio)
        min_tail = secs_to_samples(MIN_TAIL_SECONDS, self.sr)
        slice_idx = 0
        last_start = None

        for start in range(0, total, stride):
            end = start + chunk_len_smpl
            if end > total:
                # Slide the window back to end on the last sample instead of
                # padding the tail with silence, which would put a hard mel
                # floor into the dataset.
                remainder = total - start
                if remainder < min_tail:
                    break
                if total >= chunk_len_smpl:
                    start = total - chunk_len_smpl
                    if last_start is not None and start <= last_start:
                        break  # the re-anchored window would duplicate the previous one
                    chunk = audio[start:]
                else:
                    # Whole file shorter than one chunk; variable-length
                    # slices are supported downstream (Automatic emits them
                    # routinely).
                    chunk = audio
            else:
                chunk = audio[start:end]

            self.process_audio_segment(
                chunk, sid, idx0, slice_idx, loading_resampling, dataset_format
            )
            last_start = start
            slice_idx += 1

            if start + chunk_len_smpl >= total:
                break

    def chunk_segments(
        self,
        audio_segments,
        sid: int,
        idx0: int,
        loading_resampling: str,
        dataset_format: str,
    ):
        """Cut each voiced segment into overlapping ``PERCENTAGE``-second slices.

        Shared by ``Automatic`` and ``New Automatic``: the two differ only in
        how the voiced segments were found, and the slice geometry downstream
        has to stay identical either way.  Segments arrive as an iterable so a
        slicer can stay a generator.
        """
        idx1 = 0
        for audio_segment in audio_segments:
            i = 0
            while True:
                start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                i += 1
                if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                    tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                    self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                    idx1 += 1
                else:
                    tmp_audio = audio_segment[start:]
                    self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                    idx1 += 1
                    break

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str,
        normalization_mode: str = "none",
    ):
        audio_length = 0
        try:
            if loading_resampling == "librosa":
                audio = load_audio(path, self.sr)  # SoXr resampler
            else:
                audio = load_audio_ffmpeg(path, self.sr)  # windowed-sinc, Blackman-Nuttall

            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            if process_effects:
                audio = self.high_pass(audio)
            if noise_reduction:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=reduction_strength)

            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, loading_resampling, dataset_format)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, loading_resampling, dataset_format)
            elif cut_preprocess == "Automatic":
                self.chunk_segments(
                    self.slicer.slice(audio), sid, idx0, loading_resampling, dataset_format
                )
            elif cut_preprocess == "New Automatic":
                from rvc.train.preprocess import vad

                self.chunk_segments(
                    (audio[start:end] for start, end in vad.segments(audio, self.sr)),
                    sid, idx0, loading_resampling, dataset_format,
                )
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            raise e
        return audio_length

def _process_audio_worker(args):
    (
        path,
        idx0,
        sid,
        sr,
        exp_dir,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
        dataset_format,
        normalization_mode,
    ) = args
    pp = PreProcess(sr, exp_dir)
    return pp.process_audio(
        path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        loading_resampling,
        dataset_format,
        normalization_mode,
    )

def _dry_run_check_file(args):
    file_name, gt_wavs_dir, wavs16k_dir, target_rms, headroom, silence_thresh, eps, rms_norm_db = args
    worst_in_file = None
    for audio_dir in [gt_wavs_dir, wavs16k_dir]:
        audio, _ = sf.read(os.path.join(audio_dir, file_name))
        mask = np.abs(audio) > silence_thresh
        if np.any(mask):
            rms = np.sqrt(np.mean(audio[mask] ** 2) + eps)
            gain = target_rms / rms
        else:
            gain = 1.0

        peak = np.abs(audio * gain).max()
        if peak > headroom:
            gain_db = 20 * np.log10(gain)
            peak_db = 20 * np.log10(peak)
            crest_db = peak_db - rms_norm_db
            safe_max_db = -0.5 - crest_db
            if worst_in_file is None or safe_max_db < worst_in_file["safe_max_db"]:
                worst_in_file = dict(gain_db=gain_db, peak_db=peak_db, crest_db=crest_db, safe_max_db=safe_max_db)
    return worst_in_file


def _dry_run_post_rms(gt_wavs_dir, wavs16k_dir, audio_files, rms_norm_db, num_processes):
    """Compute what post_rms would do without modifying files. Returns (is_safe, worst_safe_db, summary)."""
    target_rms = 10 ** (rms_norm_db / 20)
    headroom = 10 ** (-0.5 / 20)
    silence_thresh = 10 ** (-40.0 / 20)
    eps = 1e-9

    arg_list = [
        (f, gt_wavs_dir, wavs16k_dir, target_rms, headroom, silence_thresh, eps, rms_norm_db)
        for f in audio_files
    ]

    gain_dbs, peak_dbs, crest_dbs, safe_maxs = [], [], [], []

    with multiprocessing.Pool(processes=pool_size(num_processes, len(audio_files))) as pool:
        for result in pool.imap_unordered(_dry_run_check_file, arg_list):
            if result:
                gain_dbs.append(result["gain_db"])
                peak_dbs.append(result["peak_db"])
                crest_dbs.append(result["crest_db"])
                safe_maxs.append(result["safe_max_db"])

    if safe_maxs:
        worst_safe = min(safe_maxs)
        return False, worst_safe, {
            "num_limited": len(safe_maxs),
            "avg_gain": np.mean(gain_dbs),
            "avg_peak": np.mean(peak_dbs),
            "avg_crest": np.mean(crest_dbs),
        }

    return True, None, None


def _apply_post_norm_from_gain(audio: np.ndarray, gt_audio: np.ndarray, mode: str, rms_norm_db: float):
    """Apply normalization using the gain computed from gt_audio, so gt and 16k stay loudness-consistent."""
    if mode == "post_rms":
        eps = 1e-9
        target_rms = 10 ** (rms_norm_db / 20)
        headroom = 10 ** (-0.5 / 20)
        silence_thresh = 10 ** (-40.0 / 20)
        mask = np.abs(gt_audio) > silence_thresh
        if np.any(mask):
            gt_rms = np.sqrt(np.mean(gt_audio[mask] ** 2) + eps)
            gain = target_rms / gt_rms
        else:
            gain = 1.0
        audio2 = audio * gain
        peak = np.abs(audio2).max()
        if peak > headroom:
            audio2 = audio2 / peak * headroom
        return audio2.astype(np.float32)

    elif mode == "post_peak_rvc":
        a_max = np.abs(gt_audio).max()
        if a_max <= 0:
            return audio.astype(np.float32)
        return ((audio / a_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio).astype(np.float32)

    elif mode == "post_peak":
        peak = np.max(np.abs(gt_audio))
        if peak > 0:
            return (audio / peak * 0.95).astype(np.float32)
        return audio.astype(np.float32)

    return audio.astype(np.float32)


def _apply_post_norm(audio: np.ndarray, sr: int, mode: str, rms_norm_db: float):
    """Returns (audio, stats), where stats is None or a dict of limiting info."""
    if mode == "post_rms":
        eps = 1e-9
        target_rms = 10 ** (rms_norm_db / 20)
        headroom = 10 ** (-0.5  / 20)
        silence_thresh = 10 ** (-40.0 / 20)
        mask = np.abs(audio) > silence_thresh
        if np.any(mask):
            rms = np.sqrt(np.mean(audio[mask] ** 2) + eps)
            gain = target_rms / rms
        else:
            gain = 1.0
        audio2 = audio * gain
        peak = np.abs(audio2).max()
        if peak > headroom:
            gain_db = 20 * np.log10(gain)
            peak_db = 20 * np.log10(peak)
            crest_db = peak_db - rms_norm_db
            safe_max_db = -0.5 - crest_db
            audio2 = audio2 / peak * headroom
            return audio2.astype(np.float32), dict(gain_db=gain_db, peak_db=peak_db, crest_db=crest_db, safe_max_db=safe_max_db)
        return audio2.astype(np.float32), None

    elif mode == "post_peak_rvc":
        a_max = np.abs(audio).max()
        if a_max <= 0:
            return audio.astype(np.float32), None
        return ((audio / a_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio).astype(np.float32), None

    elif mode == "post_peak":
        peak = np.max(np.abs(audio))
        if peak > 0:
            return (audio / peak * 0.95).astype(np.float32), None
        return audio.astype(np.float32), None

    return audio.astype(np.float32), None


def _process_and_save_worker(args):
    file_name, gt_wavs_dir, wavs16k_dir, mode, rms_norm_db = args
    try:
        stem, ext = file_name.split(".")[0], file_name.split(".")[1]

        gt_audio, gt_sr = sf.read(os.path.join(gt_wavs_dir, file_name))
        gt_result, gt_s = _apply_post_norm(gt_audio, gt_sr, mode, rms_norm_db)
        save_audio(gt_wavs_dir, stem, gt_sr, ext, gt_result)

        k16_audio, k16_sr = sf.read(os.path.join(wavs16k_dir, file_name))
        k16_result = _apply_post_norm_from_gain(k16_audio, gt_audio, mode, rms_norm_db)
        save_audio(wavs16k_dir, stem, k16_sr, ext, k16_result)
    except Exception as e:
        logger.error(f"Error normalizing {file_name} ({mode}): {e}")
        raise e
    return gt_s

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def format_duration_human(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} Hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} Min{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} Sec{'s' if seconds != 1 else ''}")
    return ", ".join(parts)

def save_dataset_duration(file_path, dataset_duration, normalization_mode, rms_norm_db):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": round(dataset_duration, 2),
    }
    if normalization_mode in ("post_rms", "post_peak_rvc", "post_peak"):
        new_data["normalization_method"] = normalization_mode
        if normalization_mode == "post_rms":
            new_data["normalization_rms_db"] = rms_norm_db
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def pool_size(requested, work_items):
    """Workers to actually start: never more than there are items to hand out.

    Every worker is a fresh process that re-imports this module, because
    ``multiprocessing`` spawns rather than forks on Windows.  An idle worker is
    therefore not free -- it is a second or so of interpreter startup bought
    for nothing -- which is how raising the thread setting used to make a stage
    slower instead of faster.
    """
    return max(1, min(int(requested), max(1, int(work_items))))


def _duration_seconds(audio_path, loading_resampling):
    """Length of one file in seconds, read from its header where possible.

    This runs serially in the parent for every file in the dataset before any
    work starts, purely to print the total, so it has to be a header read and
    not a decode.  It was neither: ``librosa.get_duration(path=...)`` decoded
    the file (1.87 s for 30 files against 0.01 s here, and it grows with the
    dataset), and the ffmpeg branch spawned an ``ffprobe`` process per file.

    Both remain as fallbacks -- ``soundfile`` cannot open every container the
    loaders accept -- but they are now the exception rather than the path every
    file takes.  A file whose length cannot be read at all contributes nothing:
    this total is a log line, and failing the run over it would be absurd.
    """
    try:
        info = sf.info(audio_path)
        if info.samplerate:
            return info.frames / info.samplerate
    except Exception:
        pass

    try:
        if loading_resampling == "librosa":
            return librosa.get_duration(path=audio_path)
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def cleanup_dirs(exp_dir):
    gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
    wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
    removed = []
    for directory in (gt_wavs_dir, wavs16k_dir):
        if os.path.exists(directory):
            shutil.rmtree(directory)
            removed.append(os.path.basename(directory))
    if removed:
        logger.info(f"Discarded previously sliced audio: {', '.join(removed)}.")


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    loading_resampling: str,
    dataset_format: str,
    rms_norm_db: float = -18.0
):
    start_time = time.time()

    if cut_preprocess == "New Automatic":
        # Checked here rather than in the worker: the pool would otherwise
        # raise the same error once per file, after the run had already
        # written part of a dataset with nothing usable in it.
        from rvc.train.preprocess import vad

        reason = vad.unavailable_reason()
        if reason:
            raise RuntimeError(f"'New Automatic' cutting is unavailable: {reason}")
        # Announced from here rather than from the workers: each of them builds
        # its own engine, so saying it there would repeat the line once per
        # worker.  This is the intent; a worker that cannot get the GPU says so
        # itself when it falls back.
        logger.info(f"Cutting with FireRedVAD on {vad.preferred_device()}.")

    speaker_map = {}

    root_files = [f for f in os.listdir(input_root) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
    if root_files:
        speaker_map[input_root] = [os.path.join(input_root, f) for f in root_files]

    for root, dirs, filenames in os.walk(input_root):
        if root == input_root:
            continue

        audio_files = [os.path.join(root, f) for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
        if audio_files:
            speaker_map[root] = audio_files

    speaker_count = len(speaker_map)

    if speaker_count > 1:
        detected_sids = set()
        for folder_path in speaker_map.keys():
            if folder_path == input_root:
                detected_sids.add(0)
            else:
                try:
                    folder_name = os.path.basename(folder_path)
                    sid = int(folder_name.split('_')[0])
                    detected_sids.add(sid)
                except (ValueError, IndexError):
                    logger.error(f"Folder '{folder_name}' is invalid for multi-speaker. "
                                 f"Folders must start with an integer (e.g., '0_name').")
                    sys.exit(1)

        expected_sids = set(range(speaker_count))
        if detected_sids != expected_sids:
            missing = sorted(list(expected_sids - detected_sids))
            logger.error(f"Speaker IDs are not contiguous or missing 0. "
                         f"Detected: {sorted(list(detected_sids))}. Missing: {missing}")
            sys.exit(1)
        else:
            logger.info(f"Speaker IDs 0-{speaker_count - 1} are contiguous.")

    total_dataset_duration = 0
    for audio_paths in speaker_map.values():
        for audio_path in audio_paths:
            total_dataset_duration += _duration_seconds(audio_path, loading_resampling)

    logger.info(f"Total dataset length: {format_duration_human(total_dataset_duration)}")
    logger.info(f"Total speakers count: {speaker_count} Speaker{'s' if speaker_count != 1 else ''}")
    logger.info(f"Normalization mode: {normalization_mode}")
    logger.info(f"Preprocessing start: {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}")

    cleanup_dirs(exp_dir)

    total_audio_length = 0

    logger.info("Stage 1: Slicing & Resampling")
    # Never more workers than there is work.  Every worker is a fresh process
    # that re-imports this module (Windows spawns rather than forks), so an
    # idle one is not free -- it is a second or so of startup bought for
    # nothing, and a three-file dataset was paying for twelve of them.
    stage1_workers = pool_size(num_processes, max(len(p) for p in speaker_map.values()))
    with multiprocessing.Pool(processes=stage1_workers) as pool:
        for speaker_dir, audio_paths in track(
            speaker_map.items(),
            total=len(speaker_map),
            description="Processing Speakers",
        ):

            try:
                if speaker_dir == input_root:
                    sid = 0
                else:
                    folder_name = os.path.basename(speaker_dir)
                    sid_str = folder_name.split('_')[0] 
                    sid = int(sid_str)
            except (ValueError, IndexError):
                logger.warning(f"Folder '{os.path.basename(speaker_dir)}' does not start with a valid integer ID. Using SID 0.")
                sid = 0

            current_batch_paths = audio_paths

            arg_list = [
                (
                    f_path,
                    idx,
                    sid,
                    sr,
                    exp_dir,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    chunk_len,
                    overlap_len,
                    loading_resampling,
                    dataset_format,
                    normalization_mode,
                )
                for idx, f_path in enumerate(current_batch_paths)
            ]

            for result in pool.imap_unordered(_process_audio_worker, arg_list):
                if result:
                    total_audio_length += result

    POST_NORM_MODES = {
        "post_rms":      "RMS Normalization",
        "post_peak_rvc": "Peak Normalization (RVC)",
        "post_peak":     "Peak Normalization",
    }

    if normalization_mode in POST_NORM_MODES:
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        audio_files = sorted(f for f in os.listdir(gt_wavs_dir) if f.endswith((".wav", ".flac")))

        logger.info("Stage 2: Normalization")

        if normalization_mode == "post_rms":
            logger.info("Performing a dry-run first to establish safety of chosen RMS dB...")
            is_safe, worst_safe, summary = _dry_run_post_rms(
                gt_wavs_dir, wavs16k_dir, audio_files, rms_norm_db, num_processes
            )
            if not is_safe:
                logger.warning(
                    f"Post RMS norm: {rms_norm_db:.1f} dBFS would clip {summary['num_limited']} files "
                    f"(avg crest {summary['avg_crest']:.0f} dB, peak {summary['avg_peak']:.1f} dBFS). "
                    f"Auto-adjusting to {worst_safe:.1f} dBFS."
                )
                rms_norm_db = worst_safe

        logger.info(f"Post Normalization: {POST_NORM_MODES[normalization_mode]}. Initiating...")
        arg_list = [(f, gt_wavs_dir, wavs16k_dir, normalization_mode, rms_norm_db) for f in audio_files]

        with multiprocessing.Pool(processes=pool_size(num_processes, len(audio_files))) as pool:
            with progress_task(
                len(audio_files),
                POST_NORM_MODES[normalization_mode],
            ) as (progress, task_id):
                for _ in pool.imap_unordered(_process_and_save_worker, arg_list):
                    progress.advance(task_id)

    save_dataset_duration(os.path.join(exp_dir, "model_info.json"), total_audio_length, normalization_mode, rms_norm_db)

    elapsed_time = time.time() - start_time
    success(
        f"Finished at {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')} "
        f"in {elapsed_time:.2f}s on {format_duration(total_audio_length)} of audio.",
        tag="[PREPROCESS]",
    )

if __name__ == "__main__":
    configure_logging(tag="[PREPROCESS]")
    if len(sys.argv) < 14:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len> <normalization_mode> <loading_resampling> <dataset_format> [rms_norm_db]")
        sys.exit(1)
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]

    if num_processes.lower() == "none":
        num_processes = DEFAULT_CPU_THREADS
    else:
        num_processes = int(num_processes)

    cut_preprocess = str(sys.argv[5])
    process_effects = bool(strtobool(sys.argv[6]))
    noise_reduction = bool(strtobool(sys.argv[7]))
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    loading_resampling = str(sys.argv[12])
    dataset_format = str(sys.argv[13])
    rms_norm_db = float(sys.argv[14]) if len(sys.argv) >= 15 else -18.0

    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        dataset_format,
        rms_norm_db,
    )
