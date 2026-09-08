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
import math
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
from rvc.train.preprocess.loudness import (
    apply_gain_with_ceiling,
    block_powers,
    CEILING_DB,
    integrated_lufs,
    limit_peaks,
    loudness_from_blocks,
    loudness_gain,
)

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
        # a 48 Hz cutoff normalises to 0.003 at 32 kHz and 0.002 at 48 kHz,
        # pushing the poles to |p| = 0.993-0.996 and making `ba` ill-conditioned. It
        # still holds up in float64, but `sos` is the form scipy recommends here
        # and it lets the filter run natively in float32.
        self.hp_sos = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr, output="sos"
        )
        # scipy's default padlen for this filter is 20 samples -- 0.6 ms, against
        # a 3.3 ms time constant -- which leaves a visible transient at the file
        # edges.  Six cycles of the cutoff is where the improvement saturates
        # (measured: 32 ms already gets within 0.2 dB of 3 s), and expressing it
        # in cycles keeps it correct at every project rate.
        self.hp_padlen = int(6 * self.sr / HIGH_PASS_CUTOFF)
        self.exp_dir = exp_dir

        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)


    def high_pass(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset and subsonic rumble, without touching the phase.

        Forward-backward rather than causal, for two reasons.

        The first is that ``rvc/infer/pipeline.py`` already filters this way --
        same Butterworth, same 48 Hz, applied with ``filtfilt``.  Running the
        training copy causally made the two disagree: the model learned from
        waveforms whose low end had been phase-rotated (167 degrees at 60 Hz,
        92 at 100, 45 at 200) against inference input that had not.  Whichever
        of the two is "right", they have to be the same one, and inference is
        not the side that can be changed without invalidating every model.

        The second is that the causal version was measurably destructive on
        material it should barely touch.  Over five vocal stems with a 2%%
        DC offset injected, error against the DC-free signal was -10.7 to
        -13.8 dB RMS causally and -72.9 to -83.8 dB this way, with not one
        sample worse anywhere in any file.  The material has 0.002%% of its
        energy below 150 Hz, so essentially all of that was phase.

        This also retires the old state-priming trick.  It existed because a
        zero-state IIR treats a file that begins mid-waveform as a step input
        and rings for ~45 ms at up to 80%% of the peak; ``filtfilt``'s
        odd-extension padding removes the transient instead of compensating for
        it, and does better at the start of the file too (peak error over the
        first millisecond: 1.4e-2 against 5.6e-2).
        """
        if audio.size == 0:
            return audio
        # filtfilt needs padlen < len(signal).  A clip too short to pad is too
        # short to have a meaningful sub-48 Hz component anyway -- one cycle of
        # the cutoff is 21 ms -- so it is passed through rather than filtered
        # with a padding it cannot support.
        padlen = min(self.hp_padlen, audio.size - 1)
        if padlen < 1:
            return audio
        filtered = signal.sosfiltfilt(self.hp_sos, audio, padlen=padlen)
        # sosfiltfilt promotes to float64; the rest of the pipeline is float32
        # and the extra precision is discarded on save anyway.
        return filtered.astype(np.float32, copy=False)

    def resample_16k(self, audio: np.ndarray, loading_resampling: str) -> np.ndarray:
        """The whole file at 16 kHz, resampled once.

        Once per file rather than once per slice, which is what this used to
        be.  On the ``ffmpeg`` path each slice cost a process spawn: 36 ms for
        a 3 s slice against 0.4 ms for the same resample through soxr,
        essentially all of it fork and exec.  At the ~7 slices a recording
        averages here that was the single largest cost in the stage -- more
        than the loader, the filter and the VAD together -- and it is why the
        GPU sat idle: it was waiting on ffmpeg, not on the network.

        Resampling first and cutting afterwards also removes the per-slice
        edge transient the old order introduced, since the filter no longer
        restarts at every slice boundary.  The 16 kHz offsets are computed
        from the rate ratio (:meth:`_slice_16k`), so the two copies stay
        sample-aligned to within the rounding of one 16 kHz sample.
        """
        if loading_resampling == "librosa":
            return librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        return load_audio_ffmpeg(
            audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
        )

    def _slice_16k(self, audio_16k: np.ndarray, start: int, end: int) -> np.ndarray:
        """``audio[start:end]`` mapped onto the 16 kHz copy.

        Clamped to what the resampler actually produced: its output length is
        the rate ratio give or take a sample, and the ratio is not an integer
        at every project rate.
        """
        total = len(audio_16k)
        ratio = SAMPLE_RATE_16K / self.sr
        begin = min(total, max(0, int(round(start * ratio))))
        finish = min(total, max(begin, int(round(end * ratio))))
        return audio_16k[begin:finish]

    def process_audio_segment(
        self,
        audio: np.ndarray,
        audio_16k: np.ndarray,
        start: int,
        end: int,
        sid: int,
        idx0: int,
        idx1: int,
        dataset_format: str
    ):
        name = f"{sid}_{idx0}_{idx1}"
        save_audio(self.gt_wavs_dir, name, self.sr, dataset_format, audio[start:end])
        save_audio(
            self.wavs16k_dir,
            name,
            SAMPLE_RATE_16K,
            dataset_format,
            self._slice_16k(audio_16k, start, end),
        )


    def simple_cut(
        self,
        audio: np.ndarray,
        audio_16k: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
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
                    end = total
                else:
                    # Whole file shorter than one chunk; variable-length
                    # slices are supported downstream (Automatic emits them
                    # routinely).
                    start, end = 0, total

            self.process_audio_segment(
                audio, audio_16k, start, end, sid, idx0, slice_idx, dataset_format
            )
            last_start = start
            slice_idx += 1

            if start + chunk_len_smpl >= total:
                break

    def chunk_segments(
        self,
        spans,
        audio: np.ndarray,
        audio_16k: np.ndarray,
        sid: int,
        idx0: int,
        dataset_format: str,
    ):
        """Cut each voiced segment into overlapping ``PERCENTAGE``-second slices.

        Shared by ``Automatic`` and ``New Automatic``: the two differ only in
        how the voiced segments were found, and the slice geometry downstream
        has to stay identical either way.  Segments arrive as ``(start, end)``
        offsets into ``audio`` -- an iterable, so a cutter can stay a generator
        -- rather than as arrays, because the same span has to be cut out of
        the 16 kHz copy as well.
        """
        idx1 = 0
        for segment_start, segment_end in spans:
            i = 0
            while True:
                start = segment_start + int(self.sr * (PERCENTAGE - OVERLAP) * i)
                i += 1
                if max(0, segment_end - start) > (PERCENTAGE + OVERLAP) * self.sr:
                    end = start + int(PERCENTAGE * self.sr)
                    self.process_audio_segment(
                        audio, audio_16k, start, end, sid, idx0, idx1, dataset_format
                    )
                    idx1 += 1
                else:
                    self.process_audio_segment(
                        audio, audio_16k, start, segment_end, sid, idx0, idx1,
                        dataset_format,
                    )
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

            # Once per file, before any cutting: every slice is cut out of
            # this copy instead of being resampled on its own.  See
            # ``resample_16k``.
            audio_16k = self.resample_16k(audio, loading_resampling)

            if cut_preprocess == "Skip":
                self.process_audio_segment(
                    audio, audio_16k, 0, len(audio), sid, idx0, 0, dataset_format
                )
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, audio_16k, sid, idx0, chunk_len, overlap_len, dataset_format)
            elif cut_preprocess == "Automatic":
                self.chunk_segments(
                    self.slicer.slice_spans(audio), audio, audio_16k, sid, idx0, dataset_format
                )
            elif cut_preprocess == "New Automatic":
                from rvc.train.preprocess import vad

                self.chunk_segments(
                    vad.segments(audio, self.sr), audio, audio_16k, sid, idx0, dataset_format,
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

    with worker_pool(pool_size(num_processes, len(audio_files))) as pool:
        with progress_task(len(arg_list), "Checking RMS target") as (progress, task_id):
            for result in pool.imap_unordered(_dry_run_check_file, arg_list):
                progress.advance(task_id)
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


def source_key(file_name: str) -> str:
    """The recording a slice came from.

    Slices are written as ``{sid}_{idx0}_{idx1}``: ``sid`` is the speaker and
    ``idx0`` enumerates that speaker's source files, so the first two fields
    together name one recording.  ``idx0`` restarts per speaker directory,
    which is why ``sid`` has to be part of the key.
    """

    stem = os.path.splitext(os.path.basename(file_name))[0]
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


#: How close to the target a recording has to land before the solver stops,
#: in dB, and how many times it may measure to get there.
GAIN_SOLVE_TOLERANCE_DB = 0.05
GAIN_SOLVE_MAX_EVALS = 8


def _solve_source_gain(audios, sample_rate, gain, target_lufs, ceiling_db):
    """The gain whose *limited* result measures ``target_lufs``.

    Ducking a peak removes energy that the loudness measurement had counted, so
    the gain that put a recording at the target before limiting leaves it under
    afterwards.  For most of a dataset that gap is nothing -- median 0.06 dB
    over 400 recordings here -- but it tracks how much of the energy lives in
    the transients, and on the tail it is real: whispers reached -1.9 dB and a
    recording of eating sounds -7.4 dB.

    Re-applying the shortfall as a gain does converge, but slowly, because the
    thing being solved is not slope 1: past the ceiling, some of every dB added
    goes straight back out through the limiter.  Measured slope is ~0.35, and
    eight naive rounds still left the worst case 0.37 dB short.  A secant on
    the measured slope gets there in a median of two measurements and left
    nothing above 0.1 dB.

    The slope is clamped to [0.05, 2]: it is a difference of two noisy
    measurements, and a near-zero one would otherwise throw the next guess
    somewhere absurd.  A genuinely flat slope is also how saturation shows up,
    and the floor of 0.05 is what makes the search take a big enough step to
    walk out onto the plateau and settle there.

    Returns ``(gain, achieved_lufs)`` -- the second so the caller can tell
    "solved" from "this is as loud as it goes".
    """

    def measure(value):
        pooled = [
            block_powers(limit_peaks(a * value, sample_rate, ceiling_db=ceiling_db),
                         sample_rate)
            for a in audios if a.size
        ]
        if not pooled:
            return -math.inf
        return loudness_from_blocks(np.concatenate(pooled))

    previous_db = 20.0 * math.log10(gain)
    previous = measure(gain)
    if not math.isfinite(previous):
        return gain, previous
    if abs(previous - target_lufs) <= GAIN_SOLVE_TOLERANCE_DB:
        return gain, previous

    # Second point from the naive step, which is the right first guess: below
    # the ceiling the slope really is 1, and those recordings finish here.
    current_db = previous_db + (target_lufs - previous)
    for _ in range(GAIN_SOLVE_MAX_EVALS - 1):
        current = measure(10.0 ** (current_db / 20.0))
        if not math.isfinite(current):
            current = previous
            break
        if abs(current - target_lufs) <= GAIN_SOLVE_TOLERANCE_DB:
            previous_db, previous = current_db, current
            break
        span = current_db - previous_db
        slope = (current - previous) / span if abs(span) > 1e-9 else 1.0
        slope = min(max(slope, 0.05), 2.0)
        previous_db, previous = current_db, current
        current_db = current_db + (target_lufs - current) / slope
    else:
        previous_db, previous = current_db, current

    return float(10.0 ** (previous_db / 20.0)), previous


def _apply_source_gain_worker(args):
    """Level one whole recording and write its slices back.

    A recording rather than a slice, for two reasons.  Its loudness is measured
    by pooling its slices' block powers and gating **once** -- gating per slice
    and averaging is a different (and wrong) number, because the relative gate
    is defined against the whole measurement.  And the solver measures that
    pooled loudness several times, which is only free if the slices are already
    in hand.

    Measuring here rather than in a pass of its own is what makes this mode one
    read of the dataset instead of two.  It used to need two: the target came
    down to whatever the *worst* recording could reach, so every recording had
    to be measured before any could be written.  Nothing is global any more --
    each recording goes to the target on its own -- so the only thing the
    separate pass still bought was printing the summary before the writes
    rather than after, and it cost a full pass over 43 GB to do it.

    The two copies get the same gain, which is what keeps the feature input and
    the ground truth at one level, and then each is limited at its own rate.
    Limiting the ground truth and resampling that to 16 kHz would be the purer
    construction, but it would replace the 16 kHz file the slicer already wrote
    with one that went through a second resample.  The two envelopes differ
    only where the limiter acts at all.

    Returns ``(slices_written, overshoot_db, shortfall_db)``.  ``overshoot_db``
    is how far past the ceiling the recording would have gone at the target, or
    ``None`` when the limiter had nothing to do.  ``shortfall_db`` is how far
    under the target it finished, or ``None`` when it got there: a recording
    can *saturate* below the target, because once the limiter has flattened it
    there is a loudness no further gain can pass.  Measured on one here --
    34.3 dB of crest, sparse transients over near-silence -- which tops out at
    -17.3 LUFS however hard it is driven.  Reported rather than swallowed, on
    the same principle as everything else in this mode: a target the material
    cannot reach should be a message.
    """
    key, file_names, gt_wavs_dir, wavs16k_dir, ceiling_db, target_lufs = args
    try:
        gt_audio, gt_rate = [], None
        for file_name in file_names:
            audio, rate = sf.read(os.path.join(gt_wavs_dir, file_name))
            gt_audio.append(audio)
            gt_rate = rate

        gain, overshoot, shortfall = 1.0, None, None
        pooled = [block_powers(a, gt_rate) for a in gt_audio if a.size]
        measured = (
            loudness_from_blocks(np.concatenate(pooled)) if pooled else -math.inf
        )
        if math.isfinite(measured) and math.isfinite(target_lufs):
            gain = 10.0 ** ((target_lufs - measured) / 20.0)
            peak = max((float(np.abs(a).max()) for a in gt_audio if a.size), default=0.0)
            ceiling = 10.0 ** (ceiling_db / 20.0)
            if peak * gain > ceiling:
                overshoot = 20.0 * math.log10(peak * gain / ceiling)
            gain, achieved = _solve_source_gain(
                gt_audio, gt_rate, gain, target_lufs, ceiling_db
            )
            if math.isfinite(achieved) and target_lufs - achieved > 0.1:
                shortfall = target_lufs - achieved
        # Otherwise: no measurable loudness -- silence, or shorter than the
        # gate can see.  Left alone rather than given an invented gain.

        for file_name, audio in zip(file_names, gt_audio):
            stem, ext = file_name.split(".")[0], file_name.split(".")[1]
            limited = limit_peaks(audio * gain, gt_rate, ceiling_db=ceiling_db)
            save_audio(gt_wavs_dir, stem, gt_rate, ext, limited.astype(np.float32))
            k16_audio, k16_rate = sf.read(os.path.join(wavs16k_dir, file_name))
            k16_limited = limit_peaks(
                k16_audio * gain, k16_rate, ceiling_db=ceiling_db
            )
            save_audio(
                wavs16k_dir, stem, k16_rate, ext, k16_limited.astype(np.float32)
            )
        return len(file_names), overshoot, shortfall
    except Exception as e:
        logger.error(f"Error normalizing recording {key} (pre_loudness): {e}")
        raise e


def _apply_source_peak_rvc_worker(args):
    """Apply the RVC peak blend to one whole recording and write its slices back.

    The formula is stock RVC's::

        audio / peak * (MAX_AMPLITUDE * ALPHA) + (1 - ALPHA) * audio

    -- a blend rather than a normalisation: it moves the recording ``ALPHA`` of
    the way toward ``MAX_AMPLITUDE`` and keeps the rest of the original level,
    so a quiet take stays quieter than a loud one.

    ``peak`` is pooled over the recording's slices, not taken per slice, which
    is the whole point of the ``pre_`` prefix here: RVC applies this to the
    source file before it is ever cut, and levelling each slice on its own peak
    instead flattens the dynamics between phrases.  ``post_peak_rvc`` is that
    per-slice variant, kept for configs that already name it.

    Both copies get the same scale factor, so the 16 kHz feature input and the
    ground truth stay at one level -- the same contract as
    ``_apply_source_gain_worker``.

    No limiter runs here, and unlike ``pre_loudness`` this mode has no ceiling:
    the blend puts the output peak at ``MAX_AMPLITUDE * ALPHA + (1 - ALPHA) *
    peak``, which is under ``MAX_AMPLITUDE`` only while the input peak is, and
    reaches 1.0 at an input peak of 1.3.  That is stock RVC's arithmetic and is
    left alone deliberately -- reproducing upstream is the reason this mode
    exists -- but it means a float source already at or above 0 dBFS comes out
    above the ceiling, and one above 1.3 comes out clipped.  Use ``pre_loudness``
    if the level has to be guaranteed.

    Recordings whose peak is 0 or above 2.5 are written back untouched, the
    threshold ``_normalize_audio`` uses to reject a file.  It returns ``None``
    there, which drops the audio; dropping a whole recording is worse than
    leaving it at its original level, so this passes it through instead.

    Returns the number of slices written.
    """
    key, file_names, gt_wavs_dir, wavs16k_dir = args
    try:
        gt_audio, gt_rate = [], None
        for file_name in file_names:
            audio, rate = sf.read(os.path.join(gt_wavs_dir, file_name))
            gt_audio.append(audio)
            gt_rate = rate

        peak = max((float(np.abs(a).max()) for a in gt_audio if a.size), default=0.0)
        # A silent or unreadably loud recording is left alone rather than given
        # an invented gain, matching ``_normalize_audio``'s own guard.
        scale = 0.0
        if peak > 0.0 and peak <= 2.5:
            scale = MAX_AMPLITUDE * ALPHA / peak

        for file_name, audio in zip(file_names, gt_audio):
            stem, ext = file_name.split(".")[0], file_name.split(".")[1]
            if scale:
                audio = audio * scale + (1 - ALPHA) * audio
            save_audio(gt_wavs_dir, stem, gt_rate, ext, audio.astype(np.float32))
            k16_audio, k16_rate = sf.read(os.path.join(wavs16k_dir, file_name))
            if scale:
                k16_audio = k16_audio * scale + (1 - ALPHA) * k16_audio
            save_audio(
                wavs16k_dir, stem, k16_rate, ext, k16_audio.astype(np.float32)
            )
        return len(file_names)
    except Exception as e:
        logger.error(f"Error normalizing recording {key} (pre_peak_rvc): {e}")
        raise e


def _dry_run_loudness_file(args):
    file_name, gt_wavs_dir, target_lufs, ceiling_db = args
    try:
        audio, sr = sf.read(os.path.join(gt_wavs_dir, file_name))
    except Exception:
        return None
    measured = integrated_lufs(audio, sr)
    if not math.isfinite(measured):
        return None
    peak = float(np.abs(audio).max())
    if peak <= 0.0:
        return None
    peak_db = 20.0 * math.log10(peak)
    # Crest above the *loudness*, not above the RMS: it is what the ceiling
    # costs when the file is pushed to the target.
    crest_db = peak_db - measured
    return dict(crest_db=crest_db, safe_max_db=ceiling_db - crest_db,
                peak_db=peak_db, gain_db=target_lufs - measured)


def _dry_run_slice_loudness(gt_wavs_dir, audio_files, target_lufs, num_processes,
                           ceiling_db=CEILING_DB):
    """What ``post_loudness`` would do, without writing anything.

    Same contract as the RMS dry run: ``(is_safe, worst_safe_db, summary)``, so
    a target the dataset cannot reach becomes a message and an auto-adjustment
    rather than a set of files quietly short of it.
    """

    arg_list = [(f, gt_wavs_dir, target_lufs, ceiling_db) for f in audio_files]
    crests, peaks, gains, safes = [], [], [], []
    # A full pass over the slices, printing nothing until it is done, so it
    # gets a bar for the same reason the write passes do.
    with worker_pool(pool_size(num_processes, len(audio_files))) as pool:
        with progress_task(len(arg_list), "Checking loudness target") as (progress, task_id):
            for result in pool.imap_unordered(_dry_run_loudness_file, arg_list):
                progress.advance(task_id)
                if result and result["safe_max_db"] < target_lufs:
                    crests.append(result["crest_db"])
                    peaks.append(result["peak_db"])
                    gains.append(result["gain_db"])
                    safes.append(result["safe_max_db"])
    if safes:
        return False, min(safes), {
            "num_limited": len(safes),
            "avg_gain": float(np.mean(gains)),
            "avg_peak": float(np.mean(peaks)),
            "avg_crest": float(np.mean(crests)),
        }
    return True, None, None


def _apply_post_norm_from_gain(audio: np.ndarray, gt_audio: np.ndarray, mode: str, rms_norm_db: float, gt_sample_rate: int = 40000):
    """Apply normalization using the gain computed from gt_audio, so gt and 16k stay loudness-consistent."""
    if mode == "post_loudness":
        # The gain is measured on ``gt_audio`` and applied to both copies, so
        # the 16 kHz feature input and the ground truth stay at the same level.
        # Its own rate is what the K-weighting has to be built at, hence
        # ``gt_sample_rate`` rather than the 16 kHz one.
        gain = loudness_gain(gt_audio, gt_sample_rate, rms_norm_db)
        scaled, _ = apply_gain_with_ceiling(audio, gain)
        return scaled.astype(np.float32)

    elif mode == "post_rms":
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
    if mode == "post_loudness":
        measured = integrated_lufs(audio, sr)
        gain = loudness_gain(audio, sr, rms_norm_db)
        scaled, limited_db = apply_gain_with_ceiling(audio, gain)
        if limited_db > 0.0:
            return scaled.astype(np.float32), dict(
                gain_db=20 * np.log10(gain),
                peak_db=20 * np.log10(max(np.abs(audio * gain).max(), 1e-12)),
                crest_db=limited_db,
                safe_max_db=rms_norm_db - limited_db,
            )
        return scaled.astype(np.float32), None

    elif mode == "post_rms":
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


def _stage1_pool(workers, use_threads):
    """The stage-1 pool, as processes or as threads.

    ``multiprocessing.pool.ThreadPool`` rather than
    ``concurrent.futures.ThreadPoolExecutor`` so the call site keeps working
    unchanged: it is the same class hierarchy as ``multiprocessing.Pool`` and
    has the same ``imap_unordered``, whereas the futures executors do not.

    On the thread path the VAD's engine and batcher live in this process, so
    they are started before any work is handed out and stopped after -- the
    batcher owns a daemon thread and a CUDA stream that should not outlive the
    stage.
    """
    if not use_threads:
        return worker_pool(workers)

    from multiprocessing.pool import ThreadPool

    from rvc.train.preprocess import vad

    vad.start_threaded()

    class _StoppingThreadPool(ThreadPool):
        def __exit__(self, *exc):
            try:
                return super().__exit__(*exc)
            finally:
                vad.stop_threaded()

    return _StoppingThreadPool(processes=workers)


def _process_and_save_worker(args):
    file_name, gt_wavs_dir, wavs16k_dir, mode, rms_norm_db = args
    try:
        stem, ext = file_name.split(".")[0], file_name.split(".")[1]

        gt_audio, gt_sr = sf.read(os.path.join(gt_wavs_dir, file_name))
        gt_result, gt_s = _apply_post_norm(gt_audio, gt_sr, mode, rms_norm_db)
        save_audio(gt_wavs_dir, stem, gt_sr, ext, gt_result)

        k16_audio, k16_sr = sf.read(os.path.join(wavs16k_dir, file_name))
        k16_result = _apply_post_norm_from_gain(
            k16_audio, gt_audio, mode, rms_norm_db, gt_sample_rate=gt_sr
        )
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
    if normalization_mode in ("pre_loudness", "pre_peak_rvc", "post_loudness",
                              "post_rms", "post_peak_rvc", "post_peak"):
        new_data["normalization_method"] = normalization_mode
        if normalization_mode in ("pre_loudness", "post_loudness"):
            new_data["normalization_target_lufs"] = rms_norm_db
        elif normalization_mode == "post_rms":
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


def worker_pool(processes):
    """A process pool that does not inherit this process's threads.

    Stage 1 runs the VAD in this process when it is on the GPU, so by the time
    the later stages start, the parent holds a CUDA context as well as torch's
    and OpenMP's thread pools.  ``fork`` -- the default on Linux -- copies only
    the calling thread, and every mutex the other threads happened to hold is
    copied locked with nobody left to release it: the children come up and
    block in ``futex_wait`` before running a single task, which is a
    normalisation stage sitting at 0 of 24k recordings forever.  ``spawn``
    starts a clean interpreter instead.  The workers already have to survive
    that -- it is what Windows does -- so it costs only the re-import, which
    ``pool_size`` is already written around.
    """

    return multiprocessing.get_context("spawn").Pool(processes=processes)


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
    rms_norm_db: float = -16.0
):
    start_time = time.time()

    vad_device = "cpu"
    if cut_preprocess == "New Automatic":
        # Checked here rather than in the worker: the pool would otherwise
        # raise the same error once per file, after the run had already
        # written part of a dataset with nothing usable in it.
        from rvc.train.preprocess import vad

        reason = vad.unavailable_reason()
        if reason:
            raise RuntimeError(f"'New Automatic' cutting is unavailable: {reason}")
        # Decided once, here, and pinned for the workers.  ``plan_device``
        # touches CUDA, which is why it may only run on the branch that then
        # avoids forking: see below.
        vad_device = vad.plan_device()
        logger.info(f"Cutting with FireRedVAD on {vad_device.upper()}.")

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
    # Threads rather than processes when the VAD is on the GPU, which is the
    # whole point of deciding the device up front: N worker processes means N
    # CUDA contexts of a few hundred MB each for a 2.3 MB model, and on a card
    # already holding a training run most of them lose that race.  One process
    # of N threads shares one context and one batcher.  The GIL is not the
    # constraint it looks like here -- soundfile, librosa, numpy and torch all
    # release it for the work that dominates this stage.
    #
    # Everywhere else keeps the process pool: without the GPU there is nothing
    # to share, and processes still beat threads on the pure-CPU path.
    use_threads = cut_preprocess == "New Automatic" and vad_device == "cuda"
    with _stage1_pool(stage1_workers, use_threads) as pool:
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

    #: Modes whose gain is solved once per source recording.  They write the
    #: slices back themselves, in their own branch below, so the generic
    #: per-slice pass has to skip them -- it would otherwise re-read and
    #: re-write the whole dataset to apply nothing.
    RECORDING_SCOPE_MODES = ("pre_loudness", "pre_peak_rvc")

    POST_NORM_MODES = {
        "pre_loudness": "Loudness Normalization (BS.1770, per recording)",
        "post_loudness": "Loudness Normalization (BS.1770, per slice)",
        "post_rms":      "RMS Normalization",
        "pre_peak_rvc":  "Peak Normalization (RVC, per recording)",
        "post_peak_rvc": "Peak Normalization (RVC, per slice)",
        "post_peak":     "Peak Normalization",
    }

    if normalization_mode in POST_NORM_MODES:
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        audio_files = sorted(f for f in os.listdir(gt_wavs_dir) if f.endswith((".wav", ".flac")))

        logger.info("Stage 2: Normalization")

        if normalization_mode == "pre_peak_rvc":
            # Grouped by recording for the same reason as ``pre_loudness``:
            # the peak the blend is built on belongs to the recording, not to
            # whichever slice happens to hold its loudest sample.
            by_source = {}
            for f in audio_files:
                by_source.setdefault(source_key(f), []).append(f)
            arg_list = [
                (key, sorted(names), gt_wavs_dir, wavs16k_dir)
                for key, names in sorted(by_source.items())
            ]
            logger.info(
                f"Post Normalization: {POST_NORM_MODES[normalization_mode]} "
                f"({len(arg_list)} recordings). Initiating..."
            )
            with worker_pool(pool_size(num_processes, len(arg_list))) as pool:
                with progress_task(
                    len(audio_files), POST_NORM_MODES[normalization_mode]
                ) as (progress, task_id):
                    for written in pool.imap_unordered(
                        _apply_source_peak_rvc_worker, arg_list
                    ):
                        progress.advance(task_id, written)

        elif normalization_mode == "pre_loudness":
            # Grouped by recording: that is the unit the loudness is pooled
            # over, the unit the gain is solved for, and the unit one gain is
            # shared by.  One pass -- measuring used to be a pass of its own
            # and no longer needs to be; see ``_apply_source_gain_worker``.
            by_source = {}
            for f in audio_files:
                by_source.setdefault(source_key(f), []).append(f)
            arg_list = [
                (key, sorted(names), gt_wavs_dir, wavs16k_dir, CEILING_DB, rms_norm_db)
                for key, names in sorted(by_source.items())
            ]
            logger.info(
                f"Post Normalization: {POST_NORM_MODES[normalization_mode]} "
                f"({len(arg_list)} recordings). Initiating..."
            )
            overshoots, shortfalls = [], []
            with worker_pool(pool_size(num_processes, len(arg_list))) as pool:
                with progress_task(
                    len(audio_files), POST_NORM_MODES[normalization_mode]
                ) as (progress, task_id):
                    for written, overshoot, shortfall in pool.imap_unordered(
                        _apply_source_gain_worker, arg_list
                    ):
                        progress.advance(task_id, written)
                        if overshoot is not None:
                            overshoots.append(overshoot)
                        if shortfall is not None:
                            shortfalls.append(shortfall)
            # Reported after rather than before, which is what folding the
            # measurement into the write pass costs.  Nothing here is a
            # decision the target depends on any more -- it is the record of
            # how much work the limiter did.
            if overshoots:
                logger.info(
                    f"Loudness norm: every recording reached {rms_norm_db:.1f} LUFS. "
                    f"{len(overshoots)} of {len(arg_list)} peaked above the ceiling "
                    f"there and had those peaks ducked "
                    f"(mean {np.mean(overshoots):.1f} dB, "
                    f"max {max(overshoots):.1f} dB)."
                )
            if shortfalls:
                logger.warning(
                    f"Loudness norm: {len(shortfalls)} of {len(arg_list)} recordings "
                    f"top out below {rms_norm_db:.1f} LUFS and finish there "
                    f"(worst {max(shortfalls):.1f} dB short). Their peaks are so far "
                    f"above their loudness that limiting flattens them before the "
                    f"target is reached; no gain gets them further."
                )

        elif normalization_mode == "post_loudness":
            logger.info("Dry run: checking the target is reachable without limiting...")
            is_safe, worst_safe, summary = _dry_run_slice_loudness(
                gt_wavs_dir, audio_files, rms_norm_db, num_processes
            )
            if not is_safe:
                logger.warning(
                    f"Loudness norm: {rms_norm_db:.1f} LUFS would be limited on "
                    f"{summary['num_limited']} files (avg crest {summary['avg_crest']:.0f} LU, "
                    f"peak {summary['avg_peak']:.1f} dBFS). "
                    f"Auto-adjusting to {worst_safe:.1f} LUFS."
                )
                rms_norm_db = worst_safe

        elif normalization_mode == "post_rms":
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

        if normalization_mode not in RECORDING_SCOPE_MODES:
            logger.info(f"Post Normalization: {POST_NORM_MODES[normalization_mode]}. Initiating...")
            arg_list = [(f, gt_wavs_dir, wavs16k_dir, normalization_mode, rms_norm_db) for f in audio_files]

            with worker_pool(pool_size(num_processes, len(audio_files))) as pool:
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
