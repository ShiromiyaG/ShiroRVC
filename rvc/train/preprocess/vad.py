"""FireRedVAD-backed speech segmentation for the "New Automatic" cutter.

The stock ``Automatic`` cutter is :mod:`rvc.train.preprocess.slicer`, which
finds silence by RMS energy against a fixed dBFS threshold.  That works on a
clean studio take and falls apart everywhere else: breath, room tone and a
noise floor above the threshold all read as speech, and a quiet held note reads
as silence and gets cut in half.

FireRedVAD (https://github.com/FireRedTeam/FireRedVAD, Apache-2.0) is a
588k-parameter DFSMN classifier that decides voiced/unvoiced from the audio
itself rather than from its level, so neither of those failure modes applies.
It is small enough -- 2.3 MB of weights, CPU inference -- to run inside the
preprocessing worker pool without a budget worth discussing.

Two details of its contract are easy to get wrong and are handled here rather
than at the call site:

* It is a 16 kHz model and *asserts* the rate rather than resampling, so the
  audio is resampled here and the segment times are mapped back to the caller's
  rate afterwards.
* Its feature front-end reads files with ``soundfile(dtype="int16")``, so the
  network was calibrated on int16-magnitude samples.  Handing it the float
  [-1, 1] audio the rest of this pipeline uses puts every frame ~90 dB below
  what it was trained on, and it then finds almost nothing: measured over three
  files here, 0-1% of the audio came back voiced against 91-96% once scaled.
  Nothing errors -- the dataset just comes out nearly empty.
"""

from __future__ import annotations

import logging
import os

import librosa
import numpy as np

# ``detect`` warns once per 300 s chunk on any long file, and the postprocessor
# warns about short segments it then handles itself.  Neither is actionable,
# and at one line per chunk they would bury the preprocessing log.
logging.getLogger("fireredvad").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

#: Where ``prerequisites_download`` puts the VAD checkpoint and its CMVN stats.
MODEL_DIR = os.path.join("rvc", "models", "fireredvad", "VAD")

#: The model's own rate; it asserts this rather than resampling.
VAD_SAMPLE_RATE = 16000

#: Resampler for the trip down to 16 kHz.  ``soxr_hq`` rather than the
#: ``soxr_vhq`` used for the slices that get written: this copy is thrown away
#: after the frame decisions come back, and 80-bin fbank features cannot see
#: the difference.
RES_TYPE = "soxr_hq"

#: One engine per process.  The worker pool builds a fresh ``PreProcess`` for
#: every file, so anything held on the instance would reload the model once per
#: file; a module global loads it once per worker instead.
_ENGINE = None

#: Which device the cached engine ended up on, so the fallback below can tell
#: "GPU never worked here" from "GPU worked and then stopped".
_DEVICE = None

#: Set once the GPU has failed in this process.  Without it a run whose GPU is
#: out of memory would retry it for every remaining file, paying the failed
#: allocation each time.
_GPU_REFUSED = False


def _config(use_gpu):
    """Detection thresholds, tuned for building a training set rather than for
    transcription.

    ``min_speech_frame`` and ``min_silence_frame`` are frames of 10 ms.  Both
    are shorter than the upstream defaults (20 frames): a sung phrase can turn
    around in well under 200 ms, and losing that transition is worse here than
    admitting a little extra silence, which the chunker crops anyway.

    ``extend_speech_frame`` pads 50 ms onto each end.  The network fires on the
    steady part of a phoneme and clips the onset transient; that transient is
    exactly what a vocoder has to learn to reproduce.
    """

    from fireredvad import FireRedVadConfig

    return FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=10,
        max_speech_frame=2000,
        min_silence_frame=10,
        merge_silence_frame=0,
        extend_speech_frame=5,
    )


def is_installed() -> bool:
    """Whether the Python package is importable, ignoring the weights."""
    try:
        import fireredvad  # noqa: F401
    except ImportError:
        return False
    return True


def has_weights() -> bool:
    return os.path.isfile(os.path.join(MODEL_DIR, "model.pth.tar")) and os.path.isfile(
        os.path.join(MODEL_DIR, "cmvn.ark")
    )


def unavailable_reason() -> str | None:
    """Why ``segments`` would fail, or ``None`` when it would work.

    Checked up front by the caller so a run that cannot use this cutter stops
    before it has written half a dataset with the wrong one.
    """
    if not is_installed():
        return (
            "the 'fireredvad' package is not installed. Install it with "
            "'pip install fireredvad', or pick another cutting mode."
        )
    if not has_weights():
        return (
            f"the VAD weights are missing from '{MODEL_DIR}'. Run the "
            f"prerequisites download, or pick another cutting mode."
        )
    return None


def _gpu_is_usable():
    """Whether this process should even try the GPU.

    ``torch`` is imported lazily: this module is reachable from the option
    check in the UI, which must not pay for a CUDA import to answer whether a
    dropdown entry works.
    """
    if _GPU_REFUSED:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _load(use_gpu):
    from fireredvad import FireRedVad

    return FireRedVad.from_pretrained(MODEL_DIR, _config(use_gpu))


def _engine():
    """The cached engine, on the GPU when one will have it.

    The GPU is roughly 2x on the network itself -- measured over 12 minutes of
    audio, 2.77 s to 1.36 s end to end, the remainder being the CPU-only fbank
    front-end.  It is attempted first and abandoned on any failure, because
    "no CUDA build", "no driver", "no free memory" and "another process has the
    device" all arrive as different exception types and none of them is worth
    stopping a preprocessing run over.

    Note that preprocessing runs a pool of worker processes, so this happens
    once per worker: N CUDA contexts of a few hundred MB each, for a 2.3 MB
    model.  That is the reason the fallback has to be real rather than
    decorative -- on a card already holding a training run, some workers will
    fail here and carry on over CPU.
    """
    global _ENGINE, _DEVICE, _GPU_REFUSED
    if _ENGINE is not None:
        return _ENGINE

    if _gpu_is_usable():
        try:
            _ENGINE = _load(use_gpu=True)
            _DEVICE = "cuda"
            return _ENGINE
        except Exception as error:
            _GPU_REFUSED = True
            logger.warning(f"VAD falling back to CPU: the GPU refused it ({error}).")

    _ENGINE = _load(use_gpu=False)
    _DEVICE = "cpu"
    return _ENGINE


def _detect(audio):
    """``engine.detect`` with a one-time demotion to CPU if the GPU gives out.

    Loading succeeding says nothing about inference succeeding: the model is
    tiny but its activations are not, and a long file arrives as one tensor.
    A card that was fine a moment ago can also be taken by something else
    mid-run.  Either way the file still has to be segmented, so the engine is
    rebuilt on CPU and the call retried once.
    """
    global _ENGINE, _DEVICE, _GPU_REFUSED
    try:
        return _engine().detect(audio)
    except Exception as error:
        if _DEVICE != "cuda":
            raise
        _GPU_REFUSED = True
        logger.warning(f"VAD falling back to CPU: the GPU failed mid-run ({error}).")
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        _ENGINE = _load(use_gpu=False)
        _DEVICE = "cpu"
        return _ENGINE.detect(audio)


def device() -> str | None:
    """Where the engine ended up, or ``None`` before one has been built."""
    return _DEVICE


def preferred_device() -> str:
    """Where an engine built now would go, without building one.

    For the caller to report before it forks the worker pool: probing this in
    the parent is cheap, whereas asking each worker after the fact would print
    the same line once per worker.
    """
    return "GPU" if _gpu_is_usable() else "CPU"


def segments(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Voiced spans of ``audio`` as ``(start, end)`` sample offsets at ``sr``.

    Returns one span covering the whole input when the model finds no speech,
    so a file the VAD does not understand is still cut into chunks rather than
    silently dropped from the dataset.
    """
    if audio.size == 0:
        return []

    if sr != VAD_SAMPLE_RATE:
        audio_16k = librosa.resample(
            audio, orig_sr=sr, target_sr=VAD_SAMPLE_RATE, res_type=RES_TYPE
        )
    else:
        audio_16k = audio

    # int16 magnitude, not int16 dtype: the front-end calls ``.tolist()`` on
    # whatever it gets and kaldi-native-fbank takes the numbers as they come.
    # Clipped rather than scaled to fit, so one loud sample cannot quiet the
    # whole file below the detection threshold.
    scaled = np.clip(audio_16k.astype(np.float32) * 32768.0, -32768.0, 32767.0)

    result, _probs = _detect(scaled)
    timestamps = result.get("timestamps") or []

    total = len(audio)
    spans = []
    for start_seconds, end_seconds in timestamps:
        start = max(0, int(round(start_seconds * sr)))
        end = min(total, int(round(end_seconds * sr)))
        if end > start:
            spans.append((start, end))

    if not spans:
        return [(0, total)]
    return spans
