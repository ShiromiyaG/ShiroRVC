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
import queue
import threading
from concurrent.futures import Future

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

#: Free VRAM a card has to have before this will take it.  The model is 2.3 MB
#: but its activations are not, and the interesting case is a card already
#: holding a training run: there, losing the race costs the *training* an OOM,
#: which is a far worse outcome than segmenting on CPU.  Matched to the
#: headroom upstream FireRedVAD wrappers ask for.
MIN_FREE_VRAM = 1024 ** 3

#: ``auto`` (default) takes the GPU when the checks below pass, ``cpu`` never
#: does, ``cuda`` demands it and raises rather than degrading.  The parent sets
#: it to ``cpu`` before forking once it has decided against the GPU, so workers
#: do not each rediscover that on their own.
DEVICE_ENV = "SHIRO_VAD_DEVICE"

#: The fbank front-end is stateful and not thread-safe, so each thread gets its
#: own; the network itself is shared and read-only under ``inference_mode``.
_THREAD_LOCAL = threading.local()

#: Serialises engine construction between threads.
_ENGINE_LOCK = threading.RLock()

#: The batcher, when one is running.  Only built for the GPU path.
_BATCHER = None

#: Whether this process is running the stage in threads.  Separate from the
#: device on purpose: it decides whether ``detect`` may be used as-is (one
#: caller, upstream's own code path) or has to be decomposed so the stateful
#: front-end can be per-thread.  A GPU run that is demoted to CPU mid-way stays
#: threaded, so the decomposition has to outlive the device that motivated it.
_THREADED = False


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

    ``device_count`` rather than ``is_available``: the latter probes through
    the CUDA runtime, which initializes a context in whatever process asks.
    The parent calls this (via :func:`preferred_device`) before it forks the
    worker pool, and a fork of a process that has touched CUDA can never
    initialize it again -- every worker would then hit "Cannot re-initialize
    CUDA in forked subprocess" and demote itself to CPU, i.e. the probe alone
    would cost the GPU it was asking about.  ``device_count`` answers from
    NVML instead and is documented not to poison the fork.
    """
    if _GPU_REFUSED:
        return False

    setting = os.environ.get(DEVICE_ENV, "auto").strip().lower()
    if setting == "cpu":
        return False
    if setting not in ("auto", "cuda"):
        raise ValueError(f"{DEVICE_ENV} must be 'auto', 'cpu' or 'cuda'.")
    try:
        import torch

        return torch.cuda.device_count() > 0
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

    Preprocessing used to run this inside a pool of worker *processes*, which
    meant N CUDA contexts of a few hundred MB each for a 2.3 MB model -- the
    reason the CPU fallback had to be real rather than decorative.  The GPU
    path now runs the stage in threads instead (see ``plan_device`` and the
    executor choice in ``preprocess.py``), so there is one context per run.
    The fallback still has to be real: the card can be busy, or taken mid-run.
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


def has_vram_headroom() -> bool:
    """Whether the card has :data:`MIN_FREE_VRAM` free *right now*.

    Unlike :func:`_gpu_is_usable` this **initializes CUDA in the calling
    process**, because ``mem_get_info`` has no NVML-only equivalent in torch.
    That makes it unsafe to call before a ``fork``, so it is deliberately not
    part of ``_gpu_is_usable``: only :func:`plan_device` calls it, and only on
    the path that then runs the VAD in *threads*, where there is no fork left
    to poison.

    Free memory is a snapshot and something else can take the card a moment
    later; that is what the OOM handling in :class:`_Batcher` is for.  This
    check is about not starting a fight it can avoid.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes >= MIN_FREE_VRAM
    except Exception:
        return False


def plan_device() -> str:
    """Decide GPU vs CPU once, in the parent, and record it for the workers.

    Returns ``"cuda"`` or ``"cpu"``.  On ``cpu`` it pins :data:`DEVICE_ENV` so
    that forked (or spawned) workers inherit the verdict instead of each paying
    the failed allocation to rediscover it -- which is what the old code did,
    once per worker.  ``SHIRO_VAD_DEVICE=cuda`` raises here rather than
    degrading, so a run that was told to use the GPU fails loudly.
    """
    setting = os.environ.get(DEVICE_ENV, "auto").strip().lower()
    if setting == "cpu":
        return "cpu"

    if _gpu_is_usable() and has_vram_headroom():
        return "cuda"

    if setting == "cuda":
        raise RuntimeError(
            f"{DEVICE_ENV}=cuda but no CUDA device has {MIN_FREE_VRAM // 1024 ** 2} MB free."
        )
    os.environ[DEVICE_ENV] = "cpu"
    return "cpu"


def _thread_audio_feat():
    """This thread's fbank front-end.

    ``AudioFeat`` wraps a kaldi-native-fbank computer, which carries state
    across calls and is not safe to share.  The postprocessor next to it holds
    nothing but its thresholds and *is* shared, and so is the network.  One
    front-end per thread costs a second copy of the 2.3 MB weights that come
    with it, which is the cheap half of the trade.
    """
    feat = getattr(_THREAD_LOCAL, "audio_feat", None)
    if feat is None:
        feat = _load(use_gpu=False).audio_feat
        _THREAD_LOCAL.audio_feat = feat
    return feat


class _Batcher:
    """Runs the network for every thread, one CUDA stream, with OOM backoff.

    Requests are grouped by frame count and only equal-length ones share a
    forward.  Padding a batch to its longest member is the obvious alternative
    and is **wrong here**: measured on this model, appending 10 zero frames to
    a 224-frame utterance moved 123 of its 224 outputs, by up to 1.7e-2, and
    100 frames moved 133 of them by up to 0.22.  The lookahead is not local, so
    a padded batch would silently return different speech boundaries than the
    same audio run alone -- a corrupted dataset with nothing to show for it.

    In practice that means batching fires on long files, which ``detect``
    splits into equal ``chunk_max_frame`` blocks, and rarely on short ones.
    The larger win is upstream of here: one CUDA context for the whole run
    instead of one per worker process, and the CPU fbank front-end running in
    parallel across threads.
    """

    def __init__(self, model, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self.requests = queue.Queue()
        self.failed = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, feats):
        future = Future()
        self.requests.put((feats, future))
        return future

    def _forward(self, group):
        import torch

        batch = torch.stack([feats for feats, _ in group]).cuda()
        probs, _ = self.model.forward(batch)
        return probs.cpu()

    def _run_group(self, group):
        """One equal-length group, halving on OOM until it fits or gives up."""
        import torch

        if len(group) > self.batch_size:
            for start in range(0, len(group), self.batch_size):
                self._run_group(group[start : start + self.batch_size])
            return
        try:
            probs = self._forward(group)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1:
                # One utterance does not fit: the card is gone for this run.
                # Reported to every waiter so the caller can demote to CPU.
                raise
            self.batch_size = max(1, len(group) // 2)
            logger.warning(
                f"VAD CUDA batch reduced to {self.batch_size} after OOM."
            )
            middle = len(group) // 2
            self._run_group(group[:middle])
            self._run_group(group[middle:])
            return
        # ``forward`` returns (B, T, 1); the postprocessor wants a flat list of
        # T probabilities, which is what upstream's ``.squeeze()`` produces.
        for index, (_, future) in enumerate(group):
            future.set_result(probs[index, :, 0].clone())

    def _run(self):
        import torch

        with torch.inference_mode():
            while True:
                first = self.requests.get()
                if first is None:
                    return
                pending = [first]
                # Drain whatever else is already waiting, so a burst from N
                # threads becomes one forward per length rather than N.
                while len(pending) < self.batch_size * 4:
                    try:
                        item = self.requests.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        self.requests.put(None)
                        break
                    pending.append(item)

                by_length = {}
                for feats, future in pending:
                    by_length.setdefault(feats.size(0), []).append((feats, future))
                for group in by_length.values():
                    try:
                        self._run_group(group)
                    except Exception as error:
                        self.failed = error
                        for _, future in group:
                            if not future.done():
                                future.set_exception(error)

    def stop(self):
        """Ask the thread to finish and wait for it.

        Joined rather than left to the daemon flag: a thread parked in
        ``queue.get`` when the interpreter tears down aborts noisily from the
        C++ side ("terminate called without an active exception"), which looks
        like a crash at the end of an otherwise clean run.
        """
        self.requests.put(None)
        self.thread.join(timeout=30)


def start_threaded():
    """Enter threaded mode in the process that will run the stage.

    Builds the shared engine, marks the module threaded so ``_detect`` takes
    the decomposed path, and starts a batcher when the engine is on the GPU.
    Returns the device, for the caller to report.
    """
    global _BATCHER, _THREADED
    with _ENGINE_LOCK:
        engine = _engine()
        _THREADED = True
        if _DEVICE == "cuda" and _BATCHER is None:
            _BATCHER = _Batcher(engine.vad_model)
        return _DEVICE


def stop_threaded():
    global _BATCHER, _THREADED
    with _ENGINE_LOCK:
        _THREADED = False
        if _BATCHER is not None:
            _BATCHER.stop()
            _BATCHER = None


def _demote_to_cpu(reason):
    """Rebuild the engine on CPU after the GPU has given out, once per run."""
    global _ENGINE, _DEVICE, _GPU_REFUSED, _BATCHER
    with _ENGINE_LOCK:
        if _DEVICE != "cuda":
            return
        _GPU_REFUSED = True
        logger.warning(f"VAD falling back to CPU: the GPU failed mid-run ({reason}).")
        if _BATCHER is not None:
            _BATCHER.stop()
            _BATCHER = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        _ENGINE = _load(use_gpu=False)
        _DEVICE = "cpu"


def _detect(audio):
    """Segment one clip, with a one-time demotion to CPU if the GPU gives out.

    Off the threaded path this is upstream's ``detect`` and nothing else: one
    caller, no reason to reimplement its chunking.  In threaded mode it is
    decomposed instead, so the stateful fbank front-end can be per-thread and
    the forward can go through the shared batcher.  The split at
    ``chunk_max_frame`` and the postprocessing mirror ``detect`` exactly; the
    parity is pinned by ``test_threaded_detect_matches_upstream``.
    """
    global _ENGINE, _DEVICE, _GPU_REFUSED
    if not _THREADED:
        try:
            return _engine().detect(audio)
        except Exception as error:
            if _DEVICE != "cuda":
                raise
            _demote_to_cpu(error)
            return _engine().detect(audio)

    engine = _engine()
    feats, duration = _thread_audio_feat().extract(audio)

    try:
        probs = _forward(engine, feats)
    except Exception as error:
        if _DEVICE != "cuda":
            raise
        _demote_to_cpu(error)
        engine = _engine()
        probs = _forward(engine, feats)

    decisions = engine.vad_postprocessor.process(probs.tolist())
    timestamps = engine.vad_postprocessor.decision_to_segment(decisions, duration)
    return {"dur": round(duration, 3), "timestamps": timestamps}, probs


def _forward(engine, feats):
    """Frame probabilities for ``feats``, split exactly as upstream splits."""
    import torch

    limit = engine.config.chunk_max_frame
    if feats.size(0) <= limit:
        return _forward_one(engine, feats)
    parts = [_forward_one(engine, chunk) for chunk in feats.split(limit, dim=0)]
    return torch.cat(parts, dim=0)


def _forward_one(engine, feats):
    """One forward, on whichever device the engine ended up on.

    Always returns CPU probabilities: the postprocessor calls ``.tolist()``.
    The un-batched GPU path exists for a single-threaded caller, which has no
    batcher to submit to.
    """
    import torch

    batcher = _BATCHER
    if batcher is not None and _DEVICE == "cuda":
        return batcher.submit(feats).result()
    with torch.inference_mode():
        if _DEVICE == "cuda":
            feats = feats.cuda()
        probs, _ = engine.vad_model.forward(feats.unsqueeze(0))
    return probs[0, :, 0].cpu()


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
