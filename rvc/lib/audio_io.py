"""Reading audio off disk, and nothing else.

Split out of :mod:`rvc.lib.utils` because of what importing that module costs.
It pulls in torch and transformers to serve the embedder helpers
that live alongside these loaders -- 4.6 s on this machine, measured with
``-X importtime`` -- and preprocessing pays it twice over: once in the parent,
and once more in *every* pool worker, because ``multiprocessing`` on Windows
spawns rather than forks and each child re-imports the module it came from.

That fixed cost was most of a preprocessing run and it scaled with the worker
count, which is why raising the thread setting made the stage slower instead of
faster.  Nothing here needs a tensor, so nothing here should import one.

``rvc.lib.utils`` re-exports all three names, so existing callers are
unaffected and either import path keeps working.
"""

from __future__ import annotations

import os

import ffmpeg
import librosa
import numpy as np
import soundfile as sf


def load_audio_16k(file):
    # Callers already preprocess to 16k, so no resample happens here.
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_ffmpeg(
    source: [str, np.ndarray],
    sample_rate: int = 48000,
    source_sr: int = None,
) -> np.ndarray:
    """Load (or resample, for an in-memory chunk) audio via ffmpeg, as float32."""
    if isinstance(source, str):
        source = source.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.exists(source):
            raise FileNotFoundError(f"The audio file was not found at the provided path: {source}")

        try:
            out, err = (
                ffmpeg.input(source, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio file '{source}':\n{e.stderr.decode()}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading audio: {e}") from e
    elif isinstance(source, np.ndarray):
        if source_sr is None:
            raise ValueError("source_sr must be provided when passing a NumPy array.")

        if source.dtype != np.float32:
            source = source.astype(np.float32)

        if source.ndim > 1:
            source = np.mean(source, axis=1)

        try:
            process = (
                ffmpeg
                .input('pipe:0', format='f32le', acodec='pcm_f32le', ar=source_sr, ac=1)
                .output('pipe:1', format='f32le', acodec='pcm_f32le', ar=sample_rate)
                .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
            )
            out, err = process.communicate(input=source.tobytes())
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to resample audio chunk:\n{e.stderr.decode()}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while processing audio chunk: {e}") from e
    else:
        raise ValueError("Invalid source type. Must be a file path (str) or a NumPy array (np.ndarray).")

    return np.frombuffer(out, np.float32).flatten()
