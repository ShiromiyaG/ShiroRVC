import os
import sys
import random
import gc
import re
import torch
import torch.nn.functional as F
import torchcrepe
import librosa
import numpy as np
from scipy import signal
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import (
    error as print_error,
    info,
    install_rich_print,
    warning,
)

install_rich_print()

from rvc.lib.predictors.f0 import CREPE, RMVPE, FCPE
from rvc.lib.utils import extract_features
from rvc.infer.retrieval import IndexRetriever, RetrievalConfig
from rvc.lib.terminal import get_console

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)


class AudioProcessor:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """

    @staticmethod
    def gate_to_source(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        threshold_db: float = -60.0,
        knee_db: float = 12.0,
        hold_ms: float = 120.0,
        release_ms: float = 80.0,
    ):
        """Silence the output wherever the *input* carries nothing.

        The content encoder has no absolute notion of level.  Its first conv
        layer is group-normalised over the whole chunk and its transformer
        attends across all of it, so digital silence does not come out as a
        "nothing" vector: it gets a full-magnitude embedding whose direction
        depends on whatever else is in the chunk.  Measured on contentvec, the
        same silence embeds at cosine 0.54 to itself depending on the company it
        keeps, at a feature norm of 9.65 against 9.92 -- and the decoder then
        renders that faithfully, as broadband hiss.  On a 288 s stem this put
        -56 dBFS of noise over passages where the input was exactly 0.0, and
        only over the chunks that also contained singing: chunks that were
        entirely silent came out at -107 dBFS.

        The decoder is not what is wrong.  Handed the features the *training*
        extraction produces, it tracks level to within 3 dB down to -120 dBFS
        frames.  So the fix belongs here, at the one place that still knows what
        the input actually was.

        Keyed on the input rather than the output, because that is the only
        signal that knows the difference between silence and a quiet passage the
        model rendered well.  Below ``threshold_db`` the gain fades to zero over
        ``knee_db``; ``hold_ms`` keeps the gate open across short dips so a word
        does not lose its tail, and ``release_ms`` closes it smoothly, since a
        step in the gain is a click.  Nothing above the knee is touched at all,
        which is what separates this from ``change_rms``: quiet-but-real audio
        keeps whatever dynamics the model gave it.
        """
        if threshold_db is None or not np.isfinite(threshold_db):
            return target_audio
        source = np.asarray(source_audio, dtype=np.float64)
        if source.ndim > 1:
            source = source.mean(axis=-1)
        window = max(1, int(source_rate * 0.020))
        hop = max(1, int(source_rate * 0.005))
        if source.size < window or target_audio.size == 0:
            return target_audio

        views = np.lib.stride_tricks.sliding_window_view(source, window)[::hop]
        level = 20.0 * np.log10(np.sqrt((views**2).mean(axis=-1)) + 1e-12)

        knee = max(1e-6, float(knee_db))
        gain = np.clip((level - (float(threshold_db) - knee)) / knee, 0.0, 1.0)
        if gain.min() >= 1.0:
            return target_audio

        # Hold: a dilation of the open gate, so a dip shorter than the hold
        # never closes it.  Written as a max over shifts rather than pulled from
        # scipy.ndimage to keep this dependency-free.
        hold = int(round(float(hold_ms) / 1000.0 * source_rate / hop))
        if hold > 0:
            held = gain.copy()
            for shift in range(1, hold + 1):
                held[shift:] = np.maximum(held[shift:], gain[:-shift])
            gain = held

        # Release: instantaneous to open, exponential to close.
        frames_per_second = source_rate / hop
        decay = float(
            np.exp(-1.0 / max(1e-6, float(release_ms) / 1000.0 * frames_per_second))
        )
        smoothed = np.empty_like(gain)
        running = gain[0]
        for index, value in enumerate(gain):
            running = value if value >= running else running * decay
            smoothed[index] = running

        # The gate is measured at the source rate and applied at the target's,
        # which are rarely the same; the centre of each analysis window is where
        # its measurement belongs.
        centres = (np.arange(smoothed.size) * hop + window / 2.0) / source_rate
        positions = np.arange(target_audio.shape[0]) / float(target_rate)
        envelope = np.interp(positions, centres, smoothed).astype(target_audio.dtype)
        return target_audio * envelope

    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.

        Args:
            source_audio: The source audio signal as a NumPy array.
            source_rate: The sampling rate of the source audio.
            target_audio: The target audio signal to adjust.
            target_rate: The sampling rate of the target audio.
            rate: The blending rate between the source and target RMS levels.
        """
        # Calculate RMS of both audio data
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjusted_audio


class Autotune:
    """
    A class for applying autotune to a given fundamental frequency (F0) contour.
    """

    def __init__(self):
        """
        Initializes the Autotune class with a set of reference frequencies.
        """
        self.note_dict = [
            49.00,  # G1
            51.91,  # G#1 / Ab1
            55.00,  # A1
            58.27,  # A#1 / Bb1
            61.74,  # B1
            65.41,  # C2
            69.30,  # C#2 / Db2
            73.42,  # D2
            77.78,  # D#2 / Eb2
            82.41,  # E2
            87.31,  # F2
            92.50,  # F#2 / Gb2
            98.00,  # G2
            103.83,  # G#2 / Ab2
            110.00,  # A2
            116.54,  # A#2 / Bb2
            123.47,  # B2
            130.81,  # C3
            138.59,  # C#3 / Db3
            146.83,  # D3
            155.56,  # D#3 / Eb3
            164.81,  # E3
            174.61,  # F3
            185.00,  # F#3 / Gb3
            196.00,  # G3
            207.65,  # G#3 / Ab3
            220.00,  # A3
            233.08,  # A#3 / Bb3
            246.94,  # B3
            261.63,  # C4
            277.18,  # C#4 / Db4
            293.66,  # D4
            311.13,  # D#4 / Eb4
            329.63,  # E4
            349.23,  # F4
            369.99,  # F#4 / Gb4
            392.00,  # G4
            415.30,  # G#4 / Ab4
            440.00,  # A4
            466.16,  # A#4 / Bb4
            493.88,  # B4
            523.25,  # C5
            554.37,  # C#5 / Db5
            587.33,  # D5
            622.25,  # D#5 / Eb5
            659.25,  # E5
            698.46,  # F5
            739.99,  # F#5 / Gb5
            783.99,  # G5
            830.61,  # G#5 / Ab5
            880.00,  # A5
            932.33,  # A#5 / Bb5
            987.77,  # B5
            1046.50,  # C6
        ]

    def autotune_f0(self, f0, f0_autotune_strength):
        """
        Autotunes a given F0 contour by snapping each frequency to the closest reference frequency.

        Args:
            f0: The input F0 contour as a NumPy array.
        """
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0


class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """

    def __init__(self, tgt_sr, config):
        """
        Initializes the Pipeline class with target sampling rate and configuration parameters.

        Args:
            tgt_sr: The target sampling rate for the output audio.
            config: A configuration object containing various parameters for the pipeline.
        """
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.autotune = Autotune()
        # F0 predictors and the retriever are reused across calls; loading them
        # per file dominates batch inference otherwise.
        self._f0_models = {}
        self._retriever_cache = {}

    def _get_f0_model(self, f0_method: str):
        """
        Returns the F0 predictor for the given method, loading it on first use.

        The predictors are stateless between calls, so a single instance is kept
        alive for the lifetime of the pipeline instead of reading the checkpoint
        from disk and re-uploading it to the GPU for every audio file.

        Args:
            f0_method (str): One of "crepe", "crepe-tiny", "rmvpe", "fcpe".
        """
        model = self._f0_models.get(f0_method)
        if model is not None:
            return model

        if f0_method in ("crepe", "crepe-tiny"):
            model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
        elif f0_method == "rmvpe":
            model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
        elif f0_method == "fcpe":
            model = FCPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.window
            )
        else:
            raise ValueError(f"Unknown f0 method: {f0_method}")

        self._f0_models[f0_method] = model
        return model

    def unload_f0_models(self):
        """
        Drops the cached F0 predictors and frees their GPU memory.
        """
        self._f0_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_f0(
        self,
        x,
        p_len,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        filter_radius: float = 3.0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        inp_f0=None,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.

        Args:
            input_audio_path: Path to the input audio file.
            x: The input audio signal as a NumPy array.
            p_len: Desired length of the F0 output.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation (e.g., "crepe").
            filter_radius: Radius for median filtering the F0 contour.
            f0_autotune: Whether to apply autotune to the F0 contour.
            inp_f0: Optional input F0 contour to use instead of estimating.
        """
        model = self._get_f0_model(f0_method)
        if f0_method == "crepe":
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
        elif f0_method == "crepe-tiny":
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
        elif f0_method == "rmvpe":
            f0 = model.get_f0(x, filter_radius=0.03)
        elif f0_method == "fcpe":
            f0 = model.get_f0(x, p_len, filter_radius=0.006, test_time_augmentation=True)
        # f0 adjustments
        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        else:
            f0 *= pow(2, pitch / 12)

        # Apply user-edited F0 if provided
        if inp_f0 is not None:
            replace_hz = inp_f0[:, 1].astype(np.float32)
            replace_hz_shifted = replace_hz * pow(2, pitch / 12)
            offset = self.t_pad // self.window
            n = min(len(f0) - offset, len(replace_hz))
            voiced = replace_hz[:n] > 0
            f0[offset : offset + n][voiced] = replace_hz_shifted[:n][voiced]

        # quantizing f0 to 255 buckets to make coarse f0
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        retriever,
        index_rate,
        version,
        protect,
        seed,
        retrieval_config=None,
        do_normalize=False,
    ):
        """
        Performs voice conversion on a given audio segment.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio0: The input audio segment.
            pitch: Quantized F0 contour for pitch guidance.
            pitchf: Original F0 contour for pitch guidance.
            retriever: IndexRetriever for speaker embedding retrieval, or None.
            index_rate: Blending rate for speaker embedding retrieval.
            version: Model version (Keep to support old models).
            protect: Protection level for preserving the original pitch.
            seed: Seed for randomization of noise.
            retrieval_config: Neighbour count, weighting and continuity for the retrieval.
            do_normalize: Whether the embedder wants its input layer-normalised.
        """

        with torch.no_grad():
            pitch_guidance = pitch != None and pitchf != None

            # prepare source audio
            feats = torch.from_numpy(audio0).float()

            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()

            feats = feats.view(1, -1).to(self.device)

            # extract features with contentvec on audio0
            feats = extract_features(model, feats, version, do_normalize=do_normalize)

            # make a copy for pitch guidance and protection
            feats0 = feats.clone() if pitch_guidance else None
            if retriever is not None and retriever.ready and index_rate > 0:
                feats = retriever.retrieve(feats, index_rate, retrieval_config)

            # feature upsampling
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )

            # adjust the length if the audio is short
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])

            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                # Pitch protection blending
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
            p_len = torch.tensor([p_len], device=self.device).long()

            # Inference
            audio1 = (
                net_g.infer(
                    phone=feats.float(),        # phone
                    phone_lengths=p_len,        # phone_lengths
                    pitch=pitch,                # quantized f0 curve
                    nsff0=pitchf.float(),       # float f0 curve
                    sid=sid,                    # speaker id
                    seed=seed,                  # inference seed
                )[0][0, 0]
                .detach()
                .cpu()
                .float()
                .numpy()
            )

            # clean up
            del feats, feats0, p_len
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1

    def _get_retriever(self, file_index, loaded_index, meta_payload, index_rate):
        """Return the retriever for this run, building it at most once.

        Batch inference calls ``pipeline`` per file, and the expensive parts --
        reconstructing every vector out of the index and uploading them -- do not
        depend on the file.  The cache is keyed on the index's identity so a
        second model in the same session gets its own.
        """
        if index_rate <= 0:
            return None

        if loaded_index is not None:
            key = ("loaded", id(loaded_index))
        elif file_index and os.path.exists(file_index):
            key = ("file", os.path.abspath(file_index))
        else:
            return None

        cached = self._retriever_cache.get(key)
        if cached is not None:
            return cached

        try:
            if loaded_index is not None:
                retriever = IndexRetriever.from_loaded(
                    loaded_index, meta_payload, self.device
                )
            else:
                retriever = IndexRetriever.from_path(file_index, self.device)
        except Exception as error:
            print_error(f"Could not read the index, retrieval is off: {error}", tag="[INFER]")
            return None

        info(f"Index: {retriever.describe()}.", tag="[INFER]")
        self._retriever_cache.clear()
        self._retriever_cache[key] = retriever
        return retriever

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        filter_radius,
        volume_envelope,
        version,
        protect,
        f0_autotune,
        f0_autotune_strength,
        f0_file,
        seed,
        loaded_index=None,
        index_meta_payload=None,
        retrieval_config=None,
        do_normalize=False,
        silence_gate_db=-60.0,
    ):
        """
        The main pipeline function for performing voice conversion.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            input_audio_path: Path to the input audio file.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            file_index: Path to the FAISS index file for speaker embedding retrieval.
            index_rate: Blending rate for speaker embedding retrieval.
            pitch_guidance: Whether to use pitch guidance during voice conversion.
            filter_radius: Radius for median filtering the F0 contour.
            tgt_sr: Target sampling rate for the output audio.
            resample_sr: Resampling rate for the output audio.
            volume_envelope: Blending rate for adjusting the RMS level of the output audio.
            version: Model version.
            protect: Protection level for preserving the original pitch.
            f0_autotune: Whether to apply autotune to the F0 contour.
            f0_file: Path to a file containing an F0 contour to use.
            seed: Seed for randomization of noise.
            loaded_index: A pre-loaded FAISS index object.
            index_meta_payload: Serialised sidecar accompanying ``loaded_index``.
            retrieval_config: Neighbour count, weighting and continuity for the retrieval.
            do_normalize: Whether the embedder wants its input layer-normalised.
            silence_gate_db: Input level below which the output is faded out.
                None or -inf disables it.  See ``AudioProcessor.gate_to_source``.
        """

        if seed == 0:
            seed = random.randint(1, 2**32 - 1)

        retriever = self._get_retriever(
            file_index, loaded_index, index_meta_payload, index_rate
        )

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")


        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None

        # Padding for 16k audio
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None

        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception as error:
                warning(f"F0 file ignored, could not be read: {error}", tag="[INFER]")


        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
                filter_radius,
                f0_autotune,
                f0_autotune_strength,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()


        for t in opt_ts:
            t = t // self.window * self.window
            if pitch_guidance:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        retriever,
                        index_rate,
                        version,
                        protect,
                        seed,
                        retrieval_config,
                        do_normalize,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        retriever,
                        index_rate,
                        version,
                        protect,
                        seed,
                        retrieval_config,
                        do_normalize,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )


            s = t
        if pitch_guidance:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    retriever,
                    index_rate,
                    version,
                    protect,
                    seed,
                    retrieval_config,
                    do_normalize,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    retriever,
                    index_rate,
                    version,
                    protect,
                    seed,
                    retrieval_config,
                    do_normalize,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)


        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt, self.tgt_sr, volume_envelope
            )

        # After the envelope blend, so the gate has the last word on anything
        # the input says is silent.
        audio_opt = AudioProcessor.gate_to_source(
            audio, self.sample_rate, audio_opt, self.tgt_sr, silence_gate_db
        )

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        if pitch_guidance:
            del pitch, pitchf
        del sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
