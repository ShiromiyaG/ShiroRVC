import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from rvc.lib.algorithm.commons import upsample_content
from rvc.lib.algorithm.energy import frame_energy
from rvc.lib.extras.split_audio import merge_audio, process_audio
from rvc.lib.model_bundle import (
    RECTIFIED_KIND,
    default_model_name,
    get_bundle_models,
    is_model_bundle,
    load_model_bundle,
    model_kind,
)
from rvc.lib.terminal import info, print_error_panel, print_settings_panel, progress_task, success
from rvc.lib.utils import extract_features, load_audio_infer, load_embedder_model
from rvc.rectified.aperiodicity import aperiodicity, tension
from rvc.rectified.common import (
    TENSION_SMOOTH_SECONDS,
    clean_f0,
    curve_to_mel_rate,
    f0_to_mel_rate,
    mel_frames,
    smooth_curve,
    to_mel_rate,
)
from rvc.rectified.flow_model import Conditioning, build_flow
from rvc.rectified.vocoder import load_vocoder, mel_mismatch

INPUT_RATE = 16000
INPUT_HOP = 160
#: Content frames (20 ms) per embedder pass; the embedder's receptive field is
#: 400 samples at a 320-sample stride.
EMBEDDER_CHUNK = 1500
EMBEDDER_RATE = 50
EMBEDDER_STRIDE = 320
EMBEDDER_FIELD = 400
#: Frames per flow pass, and the overlap crossfaded between passes.
FLOW_CHUNK = 3000
FLOW_OVERLAP = 100
#: Frames per vocoder pass, the context rendered on each side, and the
#: crossfade at each join: every pass starts its sine at phase zero, so a hard
#: join would click.
VOCODER_CHUNK = 6000
VOCODER_CONTEXT = 50
VOCODER_CROSSFADE = 4


@dataclass(frozen=True)
class PitchOptions:
    """How the pitch to sing is made from the input's: transposed by
    ``transpose`` semitones, or autotuned. ``median`` and ``octave_fix`` are
    ``clean_f0``'s, on the input's pitch."""

    method: str = "rmvpe"
    transpose: int = 0
    autotune: bool = False
    autotune_strength: float = 1.0
    median: int = 0
    octave_fix: bool = False

    def describe(self) -> str:
        return f"{self.transpose:+d} st, {self.method}" + (", autotune" if self.autotune else "")

    def describe_cleanup(self) -> str:
        return f"median {self.median} frames" + (", octave fix" if self.octave_fix else "")


@dataclass(frozen=True)
class IndexOptions:
    """Retrieval over the experiment's content features; ``protect`` below 0.5
    keeps the unretrieved features in unvoiced frames, as the RVC pipeline
    does."""

    rate: float = 0.5
    k: int = 8
    power: float = 2.0
    continuity: float = 0.5
    protect: float = 0.33

    def describe(self, source: str) -> str:
        """``source`` names the index; empty without one."""
        if self.rate <= 0 or not source:
            return "off"
        return f"{source}, rate {self.rate:g}, protect {self.protect:g}"


@dataclass(frozen=True)
class SamplingOptions:
    """``RectifiedFlow.sample``'s arguments, under its names."""

    steps: int = 16
    method: str = "euler"
    cfg_scale: float = 2.0
    content_guidance: float = 0.1
    guidance_rescale: float = 0.7
    rescale_mode: str = "global"
    guidance_interval: tuple = (0.0, 1.0)
    temperature: float = 1.0
    schedule: str = "uniform"
    churn: float = 0.0
    start: Optional[float] = None

    def describe(self) -> str:
        return (
            f"{self.steps} {self.method} steps, {self.schedule} schedule, temperature {self.temperature:g}"
            + (f", churn {self.churn:g}" if self.churn > 0 else "")
            + (f", start {self.start:g}" if self.start is not None and self.start > 0 else "")
        )

    def describe_guidance(self) -> str:
        guide_from, guide_until = self.guidance_interval
        return (
            f"speaker {self.cfg_scale:g}, content {self.content_guidance:g}, "
            f"rescale {self.guidance_rescale:g} ({self.rescale_mode}), "
            f"t in [{guide_from:g}, {guide_until:g})"
        )


class RectifiedConverter:
    """Audio -> content and f0 -> rectified flow -> mel -> PCPH-BigVGAN."""

    def __init__(self):
        from rvc.configs.config import Config

        self.config = Config()
        self.device = self.config.device
        self.flow = self.flow_path = self.flow_meta = None
        #: ``(index_data, index_meta)`` of a bundled flow's own index, or None.
        self.flow_index = None
        self.vocoder = self.vocoder_path = self.vocoder_config = None
        self.embedder = self.embedder_name = None
        self.embedder_normalize = False
        self.pipeline = None
        self.retriever = self.retriever_path = None

    def _load_flow(self, path, submodel=None):
        """A flow export, or the flow ``submodel`` of a bundle (its first
        flow when None)."""
        key = (path, submodel or None)
        if key == self.flow_path:
            return
        index = None
        if is_model_bundle(path):
            models = {
                name: entry for name, entry in get_bundle_models(load_model_bundle(path)).items()
                if model_kind(entry.get("model_state") or {}) == RECTIFIED_KIND
            }
            name = submodel or default_model_name(models)
            if name not in models:
                raise ValueError(
                    f"'{name}' is not a rectified flow in {os.path.basename(path)}." if name
                    else f"{os.path.basename(path)} holds no rectified flow."
                )
            entry = models[name]
            checkpoint = entry["model_state"]
            if entry.get("index_data") is not None:
                index = (entry["index_data"], entry.get("index_meta"))
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("kind") != "rectified_flow":
            raise ValueError(f"{path} is not an exported rectified-flow model.")
        model = build_flow(checkpoint["config"], checkpoint["speaker_count"])
        model.load_state_dict(checkpoint["model"])
        self.flow = model.to(self.device).eval()
        self.flow_path = key
        self.flow_meta = checkpoint
        self.flow_index = index

    def _load_vocoder(self, path, data):
        """The vocoder at ``path`` for the flow's mel ``data``; an OpenVPI one
        is rebuilt when the flow's normalisation changes."""
        key = (path, data["mel_mean"], data["mel_std"])
        if key == self.vocoder_path:
            return
        model, vocoder_data = load_vocoder(path, data)
        self.vocoder = model.to(self.device)
        self.vocoder_path = key
        self.vocoder_config = {"data": vocoder_data}

    def _load_embedder(self, name):
        if name == self.embedder_name:
            return
        model, self.embedder_normalize = load_embedder_model(name)
        self.embedder = model.to(self.device).float().eval()
        self.embedder_name = name

    def unload(self):
        """Free every loaded model."""
        import gc

        self.flow = self.flow_path = self.flow_meta = self.flow_index = None
        self.vocoder = self.vocoder_path = self.vocoder_config = None
        self.embedder = self.embedder_name = None
        self.pipeline = None
        self.retriever = self.retriever_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_retriever(self, path):
        """The loaded flow's bundled index when it has one, else the index at
        ``path``; built once per source, None without one."""
        from rvc.infer.retrieval import IndexRetriever

        if self.flow_index is not None:
            key = ("bundle", self.flow_path)
            if key != self.retriever_path:
                import faiss

                from rvc.lib import index_meta

                data, meta = self.flow_index
                self.retriever = IndexRetriever.from_loaded(
                    faiss.deserialize_index(data),
                    meta if meta is not None else index_meta.from_index_bytes(data),
                    self.device,
                )
                self.retriever_path = key
            return self.retriever
        if not path or not os.path.exists(path):
            return None
        if os.path.abspath(path) != self.retriever_path:
            self.retriever = IndexRetriever.from_path(path, self.device)
            self.retriever_path = os.path.abspath(path)
        return self.retriever

    def _pitch(self, audio, frames, options: PitchOptions):
        """The input's own pitch, which the breathiness is measured against, and
        the pitch to sing, transposed or autotuned as ``Pipeline.get_f0`` does."""
        if self.pipeline is None:
            from rvc.infer.pipeline import Pipeline

            self.pipeline = Pipeline(self.vocoder_config["data"]["sample_rate"], self.config)
        _, source = self.pipeline.get_f0(audio, frames, options.method, 0)
        source = np.asarray(source, dtype=np.float32)[:frames]
        source = np.pad(source, (0, frames - source.shape[0]))
        source = clean_f0(source, options.median, options.octave_fix)
        if options.autotune:
            f0 = self.pipeline.autotune.autotune_f0(source.copy(), options.autotune_strength)
        else:
            f0 = source * pow(2, options.transpose / 12)
        return source, np.asarray(f0, dtype=np.float32)

    @torch.no_grad()
    def _content(self, source, context_seconds=2.0):
        """Content features [1, T, C] of ``source`` [1, samples], in passes
        with ``context_seconds`` either side so long inputs fit in memory."""
        if self.embedder_normalize:
            # Over the whole input, as a single pass would.
            source = F.layer_norm(source, source.shape[-1:])
        frames = (source.shape[-1] - EMBEDDER_FIELD) // EMBEDDER_STRIDE + 1
        context = max(0, int(round(context_seconds * EMBEDDER_RATE)))
        parts = []
        for start in range(0, frames, EMBEDDER_CHUNK):
            stop = min(frames, start + EMBEDDER_CHUNK)
            left = max(0, start - context)
            right = min(frames, stop + context)
            window = source[:, left * EMBEDDER_STRIDE : (right - 1) * EMBEDDER_STRIDE + EMBEDDER_FIELD]
            part = extract_features(self.embedder, window, "v2", do_normalize=False).float()
            parts.append(part[:, start - left : stop - left])
        return torch.cat(parts, 1)

    @torch.no_grad()
    def _inputs(self, audio, full, sid, pitch: PitchOptions, index: IndexOptions, retriever, content_context,
                formant_shift, tension_strength) -> Conditioning:
        """The flow's inputs for ``audio``, a NumPy array at ``INPUT_RATE``, at
        the mel frame rate. ``full`` is the same input at the mel's sample
        rate: training measures the loudness, breathiness and tension there,
        and a 16 kHz copy has lost what a sibilant carries above 8 kHz.
        ``retriever`` is the loaded index, or None."""
        data = self.flow_meta["config"]["data"]
        rate, hop = data["sample_rate"], data["hop_length"]
        source = torch.from_numpy(audio).view(1, -1).to(self.device)
        full = torch.from_numpy(full).view(1, -1).to(self.device)
        content = self._content(source, content_context)
        original = content
        if retriever is not None:
            from rvc.infer.retrieval import RetrievalConfig

            config = RetrievalConfig.build(k=index.k, power=index.power, continuity=index.continuity)
            content = retriever.retrieve(content, float(index.rate), config)
        content = upsample_content(content, data["content_interpolation"])
        original = upsample_content(original, data["content_interpolation"])
        frames = min(audio.shape[0] // INPUT_HOP, content.shape[1])
        content, original = content[:, :frames], original[:, :frames]

        source_f0, f0 = self._pitch(audio, frames, pitch)
        source_f0 = torch.from_numpy(source_f0).view(1, -1).to(self.device)
        f0 = torch.from_numpy(f0).view(1, -1).to(self.device)
        energy = smooth_curve(frame_energy(full, rate, frames))
        breathiness = smooth_curve(aperiodicity(full, rate, source_f0, frames))
        strain = None
        if self.flow.encoder.tension is not None and tension_strength > 0:
            strain = float(tension_strength) * smooth_curve(
                tension(full, rate, source_f0, frames), TENSION_SMOOTH_SECONDS
            )

        # Everything above is at 100 frames per second; the mel has its own rate.
        frames = mel_frames(frames, rate, hop)
        content = to_mel_rate(content, frames, rate, hop)
        original = to_mel_rate(original, frames, rate, hop)
        f0 = f0_to_mel_rate(f0, frames, rate, hop)
        if retriever is not None and index.protect < 0.5:
            keep = torch.where(f0 > 0, 1.0, float(index.protect)).unsqueeze(-1)
            content = content * keep + original * (1.0 - keep)
        return Conditioning(
            content=content,
            f0=f0,
            energy=curve_to_mel_rate(energy, frames, rate, hop),
            speaker=torch.tensor([sid], device=self.device),
            mask=torch.ones(1, 1, frames, device=self.device),
            breathiness=curve_to_mel_rate(breathiness, frames, rate, hop),
            key_shift=torch.full((1,), float(formant_shift), device=self.device),
            tension=None if strain is None else curve_to_mel_rate(strain, frames, rate, hop),
        )

    @torch.no_grad()
    def _sample_mel(self, inputs: Conditioning, options: SamplingOptions):
        """Normalised mel [1, n_mels, T], in overlapping passes crossfaded."""
        frames = inputs.content.shape[1]
        noise = torch.randn(1, self.flow.n_mels, frames, device=self.device)
        churn_seed = int(torch.randint(2**31 - 1, ()))
        mel = torch.zeros_like(noise)
        weight = torch.zeros(1, 1, frames, device=self.device)
        passes = 1 + max(0, -(-(frames - FLOW_CHUNK) // (FLOW_CHUNK - FLOW_OVERLAP)))
        with progress_task(passes * options.steps, "Sampling") as (progress, task):
            start = 0
            while start < frames:
                stop = self._flow_pass(
                    inputs, options, noise, mel, weight, start, lambda: progress.advance(task), churn_seed
                )
                if stop == frames:
                    break
                start = stop - FLOW_OVERLAP
        return mel / weight.clamp_min(1e-4)

    def _flow_pass(self, inputs: Conditioning, options: SamplingOptions, noise, mel, weight, start,
                   callback, churn_seed):
        """One flow pass from ``start``, added into ``mel`` and ``weight`` with
        its crossfade ramps; returns where it stopped."""
        frames = inputs.content.shape[1]
        stop = min(frames, start + FLOW_CHUNK)

        def churn_noise(step):
            # Drawn over the whole input so overlapping passes share it.
            generator = torch.Generator(self.device).manual_seed(churn_seed + step)
            return torch.randn(noise.shape, generator=generator, device=self.device)[..., start:stop]

        part = self.flow.sample(
            inputs.crop(start, stop), noise=noise[..., start:stop], callback=callback,
            churn_noise=churn_noise, **vars(options),
        )
        ramp = torch.ones(stop - start, device=self.device)
        fade = min(FLOW_OVERLAP, stop - start)
        if start > 0:
            ramp[:fade] = torch.linspace(0.0, 1.0, fade, device=self.device)
        if stop < frames:
            ramp[-fade:] = torch.minimum(ramp[-fade:], torch.linspace(1.0, 0.0, fade, device=self.device))
        mel[..., start:stop] += part * ramp
        weight[..., start:stop] += ramp
        return stop

    @torch.no_grad()
    def _render(self, mel, f0):
        """Waveform from [1, n_mels, T], in passes crossfaded at the joins."""
        hop = self.vocoder_config["data"]["hop_length"]
        frames = mel.shape[-1]
        output = torch.zeros(frames * hop, device=self.device)
        half = VOCODER_CROSSFADE * hop // 2
        starts = range(0, frames, VOCODER_CHUNK)
        with progress_task(len(starts), "Rendering", disable=len(starts) < 2) as (progress, task):
            for start in starts:
                self._vocoder_pass(mel, f0, output, start, half)
                progress.advance(task)
        return output.cpu().numpy()

    def _vocoder_pass(self, mel, f0, output, start, half):
        """Render one pass from ``start`` and add it into ``output``, faded in
        and out over ``half`` samples either side of its joins."""
        hop = self.vocoder_config["data"]["hop_length"]
        frames = mel.shape[-1]
        stop = min(frames, start + VOCODER_CHUNK)
        left = max(0, start - VOCODER_CONTEXT)
        right = min(frames, stop + VOCODER_CONTEXT)
        audio = self.vocoder(mel[..., left:right], f0[:, left:right])[0, 0].float()
        position = torch.arange(left * hop, left * hop + audio.shape[0], device=self.device)
        weight = torch.ones_like(audio)
        if start > 0:
            weight = weight * ((position - (start * hop - half)) / (2 * half)).clamp(0, 1)
        if stop < frames:
            weight = weight * (((stop * hop + half) - position) / (2 * half)).clamp(0, 1)
        output[left * hop : left * hop + audio.shape[0]] += audio * weight

    def convert(
        self,
        audio_input_path: str,
        audio_output_path: str,
        flow_path: str,
        vocoder_path: str,
        sid: int = 0,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        steps: int = 16,
        sampler: str = "euler",
        cfg_scale: float = 2.0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        seed: int = 0,
        export_format: str = "WAV",
        index_path: str = "",
        index_rate: float = 0.5,
        index_k: int = 8,
        index_power: float = 2.0,
        index_continuity: float = 0.5,
        protect: float = 0.33,
        formant_shift: float = 0.0,
        content_guidance: float = 0.1,
        guidance_rescale: float = 0.7,
        split_audio: bool = False,
        silence_gate_db: float = -60.0,
        noise_temperature: float = 1.0,
        flow_start: float = 0.0,
        guidance_from: float = 0.0,
        guidance_until: float = 1.0,
        rescale_mode: str = "global",
        schedule: str = "uniform",
        churn: float = 0.0,
        f0_median: int = 0,
        f0_octave_fix: bool = False,
        content_context: float = 2.0,
        flow_submodel: str = "",
        match_level: bool = True,
        tension_strength: float = 1.0,
    ):
        """Convert one file; returns the path written, or None on failure.

        ``flow_submodel`` picks the flow when ``flow_path`` is a bundle; a
        bundled index, when the flow has one, is used in place of ``index_path``.

        ``index_path`` is an RVC retrieval index over this experiment's content
        features; ``protect`` below 0.5 keeps the unretrieved features in
        unvoiced frames, as the RVC pipeline does. ``formant_shift`` moves the
        formants by that many semitones, apart from the pitch.
        ``content_guidance`` and ``guidance_rescale`` are ``RectifiedFlow.sample``'s.
        ``split_audio`` converts each non-silent segment on its own and puts
        the silences back. ``silence_gate_db`` fades the output out where the
        input is quieter than that, as ``AudioProcessor.gate_to_source`` does.

        ``noise_temperature``, ``flow_start`` (0 keeps the model's own),
        ``guidance_from``/``guidance_until``, ``rescale_mode``, ``schedule``
        and ``churn`` are ``RectifiedFlow.sample``'s ``temperature``,
        ``start``, ``guidance_interval``, ``rescale_mode``, ``schedule`` and
        ``churn``; ``f0_median`` and ``f0_octave_fix`` clean the input's pitch;
        ``content_context`` is the seconds of audio each content pass sees
        either side.

        ``match_level`` peak-normalises the input as the training data was, since
        the energy input is absolute, and scales the output back to the input's
        level; off, the input is only attenuated when it would clip.

        ``tension_strength`` scales how far the input's tension, its departure
        from its own usual tilt, carries over; 0 leaves the voice at its own.
        Ignored by a model without the input."""
        try:
            started = time.time()
            self._load_flow(flow_path, flow_submodel)
            flow_data = self.flow_meta["config"]["data"]
            self._load_vocoder(vocoder_path, flow_data)
            self._load_embedder(self.flow_meta.get("embedder_model", "contentvec"))
            mismatch = mel_mismatch(flow_data, self.vocoder_config["data"])
            if mismatch:
                raise ValueError(f"The flow and vocoder models disagree on {mismatch}.")
            if not 0 <= int(sid) < int(self.flow_meta["speaker_count"]):
                raise ValueError(f"Speaker {sid} is not in this model.")
            seed = int(seed) or random.randint(1, 2**31 - 1)
            torch.manual_seed(seed)
            pitch_options = PitchOptions(
                f0_method, int(pitch), bool(f0_autotune), f0_autotune_strength,
                int(f0_median), bool(f0_octave_fix),
            )
            index = IndexOptions(index_rate, index_k, index_power, index_continuity, protect)
            sampling = SamplingOptions(
                steps=int(steps), method=sampler, cfg_scale=float(cfg_scale),
                content_guidance=float(content_guidance), guidance_rescale=float(guidance_rescale),
                rescale_mode=rescale_mode, guidance_interval=(float(guidance_from), float(guidance_until)),
                temperature=float(noise_temperature), schedule=schedule, churn=float(churn),
                start=float(flow_start) or None,
            )
            index_source = "bundled" if self.flow_index is not None else os.path.basename(index_path)
            print_settings_panel(
                [
                    ("Input", audio_input_path),
                    ("Flow", os.path.basename(flow_path) + (f" [{flow_submodel}]" if flow_submodel else "")),
                    ("Vocoder", os.path.basename(vocoder_path)),
                    ("Speaker", sid),
                    ("Pitch", pitch_options.describe()),
                    ("Pitch cleanup", pitch_options.describe_cleanup()),
                    ("Sampling", sampling.describe()),
                    ("Guidance", sampling.describe_guidance()),
                    ("Content context", f"{content_context:g} s"),
                    ("Formant shift", f"{formant_shift:+g} st"),
                    ("Tension", f"{tension_strength:g}" if self.flow.encoder.tension is not None else "not in this model"),
                    ("Index", index.describe(index_source)),
                    ("Match level", "on" if match_level else "off"),
                    ("Split audio", "on" if split_audio else "off"),
                    ("Silence gate", f"{silence_gate_db:g} dBFS"),
                    ("Seed", seed),
                ],
                title="Rectified conversion",
            )

            sample_rate = self.vocoder_config["data"]["sample_rate"]
            # Decoded once, at the mel's rate; content and pitch read a copy of it.
            full = load_audio_infer(audio_input_path, sample_rate)
            audio = librosa.resample(full, orig_sr=sample_rate, target_sr=INPUT_RATE, res_type="soxr_vhq")
            peak = float(np.abs(audio).max())
            gain = 0.95 / peak if peak > 0 else 1.0
            if not match_level:
                gain = min(gain, 1.0)
            audio = (audio * gain).astype(np.float32)
            full = (full * gain).astype(np.float32)
            # Only the boost is undone; a hot input stays attenuated.
            restore = 1.0 / max(gain, 1.0)
            retriever = self._load_retriever(index_path) if index.rate > 0 else None
            from rvc.infer.pipeline import AudioProcessor

            def convert_segment(audio, full):
                if audio.shape[0] < EMBEDDER_FIELD + INPUT_HOP:
                    return np.zeros(round(audio.shape[0] * sample_rate / INPUT_RATE), dtype=np.float32)
                inputs = self._inputs(
                    audio, full, int(sid), pitch_options, index, retriever, content_context,
                    formant_shift, tension_strength,
                )
                mel = self._sample_mel(inputs, sampling)
                # The vocoder takes the normalised mel the flow produces.
                output = self._render(mel, inputs.f0) * restore
                return AudioProcessor.gate_to_source(
                    audio * restore, INPUT_RATE, output, sample_rate, silence_gate_db
                )

            if split_audio:
                segments, intervals = process_audio(audio, INPUT_RATE)
                info(f"Audio split into {len(segments)} segments.", tag="[RECTIFIED]")
                scale = sample_rate / INPUT_RATE
                converted = [
                    convert_segment(segment, full[round(start * scale) : round(stop * scale)])
                    for segment, (start, stop) in zip(segments, intervals)
                ]
                output = merge_audio(segments, converted, intervals, INPUT_RATE, sample_rate)
            else:
                output = convert_segment(audio, full)

            os.makedirs(os.path.dirname(os.path.abspath(audio_output_path)), exist_ok=True)
            sf.write(audio_output_path, output, sample_rate, format="WAV")
            if export_format != "WAV":
                from rvc.infer.infer import VoiceConverter

                target = os.path.splitext(audio_output_path)[0] + f".{export_format.lower()}"
                VoiceConverter.convert_audio_format(audio_output_path, target, export_format)
                os.remove(audio_output_path)
                audio_output_path = target
            success(
                f"Converted in {time.time() - started:.2f}s -> '{audio_output_path}'",
                tag="[RECTIFIED]",
            )
            return audio_output_path
        except Exception as error:
            print_error_panel(error, title="Rectified conversion failed", details=traceback.format_exc())
            return None
