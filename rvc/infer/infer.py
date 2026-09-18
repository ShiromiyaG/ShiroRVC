import os
import sys
import random
import soxr
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf
import noisereduce as nr
import faiss

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.terminal import (
    error as print_error,
    info,
    install_rich_print,
    print_error_panel,
    progress_task,
    success,
    warning,
)

install_rich_print()

from rvc.infer.pipeline import Pipeline as VC
from rvc.infer.retrieval import RetrievalConfig
from rvc.lib.utils import load_audio_infer, load_embedder_model
from rvc.lib.extras.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm.commons import strip_parametrizations
from rvc.lib import index_meta
from rvc.lib.model_bundle import (
    default_model_name,
    get_bundle_model_state,
    get_bundle_models,
    is_model_bundle,
    load_model_bundle,
)
from rvc.configs.config import Config
from rvc.configs.vocoders import normalize_vocoder
from rvc.infer.messages import (
    INFER_RANDOM_SEED_EXPOSED,
    INFER_SEED_SPECIFIED,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

class VoiceConverter:
    """Performs voice conversion using the RVC method."""

    def __init__(self):
        self.config = Config()
        self.hubert_model = None
        self.last_embedder_model = None
        self.tgt_sr = None
        self.net_g = None
        self.vc = None
        self.active_cpt = None  # Active checkpoint for the selected speaker
        self.version = None
        self.n_spk = None
        self.use_f0 = None
        self.loaded_model = None
        self.loaded_index = None  # Deserialized Faiss index
        self.loaded_index_meta = None  # Metadata for the bundle index
        # Whether the embedder wants its input layer-normalised.  Extraction has
        # always honoured this; inference used to drop it on the floor, so an
        # embedder whose config asked for normalisation produced training
        # features and query features from two different distributions -- an
        # index that silently retrieved the wrong neighbours, with nothing
        # anywhere reporting a problem.
        self.hubert_do_normalize = False

    def load_hubert(self, embedder_model: str):
        self.hubert_model, self.hubert_do_normalize = load_embedder_model(
            embedder_model
        )
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        try:
            reduced_noise = nr.reduce_noise(
                y=data, sr=sr, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            warning(f"Noise reduction failed, keeping the raw audio: {error}", tag="[INFER]")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        try:
            if output_format != "WAV":
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print_error(f"Could not write the audio as {output_format}: {error}", tag="[INFER]")

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_file: str = None,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1,
        silence_gate_db: float = -60.0,
        protect: float = 0.5,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        filter_radius: float = 3.0,
        embedder_model: str = "contentvec",
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        resample_sr: int = 0,
        sid: int = 0,
        seed: int = 0,
        bundle_submodel: str = None,
        index_k: int = 8,
        index_power: float = 2.0,
        index_continuity: float = 0.5,
        noise_scale: float = None,
        **kwargs,
    ):
        """silence_gate_db: input level (dBFS) under which output is faded out. The
        content encoder gives digital silence a full-magnitude embedding whose
        direction depends on the rest of the chunk, and the decoder renders that
        as hiss; this keeps it out of passages the input says are empty. None or
        -inf disables the gate.

        noise_scale: scale of the prior draw the model decodes. None uses the
        model's own default (0.3 for RefineGAN2, 0.66666 otherwise or when the
        checkpoint carries ``prior_noise_subspace``).
        """
        if not model_path:
            print_error("No model provided. Aborting conversion.", tag="[INFER]")
            return

        self.get_vc(model_path, sid, bundle_submodel)

        if not self.vc:
            print_error(
                "The conversion pipeline did not initialise; see the model "
                "loading errors above. Aborting conversion.",
                tag="[INFER]",
            )
            return

        try:
            start_time = time.time()
            info(f"Converting '{audio_input_path}'", tag="[INFER]")

            audio = load_audio_infer(audio_input_path, 16000, **kwargs)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model)
                self.last_embedder_model = embedder_model

            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                if index_path and os.path.exists(index_path) else ""
            )

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio:
                chunks, intervals = process_audio(audio, 16000)
                info(f"Audio split into {len(chunks)} chunks.", tag="[INFER]")
            else:
                chunks = [audio]

            if seed != 0:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                info(INFER_SEED_SPECIFIED.format(seed=seed), tag="[INFER]")
            else:
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                info(INFER_RANDOM_SEED_EXPOSED.format(seed=seed), tag="[INFER]")

            converted_chunks = []
            retrieval_config = RetrievalConfig.build(
                k=index_k, power=index_power, continuity=index_continuity
            )
            # Inference.  A single chunk finishes in one step, so the bar only
            # earns its place when the audio was split.
            with progress_task(
                len(chunks),
                "Converting",
                disable=len(chunks) < 2,
            ) as (chunk_progress, chunk_task):
                for c in chunks:
                    audio_opt = self.vc.pipeline(
                        model=self.hubert_model,
                        net_g=self.net_g,
                        sid=sid,
                        audio=c,
                        pitch=pitch,
                        f0_method=f0_method,
                        file_index=file_index,
                        index_rate=index_rate,
                        pitch_guidance=self.use_f0,
                        filter_radius=filter_radius,
                        volume_envelope=volume_envelope,
                        silence_gate_db=silence_gate_db,
                        version=self.version,
                        protect=protect,
                        f0_autotune=f0_autotune,
                        f0_autotune_strength=f0_autotune_strength,
                        f0_file=f0_file,
                        seed=seed,
                        loaded_index=self.loaded_index,
                        index_meta_payload=self.loaded_index_meta,
                        retrieval_config=retrieval_config,
                        do_normalize=self.hubert_do_normalize,
                        noise_scale=noise_scale,
                    )
                    converted_chunks.append(audio_opt)
                    chunk_progress.advance(chunk_task)

            if split_audio:
                audio_opt = merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
            else:
                audio_opt = converted_chunks[0]

            if clean_audio:
                cleaned_audio = self.remove_audio_noise(
                    audio_opt, self.tgt_sr, clean_strength
                )
                if cleaned_audio is not None:
                    audio_opt = cleaned_audio

            sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            intermediate_wav = audio_output_path
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )
            if export_format != "WAV" and os.path.exists(intermediate_wav):
                try:
                    os.remove(intermediate_wav)
                except OSError:
                    pass

            elapsed_time = time.time() - start_time
            success(
                f"Converted in {elapsed_time:.2f}s -> '{audio_output_path}'",
                tag="[INFER]",
            )
        except Exception as error:
            print_error_panel(
                error,
                title="Conversion failed",
                details=traceback.format_exc(),
            )

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"), "w"
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            info(f"Converting batch '{audio_input_paths}'", tag="[INFER]")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.lower().endswith(
                    (
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "opus",
                        "m4a",
                        "mp4",
                        "aac",
                        "alac",
                        "wma",
                        "aiff",
                        "webm",
                        "ac3",
                    )
                )
            ]
            info(f"{len(audio_files)} audio files queued.", tag="[INFER]")
            for a in audio_files:
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.splitext(a)[0] + "_output.wav"
                new_output = os.path.join(audio_output_path, new_output)
                if os.path.exists(new_output):
                    continue
                self.convert_audio(
                    audio_input_path=new_input,
                    audio_output_path=new_output,
                    **kwargs,
                )
            elapsed_time = time.time() - start_time
            success(
                f"Batch of {len(audio_files)} files converted in {elapsed_time:.2f}s "
                f"-> '{audio_output_path}'",
                tag="[INFER]",
            )
        except Exception as error:
            print_error_panel(
                error,
                title="Batch conversion failed",
                details=traceback.format_exc(),
            )
        finally:
            if os.path.exists(os.path.join(now_dir, "assets", "infer_pid.txt")):
                os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid, bundle_submodel=None):
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        # A bundle is loaded one sub-model at a time, so switching sub-models
        # is a reload rather than a lookup.
        key = (weight_root, bundle_submodel or None)
        if self.loaded_model != key:
            self.load_model(weight_root, bundle_submodel)

        if self.active_cpt is not None:
            self.setup_network()
            self.setup_vc_instance()
            self.loaded_model = key
        else:
            self.vc = None
            self.loaded_model = None

    def cleanup_model(self):
        import gc
        for attr in ("net_g", "n_spk", "vc", "hubert_model", "tgt_sr", "active_cpt", "loaded_model", "loaded_index", "loaded_index_meta"):
            setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, weight_root, bundle_submodel=None):
        """Handles both plain .pth checkpoints and model bundles.

        Of a bundle only ``bundle_submodel`` is kept -- the first model when
        none is named -- so the others are not held in memory for the session.
        """
        self.active_cpt = None
        self.loaded_index = None
        self.loaded_index_meta = None

        if not os.path.isfile(weight_root):
            print_error(f"Model file not found: {weight_root}", tag="[INFER]")
            return

        info(f"Loading model '{os.path.basename(weight_root)}'", tag="[INFER]")
        if not is_model_bundle(weight_root):
            self.active_cpt = torch.load(weight_root, map_location="cpu", weights_only=True)
            return

        try:
            bundle_data = load_model_bundle(weight_root)
        except Exception as e:
            print_error(f"Could not load the model bundle: {e}", tag="[INFER]")
            return

        models = get_bundle_models(bundle_data)
        if models:
            name = bundle_submodel or default_model_name(models)
            if name not in models:
                print_error(f"Sub-model '{name}' is not in the bundle.", tag="[INFER]")
                return
            if not bundle_submodel:
                info(f"No sub-model chosen; using '{name}'.", tag="[INFER]")
            info(f"Bundle holds {len(models)} models; loaded '{name}'.", tag="[INFER]")
            entry = models[name]
            state = entry.get("model_state")
        else:  # the older single-model layout
            entry = bundle_data
            state = get_bundle_model_state(bundle_data)

        self.active_cpt = state
        self.loaded_index_meta = entry.get("index_meta")
        index_data = entry.get("index_data")
        if index_data is not None:
            # Without a separate copy, the index file's own footer has it.
            if self.loaded_index_meta is None:
                self.loaded_index_meta = index_meta.from_index_bytes(index_data)
            try:
                self.loaded_index = faiss.deserialize_index(index_data)
            except Exception as e:
                warning(f"Bundled index could not be read, retrieval is off: {e}", tag="[INFER]")

    def setup_network(self):
        if self.active_cpt is not None:
            self.tgt_sr = self.active_cpt["config"][-1]
            self.active_cpt["config"][-3] = self.active_cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = True

            self.version = self.active_cpt.get("version", "v1")
            # The width the model was actually built with, read off the text
            # encoder's own input layer.  The version rule below is only a
            # fallback: it holds for contentvec and the spin_v* pair, and
            # would be wrong for any embedder whose width is not tied to the
            # RVC version.
            emb_phone = self.active_cpt["weight"].get("enc_p.emb_phone.weight")
            if emb_phone is not None:
                self.text_enc_hidden_dim = int(emb_phone.shape[1])
            else:
                self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = normalize_vocoder(
                self.active_cpt.get(
                    "vocoder_id",
                    self.active_cpt.get("vocoder_architecture", self.active_cpt.get("vocoder", "hifi")),
                )
            )
            vocoder_config = self.active_cpt.get("vocoder_config", {}) or {}

            synth_kwargs = {
                "use_f0": self.use_f0,
                "text_enc_hidden_dim": self.text_enc_hidden_dim,
                "vocoder": self.vocoder,
                "vocoder_config": vocoder_config,
            }

            self.net_g = Synthesizer(*self.active_cpt["config"], **synth_kwargs)

            self.net_g.load_state_dict(self.active_cpt["weight"], strict=False)
            self.net_g.set_prior_noise_subspace(self.active_cpt.get("prior_noise_subspace"))
            # ``remove_training_modules`` drops the posterior on either
            # frontend and keeps the flow where inference needs it.
            self.net_g.remove_training_modules()
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            # Fold weight norm into the weights: the generator is frozen from
            # here on, so recomputing g * v/||v|| on every forward is wasted work.
            strip_parametrizations(self.net_g)

    def setup_vc_instance(self):
        if self.active_cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.active_cpt["config"][-3]
