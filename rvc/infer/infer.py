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

from rvc.lib.terminal import get_console, install_rich_print

install_rich_print()

from rvc.infer.pipeline import Pipeline as VC
from rvc.lib.utils import load_audio_infer, load_embedder_model
from rvc.lib.tools.split_audio import process_audio, merge_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.model_bundle import get_bundle_models, is_model_bundle, load_model_bundle
from rvc.configs.config import Config
from rvc.configs.vocoders import normalize_vocoder

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.active_cpt = None # Active checkpoint for the selected speaker
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.loaded_model = None
        self.loaded_index = None # Holds the deserialized Faiss index

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        self.hubert_model, _ = load_embedder_model(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.

        Args:
            data (numpy.ndarray): The audio data as a NumPy array.
            sr (int): The sample rate of the audio data.
            reduction_strength (float): Strength of the noise reduction. Default is 0.7.
        """
        try:
            reduced_noise = nr.reduce_noise(
                y=data, sr=sr, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            get_console().print(f"An error occurred removing audio noise: {error}")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output audio file.
            output_format (str): Desired audio format (e.g., "WAV", "MP3").
        """
        try:
            if output_format != "WAV":
                get_console().print(f"Saving audio as {output_format}...")
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
            get_console().print(f"An error occurred converting the audio format: {error}")

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
        protect: float = 0.5,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        filter_radius: float = 3.0,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        resample_sr: int = 0,
        sid: int = 0,
        seed: int = 0,
        bundle_submodel: str = None,
        **kwargs,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            pitch (int): Key for F0 up-sampling.
            filter_radius (float): Radius for filtering.
            index_rate (float): Rate for index matching.
            volume_envelope (int): RMS mix rate.
            protect (float): Protection rate for certain audio segments.
            f0_method (str): Method for F0 extraction.
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            split_audio (bool): Whether to split the audio for processing.
            f0_autotune (bool): Whether to use F0 autotune.
            clean_audio (bool): Whether to clean the audio.
            clean_strength (float): Strength of the audio cleaning.
            export_format (str): Format for exporting the audio.
            f0_file (str): Path to the F0 file.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            seed: (int): Seed for randomization of noise.
            **kwargs: Additional keyword arguments.
        """
        if not model_path:
            get_console().print("No model provided. Aborting conversion.")
            return

        self.get_vc(model_path, sid, bundle_submodel)
        
        if not self.vc:
            get_console().print("Voice conversion pipeline not initialized. Check for model loading errors in the logs. Aborting conversion.")
            return

        try:
            start_time = time.time()
            get_console().print(f"Converting audio '{audio_input_path}'...")

            # Loading the input audio and downsample to 16khz
            audio = load_audio_infer(audio_input_path, 16000, **kwargs)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            # Load in the feature embedder model
            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
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
                get_console().print(f"Audio split into {len(chunks)} chunks for processing.")
            else:
                chunks = [audio]

            # Seed handling
            if seed != 0:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                get_console().print(f"[INFER] Seed specified: Inference is performed in deterministic mode using seed: {seed}")
            else:
                get_console().print(f"[INFER] Seed unspecified: Inference is performed in randomized mode.")
                seed = random.randint(0, 2**32 - 1) 
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                get_console().print(f"[INFER] Randomized seed exposed for reproduction: {seed}")


            # Collect chunked inference outputs ( if chunking's used )
            converted_chunks = []
            # Inference
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
                    version=self.version,
                    protect=protect,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    f0_file=f0_file,
                    seed=seed,
                    loaded_index=self.loaded_index,
                )
                converted_chunks.append(audio_opt)
                if split_audio:
                    get_console().print(f"Converted audio chunk {len(converted_chunks)}")

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
            get_console().print(
                f"Conversion completed! Result available in: '{audio_output_path}'. Time taken: {elapsed_time:.2f} seconds."
            )
        except Exception as error:
            get_console().print(f"An error occurred during audio conversion: {error}")
            get_console().print(traceback.format_exc())

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.

        Args:
            audio_input_paths (str): List of paths to the input audio files.
            audio_output_path (str): Path to the output audio file.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"), "w"
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            get_console().print(f"Converting audio batch '{audio_input_paths}'...")
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
            get_console().print(f"Detected {len(audio_files)} audio files for inference.")
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
            get_console().print(f"Conversion completed at '{audio_input_paths}'.")
            elapsed_time = time.time() - start_time
            get_console().print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
        except Exception as error:
            get_console().print(f"An error occurred during audio batch conversion: {error}")
            get_console().print(traceback.format_exc())
        finally:
            if os.path.exists(os.path.join(now_dir, "assets", "infer_pid.txt")):
                os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid, bundle_submodel=None):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int or str): Speaker ID or Speaker Name.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)

        bundle_models = get_bundle_models(self.cpt) if isinstance(self.cpt, dict) else {}
        if bundle_models:
            target_key = bundle_submodel

            if target_key and target_key in bundle_models:
                model_data = bundle_models[target_key]
                self.active_cpt = model_data["model_state"]

                self.loaded_index = None
                if "index_data" in model_data:
                    try:
                        self.loaded_index = faiss.deserialize_index(model_data["index_data"])
                    except Exception as e:
                        get_console().print(f"Failed to deserialize index: {e}")
            else:
                get_console().print(f"Sub-model '{bundle_submodel}' not found in the model bundle.")
                self.cleanup_model()
                return
        else:
            self.active_cpt = self.cpt

        if self.active_cpt is not None:
            self.setup_network()
            self.setup_vc_instance()
            self.loaded_model = weight_root
        else:
            self.vc = None
            self.loaded_model = None

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        import gc
        for attr in ("net_g", "n_spk", "vc", "hubert_model", "tgt_sr", "cpt", "active_cpt", "loaded_model", "loaded_index"):
            setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path. Handles .pth and model bundles.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = None
        self.loaded_index = None

        if not os.path.isfile(weight_root):
            get_console().print(f"Model file not found: {weight_root}")
            return
        
        if is_model_bundle(weight_root):
            get_console().print(f"[Infer] Loading compressed model bundle: {weight_root}")
            try:
                bundle_data = load_model_bundle(weight_root)

                # Check for new multi-model format
                if "models" in bundle_data:
                    self.cpt = bundle_data
                    get_console().print(f"[Infer] Loaded model bundle with {len(bundle_data['models'])} models.")
                # Backward compatibility for old single-model bundles
                else:
                    self.cpt = bundle_data.get("model_state")
                    serialized_index = bundle_data.get("index_data")
                    if serialized_index is not None:
                        try:
                            self.loaded_index = faiss.deserialize_index(serialized_index)
                            get_console().print("[Infer] Loaded and deserialized the bundle index.")
                        except Exception as e:
                            get_console().print(f"Failed to deserialize bundle index: {e}")
            except Exception as e:
                get_console().print(f"An error occurred loading the model bundle: {e}")
                self.cpt = None

        else:
            get_console().print(f"Loading .pth file: {weight_root}")
            self.cpt = torch.load(weight_root, map_location="cpu", weights_only=True)


    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.active_cpt is not None:
            self.tgt_sr = self.active_cpt["config"][-1]
            self.active_cpt["config"][-3] = self.active_cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = True

            self.version = self.active_cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = normalize_vocoder(
                self.active_cpt.get(
                    "vocoder_id",
                    self.active_cpt.get("vocoder_architecture", self.active_cpt.get("vocoder", "hifi")),
                )
            )

            synth_kwargs = {
                "use_f0": self.use_f0,
                "text_enc_hidden_dim": self.text_enc_hidden_dim,
                "vocoder": self.vocoder,
                "vocoder_config": self.active_cpt.get("vocoder_config", {}),
            }

            # Model init
            self.net_g = Synthesizer(*self.active_cpt["config"], **synth_kwargs)

            del self.net_g.enc_q # Posterior encoder is training-only

            self.net_g.load_state_dict(self.active_cpt["weight"], strict=False)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.active_cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.active_cpt["config"][-3]
