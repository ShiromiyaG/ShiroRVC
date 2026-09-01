import os
import sys
import soxr
import librosa
import ffmpeg
import soundfile as sf
import numpy as np
import wget
import pyloudnorm as pyln
import torch
import torch.nn.functional as F

from transformers import AutoFeatureExtractor

import logging
import warnings

from rvc.lib.text import format_title
from rvc.lib.terminal import info, warning

# Moved to a torch-free module so preprocessing's worker pool can import the
# loaders without paying for torch; re-exported here so every existing
# ``from rvc.lib.utils import load_audio`` keeps working.
from rvc.lib.audio_io import (  # noqa: F401
    load_audio,
    load_audio_16k,
    load_audio_ffmpeg,
)

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
            warning(
                "Input audio is stereo; converting to mono. Prefer mono input.",
                tag="[INFER]",
            )
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


from transformers import HubertModel
from torch import nn


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_embedder_model(embedder_model, custom_embedder=None):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin_v1": os.path.join(embedder_root, "spin_v1"),
        "spin_v2": os.path.join(embedder_root, "spin_v2"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            warning(
                f"Custom embedder not found at {custom_embedder}; using contentvec.",
                tag="[INFER]",
            )
            model_path = embedding_list["contentvec"]
    elif embedder_model == "spin_v1":
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
    elif embedder_model == "spin_v2":
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
    else:
        if embedder_model not in embedding_list:
            warning(
                f"Unknown embedder {embedder_model!r}; using contentvec.",
                tag="[INFER]",
            )
            embedder_model = "contentvec"
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders.get(embedder_model)
            if url is not None:
                info(f"Downloading the {embedder_model} weights.", tag="[INFER]")
                wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files.get(embedder_model)
            if url is not None:
                info(f"Downloading the {embedder_model} config.", tag="[INFER]")
                wget.download(url, out=json_file)

    model = HubertModelWithFinalProj.from_pretrained(model_path)

    do_normalize = False
    try:
        fe = AutoFeatureExtractor.from_pretrained(model_path)
        do_normalize = getattr(fe, "do_normalize", False)
    except Exception:
        pass

    return model, do_normalize


def extract_features(model, source, version, do_normalize=False):
    """v1 (256-D) takes layer-9 hidden states through final_proj; v2 (768-D) uses
    the last hidden state directly. do_normalize layer-norms the waveform first,
    matching what ContentVec/HuBERT expects.
    """
    if do_normalize:
        # Over the sample axis only.  ``source.shape`` normalised across the
        # whole tensor, which is the same thing for the (1, T) inputs this used
        # to get and silently wrong for a batch, where it would mix every clip
        # into every other clip's statistics.
        source = F.layer_norm(source, source.shape[-1:])

    if version == "v1":
        outputs = model(
            input_values=source,
            output_hidden_states=True,
            return_dict=True,
        )
        return model.final_proj(outputs.hidden_states[9])

    return model(
        input_values=source,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state
