import datetime
import hashlib
import json
import os
import sys
from collections import OrderedDict

import torch

from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.algorithm.synthesizers import vocoder_config_from_model
from rvc.lib.terminal import error as print_error, success

now_dir = os.getcwd()
sys.path.append(now_dir)


def replace_keys_in_dict(d, old_key_part, new_key_part):
    if isinstance(d, OrderedDict):
        updated_dict = OrderedDict()
    else:
        updated_dict = {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part)
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def extract_model(
    ckpt,
    sr,
    name,
    model_path,
    epoch,
    step,
    hps,
    vocoder,
    architecture,
    pitch_guidance=True,
    version="v2",
):
    try:
        architecture = "RVC"
        vocoder_id = normalize_vocoder(vocoder)
        vocoder_spec = get_vocoder_spec(vocoder_id)
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        dataset_length = None
        embedder_model = None
        speakers_id = 1
        if os.path.exists(os.path.join(model_dir, "model_info.json")):
            with open(os.path.join(model_dir, "model_info.json"), "r") as f:
                data = json.load(f)
                dataset_length = data.get("total_dataset_duration", None)
                embedder_model = data.get("embedder_model", None)
                speakers_id = data.get("speakers_id", 1)
                vocoder_architecture = normalize_vocoder(
                    data.get("vocoder_architecture", vocoder_id)
                )
        else:
            dataset_length = None
            vocoder_architecture = vocoder_id

        vocoder_architecture = vocoder_id

        with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
            data = json.load(f)
            model_author = data.get("model_author", None)

        def is_training_only_key(key):
            return "enc_q" in key

        opt = OrderedDict(
            weight={
                key: value.half()
                for key, value in ckpt.items()
                if not is_training_only_key(key)
            }
        )

        # Base configuration list
        config_list = [
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        # Assigning to opt config
        opt["config"] = config_list

        # Whatever ``Synthesizer`` does not name in its own signature: the
        # frontend, decoder and discriminator options the checkpoint has to
        # carry so ``infer`` can rebuild the same model.
        vocoder_config = vocoder_config_from_model(dict(hps.model.items()))


        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = True
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = f"{name}-{epoch}-{step}-{sr}-{version}-{opt['config']}"
        opt["model_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["dataset_length"] = dataset_length
        opt["model_name"] = name
        opt["author"] = model_author
        opt["embedder_model"] = embedder_model
        opt["speakers_id"] = speakers_id
        opt["vocoder"] = vocoder_spec["label"]
        opt["vocoder_id"] = vocoder_id
        opt["vocoder_architecture"] = vocoder_architecture
        opt["vocoder_config"] = vocoder_config
        opt["architecture_id"] = vocoder_config.get(
            "architecture_id", get_architecture_id(vocoder_id)
        )

        # Since fork uses new API for weight norm ( parametrizations )
        # and mainline RVC ( Original ), W-okada and such rely on old API, we're performing keys conversion.
        #
        #   Old API:  .weight_g / .weight_v
        #   New API:  .parametrizations.weight.original0 (direction) / .original1 (gain)

        NEW_TO_OLD = [
            (".parametrizations.weight.original1", ".weight_g"),
            (".parametrizations.weight.original0", ".weight_v"),
        ]
        OLD_TO_NEW = [
            (".weight_g", ".parametrizations.weight.original1"),
            (".weight_v", ".parametrizations.weight.original0"),
        ]

        has_new = any("parametrizations.weight.original" in k for k in opt)
        has_old = any(k.endswith(".weight_v") or k.endswith(".weight_g") for k in opt)

        if architecture == "RVC" and has_new: # RVC Arch models TRAIN with new api, but are SAVED with old-API compatibility in mind.
            for old, new in NEW_TO_OLD:
                opt = replace_keys_in_dict(opt, old, new)

        elif architecture != "RVC" and has_old: # Fork arch doesn't use old API but this is a safety-fallback for whatever
            for old, new in OLD_TO_NEW:
                opt = replace_keys_in_dict(opt, old, new)

        torch.save(opt, model_path)
        success(
            f"Saved '{os.path.basename(model_path)}' (epoch {epoch}, step {step}).",
            tag="[EXPORT]",
        )

    except Exception as error:
        print_error(f"Could not export the model: {error}", tag="[EXPORT]")
