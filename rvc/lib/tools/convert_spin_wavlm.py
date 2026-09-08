"""Turn Lyery's SPIN WavLM 512 checkpoint into something ``transformers`` loads.

The published weights are a PyTorch Lightning training state -- optimiser slots,
a logger reference and the SPIN heads alongside the encoder -- so they cannot be
handed to ``WavLMModel.from_pretrained`` as they are.  This downloads that
checkpoint once, checks it is the one this recipe was written against, renames
its encoder tensors onto the ``transformers`` layout, and writes the encoder,
the projection head and a small ``spin_config.json`` beside each other.

Ported from Lyery's fork, https://github.com/redpanda343/redpanda-rvc; the
weights themselves are Lyery's, at https://huggingface.co/lyery/spin-wavlm512.

The conversion is pinned hard on purpose.  A checkpoint with different SPIN
hyper-parameters -- another encoder layer, another cluster count, no L2 norm --
would convert without complaint and produce features that do not match what
``spin_config.json`` then claims they are, and the mismatch would only show up
as a model that trains badly.  So the SHA-256 is checked before the load and
the recipe is validated after it.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile

import requests
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import WavLMConfig, WavLMModel


EXPECTED_SHA256 = "bf9faa02f07a2904153e29583fe044d7974cc9bd46e183080b0b21e9777f21af"
EXPECTED_SIZE = 835355303
SOURCE_FILENAME = "epoch=0-step=5000.ckpt"
SOURCE_URL = "https://huggingface.co/lyery/spin-wavlm512/resolve/main/epoch%3D0-step%3D5000.ckpt"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "spin_config.json",
    "spin_projection.safetensors",
)


class WandbLogger:
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def converted_bundle_is_current(output_path):
    if any(
        not os.path.isfile(os.path.join(output_path, name))
        for name in REQUIRED_FILES
    ):
        return False
    try:
        with open(
            os.path.join(output_path, "spin_config.json"), "r", encoding="utf-8"
        ) as config_file:
            spin_config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return False
    return spin_config.get("source_checkpoint_sha256") == EXPECTED_SHA256


def download_checkpoint(destination):
    temporary_path = destination + ".part"
    try:
        print(f"Downloading SPIN WavLM 512 from {SOURCE_URL}...")
        with requests.get(
            SOURCE_URL,
            stream=True,
            headers={"Accept-Encoding": "identity"},
            timeout=(10, 120),
        ) as response:
            response.raise_for_status()
            bytes_written = 0
            with open(temporary_path, "wb") as checkpoint_file:
                with tqdm(
                    total=EXPECTED_SIZE,
                    unit="iB",
                    unit_scale=True,
                    desc="Downloading SPIN WavLM 512",
                ) as progress:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        checkpoint_file.write(chunk)
                        bytes_written += len(chunk)
                        progress.update(len(chunk))
        if bytes_written != EXPECTED_SIZE:
            raise IOError(
                f"Incomplete SPIN checkpoint: expected {EXPECTED_SIZE} bytes, "
                f"received {bytes_written}"
            )
        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != EXPECTED_SHA256:
            raise IOError(
                f"Unexpected checkpoint SHA-256: {actual_sha256}. "
                f"Expected {EXPECTED_SHA256}."
            )
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(path):
    actual_sha256 = file_sha256(path)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected checkpoint SHA-256: {actual_sha256}. Expected {EXPECTED_SHA256}."
        )
    safe_types = [(WandbLogger, "pytorch_lightning.loggers.wandb.WandbLogger")]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint, actual_sha256


def validate_recipe(checkpoint):
    recipe = checkpoint["hyper_parameters"]
    encoder = recipe["model"]["encoder"]
    prediction = recipe["model"]["pred_head"]
    loss = recipe["model"]["loss"]
    expected = {
        "encoder_type": (encoder["type"], "WavLM"),
        "use_layer": (encoder["use_layer"], 12),
        "freeze_layers": (encoder["freeze_layers"], ["pos", 0, 1, 2, 3, 4]),
        "projection": (prediction["hid_dims"], [256]),
        "clusters": (loss["num_vars"], 512),
        "l2_norm": (loss["l2_norm"], True),
        "sample_rate": (recipe["data"]["sample_rate"], 16000),
    }
    invalid = [name for name, values in expected.items() if values[0] != values[1]]
    if invalid:
        raise ValueError("Unexpected SPIN recipe values: " + ", ".join(invalid))


def wavlm_config():
    return WavLMConfig(
        activation_dropout=0.0,
        apply_spec_augment=True,
        attention_dropout=0.1,
        conv_bias=False,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        do_stable_layer_norm=False,
        feat_extract_norm="group",
        feat_proj_dropout=0.1,
        hidden_act="gelu",
        hidden_dropout=0.1,
        hidden_size=768,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        layerdrop=0.05,
        mask_feature_prob=0.0,
        mask_time_prob=0.05,
        max_bucket_distance=800,
        num_attention_heads=12,
        num_buckets=320,
        num_conv_pos_embedding_groups=16,
        num_conv_pos_embeddings=128,
        num_hidden_layers=12,
    )


def map_encoder_key(source_key):
    key = source_key.removeprefix("encoder.model.")
    if key == "mask_emb":
        return "masked_spec_embed"
    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}"
    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    if key.startswith("layer_norm."):
        return key.replace("layer_norm.", "feature_projection.layer_norm.")
    if key.startswith("encoder.pos_conv.0."):
        suffix = key.removeprefix("encoder.pos_conv.0.")
        suffix = {
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
        }.get(suffix, suffix)
        return f"encoder.pos_conv_embed.conv.{suffix}"
    match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if match:
        layer, suffix = match.groups()
        suffix = suffix.replace("self_attn.grep_a", "attention.gru_rel_pos_const")
        suffix = suffix.replace("self_attn.grep_linear", "attention.gru_rel_pos_linear")
        suffix = suffix.replace(
            "self_attn.relative_attention_bias", "attention.rel_attn_embed"
        )
        suffix = suffix.replace("self_attn.", "attention.")
        suffix = suffix.replace("self_attn_layer_norm.", "layer_norm.")
        suffix = suffix.replace("fc1.", "feed_forward.intermediate_dense.")
        suffix = suffix.replace("fc2.", "feed_forward.output_dense.")
        return f"encoder.layers.{layer}.{suffix}"
    if key.startswith("encoder.layer_norm."):
        return key
    raise KeyError(f"Unsupported encoder tensor: {source_key}")


def convert(checkpoint_path, output_path):
    checkpoint, checkpoint_sha256 = load_checkpoint(checkpoint_path)
    validate_recipe(checkpoint)
    source = checkpoint["state_dict"]
    encoder_source = {
        key: value for key, value in source.items() if key.startswith("encoder.model.")
    }
    converted = {map_encoder_key(key): value for key, value in encoder_source.items()}

    model = WavLMModel(wavlm_config())
    model.load_state_dict(converted, strict=True)
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    projection = {
        "weight": source["pred_head.layers.0.weight"].contiguous(),
        "bias": source["pred_head.layers.0.bias"].contiguous(),
    }
    save_file(projection, os.path.join(output_path, "spin_projection.safetensors"))

    spin_config = {
        "model_type": "spin-wavlm",
        "source": SOURCE_URL,
        "source_checkpoint_sha256": checkpoint_sha256,
        "sample_rate": 16000,
        "feature_hop_samples": 320,
        "encoder_layer": 12,
        "encoder_dim": 768,
        "feature_dim": 256,
        "feature_output": "spin_projection",
        "clusters": 512,
        "l2_normalize": True,
        "audio_requires_normalization": False,
    }
    with open(
        os.path.join(output_path, "spin_config.json"), "w", encoding="utf-8"
    ) as config_file:
        json.dump(spin_config, config_file, indent=2)
        config_file.write("\n")

    preprocessor_config = {
        "do_normalize": False,
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }
    with open(
        os.path.join(output_path, "preprocessor_config.json"),
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(preprocessor_config, config_file, indent=2)
        config_file.write("\n")
    print(
        f"Converted SPIN WavLM 512 to {output_path} with SHA-256 "
        f"{checkpoint_sha256}."
    )


def ensure_converted(output_path):
    output_path = os.path.abspath(output_path)
    if converted_bundle_is_current(output_path):
        return
    parent_path = os.path.dirname(output_path)
    os.makedirs(parent_path, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".spin-wavlm-512-", dir=parent_path
    ) as temporary_directory:
        checkpoint_path = os.path.join(temporary_directory, SOURCE_FILENAME)
        converted_path = os.path.join(temporary_directory, "converted")
        download_checkpoint(checkpoint_path)
        convert(checkpoint_path, converted_path)
        if not converted_bundle_is_current(converted_path):
            raise RuntimeError("Converted SPIN WavLM 512 bundle failed validation")
        os.makedirs(output_path, exist_ok=True)
        for name in REQUIRED_FILES:
            os.replace(
                os.path.join(converted_path, name), os.path.join(output_path, name)
            )
    if not converted_bundle_is_current(output_path):
        raise RuntimeError("SPIN WavLM 512 installation failed validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to spin_wavlm_512.ckpt")
    parser.add_argument("output", help="Destination embedder directory")
    args = parser.parse_args()
    convert(os.path.abspath(args.checkpoint), os.path.abspath(args.output))


if __name__ == "__main__":
    main()
