import os
import torch
from collections import OrderedDict

from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.terminal import error as print_error, info, install_rich_print, success

install_rich_print()

deep_debug_merging = False # For dev or debugging purposes

def extract(ckpt):
    a = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {}
    for key in a.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = a[key]

    if deep_debug_merging:
        print(f"[DEBUG] extract() returning keys: {list(opt['weight'].keys())}")

    return opt

def model_blender(name, path1, path2, ratio):
    try:
        message = f"Model {path1} and {path2} are merged with alpha {ratio}."
        info(f"Blending '{path1}' and '{path2}' at alpha {ratio}.", tag="[BLEND]")

        # Load checkpoints
        ckpt1 = torch.load(path1, map_location="cpu", weights_only=True)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=True)

        if deep_debug_merging:
            print(f"[DEBUG] Loaded ckpt1 keys: {list(ckpt1.keys())}")
            print(f"[DEBUG] Loaded ckpt2 keys: {list(ckpt2.keys())}")

        # Check sample rate compatibility (normalize "48k" -> 48000)
        def _normalize_sr(v):
            return int(str(v).replace("k", "000")) if isinstance(v, str) else v
        sr1 = _normalize_sr(ckpt1["sr"])
        sr2 = _normalize_sr(ckpt2["sr"])
        if sr1 != sr2:
            err_msg = "The sample rates of the two models are not the same."
            print_error(err_msg, tag="[BLEND]")
            return err_msg, None

        # Retrieve configuration values
        cfg = ckpt1["config"]
        cfg_f0 = True
        cfg_version = ckpt1["version"]
        cfg_sr = sr1
        vocoder_id = normalize_vocoder(
            ckpt1.get(
                "vocoder_id",
                ckpt1.get("vocoder_architecture", ckpt1.get("vocoder", "hifi")),
            )
        )
        other_vocoder_id = normalize_vocoder(
            ckpt2.get(
                "vocoder_id",
                ckpt2.get("vocoder_architecture", ckpt2.get("vocoder", "hifi")),
            )
        )
        if vocoder_id != other_vocoder_id:
            err_msg = "The vocoder architectures of the two models are not the same."
            print_error(err_msg, tag="[BLEND]")
            return err_msg, None
        vocoder = get_vocoder_spec(vocoder_id)["label"]
        vocoder_config = ckpt1.get("vocoder_config", {})
        architecture_id = ckpt1.get(
            "architecture_id",
            vocoder_config.get(
                "architecture_id", get_architecture_id(vocoder_id)
            ),
        )
        other_architecture_id = ckpt2.get(
            "architecture_id",
            ckpt2.get("vocoder_config", {}).get(
                "architecture_id", get_architecture_id(other_vocoder_id)
            ),
        )
        if architecture_id != other_architecture_id:
            err_msg = "The model architecture revisions are not the same."
            print_error(err_msg, tag="[BLEND]")
            return err_msg, None

        # Extract models if needed
        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if deep_debug_merging:
            print(f"[DEBUG] ckpt1 model keys: {list(ckpt1.keys())}")
            print(f"[DEBUG] ckpt2 model keys: {list(ckpt2.keys())}")

        # Check model architecture compatibility
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            err_msg = "Fail to merge the models. The model architectures are not the same."
            print_error(err_msg, tag="[BLEND]")
            return err_msg, None

        # Blend model weights
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])

                if deep_debug_merging:
                    print(f"[DEBUG] Blending key '{key}' with different shapes, using min shape: {min_shape0}")

                opt["weight"][key] = (
                    ratio * (ckpt1[key][:min_shape0].float())
                    + (1 - ratio) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    ratio * (ckpt1[key].float())
                    + (1 - ratio) * (ckpt2[key].float())
                ).half()

            if deep_debug_merging:
                print(f"[DEBUG] Blended key '{key}': shape {opt['weight'][key].shape}")

        # Append additional configuration data
        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["info"] = message
        opt["vocoder"] = vocoder
        opt["vocoder_id"] = vocoder_id
        opt["vocoder_architecture"] = vocoder_id
        opt["vocoder_config"] = vocoder_config
        opt["architecture_id"] = architecture_id

        output_path = os.path.join("logs", f"{name}.pth")
        torch.save(opt, output_path)
        success(f"Blended model saved to '{output_path}'.", tag="[BLEND]")
        return message, output_path
    except Exception as blend_error:
        print_error(f"Model blending failed: {blend_error}", tag="[BLEND]")
        return str(blend_error), None
