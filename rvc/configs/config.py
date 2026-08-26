import torch
import os
import json

from rvc.configs.vocoders import get_vocoder_config_paths, normalize_vocoder
from rvc.lib.terminal import get_console


arch_config_paths = get_vocoder_config_paths()

# Training defaults to FP32 master weights with TF32 tensor cores: no autocast,
# no scaler, toggled per run from the training tab.
#
# FP16 is the one autocast mode offered, enabled from Settings -> Precision and
# carried into the run spec by the launcher.  It has the same 11-bit mantissa as
# TF32 (gradient cosine 0.99 against a true-FP32 reference) but a narrow
# exponent range, which is what the GradScaler is there for.  BF16 is not
# offered: an 8-bit mantissa measured a gradient cosine of only 0.77-0.82 on
# this model, whose oscillatory NSF source and L1-on-log-mel loss produce
# heavily cancelling sums.

#: Where the persisted FP16 preference lives.  Read defensively -- a missing or
#: hand-broken config must not stop the app from starting.
_APP_CONFIG_PATH = os.path.join("assets", "config.json")


def get_use_fp16() -> bool:
    """The persisted "train under FP16 autocast" preference."""
    try:
        with open(_APP_CONFIG_PATH, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("use_fp16", False))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        initial_precision = self.get_precision()

        get_console().log(
            f"[CONFIG] Running on {'CPU' if self.device == 'cpu' else 'CUDA'}, "
            f"training precision: {initial_precision}",
            markup=False,
        )
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )

        self.json_config = self.load_config_json("hifi")
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self, vocoder_arch="hifi"):
        vocoder_arch = normalize_vocoder(vocoder_arch)
        configs = {}
        for config_file in arch_config_paths.get(vocoder_arch, arch_config_paths["hifi"]):
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs


    def get_precision(self):
        return "fp16 (autocast)" if get_use_fp16() else "fp32"

    def check_precision(self, use_fp16=None):
        """Report the precision the next run would start with.

        ``use_fp16`` comes from the settings checkbox so the report matches what
        is on screen even before the change handler has written it; falling back
        to the persisted value keeps the method callable with no arguments.
        """
        if use_fp16 is None:
            use_fp16 = get_use_fp16()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        lines = [
            "Master weights and optimizer state: FP32 (always).",
            f"TF32 matmul/conv currently: {'on' if tf32 else 'off'}"
            " - toggle it per run in the Training tab.",
        ]
        if use_fp16:
            lines.append(
                "FP16 autocast: ON, with a GradScaler. Distribution math and the"
                " NSF source stay in FP32."
            )
            if not torch.cuda.is_available():
                lines.append(
                    "No CUDA device visible, so the setting will do nothing here."
                )
        else:
            lines.append("FP16 autocast: off - no autocast, no GradScaler.")
        lines.append(
            "BF16 is not offered: it lost too much mantissa here (gradient cosine"
            " 0.77-0.82 vs FP32; TF32 is 0.9997 and FP16 is 0.99)."
        )
        return "\n".join(lines)

    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max


    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024 ** 3)

def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "1"

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "Unfortunately, there is no compatible GPU available to support your training."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"


def microarchitecture_capability_checker():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8



