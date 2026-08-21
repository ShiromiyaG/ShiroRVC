import torch
import os
import json

from rvc.configs.vocoders import get_vocoder_config_paths, normalize_vocoder
from rvc.lib.terminal import get_console


arch_config_paths = get_vocoder_config_paths()

# Training is FP32-only.  FP16 needed a GradScaler and was the source of the
# AMP instability; BF16 has the range but only an 8-bit mantissa, which
# measured a gradient cosine of 0.77-0.82 against a true-FP32 reference (TF32
# and FP16, both 11-bit, sit at 0.9997 and 0.99).  TF32 already gives tensor
# cores at 11 bits with no autocast and no scaler, and it is toggled per run
# from the training tab rather than here.

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
        return "fp32"

    def check_precision(self):
        tf32 = torch.backends.cuda.matmul.allow_tf32
        return "\n".join(
            (
                "Training precision: FP32 (master weights and optimizer in FP32).",
                f"TF32 matmul/conv currently: {'on' if tf32 else 'off'}"
                " - toggle it per run in the Training tab.",
                "No autocast, no GradScaler: FP16 and BF16 training were removed.",
                "BF16 lost too much mantissa here (gradient cosine 0.77-0.82 vs"
                " FP32; TF32 is 0.9997).",
            )
        )

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



