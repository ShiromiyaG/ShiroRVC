import json
import pathlib

import torch
from einops import rearrange

from .models import CFNaiveMelPE
from .tools import (
    DotDict,
    catch_none_args_must,
    catch_none_args_opti,
    get_config_json_in_same_path,
    get_device,
    spawn_wav2mel,
)
from .torch_interp import batch_interp_with_replacement_detach


def ensemble_f0(f0s, key_shift_list, tta_uv_penalty):
    """Combine multi-key-shift f0 estimates (B, T, len(key_shift_list)) into one (B, T, 1) via DP."""
    device = f0s.device
    # convert f0 to note
    f0s = f0s / (
        torch.pow(
            2,
            torch.tensor(key_shift_list, device=device)
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 12,
        )
    )
    notes = torch.log2(f0s / 440) * 12 + 69
    notes[notes < 0] = 0

    # select best note
    # 使用动态规划选择最优的音高
    # 惩罚1：uv的惩罚固定为超参数uv_penalty ** 2，v转为uv时额外惩罚两次
    # 惩罚2：相邻帧音高的L2距离（uv和v互转的过程除外），距离小于0.5时忽略不计
    uv_penalty = tta_uv_penalty**2
    dp = torch.zeros_like(notes, device=device)
    # dp[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾的最小惩罚
    backtrack = torch.zeros_like(notes, device=device).long()
    # backtrack[b,t,c]表示，对于样本b，0到第t帧的所有选择中，选择第c个f0作为第t帧的结尾时，t-1帧结尾的选择，值域为0到len(f0_list)-1
    # init
    dp[:, 0, :] = (notes[:, 0, :] <= 0) * uv_penalty
    # forward
    for t in range(1, notes.size(1)):
        penalty = torch.zeros(
            [notes.size(0), notes.size(2), notes.size(2)], device=device
        )
        # [b,c1,c2]表示第b个样本中，t-1帧选择c1，t帧选择c2的惩罚

        # t帧是uv的情况
        t_uv = notes[:, t, :] <= 0
        penalty += uv_penalty * t_uv.unsqueeze(1)

        # t帧是v的情况
        # t-1帧也是v的情况
        t1_uv = notes[:, t - 1, :] <= 0
        l2 = torch.pow(
            (notes[:, t - 1, :].unsqueeze(-1) - notes[:, t, :].unsqueeze(1))
            * (~t1_uv).unsqueeze(-1)
            * (~t_uv).unsqueeze(1),
            2,
        )
        l2 = l2 - 0.5
        l2 = l2 * (l2 > 0)
        penalty += l2

        # t-1帧是uv的情况，uv转v的惩罚
        penalty += t1_uv.unsqueeze(-1) * (~t_uv).unsqueeze(1) * uv_penalty * 2

        # 选择最小惩罚
        min_value, min_indices = torch.min(
            dp[:, t - 1, :].unsqueeze(-1) + penalty, dim=1
        )
        dp[:, t, :] = min_value
        backtrack[:, t, :] = min_indices

    # backtrack
    t = f0s.size(1) - 1
    f0_result = torch.zeros_like(f0s[:, :, 0], device=device)
    min_indices = torch.argmin(dp[:, t, :], dim=-1)
    for i in range(0, t + 1):
        f0_result[:, t - i] = f0s[:, t - i, min_indices]
        min_indices = backtrack[:, t - i, min_indices]

    return f0_result.unsqueeze(-1)


class InferCFNaiveMelPE(torch.nn.Module):
    """Inference wrapper around CFNaiveMelPE."""

    def __init__(self, args, state_dict):
        super().__init__()
        self.wav2mel = spawn_wav2mel(args, device="cpu")
        self.model = spawn_model(args)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.args_dict = dict(args)
        self.register_buffer(
            "tensor_device_marker", torch.tensor(1.0).float(), persistent=False
        )

    def forward(
        self,
        wav: torch.Tensor,
        sr: [int, float],
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
        key_shifts: list = [0],
    ) -> torch.Tensor:
        """Returns f0 in Hz, shape (B, n_sample//hop_size + 1, len(key_shifts))."""
        with torch.no_grad():
            wav = wav.to(self.tensor_device_marker.device)
            mels = torch.stack(
                [self.wav2mel(wav, sr, keyshift=keyshift) for keyshift in key_shifts],
                -1,
            )
            mels = rearrange(mels, "B T C K -> (B K) T C")
            f0s = self.model.infer(mels, decoder=decoder_mode, threshold=threshold)
            f0s = rearrange(f0s, "(B K) T 1 -> B T (K 1)", K=len(key_shifts))
        return f0s  # (B, T, len(key_shifts))

    def infer(
        self,
        wav: torch.Tensor,
        sr: [int, float],
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
        f0_min: float = None,
        f0_max: float = None,
        interp_uv: bool = False,
        output_interp_target_length: int = None,
        return_uv: bool = False,
        test_time_augmentation: bool = False,
        tta_uv_penalty: float = 12.0,
        tta_key_shifts: list = [0, -12, 12],
        tta_use_origin_uv=False,
    ) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        """f0 in Hz; also returns the uv mask if return_uv is True.

        test_time_augmentation infers at several key shifts and combines them via
        ensemble_f0 for a more robust (but slower) estimate.
        """
        if test_time_augmentation:
            assert len(tta_key_shifts) > 0
            flag = 0
            if tta_use_origin_uv:
                if 0 not in tta_key_shifts:
                    flag = 1
                    tta_key_shifts.append(0)
            tta_key_shifts.sort(key=lambda x: (x if x >= 0 else -x / 2))
            f0s = self.__call__(wav, sr, decoder_mode, threshold, tta_key_shifts)
            f0 = ensemble_f0(
                f0s[:, :, flag:],
                tta_key_shifts[flag:],
                tta_uv_penalty,
            )
            if tta_use_origin_uv:
                f0_for_uv = f0s[:, :, [0]]
            else:
                f0_for_uv = f0
        else:
            f0 = self.__call__(wav, sr, decoder_mode, threshold)
            f0_for_uv = f0
        if f0_min is None:
            f0_min = self.args_dict["model"]["f0_min"]
        uv = (f0_for_uv < f0_min).type(f0_for_uv.dtype)
        f0 = f0 * (1 - uv)
        if interp_uv:
            f0 = batch_interp_with_replacement_detach(
                uv.squeeze(-1).bool(), f0.squeeze(-1)
            ).unsqueeze(-1)
        if f0_max is not None:
            f0[f0 > f0_max] = f0_max
        if output_interp_target_length is not None:
            f0 = torch.where(f0 == 0, float("nan"), f0)
            f0 = torch.nn.functional.interpolate(
                f0.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="linear",
            ).transpose(1, 2)
            f0 = torch.where(f0.isnan(), float(0.0), f0)
        if return_uv:
            uv = torch.nn.functional.interpolate(
                uv.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
            return f0, uv
        else:
            return f0

    def get_hop_size(self) -> int:
        return DotDict(self.args_dict).mel.hop_size

    def get_hop_size_ms(self) -> float:
        return (
            DotDict(self.args_dict).mel.hop_size / DotDict(self.args_dict).mel.sr * 1000
        )

    def get_model_sr(self) -> int:
        return DotDict(self.args_dict).mel.sr

    def get_mel_config(self) -> dict:
        return dict(DotDict(self.args_dict).mel)

    def get_device(self) -> str:
        return self.tensor_device_marker.device

    def get_model_f0_range(self) -> dict:
        return {
            "f0_min": DotDict(self.args_dict).model.f0_min,
            "f0_max": DotDict(self.args_dict).model.f0_max,
        }


class InferCFNaiveMelPEONNX:
    """ONNX inference wrapper; not implemented."""

    def __init__(self, args, onnx_path, device):
        raise NotImplementedError


def spawn_bundled_infer_model(device: str = None) -> InferCFNaiveMelPE:
    """Load the model bundled with the package (pretrained, ready to use)."""
    file_path = pathlib.Path(__file__)
    model_path = file_path.parent / "assets" / "fcpe_c_v001.pt"
    model = spawn_infer_model_from_pt(str(model_path), device, bundled_model=True)
    return model


def spawn_infer_model_from_onnx(
    onnx_path: str, device: str = None
) -> InferCFNaiveMelPEONNX:
    device = get_device(device, "torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_onnx")
    config_path = get_config_json_in_same_path(onnx_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        args = DotDict(config_dict)
    if (args.is_onnx is None) or (args.is_onnx is False):
        raise ValueError(
            "  [ERROR] spawn_infer_model_from_onnx: this model is not onnx model."
        )

    if args.model.type == "CFNaiveMelPEONNX":
        infer_model = InferCFNaiveMelPEONNX(args, onnx_path, device)
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPEONNX"
        )

    return infer_model


def spawn_infer_model_from_pt(
    pt_path: str, device: str = None, bundled_model: bool = False
) -> InferCFNaiveMelPE:
    """bundled_model is only set True from spawn_bundled_infer_model."""
    device = get_device(device, "torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_pt")
    ckpt = torch.load(pt_path, map_location=torch.device(device))
    if bundled_model:
        ckpt["config_dict"]["model"]["conv_dropout"] = 0.0
        ckpt["config_dict"]["model"]["atten_dropout"] = 0.0
    args = DotDict(ckpt["config_dict"])
    if (args.is_onnx is not None) and (args.is_onnx is True):
        raise ValueError(
            "  [ERROR] spawn_infer_model_from_pt: this model is an onnx model."
        )

    if args.model.type == "CFNaiveMelPE":
        infer_model = InferCFNaiveMelPE(args, ckpt["model"])
        infer_model = infer_model.to(device)
        infer_model.eval()
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE"
        )

    return infer_model


def spawn_model(args: DotDict) -> CFNaiveMelPE:
    if args.model.type == "CFNaiveMelPE":
        pe_model = CFNaiveMelPE(
            input_channels=catch_none_args_must(
                args.mel.num_mels,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.mel.num_mels is None",
            ),
            out_dims=catch_none_args_must(
                args.model.out_dims,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.out_dims is None",
            ),
            hidden_dims=catch_none_args_must(
                args.model.hidden_dims,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.hidden_dims is None",
            ),
            n_layers=catch_none_args_must(
                args.model.n_layers,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.n_layers is None",
            ),
            n_heads=catch_none_args_must(
                args.model.n_heads,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.n_heads is None",
            ),
            f0_max=catch_none_args_must(
                args.model.f0_max,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.f0_max is None",
            ),
            f0_min=catch_none_args_must(
                args.model.f0_min,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.f0_min is None",
            ),
            use_fa_norm=catch_none_args_must(
                args.model.use_fa_norm,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.use_fa_norm is None",
            ),
            conv_only=catch_none_args_opti(
                args.model.conv_only,
                default=False,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.conv_only is None",
            ),
            conv_dropout=catch_none_args_opti(
                args.model.conv_dropout,
                default=0.0,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.conv_dropout is None",
            ),
            atten_dropout=catch_none_args_opti(
                args.model.atten_dropout,
                default=0.0,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.atten_dropout is None",
            ),
            use_harmonic_emb=catch_none_args_opti(
                args.model.use_harmonic_emb,
                default=False,
                func_name="torchfcpe.tools.spawn_cf_naive_mel_pe",
                warning_str="args.model.use_harmonic_emb is None",
            ),
        )
    else:
        raise ValueError(
            f"  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE"
        )
    return pe_model


def bundled_infer_model_unit_test(wav_path):
    try:
        import librosa
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  [UNIT_TEST] torchfcpe.tools.spawn_infer_model_from_pt: matplotlib or librosa not found, skip test"
        )
        exit(1)

    infer_model = spawn_bundled_infer_model(device="cpu")
    wav, sr = librosa.load(wav_path, sr=16000)
    f0 = infer_model.infer(torch.tensor(wav).unsqueeze(0), sr, interp_uv=False)
    f0_interp = infer_model.infer(torch.tensor(wav).unsqueeze(0), sr, interp_uv=True)
    plt.plot(f0.squeeze(-1).squeeze(0).numpy(), color="r", linestyle="-")
    plt.plot(f0_interp.squeeze(-1).squeeze(0).numpy(), color="g", linestyle="-")
    plt.legend(["f0", "f0_interp"])
    plt.xlabel("frame")
    plt.ylabel("f0")
    plt.title("f0")
    plt.show()
