"""Measure how close a pretrain's training step comes to the FP16 range.

Runs the generator and discriminator steps of ``train.py`` under autocast on a
few real batches, without updating anything, and records per module the
largest forward activation and the largest gradient flowing back into it.
Reports the modules nearest 65504 (the FP16 maximum), and the largest
GradScaler scale the backward tolerates.

    python tools/pretrain/fp16_headroom.py logs/pretrain/pretrain_G.pth \\
        --discriminator logs/pretrain/pretrain_D.pth
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)
_train_dir = os.path.join(now_dir, "rvc", "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from data_utils import (  # noqa: E402
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)
from losses import (  # noqa: E402
    BandWeightedSpectralLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    mel_frequency_tilt_weights,
)
from mel_processing import build_ms_mel_loss  # noqa: E402
from utils import load_config_from_json  # noqa: E402

from rvc.lib.algorithm import commons  # noqa: E402
from rvc.train.setup import apply_precision_policy, get_d_model, get_g_model  # noqa: E402
from tools.pretrain._staged_pretrain import model_dir, resolve_vocoder  # noqa: E402

FP16_MAX = 65504.0
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


class RangeProbe:
    """Per-module max |activation| and max |grad wrt output|, by dtype."""

    def __init__(self):
        self.fwd = defaultdict(float)
        self.bwd = defaultdict(float)
        self.nonfinite_fwd = defaultdict(int)
        self.nonfinite_bwd = defaultdict(int)
        self.dtype = {}
        self.handles = []

    def attach(self, prefix: str, model: torch.nn.Module):
        for name, module in model.named_modules():
            if not name or "parametrizations" in name:
                continue
            self.handles.append(
                module.register_forward_hook(self._forward_hook(f"{prefix}.{name}"))
            )

    def detach(self):
        for handle in self.handles:
            handle.remove()

    def _forward_hook(self, name):
        def hook(_module, _inputs, output):
            tensors = output if isinstance(output, (list, tuple)) else (output,)
            for tensor in tensors:
                if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                    continue
                self._record(self.fwd, self.nonfinite_fwd, name, tensor)
                self.dtype[name] = tensor.dtype
                if tensor.requires_grad:
                    tensor.register_hook(self._backward_hook(name))

        return hook

    def _backward_hook(self, name):
        def hook(grad):
            self._record(self.bwd, self.nonfinite_bwd, name, grad)

        return hook

    @staticmethod
    def _record(peaks, nonfinite, name, tensor):
        tensor = tensor.detach()
        finite = torch.isfinite(tensor)
        bad = int((~finite).sum())
        if bad:
            nonfinite[name] += bad
            tensor = torch.where(finite, tensor, torch.zeros_like(tensor))
        peaks[name] = max(peaks[name], float(tensor.abs().max()) if tensor.numel() else 0.0)


def build_spectral_loss(config, device):
    tilt = float(getattr(config.train, "mel_frequency_tilt", 0.0))
    max_ratio = float(getattr(config.train, "mel_frequency_tilt_max_ratio", 8.0))
    if tilt == 0.0:
        distance = torch.nn.L1Loss()
    else:
        def weights(num_mels):
            return mel_frequency_tilt_weights(
                num_mels=num_mels,
                sample_rate=config.data.sample_rate,
                mel_fmin=config.data.mel_fmin,
                mel_fmax=config.data.mel_fmax,
                tilt=tilt,
                max_ratio=max_ratio,
            )

        distance = BandWeightedSpectralLoss(
            torch.nn.L1Loss(reduction="none"),
            weights(config.data.n_mel_channels).to(device),
            weight_factory=weights,
        ).to(device)
    return build_ms_mel_loss(config.data.sample_rate, loss_fn=distance).to(device)


def load_weights(net, path, label):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        print(f"{label}: {len(missing)} missing, {len(unexpected)} unexpected keys "
              f"(e.g. {(missing or unexpected)[:3]})")
    return blob


def report(title, peaks, nonfinite, dtypes, top):
    print(f"\n{title}")
    rows = sorted(peaks.items(), key=lambda item: -item[1])
    fp16_rows = [(n, v) for n, v in rows if dtypes.get(n) == torch.float16]
    if fp16_rows:
        name, value = fp16_rows[0]
        print(f"  largest in an FP16 tensor: {value:.4g} ({name}), "
              f"{FP16_MAX / max(value, 1e-30):.3g}x headroom")
    print(f"  {'max |x|':>11}  {'dtype':>8}  {'nonfinite':>9}  module")
    for name, value in rows[:top]:
        dtype = str(dtypes.get(name, "")).replace("torch.", "")
        flag = "  <-- over FP16" if value > FP16_MAX else ""
        print(f"  {value:11.4g}  {dtype:>8}  {nonfinite.get(name, 0):9d}  {name}{flag}")
    for name, count in nonfinite.items():
        if name not in dict(rows[:top]):
            print(f"  {'':11}  {'':8}  {count:9d}  {name} (non-finite)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generator", type=Path)
    parser.add_argument("--discriminator", type=Path, default=None)
    parser.add_argument("--model", default=None, help="run folder (default: the checkpoint's)")
    parser.add_argument("--precision", choices=DTYPES, default="fp16")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    directory = model_dir(args.model) if args.model else args.generator.parent
    config = load_config_from_json(str(directory / "config.json"))
    config.data.training_files = str(directory / "filelist.txt")
    vocoder = resolve_vocoder(directory, None)
    amp_dtype = DTYPES[args.precision]
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    g_blob = torch.load(args.generator, map_location="cpu", weights_only=True)
    config.model.spk_embed_dim = int(g_blob["model"]["emb_g.weight"].shape[0])
    del g_blob
    net_g = get_g_model(config, config.data.sample_rate, vocoder, False)
    load_weights(net_g, args.generator, "G")
    net_g = net_g.to(device).train()
    apply_precision_policy(net_g, amp_dtype)

    net_d = None
    if args.discriminator is not None:
        net_d = get_d_model(config, vocoder, False)
        load_weights(net_d, args.discriminator, "D")
        net_d = net_d.to(device).train()
    san_active = bool(getattr(net_d, "supports_san", False))
    branch_weights = (
        tuple(net_d.branch_weights)
        if net_d is not None and getattr(net_d, "uses_branch_weights", False)
        else None
    )
    san_direction_weight = max(
        0.0, min(1.0, float(getattr(config.train, "san_direction_weight", 0.25)))
    )

    dataset = TextAudioLoaderMultiNSFsid(config.data, n_mel_bins=config.model.inter_channels)
    collate = TextAudioCollateMultiNSFsid(
        pad_multiple=int(getattr(config.data, "pad_multiple", 32)),
        hop_length=config.data.hop_length,
    )
    fn_spectral = build_spectral_loss(config, device)
    c_mel = float(config.train.c_mel)
    c_kl = float(config.train.c_kl)
    segment = config.train.segment_size

    probe_g, probe_d = RangeProbe(), RangeProbe()
    probe_g.attach("G", net_g)
    if net_d is not None:
        probe_d.attach("D", net_d)

    autocast = lambda: torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                      enabled=amp_dtype is not None)
    param_peaks = defaultdict(float)
    losses = defaultdict(list)
    print(f"{args.generator.name}: {vocoder}, {args.precision}, "
          f"{args.batches} x {args.batch_size} clips from {directory}")

    for _ in range(args.batches):
        indices = random.sample(range(len(dataset)), args.batch_size)
        batch = [t.to(device) for t in collate([dataset[i] for i in indices])]
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, _, sid = batch

        with autocast():
            y_hat, ids_slice, _, z_mask, vae_parts = net_g(
                spec, spec_lengths, sid, phone, phone_lengths, pitchf, pitch
            )
            z, z_p, m_p, logs_p, m_q, logs_q = vae_parts
            y = commons.slice_segments(y, ids_slice * config.data.hop_length, segment, dim=3)

        if net_d is not None:
            net_d.zero_grad(set_to_none=True)
            with autocast():
                outputs = net_d(y, y_hat.detach(), san_training=san_active, combine_inputs=True)
                loss_disc = discriminator_loss(
                    outputs[0], outputs[1],
                    san_direction_weight=san_direction_weight,
                    branch_weights=branch_weights,
                )[0]
            loss_disc.backward()
            losses["disc"].append(loss_disc.item())
            del outputs
            net_d.requires_grad_(False)

        with autocast():
            loss = fn_spectral(y, y_hat) * c_mel
            losses["spectral"].append(loss.item())
            if net_d is not None:
                _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat, no_grad_real=True)
                loss_fm = feature_loss(fmap_r, fmap_g, branch_weights=branch_weights) * 2.0
                loss_adv = generator_loss(
                    y_d_hat_g,
                    san_direction_weight=san_direction_weight,
                    use_softplus=san_active,
                    branch_weights=branch_weights,
                )
                loss = loss + loss_fm + loss_adv
                losses["fm"].append(loss_fm.item())
                losses["adv"].append(loss_adv.item())
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl
            losses["kl"].append(loss_kl.item())
            loss = loss + loss_kl

        net_g.zero_grad(set_to_none=True)
        loss.backward()
        if net_d is not None:
            net_d.requires_grad_(True)

        for name, child in net_g.named_children():
            grads = [p.grad for p in child.parameters() if p.grad is not None]
            if grads:
                norm = float(torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
                ))
                param_peaks[name] = max(param_peaks[name], norm)
        del loss, y_hat, vae_parts

    probe_g.detach()
    probe_d.detach()

    print("\nlosses (mean over batches): " + ", ".join(
        f"{k} {sum(v) / len(v):.4g}" for k, v in losses.items()
    ))
    print("generator grad norm per submodule (max over batches): " + ", ".join(
        f"{k} {v:.4g}" for k, v in sorted(param_peaks.items(), key=lambda kv: -kv[1])
    ))

    dtypes = {**probe_g.dtype, **probe_d.dtype}
    report("Forward activations, generator", probe_g.fwd, probe_g.nonfinite_fwd, dtypes, args.top)
    if net_d is not None:
        report("Forward activations, discriminator", probe_d.fwd, probe_d.nonfinite_fwd, dtypes, args.top)
    backward = {**probe_g.bwd, **probe_d.bwd}
    report("Gradient w.r.t. module outputs (unscaled loss)", backward,
           {**probe_g.nonfinite_bwd, **probe_d.nonfinite_bwd}, dtypes, args.top)

    fp16_bwd = [v for n, v in backward.items() if dtypes.get(n) == torch.float16]
    if fp16_bwd and max(fp16_bwd) > 0:
        ceiling = FP16_MAX / max(fp16_bwd)
        print(f"\nLargest GradScaler scale before the backward overflows: "
              f"{ceiling:.3g} (2^{math.log2(ceiling):.1f}); train.py starts at 2^10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
