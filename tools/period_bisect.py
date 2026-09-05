"""Where does a periodic input stop producing a periodic output?

With the conditioning frozen to one repeated frame and f0 flat, the decoder is
periodically time-varying with the hop (320 samples) and its excitation has
period 32000/220 = 145.4545 samples.  Eleven pitch periods are 1600 samples,
which is exactly five hops, so a deterministic forward *must* repeat every 1600
output samples.  Anything that does not is state that should not be there.

The test runs at every stage's own rate: 1600 output samples are 5 frames, so
the period is ``5 * rate_per_frame`` -- 1600, 400, 100, 25, 5 down the stack.

The first row that leaves -inf is the culprit.

Usage::

    python tools/period_bisect.py
    python tools/period_bisect.py --f0 220 --frames 600
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from decoder_determinism import build  # noqa: E402
from render_constant_f0 import coarse, pick_feature  # noqa: E402

PERIOD_FRAMES = 5  # 1600 output samples = 11 pitch periods at 220 Hz = 5 hops


def residual_db(x: np.ndarray, period: int, guard: int = 0) -> float:
    """Level of ``x[n] - x[n+period]`` against ``x``, over the interior."""
    if x.size <= 2 * period + 2 * guard:
        return float("nan")
    a = x[guard : x.size - period - guard]
    b = x[guard + period : x.size - guard]
    d = a - b
    ref = math.sqrt((a**2).mean())
    rms = math.sqrt((d**2).mean())
    return -math.inf if rms == 0 else 20 * math.log10(rms / ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--f0", type=float, default=220.0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--freeze-frame", type=int, default=60)
    ap.add_argument("--feature", default=None)
    ap.add_argument("--sid", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="prior sampling scale; the default freezes z, which is what "
        "'frozen conditioning' has to mean here -- tiling the content frame "
        "alone leaves z redrawn every frame",
    )
    ap.add_argument(
        "--guard",
        type=int,
        default=60,
        help="frames dropped at each end.  Not cosmetic: the 64-tap resamplers "
        "run at 1 sample per frame, so the padding transient reaches ~40 "
        "frames in and a smaller guard measures the edge, not the steady state",
    )
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    net_g, sr = build(log_dir, ckpt_path, args.ema)
    dec = net_g.dec
    dec.m_source.noise_std = float(args.noise_std)

    feature_path, sid = pick_feature(log_dir, args.feature)
    phone = np.load(feature_path)
    index = args.freeze_frame % len(phone)
    frames = args.frames
    phone = np.repeat(phone[index : index + 1], frames, axis=0)

    phone_t = torch.from_numpy(phone).float().unsqueeze(0)
    lengths = torch.tensor([frames])
    nsff0 = np.full(frames, float(args.f0), dtype=np.float32)
    pitch = torch.from_numpy(coarse(nsff0.copy())).unsqueeze(0)
    pitchf = torch.from_numpy(nsff0).unsqueeze(0)
    sid_t = torch.tensor([args.sid if args.sid is not None else sid])

    # ---------------------------------------------------------------- inputs
    print(f"checkpoint: {ckpt_path.name}  frames={frames}  f0={args.f0:g} Hz  "
          f"noise_std={args.noise_std:g}  noise_scale={args.noise_scale:g}  "
          f"guard={args.guard} frames")
    assert torch.equal(phone_t, phone_t[:, :1].expand_as(phone_t)), "phone not tiled"
    assert torch.equal(pitch, pitch[:, :1].expand_as(pitch)), "coarse pitch not constant"
    assert torch.equal(pitchf, pitchf[:, :1].expand_as(pitchf)), "f0 not constant"
    assert float(pitchf.min()) == args.f0 and float(pitchf.max()) == args.f0
    print(f"inputs: phone tiled OK, pitch={int(pitch[0,0])} constant, "
          f"f0 exactly {float(pitchf[0,0]):.6f} Hz, sid={int(sid_t)}")

    with torch.no_grad():
        g = net_g.emb_g(sid_t).unsqueeze(-1)
        assert g.shape[-1] == 1, "speaker embedding is not a single vector"
        m_p, _logs_p, x_mask = net_g.enc_p(
            phone=phone_t, pitch=pitch, lengths=lengths
        )
        z_p = m_p
        if args.noise_scale:
            torch.manual_seed(1234)
            z_p = m_p + torch.exp(_logs_p) * torch.randn_like(m_p) * args.noise_scale
        z = net_g.flow(z_p * x_mask, x_mask, g=g, reverse=True) * x_mask

        # f0 after the decoder's own expansion: the uv gate is nearest and the
        # log path clamps at 1.0, so a single zeroed frame would open and close
        # the gate and break periodicity on its own.
        f0_up = dec._expand_f0(pitchf.unsqueeze(1), frames * dec.upp)
        print(
            f"expanded f0: min={float(f0_up.min()):.6f} max={float(f0_up.max()):.6f} "
            f"unique={int(torch.unique(f0_up).numel())}"
        )

        def spread(name, t):
            """Frame-to-frame spread of a (B, C, T) conditioning tensor."""
            ref = t[:, :, :1]
            dev = (t - ref).abs().max().item()
            rel = dev / (t.abs().max().item() + 1e-12)
            print(f"{name:14s} shape={tuple(t.shape)}  max|x - x[:,:,0]| = "
                  f"{dev:.3e}  ({100 * rel:.2f}% of peak)")

        spread("m_p", m_p)
        spread("z", z)

        # ------------------------------------------------------------ stages
        captured: list[tuple[str, torch.Tensor]] = []

        def grab(name):
            def hook(_module, _inputs, output):
                captured.append((name, output.detach()))
            return hook

        handles = [dec.pre_conv.register_forward_hook(grab("pre_conv"))]
        for i, block in enumerate(dec.downsample_blocks):
            handles.append(block.register_forward_hook(grab(f"down[{i}]")))
        for i, block in enumerate(dec.upsample_conv_blocks):
            handles.append(block.register_forward_hook(grab(f"up[{i}]")))
        handles.append(dec.conv_post.register_forward_hook(grab("conv_post")))
        handles.append(dec.m_source.register_forward_hook(grab("har_source")))

        out = dec(z, pitchf, g)
        for handle in handles:
            handle.remove()

    captured.sort(key=lambda item: item[1].shape[-1] == 0)
    stages = [("f0 expanded", f0_up), ("z (decoder input)", z)]
    stages += [(name, t) for name, t in captured]
    stages.append(("output", out))

    print(f"\n{'stage':22s} {'rate/frame':>10s} {'period':>8s}   residual vs signal")
    for name, tensor in stages:
        length = tensor.shape[-1]
        rate = length / frames
        if abs(rate - round(rate)) > 1e-9:
            print(f"{name:22s} {rate:10.3f} {'-':>8s}   (not frame-aligned)")
            continue
        rate = int(round(rate))
        period = PERIOD_FRAMES * rate
        # Average the per-channel residual in power, dropping ``--guard``
        # frames of edge at each end.
        guard = args.guard * rate
        x = tensor[0].double().numpy()
        num = den = 0.0
        for channel in x:
            if channel.size <= 2 * period + 2 * guard:
                continue
            a = channel[guard : channel.size - period - guard]
            b = channel[guard + period : channel.size - guard]
            num += ((a - b) ** 2).sum()
            den += (a**2).sum()
        db = "-inf" if num == 0 else f"{10 * math.log10(num / den):.1f} dB"
        print(f"{name:22s} {rate:10d} {period:8d}   {db:>10s}")


if __name__ == "__main__":
    main()
