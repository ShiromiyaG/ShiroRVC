"""Render one clip from a training checkpoint with a *constant* F0.

The content features come from a real extracted clip (so the decoder gets a
plausible latent); only the pitch track is replaced by a flat line.  That is
the render an aliasing/imaging question needs: with a steady fundamental,
every partial sits on an exact multiple of it, and anything that does not is
the decoder's own.

Usage::

    python tools/render_constant_f0.py --f0 220
    python tools/render_constant_f0.py --f0 110 220 440 --seconds 3 --ema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.commons import strip_parametrizations  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

F0_MIN, F0_MAX = 50.0, 1100.0


def coarse(f0: np.ndarray) -> np.ndarray:
    """The 255-bucket mel quantisation the pitch embedding was trained on."""
    mel_min = 1127 * np.log(1 + F0_MIN / 700)
    mel_max = 1127 * np.log(1 + F0_MAX / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64)


def pick_feature(log_dir: Path, wanted: str | None):
    lines = (log_dir / "filelist.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.split("|")
        if wanted is None or wanted in parts[1]:
            return Path(parts[1]), int(parts[-1])
    raise SystemExit(f"no filelist entry matching {wanted!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--checkpoint", default=None, help="default: newest G_*.pth")
    ap.add_argument("--f0", type=float, nargs="+", default=[220.0])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--feature", default=None, help="substring of a filelist feature path")
    ap.add_argument("--sid", type=int, default=None, help="override the clip's speaker")
    ap.add_argument("--ema", action="store_true", help="use the EMA weights")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--freeze-frame",
        type=int,
        default=None,
        help="tile this content frame across the whole segment (frozen conditioning)",
    )
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=0.66666,
        help="prior sampling scale; 0 freezes the latent to the prior mean",
    )
    ap.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="override the excitation's voiced-region dither (default 0.003)",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    out_dir = Path(args.out_dir or (log_dir / "constant_f0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    model_cfg = dict(config["model"])
    data = config["data"]
    sr = int(data["sample_rate"])
    hop = int(data["hop_length"])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"]
    if args.ema:
        shadow = (ckpt.get("ema") or {}).get("shadow")
        if not shadow:
            raise SystemExit("checkpoint carries no EMA shadow")
        weights = {**weights, **shadow}

    if "emb_g.weight" in weights:
        model_cfg["spk_embed_dim"] = int(weights["emb_g.weight"].shape[0])

    info = json.loads((log_dir / "model_info.json").read_text(encoding="utf-8"))
    vocoder = info.get("vocoder_architecture", "refinegan2")

    net_g = Synthesizer(
        data["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // hop,
        **model_cfg,
        use_f0=True,
        sr=sr,
        vocoder=vocoder,
        checkpointing=False,
    )
    missing, unexpected = net_g.load_state_dict(weights, strict=False)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    net_g.remove_training_modules()
    net_g = net_g.float().eval()
    strip_parametrizations(net_g)

    if args.noise_std is not None:
        source = getattr(getattr(net_g, "dec", None), "m_source", None)
        if source is None or not hasattr(source, "noise_std"):
            raise SystemExit("this decoder has no excitation noise to set")
        print(f"noise_std: {source.noise_std} -> {args.noise_std}")
        source.noise_std = float(args.noise_std)

    feature_path, sid = pick_feature(log_dir, args.feature)
    phone = np.load(feature_path)
    want = max(1, int(round(args.seconds * sr / hop)))
    if args.freeze_frame is None:
        frames = min(len(phone), want)
        phone = phone[:frames]
        print(f"content: {feature_path.name}  frames={frames}  speaker={sid}")
    else:
        # Frozen conditioning: one content frame repeated.  Nothing in the
        # input then moves at a speech rate, so any 0-40 Hz modulation left in
        # the render is the generator's own and not the utterance's envelope.
        index = args.freeze_frame % len(phone)
        frames = want
        phone = np.repeat(phone[index : index + 1], frames, axis=0)
        print(
            f"content: {feature_path.name} frame {index} tiled  "
            f"frames={frames}  speaker={sid}"
        )

    phone_t = torch.from_numpy(phone).float().unsqueeze(0)
    lengths = torch.tensor([frames])
    sid_t = torch.tensor([args.sid if args.sid is not None else sid])

    for value in args.f0:
        nsff0 = np.full(frames, float(value), dtype=np.float32)
        pitch = torch.from_numpy(coarse(nsff0.copy())).unsqueeze(0)
        with torch.no_grad():
            # ``Synthesizer.infer`` hardcodes the 0.66666 prior scale; this is
            # the same path with that scale exposed, so the latent can be
            # frozen to the prior mean as well as the conditioning.
            torch.manual_seed(args.seed)
            g = net_g.emb_g(sid_t).unsqueeze(-1)
            m_p, logs_p, x_mask = net_g.enc_p(
                phone=phone_t, pitch=pitch, lengths=lengths
            )
            z_p = m_p
            if args.noise_scale:
                z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * args.noise_scale
            z = net_g.flow(z_p * x_mask, x_mask, g=g, reverse=True)
            audio = net_g.dec(
                z * x_mask, torch.from_numpy(nsff0).unsqueeze(0), g
            )
        wav = audio[0, 0].numpy()
        tag = "_ema" if args.ema else ""
        if args.freeze_frame is not None:
            tag += f"_frozen{args.freeze_frame}"
        if args.noise_scale != 0.66666:
            tag += f"_ns{args.noise_scale:g}"
        if args.noise_std is not None:
            tag += f"_n{args.noise_std:g}"
        name = f"{ckpt_path.stem}{tag}_f0_{value:g}Hz.wav"
        sf.write(out_dir / name, wav, sr)
        print(f"  {name}  {wav.size / sr:.2f}s  peak={np.abs(wav).max():.3f}")


if __name__ == "__main__":
    main()
