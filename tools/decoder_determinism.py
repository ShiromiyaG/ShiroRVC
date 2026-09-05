"""Is the decoder's output a function of its inputs, or is a noise source live?

Renders the same input twice *in one process without reseeding* and subtracts
the two waveforms.  Reseeding between the two renders would make any stochastic
module replay its draws and report a false zero, which is why this does not
call ``Synthesizer.infer`` (it seeds) and why the prior is taken at its mean
(``noise_scale=0``): a resampled ``z_p`` would differ for reasons that have
nothing to do with the decoder.

Two configurations, because two things in the decoder draw noise:

``noise_std``  the excitation's dither, added *before* the convolutions.
``AdaIN``      a per-channel gaussian added *inside* them, before the
               nonlinearity -- signal-shaped noise, the only candidate that
               could make a skirt that tracks partial level.

Running once with the shipped ``noise_std`` and once with it at zero separates
them: whatever is left over at ``noise_std=0`` is the AdaIN's.

Usage::

    python tools/decoder_determinism.py
    python tools/decoder_determinism.py --f0 220
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.commons import strip_parametrizations  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from render_constant_f0 import coarse  # noqa: E402


def apply_checkpoint_layout(model_cfg: dict, layout: dict | None) -> dict:
    """Make the config's decoder options say what the checkpoint was trained with.

    ``decoder_layout`` exists because none of these leave a trace in the state
    dict: wrapping an activation adds no key, and redesigning its filter adds
    none either.  A tool that builds from ``config.json`` alone therefore loads
    ``decoder_layout`` exists because none of these leave a trace in the state
    dict: the stage ordering keeps every key and shape, and the upsamplers'
    interpolation kernels are non-persistent, so redesigning one adds no key.
    A tool that builds from ``config.json`` alone therefore loads happily into
    the wrong signal path and measures a decoder that never existed.  Training
    guards this with ``assert_decoder_layout_matches``; analysis has to as
    well.
    """

    if not layout:
        return model_cfg
    cfg = dict(model_cfg)
    cfg["upsample_rates"] = list(layout["upsample_rates"])
    for key in ("source_gain", "source_bandwidth", "source_normalize"):
        if key in layout:
            cfg[f"refinegan2_{key}"] = layout[key]
    up = layout.get("upsample_filter")
    if up:
        cfg["refinegan2_filter_width"], cfg["refinegan2_rolloff"], \
            cfg["refinegan2_filter_beta"] = (list(v) for v in up)
    return cfg


def build(log_dir: Path, ckpt_path: Path, ema: bool):
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    model_cfg = dict(config["model"])
    data = config["data"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"]
    if ema:
        weights = {**weights, **(ckpt["ema"]["shadow"])}
    model_cfg["spk_embed_dim"] = int(weights["emb_g.weight"].shape[0])
    model_cfg = apply_checkpoint_layout(model_cfg, ckpt.get("decoder_layout"))

    info = json.loads((log_dir / "model_info.json").read_text(encoding="utf-8"))
    net_g = Synthesizer(
        data["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // data["hop_length"],
        **model_cfg,
        use_f0=True,
        sr=int(data["sample_rate"]),
        vocoder=info.get("vocoder_architecture", "refinegan2"),
        checkpointing=False,
    )
    net_g.load_state_dict(weights, strict=False)
    # ``rvc/train`` imports its neighbours flat, so it needs to be importable
    # as a directory before the guard can be reached.
    sys.path.insert(0, str(ROOT / "rvc" / "train"))
    from rvc.train.utils import assert_decoder_layout_matches

    assert_decoder_layout_matches(net_g, ckpt)
    net_g.remove_training_modules()
    net_g = net_g.float().eval()
    strip_parametrizations(net_g)
    return net_g, int(data["sample_rate"])


def reference_inputs(ref_dir: Path, f0: float | None):
    phone = np.repeat(np.load(ref_dir / "ref_feats.npy"), 2, axis=0)
    pitch = np.load(ref_dir / "ref_f0c.npy").astype(np.int64)
    pitchf = np.load(ref_dir / "ref_f0f.npy").astype(np.float32)
    n = min(len(phone), len(pitch), len(pitchf))
    phone, pitch, pitchf = phone[:n], pitch[:n], pitchf[:n]
    if f0 is not None:
        pitchf = np.full(n, float(f0), dtype=np.float32)
        pitch = coarse(pitchf.copy())
    return (
        torch.from_numpy(phone).float().unsqueeze(0),
        torch.tensor([n]),
        torch.from_numpy(pitch).long().unsqueeze(0),
        torch.from_numpy(pitchf).float().unsqueeze(0),
    )


def render(net_g, phone, lengths, pitch, pitchf, sid):
    """One deterministic-by-construction pass: prior at its mean, no seeding."""
    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)
        m_p, _logs_p, x_mask = net_g.enc_p(
            phone=phone, pitch=pitch, lengths=lengths
        )
        z = net_g.flow(m_p * x_mask, x_mask, g=g, reverse=True)
        return net_g.dec(z * x_mask, pitchf, g)[0, 0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument(
        "--f0", type=float, default=None, help="flatten the reference's pitch track"
    )
    ap.add_argument("--save-diff", default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    net_g, sr = build(log_dir, ckpt_path, args.ema)
    phone, lengths, pitch, pitchf = reference_inputs(Path(args.ref_dir), args.f0)
    sid = torch.tensor([args.sid])

    source = net_g.dec.m_source
    shipped = source.noise_std
    print(f"checkpoint: {ckpt_path.name}   frames={lengths.item()}   sr={sr}")
    print(f"f0: {'flat %g Hz' % args.f0 if args.f0 else 'reference track'}")
    print(f"training mode: {net_g.training}\n")

    for label, noise_std in (("noise_std=%g (shipped)" % shipped, shipped), ("noise_std=0", 0.0)):
        source.noise_std = float(noise_std)
        a = render(net_g, phone, lengths, pitch, pitchf, sid)
        b = render(net_g, phone, lengths, pitch, pitchf, sid)
        d = a - b
        identical = np.array_equal(a, b)
        rms_a = float(np.sqrt((a**2).mean()))
        rms_d = float(np.sqrt((d**2).mean()))
        rel = (
            "-inf"
            if rms_d == 0
            else f"{20 * math.log10(rms_d / rms_a):.1f} dB"
        )
        print(f"{label:28s} bit-identical={identical}")
        print(
            f"{'':28s} max|a-b|={np.abs(d).max():.3e}   "
            f"rms(a-b) vs rms(a) = {rel}"
        )
        if args.save_diff and not identical:
            out = Path(args.save_diff)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, d, sr)
            print(f"{'':28s} diff written to {out}")
    source.noise_std = shipped


if __name__ == "__main__":
    main()
