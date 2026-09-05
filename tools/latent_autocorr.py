"""What temporal shape does the decoder's ``z`` actually have in training?

At inference ``z_p = m_p + exp(logs_p) * randn * noise_scale`` draws an
independent ``randn`` per frame, so the perturbation is white in time by
construction: autocorrelation 0 at every lag.  In training ``z`` comes from the
posterior, whose input is a spectrogram of real speech.  If that one is
correlated in time, inference is handing the decoder a roughness it never saw;
if it is white too, there is no train/inference mismatch to fix and the skirt
is simply what a prior costs.

Measured on real utterances from the training filelist, per channel, on the
*deviation from the posterior mean* -- the thing ``randn`` stands in for --
and, separately, on ``z`` itself, which is what the decoder receives.

Usage::

    python tools/latent_autocorr.py --clips 64
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
sys.path.insert(0, str(ROOT / "tools"))

from rvc.lib.algorithm.commons import strip_parametrizations  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402
from rvc.train.mel_processing import spectrogram_torch  # noqa: E402
from render_constant_f0 import coarse  # noqa: E402

MAX_LAG = 10


def autocorr(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Per-channel normalised autocorrelation of (C, T), averaged over C.

    Each channel is centred on its own temporal mean first: a channel with a
    large constant offset would otherwise report ~1 at every lag and say
    nothing about how it moves.
    """
    x = x - x.mean(axis=1, keepdims=True)
    var = (x**2).mean(axis=1)
    keep = var > 1e-12
    x, var = x[keep], var[keep]
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = ((x[:, : -lag] * x[:, lag:]).mean(axis=1) / var).mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clips", type=int, default=64)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    model_cfg = dict(config["model"])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"]
    model_cfg["spk_embed_dim"] = int(weights["emb_g.weight"].shape[0])
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
    # The posterior is exactly what this measures, so it must NOT be dropped.
    net_g.load_state_dict(weights, strict=False)
    net_g = net_g.float().eval()
    strip_parametrizations(net_g)

    entries = []
    for line in (log_dir / "filelist.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        entries.append((parts[0], parts[1], parts[3], int(parts[-1])))
    rng = np.random.default_rng(0)
    picked = [entries[i] for i in rng.choice(len(entries), args.clips, replace=False)]

    post_dev, post_z, prior_dev, infer_dev = [], [], [], []
    ratios = {"posterior": [], "inference": []}
    with torch.no_grad():
        for wav_path, feat_path, f0_path, sid in picked:
            audio, sr = sf.read(wav_path, dtype="float32")
            if sr != data["sample_rate"]:
                raise SystemExit(f"{wav_path} is {sr} Hz, expected {data['sample_rate']}")
            y = torch.from_numpy(audio).unsqueeze(0)
            spec = spectrogram_torch(
                y,
                data["filter_length"],
                data["hop_length"],
                data["win_length"],
                center=False,
            )
            lengths = torch.tensor([spec.shape[-1]])
            g = net_g.emb_g(torch.tensor([sid])).unsqueeze(-1)
            z, m_q, logs_q, mask = net_g.enc_q(spec, lengths, g=g)
            if z.shape[-1] <= 2 * MAX_LAG:
                continue
            post_z.append(autocorr(z[0].numpy()))
            # z = m_q + exp(logs_q) * eps.  Recovering eps is what compares
            # like for like against the prior's randn.
            eps = ((z - m_q) / torch.exp(logs_q))[0].numpy()
            post_dev.append(autocorr(eps))
            prior_dev.append(autocorr(np.random.default_rng(1).standard_normal(eps.shape)))
            ratios["posterior"].append(
                float((z - m_q).std() / (m_q - m_q.mean(dim=-1, keepdim=True)).std())
            )

            # The same question on the inference path.  What the decoder gets
            # there is ``flow(z_p)``, not ``z_p``: the flow is a temporal
            # convolution and could colour a white perturbation on its way
            # through, so the deviation has to be measured after it.
            phone = torch.from_numpy(
                np.load(feat_path)
            ).float().unsqueeze(0)
            pitchf = torch.from_numpy(np.load(f0_path)).float().unsqueeze(0)
            n = min(phone.shape[1], pitchf.shape[1])
            phone, pitchf = phone[:, :n], pitchf[:, :n]
            pitch = torch.from_numpy(coarse(pitchf[0].numpy().copy())).unsqueeze(0)
            m_p, logs_p, x_mask = net_g.enc_p(
                phone=phone, pitch=pitch, lengths=torch.tensor([n])
            )
            torch.manual_seed(1234)
            z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
            z_inf = net_g.flow(z_p * x_mask, x_mask, g=g, reverse=True)
            z_ref = net_g.flow(m_p * x_mask, x_mask, g=g, reverse=True)
            if n > 2 * MAX_LAG:
                infer_dev.append(autocorr((z_inf - z_ref)[0].numpy()))
                ratios["inference"].append(
                    float(
                        (z_inf - z_ref).std()
                        / (z_ref - z_ref.mean(dim=-1, keepdim=True)).std()
                    )
                )

    clips = len(post_dev)
    infer_dev = np.mean(infer_dev, axis=0)
    post_dev = np.mean(post_dev, axis=0)
    post_z = np.mean(post_z, axis=0)
    prior_dev = np.mean(prior_dev, axis=0)

    print(f"checkpoint: {ckpt_path.name}   clips: {clips}   "
          f"frame rate: {data['sample_rate'] / data['hop_length']:g} Hz\n")
    print(f"{'lag (frames)':>12s} {'posterior eps':>14s} {'posterior z':>13s} "
          f"{'prior randn':>13s} {'infer dev':>11s}")
    for lag in range(MAX_LAG + 1):
        print(f"{lag:12d} {post_dev[lag]:14.3f} {post_z[lag]:13.3f} "
              f"{prior_dev[lag]:13.3f} {infer_dev[lag]:11.3f}")

    print(
        "\nstochastic part vs the mean it rides on (std ratio):"
        f"\n  posterior, training           {np.mean(ratios['posterior']):.3f}"
        f"\n  prior ns=0.667, after the flow {np.mean(ratios['inference']):.3f}"
    )


if __name__ == "__main__":
    main()
