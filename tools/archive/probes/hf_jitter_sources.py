"""Does the decoder trunk make the high-band jitter, and which loss asks for it?

Two parts, on the ``logs/reference`` clip, in the direct path:

``trunk``  over the longest voiced stretch, ``m_p`` held at its mean, with the
           real f0 or a constant one, excitation noise on or off.  With both
           inputs constant and no noise the render is periodic, so any band
           jitter left is the trunk's reaction to a moving f0.

``push``   the gradient each generator loss term sends into the rendered
           waveform (the mel loss, and each discriminator branch's adversarial
           plus feature-matching term with its training weight), projected on
           the gradient of the band jitter itself.  ``dJ/step`` is how the
           jitter moves for a unit step down that term's gradient: negative
           means the term pulls the jitter down; ``dL/step`` is the same for the
           band's mean level in dB.  ``share`` is the fraction of
           the term's gradient energy that lies in the band.

Usage::

    python archive/probes/hf_jitter_sources.py --log-dir logs/pretrain_bigvgan_teste
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "probes"))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

from decoder_determinism import build, reference_inputs  # noqa: E402
from energy_ab import BANDS, band_jitter  # noqa: E402
from rvc.lib.algorithm.energy import frame_energy  # noqa: E402
from rvc.lib.audio_io import load_audio  # noqa: E402

C_FM = 2.0


def longest_voiced_run(voiced: np.ndarray):
    best, start = (0, 0), None
    for i, v in enumerate(np.append(voiced, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def decode(net_g, m_p, pitchf, g, noise):
    dec = net_g.dec
    noise_std = dec.m_source.noise_std
    if not noise:
        dec.m_source.noise_std = 0.0
    try:
        with torch.no_grad():
            return dec(m_p, pitchf, g)[0, 0].float().cpu().numpy()
    finally:
        dec.m_source.noise_std = noise_std


def torch_jitter(wave, sr, hop, pair_mask, lo, hi):
    """``band_jitter`` for one band, differentiable in ``wave`` [samples]:
    the jitter and the band's mean level over voiced frames, both in dB."""
    n_fft = 1024
    spec = torch.stft(
        wave, n_fft, hop, n_fft, torch.hann_window(n_fft, device=wave.device),
        center=False, return_complex=True,
    )
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr, device=wave.device)
    band = (freqs >= lo) & (freqs < min(hi, sr / 2))
    power = spec.real.square() + spec.imag.square()
    db = 10 * torch.log10(power[band].sum(0) + 1e-12)
    diff = (db[1:] - db[:-1]).abs()
    n = min(diff.shape[0], pair_mask.shape[0])
    mask = pair_mask[:n]
    return diff[:n][mask].mean(), db[1 : n + 1][mask].mean()


def band_share(grad, sr, lo, hi):
    power = torch.fft.rfft(grad.double()).abs().square()
    freqs = torch.fft.rfftfreq(grad.shape[-1], 1.0 / sr)
    band = (freqs >= lo) & (freqs < hi)
    return float(power[band].sum() / power.sum().clamp_min(1e-30))


def trunk(args, net_g, sr, hop, phone, lengths, pitch, pitchf, sid, energy, wave):
    voiced = pitchf[0].numpy() > 0
    start, end = longest_voiced_run(voiced)
    start, end = start + args.margin, end - args.margin
    if end - start < 50:
        raise SystemExit("No voiced stretch long enough in the reference.")
    stretch = np.zeros_like(voiced)
    stretch[start:end] = True

    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)
        m_p, _logs_p, x_mask = net_g.encode_content(phone, lengths, pitch, energy)
    held = m_p.clone()
    held[..., start:end] = m_p[..., start:end].mean(dim=-1, keepdim=True)
    flat = pitchf.clone()
    flat[:, start:end] = pitchf[:, start:end].log().mean().exp()
    held, m_p = held * x_mask, m_p * x_mask

    # Measured inside the stretch only, where every case is defined.
    inner = stretch.copy()
    inner[: start + 10] = False
    inner[end - 10 :] = False
    cents = 1200 * np.abs(np.diff(np.log2(pitchf[0, start:end].numpy())))

    print(
        f"trunk: frames {start}-{end} ({(end - start) / 100:.2f} s), "
        f"f0 frame-to-frame {cents.mean():.1f} cents mean\n"
    )
    header = "  ".join(f"{lo // 1000 if lo >= 1000 else lo}-{hi // 1000}k".rjust(8) for lo, hi in BANDS)
    print(f"{'frame-to-frame |dE| (dB)':30s}{header}")
    print(f"{'original':30s}" + "  ".join(f"{j:8.2f}" for j in band_jitter(wave, sr, hop, inner)[0]))
    cases = [
        ("as preview", m_p, pitchf, True),
        ("m_p held, real f0", held, pitchf, True),
        ("m_p held, real f0, no noise", held, pitchf, False),
        ("m_p held, flat f0", held, flat, True),
        ("m_p held, flat f0, no noise", held, flat, False),
    ]
    for label, latent, f0, noise in cases:
        out = decode(net_g, latent, f0, g, noise)[: len(wave)]
        jitter, _ = band_jitter(out, sr, hop, inner)
        print(f"{label:30s}" + "  ".join(f"{j:8.2f}" for j in jitter))


def push(args, net_g, sr, hop, phone, lengths, pitch, pitchf, sid, energy, wave, log_dir, ckpt_path):
    from rvc.train.losses import feature_loss, generator_loss
    from rvc.train.mel_processing import build_ms_mel_loss
    from rvc.train.setup import get_d_model
    from rvc.train.utils import load_config_from_json

    config = load_config_from_json(str(log_dir / "config.json"))
    spec = json.loads((log_dir / "run_spec.json").read_text(encoding="utf-8"))
    net_d = get_d_model(config, spec["vocoder"], False)
    d_path = ckpt_path.with_name(ckpt_path.name.replace("G_", "D_"))
    net_d.load_state_dict(torch.load(d_path, map_location="cpu", weights_only=True)["model"])
    net_d = net_d.float().eval()
    for p in net_d.parameters():
        p.requires_grad_(False)
    labels = net_d.branch_labels
    weights = net_d.branch_weights
    san = bool(getattr(net_d, "supports_san", False))
    c_mel = float(config.train.c_mel)
    fn_mel = build_ms_mel_loss(sr)

    with torch.no_grad():
        o, *_ = net_g.infer(phone, lengths, pitch, pitchf, sid, energy=energy)
    n = min(o.shape[-1], len(wave))
    y_hat = o[..., :n].detach().clone().requires_grad_(True)
    y = torch.from_numpy(wave[:n]).view(1, 1, -1)

    voiced = pitchf[0].numpy() > 0
    shift = int(round(1024 / 2 / hop))
    v = voiced[shift:]
    pair_mask = torch.from_numpy(v[1:] & v[:-1])

    def grad_of(loss):
        (grad,) = torch.autograd.grad(loss, y_hat, retain_graph=True)
        return grad[0, 0]

    terms = [("mel (c_mel)", grad_of(fn_mel(y, y_hat) * c_mel))]
    y_d_r, y_d_g, fmap_r, fmap_g = net_d(y, y_hat, no_grad_real=True)
    for i, label in enumerate(labels):
        w = [weights[i]]
        adv = generator_loss([y_d_g[i]], use_softplus=san, branch_weights=w)
        fm = feature_loss([fmap_r[i]], [fmap_g[i]], branch_weights=w) * C_FM
        terms.append((label, grad_of(adv + fm)))
    total_d = sum(g for label, g in terms[1:])
    terms.append(("all D branches", total_d))

    bands = [(4000, 8000), (8000, 16000)]
    jitter_grads = []
    for lo, hi in bands:
        wave_var = y_hat[0, 0].detach().clone().requires_grad_(True)
        J, L = torch_jitter(wave_var, sr, hop, pair_mask, lo, hi)
        (gJ,) = torch.autograd.grad(J, wave_var, retain_graph=True)
        (gL,) = torch.autograd.grad(L, wave_var)
        jitter_grads.append((float(J.detach()), gJ, gL))

    print(f"push: {d_path.name}, branch weights as trained, SAN={'on' if san else 'off'}")
    print("  " + "   ".join(f"{lo // 1000}-{hi // 1000}k J={J:.2f} dB" for (lo, hi), (J, _, _) in zip(bands, jitter_grads)))
    print(f"\n{'term':18s}" + "".join(
        f"{f'{lo // 1000}-{hi // 1000}k dJ/step':>17s}{'dL/step':>11s}{'share':>7s}" for lo, hi in bands
    ) + f"{'|grad|':>11s}")
    for label, grad in terms:
        row = f"{label:18s}"
        for (lo, hi), (_, gJ, gL) in zip(bands, jitter_grads):
            # A unit step down the term's gradient: y -= grad.
            dj = -float((gJ.double() * grad.double()).sum())
            dl = -float((gL.double() * grad.double()).sum())
            row += f"{dj:17.2e}{dl:11.2e}{band_share(grad, sr, lo, hi):7.3f}"
        row += f"{float(grad.norm()):11.2e}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/pretrain_bigvgan_teste")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref-dir", default="logs/reference")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--margin", type=int, default=20, help="frames trimmed off each end of the stretch")
    ap.add_argument("--part", choices=("trunk", "push", "both"), default="both")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else max(log_dir.glob("G_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    )
    # Training saves the EMA as ``model``, so these are the preview's weights.
    net_g, sr = build(log_dir, ckpt_path, ema=False)
    if net_g.latent_mode != "direct":
        raise SystemExit(f"{ckpt_path.name} is not a direct-mode model.")
    hop = sr // 100

    ref_dir = Path(args.ref_dir)
    phone, lengths, pitch, pitchf = reference_inputs(ref_dir, None, net_g.content_interpolation)
    frames = int(lengths[0])
    wave = load_audio(str(ref_dir / "ref_audio.wav"), sr)[: frames * hop].astype(np.float32)
    energy = None
    if getattr(net_g, "energy_embedding", None) is not None:
        energy = frame_energy(torch.from_numpy(wave).view(1, -1), sr, frames)
    sid = torch.tensor([args.sid])
    print(f"checkpoint: {ckpt_path.name}   sr={sr}   frames={frames}\n")

    inputs = (net_g, sr, hop, phone, lengths, pitch, pitchf, sid, energy, wave)
    if args.part in ("trunk", "both"):
        trunk(args, *inputs)
        print()
    if args.part in ("push", "both"):
        push(args, *inputs, log_dir, ckpt_path)


if __name__ == "__main__":
    main()
