"""Latent directions the decoder renders as bursts when the prior draw moves them.

The decoder reads a handful of directions of ``z`` as noise between the
harmonics.  A prior draw that moves them fills and empties those valleys over
hundreds of milliseconds, which is heard as bursts.  They are estimated from the
gradient of the valley energy over dataset clips and stored in the checkpoint as
``prior_noise_subspace``; ``Synthesizer.infer`` keeps the draw out of them.
"""

from __future__ import annotations

import random

import numpy as np
import torch

KEY = "prior_noise_subspace"
N_FFT, HOP = 2048, 160


def pick_clips(filelist: str, count: int = 24, seed: int = 0) -> list[list[str]]:
    """Mostly-voiced clips of at least 3 s, sampled from a training filelist."""
    with open(filelist, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    random.Random(seed).shuffle(lines)
    clips = []
    for line in lines:
        parts = line.split("|")
        try:
            pitchf = np.load(parts[3])
        except (IndexError, OSError, ValueError):
            continue
        if len(pitchf) >= 300 and (pitchf > 0).mean() > 0.6:
            clips.append(parts)
            if len(clips) == count:
                break
    return clips


def harmonic_masks(pitchf: np.ndarray, frames: int, sr: int, hop: int, device):
    """Bins between the partials and on them, 1-12 kHz, voiced frames only."""
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    df = freqs[1]
    valleys = np.zeros((len(freqs), frames), bool)
    partials = np.zeros((len(freqs), frames), bool)
    edge = int(0.2 * sr / HOP)
    for i in range(edge, frames - edge):
        f0 = pitchf[min(i * HOP // hop, len(pitchf) - 1)]
        if f0 <= 0:
            continue
        for k in range(max(1, int(1000 / f0)), int(12000 / f0)):
            c = int(round(k * f0 / df))
            if c + 2 >= len(freqs):
                break
            partials[c - 1 : c + 2, i] = True
            c = int(round((k + 0.5) * f0 / df))
            w = max(1, int(0.15 * f0 / df))
            valleys[c - w : c + w + 1, i] = True
    return torch.tensor(valleys, device=device), torch.tensor(partials, device=device)


def valley_gradient(net_g, parts: list[str], sr: int, hop: int, max_frames: int):
    """d log(valley / partial energy) / dz at the noiseless prior, [channels, frames]."""
    device = net_g.emb_g.weight.device
    phone = np.repeat(np.load(parts[1]), 2, axis=0)
    pitch, pitchf = np.load(parts[2]), np.load(parts[3])
    frames = min(len(phone), len(pitch), len(pitchf), max_frames)
    phone = torch.FloatTensor(phone[:frames]).unsqueeze(0).to(device)
    pitch_t = torch.LongTensor(pitch[:frames]).unsqueeze(0).to(device)
    pitchf_t = torch.FloatTensor(pitchf[:frames]).unsqueeze(0).to(device)
    sid = torch.LongTensor([int(parts[4])]).to(device)

    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)
        m_p, _, x_mask = net_g.enc_p(
            phone=phone, pitch=pitch_t, lengths=torch.LongTensor([frames]).to(device)
        )
        z0 = net_g.flow(m_p * x_mask, x_mask, g=g, reverse=True)

    z = z0.clone().requires_grad_(True)
    # The excitation draws noise; fixing it keeps the gradient about z alone.
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    audio = net_g.dec(z * x_mask, pitchf_t, g).squeeze()
    window = torch.hann_window(N_FFT, device=device)
    power = torch.stft(audio.float(), N_FFT, HOP, window=window, return_complex=True).abs().pow(2)
    valleys, partials = harmonic_masks(pitchf[:frames], power.shape[1], sr, hop, device)
    if not valleys.any():
        return None
    ratio = torch.log(torch.where(valleys, power, 0).sum() + 1e-9) - torch.log(
        torch.where(partials, power, 0).sum() + 1e-9
    )
    # Gradient with respect to z only, so no parameter's .grad is touched.
    (grad,) = torch.autograd.grad(ratio, z)
    return grad[0]


def estimate_prior_subspace(net_g, clips, sr: int, hop: int, rank: int = 16, max_frames: int = 900):
    """The top ``rank`` directions as a CPU [channels, rank] tensor, and the share they capture.

    Runs the model in eval mode and restores its mode and the global RNG state
    afterwards, so it can be called from inside a training loop.
    """
    device = net_g.emb_g.weight.device
    covariance = None
    was_training = net_g.training
    net_g.eval()
    devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            for parts in clips:
                grad = valley_gradient(net_g, parts, sr, hop, max_frames)
                if grad is None:
                    continue
                grad = grad.double()
                outer = grad @ grad.T
                # Normalised so a loud clip does not choose the directions alone.
                outer = outer / (outer.trace() + 1e-12)
                covariance = outer if covariance is None else covariance + outer
    finally:
        net_g.train(was_training)
    if covariance is None:
        return None, 0.0

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    rank = max(1, min(int(rank), eigenvectors.shape[1]))
    captured = float(eigenvalues[:rank].sum() / eigenvalues.sum())
    return eigenvectors[:, :rank].float().cpu().contiguous(), captured
