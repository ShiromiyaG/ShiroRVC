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
#: Smallest pass worth running, and the share of the free memory a pass plans to
#: take.
MIN_FRAMES = 150
BUDGET = 0.6
#: Peak device bytes one clip-frame of a pass costs, measured on the first pass
#: and kept as the largest seen.  Measured rather than assumed, and read before
#: the pass rather than after: Windows spills to host RAM instead of raising, so
#: a pass that does not fit has to be refused, never caught.
_bytes_per_frame = None


def free_bytes(device) -> int:
    """Device bytes a new allocation can have."""
    free, _ = torch.cuda.mem_get_info(device)
    # The allocator's own free blocks are usable without asking the driver.
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def plan_pass(device, batch_size: int, max_frames: int) -> tuple[int, int]:
    """Largest ``(clips, frames)`` within the caps that fits, or ``(0, 0)``.

    Until a pass has been measured the plan is the floor, one clip of
    ``MIN_FRAMES``, which is what the measurement is taken on.
    """
    if device.type != "cuda":
        return batch_size, max_frames
    if _bytes_per_frame is None:
        return 1, min(max_frames, MIN_FRAMES)
    budget = BUDGET * free_bytes(device) / _bytes_per_frame
    frames = min(max_frames, int(budget))
    if frames < MIN_FRAMES:
        return 0, 0
    return max(1, min(batch_size, int(budget // frames))), frames


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


#: Masks by ``(pitch file, frames, sr, hop)``.  They are a function of the
#: clip, and the clips are picked once per process, while the estimate runs
#: again at every checkpoint.
_MASK_CACHE: dict = {}


def harmonic_masks(pitchf: np.ndarray, frames: int, sr: int, hop: int, device, key=None):
    """Bins between the partials and on them, 1-12 kHz, voiced frames only."""
    cached = _MASK_CACHE.get((key, frames, sr, hop)) if key is not None else None
    if cached is None:
        freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
        df = freqs[1]
        bins = len(freqs)
        edge = int(0.2 * sr / HOP)
        index = np.arange(edge, max(edge, frames - edge))
        f0 = pitchf[np.minimum(index * HOP // hop, len(pitchf) - 1)]
        voiced = f0 > 0
        index, f0 = index[voiced], f0[voiced]

        valleys = np.zeros((bins, frames), bool)
        partials = np.zeros((bins, frames), bool)
        if len(f0):
            # ``k`` runs to the highest partial any frame reaches; the per-frame
            # range is a mask over it rather than a loop bound.
            orders = np.arange(1, max(2, int(12000 / f0.min()) + 1))[None, :]
            centres = np.rint(orders * f0[:, None] / df).astype(np.int64)
            low = np.maximum(1, (1000 / f0[:, None]).astype(np.int64))
            high = (12000 / f0[:, None]).astype(np.int64)
            wanted = (orders >= low) & (orders < high)
            # ``break``, not ``continue``: once a partial runs off the top of
            # the spectrum, neither it nor anything above it is marked.
            wanted &= np.cumprod(centres + 2 < bins, axis=1).astype(bool)
            rows = np.repeat(index[:, None], orders.shape[1], axis=1)

            def mark(target, centre, offsets, keep):
                for offset in offsets:
                    taken = keep & (centre + offset >= 0) & (centre + offset < bins)
                    target[(centre + offset)[taken], rows[taken]] = True

            mark(partials, centres, range(-1, 2), wanted)

            between = np.rint((orders + 0.5) * f0[:, None] / df).astype(np.int64)
            width = np.maximum(1, (0.15 * f0 / df).astype(np.int64))
            spread = range(-int(width.max()), int(width.max()) + 1)
            for offset in spread:
                taken = (
                    wanted
                    & (np.abs(offset) <= width[:, None])
                    & (between + offset >= 0)
                    & (between + offset < bins)
                )
                valleys[(between + offset)[taken], rows[taken]] = True
        cached = (valleys, partials)
        if key is not None:
            _MASK_CACHE[(key, frames, sr, hop)] = cached

    valleys, partials = cached
    return torch.tensor(valleys, device=device), torch.tensor(partials, device=device)


def valley_gradients(net_g, batch: list[list[str]], sr: int, hop: int, max_frames: int):
    """d log(valley / partial energy) / dz at the noiseless prior, per clip.

    Returns the gradients and the frame count they were taken at, which is what
    the pass actually cost.  One batched pass: each clip's ratio reads only its
    own ``z``, so the gradient of their sum is each clip's own gradient.  The
    batch is cropped to its shortest clip, which is what lets them share a pass
    at all.
    """
    device = net_g.emb_g.weight.device
    loaded = []
    for parts in batch:
        phone = np.repeat(np.load(parts[1]), 2, axis=0)
        pitch, pitchf = np.load(parts[2]), np.load(parts[3])
        loaded.append((parts, phone, pitch, pitchf))
    frames = min(
        [max_frames] + [min(len(p), len(c), len(f)) for _, p, c, f in loaded]
    )
    phone = torch.FloatTensor(np.stack([p[:frames] for _, p, _, _ in loaded])).to(device)
    pitch_t = torch.LongTensor(np.stack([c[:frames] for _, _, c, _ in loaded])).to(device)
    pitchf_t = torch.FloatTensor(np.stack([f[:frames] for _, _, _, f in loaded])).to(device)
    sid = torch.LongTensor([int(parts[4]) for parts, *_ in loaded]).to(device)
    lengths = torch.full((len(loaded),), frames, dtype=torch.long, device=device)

    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)
        m_p, _, x_mask = net_g.enc_p(phone=phone, pitch=pitch_t, lengths=lengths)
        z0 = net_g.flow(m_p * x_mask, x_mask, g=g, reverse=True)

    z = z0.clone().requires_grad_(True)
    # The excitation draws noise; fixing it keeps the gradient about z alone.
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    audio = net_g.dec(z * x_mask, pitchf_t, g).squeeze(1)
    window = torch.hann_window(N_FFT, device=device)
    power = torch.stft(
        audio.float(), N_FFT, HOP, window=window, return_complex=True
    ).abs().pow(2)

    ratios, wanted = [], []
    for index, (parts, _, _, pitchf) in enumerate(loaded):
        valleys, partials = harmonic_masks(
            pitchf[:frames], power.shape[-1], sr, hop, device, key=parts[3]
        )
        if not valleys.any():
            continue
        item = power[index]
        ratios.append(
            torch.log(torch.where(valleys, item, 0).sum() + 1e-9)
            - torch.log(torch.where(partials, item, 0).sum() + 1e-9)
        )
        wanted.append(index)
    if not ratios:
        return [], frames
    # Gradient with respect to z only, so no parameter's .grad is touched.
    (grad,) = torch.autograd.grad(torch.stack(ratios).sum(), z)
    return [grad[index] for index in wanted], frames


def valley_gradient(net_g, parts: list[str], sr: int, hop: int, max_frames: int):
    """``valley_gradients`` for one clip, or ``None`` when it has no voiced bins."""
    grads, _ = valley_gradients(net_g, [parts], sr, hop, max_frames)
    return grads[0] if grads else None


def charge(bytes_per_frame: float) -> None:
    """Records what a clip-frame costs, keeping the worst seen."""
    global _bytes_per_frame
    if _bytes_per_frame is None or bytes_per_frame > _bytes_per_frame:
        _bytes_per_frame = bytes_per_frame


def charge_failure(device, clips: int, frames: int) -> None:
    """Doubles the cost after a pass that missed, so the next plan halves it."""
    charge(2.0 * (_bytes_per_frame or free_bytes(device) / max(1, clips * frames)))


def measured_gradients(net_g, batch: list[list[str]], sr: int, hop: int, frames: int):
    """``valley_gradients``, charging the peak it allocated to a clip-frame."""
    device = net_g.emb_g.weight.device
    if device.type != "cuda":
        return valley_gradients(net_g, batch, sr, hop, frames)[0]
    # Nothing else in training reads the peak counters.
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    grads, used = valley_gradients(net_g, batch, sr, hop, frames)
    charge((torch.cuda.max_memory_allocated(device) - before) / max(1, len(batch) * used))
    return grads


def estimate_prior_subspace(
    net_g,
    clips,
    sr: int,
    hop: int,
    rank: int = 16,
    max_frames: int = 900,
    batch_size: int = 4,
):
    """The top ``rank`` directions as a CPU [channels, rank] tensor, and the share they capture.

    Runs the model in eval mode and restores its mode and the global RNG state
    afterwards, so it can be called from inside a training loop.  ``batch_size``
    clips share a pass, cropped to the shortest of them.  It runs beside a
    training step's memory, so ``batch_size`` and ``max_frames`` are caps rather
    than sizes: each pass is planned from what the device has free and what the
    measured passes cost.  Returns ``(None, 0.0)`` when not even the floor fits,
    which is the caller's cue to keep its previous basis.
    """
    device = net_g.emb_g.weight.device
    covariance = None
    was_training = net_g.training
    net_g.eval()
    devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            index = 0
            while index < len(clips):
                count, frames = plan_pass(
                    device, max(1, int(batch_size)), int(max_frames)
                )
                if not count:
                    break
                batch = clips[index : index + count]
                try:
                    grads = measured_gradients(net_g, batch, sr, hop, frames)
                except RuntimeError as error:
                    # Only a card that raises gets here, and only because the
                    # plan was optimistic; charging double re-plans it smaller.
                    if "out of memory" not in str(error).lower():
                        raise
                    torch.cuda.empty_cache()
                    charge_failure(device, len(batch), frames)
                    continue
                index += len(batch)
                for grad in grads:
                    grad = grad.double()
                    outer = grad @ grad.T
                    # Normalised so a loud clip does not choose the directions alone.
                    outer = outer / (outer.trace() + 1e-12)
                    covariance = outer if covariance is None else covariance + outer
    finally:
        net_g.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if covariance is None:
        return None, 0.0

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    rank = max(1, min(int(rank), eigenvectors.shape[1]))
    captured = float(eigenvalues[:rank].sum() / eigenvalues.sum())
    return eigenvectors[:, :rank].float().cpu().contiguous(), captured
