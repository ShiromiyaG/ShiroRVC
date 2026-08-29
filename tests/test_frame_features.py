"""The training and inference paths must measure the same audio identically.

These features are the kind of change that cannot fail loudly: a mismatched
window, hop or band between the trainer and the inference pipeline produces a
model that trains fine and sounds subtly wrong, with nothing in any metric to
point at.  So the contract is pinned here rather than left to review.
"""

import numpy as np
import pytest
import torch

from rvc.lib.algorithm.commons import rand_slice_segments
from rvc.lib.algorithm.frame_features import (
    BAND_LIMIT_HZ,
    conditioning_channels,
    frame_conditioning,
    frame_energy,
    frame_periodicity,
    matched_n_fft,
    spectrogram_for_features,
)

TRAIN_SR, TRAIN_NFFT, TRAIN_HOP = 44100, 2048, 441
INFER_SR, INFER_HOP = 16000, 160


def _tone(sample_rate, seconds, f0, harmonics=40, noise=0.0, seed=0):
    generator = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    wave = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if k * f0 >= sample_rate / 2:
            break
        wave += np.sin(2 * np.pi * k * f0 * t + generator.uniform(0, 2 * np.pi)) / k
    wave /= np.abs(wave).max() + 1e-9
    if noise:
        wave = (1 - noise) * wave + noise * generator.standard_normal(len(t)) * 0.3
    return torch.from_numpy(wave.astype(np.float32))


def test_matched_n_fft_holds_hz_per_bin():
    assert matched_n_fft(TRAIN_SR) == TRAIN_NFFT
    inferred = matched_n_fft(INFER_SR)
    assert abs(INFER_SR / inferred - TRAIN_SR / TRAIN_NFFT) < 0.1


def test_periodicity_separates_tone_from_noise():
    f0 = 160.0
    for sample_rate, hop in ((TRAIN_SR, TRAIN_HOP), (INFER_SR, INFER_HOP)):
        n_fft = matched_n_fft(sample_rate)
        harmonic = _tone(sample_rate, 1.0, f0, noise=0.0)
        noisy = torch.from_numpy(
            (np.random.default_rng(1).standard_normal(len(harmonic)) * 0.2).astype(
                np.float32
            )
        )
        for wave, expect_high in ((harmonic, True), (noisy, False)):
            spec = spectrogram_for_features(wave, n_fft, hop)
            pitch = torch.full((1, spec.shape[-1]), f0)
            value = frame_periodicity(spec, pitch, sample_rate, n_fft).mean().item()
            if expect_high:
                assert value > 0.5, (sample_rate, value)
            else:
                assert value < 0.2, (sample_rate, value)


def test_periodicity_agrees_across_sample_rates():
    """The same signal must score the same at 44.1 kHz and at 16 kHz."""
    f0 = 180.0
    scores = []
    for sample_rate, hop in ((TRAIN_SR, TRAIN_HOP), (INFER_SR, INFER_HOP)):
        n_fft = matched_n_fft(sample_rate)
        wave = _tone(sample_rate, 1.0, f0, noise=0.35, seed=3)
        spec = spectrogram_for_features(wave, n_fft, hop)
        pitch = torch.full((1, spec.shape[-1]), f0)
        scores.append(frame_periodicity(spec, pitch, sample_rate, n_fft).mean().item())
    assert abs(scores[0] - scores[1]) < 0.15, scores


def test_periodicity_is_zero_on_unvoiced_frames():
    wave = _tone(TRAIN_SR, 0.5, 200.0)
    spec = spectrogram_for_features(wave, TRAIN_NFFT, TRAIN_HOP)
    pitch = torch.zeros(1, spec.shape[-1])
    assert frame_periodicity(spec, pitch, TRAIN_SR, TRAIN_NFFT).abs().max().item() == 0.0


def test_energy_is_gain_invariant():
    """A file that arrives 6 dB hotter must produce the same conditioning."""
    wave = _tone(TRAIN_SR, 1.0, 150.0, noise=0.2)
    quiet = spectrogram_for_features(wave, TRAIN_NFFT, TRAIN_HOP)
    loud = spectrogram_for_features(wave * 2.0, TRAIN_NFFT, TRAIN_HOP)
    assert torch.allclose(frame_energy(quiet), frame_energy(loud), atol=1e-5)


def test_energy_tracks_the_envelope():
    wave = _tone(TRAIN_SR, 1.0, 150.0)
    wave[: len(wave) // 2] *= 0.05
    spec = spectrogram_for_features(wave, TRAIN_NFFT, TRAIN_HOP)
    energy = frame_energy(spec).squeeze()
    half = energy.shape[-1] // 2
    assert energy[:half].mean() < energy[half:].mean() - 0.5


def test_conditioning_channels_and_none_fallback():
    assert conditioning_channels(True, True) == 2
    assert conditioning_channels(False, True) == 1
    assert conditioning_channels(False, False) == 0
    # No spectrogram must degrade to the previous model, not to zeros.
    assert frame_conditioning(None, None, TRAIN_SR, TRAIN_NFFT, True, True) is None
    spec = spectrogram_for_features(_tone(TRAIN_SR, 0.3, 150.0), TRAIN_NFFT, TRAIN_HOP)
    assert frame_conditioning(spec, None, TRAIN_SR, TRAIN_NFFT, True, False) is None
    assert frame_conditioning(spec, None, TRAIN_SR, TRAIN_NFFT, False, True) is not None


def test_band_limit_excludes_high_frequencies():
    """Energy above the band limit must not move the measurement."""
    base = _tone(TRAIN_SR, 0.5, 150.0)
    t = np.arange(len(base)) / TRAIN_SR
    above = torch.from_numpy(
        (0.3 * np.sin(2 * np.pi * (BAND_LIMIT_HZ + 3000.0) * t)).astype(np.float32)
    )
    spec_a = spectrogram_for_features(base, TRAIN_NFFT, TRAIN_HOP)
    spec_b = spectrogram_for_features(base + above, TRAIN_NFFT, TRAIN_HOP)
    pitch = torch.full((1, spec_a.shape[-1]), 150.0)
    a = frame_periodicity(spec_a, pitch, TRAIN_SR, TRAIN_NFFT).mean()
    b = frame_periodicity(spec_b, pitch, TRAIN_SR, TRAIN_NFFT).mean()
    assert torch.allclose(a, b, atol=1e-4)


def test_silence_rejection_prefers_segments_with_signal():
    torch.manual_seed(0)
    batch, frames, segment = 4, 120, 20
    x = torch.randn(batch, 8, frames)
    energy = torch.full((batch, frames), 1e-6)
    energy[:, 60:100] = 1.0                       # the only live region
    lengths = torch.full((batch,), frames, dtype=torch.long)
    starts = []
    for _ in range(40):
        _, ids = rand_slice_segments(
            x, lengths, segment, frame_energy=energy, energy_floor=0.05
        )
        starts.append(ids)
    starts = torch.cat(starts)
    assert (starts >= 40).float().mean() > 0.95


def test_silence_rejection_off_matches_uniform_draw():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 50)
    lengths = torch.full((2,), 50, dtype=torch.long)
    _, ids = rand_slice_segments(x, lengths, 10, frame_energy=None, energy_floor=0.0)
    assert ids.shape == (2,)
    assert (ids >= 0).all() and (ids <= 40).all()


def test_silence_rejection_falls_back_when_nothing_qualifies():
    """An utterance that is quiet throughout must still yield a valid start."""
    x = torch.randn(3, 4, 60)
    energy = torch.full((3, 60), 1e-9)
    lengths = torch.full((3,), 60, dtype=torch.long)
    _, ids = rand_slice_segments(x, lengths, 15, frame_energy=energy, energy_floor=0.5)
    assert ids.shape == (3,)
    assert (ids >= 0).all() and (ids <= 45).all()


def test_infer_positional_contract_matches_eval_callers():
    """``spec`` must stay the 7th positional argument of ``Synthesizer.infer``.

    Both in-training evaluators call ``infer`` positionally -- the holdout with
    an explicit tuple and the preview by unpacking ``reference`` -- so the
    order here is load-bearing.  Scoring the model without its spectrogram
    silently evaluates a path the weights were never trained on and reports the
    gap as a quality regression; that is how it was found, at holdout 1.94
    against 1.36 for the run before it.
    """
    import inspect

    from rvc.lib.algorithm.synthesizers import Synthesizer

    names = list(inspect.signature(Synthesizer.infer).parameters)
    assert names[:8] == [
        "self",
        "phone",
        "phone_lengths",
        "pitch",
        "nsff0",
        "sid",
        "seed",
        "spec",
    ], names[:8]


def test_holdout_evaluator_passes_the_spectrogram():
    """The holdout must hand ``infer`` the held-out spectrogram, not drop it."""
    # Read the source rather than importing: ``rvc.train.train`` reads its run
    # spec from ``sys.argv`` at module scope and cannot be imported under test.
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "rvc" / "train" / "train.py"
    ).read_text(encoding="utf-8")
    body = source.split("def _holdout_spectral_loss", 1)[1].split("\nclass ", 1)[0]
    assert "model.infer(" in body
    call = body.split("model.infer(", 1)[1].split(")", 1)[0]
    assert "spec" in call, call


def test_frame_energy_does_not_depend_on_how_the_audio_was_cut_up():
    """The same audio must measure the same whether or not it was split.

    ``frame_energy`` is relative to a mean, and the two sides of the model cut
    audio differently: training reads 3 s preprocessing slices, the inference
    pipeline cuts long files into ~38 s segments.  While that mean was taken
    over "whatever tensor arrived", the same frame scored differently on each
    side and stepped at every cut -- which is inaudible in any metric and
    audible in the output.  A fixed-duration window removes the dependency.
    """
    rng = np.random.default_rng(0)
    seconds = 24
    frames = seconds * TRAIN_SR // TRAIN_HOP
    # A loud half and a half 12 dB quieter, so a whole-tensor mean and a local
    # one genuinely disagree.
    time = np.arange(seconds * TRAIN_SR) / TRAIN_SR
    quiet = np.where(time < seconds / 2, 1.0, 10 ** (-12 / 20))
    audio = (np.sin(2 * np.pi * 220 * time) * quiet * 0.3).astype(np.float32)
    audio += 0.01 * rng.standard_normal(audio.size).astype(np.float32)

    def energy(samples):
        spec = spectrogram_for_features(
            torch.from_numpy(samples), TRAIN_NFFT, TRAIN_HOP
        )
        return frame_energy(spec.unsqueeze(0))[0, 0]

    whole = energy(audio)
    cut = TRAIN_SR * seconds // 2
    halves = torch.cat((energy(audio[:cut]), energy(audio[cut:])))

    n = min(whole.shape[-1], halves.shape[-1])
    # Frames within half a window of the seam see edge padding instead of their
    # true neighbours; everything further away must agree closely.
    from rvc.lib.algorithm.frame_features import ENERGY_REFERENCE_FRAMES

    margin = ENERGY_REFERENCE_FRAMES
    seam = cut // TRAIN_HOP
    interior = torch.ones(n, dtype=torch.bool)
    interior[max(0, seam - margin) : seam + margin] = False
    difference = (whole[:n] - halves[:n])[interior].abs().max()
    assert difference < 0.05, difference
    assert frames > 0
