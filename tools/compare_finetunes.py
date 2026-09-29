"""Compare two fine-tunes of the same voice by FAD and ECAPA similarity.

Both models convert the same source audio; every output is cut into the same
fixed windows, so the two are paired clip by clip.  Against real audio of the
target voice it reports:

``FAD``   Fréchet distance of WavLM frame embeddings (16 kHz, so nothing
          above 8 kHz counts).  Lower is closer to the real voice.
``ECAPA`` cosine of each clip's ECAPA embedding to the real voice's centroid.

Every real file is split in time: the second halves are the FAD reference, the
first halves give the ECAPA centroid and the floor.  ``A - B`` is
paired-bootstrapped over clips, giving a CI and how often B came out better.
Two references: the unconverted source (how far the starting voice is) and the
real first halves (the floor), with the models also scored on as many clips as
the floor has, since FAD grows as the sample shrinks.

``--a``/``--b`` take a ``.pth`` to render with, or a folder of files already
rendered from the same sources (same file names).  Renders are cached under
``--out``.  speechbrain, for ECAPA, is installed on first use if missing.

Usage::

    python tools/compare_finetunes.py \\
        --a logs/weights/voz-hifi.pth --b logs/weights/voz-bigvgan.pth \\
        --source path/to/other_singer --target logs/voz/sliced_audios \\
        --out compare/voz
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rvc.lib.catalog import is_audio_file  # noqa: E402

SR = 16000


def audio_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and is_audio_file(p))


def load_16k(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return audio


def render(model, sources, out_dir, args):
    """Convert every source with ``model`` into ``out_dir``, skipping cached files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in sources if args.force or not (out_dir / f"{s.stem}.wav").exists()]
    if not pending:
        return out_dir

    from rvc.infer.infer import VoiceConverter

    converter = VoiceConverter()
    for source in pending:
        converter.convert_audio(
            audio_input_path=str(source),
            audio_output_path=str(out_dir / f"{source.stem}.wav"),
            model_path=str(model),
            index_path=args.index,
            index_rate=args.index_rate if args.index else 0.0,
            pitch=args.pitch,
            f0_method=args.f0_method,
            protect=args.protect,
            sid=args.sid,
            seed=args.seed,
            noise_scale=args.noise_scale,
        )
    converter.cleanup_model()
    return out_dir


def windows(audio, seconds, floor_db):
    """Start samples of the non-overlapping windows loud enough to score."""
    size = int(seconds * SR)
    starts = []
    for start in range(0, len(audio) - size + 1, size):
        rms = np.sqrt(np.mean(audio[start : start + size] ** 2) + 1e-12)
        if 20 * np.log10(rms) > floor_db:
            starts.append(start)
    return starts


def paired_clips(sources, rendered, args):
    """Clips cut at the same places from source, A and B.

    The windows are chosen on the source, so all three sets hold the same
    passages in the same order.
    """
    size = int(args.clip_seconds * SR)
    clips = {"source": [], "a": [], "b": []}
    for source in sources:
        paths = {key: rendered[key] / f"{source.stem}.wav" for key in ("a", "b")}
        if not all(path.exists() for path in paths.values()):
            print(f"skipped {source.name}: missing render")
            continue
        waves = {"source": load_16k(source)}
        waves.update({key: load_16k(path) for key, path in paths.items()})
        length = min(len(wave) for wave in waves.values())
        for start in windows(waves["source"][:length], args.clip_seconds, args.floor_db):
            for key, wave in waves.items():
                clips[key].append(wave[start : start + size])
    return clips


def real_clips(target, args):
    size = int(args.clip_seconds * SR)
    files = audio_files(target)
    halves = ([], [])
    clips = []
    for path in files:
        audio = load_16k(path)
        middle = len(audio) // 2
        for start in windows(audio, args.clip_seconds, args.floor_db):
            clips.append(audio[start : start + size])
            # Each file split in time, so both halves hold every song and the
            # floor measures the voice rather than the difference between songs.
            halves[start + size > middle].append(len(clips) - 1)
    return clips, halves


class WavLMFrames:
    def __init__(self, device, layer, stride):
        from transformers import AutoFeatureExtractor, WavLMModel

        name = "microsoft/wavlm-base-plus"
        self.extractor = AutoFeatureExtractor.from_pretrained(name)
        # Warns that pos_conv_embed's weight_g/weight_v went unused; they are
        # converted to the parametrized weight norm and do load.
        self.model = WavLMModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.layer = layer
        self.stride = stride

    @torch.inference_mode()
    def __call__(self, clips):
        """Frames of every clip stacked, and the clip each frame came from."""
        frames, owner = [], []
        for index, clip in enumerate(clips):
            inputs = self.extractor(clip, sampling_rate=SR, return_tensors="pt")
            hidden = self.model(
                inputs.input_values.to(self.device), output_hidden_states=True
            ).hidden_states[self.layer][0, :: self.stride]
            frames.append(hidden.float())
            owner.append(torch.full((hidden.shape[0],), index, device=self.device))
        return torch.cat(frames), torch.cat(owner)


def install_speechbrain():
    """Install speechbrain into this interpreter, pinning torch and torchaudio.

    The pins keep pip from swapping the CUDA build of torch for whatever
    speechbrain's resolver would pick.
    """
    pins = []
    for package in ("torch", "torchaudio"):
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pass
    print("speechbrain not found; installing it for ECAPA.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "speechbrain", *pins])
    importlib.invalidate_caches()


class Ecapa:
    def __init__(self, device, cache):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            install_speechbrain()
            from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cache),
            # speechbrain wants an explicit index ("cuda:0"), not bare "cuda".
            run_opts={"device": f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"},
        )
        self.device = device

    @torch.inference_mode()
    def __call__(self, clips):
        batch = torch.from_numpy(np.stack(clips)).to(self.device)
        out = []
        for chunk in batch.split(32):
            out.append(self.model.encode_batch(chunk).squeeze(1))
        return torch.nn.functional.normalize(torch.cat(out).float(), dim=-1)


def gaussian(frames, owner, weights):
    """Mean and covariance of ``frames``, each weighted by its clip's count."""
    w = weights[owner]
    total = w.sum()
    mean = (w[:, None] * frames).sum(0) / total
    centred = frames - mean
    # float32 for the product: consumer GPUs run float64 matmuls ~64x slower.
    cov = (centred * w[:, None]).T @ centred / (total - 1)
    return mean.double(), cov.double()


def frechet(g1, g2):
    """Fréchet distance via a symmetric eigensolve instead of ``sqrtm``."""
    (m1, c1), (m2, c2) = g1, g2
    values, vectors = torch.linalg.eigh(c1)
    root = vectors @ torch.diag(values.clamp_min(0).sqrt()) @ vectors.T
    cross = torch.linalg.eigvalsh(root @ c2 @ root).clamp_min(0).sqrt().sum()
    return float((m1 - m2).square().sum() + c1.trace() + c2.trace() - 2 * cross)


def summarise(point, samples):
    samples = np.asarray(samples)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"value": point, "ci95": [float(low), float(high)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", required=True, help=".pth or folder of renders (baseline)")
    parser.add_argument("--b", required=True, help=".pth or folder of renders")
    parser.add_argument("--source", required=True, help="audio to convert, another voice")
    parser.add_argument("--target", required=True, help="real audio of the target voice")
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", default="", help="same index for both; empty = none")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--floor-db", type=float, default=-40.0)
    parser.add_argument("--wavlm-layer", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="re-render cached files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    sources = audio_files(args.source)
    rendered = {}
    for key in ("a", "b"):
        given = Path(getattr(args, key))
        rendered[key] = given if given.is_dir() else render(given, sources, out / key, args)

    clips = paired_clips(sources, rendered, args)
    real, halves = real_clips(args.target, args)
    n = len(clips["a"])
    if n < 2 or len(real) < 4:
        raise SystemExit(f"too few clips to score: {n} converted, {len(real)} real")
    print(f"{n} paired clips, {len(real)} real clips of {args.clip_seconds:.1f}s")

    wavlm = WavLMFrames(device, args.wavlm_layer, args.frame_stride)
    feats = {key: wavlm(value) for key, value in clips.items()}
    real_feats = wavlm(real)
    del wavlm

    ecapa = Ecapa(device, out / "ecapa_model")
    emb = {key: ecapa(value) for key, value in clips.items()}
    real_emb = ecapa(real)
    del ecapa

    # The second half of every real song is the reference; the first half
    # gives the ECAPA centroid and the floor.
    first, second = (torch.tensor(h, device=device) for h in halves)
    m = len(first)

    def real_gauss(indices):
        weights = torch.zeros(len(real), device=device)
        weights.index_add_(0, indices, torch.ones(len(indices), device=device))
        return gaussian(*real_feats, weights)

    def fad(key, weights, reference):
        return frechet(gaussian(*feats[key], weights), reference)

    reference = real_gauss(second)
    ones = torch.ones(n, device=device)
    point = {key: fad(key, ones, reference) for key in clips}
    floor = frechet(real_gauss(first), reference)

    centroid = torch.nn.functional.normalize(real_emb[first].mean(0), dim=0)
    cos = {key: value @ centroid for key, value in emb.items()}
    ecapa_point = {key: float(value.mean()) for key, value in cos.items()}
    ecapa_point["real"] = float((real_emb[second] @ centroid).mean())

    generator = torch.Generator(device=device).manual_seed(args.seed)
    boot = {"fad_diff": [], "ecapa_diff": []}
    # FAD is biased by sample size, so the floor (m clips) is only comparable
    # with the models scored on m clips too.
    matched = {key: [] for key in clips}
    for _ in range(args.bootstrap):
        draw = torch.randint(n, (n,), device=device, generator=generator)
        weights = torch.bincount(draw, minlength=n).float()
        resampled = real_gauss(
            second[torch.randint(len(second), (len(second),), device=device, generator=generator)]
        )
        boot["fad_diff"].append(fad("a", weights, resampled) - fad("b", weights, resampled))
        boot["ecapa_diff"].append(float((cos["a"][draw] - cos["b"][draw]).mean()))

        subset = torch.randperm(n, device=device, generator=generator)[: min(m, n)]
        weights = torch.zeros(n, device=device)
        weights[subset] = 1.0
        for key in clips:
            matched[key].append(fad(key, weights, reference))

    fad_matched = {key: float(np.mean(value)) for key, value in matched.items()}
    report = {
        "clips": n,
        "real_clips": {"first_half": m, "second_half": len(second)},
        "fad": point,
        "fad_matched": {**fad_matched, "floor": floor},
        "fad_a_minus_b": summarise(point["a"] - point["b"], boot["fad_diff"]),
        "fad_b_better_share": float(np.mean(np.asarray(boot["fad_diff"]) > 0)),
        "ecapa": ecapa_point,
        "ecapa_a_minus_b": summarise(
            ecapa_point["a"] - ecapa_point["b"], boot["ecapa_diff"]
        ),
        "ecapa_b_better_share": float(np.mean(np.asarray(boot["ecapa_diff"]) < 0)),
        "settings": vars(args),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\nFAD (WavLM layer {args.wavlm_layer}, lower is better)")
    print(f"  {'':<8}{f'{n} clips':>10}{f'{m} clips':>12}")
    for key, label in (("a", "A"), ("b", "B"), ("source", "source")):
        print(f"  {label:<8}{point[key]:10.3f}{fad_matched[key]:12.3f}")
    print(f"  {'floor':<8}{'':>10}{floor:12.3f}   real first halves vs second halves")
    diff = report["fad_a_minus_b"]
    print(
        f"  A-B {diff['value']:+.3f}  CI {diff['ci95'][0]:+.3f} {diff['ci95'][1]:+.3f}"
        f"  B better in {report['fad_b_better_share']:.0%} of resamples"
    )
    print("\nECAPA cosine to the real first-half centroid (higher is better)")
    for key, label in (("a", "A"), ("b", "B"), ("source", "source"), ("real", "real")):
        print(f"  {label:<8}{ecapa_point[key]:8.3f}")
    diff = report["ecapa_a_minus_b"]
    print(
        f"  A-B {diff['value']:+.4f}  CI {diff['ci95'][0]:+.4f} {diff['ci95'][1]:+.4f}"
        f"  B better in {report['ecapa_b_better_share']:.0%} of resamples"
    )
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
