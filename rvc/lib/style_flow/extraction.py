"""Building a style dataset: clips, units, descriptors and the manifest."""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rvc.lib.terminal import info, success, track

from .clips import experiment_runs, folder_runs, split_frames, stitch_frames
from .data import (
    CLIP_DIR,
    CODEBOOK,
    clip_paths,
    compute_normalizer,
    read_manifest,
    save_clip,
    speaker_descriptors,
    write_manifest,
)
from .descriptors import DescriptorConfig, analyze, descriptor_vector
from .f0_repr import Normalizer, ReprConfig, decompose
from .frontend import HOP, VOICING_THRESHOLD, StyleFrontend, highpass, loudness_db
from .units import UnitCodebook, units_to_frames

DONE_FILE = "done_runs.txt"


def prefetch(runs, workers):
    """``(run, stretches)`` in order, loading ahead on threads."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(run.load) for run in runs[:workers]]
        for i, run in enumerate(runs):
            stretches = pending[i].result()
            if i + workers < len(runs):
                pending.append(pool.submit(runs[i + workers].load))
            pending[i] = None
            yield run, stretches


class Analysis:
    """F0 and features per stretch, from the experiment's files when they
    match what the frontend would compute and from the frontend otherwise."""

    def __init__(self, frontend, exp_dir=None, recompute=False):
        self.frontend = frontend
        self.f0_dir = self.feat_dir = None
        info_path = os.path.join(exp_dir or "", "model_info.json")
        if exp_dir and not recompute and os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)
            # The experiment's F0 was taken at RMVPE's usual threshold.
            if model_info.get("f0_method") == "rmvpe" and frontend.voicing_threshold == VOICING_THRESHOLD:
                self.f0_dir = os.path.join(exp_dir, "f0_voiced")
            if model_info.get("embedder_model") == frontend.embedder_name:
                self.feat_dir = os.path.join(exp_dir, "extracted")
        info(
            f"F0: {'experiment' if self.f0_dir else 'RMVPE'}; "
            f"features: {'experiment' if self.feat_dir else frontend.embedder_name}.",
            tag="[STYLE]",
        )

    def _stitched(self, stretch, directory, file_name, hop):
        if directory is None or not stretch.slices:
            return None
        try:
            arrays = [np.load(os.path.join(directory, file_name(name))) for name, _, _ in stretch.slices]
        except OSError:
            return None
        starts = [start for _, start, _ in stretch.slices]
        overlaps = [overlap for _, _, overlap in stretch.slices]
        return stitch_frames(arrays, starts, overlaps, len(stretch.audio), hop).astype(np.float32)

    def f0(self, stretch, audio):
        f0 = self._stitched(stretch, self.f0_dir, lambda name: name + ".npy", HOP)
        return f0 if f0 is not None else self.frontend.f0(audio)[: len(audio) // HOP]

    def features(self, stretch, audio):
        feats = self._stitched(stretch, self.feat_dir, lambda name: os.path.splitext(name)[0] + ".npy", 2 * HOP)
        return feats if feats is not None else self.frontend.features(audio)


def clips_of(stretches, analysis, rcfg, ccfg):
    """``(stretch, audio, k, (start, end), f0, coarse, residual, vuv)`` per clip;
    ``audio`` is the high-passed stretch.  The representation is computed over
    the whole stretch and then cut, as it is over the whole song at inference."""
    fr = rcfg.frame_rate
    for stretch in stretches:
        audio = highpass(stretch.audio)
        f0 = analysis.f0(stretch, audio)
        if len(f0) < ccfg["min_seconds"] * fr:
            continue
        coarse, residual, vuv = decompose(f0, rcfg)
        spans = split_frames(vuv, int(ccfg["min_seconds"] * fr), int(ccfg["max_seconds"] * fr))
        for k, (s, e) in enumerate(spans):
            if vuv[s:e].mean() < ccfg["min_voiced_fraction"]:
                continue
            yield stretch, audio, k, (s, e), f0[s:e], coarse[s:e], residual[s:e], vuv[s:e]


def fit_codebook(runs, analysis, rcfg, ccfg, ucfg, workers):
    rng = random.Random(ucfg.get("seed", 0))
    order = runs[:]
    rng.shuffle(order)
    target = int(ucfg["fit_max_frames"])
    per_clip = int(ucfg["fit_frames_per_clip"])
    samples, have = [], 0
    for run, stretches in track(prefetch(order, workers), total=len(order), description="Sampling features for k-means"):
        cached = {}
        for stretch, audio, _k, (s, e), *_ in clips_of(stretches, analysis, rcfg, ccfg):
            if id(stretch) not in cached:
                cached[id(stretch)] = analysis.features(stretch, audio)
            feats = cached[id(stretch)][s // 2 : e // 2]
            take = np.random.default_rng(rng.randrange(2**31)).choice(len(feats), min(per_clip, len(feats)), replace=False)
            samples.append(feats[take])
            have += len(take)
        if have >= target:
            break
    samples = np.concatenate(samples)
    info(f"Fitting {ucfg['clusters']} units on {len(samples)} frames.", tag="[STYLE]")
    return UnitCodebook.fit(samples, int(ucfg["clusters"]), int(ucfg["fit_iterations"]), int(ucfg.get("seed", 0)))



def reference_of(path: str) -> dict:
    """Embedder, codebook and statistics a dataset must share with its base:
    from a style dataset directory or a style checkpoint."""
    if os.path.isdir(path):
        manifest = read_manifest(path)
        return {
            "embedder": manifest["embedder"],
            "representation": ReprConfig.from_dict(manifest["representation"]),
            "descriptor_config": DescriptorConfig.from_dict(manifest["descriptor_config"]),
            "normalizer": Normalizer.from_dict(manifest["normalizer"]),
            "codebook": UnitCodebook.load(os.path.join(path, manifest.get("codebook", CODEBOOK))),
        }
    from .checkpoint import load

    base = load(path)
    return {
        "embedder": base.embedder,
        "representation": base.representation,
        "descriptor_config": base.descriptor_config,
        "normalizer": base.normalizer,
        "codebook": base.codebook,
    }


def build_dataset(
    out: str,
    cfg: dict,
    *,
    experiment: str | None = None,
    audio_dir: str | None = None,
    speaker: int | None = None,
    reference: str | None = None,
    embedder: str = "contentvec",
    recompute: bool = False,
    device: str = "cuda:0",
    workers: int = 4,
) -> str:
    """Extract ``experiment`` (an RVC experiment) or ``audio_dir`` into the
    style dataset ``out``.  ``reference`` (a style dataset or checkpoint)
    fixes the embedder, codebook and statistics, as a fine-tune set needs.
    Resumes an interrupted run.  Returns ``out``."""
    rcfg = ReprConfig.from_dict(cfg.get("representation"))
    dcfg = DescriptorConfig.from_dict(cfg.get("descriptors"))
    ccfg, ucfg = cfg["clips"], cfg["units"]

    if experiment:
        runs = experiment_runs(experiment, ccfg["slice_overlap_seconds"], ccfg["stitch_min_correlation"])
    else:
        runs = folder_runs(audio_dir, speaker)
    if not runs:
        raise ValueError("No audio found.")

    os.makedirs(os.path.join(out, CLIP_DIR), exist_ok=True)
    codebook_path = os.path.join(out, CODEBOOK)
    normalizer = None
    if reference:
        ref = reference_of(reference)
        if ref["representation"] != rcfg or ref["descriptor_config"] != dcfg:
            info("Using the reference's representation and descriptor settings.", tag="[STYLE]")
        rcfg, dcfg = ref["representation"], ref["descriptor_config"]
        embedder, normalizer = ref["embedder"], ref["normalizer"]
        ref["codebook"].save(codebook_path)

    fcfg = cfg.get("frontend", {})
    voicing_threshold = float(fcfg.get("voicing_threshold", VOICING_THRESHOLD))
    frontend = StyleFrontend(
        device, embedder, fcfg.get("chunk_seconds", 20.0), fcfg.get("context_seconds", 1.0), voicing_threshold
    )
    analysis = Analysis(frontend, experiment, recompute)

    if os.path.exists(codebook_path):
        codebook = UnitCodebook.load(codebook_path)
    else:
        codebook = fit_codebook(runs, analysis, rcfg, ccfg, ucfg, workers)
        codebook.save(codebook_path)

    done_path = os.path.join(out, DONE_FILE)
    done = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            done = {line.strip() for line in f if line.strip()}
    todo = [run for run in runs if run.key not in done]
    info(f"{len(runs)} recordings, {len(todo)} left to extract.", tag="[STYLE]")

    with open(done_path, "a", encoding="utf-8") as done_file:
        for run, stretches in track(prefetch(todo, workers), total=len(todo), description="Extracting style data"):
            units, levels = {}, {}
            for stretch, audio, k, (s, e), f0, coarse, residual, vuv in clips_of(stretches, analysis, rcfg, ccfg):
                if id(stretch) not in units:
                    feats = analysis.features(stretch, audio)
                    units[id(stretch)] = units_to_frames(codebook.assign(feats), len(audio) // HOP)
                    levels[id(stretch)] = loudness_db(audio, len(audio) // HOP)
                events = analyze(f0, rcfg, dcfg, residual)
                save_clip(
                    os.path.join(out, CLIP_DIR, f"{run.key}_{stretch.first}_{k}.npz"),
                    f0=f0, coarse=coarse, residual=residual, vuv=vuv, units=units[id(stretch)][s:e],
                    descriptors=descriptor_vector(events), events=events, speaker=run.speaker,
                    loudness=levels[id(stretch)][s:e],
                )
            done_file.write(run.key + "\n")
            done_file.flush()

    paths = clip_paths(out)
    if not paths:
        raise ValueError("No clip survived the length and voicing filters.")
    if normalizer is None:
        normalizer = compute_normalizer(paths)
    speakers = speaker_descriptors(paths)
    frames = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            frames += len(data["f0"])
    write_manifest(
        out,
        embedder=embedder,
        voicing_threshold=voicing_threshold,
        representation=rcfg.to_dict(),
        descriptor_config=dcfg.to_dict(),
        normalizer=normalizer.to_dict(),
        codebook=CODEBOOK,
        speaker_descriptors={str(k): v for k, v in speakers.items()},
        clips=len(paths),
        hours=frames / rcfg.frame_rate / 3600.0,
    )
    success(f"{len(paths)} clips, {frames / rcfg.frame_rate / 3600.0:.1f} h in {out}.", tag="[STYLE]")
    return out
