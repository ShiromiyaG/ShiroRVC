import os
import sys
import time

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Run as a subprocess, so ``sys.path[0]`` is this file's directory rather than
# the repository root and the ``rvc.`` import below fails.  It has failed since
# the first commit; ``run_index_script`` did not check the return code, so the
# UI reported success while nothing was ever written.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from rvc.lib.terminal import (
    error as print_error,
    info,
    install_rich_print,
    print_error_panel,
    print_settings_panel,
    success,
    warning,
)
from rvc.lib import index_meta

install_rich_print()

# ---------------------------------------------------------------------------
# Tuning
#
# At inference the pipeline calls ``index.reconstruct_n(0, ntotal)`` and, on
# CUDA, runs an exact brute-force search over the result -- the IVF structure is
# only used on the CPU path.  So the number of vectors kept here is both the
# retrieval's fidelity *and* its VRAM cost:
#
#     VRAM = vectors * 768 dims * 4 bytes      (100k -> 293 MB)
#
# The previous default reduced anything over 200k vectors to 10,000 k-means
# centroids, which inverted the relationship between dataset size and index
# quality: a 4 h dataset ended up with an index 18x smaller than a 2 h one.
#
# Left at 100k rather than raised: the retriever now falls back to a half
# precision device copy past a memory budget, so a larger index is affordable,
# but choosing to spend someone's VRAM is theirs to make.  Override with
# ``RVC_INDEX_MAX_VECTORS``.
INDEX_MAX_VECTORS = int(os.environ.get("RVC_INDEX_MAX_VECTORS", 100_000))

# k-means cost scales with the cluster count -- every mini-batch measures each
# of its points against every centroid -- so it cannot follow
# ``INDEX_MAX_VECTORS`` upward.  100k clusters on 260k vectors did not finish in
# ten minutes when this was tried.  Anyone choosing "KMeans" is asking for even
# coverage rather than raw fidelity, and 20k centroids is already twice the old
# default; subsampling is the path that scales.
KMEANS_MAX_CLUSTERS = 20_000

# Initialisation stays random, which is not an oversight.  ``k-means++`` seeds
# each centroid by measuring every candidate against the ones already chosen,
# which is O(n * k) *before* the fit begins: at 20k clusters over 115k vectors
# it had not finished after ten minutes, against 49 seconds for the whole fit
# with random init.  Given the choice between 20k centroids seeded randomly and
# 10k seeded well, twice the resolution is worth more here than a slightly
# better local minimum -- the centroids are retrieval targets, not a clustering
# anyone reads.
KMEANS_INIT = "random"

# Selection keeps whole runs of consecutive frames rather than individual ones.
# The retrieval's continuity bonus can only reward a candidate for being the
# next frame of the same recording if that next frame survived selection, and
# picking frames independently leaves roughly none of them adjacent.  16 frames
# is 320 ms of speech: long enough to hold a syllable's articulation together,
# short enough that the budget still spreads across the dataset.
RUN_LENGTH = 16

# Density estimation for run scoring.  Both numbers are deliberately small: this
# is a redundancy measure, not a clustering anyone consumes, and it runs over
# every frame in the dataset.
DENSITY_DIMS = 64
DENSITY_CLUSTERS = 1024
DENSITY_SAMPLE = 50_000

# A frame this far below the recording's own loud passages is silence or room
# tone.  Measured per file against a high percentile rather than an absolute
# level so it survives whatever normalisation preprocessing applied.
SILENCE_FLOOR_DB = -45.0
# The relative test cannot see a file that is silent all the way through, and
# training deliberately adds some: ``include_mutes`` appends pure-silence
# examples, whose features are as retrievable as any other unless they are cut
# here.  This absolute floor is what removes them.
SILENCE_ABSOLUTE_DB = -60.0
# If the threshold would throw away most of a file it is measuring something
# other than silence, so it is ignored for that file.
SILENCE_MIN_KEEP = 0.10

# Hop of the embedder in 16 kHz samples: features land at 50 Hz.
FEATURE_HOP_16K = 320

# ``nprobe`` is how many of the IVF cells a CPU search visits.  It was 1, which
# measures 18% top-8 recall against an exact search -- four of every five
# "nearest neighbours" were not near.  It is now tuned against a measurement
# instead of guessed from the cell count.
NPROBE_LADDER = (8, 16, 32, 64, 128, 256)
NPROBE_TARGET_RECALL = 0.90

# Fixed so the same dataset always produces the same index.
SEED = 1234


def _frame_energy_mask(audio_path, frames):
    """Per-frame keep mask for one recording, or ``None`` if it cannot be measured.

    The 16 kHz slices are optional -- extraction can be told to delete them --
    so every failure here is a reason to keep all frames, never to drop them.
    """
    try:
        import soundfile as sf

        audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000 or audio.size < FEATURE_HOP_16K:
        return None

    usable = min(frames, audio.size // FEATURE_HOP_16K)
    if usable < 1:
        return None
    blocks = audio[: usable * FEATURE_HOP_16K].reshape(usable, FEATURE_HOP_16K)
    db = 10.0 * np.log10(np.square(blocks).mean(axis=1) + 1e-12)

    relative = db > (np.percentile(db, 95.0) + SILENCE_FLOOR_DB)
    absolute = db > SILENCE_ABSOLUTE_DB

    keep = np.zeros(frames, dtype=bool)
    keep[:usable] = relative & absolute
    # A file that is silent throughout is not a misfiring threshold, it is a
    # mute example, so the "kept too little" guard only rescues files that had
    # real speech in them to begin with.
    if absolute.any() and keep.sum() < max(1, int(SILENCE_MIN_KEEP * frames)):
        keep[:usable] = absolute
    # Frames past the end of the audio were never measured; they follow whatever
    # the measured part decided rather than being kept by default.
    keep[usable:] = bool(keep[:usable].any())
    return keep


def _load_features(feature_dir, audio_dir):
    """Concatenate the extracted features, tagging each frame with its origin.

    ``utt``/``pos`` are what makes the continuity bonus possible at inference.
    ``pos`` counts frames in the *original* stream, before the silence mask
    removes any, so a candidate is only credited as "the next frame" when it
    genuinely was one.
    """
    features, utts, positions = [], [], []
    dropped_silence = 0
    dropped_nonfinite = 0

    for utt_id, name in enumerate(sorted(os.listdir(feature_dir))):
        if not name.endswith(".npy"):
            continue
        array = np.load(os.path.join(feature_dir, name), allow_pickle=False)
        if array.ndim != 2 or array.shape[0] == 0:
            warning(f"Skipping {name}: unexpected shape {array.shape}.", tag="[INDEX]")
            continue
        array = np.asarray(array, dtype=np.float32)

        keep = np.isfinite(array).all(axis=1)
        dropped_nonfinite += int((~keep).sum())

        if audio_dir is not None:
            stem = os.path.splitext(name)[0]
            for extension in (".wav", ".flac", ".ogg", ".mp3"):
                candidate = os.path.join(audio_dir, stem + extension)
                if os.path.exists(candidate):
                    energy = _frame_energy_mask(candidate, array.shape[0])
                    if energy is not None:
                        dropped_silence += int((keep & ~energy).sum())
                        keep &= energy
                    break

        if not keep.any():
            continue
        features.append(array[keep])
        utts.append(np.full(int(keep.sum()), utt_id, dtype=np.int32))
        positions.append(np.flatnonzero(keep).astype(np.int32))

    if not features:
        raise RuntimeError(f"No usable feature files in {feature_dir}.")

    dropped = []
    if dropped_nonfinite:
        dropped.append(f"{dropped_nonfinite} non-finite")
    if dropped_silence:
        dropped.append(f"{dropped_silence} silent")
    if dropped:
        info(f"Dropping {' and '.join(dropped)} frames.", tag="[INDEX]")

    return (
        np.ascontiguousarray(np.concatenate(features, axis=0)),
        np.concatenate(utts),
        np.concatenate(positions),
    )


def _frame_density(features, rng):
    """Rough population of each frame's neighbourhood, one value per frame.

    Used only to rank runs against each other, so it is estimated on a PCA
    projection with a coarse codebook: full-dimension k-means over every frame
    costs minutes and would buy precision nothing here consumes.
    """
    count, dims = features.shape
    sample = features
    if count > DENSITY_SAMPLE:
        sample = np.ascontiguousarray(
            features[rng.choice(count, size=DENSITY_SAMPLE, replace=False)]
        )

    target_dims = min(DENSITY_DIMS, dims)
    pca = faiss.PCAMatrix(dims, target_dims)
    pca.train(sample)
    projected_sample = pca.apply(sample)

    clusters = min(DENSITY_CLUSTERS, max(1, projected_sample.shape[0] // 39))
    kmeans = faiss.Kmeans(target_dims, clusters, niter=10, seed=SEED, verbose=False)
    kmeans.train(projected_sample)

    assignments = np.empty(count, dtype=np.int64)
    for start in range(0, count, 65536):
        block = pca.apply(np.ascontiguousarray(features[start : start + 65536]))
        _, ids = kmeans.index.search(block, 1)
        assignments[start : start + 65536] = ids[:, 0]

    population = np.bincount(assignments, minlength=clusters).astype(np.float32)
    return population[assignments]


def _select_runs(features, utts, positions, rng, budget):
    """Keep whole runs of consecutive frames, preferring under-represented ones.

    Two things are being bought at once.  Runs preserve local temporal structure,
    without which the continuity bonus has nothing to find.  Ranking them by the
    rarity of the frames they contain spends the budget on articulation the
    dataset shows once rather than on the hundredth copy of a sustained vowel --
    an index that is 40% room tone retrieves room tone.
    """
    count = features.shape[0]
    boundaries = np.flatnonzero(np.diff(utts) != 0) + 1
    groups = np.split(np.arange(count), boundaries)

    runs = []
    for group in groups:
        for start in range(0, group.size, RUN_LENGTH):
            runs.append(group[start : start + RUN_LENGTH])

    try:
        density = _frame_density(features, rng)
        # Rarity per run, averaged.  1/sqrt keeps the ranking from being decided
        # entirely by the single emptiest cell in the dataset.
        scores = np.array(
            [float(np.mean(1.0 / np.sqrt(density[run]))) for run in runs],
            dtype=np.float64,
        )
        strategy = "runs by rarity"
    except Exception as error:
        warning(f"Density estimation failed ({error}); selecting runs at random.", tag="[INDEX]")
        scores = rng.random(len(runs))
        strategy = "runs at random"

    # Break ties (and any degenerate scoring) reproducibly rather than by the
    # order files happened to land in.
    order = np.lexsort((rng.random(len(runs)), -scores))

    chosen, kept = [], 0
    for position in order:
        run = runs[position]
        if kept + run.size > budget:
            # Trim the last run rather than the sorted result: cutting after the
            # sort would drop whichever frames happen to sit highest in the
            # concatenation, which is an arbitrary utterance, not an unwanted run.
            run = run[: budget - kept]
        chosen.append(run)
        kept += run.size
        if kept >= budget:
            break

    keep = np.sort(np.concatenate(chosen))
    info(
        f"Selecting {strategy}: {count} vectors -> {keep.size} real frames "
        f"({keep.size * features.shape[1] * 4 / 1024 ** 2:.0f} MB at inference).",
        tag="[INDEX]",
    )
    return (
        np.ascontiguousarray(features[keep]),
        utts[keep],
        positions[keep],
    )


def _reduce(features, utts, positions, index_algorithm, rng):
    """Bring the vector count under ``INDEX_MAX_VECTORS``.

    Two ways to give up vectors, and they lose different things:

    * run selection keeps *real observed frames*, which is what the retrieval is
      for -- it pulls a query toward something the speaker actually produced;
    * k-means keeps cluster *means*, which cover the space more evenly but
      average away the within-cluster variation, and that variation is exactly
      the idiosyncratic articulation worth retrieving.  It also destroys the
      frame provenance, so an index built this way cannot use the continuity
      bonus at inference.

    Run selection is the default because it keeps ten times as many points for
    the same memory, every one of them is genuine, and they stay in order.
    """
    total = features.shape[0]
    if index_algorithm == "Faiss" or total <= INDEX_MAX_VECTORS:
        if total > INDEX_MAX_VECTORS:
            info(
                f"Keeping all {total} vectors "
                f"({total * features.shape[1] * 4 / 1024 ** 2:.0f} MB at inference).",
                tag="[INDEX]",
            )
        return features, utts, positions, "all"

    if index_algorithm == "KMeans":
        clusters = min(KMEANS_MAX_CLUSTERS, INDEX_MAX_VECTORS, total)
        info(f"k-means: {total} vectors -> {clusters} centroids.", tag="[INDEX]")
        started = time.time()
        centroids = (
            MiniBatchKMeans(
                n_clusters=clusters,
                batch_size=4096,
                compute_labels=False,
                init=KMEANS_INIT,
                # One run.  Each mini-batch measures its points against every
                # centroid, so at these cluster counts restarting the whole
                # fit two more times costs far more than the marginally better
                # local minimum is worth.
                n_init=1,
                random_state=SEED,
            )
            .fit(features)
            .cluster_centers_
        )
        info(f"k-means finished in {time.time() - started:.1f}s.", tag="[INDEX]")
        return (
            np.ascontiguousarray(centroids, dtype=np.float32),
            None,
            None,
            "kmeans centroids",
        )

    features, utts, positions = _select_runs(
        features, utts, positions, rng, INDEX_MAX_VECTORS
    )
    return features, utts, positions, "runs"


def _measure_recall(index, features, rng, queries=256, k=8):
    """Top-k recall of the IVF search against an exact search.

    Only the CPU inference path goes through FAISS -- on CUDA the pipeline
    reconstructs every vector and brute-forces it -- so this number is what CPU
    users actually get.  It is worth measuring because it is invisible
    otherwise: an index with 18% recall loads, searches and returns results
    that simply are not the nearest neighbours, and nothing complains.
    """
    count = features.shape[0]
    if count <= k:
        return None
    sample = rng.choice(count, size=min(queries, count), replace=False)
    probe = np.ascontiguousarray(features[sample])

    _, approximate = index.search(probe, k)

    exact = np.empty((probe.shape[0], k), dtype=np.int64)
    inner_product = index.metric_type == faiss.METRIC_INNER_PRODUCT
    norms = None if inner_product else (features**2).sum(axis=1)
    for start in range(0, probe.shape[0], 64):
        block = probe[start : start + 64]
        similarity = block @ features.T
        if inner_product:
            # Larger is better, so rank the negation to reuse one code path.
            scores = -similarity
        else:
            scores = norms[None, :] - 2.0 * similarity
        exact[start : start + 64] = np.argpartition(scores, k, axis=1)[:, :k]

    overlap = [len(set(a) & set(b)) / k for a, b in zip(approximate, exact)]
    return float(np.mean(overlap))


def _tune_nprobe(index, features, rng):
    """Raise ``nprobe`` until the CPU search actually finds nearest neighbours.

    The recall was already being measured; it printed a warning telling the user
    to edit a constant in this file, which is not a fix anyone applies.  The
    ladder stops at the first value that clears the target so CPU inference pays
    for the recall it needs and no more.
    """
    index_ivf = faiss.extract_index_ivf(index)
    # An index with fewer cells than the first rung still needs a valid nprobe,
    # and visiting every cell it has is an exact search rather than a cost.
    ladder = sorted({min(int(index_ivf.nlist), rung) for rung in NPROBE_LADDER})
    best = (ladder[0], None)

    for nprobe in ladder:
        index_ivf.nprobe = nprobe
        recall = _measure_recall(index, features, rng)
        if recall is None:
            index_ivf.nprobe = nprobe
            return nprobe, None
        best = (nprobe, recall)
        if recall >= NPROBE_TARGET_RECALL:
            break

    index_ivf.nprobe = best[0]
    return best


def _build_index(features, metric):
    """Train an IVF index over ``features``, normalising first for cosine.

    Cosine ranks by direction alone.  For embeddings whose magnitude tracks
    loudness that is the better question to ask -- a quiet and a loud rendering
    of the same phone should be equally good matches -- and it is what the
    retrieval's query-norm rescaling assumes on the other side.
    """
    count, dims = features.shape
    searchable = features
    if metric == index_meta.METRIC_COSINE:
        searchable = np.ascontiguousarray(features.copy())
        faiss.normalize_L2(searchable)

    # FAISS wants enough training points per cell; 39 is its own guideline.
    n_ivf = max(1, min(int(16 * np.sqrt(count)), count // 39))
    faiss_metric = (
        faiss.METRIC_INNER_PRODUCT
        if metric == index_meta.METRIC_COSINE
        else faiss.METRIC_L2
    )
    index = faiss.index_factory(dims, f"IVF{n_ivf},Flat", faiss_metric)
    index.train(searchable)
    for start in range(0, count, 8192):
        index.add(searchable[start : start + 8192])
    return index, searchable, n_ivf


def main(exp_dir, index_algorithm, index_metric=index_meta.METRIC_L2):
    feature_dir = os.path.join(exp_dir, "extracted")
    model_name = os.path.basename(exp_dir)

    if not os.path.exists(feature_dir):
        print_error(
            f"No features at {feature_dir}. Run preprocessing and feature "
            "extraction first.",
            tag="[INDEX]",
        )
        return 1

    if index_metric not in index_meta.METRICS:
        warning(f"Unknown metric {index_metric!r}; using l2.", tag="[INDEX]")
        index_metric = index_meta.METRIC_L2

    index_filepath = os.path.join(exp_dir, f"{model_name}.index")
    if os.path.exists(index_filepath):
        # This used to be a silent no-op, so changing the algorithm and
        # regenerating quietly handed back the previous index.
        info(f"Replacing the existing index at {index_filepath}.", tag="[INDEX]")

    audio_dir = os.path.join(exp_dir, "sliced_audios_16k")
    if not os.path.isdir(audio_dir):
        audio_dir = None
        warning("No 16 kHz slices to measure; keeping silent frames.", tag="[INDEX]")

    features, utts, positions = _load_features(feature_dir, audio_dir)
    info(f"{features.shape[0]} frames of {features.shape[1]} dims.", tag="[INDEX]")

    rng = np.random.default_rng(SEED)
    features, utts, positions, selection = _reduce(
        features, utts, positions, index_algorithm, rng
    )

    count = features.shape[0]
    index, searchable, n_ivf = _build_index(features, index_metric)
    nprobe, recall = _tune_nprobe(index, searchable, rng)

    faiss.write_index(index, index_filepath)

    meta = index_meta.IndexMeta(
        metric=index_metric,
        dim=int(features.shape[1]),
        count=int(count),
        selection=selection,
        nprobe=int(nprobe),
        recall=recall,
        utt=utts,
        pos=positions,
        extra={"n_ivf": int(n_ivf), "algorithm": index_algorithm},
    )
    index_meta.write(index_filepath, meta)

    print_settings_panel(
        [
            ("Vectors", f"{count:,}"),
            ("Metric", index_metric),
            ("Structure", f"IVF{n_ivf}, nprobe {nprobe}"),
            ("Selection", selection),
            ("Top-8 recall", "n/a" if recall is None else f"{recall:.1%}"),
        ],
        title="Index",
    )
    success(f"Saved '{index_filepath}'.", tag="[INDEX]")
    if recall is not None and recall < 0.5:
        warning(
            "Recall is low even at the highest nprobe. CUDA inference is "
            "unaffected -- it brute-forces every vector -- but CPU inference "
            "will retrieve neighbours that are not the nearest.",
            tag="[INDEX]",
        )
    if utts is None:
        warning(
            "This algorithm discards frame provenance, so the retrieval's "
            "temporal continuity bonus will be inactive for this index.",
            tag="[INDEX]",
        )
    return 0


if __name__ == "__main__":
    try:
        metric = str(sys.argv[3]) if len(sys.argv) > 3 else index_meta.METRIC_L2
        sys.exit(main(str(sys.argv[1]), str(sys.argv[2]), metric))
    except Exception as error:
        # The old handler blamed GPU memory, which this script never touches --
        # it is faiss and scikit-learn on the CPU.
        import traceback

        print_error_panel(
            error,
            title="Index extraction failed",
            details=traceback.format_exc(),
        )
        sys.exit(1)
