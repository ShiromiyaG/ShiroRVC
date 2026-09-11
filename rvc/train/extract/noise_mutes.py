"""Low-noise mute examples, generated per experiment.

The shared ``logs/mute`` example is one clip of digital zeros whose features
were extracted once, in isolation.  That teaches the model to silence one
feature vector, not the pauses it meets at inference: those carry a noise
floor, and the embedder codes silence differently depending on what is around
it (see ``AudioProcessor.gate_to_source`` in ``rvc/infer/pipeline.py``).  In a speech dataset
the quiet frames between phrases are mostly breaths, so a pause gets rendered
as a breath, however many copies of the zero clip are added.

Each clip here pairs low-level noise on the *input* side (the 16 kHz audio the
embedder sees) with digital silence on the *target* side (the audio the
decoder is trained to reproduce), so what the model learns is "a quiet noise
floor comes out silent".  The clips differ in level and colour so the lesson
is not one vector again.

The clips live under ``<experiment>/mutes/``, apart from ``sliced_audios`` and
``extracted``, so neither the dataset scan in ``extract.py`` nor the retrieval
index (which reads ``extracted/``) picks them up.
"""

import json
import os
import shutil

import numpy as np
import soundfile as sf

MUTE_DIR = "mutes"
#: Every filename starts with this; ``is_mute_path`` keys on it.
MUTE_PREFIX = "mute"
#: Bump whenever the clips change, so existing experiments regenerate them.
VERSION = 1

DURATION_S = 3.0
INPUT_RATE = 16000
#: The pitch extractors' hop at 16 kHz: RMVPE frames are ``samples // 160 + 1``.
F0_HOP = 160
#: Noise-floor range, RMS in dBFS.  Below -60 the inference silence gate
#: already fades the output out; above -50 the floor starts to overlap the
#: quiet end of real speech.  The range straddles the gate so the two agree.
LEVEL_DB_RANGE = (-70.0, -50.0)
#: Spectral slopes cycled across the clips: white, pink and brown noise.
COLOURS = (0.0, 1.0, 2.0)
SEED = 1234


def is_mute_path(path):
    """True for a mute clip from either source (shared asset or generated)."""
    return os.path.basename(str(path)).startswith(MUTE_PREFIX)


def _coloured_noise(rng, samples, slope, rate):
    """Gaussian noise with a ``1/f**slope`` power spectrum, zero mean.

    The slope is capped below 20 Hz so brown noise does not turn into a DC
    drift, and bin 0 is dropped outright.
    """
    spectrum = np.fft.rfft(rng.standard_normal(samples))
    freqs = np.fft.rfftfreq(samples, d=1.0 / rate)
    gain = np.maximum(freqs, 20.0) ** (-slope / 2.0)
    gain[0] = 0.0
    noise = np.fft.irfft(spectrum * gain, n=samples)
    return noise - noise.mean()


def mute_specs(count):
    """``(name, level_db, slope)`` for each of ``count`` clips, deterministic."""
    low, high = LEVEL_DB_RANGE
    levels = [0.5 * (low + high)] if count == 1 else np.linspace(low, high, count)
    return [
        (f"{MUTE_PREFIX}_{index:02d}", float(level), COLOURS[index % len(COLOURS)])
        for index, level in enumerate(levels)
    ]


def _meta(sample_rate, embedder_model, count):
    return {
        "version": VERSION,
        "sample_rate": int(sample_rate),
        "embedder_model": str(embedder_model),
        "count": int(count),
    }


def prepare_noise_mutes(exp_dir, sample_rate, embedder_model, count):
    """Write the mute clips and their pitch files; return embedding work items.

    Returns ``(clips, pending)``.  ``clips`` lists every clip as
    ``(target_wav, feature_npy, f0_npy, f0_voiced_npy)``.  ``pending`` holds
    ``[input_wav_16k, f0_npy, f0_voiced_npy, feature_npy]`` for the clips whose
    features still have to be extracted -- the same shape as
    ``extract.py``'s file entries, so they go through the same embedder pass.

    Pitch is written rather than extracted: a pitch tracker run on noise can
    report spurious voicing, and these clips are unvoiced by definition.  The
    arrays match the shared mute asset (coarse ``1``, fine ``0.0``).
    """
    root = os.path.join(exp_dir, MUTE_DIR)
    meta_path = os.path.join(root, "meta.json")
    meta = _meta(sample_rate, embedder_model, count)

    if os.path.isdir(root):
        try:
            with open(meta_path, "r") as handle:
                stale = json.load(handle) != meta
        except (OSError, ValueError):
            stale = True
        if stale:
            shutil.rmtree(root)

    if count <= 0:
        return [], []

    dirs = {
        key: os.path.join(root, key)
        for key in ("sliced_audios", "sliced_audios_16k", "extracted", "f0", "f0_voiced")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    input_samples = int(round(DURATION_S * INPUT_RATE))
    target_samples = int(round(DURATION_S * int(sample_rate)))
    f0_frames = input_samples // F0_HOP + 1

    clips, pending = [], []
    for index, (name, level_db, slope) in enumerate(mute_specs(count)):
        target = os.path.join(dirs["sliced_audios"], f"{name}.wav")
        source = os.path.join(dirs["sliced_audios_16k"], f"{name}.wav")
        feature = os.path.join(dirs["extracted"], f"{name}.npy")
        f0 = os.path.join(dirs["f0"], f"{name}.wav.npy")
        f0_voiced = os.path.join(dirs["f0_voiced"], f"{name}.wav.npy")

        if not os.path.isfile(target):
            sf.write(target, np.zeros(target_samples, dtype=np.float32), int(sample_rate), subtype="PCM_16")
        if not os.path.isfile(source):
            rng = np.random.default_rng(SEED + index)
            noise = _coloured_noise(rng, input_samples, slope, INPUT_RATE)
            rms = float(np.sqrt(np.mean(np.square(noise)))) or 1.0
            noise *= 10.0 ** (level_db / 20.0) / rms
            sf.write(source, noise.astype(np.float32), INPUT_RATE, subtype="FLOAT")
        if not os.path.isfile(f0):
            np.save(f0, np.ones(f0_frames, dtype=np.int32), allow_pickle=False)
        if not os.path.isfile(f0_voiced):
            np.save(f0_voiced, np.zeros(f0_frames, dtype=np.float64), allow_pickle=False)

        clips.append((target, feature, f0, f0_voiced))
        if not os.path.isfile(feature):
            pending.append([source, f0, f0_voiced, feature])

    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=4)
    return clips, pending


def generated_mutes(exp_dir):
    """Completed clips under ``<experiment>/mutes/`` as filelist tuples.

    Only clips with every file present are returned, so a failed embedding
    pass leaves a clip out instead of pointing the filelist at nothing.
    """
    root = os.path.join(exp_dir, MUTE_DIR)
    target_dir = os.path.join(root, "sliced_audios")
    if not os.path.isdir(target_dir):
        return []
    clips = []
    for name in sorted(os.listdir(target_dir)):
        if not (name.startswith(MUTE_PREFIX) and name.endswith(".wav")):
            continue
        stem = name[: -len(".wav")]
        clip = (
            os.path.join(target_dir, name),
            os.path.join(root, "extracted", f"{stem}.npy"),
            os.path.join(root, "f0", f"{name}.npy"),
            os.path.join(root, "f0_voiced", f"{name}.npy"),
        )
        if all(os.path.isfile(path) for path in clip):
            clips.append(clip)
    return clips
