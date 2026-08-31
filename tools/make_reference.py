"""Build the three ``logs/reference`` files the trainer's preview reads, from a wav.

The preview that runs every ``preview_interval`` steps needs a fixed input, and
by default it takes whatever the train loader hands it first -- which changes
whenever the dataset or the shuffle does, so two runs are not comparable by
eye.  ``get_reference_sample`` will use a custom one instead when all three of
these exist:

    logs/reference/ref_feats.npy    (T_feat, 768)  embedder features
    logs/reference/ref_f0c.npy      (T_f0,)        f0 as coarse bin indices
    logs/reference/ref_f0f.npy      (T_f0,)        f0 in Hz

and, optional but written by default:

    logs/reference/ref_audio.wav                   the ground truth to compare

They are the same three arrays the dataset extraction writes to
``extracted/``, ``f0/`` and ``f0_voiced/``, so this runs the same code with the
same defaults rather than reimplementing it -- a reference extracted a
different way is measuring the extractor, not the model.

    python tools/make_reference.py my_singer.wav
    python tools/make_reference.py my_singer.wav --seconds 6
    python tools/make_reference.py a.wav --f0-method fcpe --embedder spin_v2

Two properties of the custom path are worth knowing before you use it, because
neither is an error and neither is visible in the log:

* **Without ``ref_audio.wav`` the preview loses its "original" panel.**  The
  three-panel PNG (generated mel / original mel / difference) needs something to
  subtract; with no ground truth the trainer writes the generated waveform
  alone.  This writes the wav by default for that reason -- ``--no-audio``
  skips it.
* **The speaker is pinned to 0.**  ``get_reference_sample`` hardcodes
  ``sid = 0`` on this path, so on a multi-speaker model the preview is rendered
  in speaker 0's voice no matter whose audio you extracted.  With a
  multi-speaker model the difference panel therefore shows the *timbre* gap as
  well as the reconstruction gap, unless the audio is speaker 0's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: What ``get_reference_sample`` looks for, and in which order this writes them.
REFERENCE_FILES = ("ref_feats.npy", "ref_f0c.npy", "ref_f0f.npy")

#: The extractor runs everything at 16 kHz with a 160-sample hop, so one f0
#: frame is 10 ms regardless of the model's own sample rate.
F0_FRAMES_PER_SECOND = 100


def build(
    source: Path,
    destination: Path,
    f0_method: str,
    embedder: str,
    embedder_custom: str | None,
    device: str,
    seconds: float | None,
    write_audio: bool,
    sample_rate: int,
) -> None:
    import soundfile as sf

    from rvc.lib.utils import load_audio, load_audio_16k, load_embedder_model, extract_features
    from rvc.train.extract.extract import FeatureInput
    import torch

    audio = load_audio_16k(str(source))
    available = audio.size / 16000
    if seconds:
        audio = audio[: int(seconds * 16000)]
        if available > seconds:
            print(
                f"[REFERENCE] {source.name} is {available:.1f}s; using the first "
                f"{seconds:g}s (--seconds 0 to keep all of it)"
            )
    elif available > 30:
        print(
            f"[REFERENCE] warning: extracting all {available:.1f}s. The preview "
            "renders the whole reference on every interval, so this makes every "
            "preview that much slower for the rest of the run."
        )
    if audio.size < 16000:
        raise SystemExit(
            f"{source} is {audio.size / 16000:.2f}s of audio; use at least a second."
        )

    # --- f0, both forms.  ``coarse_f0`` is the *only* thing that should ever
    # produce ref_f0c: it is a mel-scaled bin index, not a rounded frequency,
    # and the two are impossible to tell apart by looking at the file.
    pitch = FeatureInput(f0_method=f0_method, device=device)
    f0_fine = np.asarray(pitch.compute_f0(audio), dtype=np.float32)
    f0_coarse = pitch.coarse_f0(f0_fine)

    # --- embedder features, at half the f0 rate; the trainer repeats them by 2.
    model, do_normalize = load_embedder_model(embedder, embedder_custom)
    model = model.to(device).float().eval()
    with torch.inference_mode():
        frames = torch.from_numpy(audio).to(device).float().view(1, -1)
        feats = extract_features(model, frames, "v2", do_normalize=do_normalize)
    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isfinite(feats).all():
        raise SystemExit("the embedder produced NaN features; try another file")

    _check(feats, f0_coarse, f0_fine)

    destination.mkdir(parents=True, exist_ok=True)
    for name, array in zip(REFERENCE_FILES, (feats, f0_coarse, f0_fine)):
        np.save(destination / name, array, allow_pickle=False)

    usable_frames = min(feats.shape[0] * 2, f0_fine.size)
    if write_audio:
        # Re-read at the model's rate rather than upsampling the 16 kHz copy the
        # extractor used: the preview compares mel spectrograms up to Nyquist,
        # and a 16 kHz source resampled to 44.1 kHz is empty above 8 kHz -- the
        # difference panel would then show the model inventing the whole top
        # half of the band, which is an artefact of this tool, not the model.
        wave = load_audio(str(source), sample_rate)
        if seconds:
            wave = wave[: int(seconds * sample_rate)]
        # One f0 frame is 10 ms and so is one hop at every configured rate, so
        # the frame count converts directly.
        wave = wave[: int(usable_frames * sample_rate / F0_FRAMES_PER_SECOND)]
        sf.write(destination / "ref_audio.wav", wave, sample_rate)

    voiced = f0_fine[f0_fine > 0]
    print(f"[REFERENCE] wrote {destination}")
    print(f"  ref_feats.npy  {feats.shape} {feats.dtype}")
    print(f"  ref_f0c.npy    {f0_coarse.shape} {f0_coarse.dtype}  bins {f0_coarse.min()}-{f0_coarse.max()}")
    print(f"  ref_f0f.npy    {f0_fine.shape} {f0_fine.dtype}  ", end="")
    if voiced.size:
        print(f"{voiced.mean():.0f} Hz mean over {voiced.size / f0_fine.size:.0%} voiced")
    else:
        print("no voiced frames")
    if write_audio:
        print(f"  ref_audio.wav  {sample_rate} Hz, for the three-panel preview")
    else:
        print("  ref_audio.wav  skipped -- the preview will have no mel comparison")
    print(
        f"  the trainer will use {usable_frames} frames "
        f"({usable_frames / F0_FRAMES_PER_SECOND:.2f}s)"
    )


def _check(feats: np.ndarray, f0_coarse: np.ndarray, f0_fine: np.ndarray) -> None:
    """Catch the mistakes that produce a working file with the wrong contents.

    None of these can be caught downstream: ``get_reference_sample`` loads the
    three arrays with no validation at all, so a coarse curve written to
    ``ref_f0f`` renders a reference an octave and a half flat and reports
    nothing.  They are the same shape and both numeric.
    """
    if feats.ndim != 2:
        raise SystemExit(f"features should be (frames, channels), got {feats.shape}")
    if f0_coarse.shape != f0_fine.shape:
        raise SystemExit(
            f"the two f0 curves disagree: {f0_coarse.shape} vs {f0_fine.shape}"
        )
    voiced = f0_fine[f0_fine > 0]
    if voiced.size and not (50.0 <= voiced.mean() <= 1100.0):
        raise SystemExit(
            f"ref_f0f averages {voiced.mean():.1f}, which is not Hz -- the coarse "
            "curve is bin indices and belongs in ref_f0c"
        )
    ratio = f0_fine.size / max(1, feats.shape[0])
    if not 1.8 <= ratio <= 2.2:
        raise SystemExit(
            f"f0 has {ratio:.2f}x the frames of the features, expected ~2.0; the "
            "trainer repeats the features by exactly 2 to line them up"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path, help="any audio file; it is resampled to 16 kHz")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs" / "reference",
        help="where to write the three files (default: logs/reference)",
    )
    parser.add_argument(
        "--f0-method", default="rmvpe", choices=("rmvpe", "crepe", "crepe-tiny", "fcpe"),
        help="must match what the dataset was extracted with (default: rmvpe)",
    )
    parser.add_argument(
        "--embedder", default="contentvec",
        help="must match the dataset's embedder, or the features mean something "
             "else to the model (default: contentvec)",
    )
    parser.add_argument("--embedder-custom", default=None)
    parser.add_argument(
        "--no-audio", action="store_true",
        help="skip ref_audio.wav; the preview then has no mel comparison",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100,
        help="rate to write ref_audio.wav at; match the model (default: 44100)",
    )
    parser.add_argument("--device", default=None, help="default: cuda:0 when available")
    # Capped by default.  The preview re-renders the whole reference every
    # interval for the entire run, so the cost of a long one is paid thousands
    # of times, and nothing downstream trims it -- point this at a full
    # recording without a cap and every preview drags for the rest of training.
    parser.add_argument(
        "--seconds", type=float, default=10.0,
        help="how much of the input to use (default: 10; 0 keeps all of it)",
    )
    args = parser.parse_args()

    if not args.wav.is_file():
        raise SystemExit(f"{args.wav} does not exist")

    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    build(
        source=args.wav,
        destination=args.output_dir,
        f0_method=args.f0_method,
        embedder=args.embedder,
        embedder_custom=args.embedder_custom,
        device=device,
        seconds=args.seconds,
        write_audio=not args.no_audio,
        sample_rate=args.sample_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
