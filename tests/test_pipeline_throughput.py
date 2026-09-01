"""Guards on the things that made preprocessing and extraction waste hardware.

Each test here pins one property that was silently violated and cost real time:
an import that dragged torch into every pool worker, a header read that decoded
the file instead, a worker count larger than the work, a batch axis that a
normalisation quietly reduced over.  All of them are the kind of regression
that shows up as "it got slower" rather than as a failure, which is why they
are asserted rather than left to a benchmark nobody runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy", reason="needs numpy", exc_type=ImportError)
sf = pytest.importorskip("soundfile", reason="needs soundfile", exc_type=ImportError)


def test_audio_io_does_not_import_torch():
    """The loaders must stay torch-free.

    They live in their own module precisely so preprocessing's worker pool can
    import them cheaply: ``multiprocessing`` spawns on Windows, so every worker
    re-imports whatever the module pulls in, and torch alone was ~2 s of that.
    Checked in a subprocess because torch is already imported in this one.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import rvc.lib.audio_io;"
        "print('torch' in sys.modules)" % ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "rvc.lib.audio_io pulled torch in; every preprocessing worker now pays "
        "for it again"
    )


def test_utils_still_re_exports_the_loaders():
    """Moving them must not break ``from rvc.lib.utils import load_audio``."""
    from rvc.lib import audio_io, utils

    for name in ("load_audio", "load_audio_16k", "load_audio_ffmpeg"):
        assert getattr(utils, name) is getattr(audio_io, name)


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.linspace(0, 2.5, int(2.5 * 16000), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    return path


def test_duration_is_read_from_the_header(wav, monkeypatch):
    """The dataset scan is serial over every file, so it must not decode.

    ``librosa.get_duration(path=...)`` did, at 187x the cost of a header read
    and growing with the dataset.  The fallbacks stay for containers soundfile
    cannot open, but nothing that soundfile *can* open may reach them.
    """
    from rvc.train.preprocess import preprocess

    def explode(*args, **kwargs):
        raise AssertionError("fell through to the decoding path")

    monkeypatch.setattr(preprocess.librosa, "get_duration", explode)
    monkeypatch.setattr(preprocess.subprocess, "run", explode)

    assert preprocess._duration_seconds(str(wav), "librosa") == pytest.approx(2.5)
    assert preprocess._duration_seconds(str(wav), "ffmpeg") == pytest.approx(2.5)


def test_an_unreadable_file_contributes_nothing(tmp_path):
    """The total is a log line; failing the run over it would be absurd."""
    from rvc.train.preprocess import preprocess

    junk = tmp_path / "not_audio.wav"
    junk.write_bytes(b"nope")
    assert preprocess._duration_seconds(str(junk), "librosa") == 0


@pytest.mark.parametrize(
    "requested, items, expected",
    [(12, 3, 3), (4, 100, 4), (1, 100, 1), (12, 0, 1), (0, 5, 1), (4, 4, 4)],
)
def test_pool_size_never_exceeds_the_work(requested, items, expected):
    from rvc.train.preprocess.preprocess import pool_size

    assert pool_size(requested, items) == expected


def test_extract_groups_clips_by_exact_length(tmp_path):
    """Buckets must be exact: a padded batch is not the same arithmetic.

    Measured at up to 95% relative error on the embeddings when clips of
    different lengths were padded into one tensor, which is why the grouping
    keys on the frame count rather than on anything approximate.
    """
    extract = pytest.importorskip(
        "rvc.train.extract.extract", reason="needs the extractor's dependencies",
        exc_type=ImportError,
    )

    infos = []
    for i, frames in enumerate([48000, 48000, 48000, 32000, 32000, 17520]):
        path = tmp_path / f"clip{i}.wav"
        sf.write(path, np.zeros(frames, dtype=np.float32), 16000)
        infos.append([str(path), "c", "f", "e"])

    buckets = extract._grouped_by_length(infos)
    assert sorted(len(b) for b in buckets) == [1, 2, 3]
    for bucket in buckets:
        frames = {sf.info(info[0]).frames for info in bucket}
        assert len(frames) == 1, "a bucket mixed lengths, so a batch would pad"


def test_an_unreadable_clip_gets_its_own_bucket(tmp_path):
    """It must fail alone rather than taking a batch of good clips with it."""
    extract = pytest.importorskip(
        "rvc.train.extract.extract", reason="needs the extractor's dependencies",
        exc_type=ImportError,
    )

    good = tmp_path / "good.wav"
    sf.write(good, np.zeros(48000, dtype=np.float32), 16000)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"nope")

    buckets = extract._grouped_by_length(
        [[str(good), "c", "f", "e"], [str(bad), "c", "f", "e"]]
    )
    assert sorted(len(b) for b in buckets) == [1, 1]


def test_waveform_normalisation_is_per_clip():
    """``do_normalize`` must not mix one clip's statistics into another's.

    It normalised over ``source.shape`` -- the whole tensor -- which is the
    same thing for the single (1, T) inputs it used to get and silently wrong
    once a batch has a leading dimension.
    """
    torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)
    from rvc.lib.utils import extract_features

    seen = {}

    class Recorder:
        def __call__(self, input_values, **kwargs):
            seen["source"] = input_values.clone()
            raise _Stop()

    class _Stop(Exception):
        pass

    quiet = torch.full((1, 400), 0.01)
    loud = torch.full((1, 400), 10.0)
    batch = torch.cat([quiet, loud], dim=0)

    with pytest.raises(_Stop):
        extract_features(Recorder(), batch, "v2", do_normalize=True)
    batched = seen["source"]

    with pytest.raises(_Stop):
        extract_features(Recorder(), quiet, "v2", do_normalize=True)
    alone = seen["source"]

    assert torch.allclose(batched[0], alone[0], atol=1e-5), (
        "a clip's normalisation changed because of what it was batched with"
    )
