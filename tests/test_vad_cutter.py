"""The "New Automatic" cutter: its contract with FireRedVAD, and its wiring.

The two things worth pinning are the ones that fail *silently*.  The model
asserts 16 kHz and was calibrated on int16-magnitude samples, so feeding it the
pipeline's float [-1, 1] audio at the project rate does not raise -- it returns
almost no speech, and the run finishes with a nearly empty dataset.  The rest
is wiring: the option has to exist in both UIs and reach the same chunker the
stock cutter uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy", reason="needs numpy", exc_type=ImportError)

from rvc.train.preprocess import vad  # noqa: E402


def _tone(seconds, sr, freq=220.0, amplitude=0.3):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class _FakeEngine:
    """Stands in for ``FireRedVad``, recording what it was handed."""

    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.seen = None

    def detect(self, audio, do_postprocess=True):
        self.seen = np.asarray(audio)
        return {"dur": 0.0, "timestamps": self.timestamps}, None


@pytest.fixture
def fake_engine(monkeypatch):
    def install(timestamps):
        engine = _FakeEngine(timestamps)
        monkeypatch.setattr(vad, "_engine", lambda: engine)
        return engine

    return install


def test_audio_is_scaled_to_int16_magnitude(fake_engine):
    """The model saw int16 samples in training; float [-1, 1] finds nothing.

    Nothing raises when the scaling is wrong, which is exactly why it is
    pinned: the only symptom is a dataset that comes out nearly empty.
    """
    engine = fake_engine([(0.0, 1.0)])
    vad.segments(_tone(1.0, 16000, amplitude=0.5), 16000)

    peak = np.abs(engine.seen).max()
    assert peak > 1000, f"audio reached the model at float scale (peak {peak})"
    assert peak == pytest.approx(0.5 * 32768, rel=0.05)


def test_audio_is_resampled_to_the_models_rate(fake_engine):
    """It asserts 16 kHz rather than resampling, so the caller must."""
    engine = fake_engine([])
    vad.segments(_tone(1.0, 44100), 44100)
    assert len(engine.seen) == pytest.approx(16000, rel=0.01)


def test_clipping_rather_than_rescaling(fake_engine):
    """One sample over full scale must not quiet the rest of the file.

    Normalising to fit would divide every other frame down by the overshoot,
    which is how a single click turns a whole take unvoiced.
    """
    engine = fake_engine([])
    audio = _tone(1.0, 16000, amplitude=0.5)
    audio[0] = 4.0  # far beyond full scale
    vad.segments(audio, 16000)

    assert engine.seen[0] == pytest.approx(32767.0)
    assert np.abs(engine.seen[1:]).max() == pytest.approx(0.5 * 32768, rel=0.05)


def test_timestamps_map_back_to_the_callers_rate(fake_engine):
    """Segments come back in seconds; the caller slices samples at its own rate."""
    fake_engine([(1.0, 2.0), (3.0, 3.5)])
    spans = vad.segments(_tone(4.0, 44100), 44100)
    assert spans == [(44100, 88200), (132300, 154350)]


def test_spans_are_clamped_to_the_input(fake_engine):
    """``decision_to_segment`` can round its last frame past the final sample."""
    fake_engine([(0.0, 99.0)])
    audio = _tone(1.0, 16000)
    assert vad.segments(audio, 16000) == [(0, len(audio))]


def test_silence_falls_back_to_the_whole_file(fake_engine):
    """A file the model does not understand is still cut, not dropped.

    Returning nothing here would delete it from the dataset without a word.
    """
    fake_engine([])
    audio = _tone(2.0, 16000)
    assert vad.segments(audio, 16000) == [(0, len(audio))]


def test_empty_input_yields_no_spans(fake_engine):
    fake_engine([(0.0, 1.0)])
    assert vad.segments(np.zeros(0, dtype=np.float32), 16000) == []


def test_unavailable_reason_names_the_missing_piece(monkeypatch):
    monkeypatch.setattr(vad, "is_installed", lambda: False)
    assert "fireredvad" in vad.unavailable_reason()

    monkeypatch.setattr(vad, "is_installed", lambda: True)
    monkeypatch.setattr(vad, "has_weights", lambda: False)
    assert "weights" in vad.unavailable_reason()

    monkeypatch.setattr(vad, "has_weights", lambda: True)
    assert vad.unavailable_reason() is None


def test_the_option_is_offered_by_both_interfaces():
    """A cutter the backend accepts but no interface lists is unreachable."""
    from gui.services import catalog

    assert "New Automatic" in catalog.CUT_PREPROCESS

    source = (ROOT / "tabs" / "train" / "train.py").read_text(encoding="utf-8")
    assert '"Skip", "Simple", "Automatic", "New Automatic"' in source


def test_the_weights_are_a_download_prerequisite():
    """Offering the mode without shipping a way to get its model is a dead end."""
    from rvc.lib.tools import prerequisites_download as prereq

    folders = [entry[0] for entry in prereq.models_list]
    assert "fireredvad/VAD/" in folders
    assert prereq.folder_mapping_list["fireredvad/VAD/"].replace("\\", "/") == (
        "rvc/models/fireredvad/VAD/"
    )

    entry = next(e for e in prereq.models_list if e[0] == "fireredvad/VAD/")
    assert set(entry[1]) == {"model.pth.tar", "cmvn.ark"}
    # The directory the downloader fills has to be the one the module reads.
    assert Path(vad.MODEL_DIR).as_posix() == "rvc/models/fireredvad/VAD"


def test_both_automatic_modes_share_one_chunker():
    """``New Automatic`` changes how segments are found, not how they are cut.

    Slice geometry is what the trainer sees; if the two modes chunked
    differently, switching cutters would quietly change the dataset's shape.
    """
    source = (ROOT / "rvc" / "train" / "preprocess" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index('if cut_preprocess == "Skip"'):]
    body = body[: body.index("except Exception")]
    assert body.count("self.chunk_segments(") == 2
