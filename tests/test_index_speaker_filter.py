"""Building the retrieval index from one speaker instead of the whole dataset.

On a multispeaker model the whole-dataset index is a pool of every voice, so
retrieval can hand a conversion targeting speaker 0 a frame of speaker 2's
articulation.  These tests pin the filter itself, the naming that keeps the
per-speaker index from replacing the full one, and the two independent speaker
listings (backend and Qt catalog) that have to agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy", reason="needs numpy", exc_type=ImportError)
extract_index = pytest.importorskip(
    "rvc.train.process.extract_index",
    reason="needs faiss and scikit-learn",
    exc_type=ImportError,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("0_1_2.npy", 0),
        ("12_0_7.npy", 12),
        ("3_take_1.npy", 3),
        ("mute.npy", None),
        ("speaker_1.npy", None),
        ("", None),
    ],
)
def test_speaker_is_the_leading_integer(name, expected):
    """Preprocessing names slices ``<sid>_<file>_<slice>``; anything else is
    unattributable rather than guessed at."""
    assert extract_index.speaker_of(name) == expected


@pytest.fixture
def experiment(tmp_path):
    """An experiment with three speakers and differing frame counts."""
    feature_dir = tmp_path / "extracted"
    feature_dir.mkdir()
    counts = {0: 3, 1: 5, 2: 2}
    rng = np.random.default_rng(0)
    for sid, files in counts.items():
        for i in range(files):
            array = rng.standard_normal((10, 768)).astype(np.float32)
            np.save(feature_dir / f"{sid}_{i}_0.npy", array)
    # Something unattributable, to prove it is neither claimed by a speaker nor
    # allowed to break the listing.
    np.save(feature_dir / "mute.npy", rng.standard_normal((10, 768)).astype(np.float32))
    return tmp_path


def test_available_speakers_reads_the_features(experiment):
    """From the files that exist, not from config.json: an index can only be
    built out of frames that were actually extracted."""
    assert extract_index.available_speakers(str(experiment)) == [0, 1, 2]


def test_available_speakers_on_a_bare_directory(tmp_path):
    assert extract_index.available_speakers(str(tmp_path)) == []


def test_loading_one_speaker_keeps_only_its_frames(experiment):
    feature_dir = str(experiment / "extracted")
    everything, _, _ = extract_index._load_features(feature_dir, None)
    one, _, _ = extract_index._load_features(feature_dir, None, speaker_id=1)

    assert one.shape[0] == 5 * 10
    assert everything.shape[0] == (3 + 5 + 2 + 1) * 10  # mute.npy included
    assert one.shape[1] == everything.shape[1]


def test_every_speakers_frames_are_accounted_for(experiment):
    """The parts must add up to the whole, minus the unattributable file."""
    feature_dir = str(experiment / "extracted")
    total = extract_index._load_features(feature_dir, None)[0].shape[0]
    per_speaker = sum(
        extract_index._load_features(feature_dir, None, speaker_id=sid)[0].shape[0]
        for sid in extract_index.available_speakers(str(experiment))
    )
    assert per_speaker == total - 10  # mute.npy belongs to no speaker


def test_utt_ids_stay_distinct_within_a_filtered_build(experiment):
    """Continuity keys on ``utt``; collapsing two files onto one id would let
    the bonus fire across a recording boundary."""
    utts = extract_index._load_features(
        str(experiment / "extracted"), None, speaker_id=1
    )[1]
    assert len(np.unique(utts)) == 5


def test_a_per_speaker_index_does_not_replace_the_full_one(experiment, monkeypatch):
    """Both live side by side; which one suits a conversion is decided at
    inference, not here."""
    written = []
    monkeypatch.setattr(
        extract_index.faiss, "write_index", lambda index, path: written.append(path)
    )
    monkeypatch.setattr(extract_index.index_meta, "write", lambda path, meta: None)

    extract_index.main(str(experiment), "Auto", "l2", speaker_id=1)
    extract_index.main(str(experiment), "Auto", "l2")

    name = experiment.name
    assert [Path(p).name for p in written] == [f"{name}_spk1.index", f"{name}.index"]


def test_the_sidecar_records_which_speaker(experiment, monkeypatch):
    """A file on disk must say what it was built from; the name alone is a
    convention a user can rename away."""
    captured = {}
    monkeypatch.setattr(extract_index.faiss, "write_index", lambda index, path: None)
    monkeypatch.setattr(
        extract_index.index_meta, "write",
        lambda path, meta: captured.update(meta.extra),
    )

    extract_index.main(str(experiment), "Auto", "l2", speaker_id=2)
    assert captured["speaker_id"] == 2

    extract_index.main(str(experiment), "Auto", "l2")
    assert captured["speaker_id"] == -1  # a number, so older readers still parse it


def test_an_absent_speaker_is_refused_before_anything_is_written(experiment, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("wrote an index for a speaker that has no features")

    monkeypatch.setattr(extract_index.faiss, "write_index", explode)
    assert extract_index.main(str(experiment), "Auto", "l2", speaker_id=9) == 1


def test_the_gui_catalog_agrees_with_the_backend(experiment, monkeypatch):
    """Two listings exist so the GUI's first paint stays free of faiss; they
    must not drift apart."""
    from gui.services import catalog, paths

    monkeypatch.setattr(paths, "LOGS_DIR", experiment.parent)
    assert catalog.list_experiment_speakers(experiment.name) == (
        extract_index.available_speakers(str(experiment))
    )
    assert catalog.list_experiment_speakers("no-such-model") == []
    assert catalog.list_experiment_speakers("") == []


def test_all_means_all_through_the_core_entry_point(monkeypatch):
    """The sentinel travels as a word: an empty argv entry can be dropped by a
    shell, and would read back as speaker 0 if it were not."""
    import core

    seen = []

    class _Completed:
        returncode = 0

    def fake_run(command, *args, **kwargs):
        seen.append(command)
        return _Completed()

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    for passed, expected in ((None, "all"), ("", "all"), ("all", "all"),
                             (3, "3"), ("2", "2"), (0, "0")):
        seen.clear()
        core.run_index_script("m", "Auto", "l2", passed)
        assert seen[0][-1] == expected, f"{passed!r} became {seen[0][-1]!r}"


def test_both_interfaces_offer_the_option():
    """A backend flag no interface exposes is unreachable."""
    gradio = (ROOT / "tabs" / "train" / "train.py").read_text(encoding="utf-8")
    assert "index_single_speaker" in gradio and "index_speaker" in gradio

    qt = (ROOT / "gui" / "views" / "training.py").read_text(encoding="utf-8")
    assert "index_single_speaker" in qt and "_refresh_index_speakers" in qt
