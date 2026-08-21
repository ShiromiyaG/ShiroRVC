"""The filelist must be portable.

Absolute paths bake the machine that ran the extraction into the dataset: move
``logs/<model>/`` to another drive, another user's install or a container and
every entry breaks -- as a FileNotFoundError thousands of steps into training,
long after the cause.

Written with ``ast``-free plain imports because both modules are light; neither
pulls torch at import time.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The trainer puts rvc/train on sys.path and imports its neighbours flatly
# (`from mel_processing import ...`), so `rvc.train.utils` is not importable by
# its dotted name.  Mirror the trainer's layout rather than change the module.
sys.path.insert(0, str(ROOT / "rvc" / "train"))

# Both modules reach the backend (preparing_files loads the GPU config), so
# skip cleanly rather than fail when the training environment is absent.
preparing_files = pytest.importorskip(
    "rvc.train.extract.preparing_files", reason="the training backend is optional"
)
train_utils = pytest.importorskip(
    "utils", reason="the training backend is optional"
)


def test_both_modules_agree_on_the_root():
    """A mismatch here would resolve every path to the wrong place."""
    assert os.path.normpath(preparing_files.APPLICATION_ROOT) == os.path.normpath(
        train_utils.APPLICATION_ROOT
    ) == os.path.normpath(str(ROOT))


def test_paths_are_written_relative_with_posix_separators():
    produced = preparing_files.relative_to_root(
        str(ROOT / "logs" / "demo" / "sliced_audios" / "0_1.wav")
    )
    assert produced == "logs/demo/sliced_audios/0_1.wav"
    assert not os.path.isabs(produced)
    assert "\\" not in produced


def test_paths_outside_the_root_stay_absolute():
    """Better an absolute entry than a ../../.. chain assuming a layout."""
    outside = os.path.abspath(os.path.join(os.sep, "elsewhere", "dataset", "a.wav"))
    produced = preparing_files.relative_to_root(outside)
    assert os.path.isabs(produced.replace("/", os.sep))


def _write(lines: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


def test_relative_entries_resolve_to_absolute():
    relative = "logs/demo/sliced_audios/0_1.wav"
    path = _write([f"{relative}|{relative}|{relative}|{relative}|3"])
    try:
        rows = train_utils.load_filepaths_and_text(path)
    finally:
        os.unlink(path)

    assert len(rows) == 1
    row = rows[0]
    assert all(os.path.isabs(field) for field in row[:4])
    assert row[0] == os.path.normpath(str(ROOT / relative))
    # The speaker id is not a path and must survive untouched.
    assert row[4] == "3"


def test_absolute_entries_still_load():
    """Filelists from before this change must keep working, unmigrated."""
    absolute = str(ROOT / "logs" / "demo" / "sliced_audios" / "0_1.wav")
    path = _write([f"{absolute}|{absolute}|{absolute}|{absolute}|0"])
    try:
        rows = train_utils.load_filepaths_and_text(path)
    finally:
        os.unlink(path)

    assert rows[0][0] == os.path.normpath(absolute)
    assert rows[0][4] == "0"


def test_blank_lines_are_skipped():
    """A trailing newline must not become a row with one empty field."""
    relative = "logs/demo/sliced_audios/0_1.wav"
    path = _write([f"{relative}|{relative}|{relative}|{relative}|1", "", "  "])
    try:
        rows = train_utils.load_filepaths_and_text(path)
    finally:
        os.unlink(path)

    assert len(rows) == 1


def test_round_trip_through_both_sides():
    """What the extractor writes is what the trainer reads back."""
    original = str(ROOT / "logs" / "demo" / "f0" / "0_1.wav.npy")
    stored = preparing_files.relative_to_root(original)
    path = _write([f"{stored}|{stored}|{stored}|{stored}|9"])
    try:
        rows = train_utils.load_filepaths_and_text(path)
    finally:
        os.unlink(path)

    assert rows[0][2] == os.path.normpath(original)
