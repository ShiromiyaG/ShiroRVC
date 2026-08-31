"""The held-out set is cropped to one length so it can be batched.

Batching the holdout through the training collate pads every item up to the
longest one in the batch, and padding silence into the mel puts a constant in
the metric that moves with whatever else lands beside it -- so the evaluation
used to run one item at a time, and spent most of its wall clock in Python:
measured on ``andre-multi``, 9.8 s per evaluation against 148 s of training.

Cropping to a shared length removes the padding instead of the batching.  What
these pin is that the crop is uniform, that it is chosen so most of the
excerpts survive it, and that the batch a model is handed still looks exactly
like one from ``TextAudioCollateMultiNSFsid``.

``train.py`` reads a run spec from ``sys.argv[1]`` at import, so the pieces are
lifted out with ``ast``, the same way ``test_overtrain_monitor.py`` does.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOP = 320
BINS = 513
PHONE_DIM = 768


@pytest.fixture(scope="module")
def excerpts():
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_HoldoutSet", "_uniform_excerpts"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted, "train.py no longer defines both"
    namespace: dict = {"torch": torch}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "train.py", "exec"), namespace
    )
    return namespace


class _Config:
    class data:
        hop_length = HOP
        sample_rate = 32000


class _Dataset:
    """Rows shaped like ``TextAudioLoaderMultiNSFsid`` hands them over."""

    def __init__(self, frames):
        self.frames = list(frames)

    def __getitem__(self, index):
        count = self.frames[index]
        return (
            torch.randn(BINS, count),
            torch.randn(1, count * HOP),
            torch.randn(count, PHONE_DIM),
            torch.randint(0, 255, (count,)),
            torch.randn(count),
            torch.LongTensor([index % 4]),
        )


def test_the_crop_lands_at_the_lower_quartile(excerpts):
    """Not at the shortest row, and not at whatever the ceiling says.

    The ceiling would leave the set to whichever recording happened to be
    longest; the shortest row would throw away most of the audio to keep one
    outlier.
    """
    dataset = _Dataset([100, 200, 300, 400, 500, 600, 700, 800])
    result = excerpts["_uniform_excerpts"](
        dataset, range(8), 10_000, _Config, batch_size=4
    )
    assert result.frames == 300
    assert len(result) == 6  # the two rows under the crop are dropped


def test_the_ceiling_is_a_ceiling(excerpts):
    dataset = _Dataset([400] * 8)
    result = excerpts["_uniform_excerpts"](
        dataset, range(8), 100, _Config, batch_size=4
    )
    assert result.frames == 100
    assert len(result) == 8


def test_a_fixed_crop_is_taken_literally(excerpts):
    """The training probe has to be cropped exactly like the holdout.

    Two mel L1 numbers over different amounts of audio are not comparable, and
    the whole point of the probe is that it is subtracted from the holdout.
    """
    dataset = _Dataset([150, 250, 350, 450])
    result = excerpts["_uniform_excerpts"](
        dataset, range(4), 300, _Config, batch_size=2, fixed=True
    )
    assert result.frames == 300
    assert len(result) == 2


def test_batches_look_like_the_training_collate(excerpts):
    dataset = _Dataset([400] * 5)
    result = excerpts["_uniform_excerpts"](
        dataset, range(5), 400, _Config, batch_size=2
    )
    shapes = []
    for index, batch in result.batches():
        (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spectrogram,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
        ) = batch
        size = phone.shape[0]
        shapes.append((index, size))
        assert phone.shape == (size, 400, PHONE_DIM)
        assert pitch.shape == pitchf.shape == (size, 400)
        assert spectrogram.shape == (size, BINS, 400)
        assert wave.shape == (size, 1, 400 * HOP)
        assert sid.shape == (size,)
        assert pitch.dtype == torch.int64 and sid.dtype == torch.int64
        # Uniform by construction, which is what lets the batch exist at all.
        assert torch.equal(phone_lengths, torch.full((size,), 400))
        assert torch.equal(spec_lengths, torch.full((size,), 400))
        assert torch.equal(wave_lengths, torch.full((size,), 400 * HOP))
    assert shapes == [(0, 2), (1, 2), (2, 1)]


def test_the_target_mel_is_computed_once_per_batch(excerpts):
    calls = []
    result = excerpts["_uniform_excerpts"](
        _Dataset([400] * 2), range(2), 400, _Config, batch_size=1
    )

    def factory():
        calls.append(1)
        return torch.zeros(1)

    for _ in range(3):
        for index, _batch in result.batches():
            result.target_mel(index, 128_000, factory)
    assert len(calls) == 2  # once per batch, not once per evaluation


def test_a_changed_output_length_invalidates_the_cached_target(excerpts):
    result = excerpts["_uniform_excerpts"](
        _Dataset([400]), range(1), 400, _Config, batch_size=1
    )
    assert result.target_mel(0, 128_000, lambda: "long") == "long"
    assert result.target_mel(0, 64_000, lambda: "short") == "short"


def test_shrinking_rebuilds_the_batches_and_drops_the_cache(excerpts):
    result = excerpts["_uniform_excerpts"](
        _Dataset([400] * 4), range(4), 400, _Config, batch_size=4
    )
    result.target_mel(0, 128_000, lambda: "stale")
    assert len(list(result.batches())) == 1
    assert result.shrink() is True
    assert result.batch_size == 2
    assert len(list(result.batches())) == 2
    assert result._target_mels == {}
    assert result.shrink() is True
    assert result.shrink() is False  # nothing left to halve


def test_rows_too_short_for_any_crop_yield_nothing(excerpts):
    result = excerpts["_uniform_excerpts"](
        _Dataset([10] * 4), range(4), 400, _Config, batch_size=2, fixed=True
    )
    assert result is None
