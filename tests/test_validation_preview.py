"""The validation preview must describe the audio it is actually showing.

The figure carries its own metadata -- a time axis and a footer naming the hop
length -- so a wrong parameter does not look wrong, it looks authoritative.
``hop_length`` defaulted to 256 while every shipped config uses sample_rate/100
(441 at 44.1 kHz), which labelled a 4.0 s excerpt as 2.3 s and printed
"Hop length: 256" underneath it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the preview path pulls in torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# ``rvc/train/utils.py`` imports its siblings flat (``from mel_processing
# import ...``), which resolves because train.py runs as a script from that
# directory.  Importing it from anywhere else has to reproduce that.
sys.path.insert(0, str(ROOT / "rvc" / "train"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rvc.train.utils import (  # noqa: E402
    VALIDATION_PREVIEW_DPI,
    VALIDATION_PREVIEW_FIGSIZE,
    plot_validation_preview_to_figure,
)

def _model_configs():
    """The per-sample-rate model configs, not the vocoder registry beside them.

    Identified by carrying a ``train`` section rather than by path, so a new
    vocoder directory is picked up without editing this.
    """
    found = []
    for path in sorted((ROOT / "rvc" / "configs").rglob("*.json")):
        try:
            if "train" in json.loads(path.read_text(encoding="utf-8")):
                found.append(path)
        except (json.JSONDecodeError, OSError):
            continue
    return found


CONFIGS = _model_configs()


@pytest.fixture
def mels():
    rng = np.random.default_rng(0)
    target = (rng.normal(0, 1, (128, 400)).astype("float32").cumsum(1) * 0.02)
    return target + rng.normal(0, 0.15, target.shape).astype("float32"), target


def _figure(mels, **kwargs):
    predicted, target = mels
    return plot_validation_preview_to_figure(
        predicted_mel=predicted, target_mel=target, epoch=1, global_step=100,
        sample_rate=44100, **kwargs
    )


def test_default_geometry_is_unchanged(mels):
    figure = _figure(mels)
    try:
        width, height = figure.canvas.get_width_height()
        # 24in x 67dpi = 1608px, the historical output.
        assert width == pytest.approx(
            VALIDATION_PREVIEW_FIGSIZE[0] * VALIDATION_PREVIEW_DPI, abs=2
        )
        assert height == pytest.approx(
            VALIDATION_PREVIEW_FIGSIZE[1] * VALIDATION_PREVIEW_DPI, abs=2
        )
        assert width == 1608
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "dpi,figsize",
    [(134, None), (201, None), (None, (36.0, 5.8)), (None, (24.0, 9.0)), (134, (30.0, 7.0))],
)
def test_geometry_follows_the_overrides(mels, dpi, figsize):
    figure = _figure(mels, dpi=dpi, figsize=figsize)
    try:
        width, height = figure.canvas.get_width_height()
        want_dpi = dpi or VALIDATION_PREVIEW_DPI
        want_size = figsize or VALIDATION_PREVIEW_FIGSIZE
        assert width == pytest.approx(want_size[0] * want_dpi, abs=2)
        assert height == pytest.approx(want_size[1] * want_dpi, abs=2)
    finally:
        plt.close(figure)


def test_hop_length_reaches_the_footer(mels):
    """The footer is the visible half of the bug; pin it directly."""
    figure = _figure(mels, hop_length=441)
    try:
        footers = [
            t.get_text() for t in figure.texts if "Hop length" in t.get_text()
        ]
        assert footers, "the preview lost its metadata footer"
        assert "441" in footers[0]
        assert "256" not in footers[0]
    finally:
        plt.close(figure)


def test_hop_length_sets_the_time_axis(mels):
    """400 frames at hop 441 / 44100 Hz is 4.0 s, not 2.3 s."""
    figure = _figure(mels, hop_length=441)
    try:
        # The first panel's x limit is the last frame's timestamp.
        upper = figure.axes[0].get_xlim()[1]
        assert upper == pytest.approx(400 * 441 / 44100, rel=0.02)
    finally:
        plt.close(figure)


def test_frequency_ticks_follow_the_mel_scale(mels):
    """A linear ruler put "5.5k" on bin 32, whose real centre is 1.02 kHz."""
    import librosa

    figure = _figure(mels, hop_length=441)
    try:
        axis = figure.axes[0]
        centres = librosa.mel_frequencies(n_mels=130, fmin=0.0, fmax=22050.0)[1:-1]
        labelled = {
            text.get_text(): position
            for position, text in zip(axis.get_yticks(), axis.get_yticklabels())
        }
        assert "1k" in labelled, "the voice range lost its tick"
        # The tick has to sit on the bin whose centre is nearest to it.
        for label, want in (("1k", 1000.0), ("4k", 4000.0), ("16k", 16000.0)):
            bin_index = int(round(labelled[label]))
            assert centres[bin_index] == pytest.approx(want, rel=0.05), label
        # And the axis must span bin space, not 0..Nyquist.
        assert axis.get_ylim()[1] == pytest.approx(127.5)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_config_declares_preview_geometry(path):
    """A missing key falls back silently, so the shipped configs must carry it."""
    train = json.loads(path.read_text(encoding="utf-8")).get("train", {})
    for key in (
        "validation_preview_dpi",
        "validation_preview_width",
        "validation_preview_height",
    ):
        assert key in train, f"{path.name} is missing {key}"
    assert train["validation_preview_dpi"] > 0
