"""The three selectable spectral losses have to actually work.

They are picked by name in the UI and dispatched by string in ``train.py``, so
a broken one does not fail at startup -- it fails thousands of steps in, or
worse, trains something subtly wrong.  These tests run each path end to end on
real tensors and check the gradient actually flows.

The other thing pinned here is that ``MultiScaleMelSpectrogramLoss`` honours a
configured distance.  It used to hardcode ``L1Loss``, which meant selecting
"Multi-Scale Mel Loss" silently discarded ``mel_distance``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the loss functions need torch")
pytest.importorskip("librosa", reason="rvc.train.losses imports it", exc_type=ImportError)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvc.train.losses import MultiScaleSTFTLoss  # noqa: E402
from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss  # noqa: E402

SAMPLE_RATE = 44100
SEGMENT = 17640  # what config.train.segment_size is for ChouwaGAN


@pytest.fixture
def pair():
    """A target and a close-but-imperfect reconstruction, shaped (B, 1, T)."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, SEGMENT) * 0.1
    pred = (target + torch.randn_like(target) * 0.01).requires_grad_(True)
    return target, pred


DISTANCES = {
    "l1": torch.nn.L1Loss(),
    "huber": torch.nn.SmoothL1Loss(beta=0.3),
    "mse": torch.nn.MSELoss(),
}


@pytest.mark.parametrize("distance", sorted(DISTANCES))
@pytest.mark.parametrize("safe_log", [True, False])
def test_multi_scale_mel_produces_a_finite_gradient(pair, distance, safe_log):
    target, pred = pair
    loss = MultiScaleMelSpectrogramLoss(
        sample_rate=SAMPLE_RATE, safe_log=safe_log, loss_fn=DISTANCES[distance]
    )(target, pred)
    (grad,) = torch.autograd.grad(loss, pred)
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "the loss is not connected to the prediction"


def test_multi_scale_mel_honours_the_configured_distance(pair):
    """Not just accepted -- actually used.

    Huber is L2 below beta, so on small residuals it must score well under L1.
    If the argument were ignored the two would be identical.
    """
    target, pred = pair
    def score(fn):
        return MultiScaleMelSpectrogramLoss(
            sample_rate=SAMPLE_RATE, safe_log=True, loss_fn=fn
        )(target, pred).item()

    assert score(DISTANCES["huber"]) < score(DISTANCES["l1"])


def test_multi_scale_mel_default_distance_is_a_fresh_l1(pair):
    """The default must stay L1 (HiFi-GAN relies on it) and not be shared.

    A module instance as a default argument is created once at import and
    handed to every caller that omits it.
    """
    first = MultiScaleMelSpectrogramLoss(sample_rate=SAMPLE_RATE)
    second = MultiScaleMelSpectrogramLoss(sample_rate=SAMPLE_RATE)
    assert isinstance(first.loss_fn, torch.nn.L1Loss)
    assert first.loss_fn is not second.loss_fn


def test_ms_stft_produces_a_finite_gradient(pair):
    target, pred = pair
    loss = MultiScaleSTFTLoss()(pred, target)
    (grad,) = torch.autograd.grad(loss, pred)
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_ms_stft_survives_digital_silence():
    """Spectral convergence divides by the target norm.

    A silent batch is not hypothetical -- padded and trimmed slices happen --
    and an unguarded division makes the whole run NaN from one bad batch.
    """
    silence = torch.zeros(2, 1, SEGMENT)
    pred = (silence + 1e-9).requires_grad_(True)
    loss = MultiScaleSTFTLoss()(pred, silence)
    assert torch.isfinite(loss), "silence produced a non-finite loss"


def test_losses_agree_that_a_better_reconstruction_scores_lower():
    """Sanity: all three are minimised by getting closer to the target."""
    torch.manual_seed(1)
    target = torch.randn(2, 1, SEGMENT) * 0.1
    close = target + torch.randn_like(target) * 0.005
    far = target + torch.randn_like(target) * 0.05

    for name, fn in (
        ("ms-mel", lambda a, b: MultiScaleMelSpectrogramLoss(
            sample_rate=SAMPLE_RATE, safe_log=True)(a, b)),
        ("ms-stft", lambda a, b: MultiScaleSTFTLoss()(b, a)),
    ):
        assert fn(target, close) < fn(target, far), f"{name} is not ordered"
