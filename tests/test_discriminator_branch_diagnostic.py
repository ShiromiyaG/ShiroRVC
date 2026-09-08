"""The per-branch discriminator diagnostic.

The bug these pin: ``loss_disc`` is the sum over every head, so a head that has
stopped separating (real and fake halves equal) and a head that has learned the
*opposite* of its job (fake half below real) both leave the aggregate where it
was.  A pretrain ran 87k steps with four of nine heads scoring the generator's
output above real audio, and nothing in TensorBoard said so.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="the discriminator needs torch", exc_type=ImportError)
pytest.importorskip("librosa", reason="rvc.train.losses imports it", exc_type=ImportError)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rvc" / "train"))

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined  # noqa: E402

SR = 32000


def _build(**kwargs):
    return MPD_MSD_Combined(False, sample_rate=SR, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(version="v2"),
        dict(version="v3"),
        dict(version="v4", use_univhd=True, use_san=True),
        dict(version="v3", use_msd=False),
        dict(version="v4", periods=[]),
    ],
)
def test_a_label_per_branch_in_branch_order(kwargs):
    """The labels name a per-branch series, so a mislabelled one points the
    reader at the wrong head -- worse than having none."""
    net = _build(**kwargs)
    assert len(net.branch_labels) == len(net.discriminators)


def test_the_labels_describe_what_was_actually_built():
    net = _build(version="v4", use_univhd=True)
    assert net.branch_labels[0] == "msd"
    assert net.branch_labels[1:5] == tuple(f"period_{p}" for p in net.periods)
    assert net.branch_labels[-1] == "univhd"
    assert _build(version="v3", use_msd=False).branch_labels[0].startswith("period_")


def test_the_separation_says_which_way_a_head_is_pointing():
    """Positive is a head doing its job, ~0 is one that has stopped separating,
    negative is one scoring the generator's output above real audio.

    Not derived from the loss halves: each of those is a "how wrong" measure,
    so a head that is confidently wrong about fakes and one that is right about
    them move that term the same way.
    """
    from tests._branch_separation import branch_separation

    good = ([torch.full((1, 4), 2.0)], [torch.full((1, 4), -1.0)])
    dead = ([torch.full((1, 4), 0.5)], [torch.full((1, 4), 0.5)])
    backwards = ([torch.full((1, 4), -1.0)], [torch.full((1, 4), 2.0)])

    assert branch_separation(*good)[0].item() == pytest.approx(3.0)
    assert branch_separation(*dead)[0].item() == pytest.approx(0.0)
    assert branch_separation(*backwards)[0].item() == pytest.approx(-3.0)


def test_the_separation_reads_the_function_output_under_san():
    """A SAN head returns ``(function, direction)``; the generator is scored by
    the function output, so that is the one whose separation means anything."""
    from tests._branch_separation import branch_separation

    real = [[torch.full((1, 4), 2.0), torch.full((1, 4), -9.0)]]
    fake = [[torch.full((1, 4), 0.0), torch.full((1, 4), 9.0)]]
    assert branch_separation(real, fake)[0].item() == pytest.approx(2.0)


def test_the_separation_does_not_carry_a_graph():
    """It is logged every step; holding the discriminator graph alive for a
    diagnostic would be a leak the size of the discriminator."""
    from tests._branch_separation import branch_separation

    real = [torch.full((1, 4), 2.0, requires_grad=True)]
    fake = [torch.full((1, 4), 0.0, requires_grad=True)]
    assert not branch_separation(real, fake).requires_grad
