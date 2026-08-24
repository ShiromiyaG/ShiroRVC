"""What the init panel says about the LR schedule.

``lr_final_ratio`` defines the decay by its endpoint and supersedes the
per-step/per-epoch gamma completely -- the shipped gamma is a no-op at every
realistic run length.  The panel printed that dead number anyway
("exp decay step, gamma: 0.977708"), which is how it gets tuned by someone
reasonably assuming it does something, and a separate ``[INIT]`` line existed
solely to say the number it had just printed was unused.  The panel now states
the endpoint that is in force, and the extra line is gone.

The gamma is still the live setting when there is no ratio, and it must still
be shown there.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# ``utils`` imports ``mel_processing`` as a bare module, so its own directory
# has to be importable too -- the same two lines ``test_band_weighted_loss``
# needs.
sys.path.insert(0, str(ROOT / "rvc" / "train"))

pytest.importorskip("torch", reason="the trainer's utils import torch", exc_type=ImportError)

from rvc.train.utils import print_init_setup  # noqa: E402


def panel(capsys, scheduler, gamma=0.977708, ratio=None):
    # The rich console is a module-level singleton bound to ``sys.stderr`` at
    # first use, so it has to be dropped for each call or it keeps writing to
    # the real stderr that pytest replaced.
    import rvc.lib.terminal as terminal

    terminal._console = None
    print_init_setup(
        warmup_duration=5,
        rank=0,
        use_warmup=False,
        config=None,
        optimizer_choice_g="AdamW",
        optimizer_choice_d="AdamW",
        lr_scheduler=scheduler,
        exp_decay_gamma=gamma,
        spectral_loss="L1 Mel Loss",
        lr_final_ratio=ratio,
    )
    captured = capsys.readouterr()
    return captured.err + captured.out


@pytest.mark.parametrize(
    "scheduler", ["exp decay step", "exp decay epoch", "cosine annealing"]
)
def test_gamma_is_hidden_when_a_ratio_supersedes_it(capsys, scheduler):
    out = panel(capsys, scheduler, ratio=0.25)
    assert "gamma" not in out
    assert "0.977708" not in out


def test_the_endpoint_is_shown_instead(capsys):
    out = panel(capsys, "exp decay step", ratio=0.25)
    assert "0.25x" in out


@pytest.mark.parametrize("scheduler", ["exp decay step", "exp decay epoch"])
def test_gamma_is_shown_when_it_is_the_live_setting(capsys, scheduler):
    out = panel(capsys, scheduler, ratio=None)
    assert "gamma" in out
    assert "0.977708" in out


def test_a_disabled_scheduler_says_so_either_way(capsys):
    assert "Disabled" in panel(capsys, "none", ratio=0.25)
    assert "Disabled" in panel(capsys, "none", ratio=None)


def test_cosine_without_a_ratio_is_unchanged(capsys):
    out = panel(capsys, "cosine annealing", ratio=None)
    assert "cosine annealing" in out
    assert "gamma" not in out, "cosine never used the gamma"


def test_the_superseded_init_line_is_gone():
    """The announcement the panel made redundant, pinned so it stays removed."""
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    # Only f-strings: the removed message was one, and docstrings never are, so
    # this cannot be tripped by prose explaining why the message went away.
    messages = [
        part.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.JoinedStr)
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    ]
    joined = "\n".join(messages)
    assert "is unused" not in joined
    assert "the starting LR" not in joined
