"""A saturated R1 branch has to announce itself.

The per-branch R1 controller can be defeated in one way it cannot correct: a
branch pinned at ``maximum`` while still short of target has no actuator range
left. It is invisible in every aggregate -- ``GAN/r1_scale`` and
``GAN/r1_to_disc_ratio`` are means, and a pinned branch hides behind the ones
that overshoot -- so it has now been found by hand twice on the 44.1 kHz
pretrain, both times ``stft_2048``:

    ceiling 20.0   pinned step 8k -> 28k, share 0.31 against target 1.0
    ceiling 100.0  pinned from ~11k, share 0.688, every other branch 1.20-1.81

The ceiling is 400 for the reason in ``_R1StrengthController.__init__``; these
pin the detection, which is the part that generalises past this one branch.


``train.py`` reads a run spec from ``sys.argv[1]`` at import, so the class is
lifted out with ``ast``, the same way ``test_overtrain_monitor.py`` does.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BRANCHES = ("stft_512", "stft_2048")


@pytest.fixture(scope="module")
def controller_class():
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    node = next(
        (
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "_R1StrengthController"
        ),
        None,
    )
    assert node is not None, "train.py no longer defines _R1StrengthController"
    namespace: dict = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<train.py>", "exec"), namespace)
    return namespace["_R1StrengthController"]


@pytest.fixture
def _controller(controller_class):
    def build(**kw):
        kw.setdefault("branch_names", BRANCHES)
        kw.setdefault("target_ratio", 1.0)
        return controller_class(**kw)

    return build


def _drive(controller, branch, gain, events):
    """Run a branch whose share is ``gain`` per unit of strength.

    That proportionality is the real one: the R1 gradient norm scales with the
    penalty weight and the discriminative norm it is measured against does not.
    So a branch reaches target at ``scale = target / gain``, and one whose
    ``gain * maximum`` is below target can never reach it at all -- which is the
    condition being tested.
    """
    for _ in range(events):
        controller.observe_discriminator(branch, 1.0)
        controller.observe_r1(branch, gain * controller.scale_for(branch))


def test_a_branch_short_of_target_at_the_ceiling_is_reported(_controller):
    controller = _controller(maximum=10.0)
    # 0.033 * 10 = a share of 0.33 even at full strength: unreachable.
    _drive(controller, "stft_2048", 0.033, 800)

    assert controller.scale_for("stft_2048") == pytest.approx(10.0, rel=1e-6)
    reported = controller.newly_saturated()
    assert [name for name, _ in reported] == ["stft_2048"]
    assert reported[0][1] == pytest.approx(0.33, rel=0.05)


def test_it_is_reported_once_not_every_step(_controller):
    controller = _controller(maximum=10.0)
    _drive(controller, "stft_2048", 0.033, 800)

    assert controller.newly_saturated()
    _drive(controller, "stft_2048", 0.033, 400)
    assert controller.newly_saturated() == []


def test_a_branch_that_recovers_can_warn_again(_controller):
    """Pinning is a state, not a one-off event."""
    controller = _controller(maximum=10.0)
    _drive(controller, "stft_2048", 0.033, 800)
    assert controller.newly_saturated()

    # Overshoot pulls it off the ceiling and clears the latch.
    _drive(controller, "stft_2048", 4.0, 400)
    assert controller.scale_for("stft_2048") < 10.0

    _drive(controller, "stft_2048", 0.033, 1200)
    assert [name for name, _ in controller.newly_saturated()] == ["stft_2048"]


def test_a_branch_climbing_toward_a_reachable_target_does_not_trip_it(_controller):
    """The ceiling being touched is not the failure; being stuck under it is."""
    controller = _controller(maximum=10.0)
    # Reaches target at scale 5, climbing from 1: never touches the ceiling.
    _drive(controller, "stft_512", 0.2, 800)

    assert controller.scale_for("stft_512") < 10.0
    assert controller.newly_saturated() == []


def test_a_branch_pinned_at_the_floor_is_not_reported_as_saturated(_controller):
    """The floor is a separate condition and this warning is about the ceiling."""
    controller = _controller(maximum=10.0, minimum=0.01)
    _drive(controller, "stft_2048", 500.0, 800)

    assert controller.scale_for("stft_2048") == pytest.approx(0.01, rel=1e-6)
    assert controller.newly_saturated() == []


# --------------------------------------------------------------------------
# The floor is the same failure at the other bound, and it went unreported for
# 55k steps of the 2026-08-25 pretrain because only the ceiling had a warning:
# ``period_2`` pinned at 0.005 from step 48.4k at a share of 6.26, ``period_11``
# from 52.1k at 2.18, both against a target of 1.0.  Neither is visible in
# ``GAN/r1_scale`` or ``GAN/r1_to_disc_ratio`` -- they are means, and the mean
# read 1.96 while two of seven branches had no authority at all.


def test_a_branch_over_target_at_the_floor_is_reported(_controller):
    controller = _controller(maximum=10.0, minimum=0.01)
    # A share of 5.0 even at the weakest strength the bound allows.
    _drive(controller, "stft_2048", 500.0, 800)

    assert controller.scale_for("stft_2048") == pytest.approx(0.01, rel=1e-6)
    reported = controller.newly_floored()
    assert [name for name, _ in reported] == ["stft_2048"]
    assert reported[0][1] == pytest.approx(5.0, rel=0.05)


def test_the_floor_is_reported_once_not_every_step(_controller):
    controller = _controller(maximum=10.0, minimum=0.01)
    _drive(controller, "stft_2048", 500.0, 800)

    assert controller.newly_floored()
    _drive(controller, "stft_2048", 500.0, 400)
    assert controller.newly_floored() == []


def test_a_branch_recovering_off_the_floor_can_warn_again(_controller):
    controller = _controller(maximum=10.0, minimum=0.01)
    _drive(controller, "stft_2048", 500.0, 800)
    assert controller.newly_floored()

    # Undershoot pulls it off the floor and clears the latch.
    _drive(controller, "stft_2048", 0.05, 400)
    assert controller.scale_for("stft_2048") > 0.01

    _drive(controller, "stft_2048", 500.0, 1200)
    assert [name for name, _ in controller.newly_floored()] == ["stft_2048"]


def test_a_branch_descending_toward_a_reachable_target_does_not_trip_it(_controller):
    """Touching the floor is not the failure; being stuck over target on it is."""
    controller = _controller(maximum=10.0, minimum=0.01)
    # Reaches target at scale 0.05, descending from 1: never touches the floor.
    _drive(controller, "stft_512", 20.0, 800)

    assert controller.scale_for("stft_512") > 0.01
    assert controller.newly_floored() == []


def test_a_branch_pinned_at_the_ceiling_is_not_reported_as_floored(_controller):
    """The two counters must not read each other's failure."""
    controller = _controller(maximum=10.0, minimum=0.01)
    _drive(controller, "stft_2048", 0.033, 800)

    assert controller.newly_saturated()
    assert controller.newly_floored() == []


def test_the_floor_clears_the_period_branches_the_run_actually_measured(controller_class):
    """0.005 left ``period_2`` at a share of 6.26; the share is linear in the strength."""
    needed = 0.005 / 6.26
    assert controller_class().minimum < needed / 2


def test_the_ceiling_clears_the_2048_branch_the_run_actually_measured(controller_class):
    """100.0 left it at a share of 0.688; the share is linear in the strength."""
    needed = 100.0 / 0.688
    assert controller_class().maximum > needed * 2


def test_saturation_state_does_not_break_the_resume_contract(_controller):
    controller = _controller(maximum=10.0)
    _drive(controller, "stft_2048", 0.033, 800)

    restored = _controller(maximum=400.0)
    assert restored.load_state_dict(controller.state_dict)
    # The strength carries over and the raised ceiling gives it room again.
    assert restored.scale_for("stft_2048") == pytest.approx(10.0, rel=1e-6)
    assert math.exp(restored.log_scale["stft_2048"]) < restored.maximum
    assert restored.newly_saturated() == []


def test_a_lowered_floor_gives_a_pinned_branch_room_on_resume(_controller):
    """The fix has to be applicable to a run already in progress.

    This is the 2026-08-25 pretrain's situation exactly: two branches sat on the
    old floor, the checkpoint carries that strength, and a resume under a lower
    floor has to walk them down rather than reset them.
    """
    controller = _controller(maximum=10.0, minimum=0.005)
    _drive(controller, "stft_2048", 1252.0, 800)
    assert controller.scale_for("stft_2048") == pytest.approx(0.005, rel=1e-6)

    restored = _controller(maximum=10.0, minimum=1e-4)
    assert restored.load_state_dict(controller.state_dict)
    # Picks up where it stopped, with room below it and no warning yet.
    assert restored.scale_for("stft_2048") == pytest.approx(0.005, rel=1e-6)
    assert restored.newly_floored() == []

    # And walks to the target it could not reach before.
    _drive(restored, "stft_2048", 1252.0, 400)
    assert restored.ratio_for("stft_2048") == pytest.approx(1.0, rel=0.1)
    assert restored.scale_for("stft_2048") == pytest.approx(1.0 / 1252.0, rel=0.15)


# --------------------------------------------------------------------------
# Seeding the strength from the first measurement: tried, and wrong.
#
# Kept as a test rather than only a comment, because "one sample names the
# answer" is a genuinely appealing argument and the code it produces looks
# correct.  What defeats it is that the first R1 event lands at step ~16 with
# the discriminator at its initialisation.


def test_the_strength_is_earned_by_walking_not_seeded(_controller):
    """A first measurement must move the strength by ``lr``, not by its log.

    Seeding produced scales of 970 and 1973 for branches needing 0.2 and 0.3,
    and 19.8 for the one needing 145.3 -- wrong by up to 6577x, and wrong in the
    wrong direction for the branch it existed for.  ``disc_headroom`` sat at
    0.383 flat against 0.625 without it.
    """
    controller = _controller(maximum=10000.0)
    _drive(controller, "stft_2048", 1.0 / 145.0, 1)

    # One event, one ``lr``-sized step in log space: nowhere near 145.
    assert controller.scale_for("stft_2048") == pytest.approx(
        math.exp(controller.lr), rel=0.05
    )


def test_it_still_arrives_given_the_events(_controller):
    """Walking is slow, not broken."""
    controller = _controller(maximum=10000.0)
    _drive(controller, "stft_2048", 1.0 / 145.0, 400)

    assert controller.ratio_for("stft_2048") == pytest.approx(1.0, rel=0.1)
    assert controller.scale_for("stft_2048") == pytest.approx(145.0, rel=0.15)


def test_a_resumed_strength_is_not_relearned(_controller):
    """Which is why the state is persisted: the walk happens once."""
    controller = _controller(maximum=10000.0)
    _drive(controller, "stft_2048", 1.0 / 145.0, 400)
    earned = controller.scale_for("stft_2048")

    restored = _controller(maximum=10000.0)
    assert restored.load_state_dict(controller.state_dict)
    assert restored.scale_for("stft_2048") == pytest.approx(earned, rel=1e-6)
