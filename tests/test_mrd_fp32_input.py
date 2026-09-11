"""``DiscriminatorR``'s FP32 input stage: the switch, its default, what it keeps."""

from pathlib import Path

import pytest
import torch

from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined
from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import DiscriminatorR

ROOT = Path(__file__).resolve().parents[1]
RESOLUTION = [512, 50, 240]


def _resolution_branches(discriminator):
    return [b for b in discriminator.discriminators if isinstance(b, DiscriminatorR)]


def test_on_by_default_and_reaches_every_resolution_branch():
    default = MPD_MSD_Combined(False, version="v4", sample_rate=32000)
    assert default.mrd_fp32_input is True
    branches = _resolution_branches(default)
    assert branches and all(b.fp32_input for b in branches)

    off = MPD_MSD_Combined(
        False, version="v4", sample_rate=32000, mrd_fp32_input=False
    )
    assert not any(b.fp32_input for b in _resolution_branches(off))


def test_adds_no_state_and_changes_nothing_outside_autocast():
    torch.manual_seed(0)
    on = DiscriminatorR(RESOLUTION, fp32_input=True)
    off = DiscriminatorR(RESOLUTION, fp32_input=False)
    # Strict both ways: the switch must not cost a checkpoint its load.
    off.load_state_dict(on.state_dict(), strict=True)

    x = torch.randn(2, 1, 4800) * 0.3
    out_on, fmap_on = on(x)
    out_off, fmap_off = off(x)
    torch.testing.assert_close(out_on, out_off, rtol=0, atol=0)
    for a, b in zip(fmap_on, fmap_off):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 autocast needs CUDA")
def test_under_autocast_only_the_input_stage_leaves_fp16():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 4800, device="cuda") * 0.3
    for flag, first_dtype in ((True, torch.float32), (False, torch.float16)):
        branch = DiscriminatorR(RESOLUTION, fp32_input=flag).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            _, fmap = branch(x)
        assert fmap[0].dtype == first_dtype
        assert all(f.dtype == torch.float16 for f in fmap[1:])


def test_the_trainer_reads_the_flag_with_the_constructor_default():
    source = (ROOT / "rvc" / "train" / "train.py").read_text(encoding="utf-8")
    assert 'setting("d_mrd_fp32_input", True)' in source
