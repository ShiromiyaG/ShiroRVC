"""Which speaker table a run starts from, and when a pretrain's is discarded.

A multispeaker fine-tune used to inherit the pretrained checkpoint's ``emb_g``
whole, because ``verify_spk_dim`` let the checkpoint's row count win over the
dataset's.  Speaker *k* of the dataset therefore started life *as* speaker *k*
of the pretrain -- not a neutral initialisation.  Measured on ``f0G40k``,
``dec.cond`` reads the directions the trained embedding occupies with 1.53x the
gain it gives random ones, so the decoder renders those identities confidently
from step 0 and the run has to move away from a wrong answer rather than toward
a right one.  On a small dataset it does not move far enough, and the pretrain's
timbre stays audible.

Discarding the table costs nothing in separation -- after ``dec.cond`` four
fresh speakers sit 62.3 apart against 41.0 for four inherited ones -- and the
unused rows were never the problem: a 109-row table with four speakers in use
trains its four rows *bit-identically* to a 4-row table, because the untouched
rows get a zero gradient and weight decay does not couple them.

So the rule is about the rows that *are* used, and it keys off the shape: a row
count that disagrees with the dataset's speaker count cannot be describing this
dataset's speakers.  A count that agrees almost certainly is -- that is staged
pretraining handing its own table forward -- and is kept.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rvc" / "train"))

torch = pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

from rvc.train.utils import (  # noqa: E402
    SpeakerLayout,
    substitute_speaker_embeddings,
    verify_spk_dim,
)


def _config(default_speakers=109):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(spk_embed_dim=default_speakers)
    )


def _write_checkpoint(path: Path, speakers: int) -> str:
    torch.save({"model": {"emb_g.weight": torch.zeros(speakers, 256)}}, path)
    return str(path)


def _write_model_info(directory: Path, speakers: int) -> str:
    path = directory / "model_info.json"
    path.write_text(
        json.dumps({"embedder_model": "contentvec", "speakers_id": speakers})
    )
    return str(path)


def _resolve(tmp_path, *, dataset=None, resume=None, pretrain=None):
    """Run ``verify_spk_dim`` over a synthetic experiment folder."""

    info_path = (
        _write_model_info(tmp_path, dataset)
        if dataset is not None
        else str(tmp_path / "missing.json")
    )
    resume_path = (
        _write_checkpoint(tmp_path / "G_100.pth", resume) if resume else None
    )
    pretrain_path = (
        _write_checkpoint(tmp_path / "pretrain_G.pth", pretrain) if pretrain else "None"
    )
    return verify_spk_dim(
        _config(),
        info_path,
        str(tmp_path),
        lambda _dir, _pattern: resume_path,
        1,  # rank 1 keeps the console quiet
        pretrain_path,
    )


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_a_multispeaker_pretrain_of_the_wrong_width_is_not_inherited(tmp_path):
    """The case that motivated this: 4 speakers, the stock 109-row pretrain."""

    layout = _resolve(tmp_path, dataset=4, pretrain=109)
    assert layout == SpeakerLayout(embed_dim=4, reset_pretrained=True)


def test_a_single_speaker_run_keeps_the_old_behaviour(tmp_path):
    """There is no cross-speaker leak to fix with one speaker, and row 0 of the
    pretrain is a working starting timbre.  Nothing here should change."""

    layout = _resolve(tmp_path, dataset=1, pretrain=109)
    assert layout == SpeakerLayout(embed_dim=109, reset_pretrained=False)


def test_a_pretrain_whose_width_matches_the_dataset_is_kept(tmp_path):
    """Staged pretraining hands its own table forward as ``pretrainG``.  That
    table *is* this dataset's speakers and throwing it away would discard real
    progress, so the widths agreeing is taken as the lineage agreeing."""

    layout = _resolve(tmp_path, dataset=50, pretrain=50)
    assert layout == SpeakerLayout(embed_dim=50, reset_pretrained=False)


def test_a_resume_is_never_touched(tmp_path):
    """``G_*.pth`` in the experiment folder *is* this run's table; its width is
    not negotiable, whatever the dataset or the pretrain say."""

    layout = _resolve(tmp_path, dataset=4, resume=7, pretrain=109)
    assert layout == SpeakerLayout(embed_dim=7, reset_pretrained=False)


def test_without_model_info_the_pretrain_still_wins(tmp_path):
    """The dataset's count is unknown, so there is nothing to disagree with.
    Falling back to the old behaviour is the safe direction: it inherits, which
    is what every run did before."""

    layout = _resolve(tmp_path, dataset=None, pretrain=109)
    assert layout == SpeakerLayout(embed_dim=109, reset_pretrained=False)


def test_with_no_checkpoint_at_all_the_dataset_decides(tmp_path):
    layout = _resolve(tmp_path, dataset=4)
    assert layout == SpeakerLayout(embed_dim=4, reset_pretrained=False)


# --------------------------------------------------------------------------
# the substitution
# --------------------------------------------------------------------------


class _Stub(torch.nn.Module):
    def __init__(self, speakers):
        super().__init__()
        self.emb_g = torch.nn.Embedding(speakers, 256)
        self.other = torch.nn.Linear(4, 4)


def test_the_substitution_makes_a_mismatched_pretrain_load_strictly():
    """Substituting rather than dropping the key: the VITS-latent vocoders load
    their pretrained generator strictly, where a missing key is an error."""

    torch.manual_seed(0)
    model = _Stub(4)
    fresh = model.emb_g.weight.detach().clone()
    pretrained = dict(_Stub(109).state_dict())

    with pytest.raises(RuntimeError):
        model.load_state_dict(pretrained, strict=True)

    patched = substitute_speaker_embeddings(pretrained, model)
    model.load_state_dict(patched, strict=True)

    assert torch.equal(model.emb_g.weight, fresh)
    assert torch.equal(model.other.weight, pretrained["other.weight"])
    # The caller's dict is left alone -- it is still read for other keys.
    assert pretrained["emb_g.weight"].shape[0] == 109


def test_the_substitution_is_a_no_op_without_the_key():
    model = _Stub(4)
    state = {"other.weight": torch.zeros(4, 4)}
    assert substitute_speaker_embeddings(state, model) is state


def test_the_substitution_reaches_through_a_ddp_wrapper():
    model = _Stub(4)
    wrapper = types.SimpleNamespace(module=model)
    patched = substitute_speaker_embeddings(
        {"emb_g.weight": torch.zeros(109, 256)}, wrapper
    )
    assert torch.equal(patched["emb_g.weight"], model.emb_g.weight.detach())


# --------------------------------------------------------------------------
# why the row count itself was never the problem
# --------------------------------------------------------------------------


def test_unused_rows_do_not_couple_to_the_used_ones():
    """Weight decay is the only channel that touches a zero-gradient row, and it
    is per-parameter.  Pinned so nobody re-derives the row count as a fix."""

    class Toy(torch.nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.emb = torch.nn.Embedding(rows, 32)
            self.head = torch.nn.Linear(32, 8)

        def forward(self, sid):
            return self.head(self.emb(sid))

    torch.manual_seed(1)
    wide = Toy(109)
    narrow = Toy(4)
    with torch.no_grad():
        narrow.emb.weight.copy_(wide.emb.weight[:4])
        narrow.head.load_state_dict(wide.head.state_dict())

    def optimizer(model):
        return torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=0.01)

    optimizers = [optimizer(wide), optimizer(narrow)]
    torch.manual_seed(2)
    for _ in range(50):
        sid = torch.randint(0, 4, (16,))
        target = torch.randn(16, 8)
        for model, optim in zip((wide, narrow), optimizers):
            optim.zero_grad()
            torch.nn.functional.mse_loss(model(sid), target).backward()
            optim.step()

    assert torch.equal(wide.emb.weight[:4], narrow.emb.weight)
