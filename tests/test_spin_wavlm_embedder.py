"""SPIN WavLM 512 -- the parts of it that are not the model.

The weights are Lyery's (https://huggingface.co/lyery/spin-wavlm512) and the
integration is ported from Lyery's fork,
https://github.com/redpanda343/redpanda-rvc.  Nothing here downloads them: an
835 MB checkpoint is not a test dependency.  What is pinned instead is the
plumbing that gets its 256-wide features to the synthesizer, because that is
what silently breaks -- the shipped configs all say 768, and a model built at
the wrong width either fails on the first matmul or, on a resumed run, keeps a
stale one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rvc.lib.utils import (  # noqa: E402
    EMBEDDER_FEATURE_DIMS,
    embedder_feature_dim,
)


def test_the_width_is_256_not_512():
    """The 512 in the name is SPIN's cluster count, not the feature width.

    Reading it as the width is the obvious mistake, and it would build a text
    encoder twice as wide as the features it is about to be handed.
    """
    assert embedder_feature_dim("spin_wavlm_512") == 256
    assert EMBEDDER_FEATURE_DIMS["spin_wavlm_512"] == 256
    # The ones it sits beside, so a change to the table has to be deliberate.
    assert EMBEDDER_FEATURE_DIMS["contentvec"] == 768
    assert EMBEDDER_FEATURE_DIMS["spin_v2"] == 768


def test_an_unknown_embedder_falls_back_rather_than_raising():
    """This is called on paths where guessing wrong is recoverable and
    stopping is not."""
    assert embedder_feature_dim("something-else") == 768
    assert embedder_feature_dim("custom", None) == 768
    assert embedder_feature_dim("custom", "/nonexistent") == 768


def test_a_custom_embedder_is_read_from_its_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    assert embedder_feature_dim("custom", str(tmp_path)) == 1024


@pytest.mark.parametrize("embedder,expected", [
    ("spin_wavlm_512", 256),
    ("contentvec", 768),
])
def test_the_generated_config_carries_the_embedder_width(
    tmp_path, embedder, expected
):
    """``apply_embedder_width`` is what stops the shipped 768 reaching a
    256-wide run."""
    from rvc.train.extract.preparing_files import apply_embedder_width

    shipped = json.loads(
        (ROOT / "rvc" / "configs" / "refinegan2" / "32000.json").read_text()
    )
    assert shipped["model"]["text_enc_hidden_dim"] == 768, "fixture is stale"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(shipped))
    apply_embedder_width(str(config_path), embedder)
    written = json.loads(config_path.read_text())
    assert written["model"]["text_enc_hidden_dim"] == expected
    # Nothing else moved.
    written["model"]["text_enc_hidden_dim"] = 768
    assert written == shipped


def test_a_missing_config_is_not_an_error(tmp_path):
    """It runs after the config is written; if that failed, the failure the
    caller sees should be the one that actually happened."""
    from rvc.train.extract.preparing_files import apply_embedder_width

    apply_embedder_width(str(tmp_path / "absent.json"), "spin_wavlm_512")


def test_inference_reads_the_width_off_the_checkpoint():
    """``text_enc_hidden_dim`` used to be inferred from the RVC version, which
    is 768 for anything tagged v2.  This embedder is 256 wide under v2, so the
    width now comes from the text encoder's own input layer."""
    source = (ROOT / "rvc" / "infer" / "infer.py").read_text()
    assert 'self.active_cpt["weight"].get("enc_p.emb_phone.weight")' in source
    assert "self.text_enc_hidden_dim = int(emb_phone.shape[1])" in source

    # And the layer it reads is the one the synthesizer actually builds.
    from rvc.lib.algorithm.text_encoder import TextEncoder

    encoder = TextEncoder(
        out_channels=8, hidden_channels=8, filter_channels=8, n_heads=2,
        n_layers=1, kernel_size=3, p_dropout=0.0, embedding_dim=256, f0=True,
    )
    assert encoder.emb_phone.weight.shape[1] == 256


def test_the_mute_fixture_matches_the_ones_beside_it():
    """Extraction pads every dataset with a mute clip, and the entry pairs one
    audio file with one feature file.  A fixture with a different frame count
    or width from its neighbours would line up wrongly at exactly the place
    nobody looks."""
    import numpy as np
    import soundfile as sf

    mute = ROOT / "logs" / "mute_spin_wavlm_512"
    reference = ROOT / "logs" / "mute_spin_v2"
    assert mute.is_dir(), "the fixture is shipped, not generated on demand"

    # Same files as its neighbour, and the shared ones byte-identical: the
    # silence and its pitch do not depend on the embedder.
    names = sorted(p.relative_to(reference).as_posix() for p in reference.rglob("*") if p.is_file())
    assert sorted(p.relative_to(mute).as_posix() for p in mute.rglob("*") if p.is_file()) == names
    for name in names:
        if name == "extracted/mute.npy":
            continue
        assert (mute / name).read_bytes() == (reference / name).read_bytes(), name

    feature = np.load(mute / "extracted" / "mute.npy")
    neighbour = np.load(reference / "extracted" / "mute.npy")
    assert feature.shape[0] == neighbour.shape[0], "frame count drifted"
    assert feature.shape[1] == EMBEDDER_FEATURE_DIMS["spin_wavlm_512"]
    assert np.isfinite(feature).all()
    # SPIN's features live on the unit sphere; a fixture that is not normalised
    # was produced by something other than the shipped forward pass.
    assert np.allclose(np.linalg.norm(feature, axis=-1), 1.0, atol=1e-5)

    # And it is the feature of that silence, at the rate the model expects.
    info = sf.info(str(mute / "sliced_audios_16k" / "mute.wav"))
    assert info.samplerate == 16000
    assert feature.shape[0] == pytest.approx(
        info.frames / 320, abs=2
    ), "hop is not the 320 samples spin_config declares"


def _reference_sample_function():
    """``get_reference_sample`` out of ``train.py`` without importing it.

    ``rvc/train/train.py`` is a script -- importing it starts CUDA, argument
    parsing and a config load -- so the one function is compiled on its own,
    with the two loggers it calls stubbed to collect messages.
    """
    import numpy as np
    import torch

    source = (ROOT / "rvc" / "train" / "train.py").read_text()
    start = source.index("def get_reference_sample(")
    end = source.index("\ndef ", start + 10)
    messages = []
    namespace = {
        "os": __import__("os"),
        "np": np,
        "torch": torch,
        "info": lambda text, tag=None: messages.append(("info", text)),
        "warning": lambda text, tag=None: messages.append(("warning", text)),
    }
    exec(compile(source[start:end], "train.py", "exec"), namespace)
    return namespace["get_reference_sample"], messages


def _fake_loader(width):
    import torch

    class Dataset:
        def get_file_paths(self, indices):
            return ["0_0_0.wav"]

    class Loader:
        dataset = Dataset()
        batch_sampler = [[0]]

        def __iter__(self):
            frames = 8
            yield (
                torch.randn(1, frames, width), torch.LongTensor([frames]),
                torch.zeros(1, frames, dtype=torch.long), torch.zeros(1, frames),
                None, None, torch.randn(1, 1, frames * 320), None,
                torch.LongTensor([0]),
            )

    return Loader()


def _config(width):
    import types

    return types.SimpleNamespace(
        model=types.SimpleNamespace(text_enc_hidden_dim=width),
        data=types.SimpleNamespace(sample_rate=32000, hop_length=320),
        train=types.SimpleNamespace(seed=1234),
    )


def test_a_reference_from_another_embedder_is_refused_not_crashed_on():
    """``logs/reference/ref_feats.npy`` is embedder-specific and nothing in its
    name says which one wrote it.

    Handing 768-wide features to a 256-wide text encoder reached ``F.linear``
    and died there -- "mat1 and mat2 shapes cannot be multiplied (694x768 and
    256x192)" -- at the first preview, which is thousands of steps into a run.
    """
    import numpy as np

    reference = ROOT / "logs" / "reference" / "ref_feats.npy"
    if not reference.is_file():
        pytest.skip("no custom reference in this tree")
    width = int(np.load(reference).shape[1])

    get_reference_sample, messages = _reference_sample_function()

    # A model whose width does not match: warned, and carried on with a
    # reference from the dataset instead.
    other = 256 if width != 256 else 768
    messages.clear()
    (phone, *_), _audio, origin = get_reference_sample(
        _fake_loader(other), "cpu", _config(other)
    )
    assert phone.shape[-1] == other
    assert origin != str(ROOT / "logs" / "reference")
    warned = [text for kind, text in messages if kind == "warning"]
    assert warned and "ref_feats.npy" in warned[0]
    assert "make_reference.py" in warned[0], "the message has to say how to fix it"

    # A model whose width does match still gets the custom reference.
    messages.clear()
    (phone, *_), _audio, origin = get_reference_sample(
        _fake_loader(width), "cpu", _config(width)
    )
    assert phone.shape[-1] == width
    assert origin.endswith("reference")


def test_the_embedder_is_offered_everywhere_the_others_are():
    """A choice list that forgets it makes the embedder unreachable from that
    surface, which is the kind of thing nobody notices until someone asks."""
    from gui.services import catalog

    assert "spin_wavlm_512" in catalog.EMBEDDER_MODELS
    assert "spin_wavlm_512" in catalog.TRAINING_EMBEDDER_MODELS

    for relative in (
        "core.py",
        "tabs/train/train.py",
        "tabs/inference/inference.py",
        "tabs/tts/tts.py",
    ):
        source = (ROOT / relative).read_text()
        assert "spin_wavlm_512" in source, relative
        # Offered exactly as often as spin_v2, never twice in one list.
        assert source.count('"spin_wavlm_512"') == source.count('"spin_v2"'), relative
