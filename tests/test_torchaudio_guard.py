"""A torch/torchaudio CUDA mismatch has to surface as the command that fixes it.

torchaudio's own error only says the builds differ.  Users who installed
``requirements.txt`` over a cloud image's torch stack hit it at startup and had
no way to tell which wheel to fetch, so the guard names it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="needs torch", exc_type=ImportError)

GUARD = "rvc.lib.torchaudio_guard"


class _MismatchedTorchaudio:
    """Meta path finder that fails the way a cu128 torchaudio does on cu130 torch."""

    def find_spec(self, name, path=None, target=None):
        if name == "torchaudio":
            raise RuntimeError(
                "Detected that PyTorch and TorchAudio were compiled with "
                "different CUDA versions. PyTorch has CUDA version 13.0 "
                "whereas TorchAudio has CUDA version 12.8."
            )
        return None


@pytest.fixture
def mismatched(monkeypatch):
    for name in [m for m in sys.modules if m == "torchaudio" or m.startswith("torchaudio.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, GUARD, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_MismatchedTorchaudio(), *sys.meta_path])
    monkeypatch.setattr("torch.version.cuda", "13.0")
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: "2.11.0+cu128" if name == "torchaudio" else "0",
    )


def test_mismatch_names_the_matching_wheel(mismatched):
    with pytest.raises(RuntimeError) as caught:
        importlib.import_module(GUARD)
    message = str(caught.value)
    assert "torchaudio==2.11.0 " in message
    assert "--force-reinstall --no-deps" in message
    assert "https://download.pytorch.org/whl/cu130" in message
    assert sys.executable in message


def test_every_torchaudio_import_goes_through_the_guard():
    importers = [
        path
        for path in (ROOT / "rvc").rglob("*.py")
        if "import torchaudio" in path.read_text(encoding="utf-8")
        or "from torchaudio" in path.read_text(encoding="utf-8")
    ]
    importers = [p for p in importers if p.name != "torchaudio_guard.py"]
    assert importers
    for path in importers:
        source = path.read_text(encoding="utf-8")
        guard = source.index("import rvc.lib.torchaudio_guard")
        first = min(
            i for i in (source.find("import torchaudio"), source.find("from torchaudio")) if i >= 0
        )
        assert guard < first, f"{path.relative_to(ROOT)} imports torchaudio before the guard"
