"""The prior draw ``infer`` decodes, per vocoder and per call.

The bug this pins: a RefineGAN2 decoder renders the prior draw as
formant-coloured bursts between the harmonics, and at the historical 0.66666
they are loud enough to read as a defect of the model.  0.3 removes nearly
all of them for 0.8 dB at 12-16 kHz, so it is that decoder's default; every
other vocoder keeps 0.66666.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the synthesizer needs torch", exc_type=ImportError)

from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

CONFIG = json.loads((ROOT / "rvc" / "configs" / "refinegan2" / "32000.json").read_text())


def _build(vocoder="refinegan2"):
    data, model = CONFIG["data"], dict(CONFIG["model"])
    model.pop("architecture_id", None)
    return Synthesizer(
        spec_channels=data["filter_length"] // 2 + 1,
        segment_size=CONFIG["train"]["segment_size"] // data["hop_length"],
        sr=data["sample_rate"],
        use_f0=True,
        vocoder=vocoder,
        **model,
    ).eval()


def _latent(net, seed, noise_scale=None, frames=40):
    phone = torch.randn(1, frames, CONFIG["model"]["text_enc_hidden_dim"],
                        generator=torch.Generator().manual_seed(0))
    lengths = torch.tensor([frames])
    pitch = torch.full((1, frames), 100, dtype=torch.long)
    pitchf = torch.full((1, frames), 220.0)
    with torch.no_grad():
        _, _, (_, z_p, m_p, logs_p) = net.infer(
            phone, lengths, pitch, pitchf, torch.tensor([0]),
            seed=seed, noise_scale=noise_scale,
        )
    return z_p, m_p, logs_p


def _drawn_scale(net, **kwargs):
    """The scale ``infer`` actually applied, read back off its own draw."""
    z_p, m_p, logs_p = _latent(net, seed=7, **kwargs)
    torch.manual_seed(7)
    eps = torch.randn_like(m_p)
    return float(((z_p - m_p) / (torch.exp(logs_p) * eps)).mean())


def test_refinegan2_defaults_to_0_3():
    net = _build()
    assert net.prior_noise_scale == pytest.approx(0.3)
    assert _drawn_scale(net) == pytest.approx(0.3, rel=1e-4)


def test_other_vocoders_keep_the_historical_draw():
    net = _build("hifi")
    assert net.prior_noise_scale == pytest.approx(0.66666)
    assert _drawn_scale(net) == pytest.approx(0.66666, rel=1e-4)


@pytest.mark.parametrize("value", [0.0, 0.66666])
def test_the_caller_overrides_the_default(value):
    assert _drawn_scale(_build(), noise_scale=value) == pytest.approx(value, abs=1e-5)


def test_zero_decodes_the_prior_mean_whatever_the_seed():
    net = _build()
    a = _latent(net, seed=1, noise_scale=0.0)[0]
    b = _latent(net, seed=2, noise_scale=0.0)[0]
    torch.testing.assert_close(a, b)
