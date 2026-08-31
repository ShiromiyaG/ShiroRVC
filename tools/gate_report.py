"""What the latent gate actually does, read off a checkpoint and real audio.

The in-loop series (``Source/latent_gate_*``) report the gate's mean and its
saturation share, which is enough to know whether the excitation is being opened
or closed at each stage.  They cannot show the *shape* of the distribution, and
the shape is the whole story: a gate whose mean is 1.0 may be a smooth
multiplier sitting near unity or a hard mask that is at 0 a third of the time
and at 2 a third of the time, and those are different decoders.

This prints the quantiles, so the two are distinguishable.  It runs against any
``G_*.pth`` without touching the training run, which is the point -- adding a
series costs a restart, and reading a checkpoint costs nothing.

Usage::

    python tools/gate_report.py logs/pretrain
    python tools/gate_report.py logs/pretrain --checkpoint logs/pretrain/G_8139.pth
    python tools/gate_report.py logs/pretrain --clips 96 --device cuda

Note that the sampled clips drive ``enc_q``, so the latent this measures is the
*posterior's*, matching training.  Inference decodes from the prior instead, and
a gate that behaves differently there is a real finding rather than a bug in
this script.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# ``mel_processing`` is imported by its bare name across the training package.
sys.path.insert(0, str(ROOT / "rvc" / "train"))

from mel_processing import spectrogram_torch  # noqa: E402
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402


def _latest_checkpoint(experiment: Path) -> Path:
    candidates = sorted(
        experiment.glob("G_*.pth"),
        key=lambda path: int("".join(ch for ch in path.stem if ch.isdigit()) or 0),
    )
    if not candidates:
        raise SystemExit(f"No G_*.pth in {experiment}")
    return candidates[-1]


def _build(config: dict, state: dict, device: torch.device) -> Synthesizer:
    data = config["data"]
    model = dict(config["model"])
    # The run sizes the speaker table from its own dataset, so the shipped
    # ``spk_embed_dim`` is usually wrong for a given checkpoint.  The weight is
    # the authority.
    model["spk_embed_dim"] = state["emb_g.weight"].shape[0]
    net = Synthesizer(
        spec_channels=data["filter_length"] // 2 + 1,
        segment_size=config["train"]["segment_size"] // data["hop_length"],
        **model,
        use_f0=True,
        sr=data["sample_rate"],
        vocoder=config["model"].get("vocoder", "chouwagan"),
    )
    net.load_state_dict(state)
    return net.to(device).eval()


def _batch(experiment: Path, data: dict, clips: int, seed: int, device: torch.device):
    rows = [
        line.split("|")
        for line in (experiment / "filelist.txt").read_text().splitlines()
        if line and "mute" not in line
    ]
    if not rows:
        raise SystemExit(f"{experiment / 'filelist.txt'} has no usable entries")
    random.Random(seed).shuffle(rows)

    spectrograms, speakers, frames = [], [], None
    for wav, _feature, _coarse, _fine, sid in rows[:clips]:
        audio, _ = sf.read(ROOT / wav, dtype="float32")
        spectrogram = spectrogram_torch(
            torch.from_numpy(audio).unsqueeze(0),
            data["filter_length"],
            data["hop_length"],
            data["win_length"],
            center=False,
        ).squeeze(0)
        length = spectrogram.shape[-1]
        frames = length if frames is None else min(frames, length)
        spectrograms.append(spectrogram)
        speakers.append(int(sid))

    # Every clip is cropped to the shortest, so the batch is rectangular without
    # padding -- padding would feed silence through ``enc_q`` and pull the gate
    # statistics toward whatever it does on a dead frame.
    frames = min(frames, 200)
    spec = torch.stack([item[:, :frames] for item in spectrograms]).to(device)
    sid = torch.tensor(speakers, device=device)
    lengths = torch.full((spec.shape[0],), frames, dtype=torch.long, device=device)
    return spec, lengths, sid


@torch.no_grad()
def _gates(net: Synthesizer, spec, lengths, sid):
    decoder = net.dec
    if getattr(decoder, "latent_gates", None) is None:
        raise SystemExit(
            "This checkpoint's decoder has no latent gates "
            "(chouwagan_latent_source_gate was off)."
        )
    speaker = net.emb_g(sid).unsqueeze(-1)
    z, _, _, _ = net.enc_q(spec, lengths, g=speaker)

    # The decoder's own front end, up to the point the gates read from.  Kept in
    # step with ``ChouwaGANGenerator.forward``: the gate sees the latent and the
    # speaker, never the excitation.
    latent = decoder.conv_pre(z)
    if decoder.cond is not None:
        latent = latent + decoder.cond(speaker)
    latent = decoder.latent_mixer(latent)

    return latent, [
        (2.0 * torch.sigmoid(gate(latent))).flatten().float()
        for gate in decoder.latent_gates
    ]


def _raw_ratios(decoder) -> list[float]:
    widths = tuple(reversed(getattr(decoder, "exc_skip_channels", ()) or ()))
    ratios = []
    for index, stage in enumerate(decoder.fusion_proj):
        skip = int(widths[index])
        weight = stage.weight
        main = weight[:, :-skip].norm()
        ratios.append(float((weight[:, -skip:].norm() / main.clamp_min(1e-8)).detach()))
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="e.g. logs/pretrain")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    checkpoint = args.checkpoint or _latest_checkpoint(experiment)
    device = torch.device(args.device)

    config = json.loads((experiment / "config.json").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    net = _build(config, state, device)
    spec, lengths, sid = _batch(experiment, config["data"], args.clips, args.seed, device)
    latent, gates = _gates(net, spec, lengths, sid)
    ratios = _raw_ratios(net.dec)

    print(f"checkpoint  {checkpoint.name}")
    print(f"batch       {spec.shape[0]} clips x {spec.shape[-1]} frames")
    print(f"latent      std {latent.std():.3f}\n")

    probabilities = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    header = (
        "stage   mean    std     p1     p25    p50    p75    p99   "
        "floor  ceil  |  raw    effective"
    )
    print(header)
    print("-" * len(header))
    for index, values in enumerate(gates):
        quantiles = torch.quantile(values.cpu(), probabilities)
        floor = float((values < 0.02).float().mean())
        ceiling = float((values > 1.98).float().mean())
        mean = float(values.mean())
        print(
            f"  {index}    {mean:.3f}  {values.std():.3f}  "
            + "  ".join(f"{q:.3f}" for q in quantiles)
            + f"  {floor * 100:4.1f}% {ceiling * 100:4.1f}% |  "
            f"{ratios[index]:.3f}  {ratios[index] * mean:.3f}"
        )

    print(
        "\nfloor/ceil are the shares pinned at 0 and 2.  Large on both sides "
        "means the gate\nis a hard mask, and its mean is then a duty cycle "
        "rather than a multiplier."
    )


if __name__ == "__main__":
    main()
