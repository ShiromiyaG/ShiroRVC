"""Turn a stage-0 mel -> wav decoder into an RVC pretrain.

WHAT THIS DOES
--------------
``tools/pretrain_stage0_vocoder.py`` trains a decoder whose input is a mel
(``n_mel_channels`` bins).  RVC's decoder reads ``z`` (``inter_channels``).
Only the input projection differs in shape -- ``mel_conv`` on RefineGAN2,
``conv_pre`` on HiFi-GAN NSF -- so this script builds a full ``Synthesizer``,
copies every decoder weight whose shape matches, leaves the projection at its
fresh initialisation, and writes the result as a pretrain ``G``.

The discriminator is copied whole: it only ever sees waveforms, so nothing
about it depends on the decoder's input representation.

What comes out is a pretrain with a **trained decoder and a fresh frontend**.
``enc_p``, ``enc_q``, ``flow`` and ``emb_g`` are at their initialisation --
stage 1 is what trains them.

WHY A GRAFT AND NOT A FREEZE
----------------------------
Feeding a mel-trained decoder from the encoders while holding it frozen would
pin ``z`` to mel, which makes ``enc_q`` a closed-form transform of the audio
and the KL meaningless.  The decoder is grafted as an *initialisation* and
keeps training in stages 1 and 3.  See the header of
``tools/pretrain_stage0_vocoder.py``.

USAGE
-----
    # defaults: newest stage0_*.pth in the stage-0 folder, EMA weights
    python tools/graft_vocoder_pretrain.py --model vctk-vocoder \\
        --out-g logs/pretrain_G.pth --out-d logs/pretrain_D.pth

    # graft onto a different model's config (different speaker table)
    python tools/graft_vocoder_pretrain.py --model vctk-vocoder \\
        --target-model my-voice \\
        --out-g logs/pretrain_G.pth --out-d logs/pretrain_D.pth

Then start stage 1 from it, in the target model's folder:

    python tools/pretrain_stage1_vocoder.py --model my-voice \\
        --vocoder refinegan2 --total-epochs 40 \\
        --pretrain-g logs/pretrain_G.pth --pretrain-d logs/pretrain_D.pth

Stages 2 and 3 are unchanged: they resume from the ``G_*.pth`` / ``D_*.pth``
stage 1 leaves behind, exactly as they do without a stage 0.

The target config decides the decoder that is built, so it has to name the
same vocoder and the same upsample schedule the stage-0 run used -- a
mismatch is reported here rather than at load time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
_train_dir = os.path.join(now_dir, "rvc", "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from rvc.train.setup import get_d_model, get_g_model  # noqa: E402
from rvc.train.utils import (  # noqa: E402
    decoder_layout,
    discriminator_periods,
    excitation_source,
    load_config_from_json,
)
from tools._staged_pretrain import (  # noqa: E402
    StageError,
    model_dir,
    resolve_vocoder,
)


def newest(directory: Path, kind: str) -> Path:
    """Newest ``stage0_<kind>_<step>.pth`` in ``directory``."""
    best: tuple[int, Path] | None = None
    for entry in directory.iterdir():
        name = entry.name
        if not name.startswith(f"stage0_{kind}_") or entry.suffix != ".pth":
            continue
        try:
            step = int(name[: -len(".pth")].split("_")[2])
        except (IndexError, ValueError):
            continue
        if best is None or step > best[0]:
            best = (step, entry)
    if best is None:
        raise StageError(
            f"No stage0_{kind}_*.pth in {directory}. Run "
            f"tools/pretrain_stage0_vocoder.py there first."
        )
    return best[1]


def decoder_weights(checkpoint: dict, use_ema: bool) -> tuple[dict, str]:
    """The stage-0 decoder's weights, preferring the EMA average."""
    if use_ema:
        shadow = (checkpoint.get("ema") or {}).get("shadow")
        if shadow:
            return shadow, "EMA average"
    return checkpoint["model"], "live weights"


def graft(net_g, source: dict) -> tuple[list[str], list[str]]:
    """Copy stage-0 decoder weights into ``net_g.dec``, by shape.

    Skipping on a shape mismatch rather than by name is what makes this work
    for either decoder: the input projection is the only tensor whose shape
    depends on the input representation, so it is the only one that can be
    skipped, and a second mismatch means the two runs disagree about
    something else and has to be reported rather than papered over.
    """
    target = net_g.state_dict()
    copied, skipped = [], []
    for key, value in source.items():
        full = f"dec.{key}"
        if full not in target:
            raise StageError(
                f"The stage-0 checkpoint has '{key}', which this decoder does "
                f"not. The target config builds a different vocoder than the "
                f"stage-0 run did."
            )
        if target[full].shape != value.shape:
            skipped.append(f"{full}  {tuple(value.shape)} -> "
                           f"{tuple(target[full].shape)}")
            continue
        target[full] = value.clone()
        copied.append(full)
    net_g.load_state_dict(target, strict=True)
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Graft a stage-0 mel vocoder onto an RVC pretrain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="The stage-0 log directory.")
    parser.add_argument(
        "--target-model",
        default=None,
        help="Log directory whose config.json defines the pretrain to write "
             "(default: the same as --model).",
    )
    parser.add_argument("--stage0-g", default=None)
    parser.add_argument("--stage0-d", default=None)
    parser.add_argument("--out-g", required=True)
    parser.add_argument("--out-d", default=None)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    source_dir = model_dir(args.model)
    if not source_dir.is_dir():
        raise StageError(f"{source_dir} does not exist.")
    target_dir = model_dir(args.target_model) if args.target_model else source_dir
    config_path = target_dir / "config.json"
    if not config_path.is_file():
        raise StageError(f"{config_path} is missing.")
    config = load_config_from_json(str(config_path))
    vocoder = resolve_vocoder(target_dir, args.vocoder)

    g_path = Path(args.stage0_g) if args.stage0_g else newest(source_dir, "G")
    checkpoint_g = torch.load(g_path, map_location="cpu", weights_only=True)

    stored_mels = checkpoint_g.get("num_mels")
    if stored_mels is not None and int(stored_mels) != int(
        config.data.n_mel_channels
    ):
        raise StageError(
            f"{g_path.name} was trained on {stored_mels} mel bins and "
            f"{config_path} says {config.data.n_mel_channels}. The graft "
            f"would silently keep a projection sized for neither."
        )

    net_g = get_g_model(config, int(config.data.sample_rate), vocoder, False)
    source, provenance = decoder_weights(checkpoint_g, bool(args.use_ema))
    copied, skipped = graft(net_g, source)

    print("Graft -- stage 0 decoder -> RVC pretrain")
    print(f"  stage 0   {g_path}  ({provenance})")
    print(f"  target    {config_path}")
    print(f"  vocoder   {vocoder}")
    print(f"  decoder   mel {config.data.n_mel_channels} -> "
          f"z {config.model.inter_channels}")
    print(f"  copied    {len(copied)} decoder tensors")
    if skipped:
        print(f"  fresh     {len(skipped)} (input projection, shape-dependent):")
        for line in skipped:
            print(f"              {line}")
    else:
        print("  fresh     nothing -- suspicious: the input projection should "
              "not have matched")
    print("  fresh     enc_p, enc_q, flow, emb_g (stage 1 trains these)")

    payload = {
        "model": net_g.state_dict(),
        "iteration": 0,
        "learning_rate": float(config.train.learning_rate_g),
    }
    architecture_id = getattr(net_g, "architecture_id", None)
    if architecture_id is not None:
        payload["architecture_id"] = architecture_id
    # The guards the pretrain path runs before its strict load read these
    # keys; a pretrain without them is read as the legacy layout and refused.
    source_name = excitation_source(net_g)
    if source_name is not None:
        payload["excitation_source"] = source_name
    layout = decoder_layout(net_g)
    if layout is not None:
        payload["decoder_layout"] = layout

    out_g = Path(args.out_g)
    out_g.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_g)
    print(f"\n  wrote     {out_g}")

    if args.out_d:
        d_path = Path(args.stage0_d) if args.stage0_d else newest(source_dir, "D")
        checkpoint_d = torch.load(d_path, map_location="cpu", weights_only=True)
        net_d = get_d_model(config, vocoder, False)
        # Strict: the discriminator is representation-agnostic, so anything
        # that does not fit means the two runs built different branch sets.
        net_d.load_state_dict(checkpoint_d["model"], strict=True)
        payload_d = {
            "model": net_d.state_dict(),
            "iteration": 0,
            "learning_rate": float(config.train.learning_rate_d),
        }
        periods = discriminator_periods(net_d)
        if periods is not None:
            payload_d["discriminator_periods"] = periods
        out_d = Path(args.out_d)
        out_d.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload_d, out_d)
        print(f"  wrote     {out_d}  (from {d_path.name})")

    print(
        f"\nNext: python tools/pretrain_stage1_vocoder.py --model "
        f"{target_dir.name} --vocoder {vocoder} --total-epochs 40 \\\n"
        f"        --pretrain-g {args.out_g}"
        + (f" --pretrain-d {args.out_d}" if args.out_d else "")
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as error:
        print(f"\n[graft] {error}\n", file=sys.stderr)
        sys.exit(1)
