"""Distributable style checkpoints (``style_base.pt``, ``<model>_style.pt``).

Everything inference needs travels in the file: weights, the representation,
normalization, the embedder name and its k-means codebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .descriptors import DESCRIPTOR_NAMES, DescriptorConfig
from .f0_repr import Normalizer, ReprConfig
from .frontend import VOICING_THRESHOLD
from .model import ModelConfig, StyleDiT
from .units import UnitCodebook

FORMAT_VERSION = 1


@dataclass
class StyleModel:
    model: StyleDiT
    representation: ReprConfig
    descriptor_config: DescriptorConfig
    normalizer: Normalizer
    embedder: str
    codebook: UnitCodebook
    descriptor_names: list
    #: Pooled descriptors of the singer (fine-tuned models) or the whole
    #: training set (base); ``None`` entries are unknown.
    style_descriptors: list | None
    kind: str
    step: int
    #: RMVPE salience threshold the training data was voiced with.
    voicing_threshold: float = VOICING_THRESHOLD


def export(path, model: StyleDiT, *, representation, descriptor_config, normalizer, embedder, codebook,
           style_descriptors=None, kind="base", step=0, voicing_threshold=VOICING_THRESHOLD):
    state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "format": FORMAT_VERSION,
            "kind": kind,
            "step": int(step),
            "model_config": model.cfg.to_dict(),
            "model": state,
            "representation": representation.to_dict(),
            "descriptor_config": descriptor_config.to_dict(),
            "normalizer": normalizer.to_dict(),
            "embedder": embedder,
            "codebook": np.asarray(codebook.centroids, dtype=np.float32),
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "style_descriptors": style_descriptors,
            "voicing_threshold": float(voicing_threshold),
        },
        path,
    )


def load(path, device="cpu") -> StyleModel:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if list(data["descriptor_names"]) != list(DESCRIPTOR_NAMES):
        raise ValueError(f"{path} was trained with different descriptors.")
    model = StyleDiT(ModelConfig.from_dict(data["model_config"]))
    model.load_state_dict(data["model"])
    model.to(device).eval()
    return StyleModel(
        model=model,
        representation=ReprConfig.from_dict(data["representation"]),
        descriptor_config=DescriptorConfig.from_dict(data["descriptor_config"]),
        normalizer=Normalizer.from_dict(data["normalizer"]),
        embedder=data["embedder"],
        codebook=UnitCodebook(data["codebook"]),
        descriptor_names=list(data["descriptor_names"]),
        style_descriptors=data.get("style_descriptors"),
        kind=data.get("kind", "base"),
        step=int(data.get("step", 0)),
        voicing_threshold=float(data.get("voicing_threshold", VOICING_THRESHOLD)),
    )
