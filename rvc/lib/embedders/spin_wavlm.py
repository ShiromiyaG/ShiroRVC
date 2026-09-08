"""SPIN WavLM 512 -- a WavLM encoder with a speaker-invariant projection head.

Trained by Lyery and published at https://huggingface.co/lyery/spin-wavlm512;
the integration this is ported from is Lyery's fork,
https://github.com/redpanda343/redpanda-rvc.

SPIN (Speaker-INvariant clustering) fine-tunes a self-supervised encoder so that
its features cluster by phonetic content rather than by who is speaking, which
is the property a voice conversion frontend wants: whatever survives the
projection is what the decoder has to be told, and speaker identity is supposed
to come from the speaker embedding instead.

Two things make this not a drop-in for the other embedders here:

* It is **not** a HuBERT, so ``HubertModelWithFinalProj`` cannot load it.  The
  checkpoint is a PyTorch Lightning training state, converted once by
  :mod:`rvc.lib.tools.convert_spin_wavlm` into a ``transformers`` WavLM plus a
  separate projection.
* Its features are **256-dimensional**, not 768.  The 512 in the name is the
  number of SPIN clusters, not the width.  That width reaches the synthesizer
  as ``text_enc_hidden_dim``, so a model trained against this embedder is not
  weight-compatible with a contentvec or spin_v2 one.
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import WavLMModel


class SpinWavLMModel(torch.nn.Module):
    """The converted encoder and its projection, as one module.

    ``forward`` returns a dict with ``last_hidden_state`` rather than the
    tensor, so callers can treat it like the ``transformers`` models beside it.
    """

    def __init__(self, model_path):
        super().__init__()
        config_path = os.path.join(model_path, "spin_config.json")
        projection_path = os.path.join(model_path, "spin_projection.safetensors")
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.spin_config = json.load(config_file)

        self.encoder = WavLMModel.from_pretrained(model_path, local_files_only=True)
        input_dim = int(self.spin_config["encoder_dim"])
        output_dim = int(self.spin_config["feature_dim"])
        self.projection = torch.nn.Linear(input_dim, output_dim)
        self.projection.load_state_dict(load_file(projection_path), strict=True)
        # The converted preprocessor sets ``do_normalize`` false, and the flag
        # is carried here as well so the loader can report it without going
        # back to disk for a second config.
        self.audio_requires_normalization = bool(
            self.spin_config.get("audio_requires_normalization", False)
        )
        self.feature_dim = output_dim
        self.feature_output = self.spin_config["feature_output"]
        self.feature_fingerprint = self.spin_config["source_checkpoint_sha256"]

    def forward(self, input_values):
        hidden_states = self.encoder(input_values).last_hidden_state
        features = self.projection(hidden_states)
        # SPIN's objective is defined on the unit sphere, so the features it
        # was trained to produce are normalised ones; skipping this would hand
        # the decoder a different quantity from the one that was clustered.
        if self.spin_config.get("l2_normalize", True):
            features = F.normalize(features, dim=-1)
        return {"last_hidden_state": features}
