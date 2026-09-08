"""Embedders that are not HuBERT-shaped, and so cannot go through
``HubertModelWithFinalProj``."""

from rvc.lib.embedders.spin_wavlm import SpinWavLMModel

__all__ = ["SpinWavLMModel"]
