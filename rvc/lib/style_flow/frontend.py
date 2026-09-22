"""RMVPE and ContentVec on 16 kHz audio: the one path extraction and inference share."""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

SAMPLE_RATE = 16000
HOP = 160

# Same high-pass as ``rvc/infer/pipeline.py`` (``bh``/``ah``), duplicated
# because the pipeline imports this package.
_bh, _ah = signal.butter(N=5, Wn=48, btype="high", fs=SAMPLE_RATE)


def highpass(audio: np.ndarray) -> np.ndarray:
    return signal.filtfilt(_bh, _ah, audio).astype(np.float32)


#: RMVPE's usual salience threshold, and what every dataset and model
#: without its own ``voicing_threshold`` was made with.
VOICING_THRESHOLD = 0.03


class StyleFrontend:
    """RMVPE (high-register corrector off) and the base model's embedder,
    whatever the RVC model's F0 method was, so the style model sees one F0
    distribution.  Models load on first use."""

    def __init__(
        self, device: str = "cuda", embedder: str = "contentvec", chunk_seconds: float = 20.0,
        context_seconds: float = 1.0, voicing_threshold: float = VOICING_THRESHOLD,
    ):
        self.device = device
        self.voicing_threshold = float(voicing_threshold)
        self.embedder_name = embedder
        self._rmvpe = None
        self._embedder = None
        self.chunk = int(chunk_seconds * SAMPLE_RATE)
        self.context = int(context_seconds * SAMPLE_RATE)

    @property
    def rmvpe(self):
        if self._rmvpe is None:
            from rvc.lib.predictors.f0 import RMVPE

            self._rmvpe = RMVPE(device=self.device, sample_rate=SAMPLE_RATE, hop_size=HOP, high_register={"enabled": False})
        return self._rmvpe

    @property
    def embedder(self):
        if self._embedder is None:
            from rvc.lib.utils import load_embedder_model

            model, self.do_normalize = load_embedder_model(self.embedder_name)
            self._embedder = model.to(self.device).float().eval()
        return self._embedder

    def _chunked(self, audio: np.ndarray, fn, step: int) -> np.ndarray:
        """``fn`` over ``audio`` in ``chunk``-sample windows with ``context``
        on each side, keeping each window's centre frames (``step`` samples
        per frame).  Memory then stays that of one window, not of the file."""
        if len(audio) <= self.chunk + 2 * self.context:
            return fn(audio)
        out = []
        for start in range(0, len(audio), self.chunk):
            lo = max(0, start - self.context)
            hi = min(len(audio), start + self.chunk + self.context)
            frames = fn(audio[lo:hi])
            skip = (start - lo) // step
            keep = min(self.chunk, len(audio) - start) // step
            out.append(frames[skip : skip + keep])
        frames = np.concatenate(out)
        expected = len(audio) // step
        if frames.shape[0] < expected:
            frames = np.concatenate([frames, np.repeat(frames[-1:], expected - frames.shape[0], axis=0)])
        return frames[:expected]

    def f0(self, audio: np.ndarray) -> np.ndarray:
        """F0 in Hz at 100 fps; ``audio`` is 16 kHz and already high-passed."""
        rmvpe = self.rmvpe
        return self._chunked(
            audio, lambda x: np.asarray(rmvpe.get_f0(x, filter_radius=self.voicing_threshold), dtype=np.float32), HOP
        )

    @torch.no_grad()
    def _features(self, audio: np.ndarray) -> np.ndarray:
        from rvc.lib.utils import extract_features

        model = self.embedder
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device).view(1, -1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            feats = extract_features(model, x, "v2", do_normalize=self.do_normalize)
        return feats.squeeze(0).float().cpu().numpy()

    def features(self, audio: np.ndarray) -> np.ndarray:
        """``(T/320, 768)`` embedder features at 50 fps."""
        return self._chunked(audio, self._features, 2 * HOP)
