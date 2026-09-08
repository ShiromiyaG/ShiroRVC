import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import spawn_infer_model_from_pt as fcpe_f0_predictor
import torchcrepe

# Defaults for the "rmvpe_high_register" section of assets/config.json
# (edited from the Settings tab).
HIGH_REGISTER_DEFAULTS = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250.0}

#: assets/config.json at the application root, three levels up from
#: rvc/lib/predictors/.  Resolved from this file rather than the working
#: directory so extraction subprocesses read the same settings as the UI.
CONFIG_PATH = Path(__file__).resolve().parents[3] / "assets" / "config.json"


def load_high_register_settings():
    # Missing file, malformed JSON or missing keys fall back to the defaults
    # (corrector disabled), so RMVPE keeps stock behaviour outside the UI.
    settings = dict(HIGH_REGISTER_DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            section = json.load(f).get("rmvpe_high_register")
        if isinstance(section, dict):
            settings.update(section)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return settings


class RMVPE:
    #: Lowest normal-pass reading that can still be rewritten (Hz).  See
    #: _candidate_mask for the derivation.
    HR_MIN_NORMAL = 210.0
    #: Unvoiced frames quieter than this fraction of the clip's loud level
    #: (99th percentile frame RMS), or than this absolute floor, are skipped.
    HR_SILENCE_REL = 0.001
    HR_SILENCE_ABS = 1e-5
    #: Context frames kept on each side of a span so the UNet sees the same
    #: neighbourhood it would in a full-file pass.
    HR_CONTEXT_FRAMES = 64
    #: Once the spans cover this much of the clip, splitting costs more in
    #: per-span resampling and kernel launches than it saves, so the guide
    #: runs as one full pass instead.
    HR_FULL_PASS_FRACTION = 0.7

    def __init__(
        self,
        device,
        model_name="rmvpe.pt",
        sample_rate=16000,
        hop_size=160,
        high_register=None,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        # high_register: settings for the high-register corrector (keys:
        # enabled, mode, f0_ceil).  None reads them from assets/config.json;
        # callers pass a dict to override (extraction pins the mode).
        if high_register is None:
            high_register = load_high_register_settings()
        self.high_register = {**HIGH_REGISTER_DEFAULTS, **high_register}
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03):
        f0 = self.model.infer_from_audio(x, thred=filter_radius)
        if self.high_register["enabled"]:
            f0 = self._fix_high_register(x, f0, filter_radius)
        return f0

    def _fix_high_register(self, x, normal, thred, gate=150.0):
        # rmvpe.pt cannot track fundamentals above ~1040 Hz: its training data
        # (vocal pitch corpora) tops out near C6, so for higher notes the
        # salience net confidently reports f/2 or f/3 instead.  A second rmvpe
        # pass on octave-down-resampled audio reads those registers correctly;
        # it is used only to fix the octave class of the normal pass, keeping
        # the normal 10 ms contour wherever it already agrees.
        # Mode "true_pitch" (default) writes the true pitch, capped at f0_ceil
        # (default 1250 Hz) because RVC models trained on stock rmvpe labels
        # cannot synthesize above that and collapse; above the ceiling stock's
        # octave-down drive is kept (the vocoder folds its harmonics back up).
        # Mode "fold" is a compatibility mode for such models: it fixes only
        # wrong-pitch-class frames, using half the true pitch so every drive
        # stays in the model's trained register.
        ceil = float(self.high_register["f0_ceil"])
        guide = self._half_speed_guide(
            x, len(normal), thred, mask=self._candidate_mask(x, normal)
        )
        out = normal.copy()
        nv, gv = normal > 0, guide > 0
        both = nv & gv
        c_keep = np.full(len(normal), 1e9)
        c_double = np.full(len(normal), 1e9)
        c_keep[both] = np.abs(1200.0 * np.log2(normal[both] / guide[both]))
        c_double[both] = np.abs(1200.0 * np.log2(2.0 * normal[both] / guide[both]))
        agree = both & (c_keep < gate)
        cand_dbl = both & ~agree & (c_double < gate)
        if self.high_register["mode"] == "fold":
            fold_other = (both & ~agree & ~cand_dbl) & (guide >= 990.0)
            fold_fill = (~nv) & gv & (guide >= 900.0)
            out[fold_other] = guide[fold_other] / 2.0
            out[fold_fill] = guide[fold_fill] / 2.0
            return out
        dbl = cand_dbl & (guide >= 460.0) & (2.0 * normal <= ceil)
        other = (both & ~agree & ~cand_dbl) & (guide >= 990.0) & (guide <= ceil)
        filled = (~nv) & gv & (guide >= 460.0) & (guide <= ceil)
        out[dbl] = 2.0 * normal[dbl]
        out[other] = guide[other]
        out[filled] = guide[filled]
        return out

    def _candidate_mask(self, x, normal):
        # Frames the corrector could possibly rewrite.  The guide pass costs
        # twice a normal pass (half speed means twice the frames), so it only
        # runs where a rewrite is reachable at all.
        #
        # Voiced frames: every write path needs the guide to read >= 460 Hz.
        # The doubling path additionally needs 2*normal within `gate` cents of
        # the guide, which bounds normal at 460 * 2**(-150/1200) / 2 = 211 Hz.
        # The wrong-octave path (guide >= 990 Hz) has no such algebraic bound,
        # but it repairs the f/2 and f/3 misreads rmvpe actually makes, whose
        # lowest reading for a 990 Hz fundamental is 330 Hz; MIN_NORMAL keeps
        # margin down to f/4.  A misread deeper than that is not repaired.
        #
        # Unvoiced frames: the fill path needs the guide to find a pitch where
        # the normal pass found none, which cannot happen in silence.  Frames
        # 60 dB below the loud level of the clip are treated as silent.
        voiced = normal > 0
        candidate = voiced & (normal >= self.HR_MIN_NORMAL)
        quiet = ~voiced
        if quiet.any():
            y = np.asarray(x, dtype=np.float32)
            frames = np.pad(y, (0, self.hop_size), mode="constant")
            idx = np.arange(len(normal))[:, None] * self.hop_size + np.arange(
                self.hop_size
            )
            rms = np.sqrt(
                np.mean(frames[np.minimum(idx, len(frames) - 1)] ** 2, axis=1)
            )
            loud = np.percentile(rms, 99) if rms.size else 0.0
            floor = max(loud * self.HR_SILENCE_REL, self.HR_SILENCE_ABS)
            candidate |= quiet & (rms > floor)
        return candidate

    def _mask_spans(self, mask, n_target):
        # Contiguous frame ranges covering every candidate, widened by the
        # context the UNet needs to reproduce its full-file output and merged
        # when the gap between them is smaller than that context anyway.
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 2 * self.HR_CONTEXT_FRAMES)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
        starts = np.maximum(starts - self.HR_CONTEXT_FRAMES, 0)
        ends = np.minimum(ends + self.HR_CONTEXT_FRAMES, n_target)
        return list(zip(starts.tolist(), ends.tolist()))

    def _half_speed_guide(self, x, n_target, thred, mask=None):
        # Resampling to 2x and feeding it to a 16 kHz model plays the audio at
        # half speed: every pitch lands an octave lower, inside the register
        # rmvpe was trained on, at twice the frame rate.  The result is mapped
        # back onto the normal pass's 10 ms grid and doubled in frequency.
        #
        # With a mask, only the spans that can affect the result are read.
        # Frame alignment survives slicing because spans start on a hop
        # boundary: target frame i sits at sample i*hop, which after the 2x
        # resample is guide frame 2*i, so a span starting at frame s maps
        # target frame i to guide frame 2*(i - s).
        y = np.asarray(x, dtype=np.float32)
        guide = np.zeros(n_target, dtype=np.float64)
        spans = [(0, n_target)] if mask is None else self._mask_spans(mask, n_target)
        covered = sum(end - start for start, end in spans)
        if covered >= self.HR_FULL_PASS_FRACTION * n_target:
            spans = [(0, n_target)]
        for start, end in spans:
            s0 = start * self.hop_size
            s1 = len(y) if end >= n_target else min(end * self.hop_size, len(y))
            if s1 - s0 < self.hop_size:
                continue
            y2 = librosa.resample(
                y[s0:s1],
                orig_sr=self.sample_rate,
                target_sr=2 * self.sample_rate,
                res_type="soxr_vhq",
            )
            f0h = self.model.infer_from_audio(y2, thred=thred)
            j = np.minimum(2 * (np.arange(start, end) - start), len(f0h) - 1)
            v = f0h[j]
            guide[start:end] = np.where(v > 0, 2.0 * v, 0.0)
        return guide

class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160, model_name="fcpe_ddsp.pt"):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = fcpe_f0_predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            self.device
        )

    def get_f0(self, x, p_len=None, filter_radius=0.006, test_time_augmentation=False):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
                test_time_augmentation=test_time_augmentation,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0

class CREPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0