import os
import glob
import json
import signal
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

import librosa
import numpy as np
import soundfile as sf

from collections import OrderedDict
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rvc.train.messages import (
    TENSORBOARD_VALIDATION_AUDIO_NAMES,
    TENSORBOARD_VALIDATION_AUDIO_TAG,
    TENSORBOARD_VALIDATION_AXIS_X,
    TENSORBOARD_VALIDATION_AXIS_Y,
    TENSORBOARD_VALIDATION_DB_LABEL,
    TENSORBOARD_VALIDATION_DIFFERENCE_LABEL,
    TENSORBOARD_VALIDATION_FOOTER,
    TENSORBOARD_VALIDATION_MEL_TAG,
    TENSORBOARD_VALIDATION_MEL_TITLES,
    TENSORBOARD_VALIDATION_PREVIEW_DIR,
    TENSORBOARD_VALIDATION_SOURCE_TAG,
)


MATPLOTLIB_FLAG = False
#: Defaults for the validation preview figure.  Overridable per model through
#: ``validation_preview_dpi`` / ``_width`` / ``_height`` in the config JSON.
#:
#: The two knobs are not interchangeable.  ``figsize`` is in inches and dpi
#: converts to pixels, so raising **dpi** scales the whole figure -- panels and
#: text together -- and is what "make it sharper" means.  Raising **figsize**
#: at a fixed dpi gives the panels more room while the text stays the same
#: physical size, so labels shrink relative to the plot.  67 x 24in lands on
#: 1608px wide, which is the historical output.
VALIDATION_PREVIEW_DPI = 67
VALIDATION_PREVIEW_FIGSIZE = (24.0, 5.8)

#: Candidate ticks for the frequency axis, in Hz.  A mel axis is close to
#: logarithmic, so the evenly spaced ticks a linear axis wants would crowd the
#: bottom octaves into a few pixels and label almost nothing where the voice
#: actually lives.  Whatever falls outside the filterbank's range is dropped.
VALIDATION_PREVIEW_FREQUENCY_TICKS = (
    0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
)

debug_save_load = False

from itertools import chain
from mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.lib.terminal import (
    error as print_error,
    info,
    print_settings_panel,
    warning,
)

def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def remap_optimizer_state(optimizer, model, opt_state):
    """Prune a saved optimizer state dict so it fits an optimizer whose parameter set changed
        ( e.g. layers were frozen between runs ).

    The moments of params that still exist are kept and removed params' state is dropped.
    Returns None if nothing can be salvaged.
    """
    model_params = list(model.parameters())
    old_index = {id(p): i for i, p in enumerate(model_params)}
    new_params = [p for g in optimizer.param_groups for p in g["params"]]
    new_index = {id(p): j for j, p in enumerate(new_params)}

    state = {}
    for old_i, group_state in opt_state.get("state", {}).items():
        if isinstance(old_i, int) and old_i < len(model_params):
            p = model_params[old_i]
            if id(p) in new_index:
                state[new_index[id(p)]] = group_state

    saved_groups = opt_state.get("param_groups", [])
    if not saved_groups:
        return None

    param_groups = []
    for i, group in enumerate(optimizer.param_groups):
        new_group = {}
        for key, value in saved_groups[min(i, len(saved_groups) - 1)].items():
            if key != "params":
                new_group[key] = value
        new_group["params"] = group["params"]
        if "lr_scale" in group:
            new_group["lr_scale"] = group["lr_scale"]
        param_groups.append(new_group)

    if not state and not param_groups:
        return None
    return {"state": state, "param_groups": param_groups}


def load_checkpoint(checkpoint_path, model, optimizer=None, strict_load=True, ema=None):
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state = model.module if hasattr(model, "module") else model
    expected_architecture = getattr(model_state, "architecture_id", None)
    checkpoint_architecture = checkpoint_dict.get("architecture_id")
    if expected_architecture and expected_architecture != "vits_gaussian_v1":
        if checkpoint_architecture != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch: expected '{expected_architecture}', "
                f"received '{checkpoint_architecture or 'unknown'}'."
            )
    assert_excitation_matches(model_state, checkpoint_dict)
    assert_decoder_layout_matches(model_state, checkpoint_dict)
    assert_periods_match(model_state, checkpoint_dict)
    model_state.load_state_dict(checkpoint_dict["model"], strict=strict_load)

    if ema is not None:
        # Absent from every checkpoint written before the EMA existed.  Seeding
        # the shadow from the restored weights is the correct restart: it is
        # what the average would be if this were step zero, and it costs only
        # the averaging already done.
        if ema.load_state_dict(checkpoint_dict.get("ema"), model_state):
            info(f"Loaded EMA state ({ema.updates} updates).", tag="[RESUME]")
        else:
            warning(
                "No usable EMA state in the checkpoint; seeding it from the "
                "weights.",
                tag="[RESUME]",
            )

    if optimizer:
        opt_state = checkpoint_dict.get("optimizer")
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
                info("Loaded optimizer state.", tag="[RESUME]")
            except ValueError:
                warning(
                    "The optimizer's parameter set changed (layers were frozen, "
                    "for instance); pruning the saved state to the surviving "
                    "params, with the LR re-anchored to the saved value.",
                    tag="[RESUME]",
                )
                pruned = remap_optimizer_state(optimizer, model_state, opt_state)
                if pruned is not None:
                    optimizer.load_state_dict(pruned)
                    info(
                        "Loaded optimizer state (pruned to the surviving params).",
                        tag="[RESUME]",
                    )
                else:
                    warning(
                        "Could not remap the optimizer state; starting the "
                        "optimizer fresh.",
                        tag="[RESUME]",
                    )
        else:
            if strict_load:
                raise ValueError(f"[ERROR] Missing optimizer state...")
            else:
                warning(
                    "No optimizer state in the checkpoint; starting the "
                    "optimizer fresh.",
                    tag="[RESUME]",
                )


    info(
        f"Loaded '{os.path.basename(checkpoint_path)}' at iteration "
        f"{checkpoint_dict['iteration']}.",
        tag="[RESUME]",
    )
    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        # ``save_checkpoint``'s ``extra``: an empty dict when absent, which is
        # what every checkpoint written before the key existed returns.  The
        # slot originally held FP16 scaler state directly; now the scaler state
        # is one key *inside* ``extra`` alongside the R1 controller, so a
        # checkpoint from either era loads without a version check.
        checkpoint_dict.get("extra") or {},
    )

def excitation_source(model):
    """A short name for the decoder's excitation, or ``None`` if it has no say.

    Only ``sine`` is built since 2026-09-03, but the guard stays: the removed
    sources owned different state-dict keys (``bank`` a ``phase_offset`` sized
    by its harmonic count, ``comb`` none at all) and the generator resumes
    *non-strictly*, so a bank or comb checkpoint loads into this synthesiser
    without raising and leaves ``m_source.merge`` at its random init.  A
    sentence beats training on from a silently wrong excitation.
    """
    decoder = getattr(model, "dec", None)
    source = getattr(decoder, "source_type", None)
    return None if source is None else str(source)


def assert_excitation_matches(model, checkpoint_dict, origin="checkpoint"):
    """Absent means ``sine``: every checkpoint predating the key is one."""
    expected = excitation_source(model)
    if expected is None:
        return
    found = checkpoint_dict.get("excitation_source") or "sine"
    if found != expected:
        raise ValueError(
            f"Excitation mismatch: this run builds '{expected}' but the "
            f"{origin} was trained with '{found}'. The 'comb' and 'bank' "
            f"sources were removed on 2026-09-03, so such a checkpoint cannot "
            f"be resumed -- start a fresh run."
        )


#: What the shipped configs carried before the decoder layout became
#: configurable.  A checkpoint written before the key exists is one of these,
#: and reading it as "whatever this run builds" would make the guard useless in
#: exactly the case it exists for.
LEGACY_UPSAMPLE_RATES = {32000: [4, 4, 4, 5]}

#: ``AntiAliasedActivation``'s design before 2026-09-03: 2x oversampling and a
#: 25-tap filter cutting at 0.90 of the stage's Nyquist.  Same reasoning as
#: ``LEGACY_UPSAMPLE_RATES`` -- reading a key-less checkpoint as "whatever this
#: run builds" makes the guard useless where it matters, and this one is worth
#: 19 dB of round-trip error, so it is not a cosmetic difference.
LEGACY_ANTIALIAS_FILTER = [2, 6, 0.9, 6.0]

#: The trunk upsamplers' interpolation filter before 2026-09-03: flat
#: ``12 / 0.90 / 6.0`` at every stage.  Same contract as the two above -- the
#: kernels are non-persistent, so a checkpoint cannot tell the designs apart,
#: and this one moves the worst image by 57 dB.
LEGACY_UPSAMPLE_FILTER = [[12, 12, 12, 12], [0.9] * 4, [6.0] * 4]


def upsample_filter(decoder):
    """The trunk upsamplers' per-stage filter design, or ``None`` if absent."""

    width = getattr(decoder, "filter_width", None)
    if width is None:
        return None
    return [
        [int(v) for v in width],
        [float(v) for v in decoder.rolloff],
        [float(v) for v in decoder.filter_beta],
    ]


def antialias_adain(decoder):
    """Whether the stages' ``AdaIN`` activations are anti-aliased.

    A checkpoint trained before 2026-09-03 has the same ``antialias`` and
    ``antialias_stages`` as one trained after and a different signal path: the
    six ``AdaIN`` per stage were raw either way until the flag reached them,
    and they turned out to be the site that decides whether the inharmonic
    lines are there.  Nothing else in the layout separates the two.
    """

    from rvc.lib.algorithm.generators.refinegan2 import AdaIN

    found = [m.antialias for m in decoder.modules() if isinstance(m, AdaIN)]
    return None if not found else bool(any(found))


def antialias_filter(decoder):
    """The anti-aliased activations' filter design, or ``None`` when there are none.

    Read off a built module rather than off the config: the design is a set of
    constructor defaults, so a decoder built by hand and one built from a
    config have to report the same thing.
    """

    from rvc.lib.algorithm.resampling import AntiAliasedActivation

    for module in decoder.modules():
        if isinstance(module, AntiAliasedActivation):
            factor, width, rolloff, beta = module.design
            return [int(factor), int(width), float(rolloff), float(beta)]
    return None


def decoder_layout(model):
    """The decoder arrangement that leaves no trace in the weights.

    Two of these are invisible to ``load_state_dict``: reordering
    ``upsample_rates`` keeps all 271 tensors' keys *and* shapes (channel counts
    follow the stage index, not the rate), and ``AntiAliasedActivation``'s
    kernels are non-persistent so wrapping an activation adds no key -- nor
    does *redesigning* its filter, which is why the design itself is reported
    here.  Any of them loads silently into a decoder trained for a different
    signal path.

    The source gain a strict load *would* catch, but it is reported here too:
    a named mismatch beats a missing-key list.  Same additive-key contract as
    ``excitation_source``.  ``None`` for anything that is not this decoder.
    """

    decoder = getattr(model, "dec", None)
    rates = getattr(decoder, "upsample_rates", None)
    if rates is None:
        return None
    return {
        "upsample_rates": [int(rate) for rate in rates],
        "antialias_stages": [int(s) for s in getattr(decoder, "antialias_stages", ())],
        "antialias": str(getattr(decoder, "antialias", "none")),
        "source_gain": bool(getattr(decoder, "has_source_gain", False)),
        "source_bands": int(getattr(decoder, "source_bands", 0)),
        "antialias_rates": [
            int(r) for r in getattr(decoder, "antialias_rates", ())
        ],
        # ``None`` when nothing is anti-aliased, so a run with the activations
        # raw does not carry a filter design it never used -- and does not
        # mismatch every other such run over one.
        "antialias_filter": antialias_filter(decoder),
        "antialias_adain": antialias_adain(decoder),
        # Imaging, not aliasing: what the interpolation filter leaves of the
        # spectral copies zero-stuffing makes.  No ``antialias_*`` option
        # touches it, and it is just as invisible to ``load_state_dict``.
        "upsample_filter": upsample_filter(decoder),
    }


def assert_decoder_layout_matches(model, checkpoint_dict, origin="checkpoint"):
    """Absent means the shipped legacy layout: no anti-aliasing, old ordering."""

    expected = decoder_layout(model)
    if expected is None:
        return
    found = checkpoint_dict.get("decoder_layout")
    if found is None:
        sample_rate = int(getattr(model, "sr", 0) or 0)
        found = {
            "upsample_rates": LEGACY_UPSAMPLE_RATES.get(
                sample_rate, expected["upsample_rates"]
            ),
            "antialias_stages": [],
            "antialias": "none",
            "source_gain": False,
            "source_bands": 0,
            "antialias_rates": [],
            "antialias_filter": None,
            "upsample_filter": None,
            "antialias_adain": False,
        }
    design = found.get("antialias_filter") or None
    imaging = found.get("upsample_filter") or None
    # Absent means raw, which is what every run before the flag existed had.
    adain = bool(found.get("antialias_adain", False))
    found = {
        "upsample_rates": [int(r) for r in found.get("upsample_rates", [])],
        "antialias_stages": [int(s) for s in found.get("antialias_stages", [])],
        "antialias": str(found.get("antialias", "none")),
        "source_gain": bool(found.get("source_gain", False)),
        "source_bands": int(found.get("source_bands", 0)),
        # Absent means none: the two loops' activations were raw in every run
        # before 2026-09-03, and wrapping one adds no state-dict key.
        "antialias_rates": [int(r) for r in found.get("antialias_rates", [])],
    }
    # Absent means the legacy design, but only for a checkpoint that had
    # anti-aliased activations at all: a run with none has no design, and
    # inventing one would make every raw-activation checkpoint mismatch over
    # a field neither of them uses.
    if design is None and (
        found["antialias_rates"]
        or (found["antialias_stages"] and found["antialias"] != "none")
    ):
        design = LEGACY_ANTIALIAS_FILTER
    found["antialias_filter"] = (
        None
        if design is None
        else [int(design[0]), int(design[1]), float(design[2]), float(design[3])]
    )
    # Absent means the flat legacy design, sized to the checkpoint's own stage
    # count -- unlike the anti-aliasing, every RefineGAN run ever had these
    # upsamplers, so there is no "had none" case to leave as ``None``.  A
    # decoder that reports no schedule at all is a different vocoder, and then
    # ``expected`` carries ``None`` too.
    if imaging is None and expected["upsample_filter"] is not None:
        stages = len(found["upsample_rates"]) or len(expected["upsample_rates"])
        imaging = [[12] * stages, [0.9] * stages, [6.0] * stages]
    found["upsample_filter"] = (
        None
        if imaging is None
        else [
            [int(v) for v in imaging[0]],
            [float(v) for v in imaging[1]],
            [float(v) for v in imaging[2]],
        ]
    )
    found["antialias_adain"] = (
        None if expected["antialias_adain"] is None else adain
    )
    if found != expected:
        raise ValueError(
            f"Decoder layout mismatch: this run builds {expected} but the "
            f"{origin} was trained with {found}. Neither the stage ordering "
            f"nor the anti-aliased activations appear in any weight, so this "
            f"is the only thing that can tell them apart. Set "
            f"upsample_rates / refinegan2_antialias_stages / "
            f"refinegan2_antialias / refinegan2_antialias_rates / "
            f"refinegan2_source_gain to match, or start a fresh run. "
            f"``antialias_adain`` says whether the stages' AdaIN activations "
            f"are anti-aliased; it follows refinegan2_antialias and is False in "
            f"every checkpoint written before 2026-09-03. "
            f"``antialias_filter`` is "
            f"[factor, width, rolloff, beta] and is a constructor default of "
            f"AntiAliasedActivation; ``upsample_filter`` is "
            f"[widths, rolloffs, betas] per stage for the trunk's "
            f"interpolation filters. Neither is a config key: a mismatch "
            f"there means the checkpoint predates the current filter design."
        )


def discriminator_periods(model):
    """The period set a discriminator was built with, or ``None`` if it has none."""
    periods = getattr(model, "periods", None)
    return None if periods is None else [int(p) for p in periods]


def assert_periods_match(model, checkpoint_dict, origin="checkpoint"):
    """Refuse to load a discriminator whose periods differ from this run's.

    This one is *only* catchable by an explicit key, which is why the key
    exists.  A period never appears in a parameter shape -- ``DiscriminatorP``
    reshapes the waveform and then convolves with ``(k, 1)`` kernels, so the
    period lives in the ``view`` and nowhere in the weights -- and both the
    stock set and every rate-scaled one have five branches.  A strict
    ``load_state_dict`` therefore succeeds perfectly while every branch starts
    folding at a frequency its weights were never trained on, which shows up as
    a discriminator that appears to have forgotten how to discriminate and no
    error anywhere.  Nothing else in the checkpoint can tell the two apart.

    ``None`` means "written before the key existed", and every such checkpoint
    is one of the un-scaled runs -- so it is compared against the stock set
    rather than waved through.  Which stock set is the one belonging to *this*
    run's discriminator version: ``v2``'s eight periods are Applio's and the
    HiFi-GAN pretrains shipped with this fork are trained against exactly them,
    so charging an unkeyed checkpoint with ``v3``'s five would reject every
    bundled pretrained D on a stock HiFi-GAN run.  See ``PERIODS_BY_RATE`` for
    why the scaling reaches a run through its config and not through code.
    """
    expected = discriminator_periods(model)
    if expected is None:
        return
    found = checkpoint_dict.get("discriminator_periods")
    if found is None:
        from rvc.lib.algorithm.discriminators.multi.mpd_msd_combined import (
            DISCRIMINATOR_VERSIONS,
        )

        target = getattr(model, "module", model)
        version = getattr(target, "version", "v3")
        stock, _resolutions, _strides = DISCRIMINATOR_VERSIONS.get(
            version, DISCRIMINATOR_VERSIONS["v3"]
        )
        found = list(stock)
    found = [int(p) for p in found]
    if found != expected:
        raise ValueError(
            f"Discriminator period mismatch: this run builds {expected} but the "
            f"{origin} was trained with {found}. The periods are rate-scaled "
            f"now (see PERIODS_BY_RATE); set d_periods to {found} in the "
            f"experiment's config.json to keep resuming this run, or start a "
            f"fresh one."
        )


def save_checkpoint(
    model, optimizer, learning_rate, iteration, checkpoint_path, ema=None, extra=None
):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    model_instance = model.module if hasattr(model, "module") else model

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }
    architecture_id = getattr(model_instance, "architecture_id", None)
    if architecture_id is not None:
        checkpoint_data["architecture_id"] = architecture_id
    # Additive key.  "model" still holds the live weights, so anything that
    # reads these checkpoints without knowing about the EMA -- older code, the
    # extractor, the blender -- keeps seeing exactly what it saw before.
    # Same additive contract, and the reason it is a separate key rather than a
    # suffix on ``architecture_id``: the id is what the wider RVC ecosystem
    # reads off a checkpoint, and it has to stay ``vits_gaussian_v1``.
    source = excitation_source(model_instance)
    if source is not None:
        checkpoint_data["excitation_source"] = source
    # Same additive contract again, and for the same reason as the excitation:
    # the period set is invisible in the weights, so without this key a resume
    # cannot tell that it changed.  Only discriminators have it.
    periods = discriminator_periods(model_instance)
    if periods is not None:
        checkpoint_data["discriminator_periods"] = periods
    # Same additive contract once more: the stage ordering and the anti-aliased
    # activations are invisible in the weights, so a resume cannot tell that
    # either changed. Only generators have it.
    layout = decoder_layout(model_instance)
    if layout is not None:
        checkpoint_data["decoder_layout"] = layout
    if ema is not None:
        checkpoint_data["ema"] = ema.state_dict()
    # Same additive contract: plain Python scalars for training-loop controllers
    # whose state is expensive to re-earn on a resume.  Nothing that reads these
    # checkpoints without knowing the key is affected, and ``weights_only=True``
    # unpickles them, so the loader stays hardened.
    if extra:
        checkpoint_data["extra"] = dict(extra)

    torch.save(checkpoint_data, checkpoint_path)
    info(f"Saved '{os.path.basename(checkpoint_path)}'.", tag="[SAVE]")

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def limit_audio_peak(audio, max_peak=0.98):
    """Return a finite mono preview whose peak stays inside TensorBoard's range."""
    if torch.is_tensor(audio):
        preview = audio.detach().float().cpu()
        preview = torch.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        preview = preview.reshape(-1)
        if preview.numel() == 0:
            return preview
        peak = preview.abs().amax()
        if torch.isfinite(peak) and peak.item() > max_peak:
            preview = preview * (max_peak / peak)
        return preview.clamp(-max_peak, max_peak)

    preview = np.asarray(audio, dtype=np.float32)
    preview = np.nan_to_num(
        preview,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=True,
    ).reshape(-1)
    if preview.size == 0:
        return preview
    peak = float(np.max(np.abs(preview)))
    if np.isfinite(peak) and peak > max_peak:
        preview *= max_peak / peak
    return np.clip(preview, -max_peak, max_peak).astype(np.float32, copy=False)


def _mel_to_numpy(mel):
    if torch.is_tensor(mel):
        mel = mel.detach().float().cpu().numpy()
    mel = np.nan_to_num(np.asarray(mel, dtype=np.float32), copy=True)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError("Validation mel spectrograms must be two-dimensional.")
    return mel


def _mel_axis_positions(frequencies, mel_bins, mel_fmin, mel_fmax):
    """Where ``frequencies`` land on a mel filterbank's bin-index axis.

    The filterbank's ``mel_bins + 2`` band edges are evenly spaced *in mel*, so
    a frequency's position in bin-index space is linear in ``hz_to_mel``.  Bin 0
    is centred on edge 1 and the last bin on edge ``mel_bins``, which is where
    the ``mel_bins + 1`` scaling and the -1 offset come from.

    ``htk=False`` is not a preference: it is librosa's default, and so it is
    what ``librosa.filters.mel`` used to build the basis in ``mel_processing``.
    Reading the axis off a different mel scale than the data was binned with
    would swap one wrong answer for another.
    """
    low = librosa.hz_to_mel(float(mel_fmin), htk=False)
    high = librosa.hz_to_mel(float(mel_fmax), htk=False)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("The mel axis needs a positive frequency range.")
    mels = librosa.hz_to_mel(np.asarray(frequencies, dtype=float), htk=False)
    return (mels - low) / (high - low) * (mel_bins + 1) - 1.0


def _format_frequency(frequency):
    return f"{frequency:g}" if frequency < 1000.0 else f"{frequency / 1000:g}k"


def plot_validation_preview_to_figure(
    predicted_mel,
    target_mel,
    epoch,
    global_step,
    sample_index=0,
    sample_rate=22050,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
):
    """Create a dark TensorBoard-style three-panel validation report.

    ``hop_length`` sets the time axis and must match the config the mels were
    produced with -- the default is a fallback, not a good guess.  The same goes
    for ``mel_fmin`` / ``mel_fmax``, which set the frequency axis; ``mel_fmax``
    of ``None`` means Nyquist, matching the mel extraction.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    target_mel = _mel_to_numpy(target_mel)
    predicted_mel = _mel_to_numpy(predicted_mel)
    mel_bins = min(target_mel.shape[0], predicted_mel.shape[0])
    frames = min(target_mel.shape[1], predicted_mel.shape[1])
    if mel_bins <= 0 or frames <= 0:
        raise ValueError("Validation mel spectrograms must not be empty.")

    target_mel = target_mel[:mel_bins, :frames]
    predicted_mel = predicted_mel[:mel_bins, :frames]
    difference_mel = predicted_mel - target_mel

    shared_values = np.concatenate((target_mel.ravel(), predicted_mel.ravel()))
    shared_low = float(np.quantile(shared_values, 0.01))
    shared_high = float(np.quantile(shared_values, 0.99))
    if not np.isfinite(shared_low) or not np.isfinite(shared_high):
        shared_low, shared_high = 0.0, 1.0
    if shared_low >= shared_high:
        shared_high = shared_low + 1.0

    difference_limit = max(
        1e-5,
        float(np.quantile(np.abs(difference_mel), 0.99)),
    )
    time_axis = np.arange(frames, dtype=np.float32) * float(hop_length) / float(sample_rate)
    # The mel filterbank spans DC to Nyquist, not DC to the sample rate. Using
    # the sample rate as the extent labelled every frequency tick at twice its
    # true value.
    top_frequency = float(sample_rate) / 2.0 if mel_fmax is None else float(mel_fmax)
    # The vertical axis is bin index, and mel bins are not evenly spaced in Hz.
    # Stretching them onto a linear 0..Nyquist ruler -- which is what an extent
    # of ``(0, top_frequency)`` does -- put every label except the two endpoints
    # in the wrong place, and not by a little: at 128 bins over 32 kHz the
    # tick reading "4.0k" sits on bin 32, whose real centre is 0.94 kHz. A
    # deficit under 1 kHz reads as a defect at 4 kHz, which is a diagnosis this
    # axis has no business making. Keep the image in bin space and move the
    # ticks instead, so what the eye measures is the filterbank's own spacing.
    tick_positions = _mel_axis_positions(
        VALIDATION_PREVIEW_FREQUENCY_TICKS, mel_bins, mel_fmin, top_frequency
    )
    visible = (tick_positions >= -0.5) & (tick_positions <= mel_bins - 0.5)
    tick_positions = tick_positions[visible]
    tick_labels = [
        _format_frequency(frequency)
        for frequency, keep in zip(VALIDATION_PREVIEW_FREQUENCY_TICKS, visible)
        if keep
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=tuple(figsize) if figsize else VALIDATION_PREVIEW_FIGSIZE,
        dpi=float(dpi) if dpi else VALIDATION_PREVIEW_DPI,
        facecolor="#10161f",
        gridspec_kw={
            "left": 0.045,
            "right": 0.955,
            "wspace": 0.16,
            "bottom": 0.25,
            "top": 0.86,
        },
    )
    figure.patch.set_facecolor("#10161f")
    panels = (predicted_mel, target_mel, difference_mel)
    panel_cmaps = ("inferno", "inferno", "turbo")
    panel_norms = (
        (shared_low, shared_high),
        (shared_low, shared_high),
        (-difference_limit, difference_limit),
    )
    for axis, mel, title, cmap, (vmin, vmax) in zip(
        axes,
        panels,
        TENSORBOARD_VALIDATION_MEL_TITLES,
        panel_cmaps,
        panel_norms,
    ):
        axis.set_facecolor("#10161f")
        rendered = axis.imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # Half a bin past the first and last centres, so a data coordinate
            # of ``i`` lands exactly on the centre of bin ``i``.
            extent=(time_axis[0], time_axis[-1], -0.5, mel_bins - 0.5),
        )
        axis.set_title(title, color="#f4f4f4", fontsize=17, fontweight="bold", pad=12)
        axis.set_xlabel(TENSORBOARD_VALIDATION_AXIS_X, color="#f4f4f4", fontsize=11)
        axis.set_ylabel(TENSORBOARD_VALIDATION_AXIS_Y, color="#f4f4f4", fontsize=11)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)
        axis.tick_params(colors="#f4f4f4", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#c9cdd3")
        colorbar = figure.colorbar(
            rendered,
            ax=axis,
            orientation="horizontal",
            fraction=0.055,
            pad=0.12,
        )
        colorbar.set_label(
            TENSORBOARD_VALIDATION_DIFFERENCE_LABEL
            if title == TENSORBOARD_VALIDATION_MEL_TITLES[-1]
            else TENSORBOARD_VALIDATION_DB_LABEL,
            color="#d7dbe1",
            fontsize=8,
        )
        colorbar.ax.tick_params(colors="#d7dbe1", labelsize=8)
        colorbar.outline.set_edgecolor("#c9cdd3")

    figure.text(
        0.5,
        0.055,
        TENSORBOARD_VALIDATION_FOOTER.format(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            n_mels=int(mel_bins),
            epoch=int(epoch),
            step=int(global_step),
        ),
        color="#bfc5ce",
        fontsize=10,
        ha="center",
    )
    return figure


def log_validation_preview(
    writer,
    experiment_dir,
    epoch,
    sample_index,
    global_step,
    sample_rate,
    predicted_mel,
    target_mel,
    predicted_wave=None,
    target_wave=None,
    source=None,
    hop_length=256,
    mel_fmin=0.0,
    mel_fmax=None,
    dpi=None,
    figsize=None,
):
    """Save and log one organized validation preview with mel difference."""
    sample_stem = f"sample_{int(sample_index):02d}"
    preview_dir = os.path.join(
        str(experiment_dir),
        TENSORBOARD_VALIDATION_PREVIEW_DIR,
        f"epoch_{int(epoch):04d}",
    )
    mel_dir = os.path.join(preview_dir, "mel")
    audio_dir = os.path.join(preview_dir, "audio")
    os.makedirs(mel_dir, exist_ok=True)

    figure = plot_validation_preview_to_figure(
        predicted_mel=predicted_mel,
        target_mel=target_mel,
        epoch=epoch,
        global_step=global_step,
        sample_index=sample_index,
        sample_rate=sample_rate,
        hop_length=hop_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        dpi=dpi,
        figsize=figsize,
    )
    image_path = os.path.join(mel_dir, f"{sample_stem}.png")
    try:
        figure.canvas.draw()
        composite = np.asarray(
            figure.canvas.buffer_rgba(),
            dtype=np.uint8,
        )[..., :3].copy()
        # ``figure.dpi`` rather than the module default: the figure may have
        # been built at an overridden dpi, and saving at a different one would
        # write a PNG that does not match what TensorBoard received.
        figure.savefig(image_path, dpi=figure.dpi)

        if writer is not None:
            writer.add_image(
                TENSORBOARD_VALIDATION_MEL_TAG.format(sample=sample_stem),
                torch.from_numpy(composite).permute(2, 0, 1),
                global_step,
                dataformats="CHW",
            )
            if source:
                writer.add_text(
                    TENSORBOARD_VALIDATION_SOURCE_TAG.format(sample=sample_stem),
                    str(source),
                    global_step=global_step,
                )
    finally:
        plt.close(figure)

    if predicted_wave is None or target_wave is None:
        return image_path

    os.makedirs(audio_dir, exist_ok=True)
    predicted_wave = limit_audio_peak(predicted_wave)
    target_wave = limit_audio_peak(target_wave)
    audio_length = min(predicted_wave.numel(), target_wave.numel())
    if audio_length <= 0:
        return image_path
    predicted_wave = predicted_wave[:audio_length]
    target_wave = target_wave[:audio_length]

    generated_path = os.path.join(audio_dir, f"{sample_stem}_generated.wav")
    original_path = os.path.join(audio_dir, f"{sample_stem}_original.wav")
    sf.write(generated_path, predicted_wave.numpy(), int(sample_rate), subtype="PCM_16")
    sf.write(original_path, target_wave.numpy(), int(sample_rate), subtype="PCM_16")

    if writer is not None:
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["generated"],
            ),
            predicted_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )
        writer.add_audio(
            TENSORBOARD_VALIDATION_AUDIO_TAG.format(
                sample=sample_stem,
                kind=TENSORBOARD_VALIDATION_AUDIO_NAMES["original"],
            ),
            target_wave.unsqueeze(0),
            global_step,
            sample_rate=int(sample_rate),
        )

    return image_path


def log_tensorboard_media(
    writer,
    namespace,
    global_step,
    sample_rate,
    figure=None,
    audio=None,
    text=None,
):
    """Write one sample's text, mel figure, and audio under a single namespace."""
    if writer is None:
        return

    for name, value in (text or {}).items():
        writer.add_text(
            f"{namespace}/{name}",
            str(value),
            global_step=global_step,
        )
    if figure is not None:
        writer.add_figure(
            f"{namespace}/mel",
            figure,
            global_step=global_step,
        )
    for name, value in (audio or {}).items():
        writer.add_audio(
            f"{namespace}/audio/{name}",
            limit_audio_peak(value),
            global_step=global_step,
            sample_rate=sample_rate,
        )


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


#: Application root, used to resolve the relative paths in a filelist.  Taken
#: from this file's location so it holds regardless of the working directory
#: the trainer was launched with.
APPLICATION_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_filepaths_and_text(filename, split="|", path_columns=(0, 1, 2, 3), root=None):
    """
    Load filepaths and associated text from a file.

    Paths are stored relative to the application root (see
    ``rvc/train/extract/preparing_files.py``) and are resolved back to absolute
    here.  ``os.path.join`` returns an absolute second argument unchanged, so
    filelists written by older versions -- which stored absolute paths -- keep
    loading without a migration step.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
        path_columns (tuple, optional): Which columns hold paths. The trailing
            speaker id must not be resolved, so the columns are explicit.
        root (str, optional): Base for relative paths. Defaults to the
            application root.
    """
    base = root or APPLICATION_ROOT
    rows = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(split)
            for column in path_columns:
                if column < len(fields):
                    fields[column] = os.path.normpath(
                        os.path.join(base, fields[column])
                    )
            rows.append(fields)
    return rows


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config_from_json(config_save_path):
    try:
        with open(config_save_path, "r") as f:
            config = json.load(f)
        config = HParams(**config)
        return config
    except FileNotFoundError:
        print_error(
            f"No model config at {config_save_path}. Run preprocessing and "
            "feature extraction first.",
            tag="[INIT]",
        )
        sys.exit(1)

def flush_writer(writer, rank):
    if rank == 0 and writer is not None:
        writer.flush()


# Currently has no use, kept for future ig.
def flush_writer_grad(writer, rank, global_step):
    if rank == 0 and writer is not None and global_step % 10 == 0:
        writer.flush()


def block_tensorboard_flush_on_exit(writer):
    def handler(signum, frame):
        warning(
            "Training interrupted; skipping the flush to avoid partial logs.",
            tag="[TRAIN]",
        )
        try:
            writer.close()
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def si_sdr(preds, target, eps=1e-8):
    """Scale-Invariant SDR"""
    preds = preds - preds.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    scaling_factor = (preds * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scaling_factor * target

    noise = preds - projection

    si_sdr_value = 10 * torch.log10((projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + eps)

    return si_sdr_value.mean()


def wave_to_mel(config, waveform, num_mels=None, for_loss=False):
    mel_spec = mel_spectrogram_torch(
        waveform.float().squeeze(1),
        config.data.filter_length,
        num_mels if num_mels is not None else config.data.n_mel_channels,
        config.data.sample_rate,
        config.data.hop_length,
        config.data.win_length,
        config.data.mel_fmin,
        config.data.mel_fmax,
        log_compression=not for_loss,
    )
    # Mel/STFT values and their gradients remain in FP32.  Converting this
    # tensor to FP16 after the transform makes the spectral loss a low
    # precision loss even when its autocast region is disabled.
    mel_spec = mel_spec.float()
    if for_loss:
        mel_spec = torch.log1p(mel_spec * 1000.0)
    return mel_spec


def small_model_naming(model_name, epoch, global_step):
    return f"{model_name}_{epoch}e_{global_step}s.pth"


def old_session_cleanup(now_dir, model_name):
    for root, dirs, files in os.walk(os.path.join(now_dir, "logs", model_name), topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension == ".0"
                or (file_name.startswith("D_") and file_extension == ".pth")
                or (file_name.startswith("G_") and file_extension == ".pth")
                or (file_name.startswith("added") and file_extension == ".index")
            ):
                os.remove(file_path)
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                os.rmdir(folder_path)

    info("Cleanup done.", tag="[INIT]")


#: Schedulers whose decay is defined by an endpoint when ``lr_final_ratio`` is
#: set, which makes ``lr_decay``/``exp_decay_gamma`` dead config for them.  Kept
#: next to the panel because the panel is the only place that has to know.
_ENDPOINT_SCHEDULERS = frozenset(
    {"exp decay epoch", "exp decay step", "cosine annealing", "cosine annealing epoch"}
)


def print_init_setup(
    warmup_duration,
    rank,
    use_warmup,
    config,
    optimizer_choice_g,
    optimizer_choice_d,
    lr_scheduler,
    exp_decay_gamma,
    spectral_loss,
    lr_final_ratio=None,
    amp_dtype=None,
):
    if rank != 0:
        return

    tf32_enabled = (
        torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.allow_tf32
    )
    # TF32 is orthogonal to autocast: it only changes how cuDNN/cuBLAS run the
    # convolutions internally (11-bit mantissa), and it applies to the FP32 ops
    # that remain under autocast just as much as it does without it.  So it is
    # reported as a qualifier on either line rather than as a precision of its
    # own.
    #
    # ``amp_dtype`` has to be passed in rather than inferred here.  This row read
    # the TF32 flags alone until 2026-08-26, so it printed "FP32 (TF32
    # matmul/conv)" on every run -- including FP16 ones, where autocast and the
    # GradScaler were both live.  The banner prints well before the "AMP enabled"
    # line, so it was the only precision statement most runs ever saw.
    #
    # Master weights stay FP32 in every case, so the row names only what
    # changes: the autocast dtype, whether a GradScaler is live, and TF32.  The
    # panel's value column is narrow enough that a longer string is silently
    # truncated, which is how this row came to be misleading in the first place.
    if amp_dtype is None:
        precision = "FP32 (TF32 matmul/conv)" if tf32_enabled else "FP32"
    else:
        name = {torch.float16: "FP16", torch.bfloat16: "BF16"}.get(
            amp_dtype, str(amp_dtype).rsplit(".", 1)[-1].upper()
        )
        # BF16 carries FP32's exponent range, so it needs no loss scaling.
        scaler = " + GradScaler" if amp_dtype == torch.float16 else ""
        tf32_note = " (TF32 conv)" if tf32_enabled else ""
        precision = f"{name} autocast{scaler}{tf32_note}"

    rows = [
        ("PRECISION", precision),
        ("cudnn.benchmark", torch.backends.cudnn.benchmark),
        ("cudnn.deterministic", torch.backends.cudnn.deterministic),
        (
            "Optimizer (G/D)",
            optimizer_choice_g
            if optimizer_choice_g == optimizer_choice_d
            else f"G: {optimizer_choice_g} | D: {optimizer_choice_d}",
        ),
        ("Spectral loss", spectral_loss),
    ]

    def scheduler_value(name, gamma):
        if name == "none":
            return "Disabled"
        # ``lr_final_ratio`` decides the decay by its endpoint and supersedes
        # the per-step/per-epoch gamma entirely, so printing the gamma there
        # advertises a number that has no effect -- and invites tuning it.  Show
        # the endpoint that is actually in force instead; the gamma comes back
        # the moment there is no ratio to override it.
        if lr_final_ratio is not None and name in _ENDPOINT_SCHEDULERS:
            shown = "cosine annealing" if name.startswith("cosine annealing") else name
            return f"{shown}, to {lr_final_ratio:g}x over the run"
        if name in ("cosine annealing", "cosine annealing epoch"):
            return "cosine annealing"
        return f"{name}, gamma: {gamma}"

    rows.extend(
        [
            ("LR scheduler (G/D)", scheduler_value(lr_scheduler, exp_decay_gamma)),
        ]
    )

    if use_warmup:
        rows.append(("Warmup", f"{warmup_duration} epochs"))

    print_settings_panel(rows)

def train_loader_safety(train_loader):
    if len(train_loader) < 3:
        print_error(
            "Not enough data in the training set. Did the preprocessing step "
            "slice the audio files?",
            tag="[INIT]",
        )
        os._exit(1)


def substitute_speaker_embeddings(state_dict, net_g):
    """Swap a pretrain's ``emb_g.weight`` for the model's own fresh rows.

    Substituting rather than dropping the key on purpose: the VITS-latent
    vocoders load their pretrained generator *strictly*, where a missing key is
    an error, and a load that silently skips a tensor is exactly what this path
    must not produce.  Returns a shallow copy -- the caller's checkpoint dict is
    left alone, because it may still be read for other keys.
    """

    if "emb_g.weight" not in state_dict:
        return state_dict
    target = net_g.module if hasattr(net_g, "module") else net_g
    updated = dict(state_dict)
    updated["emb_g.weight"] = target.emb_g.weight.detach().cpu().clone()
    return updated


class SpeakerLayout(NamedTuple):
    """How many rows ``emb_g`` gets, and whether a pretrain's rows are usable.

    ``embed_dim`` is what the synthesizer is built with.  ``reset_pretrained``
    says the pretrained checkpoint's speaker table describes *other* speakers
    and must not be inherited -- see :func:`verify_spk_dim`.
    """

    embed_dim: int
    reset_pretrained: bool


def verify_spk_dim(
    config,
    model_info_path,
    experiment_dir,
    latest_checkpoint_path,
    rank,
    pretrainG
):
    """Resolve ``spk_embed_dim`` and decide whether to keep a pretrain's ``emb_g``.

    Three sources, most specific last: the config's default, the dataset's own
    count from ``model_info.json``, and a checkpoint's row count.

    The last one used to win unconditionally, which is what made a multispeaker
    fine-tune start with *the pretrain's* speaker table.  That is not a harmless
    initialisation.  Measured on ``f0G40k``: ``dec.cond`` reads the directions
    the trained embedding occupies with 1.53x the gain it gives random ones, so
    at step 0 the decoder renders each of your speakers as a specific, confident
    pretrain identity, and the run has to move away from a wrong answer rather
    than toward a right one.  On a small dataset it does not move far enough and
    the pretrain's timbre stays audible -- the leak people report.  Dropping the
    table costs nothing in separation: after ``dec.cond`` four freshly
    initialised speakers sit 62.3 apart against 41.0 for four inherited ones.

    The rule is the shape, because the shape is what actually carries the
    question.  Rows that disagree with the dataset's speaker count cannot be
    this dataset's speakers, so they go.  Rows that agree almost certainly are
    -- that is staged pretraining handing its own table forward -- so they stay.
    Single-speaker runs keep the old behaviour outright: there is no
    cross-speaker leak to fix, and row 0 is a working starting timbre.

    A resume is never touched: ``G_*.pth`` in the experiment folder *is* this
    run's table and its width is not negotiable.
    """

    embedder_name = "contentvec"  # Default embedder
    spk_dim = config.model.spk_embed_dim  # 109 default speakers
    dataset_speakers = None

    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            dataset_speakers = int(model_info["speakers_id"])
            spk_dim = dataset_speakers
    except Exception as e:
        if rank == 0:
            warning(
                f"Could not read the model info file ({e}); using defaults.",
                tag="[INIT]",
            )

    reset_pretrained = False
    try:
        last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
        chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            checkpoint_speakers = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
            if (
                last_g is None
                and dataset_speakers is not None
                and dataset_speakers > 1
                and checkpoint_speakers != dataset_speakers
            ):
                spk_dim = dataset_speakers
                reset_pretrained = True
            else:
                spk_dim = checkpoint_speakers
    except Exception as e:
        # Rank-gated like every other message here: without it each DDP rank
        # prints the same warning, which reads as several different failures.
        if rank == 0:
            warning(
                f"Could not read the checkpoint ({e}); using the default speaker count.",
                tag="[INIT]",
            )

    if rank == 0:
        info(f"Initializing the generator with {spk_dim} speakers.", tag="[INIT]")
        if reset_pretrained:
            info(
                "The pretrained speaker table describes a different set of "
                "speakers and will not be inherited; this run's embeddings "
                "start fresh.",
                tag="[INIT]",
            )

    return SpeakerLayout(spk_dim, reset_pretrained)
