"""Saving and loading training checkpoints, and the guards that refuse one
built for a different architecture."""

import os
from collections import OrderedDict

import torch

from rvc.lib.terminal import info, warning
from rvc.train.optimizers import restart_schedule_free_average


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


def optimizer_param_names(optimizer, model):
    """Parameter names in the order ``optimizer.state_dict()`` indexes them."""
    names = {id(p): n for n, p in model.named_parameters()}
    return [names.get(id(p)) for g in optimizer.param_groups for p in g["params"]]


def remap_optimizer_state(optimizer, model, opt_state, saved_names=None):
    """Fit a saved optimizer state dict to an optimizer whose parameter set changed
        ( e.g. layers were frozen between runs ).

    Saved state is indexed by the saved optimizer's own parameter list, which
    skips frozen params, so it is matched by ``saved_names`` rather than by
    position in ``model.parameters()``.  Params without a match start fresh.
    Returns None if nothing can be salvaged.
    """
    saved_groups = opt_state.get("param_groups", [])
    if not saved_groups:
        return None

    if saved_names is None:
        # Checkpoints from before the names were saved: a position only means
        # something if the saved optimizer held every parameter.
        model_names = [n for n, _ in model.named_parameters()]
        saved_count = sum(len(g["params"]) for g in saved_groups)
        saved_names = model_names if saved_count == len(model_names) else None

    new_params = [p for g in optimizer.param_groups for p in g["params"]]
    new_index = {n: j for j, n in enumerate(optimizer_param_names(optimizer, model))}

    state = {}
    if saved_names is not None:
        for old_i, group_state in opt_state.get("state", {}).items():
            if not isinstance(old_i, int) or old_i >= len(saved_names):
                continue
            j = new_index.get(saved_names[old_i])
            if j is None:
                continue
            if all(
                v.shape == new_params[j].shape
                for v in group_state.values()
                if torch.is_tensor(v) and v.dim() > 0
            ):
                state[j] = group_state

    # Groups carry the saved hyperparameters, with the LR rebased onto each
    # new group's ``lr_scale`` instead of inheriting the saved group's scale.
    reference = saved_groups[0]
    saved_scale = reference.get("lr_scale", 1.0) or 1.0
    param_groups = []
    for i, group in enumerate(optimizer.param_groups):
        new_group = {
            key: value
            for key, value in saved_groups[min(i, len(saved_groups) - 1)].items()
            if key != "params"
        }
        scale = group.get("lr_scale", 1.0)
        new_group["lr"] = reference["lr"] / saved_scale * scale
        if "initial_lr" in reference:
            new_group["initial_lr"] = reference["initial_lr"] / saved_scale * scale
        if "lr_scale" in group:
            new_group["lr_scale"] = scale
        new_group["params"] = group["params"]
        param_groups.append(new_group)

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
    assert_msd_matches(model_state, checkpoint_dict)
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
            saved_names = checkpoint_dict.get("optimizer_param_names")
            # Same counts do not mean the same params: a stage switch can swap
            # one frozen set for another of equal size and still load cleanly.
            names_changed = saved_names is not None and list(saved_names) != (
                optimizer_param_names(optimizer, model_state)
            )
            try:
                if names_changed:
                    raise ValueError("optimizer parameter names changed")
                optimizer.load_state_dict(opt_state)
                info("Loaded optimizer state.", tag="[RESUME]")
            except ValueError:
                warning(
                    "The optimizer's parameter set changed (layers were frozen, "
                    "for instance); matching the saved state to the surviving "
                    "params by name, with the LR re-anchored to the saved value.",
                    tag="[RESUME]",
                )
                pruned = remap_optimizer_state(
                    optimizer, model_state, opt_state, saved_names
                )
                if pruned is not None:
                    optimizer.load_state_dict(pruned)
                    restart_schedule_free_average(optimizer)
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

    Only ``sine`` is built, but the guard stays: every source this fork tried
    owned different state-dict keys and the generator resumes *non-strictly*,
    so such a checkpoint would load without raising and leave the new modules
    at their random init.
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
            f"{origin} was trained with '{found}'. That source was removed "
            f"and only the sine is built, so such a checkpoint cannot be "
            f"resumed -- start a fresh run."
        )


#: What the shipped configs carried before the decoder layout became
#: configurable.  A checkpoint written before the key exists is one of these,
#: and reading it as "whatever this run builds" would make the guard useless in
#: exactly the case it exists for.
LEGACY_UPSAMPLE_RATES = {32000: [4, 4, 4, 5]}

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


def decoder_layout(model):
    """The decoder arrangement that leaves no trace in the weights.

    Reordering ``upsample_rates`` keeps all 271 tensors' keys *and* shapes
    (channel counts follow the stage index, not the rate), and the upsamplers'
    interpolation kernels are non-persistent, so redesigning one adds no key.
    Either loads silently into a decoder trained for a different signal path.

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
        "source_gain": bool(getattr(decoder, "has_source_gain", False)),
        # ``off`` drops one parameter per ``AdaIN``, which a strict load
        # already refuses, but ``always`` and ``train`` differ only in what
        # inference does -- and nothing in the weights records that.
        "adain_noise": str(getattr(decoder, "adain_noise", "always")),
        "source_bands": int(getattr(decoder, "source_bands", 0)),
        # The excitation's harmonic slope: partial ``j`` at ``j ** -tilt``.
        # The *count* sizes ``m_source.merge.0.weight`` and so a strict load
        # already refuses a mismatch, but the tilt is a non-persistent buffer
        # and changes every partial's level while leaving the state dict
        # byte-identical -- the same contract as ``upsample_filter`` below.
        # The count rides along so the message names both rather than leaving
        # the tilt to explain a shape error.
        "source_harmonics": int(getattr(decoder, "source_harmonics", 0)),
        "source_tilt": float(getattr(decoder, "source_tilt", 1.0)),
        # Imaging, not aliasing: what the interpolation filter leaves of the
        # spectral copies zero-stuffing makes.  It is just as invisible to
        # ``load_state_dict`` as the stage ordering, and with the anti-aliased
        # activations gone it is the last signal-path choice in this decoder
        # that no weight records.
        "upsample_filter": upsample_filter(decoder),
    }


def assert_decoder_layout_matches(model, checkpoint_dict, origin="checkpoint"):
    """Absent means the shipped legacy layout: old ordering, unit-peak source."""

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
            "source_gain": False,
            "source_bands": 0,
            "adain_noise": "train",
            "source_harmonics": 0,
            "source_tilt": 1.0,
            "upsample_filter": None,
        }
    imaging = found.get("upsample_filter") or None
    found = {
        "upsample_rates": [int(r) for r in found.get("upsample_rates", [])],
        "source_gain": bool(found.get("source_gain", False)),
        "source_bands": int(found.get("source_bands", 0)),
        "adain_noise": str(found.get("adain_noise", "train")),
        # Absent means the one-partial sine at the tilt a single partial
        # cannot express: every run before 2026-09-09 is that, and reading
        # the tilt as "whatever this run builds" would let a harmonic-rich
        # checkpoint load into a bare one without a word.
        "source_harmonics": int(found.get("source_harmonics", 0)),
        "source_tilt": float(found.get("source_tilt", 1.0)),
    }
    # Absent means the flat legacy design, sized to the checkpoint's own stage
    # count -- every RefineGAN run ever had these upsamplers, so unlike the
    # anti-aliasing that was removed there is no "had none" case to leave as
    # ``None``.  A decoder that reports no schedule at all is a different
    # vocoder, and then ``expected`` carries ``None`` too.
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
    # A checkpoint written while this decoder still had anti-aliased
    # activations carries their fields.  They describe modules that no longer
    # exist, so they are not compared -- but a checkpoint that used them was
    # trained through a different signal path, and loading it here is a real
    # mismatch even though nothing in this dict can see it.
    stale = [
        key
        for key in ("antialias_stages", "antialias", "antialias_rates")
        if checkpoint_dict.get("decoder_layout", {}).get(key)
    ]
    if found != expected or stale:
        raise ValueError(
            f"Decoder layout mismatch: this run builds {expected} but the "
            f"{origin} was trained with {found}. The stage ordering does not "
            f"appear in any weight, so this is the only thing that can tell "
            f"them apart. Set upsample_rates / refinegan2_source_gain / "
            f"refinegan2_source_harmonics / refinegan2_source_tilt to "
            f"match, or start a fresh run. ``upsample_filter`` is "
            f"[widths, rolloffs, betas] per stage for the trunk's "
            f"interpolation filters and is not a config key: a mismatch there "
            f"means the checkpoint predates the current filter design."
            + (
                f" The {origin} also names {stale}: it was trained with the "
                f"anti-aliased activations this decoder no longer has, so its "
                f"weights were fitted to a different signal path. That cannot "
                f"be configured back -- start a fresh run."
                if stale
                else ""
            )
        )


def discriminator_periods(model):
    """The period set a discriminator was built with, or ``None`` if it has none."""
    periods = getattr(model, "periods", None)
    return None if periods is None else [int(p) for p in periods]


def discriminator_has_msd(model):
    """Whether a discriminator has the waveform (MSD) branch, or ``None`` if it is not one."""
    target = getattr(model, "module", model)
    use_msd = getattr(target, "use_msd", None)
    return None if use_msd is None else bool(use_msd)


def assert_msd_matches(model, checkpoint_dict, origin="checkpoint"):
    """Refuse a discriminator checkpoint built with the other ``d_use_msd``.

    The MSD is branch 0, so toggling it shifts every other branch's index and
    the load fails on unrelated-looking keys.  An absent key means the MSD was
    on: every checkpoint written before the key existed had it.
    """
    expected = discriminator_has_msd(model)
    if expected is None:
        return
    found = checkpoint_dict.get("discriminator_msd")
    found = True if found is None else bool(found)
    if found != expected:
        raise ValueError(
            f"Discriminator MSD mismatch: this run builds d_use_msd={expected} "
            f"but the {origin} was trained with d_use_msd={found}. Set "
            f'"d_use_msd": {str(found).lower()} in the experiment\'s config.json '
            f"to keep using it, or start the discriminator fresh."
        )


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
        from rvc.lib.algorithm.discriminators.discriminator import (
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
    model,
    optimizer,
    learning_rate,
    iteration,
    checkpoint_path,
    ema=None,
    extra=None,
    prior_noise_subspace=None,
):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    model_instance = model.module if hasattr(model, "module") else model

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        # What the optimizer state's integer indices refer to, so a resume
        # with a different frozen set can match moments by name.
        "optimizer_param_names": optimizer_param_names(optimizer, model_instance),
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
    has_msd = discriminator_has_msd(model_instance)
    if has_msd is not None:
        checkpoint_data["discriminator_msd"] = has_msd
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
    if prior_noise_subspace is not None:
        checkpoint_data["prior_noise_subspace"] = prior_noise_subspace

    torch.save(checkpoint_data, checkpoint_path)
    info(f"Saved '{os.path.basename(checkpoint_path)}'.", tag="[SAVE]")
