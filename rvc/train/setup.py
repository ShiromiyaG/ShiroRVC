"""Assembling a run: build the models, place them, compile them, and build the
optimizers that will move them.

``train.py`` owns the loop; this owns everything that happens once, before it.
The tuning knobs these used to read off ``train.py``'s module globals
(``freeze_vae``, ``dec_lr_scale``, ``resume_lr`` and so on) are arguments here,
so each function states what it actually depends on.
"""

import os
import re

import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from rvc.configs.vocoders import get_discriminator_id, normalize_vocoder
from rvc.lib.terminal import info
from rvc.train.messages import (
    DISCRIMINATOR_COMPILE_ENABLED,
    DISCRIMINATOR_COMPILE_NO_CUDA,
    DISCRIMINATOR_COMPILE_NOT_SUPPORTED,
    FRONTEND_COMPILE_ENABLED,
    FRONTEND_COMPILE_NO_CUDA,
    FRONTEND_COMPILE_NOT_SUPPORTED,
    VOCODER_COMPILE_ENABLED,
    VOCODER_COMPILE_NO_CUDA,
    VOCODER_COMPILE_NOT_SUPPORTED,
)
from rvc.train.optimizers import _make_optimizer


def _inductor_cache_dir():
    """One Inductor cache for the whole fork, under ``logs/``.

    Kept out of the per-model folder on purpose: the compiled artefacts are a
    function of the code and the GPU, not of the experiment, so every run
    should be warming the same cache.
    """
    cache_dir = os.path.join(os.getcwd(), "logs", ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    return cache_dir


def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    model_config = config.model.__dict__.copy()
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **model_config,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
    )

def get_d_model(config, vocoder, use_checkpointing):
    vocoder = normalize_vocoder(vocoder)
    discriminator_id = get_discriminator_id(vocoder)
    # JSON has no tuples, so a configured schedule arrives as nested lists.  The
    # discriminators normalise them themselves; passing them through untouched
    # keeps this function free of the branch layout.
    def setting(name, default=None):
        """``None``/absent means "use the default"; ``[]`` means "none of these".

        This was ``value or default``, which collapsed the two: a config asking
        for *no* period branches got the full set back, and there was no way to
        turn a whole family off from JSON.
        """

        value = getattr(config.model, name, None)
        return default if value is None else value

    from rvc.lib.algorithm.discriminators.multi import (
        DISCRIMINATOR_VERSIONS,
        MPD_MSD_Combined,
    )

    # The registry names the branch layout directly -- ``v2`` is Applio's (8
    # periods), ``v3`` is what it picks for RefineGAN (5 periods + 3
    # multi-resolution spectrogram branches) -- so this only has to check that
    # the name exists.  Checked rather than passed through: it used to be
    # ``"v3" if id == "mpd_msd_v3" else "v2"``, which handed Applio's v2 to any
    # id it did not recognise, and a vocoder registered against the wrong name
    # would have trained against the wrong discriminator with nothing said.
    #
    # ``d_version`` overrides it: v3 does not fit an 8 GB card at batch 8
    # (6.42 GiB / 5912 ms/step against v2's 4.64 GiB / 498 ms/step), and the
    # RefineGAN2 config names ``v4`` -- v3 minus its longest period branch.
    # The registry stays at ``v3`` so a config predating that key builds what
    # it always did.
    if discriminator_id not in DISCRIMINATOR_VERSIONS:
        raise ValueError(
            f"Unknown discriminator {discriminator_id!r} for vocoder "
            f"{vocoder!r}; known: {sorted(DISCRIMINATOR_VERSIONS)}."
        )
    version = str(getattr(config.model, "d_version", None) or discriminator_id)
    # ``d_use_*`` switches a whole family off; ``d_periods``/``d_resolutions``
    # replace its content when it's on (``None`` keeps the preset's).
    return MPD_MSD_Combined(
        config.model.use_spectral_norm,
        use_checkpointing=use_checkpointing,
        version=version,
        periods=[] if not setting("d_use_periods", True) else setting("d_periods"),
        resolutions=(
            [] if not setting("d_use_resolutions", True) else setting("d_resolutions")
        ),
        frequency_strides=setting("d_frequency_strides"),
        use_msd=bool(setting("d_use_msd", True)),
        # Opt-in, so an absent key builds the branches every existing
        # discriminator was trained with.
        use_fast_mpd=bool(setting("d_use_fast_mpd", False)),
        # On by default: the spectrogram branches' STFT and first conv run
        # outside FP16 autocast, where their unnormalised magnitude overflowed
        # at a raised learning rate.  ``false`` restores the all-FP16 path.
        mrd_fp32_input=bool(setting("d_mrd_fp32_input", True)),
        # UnivHD (arXiv 2512.03486) is opt-in and *additive*: it appends a
        # harmonic-order branch and removes nothing, which is how the paper
        # runs it.  Off by default because it is unmeasured on this fork -- the
        # gains it reports are on harmonic structure and F0RMSE for singing,
        # not on the frame-rate mirroring, which its ERB bandwidths are too
        # wide to separate above ~700 Hz.
        sample_rate=int(config.data.sample_rate),
        use_univhd=bool(setting("d_use_univhd", False)),
        # SAN (arXiv 2301.12811): splits every branch's last projection into a
        # unit-norm direction and a scale.  Off by default and unmeasured on
        # this fork -- it changes conv_post's state-dict keys, so it is a
        # fresh-run decision, and it changes the loss floor, so every number
        # read off loss_disc has to be relearned.
        use_san=bool(setting("d_use_san", False)),
        univhd_n_fft=int(setting("d_univhd_n_fft", 2048)),
        univhd_hop_length=int(setting("d_univhd_hop_length", 256)),
        univhd_harmonics=int(setting("d_univhd_harmonics", 10)),
        univhd_bins_per_octave=int(setting("d_univhd_bins_per_octave", 24)),
        univhd_f_min=float(setting("d_univhd_f_min", 80.0)),
        univhd_channels=int(setting("d_univhd_channels", 32)),
        univhd_half_harmonic=bool(setting("d_univhd_half_harmonic", True)),
        # ``None`` keeps the version's pinned weight; a config key overrides it
        # on any version.  See ``UNIVHD_WEIGHT_BY_VERSION``.
        univhd_weight=(
            None
            if setting("d_univhd_weight", None) is None
            else float(setting("d_univhd_weight", None))
        ),
    )


def setup_models_for_training(net_g, net_d, device, device_id, n_gpus):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)
    net_d = net_d.to(device_id) if device.type == "cuda" else net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(
            net_d,
            device_ids=[device_id],
            find_unused_parameters=bool(
                getattr(net_d, "uses_branchwise_r1", False)
            ),
        )

    return net_g, net_d


def enable_vocoder_compile(net_g, device, rank, enabled=False, mode="default"):
    """Compile the decoder, driven by the run spec's ``compile_vocoder``.

    Unlike the frontend and the discriminator, this one honours the run's
    ``torch_compile_mode``: the decoder sees one fixed segment length per step,
    which is what ``reduce-overhead``'s CUDA graphs need.
    """
    if not enabled:
        return False
    if device.type != "cuda":
        if rank == 0:
            info(VOCODER_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    _inductor_cache_dir()

    model = net_g.module if hasattr(net_g, "module") else net_g
    enabled = model.enable_decoder_compile(mode=mode)
    if not enabled and rank == 0:
        info(VOCODER_COMPILE_NOT_SUPPORTED, tag="[INIT]")
    if enabled and rank == 0:
        info(VOCODER_COMPILE_ENABLED.format(mode=mode), tag="[INIT]")
    return enabled


def normalize_san_weights(net_d):
    """Put every SAN direction back on the unit sphere after ``optim_d.step``.

    The projection is only unit-norm because it is *kept* there; an Adam step
    moves it off, and a direction that is not a direction is the one thing the
    method cannot tolerate.  A no-op when no branch carries a SAN head.
    """

    model = net_d.module if hasattr(net_d, "module") else net_d
    for module in model.modules():
        normalize = getattr(module, "normalize_weight", None)
        if normalize is not None:
            normalize()


def enable_frontend_compile(net_g, config, device, rank):
    """Compile the prior/posterior/flow, driven by ``compile_frontend``.

    In the config rather than the run spec, for the same reason as
    ``compile_discriminator``: what is compiled travels with the architecture,
    not with the button the run was started from.  ``torch_compile_mode`` is not
    consulted here either -- ``reduce-overhead`` records CUDA graphs, and these
    modules see a different length almost every batch.
    """

    if not bool(getattr(config.train, "compile_frontend", False)):
        return False
    if device.type != "cuda":
        if rank == 0:
            info(FRONTEND_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    _inductor_cache_dir()

    model = net_g.module if hasattr(net_g, "module") else net_g
    enable = getattr(model, "enable_frontend_compile", None)
    if enable is None:
        if rank == 0:
            info(FRONTEND_COMPILE_NOT_SUPPORTED, tag="[INIT]")
        return False
    enabled = enable(mode="default")
    if not enabled and rank == 0:
        info(FRONTEND_COMPILE_NOT_SUPPORTED, tag="[INIT]")
    if enabled and rank == 0:
        info(FRONTEND_COMPILE_ENABLED.format(mode="default"), tag="[INIT]")
    return enabled


def enable_discriminator_compile(net_d, config, device, rank):
    """Compile the discriminator, driven by ``compile_discriminator`` in the
    config (not a run-spec flag, since it travels with the architecture).
    ``torch_compile_mode`` is not consulted: ``reduce-overhead`` records CUDA
    graphs, which this loop can't support, so the mode is fixed to plain
    fusion.
    """
    if not bool(getattr(config.train, "compile_discriminator", False)):
        return False
    if device.type != "cuda":
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    _inductor_cache_dir()

    model = net_d.module if hasattr(net_d, "module") else net_d
    enable = getattr(model, "enable_compile", None)
    if enable is None:
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NOT_SUPPORTED, tag="[INIT]")
        return False
    enabled = enable(mode="default")
    if enabled and rank == 0:
        info(DISCRIMINATOR_COMPILE_ENABLED.format(mode="default"), tag="[INIT]")
    return enabled


def assert_resumable_architecture(net_g, checkpoint_path):
    """Refuse to resume from a checkpoint built for a different architecture.

    Unlike the pretrained-path guard, a missing id here counts as a mismatch:
    a resumed run predating the id is one of this fork's own old runs, whose
    layout has since changed.
    """
    # Imported here, not at module scope: ``rvc.train.utils`` reaches
    # ``mel_processing`` by the trainer's script-relative path, so importing it
    # eagerly would make this module unimportable outside a training run.
    from rvc.train.utils import (
        assert_decoder_layout_matches,
        assert_excitation_matches,
    )

    model = net_g.module if hasattr(net_g, "module") else net_g
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The excitation is checked whatever the id says, because the id is pinned
    # to ``vits_gaussian_v1`` for every RefineGAN source on purpose -- see
    # ``get_architecture_id`` -- and this stack resumes non-strictly.
    assert_excitation_matches(model, checkpoint, origin="checkpoint")
    # Same reasoning, and the same door: the stage ordering and the
    # anti-aliased activations leave no trace in any weight either.
    assert_decoder_layout_matches(model, checkpoint, origin="checkpoint")
    expected = getattr(model, "architecture_id", None)
    if not expected or expected == "vits_gaussian_v1":
        return
    found = checkpoint.get("architecture_id")
    if found == expected:
        return
    raise ValueError(
        f"Cannot resume from '{os.path.basename(checkpoint_path)}': it was "
        f"written for architecture '{found or 'unknown'}', but this run builds "
        f"'{expected}'. Move the old checkpoints out of the experiment folder "
        f"to start this architecture fresh."
    )


def checkpoint_step_from_path(path):
    if not path or path in ("", "None"):
        return 0
    match = re.search(r"(?:^|[\\/])G_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else 0


def build_decoder_param_groups(net_g, base_lr, dec_lr_scale=None, vae_lr_scale=None):
    """
    Build optimizer param groups with differential LR.

    `dec_lr_scale` covers the decoder/vocoder, `vae_lr_scale` everything else
    ( the frontend and the speaker embedding ).  Returns None when neither is
    configured, so the optimizer falls back to a single group.
    """
    model = net_g.module if hasattr(net_g, "module") else net_g
    dec = getattr(model, "dec", None)
    if dec is None:
        return None

    if dec_lr_scale is None and vae_lr_scale is None:
        return None

    # scale -> [params]  ( same scale == same effective LR )
    groups = {}

    def add(param, scale):
        groups.setdefault(scale, []).append(param)

    decoder_scale = dec_lr_scale if dec_lr_scale is not None else 1.0
    rest_scale = vae_lr_scale if vae_lr_scale is not None else 1.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            add(param, decoder_scale if name.startswith("dec.") else rest_scale)

    return [
        {"params": params, "lr": base_lr * scale, "lr_scale": scale}
        for scale, params in groups.items()
    ]


def get_optimizers(
    net_g,
    net_d,
    config,
    optimizer_choice_g,
    optimizer_choice_d,
    custom_lr_g,
    custom_lr_d,
    use_custom_lr,
    total_epoch_count,
    train_loader,
    dec_lr_scale=None,
    vae_lr_scale=None,
):
    lr_g = custom_lr_g if use_custom_lr else config.train.learning_rate_g
    lr_d = custom_lr_d if use_custom_lr else config.train.learning_rate_d
    num_batches = len(train_loader)

    g_param_groups = build_decoder_param_groups(
        net_g, lr_g, dec_lr_scale=dec_lr_scale, vae_lr_scale=vae_lr_scale
    )

    optim_g = _make_optimizer(net_g, optimizer_choice_g, lr_g, num_epochs=total_epoch_count, num_batches=num_batches, param_groups=g_param_groups)
    optim_d = _make_optimizer(
        net_d,
        optimizer_choice_d,
        lr_d,
        num_epochs=total_epoch_count,
        num_batches=num_batches,
        lazy_reg_interval=None,
    )

    return optim_g, optim_d


#: What each staged-pretrain mode leaves trainable, by parameter-name prefix.
#:
#: The decoder does not read a mel spectrogram -- it reads ``z``, which comes
#: out of the *posterior* ``enc_q(spec)``.  So the vocoder-only stage is not
#: ``dec`` on its own: it is the spectral autoencoder ``enc_q -> dec``, with
#: the prior side of the model held still.  Freezing ``enc_q`` alongside the
#: frontend (which is what ``"frontend"`` does) is a fine-tune mode, not a
#: from-scratch one: it hands the decoder a ``z`` from whatever encoder the
#: checkpoint already had, and from a random init that is noise.
#:
#: A frozen module still passes gradient *through* itself, so freezing ``dec``
#: in stage 2 leaves the spectral and adversarial terms training the frontend
#: through the decoder rather than making them inert.
FREEZE_MODES = {
    # Nothing held.  End-to-end, the way a normal run trains.
    "none": None,
    # Fine-tune the vocoder onto an existing frontend: everything outside
    # ``dec``/``emb_g`` is held.  This is the old ``freeze_vae`` flag.
    "frontend": ("enc_p.", "enc_q.", "flow."),
    # Stage 1 -- spectral autoencoder.  ``enc_q`` + ``dec`` + ``emb_g`` learn
    # to reconstruct; the prior and the flow are held so the KL cannot drag
    # the latent around while the decoder is being judged on reconstruction.
    "vocoder": ("enc_p.", "flow."),
    # Stage 2 -- the encoders.  The decoder is held at the weights stage 1
    # validated, and ``enc_p``/``flow``/``enc_q`` learn to feed it.
    "encoders": ("dec.",),
}


def apply_frontend_freeze(net_g, rank, freeze_vae=False, freeze_mode="none"):
    """Hold part of the generator for a staged pretrain.

    ``freeze_mode`` names the stage (see ``FREEZE_MODES``).  ``freeze_vae`` is
    the older boolean spelling of ``freeze_mode="frontend"`` and still wins
    when set, so a run configured the old way keeps its meaning.

    Must run *before* ``get_optimizers``: ``build_decoder_param_groups`` skips
    parameters whose ``requires_grad`` is already off, which is what keeps
    frozen weights out of the optimizer's state entirely rather than merely
    unupdated.
    """
    if freeze_vae:
        freeze_mode = "frontend"

    if freeze_mode not in FREEZE_MODES:
        raise ValueError(
            f"Unknown freeze_mode {freeze_mode!r}; expected one of "
            f"{sorted(FREEZE_MODES)}."
        )

    prefixes = FREEZE_MODES[freeze_mode]
    if prefixes is None:
        if rank == 0:
            info("Freeze: nothing frozen.", tag="[INIT]")
        return

    model = net_g.module if hasattr(net_g, "module") else net_g
    frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith(prefixes) and param.requires_grad:
            param.requires_grad = False
            frozen_params += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        info(
            f"Freeze '{freeze_mode}': held {', '.join(prefixes)} "
            f"({frozen_params:,} params); {trainable:,} still training.",
            tag="[INIT]",
        )


def apply_training_freezes(net_g, rank, freeze_vae=False, freeze_mode="none"):
    apply_frontend_freeze(
        net_g, rank, freeze_vae=freeze_vae, freeze_mode=freeze_mode
    )


def apply_resume_lr_override(optim_g, optim_d=None, resume_lr=None, resume_lr_target="full"):
    """Re-anchor G and/or D param groups to `resume_lr` after loading optimizer
    state. Must run after load_checkpoint (needs the saved lr/initial_lr) and
    before prepare_schedulers (which snapshots the new initial_lr). Each
    group's saved decay ratio is preserved so the schedule keeps its position
    while the value restarts at resume_lr x its per-group lr_scale.
    """
    if resume_lr is None:
        return

    targets = {"full": ("g", "d"), "g": ("g",), "d": ("d",)}.get(resume_lr_target, ("g", "d"))
    optims = [(optim_g, "G")] if "g" in targets else []
    if "d" in targets and optim_d is not None:
        optims.append((optim_d, "D"))

    for optim, label in optims:
        for param_group in optim.param_groups:
            saved_lr = param_group.get("lr", 0.0)
            saved_initial = param_group.get("initial_lr", 0.0)
            decay = (saved_lr / saved_initial) if saved_initial else 1.0
            scale = param_group.get("lr_scale", 1.0)
            new_lr = resume_lr * scale * param_group.get("lazy_reg_scale", 1.0)
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr / decay if decay else new_lr

    parts = []
    for optim, label in optims:
        lrs = ", ".join("{:.2e}".format(g["lr"]) for g in optim.param_groups)
        parts.append(f"{label}: {lrs}")
    info(f"Resume LR override: base {resume_lr:.2e} -> " + " | ".join(parts), tag="[OVERRIDE]")
