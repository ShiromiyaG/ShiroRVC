"""Optimizer construction and the schedule-free evaluation contract.

Deliberately importable on its own.  ``rvc.train.train`` parses ``sys.argv`` and
loads a run's config at import time, so anything living there can only be
exercised by starting a real training run -- which is a poor place to keep the
one function whose job is to be correct for four different update rules.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from rvc.lib.terminal import warning


#: Every optimizer the interfaces offer.  Kept here rather than in the UI layer
#: so ``core.py``, both GUIs and the trainer cannot drift apart.
OPTIMIZER_CHOICES = ("AdamW", "Sched-Free AdamW", "Muon", "Lion")

#: The choices whose live weights are *not* the weights to evaluate or ship.
#: Anything reading the generator for a purpose other than the next training
#: step has to go through ``averaged_weights`` for these.
SCHEDULE_FREE_CHOICES = ("Sched-Free AdamW",)

#: Shared across optimizers on purpose: switching optimizer should change the
#: update rule, not silently change how much momentum the run carries.
BASE_BETAS = (0.8, 0.99)
BASE_WEIGHT_DECAY = 0.01


def is_schedule_free(choice) -> bool:
    return choice in SCHEDULE_FREE_CHOICES


@contextmanager
def averaged_weights(*pairs):
    """Hold schedule-free optimizers at their evaluation iterate.

    Schedule-free keeps the parameters at the extrapolated ``y`` point during
    training; the sequence the method actually converges is the averaged ``x``,
    and only ``optimizer.eval()`` moves the weights there.  So every read of the
    weights that is not "compute the next gradient" -- holdout scoring, preview
    rendering, checkpointing, export -- has to happen inside this block, or it
    measures and ships a point the method never claimed was good.

    Takes ``(choice, optimizer)`` pairs and ignores the ones that are not
    schedule-free, so call sites do not have to test first.
    """
    switched = [
        optimizer
        for choice, optimizer in pairs
        if optimizer is not None and is_schedule_free(choice)
    ]
    for optimizer in switched:
        optimizer.eval()
    try:
        yield
    finally:
        for optimizer in switched:
            optimizer.train()


def _embedding_parameters(model):
    """Parameters that live in an ``nn.Embedding``, by identity.

    Muon orthogonalises the rows of a matrix, which for an embedding table
    would flatten exactly the per-entry magnitude the table exists to learn --
    here, the speaker identity in ``emb_g``.
    """
    module = getattr(model, "module", model)
    found = []
    for child in module.modules():
        if isinstance(child, torch.nn.Embedding):
            found.extend(child.parameters(recurse=False))
    return found


def _make_optimizer(
    model,
    choice,
    lr,
    num_epochs=None,
    num_batches=None,
    param_groups=None,
    lazy_reg_interval=None,
):
    params = param_groups if param_groups is not None else list(
        filter(lambda p: p.requires_grad, model.parameters())
    )

    if choice not in OPTIMIZER_CHOICES:
        # Reachable from a saved preset or a stored GUI preference naming an
        # optimizer that has since been removed.  Falling back beats aborting a
        # run that is otherwise fully configured.
        warning(
            f"Unknown optimizer {choice!r}; using AdamW. "
            f"Available: {', '.join(OPTIMIZER_CHOICES)}.",
            tag="[INIT]",
        )
        choice = "AdamW"

    lazy_scale = 1.0
    if lazy_reg_interval is not None:
        interval = max(1, int(lazy_reg_interval))
        lazy_scale = interval / (interval + 1.0)
    lr = lr * lazy_scale

    def lazy_betas(betas):
        return tuple(float(beta) ** lazy_scale for beta in betas)

    if choice == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=lazy_betas(BASE_BETAS),
            eps=1e-9,
            weight_decay=BASE_WEIGHT_DECAY,
            fused=torch.cuda.is_available(),
        )

    elif choice == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree

        # ``warmup_steps`` stays 0 because this repository runs its own linear
        # warmup over ``group['lr']``.  That is compatible: the averaging weight
        # is ``lr_max ** weight_lr_power`` and ``lr_max`` is a running maximum,
        # so an external ramp down-weights the warmup steps exactly as the
        # built-in one would.
        optimizer = AdamWScheduleFree(
            params,
            lr=lr,
            betas=lazy_betas(BASE_BETAS),
            eps=1e-9,
            weight_decay=BASE_WEIGHT_DECAY,
            warmup_steps=0,
        )

    elif choice == "Muon":
        from rvc.train.custom_optimizers.muon import Muon

        # The update is RMS-matched to AdamW inside the optimizer, so it shares
        # this run's learning rate instead of needing one of its own.
        optimizer = Muon(
            params,
            lr=lr,
            momentum=0.95,
            nesterov=True,
            betas=lazy_betas(BASE_BETAS),
            eps=1e-9,
            weight_decay=BASE_WEIGHT_DECAY,
            adamw_params=_embedding_parameters(model),
        )

    elif choice == "Lion":
        from rvc.train.custom_optimizers.lion import (
            LR_SCALE,
            WEIGHT_DECAY_SCALE,
            Lion,
        )

        # Lion's step size is a sign, so it carries no gradient scale: it needs
        # a smaller learning rate than AdamW and a proportionally larger decay.
        # Applied here rather than asking the user to re-tune ``learning_rate_g``
        # when they switch optimizer.
        optimizer = Lion(
            params,
            lr=lr * LR_SCALE,
            betas=(0.9, 0.99),
            weight_decay=BASE_WEIGHT_DECAY * WEIGHT_DECAY_SCALE,
        )
        # The per-group ``lr`` set from ``param_groups`` bypasses the ``lr=``
        # argument above, so the differential decoder/VAE rates need scaling too.
        if param_groups is not None:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * LR_SCALE

    for group in optimizer.param_groups:
        group["lazy_reg_scale"] = lazy_scale
    return optimizer
