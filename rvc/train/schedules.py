"""Step-denominated schedules, sized against the run that will execute them.

Two separate jobs: fitting literals written for a pretrain into whatever run
is actually starting (:func:`fit_schedule` and friends), and building the LR
schedulers themselves.
"""

import math

import torch

from rvc.lib.terminal import warning
from rvc.train.optimizers import is_schedule_free


def planned_step_count(total_epoch_count: int, train_loader, max_steps: int = 0) -> int:
    """How many optimizer steps this run will take.

    Every schedule below this line is step-denominated but stated as an
    absolute literal, so it means something different on an 8k fine-tune than
    on a hundred-thousand-step pretrain; this is what :func:`fit_schedule` and
    :func:`fit_eval_interval` rescale against.  Returns 0 when the budget is
    unknown, which every caller treats as "leave the configured value alone".
    """
    per_epoch = max(1, len(train_loader))
    planned = max(0, int(total_epoch_count)) * per_epoch
    limit = max(0, int(max_steps))
    if limit:
        planned = min(planned, limit) if planned else limit
    return planned


def fit_schedule(configured: int, planned_steps: int, fraction: float, minimum: int = 1) -> int:
    """Shrink a step-denominated schedule to fit inside the run.

    Only ever shrinks: a run long enough for the configured value gets it back
    untouched.  ``fraction`` is the share of the run the schedule may occupy.
    """
    configured = max(0, int(configured))
    if planned_steps <= 0 or configured <= 0:
        return configured
    return max(minimum, min(configured, int(planned_steps * fraction)))


def fit_eval_interval(configured: int, planned_steps: int, patience: int) -> int:
    """An evaluation interval that lets ``patience`` actually be reached.

    At one evaluation per 2000 steps, an 8k fine-tune only gets 4 evaluations
    against a patience of 8, so the detector can never fire. Sizing for ~3x
    patience keeps it a detector; only ever shrinks, like :func:`fit_schedule`.
    """
    configured = max(1, int(configured))
    if planned_steps <= 0:
        return configured
    wanted = planned_steps // max(1, 3 * max(1, int(patience)))
    return max(1, min(configured, wanted)) if wanted else configured


def prepare_schedulers(
    optim_g, optim_d,
    use_lr_scheduler, lr_scheduler, exp_decay_gamma,
    total_epoch_count, epoch_str, global_step, train_loader,
    fresh_start=False,
    optimizer_choice_g="AdamW",
    optimizer_choice_d="AdamW",
    lr_final_ratio=None,
    exp_decay_step_raw=False,
):
    def _horizon_decay(final_ratio, total_units):
        """Exponential decay reaching ``final_ratio`` at the end of the run,
        so the same config gives the same endpoint at any run length. Progress
        is clamped at 1.0 so a run extended past its horizon holds the final
        LR instead of decaying straight through it.
        """
        ratio = min(1.0, max(1e-6, float(final_ratio)))
        total = max(1, int(total_units))

        def scale(unit):
            return ratio ** min(1.0, max(0, unit) / total)

        return scale

    def _horizon_cosine(final_ratio, total_units):
        """Cosine anneal from the starting LR to ``final_ratio`` of it.

        Unlike stock ``CosineAnnealingLR``'s shared absolute ``eta_min``,
        the endpoint here is a fraction of each group's own base LR -- G and D
        start at different rates, and a shared floor would drive their ratio
        to 1.0 by the end of the run, which is what keeps the discriminator
        alive. Clamped past the horizon like ``_horizon_decay``.
        """
        ratio = min(1.0, max(1e-6, float(final_ratio)))
        total = max(1, int(total_units))

        def scale(unit):
            progress = min(1.0, max(0, unit) / total)
            return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return scale

    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    scheduler_resume_epoch = -1 if fresh_start else epoch_str - 1
    scheduler_resume_step = -1 if fresh_start else global_step - 1

    for param_group in optim_g.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    for param_group in optim_d.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler and (
        is_schedule_free(optimizer_choice_g) or is_schedule_free(optimizer_choice_d)
    ):
        # Not an error -- the optimizer survives it, because its averaging
        # weights key off ``lr_max`` rather than the current lr -- but a decay
        # schedule is the thing schedule-free exists to remove, so a run using
        # both is almost certainly a leftover setting.
        warning(
            f"'{lr_scheduler}' is set together with a schedule-free optimizer, "
            "which is designed to run without one. The schedule will still be "
            "applied; set the scheduler to 'none' if that was not intended.",
            tag="[INIT]",
        )

    if use_lr_scheduler:
        scheduler_name = (
            "cosine annealing epoch"
            if lr_scheduler == "cosine annealing"
            else lr_scheduler
        )

        horizon_shapes = {
            "exp decay epoch": _horizon_decay,
            "exp decay step": _horizon_decay,
            "cosine annealing epoch": _horizon_cosine,
        }
        if lr_final_ratio is not None and scheduler_name in horizon_shapes:
            # Only one variant is stepped per optimizer step; the others are
            # stepped per epoch, so the ratio has to land at the end of the run
            # in whichever unit this scheduler counts.
            per_epoch = scheduler_name != "exp decay step"
            total_units = total_epoch_count * (1 if per_epoch else num_batches_per_epoch)
            shape = horizon_shapes[scheduler_name](lr_final_ratio, total_units)
            resume_at = scheduler_resume_epoch if per_epoch else scheduler_resume_step
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(
                optim_g, shape, last_epoch=resume_at
            )
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(
                optim_d, shape, last_epoch=resume_at
            )
        elif scheduler_name == "exp decay epoch":
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch
            )
        elif scheduler_name == "exp decay step":
            scheduler_gamma = (
                exp_decay_gamma
                if exp_decay_step_raw
                else exp_decay_gamma ** (1.0 / num_batches_per_epoch)
            )
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g, gamma=scheduler_gamma, last_epoch=scheduler_resume_step
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, gamma=scheduler_gamma, last_epoch=scheduler_resume_step
            )
        elif scheduler_name == "cosine annealing epoch":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_g, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch
            )
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_d, T_max=total_epoch_count, eta_min=3e-5, last_epoch=scheduler_resume_epoch
            )

    return scheduler_g, scheduler_d
