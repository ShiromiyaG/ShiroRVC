"""Discriminator learning rate driven by how well each family of heads separates."""

import math
from contextlib import contextmanager

import torch
import torch.distributed as dist


def head_family(label):
    """The family a ``MPD_MSD_Combined.branch_labels`` entry belongs to."""
    if label.startswith("period_"):
        return "mpd"
    if label.startswith("resolution_"):
        return "mrd"
    return label


def discriminator_param_groups(net_d, lr):
    """One optimizer param group per head family, tagged with ``family``."""
    model = getattr(net_d, "module", net_d)
    groups = {}
    for label, branch in zip(model.branch_labels, model.discriminators):
        params = [p for p in branch.parameters() if p.requires_grad]
        groups.setdefault(head_family(label), []).extend(params)
    covered = {id(p) for params in groups.values() for p in params}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if rest:
        groups["other"] = rest
    return [
        {"params": params, "lr": lr, "family": family}
        for family, params in groups.items()
    ]


def head_accuracies(real_outputs, fake_outputs):
    """Per-head fraction of logits on the right side of 0.5 -- the equilibrium
    of both the LSGAN and the SAN softplus losses.

    0.5 is chance (the generator fools the head), 1.0 is a head that is never
    wrong.  Returns a detached ``(heads,)`` device tensor; under SAN the
    function output is the one read.
    """
    accuracies = []
    for dr, dg in zip(real_outputs, fake_outputs):
        if isinstance(dr, (list, tuple)):
            dr, dg = dr[0], dg[0]
        dr, dg = dr.detach().float(), dg.detach().float()
        accuracies.append(
            0.5 * ((dr > 0.5).float().mean() + (dg < 0.5).float().mean())
        )
    return torch.stack(accuracies)


class FamilyReducer:
    """Reduces per-head accuracies to one weighted mean per family.

    Built once per epoch so the step only pays one matrix product.
    """

    def __init__(self, labels, weights, device):
        self.names = []
        for label in labels:
            family = head_family(label)
            if family not in self.names:
                self.names.append(family)
        weights = [1.0] * len(labels) if weights is None else [float(w) for w in weights]
        matrix = torch.zeros(len(self.names), len(labels))
        for index, (label, weight) in enumerate(zip(labels, weights)):
            matrix[self.names.index(head_family(label)), index] = weight
        self.matrix = (matrix / matrix.sum(dim=1, keepdim=True)).to(device)

    def __call__(self, accuracies):
        return self.matrix @ accuracies


class DiscriminatorLRBalancer:
    """Multiplies each head family's learning rate so its accuracy tracks ``target``.

    Below the target (the family is being fooled) its ratio grows, above it
    (the family is winning) it shrinks, always within ``[min_ratio, max_ratio]``.
    Param groups are matched by their ``family`` key; a group without one gets
    the geometric mean of the family ratios.

    Args:
        target: accuracy to hold, between 0.5 and 1.
        min_ratio, max_ratio: bounds on each multiplier.
        speed: steps a steady accuracy error of 0.1 takes to double or halve
            a ratio.
        interval: steps between updates; a multiple of ``sample_every``.
        sample_every: steps between accuracy samples.
        ema_decay: per-step decay of the smoothed accuracies.
    """

    def __init__(
        self,
        target=0.6,
        min_ratio=0.5,
        max_ratio=2.0,
        speed=5000,
        interval=8,
        sample_every=4,
        ema_decay=0.99,
    ):
        if not 0.5 < float(target) < 1.0:
            raise ValueError(f"d_lr_balance_target must be in (0.5, 1); got {target}.")
        if not 0.0 < float(min_ratio) <= 1.0 <= float(max_ratio):
            raise ValueError(
                "d_lr_balance_min must be in (0, 1] and d_lr_balance_max >= 1; "
                f"got {min_ratio} and {max_ratio}."
            )
        self.target = float(target)
        self.log_min = math.log(float(min_ratio))
        self.log_max = math.log(float(max_ratio))
        self.gain = math.log(2.0) / (0.1 * max(1.0, float(speed)))
        self.sample_every = max(1, int(sample_every))
        self.interval = max(1, int(interval) // self.sample_every) * self.sample_every
        self.ema_decay = float(ema_decay)
        self.names = []
        self.log_ratios = {}
        self.accuracies = {}
        self._sum = None
        self._count = 0
        self._pending = None
        self._pending_names = None
        self._pending_event = None

    def ratio(self, family):
        return math.exp(self.log_ratios.get(family, 0.0))

    def _default_ratio(self):
        if not self.log_ratios:
            return 1.0
        return math.exp(sum(self.log_ratios.values()) / len(self.log_ratios))

    def observe(self, family_accuracies, names, step):
        """Feed a sampled step's ``(families,)`` accuracies, in ``names`` order.

        The ratios move once per ``interval``.  The window mean reaches the
        host through a non-blocking copy and is applied at the next update, so
        the loop never waits on the GPU for it.
        """
        if list(names) != self.names:
            self.names = list(names)
            self._sum, self._count = None, 0
        self._sum = family_accuracies if self._sum is None else self._sum + family_accuracies
        self._count += 1
        if step % self.interval != 0:
            return
        self._apply_pending()
        mean = self._sum / self._count
        self._sum, self._count = None, 0
        # Every rank must step D with the same learning rates.
        if dist.is_available() and dist.is_initialized():
            mean = mean.clone()
            dist.all_reduce(mean)
            mean /= dist.get_world_size()
        self._pending_names = list(self.names)
        if mean.device.type == "cuda":
            self._pending = mean.to("cpu", non_blocking=True)
            self._pending_event = torch.cuda.Event()
            self._pending_event.record()
        else:
            self._pending, self._pending_event = mean, None

    def _apply_pending(self):
        if self._pending is None:
            return
        if self._pending_event is not None and not self._pending_event.query():
            # Still in flight after a whole interval: drop it for the fresh one.
            self._pending = None
            return
        values = self._pending.tolist()
        self._pending = None
        decay = self.ema_decay ** self.interval
        for family, value in zip(self._pending_names, values):
            previous = self.accuracies.get(family)
            accuracy = value if previous is None else decay * previous + (1.0 - decay) * value
            self.accuracies[family] = accuracy
            log_ratio = self.log_ratios.get(family, 0.0)
            log_ratio += self.gain * self.interval * (self.target - accuracy)
            self.log_ratios[family] = min(self.log_max, max(self.log_min, log_ratio))

    @contextmanager
    def applied(self, optimizer):
        """Scale each param group's lr for the duration of ``optimizer.step``.

        Restored afterwards so the scheduler and warmup never see the ratios.
        """
        saved = [group["lr"] for group in optimizer.param_groups]
        fallback = self._default_ratio()
        for group in optimizer.param_groups:
            family = group.get("family")
            ratio = self.ratio(family) if family in self.log_ratios else fallback
            group["lr"] = group["lr"] * ratio
        try:
            yield
        finally:
            for group, lr in zip(optimizer.param_groups, saved):
                group["lr"] = lr

    def state_dict(self):
        return {
            "log_ratios": dict(self.log_ratios),
            "accuracies": dict(self.accuracies),
        }

    def load_state_dict(self, state):
        def clamp(value):
            return min(self.log_max, max(self.log_min, float(value)))

        self.log_ratios = {k: clamp(v) for k, v in (state.get("log_ratios") or {}).items()}
        self.accuracies = {k: float(v) for k, v in (state.get("accuracies") or {}).items()}


def build_balancer(train_config):
    """The balancer ``train_config`` asks for, or ``None`` when it is off."""
    if not bool(getattr(train_config, "d_lr_balance", False)):
        return None
    return DiscriminatorLRBalancer(
        target=getattr(train_config, "d_lr_balance_target", 0.6),
        min_ratio=getattr(train_config, "d_lr_balance_min", 0.5),
        max_ratio=getattr(train_config, "d_lr_balance_max", 2.0),
        speed=getattr(train_config, "d_lr_balance_speed", 5000),
    )
