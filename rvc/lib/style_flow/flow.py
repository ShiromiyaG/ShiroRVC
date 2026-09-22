"""Conditional flow matching: training loss and the Euler sampler."""

from __future__ import annotations

import torch


def sample_t(n: int, device, mean: float = 0.0, std: float = 1.0, generator=None) -> torch.Tensor:
    """Logit-normal timesteps."""
    z = torch.randn(n, device=device, generator=generator) * std + mean
    return torch.sigmoid(z)


def flow_loss(model, batch: dict, *, unit_dropout=0.0, descriptor_dropout=0.0, t=None, noise=None, t_mean=0.0, t_std=1.0):
    """MSE between predicted and target velocity over valid frames.
    Returns ``(loss, per_sample_loss, t)``."""
    x1, mask = batch["target"], batch["mask"]
    B = x1.shape[0]
    if t is None:
        t = sample_t(B, x1.device, t_mean, t_std)
    x0 = torch.randn_like(x1) if noise is None else noise
    tt = t[:, None, None]
    x_t = (1.0 - tt) * x0 + tt * x1
    drop_units = torch.rand(B, device=x1.device) < unit_dropout
    drop_desc = torch.rand(B, device=x1.device) < descriptor_dropout
    v = model(
        x_t, t, batch["units"], batch["coarse"], batch["desc"], batch["desc_mask"], mask,
        drop_units=drop_units, drop_desc=drop_desc,
    )
    m = mask[:, None].float()
    err = ((v.float() - (x1 - x0)) ** 2 * m).sum(dim=(1, 2))
    per_sample = err / (m.sum(dim=(1, 2)) * x1.shape[1]).clamp(min=1.0)
    return per_sample.mean(), per_sample, t


@torch.no_grad()
def sample(
    model,
    units,
    coarse,
    desc,
    desc_mask,
    mask,
    *,
    steps: int = 32,
    cfg_scale: float = 1.0,
    cfg_drop: str = "descriptors",
    source=None,
    strength: float = 1.0,
    generator=None,
):
    """Euler integration from noise to target.  With ``source`` (a target,
    e.g. the source's own residual) and ``strength`` < 1, it starts at
    ``t0 = 1 - strength`` from that point on the path instead of pure noise.
    CFG contrasts against the descriptors dropped, or everything with
    ``cfg_drop="all"``."""
    B, T = units.shape
    C = model.cfg.target_channels
    x = torch.randn(B, C, T, device=units.device, generator=generator)
    t0 = 0.0
    if source is not None and strength < 1.0:
        t0 = 1.0 - max(0.0, strength)
        x = (1.0 - t0) * x + t0 * source
    if t0 >= 1.0:
        return source

    guided = cfg_scale != 1.0
    if guided:
        null = torch.zeros(B, dtype=torch.bool, device=units.device)
        full = torch.ones(B, dtype=torch.bool, device=units.device)
        drop_units = torch.cat([null, full if cfg_drop == "all" else null])
        drop_desc = torch.cat([null, full])
        cond = [torch.cat([a, a]) for a in (units, coarse, desc, desc_mask, mask)]

    ts = torch.linspace(t0, 1.0, steps + 1, device=units.device)
    for i in range(steps):
        t = ts[i].expand(B)
        if guided:
            v = model(torch.cat([x, x]), torch.cat([t, t]), *cond, drop_units=drop_units, drop_desc=drop_desc).float()
            v_cond, v_uncond = v.chunk(2)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, units, coarse, desc, desc_mask, mask).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x * mask[:, None]
