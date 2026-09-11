"""Per-step diagnostics: the numbers the loop logs but never optimises.

Everything here is measurement -- latent gaps, per-head separation, gradient
norms, rolling means.  None of it feeds a backward pass, which is why it can
live outside the training loop.
"""

import math

import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

from rvc.lib.algorithm import commons


def prior_gap(
    model,
    m_p,
    x_mask,
    ids_slice,
    segment_size,
    pitchf,
    sid,
    target,
    full_output,
    config,
):
    """Is the Gaussian posterior carrying anything the prior does not have?

    Decodes twice -- once from the posterior sample ``z`` that produced
    ``full_output``, once from the prior mean pushed through the flow -- and
    returns the mel L1 gap between them.  A rate near zero in ``diag/kl_*`` is
    ambiguous between "prior predicts posterior" (fine) and "posterior
    collapsed" (not); a large gap here means the former, a small one the
    latter.  Diagnostic only, under ``no_grad``: optimising this gap directly
    would reward a destructive latent rather than an informative one.
    """
    # Imported here, not at module scope: ``rvc.train.utils`` reaches
    # ``mel_processing`` by the trainer's script-relative path, so importing it
    # eagerly would make this module unimportable outside a training run.
    from rvc.train.utils import wave_to_mel

    with torch.no_grad():
        speaker = model.emb_g(sid).unsqueeze(-1)
        z_prior = model.flow(m_p * x_mask, x_mask, g=speaker, reverse=True) * x_mask
        z_slice = commons.slice_segments(z_prior, ids_slice, segment_size, dim=3)
        pitchf_slice = commons.slice_segments(pitchf, ids_slice, segment_size, dim=2)
        prior_output = model.dec(z_slice, pitchf_slice, g=speaker)

        target_mel = wave_to_mel(config, target)
        full_error = F.l1_loss(
            wave_to_mel(config, full_output).float(), target_mel.float()
        )
        prior_error = F.l1_loss(
            wave_to_mel(config, prior_output).float(), target_mel.float()
        )
    return (prior_error - full_error).detach(), prior_error.detach()


def cache_mean(cache) -> float:
    """Mean of a rolling cache of device scalars, in one transfer.

    The caches these read are filled every step and read once per logging
    interval, so they hold tensors rather than floats: a per-step ``.item()``
    is a synchronisation in the middle of the training step, and this is the
    one place the window actually has to reach the host.
    """
    return torch.stack(list(cache)).mean().item()


def split_branch_outputs(outputs, sizes):
    """Split every branch's output along the batch axis into ``len(sizes)`` groups.

    The discriminator batches its fake side into one pass, so a second class of
    fake -- the synthetic negative -- rides in the same tensor and has to come
    back out before the losses can weight the two differently.  Under SAN a
    branch returns ``(function, direction)`` rather than one tensor, and both
    halves are split.
    """

    groups = [[] for _ in sizes]
    for output in outputs:
        if isinstance(output, (list, tuple)):
            parts = [torch.split(half, sizes, dim=0) for half in output]
            for index in range(len(sizes)):
                groups[index].append([half[index] for half in parts])
        else:
            parts = torch.split(output, sizes, dim=0)
            for index in range(len(sizes)):
                groups[index].append(parts[index])
    return groups


def branch_separation(disc_real_outputs, disc_generated_outputs):
    """Per-head ``mean(real logit) - mean(fake logit)``, detached.

    Under SAN a head returns ``(function, direction)`` rather than one tensor;
    the function output is the one the generator is scored by, so it is the one
    whose separation means anything.
    """

    def logits(output):
        return (output[0] if isinstance(output, (list, tuple)) else output).detach()

    return torch.stack(
        [
            logits(dr).float().mean() - logits(dg).float().mean()
            for dr, dg in zip(disc_real_outputs, disc_generated_outputs)
        ]
    )


def clip_or_sample_grad_norm(
    parameters,
    max_norm,
    step,
    sample_interval,
):
    max_norm = float(max_norm)
    should_measure = math.isfinite(max_norm) or step % max(1, sample_interval) == 0
    if not should_measure:
        return None
    return clip_grad_norm_(parameters, max_norm=max_norm)


def generator_gradient_metrics(net_g):
    """Return gradient and gradient-to-parameter norms for generator subsystems."""
    model = net_g.module if hasattr(net_g, "module") else net_g
    groups = {
        "content_encoder": [],
        "posterior": [],
        "prior": [],
        "decoder": [],
        "speaker": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("enc_p."):
            group = "content_encoder"
        elif name.startswith("enc_q."):
            group = "posterior"
        elif name.startswith("flow."):
            group = "prior"
        elif name.startswith("dec."):
            group = "decoder"
        elif name.startswith("emb_g."):
            group = "speaker"
        else:
            group = "other"
        groups[group].append(parameter)

    metrics = {}
    for group, parameters in groups.items():
        if not parameters:
            continue
        parameter_norm = torch.sqrt(
            sum(parameter.detach().float().square().sum() for parameter in parameters)
        )
        gradient_terms = [
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        gradient_norm = (
            torch.sqrt(sum(gradient_terms))
            if gradient_terms
            else parameter_norm.new_zeros(())
        )
        metrics[f"grad_norm_{group}"] = gradient_norm
        metrics[f"grad_to_param_{group}"] = gradient_norm / parameter_norm.clamp_min(1e-8)
    return metrics
