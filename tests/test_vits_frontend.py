"""The contract ``train.py`` and ``synthesizers.py`` hold the frontend to.

Both reach into it by name in a dozen places -- parameter grouping, the ablation
sampler, the KL weighting branch, the rolling diagnostics -- and ``net_g`` loads
checkpoints non-strictly, so a missing attribute surfaces as a crash mid-epoch
and a wrong architecture id surfaces as a silently half-initialised generator.
Both are pinned here.
"""

import inspect

import pytest
import torch

from rvc.lib.algorithm.chouwa_vits import (
    ARCHITECTURE_ID as VITS_ARCHITECTURE_ID,
    RefineVitsLatent,
)

# Everything train.py or synthesizers.py touches on the frontend.
REQUIRED_METHODS = (
    "forward_train",
    "prior_losses",
    "diagnostics",
    "ablate_dimension",
    "infer",
    "set_training_step",
    "remove_posterior",
    "prior_replacement_fraction",
)
REQUIRED_ATTRIBUTES = (
    "architecture_id",
    "fast_levels",
    "kl_target_fast",
    "kl_rate_control_active",
    "posterior_available",
)


def _build(latent_channels=64, **overrides):
    kwargs = dict(
        input_channels=32,
        spec_channels=129,
        content_channels=32,
        detail_channels=16,
        gin_channels=16,
        posterior_channels=32,
        prior_hidden_channels=32,
        latent_channels=latent_channels,
        posterior_layers=2,
        flow_blocks=2,
        flow_layers=2,
        prior_blocks=1,
        prior_heads=2,
        prior_kernel_size=7,
        content_feature_channels=48,
        frame_conditioning_channels=2,
        prior_uses_logs=True,
        prior_replacement_max=0.7,
        prior_replacement_start=0,
        prior_replacement_ramp=1,
    )
    kwargs.update(overrides)
    return RefineVitsLatent(**kwargs)


def _inputs(model, batch=2, frames=24):
    return dict(
        content_stats=torch.randn(batch, 64, frames),
        spec=torch.rand(batch, 129, frames) * 3,
        g=torch.randn(batch, 16, 1),
        mask=torch.ones(batch, 1, frames),
        pitchf=torch.full((batch, frames), 150.0),
        content=torch.randn(batch, 48, frames),
        frame_conditioning=torch.rand(batch, 2, frames),
    )


def test_the_architecture_id_marks_the_decoder_it_was_trained_against():
    """A checkpoint from an older decoder must fail the guard, not load."""
    assert VITS_ARCHITECTURE_ID.endswith("chouwagan_v1")


def test_every_attribute_the_trainer_reaches_for_exists():
    for name in REQUIRED_METHODS:
        assert callable(getattr(RefineVitsLatent, name, None)), name
    model = _build()
    for name in REQUIRED_ATTRIBUTES:
        assert hasattr(model, name), name


def test_forward_train_emits_every_key_the_decoder_and_trainer_read():
    model = _build().train()
    model.set_training_step(10_000)
    parts = model.forward_train(**_inputs(model))
    for key in (
        "content",
        "detail",
        "prior_replacement",
        "prior_replacement_mean",
    ):
        assert key in parts, key
    assert parts["content"].shape[1] == 32
    assert parts["detail"].shape[1] == 16
    # The two-rate SVAE's "slow" halves are gone; the decoder's contract is the
    # content/detail pair and nothing beside it.
    assert "detail_slow" not in parts and "detail_fast" not in parts


def test_losses_and_diagnostics_are_finite_and_differentiable():
    model = _build().train()
    model.set_training_step(10_000)
    parts = model.forward_train(**_inputs(model))
    fast, total = model.prior_losses(parts, torch.ones(2, 1, 24))
    assert torch.isfinite(fast)
    assert torch.isfinite(total)
    total.backward()
    assert model.prior_content.weight.grad is not None
    assert any(p.grad is not None for p in model.flow.parameters())

    diagnostics = model.diagnostics(parts, torch.ones(2, 1, 24))
    assert "kl_fast_per_dim" in diagnostics
    assert diagnostics["kl_fast_per_dim"].numel() == 64
    # The shared logger only emits keys that are present, and there is no
    # second branch to emit: a "slow" series would be a permanent zero line.
    assert not any("slow" in key for key in diagnostics)
    for value in diagnostics.values():
        assert torch.isfinite(value).all()


def test_the_ablation_sampler_addresses_the_one_branch_directly():
    """``fast_levels`` is the whole index space the trainer draws from, and
    ``ablate_dimension`` no longer takes a branch to disambiguate."""
    model = _build()
    assert len(model.fast_levels) == model.latent_channels
    assert not hasattr(model, "slow_levels")

    model.train()
    model.set_training_step(10_000)
    parts = model.forward_train(**_inputs(model))
    ablated = model.ablate_dimension(parts, 0, parts["content"].shape[-1])
    assert ablated.shape == parts["detail"].shape
    assert not torch.equal(ablated, parts["detail"])


def test_prior_replacement_reaches_the_flow_and_the_prior():
    """The point of the frontend: the inference path gets reconstruction gradient."""
    model = _build(prior_replacement_max=1.0).train()
    model.set_training_step(10_000)
    parts = model.forward_train(**_inputs(model))
    assert float(parts["prior_replacement"]) == 1.0
    # With every item replaced, the decoder-facing tensors depend on the prior
    # and the flow, so a reconstruction-side loss alone must reach both.
    parts["content"].square().mean().backward()
    assert model.prior_fast_head.weight.grad is not None
    assert any(p.grad is not None for p in model.flow.parameters())


def test_infer_is_prior_only_and_deterministic():
    model = _build().eval()
    inputs = _inputs(model)
    with torch.no_grad():
        first = model.infer(
            inputs["content_stats"],
            inputs["g"],
            inputs["mask"],
            pitchf=inputs["pitchf"],
            content=inputs["content"],
            frame_conditioning=inputs["frame_conditioning"],
        )
        second = model.infer(
            inputs["content_stats"],
            inputs["g"],
            inputs["mask"],
            pitchf=inputs["pitchf"],
            content=inputs["content"],
            frame_conditioning=inputs["frame_conditioning"],
        )
    assert torch.allclose(first[0], second[0])
    assert first[0].shape[1] == 32


def test_remove_posterior_keeps_the_flow():
    """The flow runs at inference, so export must not strip it with the rest."""
    model = _build()
    model.remove_posterior()
    assert model.posterior_available is False
    assert model.posterior_enc is None
    assert model.flow is not None
    exported = {
        name for name, _ in model.named_parameters() if name.startswith("posterior")
    }
    assert not exported
    assert any(name.startswith("flow") for name, _ in model.named_parameters())


def test_parameter_names_land_in_the_trainer_groups():
    """train.py groups by prefix; names outside those prefixes fall to 'other'."""
    model = _build()
    known = ("posterior", "prior", "fast_to_", "content", "flow")
    unmatched = {
        name
        for name, _ in model.named_parameters()
        if not name.startswith(known)
    }
    assert not unmatched, unmatched
