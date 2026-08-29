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
    assert VITS_ARCHITECTURE_ID.endswith("chouwagan_v2")


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


def _prior_block(window):
    from rvc.lib.algorithm.chouwa_vits import EBranchformerBlock

    torch.manual_seed(0)
    block = EBranchformerBlock(32, heads=4, kernel_size=7, attention_window=window)
    return block.eval()


@pytest.mark.parametrize("window,length_invariant", [(8, True), (0, False)])
def test_the_prior_block_encodes_a_frame_the_same_whatever_arrived_with_it(
    window, length_invariant
):
    """A bounded window is the only thing making training and inference agree.

    The dataset's preprocessing slices are all exactly 3.00 s, so the prior
    stack never sees more than ~298 frames in training, while the inference
    pipeline hands it a whole silence-delimited span.  With unbounded attention
    the softmax runs over whatever arrived, so the same frame is encoded
    differently in the two regimes -- measured on ``logs/pretrain`` at step
    24393 as 0.27-0.40 sigma of drift in the prior mean, against the 0.82 sigma
    of posterior-to-prior mismatch inference already pays.

    The window is checked away from both edges: a frame at the boundary of a
    short clip genuinely has neighbours a frame mid-utterance does not, and no
    window can or should hide that.
    """
    torch.manual_seed(1)
    short = 40
    value = torch.randn(1, 32, short)
    context = torch.cat((value, torch.randn(1, 32, 3 * short)), dim=-1)

    with torch.no_grad():
        alone = _prior_block(window)(value, None)
        together = _prior_block(window)(context, None)[..., :short]

    # Both ends: a query near the end of the short tensor has a window that
    # runs off it, which is the same edge effect as at the start.  The conv
    # branch's radius is added on top.
    margin = window + 4 if window else short // 4
    interior = slice(margin, short - margin)
    drift = (alone - together)[..., interior].abs().max().item()
    if length_invariant:
        assert drift < 1e-5, drift
    else:
        assert drift > 1e-3, drift


def test_a_padded_query_row_cannot_produce_a_nan():
    """A short item in a padded batch has rows whose whole window is padding.

    ``softmax`` over an all-masked row is NaN, and the block masks its *output*
    rather than its input, so ``0 * NaN`` would carry it into the prior head.
    """
    block = _prior_block(4)
    value = torch.randn(1, 32, 64)
    mask = torch.zeros(1, 1, 64)
    mask[..., :8] = 1.0
    with torch.no_grad():
        out = block(value, mask)
    assert torch.isfinite(out).all()


def _latent(**kwargs):
    defaults = dict(
        input_channels=8, spec_channels=16, content_channels=4, detail_channels=4,
        gin_channels=8, posterior_channels=8, prior_hidden_channels=8,
        latent_channels=4, posterior_layers=2, flow_blocks=1, flow_layers=1,
        prior_blocks=1, prior_heads=1, prior_kernel_size=3,
    )
    defaults.update(kwargs)
    return RefineVitsLatent(**defaults)


def test_the_kl_target_is_a_ceiling_and_never_pushes_the_rate_back_up():
    """``kl_beta_min = 1`` is what separates a cap from a rate equaliser.

    With the old 1e-4 floor the controller was two-sided: a rate *below* target
    decayed beta until the reconstruction term pulled the rate back up to it.
    Measured on ``logs/pretrain`` at step 24393, that held every one of the 192
    dimensions at 0.132-0.167 nats against a 0.15 target -- so every dimension
    was equally, deliberately, 0.82 sigma wrong at inference.  There is no
    posterior collapse to protect against here (swapping ``z`` moves mel L1
    from 0.50 to 1.4-2.35), so the floor bought nothing.

    Below the cap beta must sit exactly at its minimum, which makes the loss a
    plain ``c_kl`` sum again.
    """
    latent = _latent(kl_target=0.15, kl_beta_min=1.0, kl_beta_max=10.0, kl_beta_lr=0.5)
    latent.train()
    quiet = torch.full((latent.latent_channels,), 0.001)
    for _ in range(50):
        beta = latent._kl_beta(quiet, latent.kl_target_fast)
    assert torch.allclose(beta, torch.ones_like(beta)), beta

    loud = torch.full((latent.latent_channels,), 5.0)
    for _ in range(50):
        beta = latent._kl_beta(loud, latent.kl_target_fast)
    assert (beta > 1.0).all(), beta


def test_the_posterior_reads_a_log_spectrogram_not_a_linear_one():
    """``log1p`` is the identity over the range most bins occupy.

    The pretrain set's median magnitude is 0.019, where ``log1p(x)`` and ``x``
    agree to 1%.  The stem has no per-bin normalisation, so that left it
    reading a linear spectrogram, and the band above 12 kHz carried 5.2% of the
    per-bin temporal variance it saw against 32.2% under a real log.

    Pinned as the property that matters: equal *ratios* of magnitude must map
    to equal steps.  Under ``log1p`` the quiet decade collapses to ~1% of the
    loud one, which is exactly how the fine structure went missing.
    """
    from rvc.lib.algorithm.chouwa_vits import (
        LOG_SPEC_FLOOR, LOG_SPEC_MEAN, LOG_SPEC_STD,
    )

    decades = torch.tensor([1e-3, 1e-2, 1e-1, 1.0])
    transformed = (decades.clamp_min(LOG_SPEC_FLOOR).log() - LOG_SPEC_MEAN) / LOG_SPEC_STD
    steps = transformed.diff()
    assert torch.allclose(steps, steps[0].expand_as(steps), rtol=1e-5), steps

    compressed = torch.log1p(decades).diff()
    assert compressed[0] / compressed[-1] < 0.02, compressed


def test_a_padded_posterior_frame_stays_zero():
    """``log`` of a padded frame is a large negative constant, not zero.

    The stem's time kernel is three wide, so an unmasked padded column would
    smear into the real frames beside it -- a hazard ``log1p`` did not have,
    because ``log1p(0)`` is 0.
    """
    latent = _latent().eval()
    spec = torch.rand(1, 16, 6) + 0.01
    mask = torch.ones(1, 1, 6)
    mask[..., 3:] = 0.0
    with torch.no_grad():
        (mean, _), _ = latent._posterior_distribution(
            spec, torch.zeros(1, latent.prior_feature_channels, 6), None, mask
        )
    assert torch.count_nonzero(mean[..., 3:]) == 0


def test_the_speaker_bottleneck_is_exact_at_the_shipped_speaker_count():
    """``cond_rank`` factors a map whose input is an embedding lookup.

    ``g`` is one row of ``emb_g``'s ``(spk_embed_dim, gin_channels)`` table and
    is constant over time, so the set of vectors ``cond_layer`` can ever be
    shown has rank at most ``spk_embed_dim``.  A bottleneck at least that wide
    can reproduce any map on those speakers exactly -- this is not an
    approximation, and the guarantee is what the config has to keep true.
    """
    import json

    for vocoder in ("chouwagan", "wavehax"):
        with open(f"rvc/configs/{vocoder}/44100.json", encoding="utf-8") as handle:
            model = json.load(handle)["model"]
        assert model["cond_rank"] >= model["spk_embed_dim"], vocoder


def test_the_flow_stays_invertible_under_every_coupling_net():
    """A coupling that does not round-trip is not a normalizing flow.

    ``infer`` runs it in reverse and the KL is evaluated on the forward pass, so
    a mismatch between the two is silent: no error, just a prior the decoder was
    never trained against.
    """
    from rvc.lib.algorithm.normalizing_flow import ResidualCouplingBlock

    shared = dict(
        channels=64, hidden_channels=64, n_flows=2, kernel_size=5, gin_channels=32
    )
    variants = {
        "wavenet": dict(n_layers=4, dilation_rate=2, coupling_net="wavenet"),
        "transformer": dict(
            n_layers=2, dilation_rate=2, coupling_net="transformer",
            filter_channels=128, attention_window=8,
        ),
    }
    x = torch.randn(2, 64, 40)
    mask = torch.ones(2, 1, 40)
    g = torch.randn(2, 32, 1)
    for name, extra in variants.items():
        torch.manual_seed(0)
        flow = ResidualCouplingBlock(**shared, **extra).eval()
        with torch.no_grad():
            back = flow(flow(x, mask, g=g), mask, g=g, reverse=True)
        assert torch.allclose(back, x, atol=1e-5), (name, (back - x).abs().max())


def test_the_transformer_coupling_is_bounded_in_time_like_the_prior():
    """``block_length`` carries the same guarantee as ``ATTENTION_WINDOW``.

    Without it the flow would reintroduce, at inference, exactly the
    length-dependence the prior stack was just fixed for -- and the flow is the
    half of the frontend that ships.
    """
    from rvc.lib.algorithm.normalizing_flow import TransformerCouplingNet

    net = TransformerCouplingNet(
        32, n_layers=1, n_heads=2, filter_channels=64, block_length=4
    ).eval()
    assert net.attn_layers[0].block_length == 4

    unbounded = TransformerCouplingNet(
        32, n_layers=1, n_heads=2, filter_channels=64, block_length=0
    ).eval()
    assert unbounded.attn_layers[0].block_length is None
