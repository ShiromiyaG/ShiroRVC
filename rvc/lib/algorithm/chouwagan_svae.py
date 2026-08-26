"""Continuous stochastic latent frontend used exclusively by ChouwaGAN.

This module replaces the former FSQ frontend with a two-rate conditional VAE:
an E-Branchformer prior predicts fast and slow diagonal Gaussians, while a
training-only ConvNeXt posterior observes the target spectrogram.  The decoder
receives only sampled latent projections; content features never bypass the
latent distributions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# v2 splits the E-Branchformer's tied macaron FFN into two independent
# sublayers, which renames parameters.  ``net_g`` loads ChouwaGAN checkpoints
# non-strictly, so a v1 checkpoint would otherwise load with both feed-forward
# sublayers left at their init instead of failing.  The id is the only thing
# standing between that and a silently half-random generator.
#
# v3 halves ``fast_latent_channels`` (64 -> 32) and makes the KL rate
# multipliers per dimension, so ``kl_log_beta_*`` and ``kl_rate_ema_*`` change
# from scalars to vectors.  Every latent-facing tensor changes shape; the same
# non-strict load would leave them at init.
ARCHITECTURE_ID = "shiro_vits_svae_v3"
SUPPORTED_ARCHITECTURE_IDS = frozenset((ARCHITECTURE_ID,))

# Bounds on the predicted log standard deviations.  These exist to stop a bad
# init from producing inf/nan, not to shape the solution -- a run that *sits* on
# either bound has lost the gradient on its variance head, because ``clamp``
# passes nothing through outside its range.  The previous ceiling of 1.0 was
# tight enough that the slow branch reached it and stayed there, so it is set
# well clear of any scale the model should plausibly want.  The real defence
# against drift is ``kl_scale_anchor`` below; this is only the guard rail.
LOGS_MIN = -6.0
LOGS_MAX = 2.0


def _resize_mask(mask: Optional[Tensor], length: int) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.shape[-1] == int(length):
        return mask
    return F.interpolate(mask.float(), size=int(length), mode="nearest").to(mask.dtype)


def _resize_sequence(value: Tensor, length: int) -> Tensor:
    if value.shape[-1] == int(length):
        return value
    return F.interpolate(value, size=int(length), mode="linear", align_corners=False)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class EBranchformerBlock(nn.Module):
    """Compact local/global branchformer block for the 100 Hz prior stream."""

    def __init__(self, channels: int, heads: int = 4, kernel_size: int = 31):
        super().__init__()
        channels = int(channels)
        heads = max(1, min(int(heads), channels))
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        # Macaron feed-forward pair.  These were previously a single ``ffn`` /
        # ``ffn_norm`` pair *applied twice*, once before the branches and once
        # after the merge.  Tying them forces one set of weights to serve two
        # different roles and hands it the sum of two unrelated gradients; the
        # half-step residual only makes sense with independent sublayers.
        self.ffn_norm_in = ChannelLayerNorm(channels)
        self.ffn_in = self._feed_forward(channels)
        self.ffn_norm_out = ChannelLayerNorm(channels)
        self.ffn_out = self._feed_forward(channels)
        self.local_norm = ChannelLayerNorm(channels)
        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(channels, channels, 1),
        )
        self.global_norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.global_out = nn.Conv1d(channels, channels, 1)
        self.merge = nn.Conv1d(channels * 2, channels, 1)
        self.final_norm = ChannelLayerNorm(channels)

    @staticmethod
    def _feed_forward(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1),
        )

    @staticmethod
    def _normalize(norm: ChannelLayerNorm, value: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Layer-norm, then re-zero padding.

        ``LayerNorm`` maps an all-zero padded frame onto its own bias, so a
        block that only masks its output still feeds a nonzero constant into
        every sublayer input.  The conv branch spreads that a kernel radius;
        attention is worse, because it becomes a key at every padded position
        that all the real frames then attend to.
        """
        value = norm(value)
        return value if mask is None else value * mask

    def forward(self, value: Tensor, mask: Optional[Tensor]) -> Tensor:
        mask = _resize_mask(mask, value.shape[-1])
        # ``(batch, 1, 1, key)`` broadcasts over heads and query positions.  The
        # key set always retains at least one real frame, so no query row is
        # fully masked and the softmax cannot produce NaN.
        attention_mask = None if mask is None else mask.bool().unsqueeze(1)
        value = value + 0.5 * self.ffn_in(self._normalize(self.ffn_norm_in, value, mask))
        local = self.local(self._normalize(self.local_norm, value, mask))
        normalized = self._normalize(self.global_norm, value, mask)
        batch, channels, length = normalized.shape
        q, k, v = self.qkv(normalized).reshape(
            batch, 3, self.heads, self.head_dim, length
        ).unbind(1)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attended = attended.permute(0, 1, 3, 2).reshape(batch, channels, length)
        global_branch = self.global_out(attended)
        value = value + self.merge(torch.cat((local, global_branch), dim=1))
        value = value + 0.5 * self.ffn_out(self._normalize(self.ffn_norm_out, value, mask))
        value = self.final_norm(value)
        return value if mask is None else value * mask


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        channels = int(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3 * int(dilation),
            dilation=int(dilation),
            groups=channels,
        )
        self.norm = ChannelLayerNorm(channels)
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.project = nn.Conv1d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1), 1e-2))

    def forward(self, value: Tensor, mask: Optional[Tensor]) -> Tensor:
        mask = _resize_mask(mask, value.shape[-1])
        residual = value
        # Masked twice, and both are load-bearing.  On entry, because the first
        # block in a stack is fed an unmasked projection whose bias makes the
        # padded frames nonzero, and the dilated depthwise conv would spread
        # that up to twelve frames into the real signal.  After the norm,
        # because ``LayerNorm`` maps a zeroed frame onto its own bias and hands
        # the same problem to the pointwise pair.
        if mask is not None:
            value = value * mask
        value = self.norm(self.depthwise(value))
        if mask is not None:
            value = value * mask
        value = self.project(F.gelu(self.expand(value)))
        value = residual + self.scale * value
        return value if mask is None else value * mask


class ChouwaContinuousLatent(nn.Module):
    """Two-rate continuous stochastic VAE frontend for the Chouwa decoder."""

    architecture_id = ARCHITECTURE_ID

    def __init__(
        self,
        input_channels: int,
        spec_channels: Optional[int] = None,
        content_channels: int = 128,
        detail_channels: int = 64,
        gin_channels: int = 256,
        posterior_channels: int = 192,
        prior_hidden_channels: int = 192,
        slow_latent_channels: int = 32,
        fast_latent_channels: int = 32,
        prior_blocks: int = 4,
        prior_heads: int = 4,
        prior_kernel_size: int = 31,
        posterior_blocks: int = 4,
        posterior_slow_blocks: int = 2,
        kl_free_bits_slow: float = 0.05,
        kl_free_bits_fast: float = 0.05,
        kl_target_slow: float = 0.0,
        kl_target_fast: float = 0.0,
        kl_beta_lr: float = 0.01,
        kl_beta_min: float = 1e-4,
        kl_beta_max: float = 10.0,
        kl_rate_momentum: float = 0.01,
        kl_scale_anchor: float = 1.0,
        feature_scale_anchor: float = 1.0,
        prior_uses_logs: bool = False,
        prior_replacement_max: float = 0.0,
        prior_replacement_start: int = 5000,
        prior_replacement_ramp: int = 20000,
        prior_replacement_mean_share: float = 0.5,
        **_: object,
    ):
        super().__init__()
        self.architecture_id = ARCHITECTURE_ID
        self.input_channels = int(input_channels)
        self.spec_channels = int(spec_channels or input_channels)
        self.content_channels = int(content_channels)
        self.detail_channels = int(detail_channels)
        self.slow_latent_channels = int(slow_latent_channels)
        self.fast_latent_channels = int(fast_latent_channels)
        self.slow_levels = tuple(range(self.slow_latent_channels))
        self.fast_levels = tuple(range(self.fast_latent_channels))
        self.slow_rate_factor = 4
        self.kl_free_bits_slow = float(kl_free_bits_slow)
        self.kl_free_bits_fast = float(kl_free_bits_fast)

        # ---- Rate-targeting KL controller -------------------------------
        # Beta is a Lagrange multiplier for the constraint "KL == target": it
        # rises while the rate overshoots and falls while it undershoots, so
        # the KL gradient stays two-sided rather than vanishing once the rate
        # is low enough.  Targets are expressed in nats per dimension per
        # frame, which is the unit the per-dim diagnostics already report.
        #
        # The floor stays on underneath it.  Dropping it looks right -- free
        # bits zero the gradient below the floor, and a latent that is not being
        # trained is not being kept honest either -- but the constraint the
        # controller enforces is on the *mean*, and a mean is satisfied just as
        # exactly by four live dimensions and sixty dead ones.  The zeroed
        # gradient is the point rather than the cost: it stops a dimension
        # already at the floor from being pushed further into the prior.  See
        # ``prior_losses`` for how the two interact.
        self.kl_target_slow = max(0.0, float(kl_target_slow))
        self.kl_target_fast = max(0.0, float(kl_target_fast))
        self.kl_beta_lr = max(0.0, float(kl_beta_lr))
        self.kl_beta_min = max(1e-8, float(kl_beta_min))
        self.kl_beta_max = max(self.kl_beta_min, float(kl_beta_max))
        self.kl_rate_momentum = min(1.0, max(1e-6, float(kl_rate_momentum)))

        # ---- Latent scale anchor ----------------------------------------
        # The KL only ever sees ``sigma_p / sigma_q`` and the mean gap measured
        # in units of ``sigma_p``, so inflating prior and posterior together
        # leaves it exactly unchanged.  The decoder's first layer is linear and
        # the blocks behind it are layer-normalised, so scaling the latent up
        # and the projection down is very nearly a symmetry of the whole
        # objective -- a flat direction the optimiser is free to wander along,
        # and it does, carrying the decoder's feature RMS with it while the KL
        # sits on target the whole time.
        # Penalising the *mean* log scale toward zero removes that one degree
        # of freedom and nothing else: per-dimension structure is untouched,
        # since only the average is constrained.
        self.kl_scale_anchor = max(0.0, float(kl_scale_anchor))

        # ---- Decoder-facing feature anchor -------------------------------
        # ``kl_scale_anchor`` holds sigma at 1 successfully, and the drift simply
        # moves.  The KL is a function of ``mu_q - mu_p`` measured in units of
        # ``sigma_p``, so inflating *both* means together is invisible to it in
        # exactly the way inflating both scales was, and the ``*_to_content`` /
        # ``*_to_detail`` projection gains are outside every term in the
        # objective.  Anchor a cause and the drift finds the next flat
        # direction; anchor the observable and it has nowhere to go.
        # Anchoring the log RMS of what the decoder actually receives closes
        # both leaks at once, and closes them wherever the next one opens: it
        # constrains the observable rather than one of its causes.  Log-space
        # and mean-only for the same reasons as above -- 2x too large costs the
        # same as 2x too small, and per-channel structure stays free.
        self.feature_scale_anchor = max(0.0, float(feature_scale_anchor))
        # One multiplier *per dimension*, not one per branch.  A single beta
        # applies the same marginal pressure to a dimension carrying 1.5 nats
        # and one carrying 0.04, because the penalty is a plain sum and its
        # gradient with respect to each dimension is 1 either way.  Nothing in
        # that objective resists concentration -- it rewards it, since the model
        # is free to spend the whole budget wherever reconstruction pays best
        # and the constraint is still satisfied on average.  A mean sitting
        # exactly on target is compatible with most of the dimensions being
        # dead, so the mean cannot detect the failure it permits.
        #
        # Per dimension the error signal is per dimension too: a dimension over
        # target gets its own rising beta and a starved one falls to
        # ``kl_beta_min`` and is left alone for reconstruction to claim.  Total
        # budget is unchanged -- the target is still 0.15 per dimension -- so
        # this redistributes rather than adds rate.  The free-bits floor stays
        # underneath as the lower guard; it is not the lever here, and raising
        # it only parks dimensions on it (see the note above).
        for branch, target, width in (
            ("slow", self.kl_target_slow, self.slow_latent_channels),
            ("fast", self.kl_target_fast, self.fast_latent_channels),
        ):
            self.register_buffer(
                f"kl_log_beta_{branch}", torch.zeros(width), persistent=True
            )
            self.register_buffer(
                f"kl_rate_ema_{branch}",
                torch.full((width,), float(target)),
                persistent=True,
            )

        self.training_step = 0
        self.current_prior_replacement = 0.0
        self.coarse_spectral_loss_weight = 0.0
        self.usage_loss_weight = 0.0
        self.content_path_dropout = 0.0

        # ---- Prior replacement ------------------------------------------
        # The decoder is trained exclusively on posterior samples, so ``enc_p``
        # and the prior only ever see the KL gradient -- which the rate
        # controller deliberately keeps small.  Inference runs entirely through
        # ``enc_p -> prior -> dec``, so that path is the one that has to be good.
        # Swapping a scheduled fraction of the batch onto prior samples gives
        # both modules real reconstruction gradient, at no extra forward cost.
        #
        # It closes only half of the train/infer mismatch on its own -- the
        # module half.  ``prior_replacement_mean_share`` closes the other half,
        # the operating point; see ``_replacement_scales`` for the measurement
        # that sized it.
        self.prior_replacement_max = min(1.0, max(0.0, float(prior_replacement_max)))
        self.prior_replacement_start = max(0, int(prior_replacement_start))
        self.prior_replacement_ramp = max(1, int(prior_replacement_ramp))
        self.prior_replacement_mean_share = min(
            1.0, max(0.0, float(prior_replacement_mean_share))
        )

        # ``enc_p`` emits a full Gaussian but the ChouwaGAN path historically
        # forwarded only its mean, leaving half of the projection untrained and
        # the encoder's own uncertainty estimate unused.
        self.prior_uses_logs = bool(prior_uses_logs)
        prior_input_channels = self.input_channels * (2 if self.prior_uses_logs else 1)
        self.prior_input = nn.Conv1d(prior_input_channels, int(prior_hidden_channels), 1)
        self.prior_feature_channels = int(prior_hidden_channels)
        self.prior_f0 = nn.Conv1d(2, int(prior_hidden_channels), 1)
        self.prior_speaker = nn.Conv1d(int(gin_channels), int(prior_hidden_channels), 1)
        self.prior_blocks = nn.ModuleList(
            [EBranchformerBlock(prior_hidden_channels, prior_heads, prior_kernel_size) for _ in range(int(prior_blocks))]
        )
        self.prior_fast = nn.Conv1d(prior_hidden_channels, self.fast_latent_channels * 2, 1)
        self.prior_slow = nn.Conv1d(prior_hidden_channels, self.slow_latent_channels * 2, 1)
        self.prior_slow_down = nn.Sequential(
            nn.Conv1d(prior_hidden_channels, prior_hidden_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(prior_hidden_channels, prior_hidden_channels, 5, stride=2, padding=2),
            nn.SiLU(),
        )

        self.posterior_input = nn.Conv1d(self.spec_channels, int(posterior_channels), 1)
        self.posterior_condition = nn.Conv1d(
            self.prior_feature_channels,
            int(posterior_channels),
            1,
        )
        self.posterior_blocks = nn.ModuleList(
            [ConvNeXtBlock1d(posterior_channels, (1, 2, 4, 1)[index % 4]) for index in range(int(posterior_blocks))]
        )
        self.posterior_slow_down = nn.Sequential(
            nn.Conv1d(posterior_channels, posterior_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(posterior_channels, posterior_channels, 5, stride=2, padding=2),
            nn.SiLU(),
        )
        self.posterior_slow_blocks = nn.ModuleList(
            [ConvNeXtBlock1d(posterior_channels, (1, 2)[index % 2]) for index in range(int(posterior_slow_blocks))]
        )
        self.posterior_fast = nn.Conv1d(posterior_channels, self.fast_latent_channels * 2, 1)
        self.posterior_slow = nn.Conv1d(posterior_channels, self.slow_latent_channels * 2, 1)

        self.fast_to_content = nn.Conv1d(self.fast_latent_channels, self.content_channels, 1)
        self.slow_to_detail = nn.Conv1d(self.slow_latent_channels, self.detail_channels, 1)
        self.fast_to_detail = nn.Conv1d(self.fast_latent_channels, self.detail_channels, 1)
        self.posterior_available = True

    @staticmethod
    def _distribution(value: Tensor) -> Tuple[Tensor, Tensor]:
        mean, logs = value.chunk(2, 1)
        return mean, logs.clamp(LOGS_MIN, LOGS_MAX)

    @staticmethod
    def _sample(mean: Tensor, logs: Tensor, scale=1.0) -> Tensor:
        # ``scale`` may be a per-item tensor broadcast over (B, 1, 1); the draw
        # happens either way so that the RNG stream does not depend on it.
        if not torch.is_tensor(scale):
            scale = float(scale)
        return mean + torch.randn_like(mean) * logs.exp() * scale

    def _prior_features(self, content_stats: Tensor, g: Optional[Tensor], mask: Optional[Tensor], pitchf: Optional[Tensor]):
        value = self.prior_input(content_stats)
        if pitchf is None:
            pitchf = content_stats.new_zeros(content_stats.shape[0], content_stats.shape[-1])
        if pitchf.ndim == 3:
            f0 = pitchf[:, :2]
        else:
            f0 = torch.stack((torch.log1p(pitchf.clamp_min(0)) / 7.0, (pitchf > 0).to(content_stats.dtype)), 1)
        value = value + self.prior_f0(_resize_sequence(f0, content_stats.shape[-1]))
        if g is not None:
            value = value + self.prior_speaker(g).expand(-1, -1, value.shape[-1])
        for block in self.prior_blocks:
            value = block(value, mask)
        return value if mask is None else value * mask

    def _posterior_features(self, spec: Tensor, condition: Tensor, mask: Optional[Tensor]):
        target = torch.log1p(spec.float().clamp_min(0.0))
        target_mask = _resize_mask(mask, target.shape[-1])
        value = self.posterior_input(target) + self.posterior_condition(_resize_sequence(condition.detach(), target.shape[-1]))
        for block in self.posterior_blocks:
            value = block(value, target_mask)
        return value if target_mask is None else value * target_mask, target_mask

    def _latent_to_decoder(self, slow: Tensor, fast: Tensor, target_length: int):
        slow_detail = self.slow_to_detail(_resize_sequence(slow, target_length))
        fast_detail = self.fast_to_detail(_resize_sequence(fast, target_length))
        content = self.fast_to_content(_resize_sequence(fast, target_length))
        return content, slow_detail + fast_detail, slow_detail, fast_detail

    @staticmethod
    def _resize_distribution(distribution, length: int):
        return tuple(_resize_sequence(value, length) for value in distribution)

    def set_training_step(self, step: int) -> None:
        self.training_step = max(0, int(step))

    def prior_replacement_fraction(self) -> float:
        """Fraction of the batch decoded from prior samples at this step."""
        if not self.training or self.prior_replacement_max <= 0.0:
            return 0.0
        elapsed = self.training_step - self.prior_replacement_start
        if elapsed <= 0:
            return 0.0
        progress = min(1.0, elapsed / self.prior_replacement_ramp)
        self.current_prior_replacement = self.prior_replacement_max * progress
        return self.current_prior_replacement

    @staticmethod
    def _replacement_selector(batch_size, fraction, device, dtype):
        """Pick exactly ``round(batch * fraction)`` items, unbiased.

        A per-item Bernoulli draw has the right expectation but, at the batch
        sizes this trains on, frequently selects zero items -- which makes the
        reconstruction gradient reaching the prior arrive in bursts.  Drawing a
        fixed count with a stochastically rounded remainder keeps the same
        expectation with far less variance.
        """
        if fraction <= 0.0 or batch_size < 1:
            return None
        expected = batch_size * float(fraction)
        count = int(expected)
        if torch.rand((), device=device).item() < (expected - count):
            count += 1
        count = min(batch_size, count)
        if count < 1:
            return None
        chosen = torch.randperm(batch_size, device=device)[:count]
        selector = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        selector[chosen] = 1.0
        return selector

    def _replacement_scales(self, selector: Tensor) -> Tensor:
        """Per-item prior sampling scale for the replaced batch items.

        Replacement fixes *which module* gets reconstruction gradient, but on
        its own it leaves the operating point mismatched: it feeds the decoder
        prior samples at scale 1, while ``infer`` runs ``deterministic=True`` and
        hands it the prior *mean*.  The mean is not a mild case of the sample --
        it is systematically less energetic, which is what ``prior_detail_rms``
        against ``posterior_detail_rms`` reports -- so a decoder trained only on
        samples pays a large penalty purely for being evaluated somewhere it was
        never trained.

        So the replaced items are split across both operating points rather than
        moved to either one.  Scale 0 is what deterministic inference uses and
        has to be in distribution; scale 1 keeps the sampling regulariser and is
        the only thing that puts reconstruction gradient on the prior's
        ``logs``, which would otherwise be left to the KL alone.  Same
        stochastically rounded count as the selector above, and for the same
        reason: a Bernoulli draw at batch 8 too often yields none of one kind.
        """
        if self.prior_replacement_mean_share <= 0.0:
            return selector
        chosen = selector.reshape(-1).nonzero(as_tuple=True)[0]
        count = chosen.numel()
        if count < 1:
            return selector
        expected = count * self.prior_replacement_mean_share
        mean_count = int(expected)
        if torch.rand((), device=selector.device).item() < (expected - mean_count):
            mean_count += 1
        if mean_count < 1:
            return selector
        scales = selector.clone()
        order = chosen[torch.randperm(count, device=selector.device)]
        scales[order[:mean_count]] = 0.0
        return scales

    def forward_train(self, content_stats, spec, g, mask, pitchf=None):
        # Gaussian statistics and their reparameterized samples are especially
        # sensitive to FP16's narrow exponent range.  Keep the complete SVAE
        # frontend in FP32; the surrounding decoder autocast still provides the
        # bulk of AMP's memory and throughput savings.
        device_type = content_stats.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            content_stats = content_stats.float()
            spec = spec.float()
            g = None if g is None else g.float()
            mask = None if mask is None else mask.float()
            pitchf = None if pitchf is None else pitchf.float()

            prior_features = self._prior_features(content_stats, g, mask, pitchf)
            prior_fast = self._distribution(self.prior_fast(prior_features))
            prior_slow_features = self.prior_slow_down(prior_features)
            prior_slow = self._distribution(self.prior_slow(prior_slow_features))
            if not self.posterior_available:
                raise RuntimeError("The training-only SVAE posterior is unavailable.")
            posterior_features, posterior_mask = self._posterior_features(spec, prior_features, mask)
            posterior_fast = self._resize_distribution(
                self._distribution(self.posterior_fast(posterior_features)),
                prior_fast[0].shape[-1],
            )
            posterior_slow_features = self.posterior_slow_down(posterior_features)
            for block in self.posterior_slow_blocks:
                posterior_slow_features = block(
                    posterior_slow_features,
                    _resize_mask(posterior_mask, posterior_slow_features.shape[-1]),
                )
            posterior_slow = self._resize_distribution(
                self._distribution(self.posterior_slow(posterior_slow_features)),
                prior_slow[0].shape[-1],
            )
            fast = self._sample(*posterior_fast)
            slow = self._sample(*posterior_slow)

            # Scheduled prior replacement.  Whole batch items are swapped rather
            # than individual frames so the decoder always receives a coherent
            # sequence, and the prior sample keeps its graph so the
            # reconstruction loss reaches ``prior_*`` and ``enc_p``.
            replacement_fraction = self.prior_replacement_fraction()
            selector = self._replacement_selector(
                fast.shape[0], replacement_fraction, fast.device, fast.dtype
            )
            if selector is not None:
                scales = self._replacement_scales(selector)
                fast = torch.lerp(fast, self._sample(*prior_fast, scale=scales), selector)
                slow = torch.lerp(slow, self._sample(*prior_slow, scale=scales), selector)
                prior_replacement = selector.mean().detach()
                prior_replacement_mean = (
                    (selector - scales).mean().detach() / selector.mean().clamp_min(1e-6)
                )
            else:
                prior_replacement = fast.new_zeros(())
                prior_replacement_mean = fast.new_zeros(())

            content, detail, slow_detail, fast_detail = self._latent_to_decoder(
                slow, fast, content_stats.shape[-1]
            )
            with torch.no_grad():
                _, prior_detail, _, _ = self._latent_to_decoder(
                    prior_slow[0], prior_fast[0], content_stats.shape[-1]
                )
        return {
            "content": content,
            "detail": detail,
            "detail_slow": slow_detail,
            "detail_fast": fast_detail,
            "posterior_slow_distribution": posterior_slow,
            "posterior_fast_distribution": posterior_fast,
            "prior_slow_distribution": prior_slow,
            "prior_fast_distribution": prior_fast,
            "prior_detail": prior_detail,
            "posterior_slow_values": slow,
            "posterior_fast_values": fast,
            "prior_replacement": prior_replacement,
            "prior_replacement_mean": prior_replacement_mean,
            "coarse_spectral_loss": content.new_zeros(()),
        }

    @staticmethod
    def _kl(q, p, free_bits: float, mask: Optional[Tensor] = None):
        mu_q, logs_q = (value.float() for value in q)
        mu_p, logs_p = (value.float() for value in p)
        value = logs_p - logs_q + 0.5 * (
            (2.0 * (logs_q - logs_p)).exp()
            + (mu_q - mu_p).square() * (-2.0 * logs_p).exp()
            - 1.0
        )
        if mask is None:
            dimensions = value.float().mean(dim=(0, 2))
        else:
            valid = _resize_mask(mask, value.shape[-1]).float()
            dimensions = (value.float() * valid).sum(dim=(0, 2)) / valid.sum().clamp_min(1.0)
        # Per dimension and still differentiable: the per-branch multiplier is
        # now a vector, so the caller weights these before summing.
        return dimensions.clamp_min(float(free_bits)), dimensions.detach()

    @property
    def kl_rate_control_active(self) -> bool:
        return self.kl_target_slow > 0.0 or self.kl_target_fast > 0.0

    def _kl_beta(self, branch: str, rate: Tensor, target: float) -> Tensor:
        """Advance the Lagrange multipliers for one branch and return them.

        ``rate`` is the detached per-dimension KL vector in nats per dimension
        per frame, and everything below is elementwise: each dimension carries
        its own multiplier and sees only its own error.  The multiplicative
        update on ``log beta`` keeps the multipliers positive and makes the
        response symmetric in relative error, so overshooting by 2x costs the
        same as undershooting by 2x.
        """
        log_beta = getattr(self, f"kl_log_beta_{branch}")
        if target <= 0.0:
            return log_beta.exp()
        rate_ema = getattr(self, f"kl_rate_ema_{branch}")
        if self.training:
            with torch.no_grad():
                rate_ema.mul_(1.0 - self.kl_rate_momentum).add_(
                    self.kl_rate_momentum * rate
                )
                error = ((rate_ema - target) / target).clamp(-1.0, 1.0)
                log_beta.add_(self.kl_beta_lr * error).clamp_(
                    math.log(self.kl_beta_min), math.log(self.kl_beta_max)
                )
        return log_beta.exp()

    def prior_losses(self, parts: Dict[str, Tensor], mask: Optional[Tensor]):
        # ``*_dims`` is the raw per-dimension rate the controller reads;
        # ``clamped_*`` is the same vector under the free-bits floor and is what
        # the loss is built from.
        clamped_slow, slow_dims = self._kl(
            parts["posterior_slow_distribution"],
            parts["prior_slow_distribution"],
            self.kl_free_bits_slow,
            mask,
        )
        clamped_fast, fast_dims = self._kl(
            parts["posterior_fast_distribution"],
            parts["prior_fast_distribution"],
            self.kl_free_bits_fast,
            mask,
        )
        parts["kl_slow_per_dim"] = slow_dims
        parts["kl_fast_per_dim"] = fast_dims
        anchor = self._scale_anchor(parts)
        parts["scale_anchor"] = anchor.detach()

        if not self.kl_rate_control_active:
            parts["kl_beta_slow"] = slow_dims.new_ones(())
            parts["kl_beta_fast"] = fast_dims.new_ones(())
            total_slow = clamped_slow.sum()
            total_fast = clamped_fast.sum()
            return total_slow, total_fast, total_slow + total_fast + anchor

        # Rate control and the free-bits floor compose; they are not
        # alternatives, and neither is the lever the other one is.
        #
        # The controller targets each dimension, not ``dims.mean()``.  A mean is
        # blind to how the KL is distributed -- every dimension at the target and
        # a handful holding everything produce the same mean -- so a concentrated
        # latent satisfies the constraint exactly and the multiplier stops
        # moving.  Per dimension (see ``_kl_beta``) removes that blind spot at
        # the source rather than asking the floor to cover for it.
        #
        # The floor still does its own job underneath: clamping stops the KL
        # gradient on dimensions already below it, which is what lets
        # reconstruction claim them -- on the raw KL those dead dimensions were
        # still being actively pushed at the prior.  The controller is unharmed
        # because it reads the *raw* per-dim rate below, so a dimension the
        # floor has released still reports its true rate.
        beta_slow = self._kl_beta("slow", slow_dims, self.kl_target_slow).detach()
        beta_fast = self._kl_beta("fast", fast_dims, self.kl_target_fast).detach()
        # Reported as the branch mean so the series keeps the scale it had when
        # there was one multiplier per branch.
        parts["kl_beta_slow"] = beta_slow.mean()
        parts["kl_beta_fast"] = beta_fast.mean()
        loss_slow = (clamped_slow * beta_slow).sum()
        loss_fast = (clamped_fast * beta_fast).sum()
        # The anchor rides on the total only: ``loss_slow``/``loss_fast`` are
        # reported as the per-branch divergences and should keep meaning that.
        return loss_slow, loss_fast, loss_slow + loss_fast + anchor

    def _scale_anchor(self, parts: Dict[str, Tensor]) -> Tensor:
        """Pull the average log scale of every branch toward zero.

        Squared so the pull is proportional to how far the drift has already
        gone, and taken over the mean rather than per element so a dimension
        that genuinely wants to be narrow or wide is free to be.

        The second half does the same job one step downstream, on the RMS of
        the tensors the decoder is handed, which is the quantity the two flat
        directions actually move.  See ``feature_scale_anchor``.
        """
        distributions = (
            parts["prior_slow_distribution"],
            parts["prior_fast_distribution"],
            parts["posterior_slow_distribution"],
            parts["posterior_fast_distribution"],
        )
        total = distributions[0][1].float().new_zeros(())
        if self.kl_scale_anchor > 0.0:
            for _mean, logs in distributions:
                total = total + logs.float().mean().square()
            total = total * self.kl_scale_anchor
        if self.feature_scale_anchor > 0.0:
            features = total.new_zeros(())
            for key in ("content", "detail"):
                value = parts[key].float()
                rms = value.square().mean().clamp_min(1e-12).sqrt()
                features = features + rms.log().square()
            total = total + features * self.feature_scale_anchor
        return total

    def usage_regularization(self, parts, mask):
        zero = parts["content"].new_zeros(())
        return zero, {"usage_variance_floor": zero, "usage_entropy_floor": zero, "usage_covariance": zero, "usage_gate_fraction": zero}

    @staticmethod
    def _kl_effective_dims(per_dim: Tensor) -> Tensor:
        """How many dimensions actually carry this branch's KL.

        The participation ratio ``(sum x)^2 / sum x^2``: equal to the dimension
        count when every dimension carries the same divergence, and to 1 when a
        single one carries all of it.  It needs no threshold, and unlike the
        mean -- which is what the rate controller targets -- it cannot be
        satisfied by a collapsed latent, so it is the series to watch for
        posterior collapse.  The equivalent metrics that already exist further
        up (``diag/kl_active_fraction`` and friends) are computed inside the
        Gaussian VITS branch, which this model never enters.
        """
        value = per_dim.clamp_min(0.0)
        return value.sum().square() / value.square().sum().clamp_min(1e-12)

    def diagnostics(self, parts, mask):
        with torch.no_grad():
            slow_kl = parts.get("kl_slow_per_dim", parts["posterior_slow_distribution"][0].new_zeros(self.slow_latent_channels))
            fast_kl = parts.get("kl_fast_per_dim", parts["posterior_fast_distribution"][0].new_zeros(self.fast_latent_channels))
            return {
                "kl_effective_dims_slow": self._kl_effective_dims(slow_kl),
                "kl_effective_dims_fast": self._kl_effective_dims(fast_kl),
                # Fraction sitting above the free-bits floor, i.e. the fraction
                # still receiving a KL gradient at all.
                "kl_above_floor_slow": (
                    slow_kl > self.kl_free_bits_slow
                ).float().mean(),
                "kl_above_floor_fast": (
                    fast_kl > self.kl_free_bits_fast
                ).float().mean(),
                "kl_median_slow": slow_kl.median(),
                "kl_median_fast": fast_kl.median(),
                "prior_kl_slow": slow_kl.mean(),
                "prior_kl_fast": fast_kl.mean(),
                "prior_std_slow": parts["prior_slow_distribution"][1].exp().mean(),
                "prior_std_fast": parts["prior_fast_distribution"][1].exp().mean(),
                # The posterior side was previously invisible, which is what
                # let both branches drift into the log-scale clamp together
                # without any series showing it -- the KL cannot, since it only
                # measures the ratio of the two.
                "posterior_std_slow": parts["posterior_slow_distribution"][1].exp().mean(),
                "posterior_std_fast": parts["posterior_fast_distribution"][1].exp().mean(),
                "scale_anchor": parts.get("scale_anchor", slow_kl.new_zeros(())),
                "prior_replacement": parts["prior_replacement"],
                # Share of the replaced items decoded from the prior *mean*,
                # i.e. from the exact input deterministic inference supplies.
                "prior_replacement_mean": parts.get(
                    "prior_replacement_mean", slow_kl.new_zeros(())
                ),
                "kl_beta_slow": parts.get(
                    "kl_beta_slow", slow_kl.new_ones(())
                ),
                "kl_beta_fast": parts.get(
                    "kl_beta_fast", fast_kl.new_ones(())
                ),
                "content_rms": parts["content"].float().square().mean().sqrt(),
                "posterior_detail_rms": parts["detail"].float().square().mean().sqrt(),
                "prior_detail_rms": parts["prior_detail"].float().square().mean().sqrt(),
                "kl_slow": slow_kl.mean(),
                "kl_fast": fast_kl.mean(),
                "kl_slow_per_dim": slow_kl,
                "kl_fast_per_dim": fast_kl,
            }

    def ablate_dimension(self, parts, branch: str, dimension: int, target_length: int):
        slow = parts["posterior_slow_values"].clone()
        fast = parts["posterior_fast_values"].clone()
        if str(branch).lower() == "slow":
            slow[:, int(dimension)] = 0.0
        elif str(branch).lower() == "fast":
            fast[:, int(dimension)] = 0.0
        else:
            raise ValueError("SVAE ablation branch must be slow or fast.")
        content, detail, slow_detail, fast_detail = self._latent_to_decoder(slow, fast, target_length)
        return detail, slow_detail, fast_detail

    def infer(self, content_stats, g, mask, pitchf=None, deterministic=True, temperature=1.0):
        prior_features = self._prior_features(content_stats, g, mask, pitchf)
        prior_fast = self._distribution(self.prior_fast(prior_features))
        prior_slow = self._distribution(self.prior_slow(self.prior_slow_down(prior_features)))
        scale = 0.0 if deterministic else max(0.0, float(temperature))
        fast = self._sample(*prior_fast, scale=scale)
        slow = self._sample(*prior_slow, scale=scale)
        content, detail, slow_detail, fast_detail = self._latent_to_decoder(slow, fast, content_stats.shape[-1])
        return content, detail, slow, fast, slow_detail, fast_detail

    def remove_posterior(self) -> None:
        self.posterior_input = None
        self.posterior_condition = None
        self.posterior_blocks = None
        self.posterior_slow_down = None
        self.posterior_slow_blocks = None
        self.posterior_fast = None
        self.posterior_slow = None
        self.posterior_available = False


ChouwaDiscreteLatent = ChouwaContinuousLatent
