"""VITS-style flow latent frontend, shared by every decoder in this fork.

It exposes ``forward_train`` / ``prior_losses`` / ``diagnostics`` / ``infer`` /
``remove_posterior``, which is the contract the decoder, the discriminator, the
balance controllers and every metric series are written against.  ``content``
and ``detail`` are concatenated into one frame-rate conditioning sequence; what
the decoder does with it is the decoder's business.  RefineGAN refines a pitch
template against it, Wavehax projects it into the frequency axis of a complex
spectrogram, and neither choice reaches anything in this file.

This is RVC's VITS posterior-plus-flow, with the four modifications that the
measurements in this repo argued for:

1. **Prior replacement.**  Vanilla VITS trains the decoder exclusively on
   posterior ``z`` and runs it on ``flow^-1(prior sample)``.  A stronger vocoder
   therefore improves the *training* path without touching the inference one and
   the gap widens -- which is what "the vocoder is too strong" actually is.  A
   scheduled fraction of the batch is decoded from the prior instead, keeping
   the graph, so reconstruction gradient reaches the flow and ``enc_p``.

2. **Direct ContentVec path.**  ``enc_p`` opens with a single
   ``Linear(768, 192)`` applied before any nonlinearity.  Measured on this
   dataset's extracted features, the best possible rank-192 projection retains
   88.7% of their variance, so ~11% is destroyed before any module sees it and
   no downstream capacity recovers it.  The prior reads the raw features too.

3. **A frequency-aware posterior input.**  ``PosteriorEncoder.pre`` is a
   ``Conv1d(1025, 192, kernel_size=1)``: one matrix collapsing every bin before
   the WaveNet sees them, with no notion that neighbouring bins are neighbours.

4. **Rate-targeted KL.**  ``c_kl = 1.0`` with no floor and no target lets the
   latent carry unlimited information about the target, which is exactly what
   the decoder learns to lean on and then does not find at inference.  The
   per-dimension rate controller below holds it at a target instead; set
   ``kl_target = 0`` for plain VITS behaviour.

The single latent branch is called "fast" throughout -- ``fast_to_content``,
``kl_fast_per_dim``, ``posterior_fast_head`` -- because train.py groups
parameters and names metric series off exactly those prefixes.  The "slow"
slots stay empty, so the shared logging simply omits them.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_ids,
    uses_vits_latent,
)
from rvc.lib.algorithm.normalizing_flow import ResidualCouplingBlock
from rvc.lib.algorithm.wavenet import WaveNet

# ``net_g`` loads checkpoints non-strictly, so the architecture id is the only
# thing standing between an incompatible checkpoint and a silently half-random
# model.  It identifies the latent *and* the decoder that consumes it, so it
# lives per vocoder in ``rvc/configs/vocoders.json`` rather than here -- a
# second copy of the string in this file could only ever drift out of step with
# the one the trainer, the exporter and the loader all read.
ARCHITECTURE_ID = get_architecture_id("chouwagan")
SUPPORTED_ARCHITECTURE_IDS = frozenset(
    get_architecture_id(vocoder_id)
    for vocoder_id in get_vocoder_ids()
    if uses_vits_latent(vocoder_id)
)

# Bounds on the predicted log standard deviations.  These exist to stop a bad
# init from producing inf/nan, not to shape the solution -- a run that *sits* on
# either bound has lost the gradient on its variance head, because ``clamp``
# passes nothing through outside its range.  The real defence against drift is
# ``kl_scale_anchor``; this is only the guard rail.
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


class SpectrogramFrequencyStem(nn.Module):
    """Read a log-magnitude spectrogram along frequency before flattening it.

    The posterior used to enter through a single ``Conv1d(spec_channels,
    posterior_channels, 1)``: one matrix collapsing all 1025 bins into 192
    channels before anything looked at them.  A kernel of width 1 over the
    channel axis has no notion that bin *k* and bin *k+1* are neighbours, so a
    harmonic comb -- the one structure the posterior most needs to describe --
    arrives as an arbitrary permutation of coordinates and has to be relearned
    as one.

    Striding down the frequency axis with a real kernel keeps that adjacency.
    The stack is deliberately cheap (1 -> 16 -> 32 -> 32 channels, frequency
    quartered twice then halved) because it runs on every training step and is
    then thrown away: ``remove_posterior`` deletes it before export, so none of
    this cost or capacity reaches inference.
    """

    def __init__(self, spec_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, (7, 3), stride=(4, 1), padding=(3, 1)),
                nn.Conv2d(16, 32, (5, 3), stride=(4, 1), padding=(2, 1)),
                nn.Conv2d(32, 32, (5, 3), stride=(2, 1), padding=(2, 1)),
            ]
        )
        bins = int(spec_channels)
        for stride in (4, 4, 2):
            bins = (bins + stride - 1) // stride
        self.flattened_channels = 32 * bins
        self.project = nn.Conv1d(self.flattened_channels, int(out_channels), 1)

    def forward(self, spectrogram: Tensor, mask: Optional[Tensor]) -> Tensor:
        # ``(batch, 1, freq, time)``; the mask is over time only and broadcasts
        # across every remaining frequency row.
        value = spectrogram.unsqueeze(1)
        frame_mask = None if mask is None else mask.unsqueeze(1)
        for layer in self.layers:
            value = F.silu(layer(value))
            if frame_mask is not None:
                value = value * frame_mask
        batch, channels, bins, length = value.shape
        return self.project(value.reshape(batch, channels * bins, length))


class RefineVitsLatent(nn.Module):
    """Posterior + normalizing flow latent frontend, decoder-agnostic.

    The name is kept because ``Synthesizer.refinegan_latent`` is a checkpoint
    key prefix: renaming it would orphan every weight under it.
    """

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
        latent_channels: int = 192,
        posterior_layers: int = 16,
        posterior_kernel_size: int = 5,
        posterior_dilation_rate: int = 1,
        flow_blocks: int = 4,
        flow_layers: int = 4,
        flow_kernel_size: int = 5,
        prior_blocks: int = 4,
        prior_heads: int = 4,
        prior_kernel_size: int = 31,
        kl_free_bits: float = 0.03,
        kl_target: float = 0.15,
        kl_beta_lr: float = 0.01,
        kl_beta_min: float = 1e-4,
        kl_beta_max: float = 10.0,
        kl_rate_momentum: float = 0.01,
        kl_scale_anchor: float = 1.0,
        feature_scale_anchor: float = 1.0,
        content_feature_channels: int = 0,
        frame_conditioning_channels: int = 0,
        prior_uses_logs: bool = False,
        prior_replacement_max: float = 0.0,
        prior_replacement_start: int = 5000,
        prior_replacement_ramp: int = 20000,
        prior_replacement_mean_share: float = 0.5,
        checkpointing: bool = False,
        **_: object,
    ):
        super().__init__()
        self.architecture_id = ARCHITECTURE_ID
        # Recompute the prior stack and the posterior encoder in the backward
        # pass instead of storing their activations.  Separate from the
        # decoder's ``checkpointing`` flag on purpose: the two stacks have
        # different shapes -- the decoder runs at the sample rate, this runs at
        # 100 Hz over ``segment_size / hop_length`` frames -- so the trade of
        # one extra forward against the saved activations is not the same
        # decision, and the profitable one depends on which stack the run is
        # actually short on.
        self.checkpointing = bool(checkpointing)
        self.input_channels = int(input_channels)
        self.spec_channels = int(spec_channels or input_channels)
        self.content_channels = int(content_channels)
        self.detail_channels = int(detail_channels)
        self.latent_channels = int(latent_channels)

        # One branch.  The empty ``slow_levels`` keeps the shared ablation
        # sampler and the shared logging from ever selecting a slow dimension.
        self.slow_levels: Tuple[int, ...] = ()
        self.fast_levels = tuple(range(self.latent_channels))
        self.slow_latent_channels = 0
        self.fast_latent_channels = self.latent_channels

        self.kl_free_bits_fast = float(kl_free_bits)
        self.kl_free_bits_slow = float(kl_free_bits)
        self.kl_target_fast = max(0.0, float(kl_target))
        self.kl_target_slow = 0.0
        self.kl_beta_lr = max(0.0, float(kl_beta_lr))
        self.kl_beta_min = max(1e-8, float(kl_beta_min))
        self.kl_beta_max = max(self.kl_beta_min, float(kl_beta_max))
        self.kl_rate_momentum = min(1.0, max(1e-6, float(kl_rate_momentum)))
        self.kl_scale_anchor = max(0.0, float(kl_scale_anchor))
        self.feature_scale_anchor = max(0.0, float(feature_scale_anchor))

        self.register_buffer(
            "kl_log_beta_fast", torch.zeros(self.latent_channels), persistent=True
        )
        self.register_buffer(
            "kl_rate_ema_fast",
            torch.full((self.latent_channels,), float(self.kl_target_fast)),
            persistent=True,
        )

        self.training_step = 0
        self.current_prior_replacement = 0.0

        self.prior_replacement_max = min(1.0, max(0.0, float(prior_replacement_max)))
        self.prior_replacement_start = max(0, int(prior_replacement_start))
        self.prior_replacement_ramp = max(1, int(prior_replacement_ramp))
        self.prior_replacement_mean_share = min(
            1.0, max(0.0, float(prior_replacement_mean_share))
        )

        # ---- Prior ---------------------------------------------------------
        self.prior_uses_logs = bool(prior_uses_logs)
        prior_input_channels = self.input_channels * (2 if self.prior_uses_logs else 1)
        self.prior_input = nn.Conv1d(prior_input_channels, int(prior_hidden_channels), 1)
        self.prior_feature_channels = int(prior_hidden_channels)
        self.content_feature_channels = max(0, int(content_feature_channels))
        self.prior_content = (
            nn.Conv1d(self.content_feature_channels, int(prior_hidden_channels), 1)
            if self.content_feature_channels
            else None
        )
        self.frame_conditioning_channels = max(0, int(frame_conditioning_channels))
        self.prior_frame = (
            nn.Conv1d(self.frame_conditioning_channels, int(prior_hidden_channels), 1)
            if self.frame_conditioning_channels
            else None
        )
        self.prior_f0 = nn.Conv1d(2, int(prior_hidden_channels), 1)
        self.prior_speaker = nn.Conv1d(int(gin_channels), int(prior_hidden_channels), 1)

        self.prior_blocks = nn.ModuleList(
            [
                EBranchformerBlock(prior_hidden_channels, prior_heads, prior_kernel_size)
                for _ in range(int(prior_blocks))
            ]
        )
        self.prior_fast_head = nn.Conv1d(
            int(prior_hidden_channels), self.latent_channels * 2, 1
        )

        # ---- Flow ----------------------------------------------------------
        # Kept at inference: the prior is over ``z_p`` and the decoder consumes
        # ``z``, so this is not a training-only module and must not be stripped
        # by ``extract_model`` -- which is why it is not named ``posterior_*``.
        self.flow = ResidualCouplingBlock(
            channels=self.latent_channels,
            hidden_channels=int(posterior_channels),
            n_flows=int(flow_blocks),
            n_layers=int(flow_layers),
            kernel_size=int(flow_kernel_size),
            dilation_rate=1,
            gin_channels=int(gin_channels),
        )

        # ---- Posterior (training only) --------------------------------------
        self.posterior_input = SpectrogramFrequencyStem(
            self.spec_channels, int(posterior_channels)
        )
        self.posterior_condition = nn.Conv1d(
            self.prior_feature_channels, int(posterior_channels), 1
        )
        self.posterior_enc = WaveNet(
            hidden_channels=int(posterior_channels),
            kernel_size=int(posterior_kernel_size),
            dilation_rate=int(posterior_dilation_rate),
            n_layers=int(posterior_layers),
            gin_channels=int(gin_channels),
        )
        self.posterior_fast_head = nn.Conv1d(
            int(posterior_channels), self.latent_channels * 2, 1
        )
        self.posterior_available = True

        # ---- Latent -> decoder ----------------------------------------------
        self.fast_to_content = nn.Conv1d(self.latent_channels, self.content_channels, 1)
        self.fast_to_detail = nn.Conv1d(self.latent_channels, self.detail_channels, 1)

    # -- shared helpers ------------------------------------------------------

    def _maybe_checkpoint(self, function, *args):
        """Run ``function`` under activation checkpointing where it pays.

        Only in training and only with grad enabled: ``infer`` and every
        ``no_grad`` diagnostic store nothing to begin with, so recomputing
        there would be pure cost.  ``use_reentrant=False`` is required rather
        than preferred -- the reentrant implementation loses the autocast state
        and cannot handle the ``None`` mask these callables take.
        """
        if not (self.checkpointing and self.training and torch.is_grad_enabled()):
            return function(*args)
        return checkpoint(function, *args, use_reentrant=False)

    @staticmethod
    def _distribution(value: Tensor) -> Tuple[Tensor, Tensor]:
        mean, logs = value.chunk(2, 1)
        return mean, logs.clamp(LOGS_MIN, LOGS_MAX)

    @staticmethod
    def _sample(mean: Tensor, logs: Tensor, scale=1.0) -> Tensor:
        if not torch.is_tensor(scale):
            scale = float(scale)
        return mean + torch.randn_like(mean) * logs.exp() * scale

    def set_training_step(self, step: int) -> None:
        self.training_step = max(0, int(step))

    def prior_replacement_fraction(self) -> float:
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
        """Split the replaced items between the prior mean and prior samples.

        ``infer`` is deterministic and consumes the prior *mean*, which is
        systematically less energetic than a sample; a decoder trained only on
        samples pays for being evaluated somewhere it never trained.  Scale 0 is
        the deterministic operating point, scale 1 keeps the sampling
        regulariser and is the only thing that puts reconstruction gradient on
        the prior's ``logs``.
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

    # -- the two encoders ----------------------------------------------------

    def _prior_features(
        self,
        content_stats: Tensor,
        g: Optional[Tensor],
        mask: Optional[Tensor],
        pitchf: Optional[Tensor],
        content: Optional[Tensor] = None,
        frame_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        value = self.prior_input(content_stats)
        if self.prior_content is not None and content is not None:
            value = value + self.prior_content(
                _resize_sequence(content.float(), value.shape[-1])
            )
        if self.prior_frame is not None and frame_conditioning is not None:
            value = value + self.prior_frame(
                _resize_sequence(frame_conditioning.float(), value.shape[-1])
            )
        if pitchf is None:
            pitchf = content_stats.new_zeros(
                content_stats.shape[0], content_stats.shape[-1]
            )
        if pitchf.ndim == 3:
            f0 = pitchf[:, :2]
        else:
            f0 = torch.stack(
                (
                    torch.log1p(pitchf.clamp_min(0)) / 7.0,
                    (pitchf > 0).to(content_stats.dtype),
                ),
                1,
            )
        value = value + self.prior_f0(_resize_sequence(f0, content_stats.shape[-1]))
        if g is not None:
            value = value + self.prior_speaker(g).expand(-1, -1, value.shape[-1])
        for block in self.prior_blocks:
            value = self._maybe_checkpoint(block, value, mask)
        return value if mask is None else value * mask

    def _posterior_distribution(self, spec, condition, g, mask):
        target = torch.log1p(spec.float().clamp_min(0.0))
        target_mask = _resize_mask(mask, target.shape[-1])
        value = self._maybe_checkpoint(self.posterior_input, target, target_mask)
        value = value + self.posterior_condition(
            _resize_sequence(condition.detach(), target.shape[-1])
        )
        if target_mask is not None:
            value = value * target_mask
        ones = (
            torch.ones_like(value[:, :1])
            if target_mask is None
            else target_mask.to(value.dtype)
        )
        # ``g`` is keyword-only at the call site but positional in the
        # signature; checkpointing takes positional arguments, and the WaveNet
        # accepts it there.
        value = self._maybe_checkpoint(self.posterior_enc, value, ones, g)
        stats = self.posterior_fast_head(value)
        if target_mask is not None:
            stats = stats * target_mask
        return self._distribution(stats), target_mask

    def _latent_to_decoder(self, z: Tensor, target_length: int):
        resized = _resize_sequence(z, target_length)
        content = self.fast_to_content(resized)
        detail = self.fast_to_detail(resized)
        # One branch, so the slow half of the decoder's detail contract is the
        # zero tensor rather than a second projection.
        return content, detail, torch.zeros_like(detail), detail

    # -- the training pass ---------------------------------------------------

    def forward_train(
        self,
        content_stats,
        spec,
        g,
        mask,
        pitchf=None,
        content=None,
        frame_conditioning=None,
    ):
        device_type = content_stats.device.type

        prior_features = self._prior_features(
            content_stats, g, mask, pitchf, content, frame_conditioning
        )
        # Distribution statistics (exp, log, sampling) and the normalizing flow
        # need FP32 precision; the conv blocks above stay in the caller's dtype.
        with torch.autocast(device_type=device_type, enabled=False):
            prior = self._distribution(self.prior_fast_head(prior_features.float()))

        if not self.posterior_available:
            raise RuntimeError("The training-only VITS posterior is unavailable.")
        posterior, posterior_mask = self._posterior_distribution(
            spec, prior_features, g, mask
        )
        with torch.autocast(device_type=device_type, enabled=False):
            length = prior[0].shape[-1]
            posterior = tuple(_resize_sequence(value.float(), length) for value in posterior)

            z = self._sample(*posterior)
            flow_mask = (
                torch.ones_like(z[:, :1])
                if mask is None
                else _resize_mask(mask, z.shape[-1]).to(z.dtype)
            )
            z_p = self.flow(z, flow_mask, g=g.float() if g is not None else None)

            replacement_fraction = self.prior_replacement_fraction()
            selector = self._replacement_selector(
                z.shape[0], replacement_fraction, z.device, z.dtype
            )
            if selector is not None:
                scales = self._replacement_scales(selector)
                prior_sample = self._sample(*prior, scale=scales)
                prior_z = self.flow(
                    prior_sample, flow_mask,
                    g=g.float() if g is not None else None, reverse=True,
                )
                z = torch.lerp(z, prior_z, selector)
                prior_replacement = selector.mean().detach()
                prior_replacement_mean = (
                    (selector - scales).mean().detach()
                    / selector.mean().clamp_min(1e-6)
                )
            else:
                prior_replacement = z.new_zeros(())
                prior_replacement_mean = z.new_zeros(())

        content_out, detail, slow_detail, fast_detail = self._latent_to_decoder(
            z, content_stats.shape[-1]
        )
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=False):
                _, prior_detail, _, _ = self._latent_to_decoder(
                    self.flow(
                        prior[0], flow_mask,
                        g=g.float() if g is not None else None, reverse=True,
                    ),
                    content_stats.shape[-1],
                )

        return {
            "content": content_out,
            "detail": detail,
            "detail_slow": slow_detail,
            "detail_fast": fast_detail,
            "posterior_fast_distribution": posterior,
            "prior_fast_distribution": prior,
            "posterior_z_p": z_p,
            "prior_detail": prior_detail,
            "posterior_fast_values": z,
            "prior_replacement": prior_replacement,
            "prior_replacement_mean": prior_replacement_mean,
        }

    # -- losses --------------------------------------------------------------

    @staticmethod
    def _kl(z_p, posterior, prior, free_bits: float, mask: Optional[Tensor] = None):
        """Per-dimension KL of the flow-transformed posterior against the prior.

        The VITS form: a Monte Carlo estimate at the sampled ``z_p`` rather than
        the closed-form Gaussian pair, because the flow makes the pushed-forward
        posterior non-Gaussian.
        """
        mu_q, logs_q = (value.float() for value in posterior)
        mu_p, logs_p = (value.float() for value in prior)
        z_p = z_p.float()
        value = logs_p - logs_q - 0.5
        value = value + 0.5 * ((z_p - mu_p).square()) * (-2.0 * logs_p).exp()
        if mask is None:
            dimensions = value.mean(dim=(0, 2))
        else:
            valid = _resize_mask(mask, value.shape[-1]).float()
            dimensions = (value * valid).sum(dim=(0, 2)) / valid.sum().clamp_min(1.0)
        return dimensions.clamp_min(float(free_bits)), dimensions.detach()

    @property
    def kl_rate_control_active(self) -> bool:
        return self.kl_target_fast > 0.0

    def _kl_beta(self, rate: Tensor, target: float) -> Tensor:
        log_beta = self.kl_log_beta_fast
        if target <= 0.0:
            return log_beta.exp()
        if self.training:
            with torch.no_grad():
                self.kl_rate_ema_fast.mul_(1.0 - self.kl_rate_momentum).add_(
                    self.kl_rate_momentum * rate
                )
                error = ((self.kl_rate_ema_fast - target) / target).clamp(-1.0, 1.0)
                log_beta.add_(self.kl_beta_lr * error).clamp_(
                    math.log(self.kl_beta_min), math.log(self.kl_beta_max)
                )
        return log_beta.exp()

    def _scale_anchor(self, parts: Dict[str, Tensor]) -> Tensor:
        distributions = (
            parts["prior_fast_distribution"],
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

    def prior_losses(self, parts: Dict[str, Tensor], mask: Optional[Tensor]):
        clamped, dims = self._kl(
            parts["posterior_z_p"],
            parts["posterior_fast_distribution"],
            parts["prior_fast_distribution"],
            self.kl_free_bits_fast,
            mask,
        )
        parts["kl_fast_per_dim"] = dims
        anchor = self._scale_anchor(parts)
        parts["scale_anchor"] = anchor.detach()

        zero = dims.new_zeros(())
        if not self.kl_rate_control_active:
            parts["kl_beta_fast"] = dims.new_ones(())
            total_fast = clamped.sum()
            return zero, total_fast, total_fast + anchor

        beta = self._kl_beta(dims, self.kl_target_fast).detach()
        parts["kl_beta_fast"] = beta.mean()
        loss_fast = (clamped * beta).sum()
        return zero, loss_fast, loss_fast + anchor

    @staticmethod
    def _kl_effective_dims(per_dim: Tensor) -> Tensor:
        value = per_dim.clamp_min(0.0)
        return value.sum().square() / value.square().sum().clamp_min(1e-12)

    def diagnostics(self, parts, mask):
        with torch.no_grad():
            fast_kl = parts.get(
                "kl_fast_per_dim",
                parts["posterior_fast_distribution"][0].new_zeros(self.latent_channels),
            )
            return {
                "kl_effective_dims_fast": self._kl_effective_dims(fast_kl),
                "kl_above_floor_fast": (fast_kl > self.kl_free_bits_fast).float().mean(),
                "kl_median_fast": fast_kl.median(),
                "prior_kl_fast": fast_kl.mean(),
                "prior_std_fast": parts["prior_fast_distribution"][1].exp().mean(),
                "posterior_std_fast": parts["posterior_fast_distribution"][1]
                .exp()
                .mean(),
                "scale_anchor": parts.get("scale_anchor", fast_kl.new_zeros(())),
                "prior_replacement": parts["prior_replacement"],
                "prior_replacement_mean": parts.get(
                    "prior_replacement_mean", fast_kl.new_zeros(())
                ),
                "kl_beta_fast": parts.get("kl_beta_fast", fast_kl.new_ones(())),
                "content_rms": parts["content"].float().square().mean().sqrt(),
                "posterior_detail_rms": parts["detail"].float().square().mean().sqrt(),
                "prior_detail_rms": parts["prior_detail"].float().square().mean().sqrt(),
                "kl_fast": fast_kl.mean(),
                "kl_fast_per_dim": fast_kl,
            }

    def ablate_dimension(self, parts, branch: str, dimension: int, target_length: int):
        if str(branch).lower() != "fast":
            raise ValueError("The VITS frontend has a single latent branch.")
        z = parts["posterior_fast_values"].clone()
        z[:, int(dimension)] = 0.0
        _, detail, slow_detail, fast_detail = self._latent_to_decoder(z, target_length)
        return detail, slow_detail, fast_detail

    # -- inference -----------------------------------------------------------

    def infer(
        self,
        content_stats,
        g,
        mask,
        pitchf=None,
        deterministic=True,
        temperature=1.0,
        content=None,
        frame_conditioning=None,
    ):
        prior_features = self._prior_features(
            content_stats, g, mask, pitchf, content, frame_conditioning
        )
        prior = self._distribution(self.prior_fast_head(prior_features))
        scale = 0.0 if deterministic else max(0.0, float(temperature))
        z_p = self._sample(*prior, scale=scale)
        flow_mask = (
            torch.ones_like(z_p[:, :1])
            if mask is None
            else _resize_mask(mask, z_p.shape[-1]).to(z_p.dtype)
        )
        z = self.flow(z_p, flow_mask, g=g, reverse=True)
        content_out, detail, slow_detail, fast_detail = self._latent_to_decoder(
            z, content_stats.shape[-1]
        )
        return content_out, detail, z, z, slow_detail, fast_detail

    def remove_posterior(self) -> None:
        self.posterior_input = None
        self.posterior_condition = None
        self.posterior_enc = None
        self.posterior_fast_head = None
        self.posterior_available = False
