"""The per-branch discriminator contract the training loop is written against.

The loop does several things no plain ``forward`` can express: it drives R1 on
one branch at a time under a per-branch strength controller, it logs a loss
series per branch, it computes each real spectrogram once and reuses it across
the paired real/fake forward, and it compiles the paired path while keeping the
double-backward of R1 eager.  All of that is stated here, once, as
``num_branches`` / ``branch_names`` / ``prepare_spectrograms`` /
``forward_branch`` / ``r1_penalty`` / ``enable_compile``.

A subclass only has to build its branches and say how many of them are periods
and how many are spectral; everything below is architecture-agnostic.  It was
extracted from ``RefineGANDiscriminator`` when Wavehax brought a second
discriminator with the same contract and a different network -- the alternative
was a second copy of this file, which would have drifted the moment either one
was touched.

Branch protocol.  Period branches are called with the waveform.  Spectral
branches expose ``spectrogram(audio) -> Tensor`` and
``forward_spectrogram(spectrogram) -> (logits, feature_maps)``, so the
transform can be lifted out of the graph; their plain ``forward`` is just the
composition of the two.  A subclass that sets ``supports_san`` additionally
takes ``san_training`` on both, and returns ``[function, direction]`` in place
of a single logit when it is set -- the base only has to slice both halves the
same way, which is what ``_slice_logit`` is for.

Per-branch driving is optional.  ``branchwise=False`` collapses the whole
contract onto a single branch as far as the trainer can see: one R1 controller,
one ``loss_disc`` series, and an R1 penalty over every branch at once.  Nothing
in the loop changes -- it reads ``num_branches`` and ``branch_names`` and both
report one -- which is the reason the switch lives here and not there.  What it
is *not* is a cheaper R1: the collapsed penalty still touches every branch, it
just spends the whole ``r1_interval`` budget on all of them together instead of
rotating.  Turn it off when a per-branch controller is the thing under
suspicion, or to reproduce a run from before the rotation existed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint


def _slice_logit(value, window: slice):
    """Take a batch slice of a logit that may be a SAN ``[function, direction]``."""
    if isinstance(value, (list, tuple)):
        return [part[window] for part in value]
    return value[window]


class BranchwiseDiscriminator(nn.Module):
    """Base for multi-branch discriminators driven per branch by the trainer.

    Subclasses call :meth:`register_branches` from their ``__init__``.
    """

    uses_branchwise_r1 = True

    #: Whether the *generator* step drives the branches one at a time.  Set by
    #: the trainer, read only by the trainer: nothing in this module's own
    #: forward consumes it, because it decides how a caller walks the branches
    #: rather than what any branch is.
    #:
    #: Deliberately independent of ``branchwise``.  That one changes the shape
    #: of the regularisation -- which branch R1 penalises, how many
    #: ``loss_disc`` series exist, how the R1 strength controller is grouped --
    #: and switching it changes what the run optimises.  This one only decides
    #: whether the generator's adversarial and feature-matching terms are
    #: accumulated branch by branch or in one joint pass; the losses and the
    #: gradients are identical either way, and the only trade is peak memory
    #: against step time.  Tying them together would make a memory decision
    #: silently move the R1 layout, and vice versa.
    generator_branchwise = True

    #: Whether the branches accept ``san_training`` and can return a
    #: ``[function, direction]`` logit pair.  A subclass sets this on the
    #: instance when its heads are actually built with SAN projections; the
    #: default keeps the flag out of the branch signatures entirely.
    supports_san = False

    #: Run every branch in FP32 regardless of autocast.  SAN's projection is a
    #: normalised weight times a learned scale, and in FP16 the normalisation
    #: overflows on the loud frames the discriminator most needs to judge.
    force_fp32 = False

    def register_branches(
        self,
        period_branches: Sequence[nn.Module],
        spectral_branches: Sequence[nn.Module],
        branch_names: Sequence[str],
        use_checkpointing: bool = False,
        branchwise: bool = True,
        waveform_branches: Sequence[nn.Module] = (),
    ) -> None:
        """Register the branches, grouped by what they consume.

        Order is fixed and load-bearing: periods, then everything that owns a
        fixed transform, then any further waveform branch.  ``spectral_branches``
        has to be one contiguous run because ``_spectrogram_index`` maps a branch
        index onto a slot in the precomputed spectrogram list by subtraction.
        ``waveform_branches`` is for anything else that judges the waveform
        directly and therefore has no such slot -- ChouwaGAN's PQMF sub-band
        branch is the one that does.
        """
        self.period_count = len(period_branches)
        self.spectral_count = len(spectral_branches)
        self.use_checkpointing = bool(use_checkpointing)
        self.discriminators = nn.ModuleList(
            list(period_branches) + list(spectral_branches) + list(waveform_branches)
        )
        # Taken from what was actually constructed rather than from a literal
        # list, so a changed ``periods`` or resolution set cannot mislabel a
        # metric series -- which would be invisible, since the series would
        # still be populated and still move.
        names = tuple(branch_names)
        if len(names) != len(self.discriminators):
            raise ValueError(
                f"{len(names)} branch names for {len(self.discriminators)} branches"
            )
        #: The real names, kept whatever ``branchwise`` says, because they name
        #: the modules and not just the metric series.
        self.all_branch_names = names
        self.branchwise = bool(branchwise)
        # Instance attributes shadowing the class default: the trainer reads
        # both off the model, and reporting no names is what turns off the
        # per-branch loss series and the per-branch R1 controller.
        self.uses_branchwise_r1 = self.branchwise
        self.branch_names = names if self.branchwise else ()

    @property
    def num_branches(self) -> int:
        return len(self.discriminators) if self.branchwise else 1

    def branch_parameter_groups(self) -> list[list[nn.Parameter]]:
        """Parameters per R1 branch, in the order ``branch_names`` reports.

        With ``branchwise`` off there is one group holding every branch, so a
        per-branch gradient measurement collapses to the same aggregate the
        single controller is steering.
        """
        if self.branchwise:
            return [list(branch.parameters()) for branch in self.discriminators]
        return [list(self.parameters())]

    def _spectrogram_index(self, index: int) -> Optional[int]:
        """Map a branch index onto its slot in a precomputed spectrogram list."""
        offset = int(index) - self.period_count
        return offset if 0 <= offset < self.spectral_count else None

    def _forward_one(
        self,
        discriminator: nn.Module,
        audio: Tensor,
        spectrogram: Optional[Tensor] = None,
        san_training: bool = False,
    ):
        forward = (
            discriminator.forward_spectrogram
            if spectrogram is not None
            else discriminator
        )
        value = spectrogram if spectrogram is not None else audio
        # A subclass without SAN heads has a branch signature that does not
        # take the flag at all, so it is passed only where it means something.
        # The alternative -- every branch everywhere accepting and discarding it
        # -- puts an argument in three networks to serve one.
        kwargs = {"san_training": san_training} if self.supports_san else {}
        if self.force_fp32:
            value = value.float()
            with torch.autocast(device_type=value.device.type, enabled=False):
                if self.training and self.use_checkpointing:
                    return checkpoint(forward, value, use_reentrant=False, **kwargs)
                return forward(value, **kwargs)
        if self.training and self.use_checkpointing:
            return checkpoint(forward, value, use_reentrant=False, **kwargs)
        return forward(value, **kwargs)

    def prepare_spectrograms(self, audio: Tensor) -> list[Tensor]:
        return [
            discriminator.spectrogram(audio)
            for discriminator in self.discriminators[
                self.period_count : self.period_count + self.spectral_count
            ]
        ]

    def forward_branch(self, audio: Tensor, branch_index: int):
        if not self.branchwise:
            # One "branch" that is all of them: logits concatenated along the
            # flattened axis they already live on, feature maps in branch order.
            logits, feature_maps = self._forward_audio(audio)
            flat_maps = [value for branch in feature_maps for value in branch]
            return torch.cat(logits, dim=1), flat_maps
        return self._forward_one(self.discriminators[int(branch_index)], audio)

    def _forward_audio(self, audio: Tensor, spectrograms=None, san_training=False):
        logits, feature_maps = [], []
        for index, discriminator in enumerate(self.discriminators):
            spectrogram = None
            if spectrograms is not None:
                slot = self._spectrogram_index(index)
                if slot is not None:
                    spectrogram = spectrograms[slot]
            logit, fmap = self._forward_one(
                discriminator, audio, spectrogram, san_training=san_training
            )
            logits.append(logit)
            feature_maps.append(fmap)
        return logits, feature_maps

    def forward(
        self,
        y: Tensor,
        y_hat: Optional[Tensor] = None,
        real_spectrograms=None,
        fake_spectrograms=None,
        branch_index: Optional[int] = None,
        pair_batches: bool = False,
        san_training: bool = False,
    ):
        # ``san_training`` splits every head into a function and a direction
        # output; it is honoured only by a subclass that declares SAN heads, and
        # accepted by the rest so the loop needs no discriminator-specific
        # branch.  Without them every head returns a plain logit and the losses
        # take their LSGAN path.
        san_training = bool(san_training) and self.supports_san

        if branch_index is not None:
            # The R1 path.  Deliberately never SAN: the penalty needs one score
            # to differentiate, and the direction output carries no gradient to
            # the trunk the penalty is meant to constrain.
            return self.forward_branch(y, branch_index)

        if y_hat is None:
            return self._forward_audio(y, real_spectrograms, san_training=san_training)

        if not pair_batches:
            real_logits, real_features = self._forward_audio(
                y, real_spectrograms, san_training=san_training
            )
            fake_logits, fake_features = self._forward_audio(
                y_hat, fake_spectrograms, san_training=san_training
            )
            return real_logits, fake_logits, real_features, fake_features

        # One batched forward over ``cat(real, fake)``: half the kernel launches
        # of two passes, and every branch sees both halves under identical
        # weights.
        if real_spectrograms is None:
            real_spectrograms = self.prepare_spectrograms(y)
        if fake_spectrograms is None:
            fake_spectrograms = self.prepare_spectrograms(y_hat)

        real_batch_size = y.shape[0]
        paired_audio = torch.cat((y, y_hat), dim=0)
        paired_spectrograms = [
            torch.cat((real, fake), dim=0)
            for real, fake in zip(real_spectrograms, fake_spectrograms, strict=True)
        ]
        paired_logits, paired_features = self._forward_audio(
            paired_audio, paired_spectrograms, san_training=san_training
        )
        # A SAN head returns ``[function, direction]`` rather than one tensor,
        # and both halves have to be split the same way.  Slicing the list
        # itself would hand the loss the function output as "real" and the
        # direction output as "fake".
        real_logits = [_slice_logit(value, slice(None, real_batch_size)) for value in paired_logits]
        fake_logits = [_slice_logit(value, slice(real_batch_size, None)) for value in paired_logits]
        real_features = [
            [value[:real_batch_size] for value in branch] for branch in paired_features
        ]
        fake_features = [
            [value[real_batch_size:] for value in branch] for branch in paired_features
        ]
        return real_logits, fake_logits, real_features, fake_features

    def enable_compile(self, mode: str = "default") -> bool:
        """Compile the paired real/fake path without wrapping the module.

        Only :meth:`forward` is compiled.  The R1 penalty reaches its branch
        through :meth:`forward_branch` and differentiates it twice with
        ``create_graph=True``; double-backward through a compiled region is the
        part of this loop most likely to break, and it runs on one branch every
        ``r1_interval`` steps, so there is nothing to win by including it.
        """
        if getattr(self, "_compile_enabled", False):
            return True

        eager_forward = self.forward
        try:
            compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
        except Exception as error:
            from rvc.train.messages import DISCRIMINATOR_COMPILE_ENABLE_FAILED

            print(DISCRIMINATOR_COMPILE_ENABLE_FAILED, str(error))
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            # ``branch_index`` is the R1 path; keep it on the eager callable.
            if (
                not self.training
                or compile_failed
                or kwargs.get("branch_index") is not None
            ):
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                from rvc.train.messages import DISCRIMINATOR_COMPILE_RUNTIME_FAILED

                print(DISCRIMINATOR_COMPILE_RUNTIME_FAILED, str(error))
                return eager_forward(*args, **kwargs)

        self.forward = training_forward
        self._compile_enabled = True
        self._compile_mode = mode
        return True

    def r1_penalty(self, real_audio: Tensor, branch_index: int) -> Tensor:
        """Gradient penalty on the input the branch actually judges.

        For a period branch that is the waveform.  For a spectral branch it is
        the **magnitude spectrogram**, with ``torch.stft`` held outside the
        graph.  R1 exists to keep the *discriminator* from becoming too sharp,
        and a branch can do nothing about the conditioning of a fixed, unlearned
        front end except shrink its own weights toward zero.  The units change
        -- per bin rather than per sample -- which is harmless because the
        per-branch controller normalises against each branch's own
        discriminative gradient and never compares absolute scales across
        branches.  It is also cheaper: the double backward no longer runs
        through the FFT.
        """
        if not self.branchwise:
            return self._global_r1_penalty(real_audio)
        index = int(branch_index)
        source, real_logits = self._r1_source(index, real_audio)
        score = real_logits.float().mean()
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=source,
            create_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().flatten(1).sum(dim=1).mean()

    def _r1_source(self, index: int, real_audio: Tensor):
        """``(input the branch judges, its logits)``, the input made a leaf."""
        slot = self._spectrogram_index(index)
        discriminator = self.discriminators[index]
        if slot is not None:
            source = discriminator.spectrogram(real_audio).detach().requires_grad_(True)
            return source, self._forward_one(
                discriminator, real_audio, spectrogram=source
            )[0]
        source = real_audio.detach().requires_grad_(True)
        return source, self._forward_one(discriminator, source)[0]

    def _global_r1_penalty(self, real_audio: Tensor) -> Tensor:
        """The same penalty, summed over every branch instead of rotating.

        Each branch still contributes the gradient of *its own* input -- the
        waveform for a period branch, the spectrogram for a spectral one -- so
        this is the sum of what the rotation would have measured one branch at a
        time, not a different quantity computed a cheaper way.  In particular
        the fixed transforms stay outside the graph here too; see the branchwise
        docstring above for why differentiating through them is not an option.
        """
        sources, score = [], None
        for index in range(len(self.discriminators)):
            source, real_logits = self._r1_source(index, real_audio)
            sources.append(source)
            branch_score = real_logits.float().mean()
            score = branch_score if score is None else score + branch_score
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=sources,
            create_graph=True,
            only_inputs=True,
        )
        penalty = None
        for gradient in gradients:
            term = gradient.square().flatten(1).sum(dim=1).mean()
            penalty = term if penalty is None else penalty + term
        return penalty

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)
