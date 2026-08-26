import math
import torch
from typing import Optional, List
import random
from torch.nn.utils.parametrize import remove_parametrizations

from rvc.lib.algorithm import generators
from rvc.configs.vocoders import get_vocoder_spec, normalize_vocoder
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.chouwagan_vits import (
    ARCHITECTURE_ID as REFINEGAN_VITS_ARCHITECTURE_ID,
    RefineVitsLatent,
)
from rvc.lib.algorithm.frame_features import (
    conditioning_channels,
    frame_conditioning,
    frame_periodicity,
    template_amplitude,
)
from rvc.lib.terminal import get_console, info, warning
from rvc.train.messages import (
    VOCODER_COMPILE_ENABLE_FAILED,
    VOCODER_COMPILE_RUNTIME_FAILED,
)

# Normalizing Flow
from rvc.lib.algorithm.normalizing_flow import ResidualCouplingBlock
# Text Encoder
from rvc.lib.algorithm.text_encoder import TextEncoder
# Posterior Encoder
from rvc.lib.algorithm.posterior_encoder import PosteriorEncoder


debug_shapes = False

class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        checkpointing: bool = False,
        # Other
        vocoder_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        vocoder_id = normalize_vocoder(vocoder)
        vocoder_spec = get_vocoder_spec(vocoder_id)
        self.segment_size = segment_size
        if not use_f0:
            raise ValueError(f"{vocoder_spec['label']} requires F0 guidance.")

        self.use_f0 = True
        self.vocoder = vocoder_id
        self.sr = sr
        self.architecture_id = "vits_gaussian_v1"

        refinegan_options = {}
        if vocoder_config:
            refinegan_options.update(vocoder_config)
        refinegan_options.update(
            {
                key: value
                for key, value in kwargs.items()
                if key.startswith("refinegan_")
            }
        )

        # ---- Measured frame conditioning and segment sampling -------------
        # All three default to off, so a config that predates them builds the
        # previous model exactly.
        self.use_periodicity = vocoder_id == "refinegan" and bool(
            refinegan_options.get("refinegan_use_periodicity", False)
        )
        self.use_frame_energy = vocoder_id == "refinegan" and bool(
            refinegan_options.get("refinegan_use_frame_energy", False)
        )
        self.segment_energy_floor = (
            float(refinegan_options.get("refinegan_segment_energy_floor", 0.0))
            if vocoder_id == "refinegan"
            else 0.0
        )
        # ``spec_channels`` is ``filter_length // 2 + 1``; the frame features
        # need the FFT size back to turn bin indices into frequencies.
        self.spec_n_fft = 2 * (int(spec_channels) - 1)
        # Must match the decoder's own nominal template amplitude, since it is
        # what the measured envelope is expressed relative to.
        self.template_wave_amp = float(
            refinegan_options.get("refinegan_wave_amp", 0.1)
        )


        # ------   [ Decoder / Vocoder ] Reconstructs audio from latents (z)   ------------------------------------------------------
        dec_kwargs = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "sr": sr,
            "checkpointing": checkpointing,
        }

        dec_kwargs["initial_channel"] = inter_channels
        if vocoder_config:
            dec_kwargs.update(vocoder_config)
        if vocoder_id == "hifi":
            self.dec = generators.HiFiGANNSFGenerator(**dec_kwargs)
        elif vocoder_id == "refinegan":
            if int(sr) != 44100:
                raise ValueError("RefineGAN only supports 44.1 kHz configurations.")
            refinegan_decoder_config = {
                key: dec_kwargs.pop(key)
                for key in list(dec_kwargs)
                if key.startswith("refinegan_")
            }
            refinegan_decoder_config.update(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key.startswith("refinegan_")
                }
            )
            self.dec = generators.RefineGANGenerator(
                **dec_kwargs,
                **refinegan_decoder_config,
            )
        else:
            raise ValueError(f"Unsupported vocoder: {vocoder_id}")

        self.refinegan_latent = None
        if vocoder_id == "refinegan":
            # The architecture id travels with the checkpoint and is checked
            # before loading.  ``net_g`` loads non-strictly, so without it a
            # checkpoint from a different latent layout would load with every
            # latent module silently left at its init.
            self.architecture_id = str(
                refinegan_options.get(
                    "refinegan_architecture_id",
                    REFINEGAN_VITS_ARCHITECTURE_ID,
                )
            )
            self.refinegan_latent = RefineVitsLatent(
                input_channels=inter_channels,
                spec_channels=spec_channels,
                content_channels=int(
                    refinegan_options.get("refinegan_content_channels", 128)
                ),
                detail_channels=int(
                    refinegan_options.get("refinegan_detail_channels", 64)
                ),
                gin_channels=gin_channels,
                posterior_channels=int(
                    refinegan_options.get("refinegan_posterior_channels", 192)
                ),
                prior_hidden_channels=int(
                    refinegan_options.get("refinegan_prior_hidden_channels", 192)
                ),
                latent_channels=int(
                    refinegan_options.get("refinegan_vits_latent_channels", 192)
                ),
                posterior_layers=int(
                    refinegan_options.get("refinegan_vits_posterior_layers", 16)
                ),
                flow_blocks=int(refinegan_options.get("refinegan_vits_flow_blocks", 4)),
                flow_layers=int(refinegan_options.get("refinegan_vits_flow_layers", 4)),
                prior_blocks=int(
                    refinegan_options.get("refinegan_prior_blocks", 4)
                ),
                prior_heads=int(refinegan_options.get("refinegan_prior_heads", 4)),
                prior_kernel_size=int(
                    refinegan_options.get("refinegan_prior_kernel_size", 31)
                ),
                kl_free_bits=float(
                    refinegan_options.get("refinegan_vits_free_bits", 0.03)
                ),
                kl_target=float(refinegan_options.get("refinegan_vits_kl_target", 0.15)),
                kl_beta_lr=float(
                    refinegan_options.get("refinegan_kl_beta_lr", 0.01)
                ),
                kl_beta_min=float(
                    refinegan_options.get("refinegan_kl_beta_min", 1e-4)
                ),
                kl_beta_max=float(
                    refinegan_options.get("refinegan_kl_beta_max", 10.0)
                ),
                kl_rate_momentum=float(
                    refinegan_options.get("refinegan_kl_rate_momentum", 0.01)
                ),
                kl_scale_anchor=float(
                    refinegan_options.get("refinegan_kl_scale_anchor", 1.0)
                ),
                feature_scale_anchor=float(
                    refinegan_options.get("refinegan_feature_scale_anchor", 1.0)
                ),
                content_feature_channels=(
                    int(text_enc_hidden_dim)
                    if bool(
                        refinegan_options.get("refinegan_prior_direct_content", True)
                    )
                    else 0
                ),
                frame_conditioning_channels=conditioning_channels(
                    self.use_periodicity, self.use_frame_energy
                ),
                prior_uses_logs=bool(
                    refinegan_options.get("refinegan_prior_uses_logs", False)
                ),
                prior_replacement_max=float(
                    refinegan_options.get("refinegan_prior_replacement_max", 0.0)
                ),
                prior_replacement_start=int(
                    refinegan_options.get("refinegan_prior_replacement_start", 5000)
                ),
                prior_replacement_ramp=int(
                    refinegan_options.get("refinegan_prior_replacement_ramp", 20000)
                ),
                prior_replacement_mean_share=float(
                    refinegan_options.get(
                        "refinegan_prior_replacement_mean_share", 0.5
                    )
                ),
            )
        info(f"Vocoder: {vocoder_spec['label']}", tag="[INIT]")


        # ------   [ TextEncoder ] Maps extracted features to latent space (p)   ----------------------------------------------------
        self.enc_p = TextEncoder(
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            embedding_dim=text_enc_hidden_dim,
            f0=use_f0,
        )


        # ------   [ Posterior Encoder ] Extracts latents (z) from target audio (training only)   -----------------------------------
        if self.refinegan_latent is None:
            self.enc_q = PosteriorEncoder(
                in_channels=spec_channels,
                out_channels=inter_channels,
                hidden_channels=hidden_channels,
                gin_channels=gin_channels,
                kernel_size=5,
                dilation_rate=1,
                n_layers=16,
            )

            # ------   [ Flow ] Reversible transformation between content priors (p) and speaker-conditioned latents (z)   --------------
            self.flow = ResidualCouplingBlock(
                channels=inter_channels,
                hidden_channels=hidden_channels,
                n_flows=4,
                n_layers=3,
                kernel_size=5,
                dilation_rate=1,
                gin_channels=gin_channels,
            )
        else:
            # The discrete RefineGAN frontend owns its training-only posterior
            # and intentionally has no Gaussian posterior or normalizing flow.
            self.enc_q = None
            self.flow = None


        # ------   [ Speaker Embedding ] Maps identity to global conditioning (g)   -------------------------------------------------
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)


    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for child in list(module.modules()):
            if hasattr(child, "parametrizations") and hasattr(
                child.parametrizations, "weight"
            ):
                remove_parametrizations(child, "weight", leave_parametrized=True)
                continue
            try:
                torch.nn.utils.remove_weight_norm(child)
            except (AttributeError, ValueError):
                pass

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q, self.refinegan_latent]:
            if module is not None:
                self._remove_weight_norm_from(module)

    def _prior_content_stats(
        self, m_p: torch.Tensor, logs_p: torch.Tensor
    ) -> torch.Tensor:
        """Assemble the tensor the RefineGAN prior consumes from ``enc_p``.

        ``enc_p`` projects to ``2 * inter_channels`` and the RefineGAN path used
        to discard the ``logs_p`` half entirely, leaving those weights untrained
        and throwing away the encoder's uncertainty estimate.
        """
        if self.refinegan_latent is not None and getattr(
            self.refinegan_latent, "prior_uses_logs", False
        ):
            return torch.cat((m_p, logs_p), dim=1)
        return m_p

    def _template_amplitude(
        self, conditioning: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Extract RefineGAN's intensity envelope from the frame conditioning.

        Returns ``None`` when frame energy is not measured, which makes the
        template fall back to its constant nominal amplitude -- the vocoder then
        has to infer loudness from the latent, which is the behaviour of every
        implementation that has no intensity input at all.
        """
        if conditioning is None or not self.use_frame_energy:
            return None
        # ``frame_conditioning`` stacks periodicity first when it is enabled, so
        # energy is always the last channel.
        return template_amplitude(conditioning[:, -1:], self.template_wave_amp)

    def set_training_step(self, step: int) -> None:
        """Update the discrete prior replacement schedule for the next batch."""
        if self.refinegan_latent is not None:
            self.refinegan_latent.set_training_step(step)

    def remove_training_modules(self) -> None:
        """Drop training-only posterior modules before exporting/inference."""
        if self.refinegan_latent is not None:
            self.refinegan_latent.remove_posterior()
        self.enc_q = None
        self.flow = None

    def enable_decoder_compile(self, mode: str = "default") -> bool:
        """Compile the selected vocoder's training forward without wrapping it."""
        if self.vocoder not in {"hifi", "refinegan"}:
            return False
        if getattr(self, "_decoder_compile_enabled", False):
            return getattr(self, "_decoder_compile_mode", mode) == mode

        decoder = self.dec
        eager_forward = decoder.forward
        try:
            compiled_forward = torch.compile(
                eager_forward,
                dynamic=False,
                mode=mode,
            )
        except Exception as error:
            get_console().print(
                VOCODER_COMPILE_ENABLE_FAILED,
                str(error),
                style="yellow",
                markup=False,
            )
            return False
        compile_failed = False

        def training_forward(*args, **kwargs):
            nonlocal compile_failed
            if not decoder.training or compile_failed:
                return eager_forward(*args, **kwargs)
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as error:
                compile_failed = True
                get_console().print(
                    VOCODER_COMPILE_RUNTIME_FAILED,
                    str(error),
                    style="yellow",
                    markup=False,
                )
                return eager_forward(*args, **kwargs)

        decoder.forward = training_forward
        self._decoder_compile_enabled = True
        self._decoder_compile_mode = mode
        return True

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        ds: torch.Tensor,
        phone: Optional[torch.Tensor] = None,
        phone_lengths: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            spec (torch.Tensor): Target linear spectrogram.
            spec_lengths (torch.Tensor): Lengths of the target spectrograms.
            ds (torch.Tensor): Speaker embedding.
            phone (torch.Tensor, optional): Contentvec features.
            phone_lengths (torch.Tensor, optional): Lengths of the contentvec features.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            pitch (torch.Tensor, optional): Quantized pitch sequence.
        """
        g = self.emb_g(ds).unsqueeze(-1)

        # Full RVC / VAE-GAN path
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if spec is not None:
            if self.refinegan_latent is not None:
                conditioning = frame_conditioning(
                    spec,
                    pitchf,
                    self.sr,
                    self.spec_n_fft,
                    self.use_periodicity,
                    self.use_frame_energy,
                    x_mask,
                )
                discrete_parts = self.refinegan_latent.forward_train(
                    self._prior_content_stats(m_p, logs_p),
                    spec,
                    g,
                    x_mask,
                    pitchf=pitchf,
                    content=phone.transpose(1, 2),
                    frame_conditioning=conditioning,
                )
                content = discrete_parts["content"]
                detail = discrete_parts["detail"]
                content_slice, ids_slice = rand_slice_segments(
                    content,
                    spec_lengths,
                    self.segment_size,
                    frame_energy=(
                        spec.detach().float().square().mean(dim=1)
                        if self.segment_energy_floor > 0.0
                        else None
                    ),
                    energy_floor=self.segment_energy_floor,
                )
                detail_slice = slice_segments(
                    detail,
                    ids_slice,
                    self.segment_size,
                    dim=3,
                )
                if self.use_f0:
                    pitchf_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                z_slice = torch.cat((content_slice, detail_slice), dim=1)
                amplitude = self._template_amplitude(conditioning)
                if amplitude is not None:
                    amplitude = slice_segments(
                        amplitude, ids_slice, self.segment_size, dim=3
                    )
                o = self.dec(
                    z_slice,
                    pitchf_slice,
                    g=g,
                    template_amplitude=amplitude,
                )
                return o, ids_slice, x_mask, x_mask, discrete_parts

            # Posterior
            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
            # Flow
            z_p = self.flow(z, spec_mask, g=g)

            # Slicing operations
            z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
            if self.use_f0:
                pitchf_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2)

            o = self.dec(z_slice, pitchf_slice, g=g)
            return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            warning(
                "No spectrogram was passed to the forward pass; skipping this "
                "batch.",
                tag="[TRAIN]",
            )
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        seed: int = 0,
        spec: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        temperature: float = 1.0,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Contentvec features.
            phone_lengths (torch.Tensor): Lengths of the contentvec features.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            seed (int, optional): Seed for randomization of noise.
            spec (torch.Tensor, optional): Precomputed spectrogram.
            deterministic (bool, optional): Use argmax FSQ codes and deterministic NSF excitation.
            temperature (float, optional): Sampling temperature for non-deterministic FSQ inference.

        """

        # Seed handler
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Embedding
        g = self.emb_g(sid).unsqueeze(-1)

        # TextEncoder
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if self.refinegan_latent is not None:
            temperature = max(1e-3, float(temperature))
            # Measured from the *source* spectrogram, by the same code the
            # trainer uses.  ``frame_conditioning`` returns None when the
            # caller supplies no spectrogram, and every consumer then falls
            # back to its pre-feature behaviour rather than to zeros.
            conditioning = frame_conditioning(
                spec,
                nsff0,
                self.sr,
                self.spec_n_fft,
                self.use_periodicity,
                self.use_frame_energy,
                x_mask,
            )
            content, detail, _, _, slow_detail, fast_detail = self.refinegan_latent.infer(
                self._prior_content_stats(m_p, logs_p),
                g,
                x_mask,
                pitchf=nsff0,
                deterministic=deterministic,
                temperature=temperature,
                content=phone.transpose(1, 2),
                frame_conditioning=conditioning,
            )
            o = self.dec(
                torch.cat((content, detail), dim=1),
                nsff0,
                g,
                template_amplitude=self._template_amplitude(conditioning),
            )
            return o, x_mask, (content, detail, m_p, logs_p)
        else:
            # Flow
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, nsff0, g)

            return o, x_mask, (z, z_p, m_p, logs_p)
