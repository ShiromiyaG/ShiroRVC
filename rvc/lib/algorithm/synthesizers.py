import math
import torch
from typing import Optional, List
import random
from torch.nn.utils.parametrize import remove_parametrizations

from rvc.lib.algorithm import generators
from rvc.configs.vocoders import get_vocoder_spec, normalize_vocoder
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.terminal import get_console
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
        use_2_sample_kl: bool = False,
        train_voc_only: bool = False,
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
        self.use_2_sample_kl = use_2_sample_kl
        self.train_voc_only = train_voc_only


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
        elif vocoder_id == "chouwagan":
            if int(sr) != 44100:
                raise ValueError("ChouwaGAN only supports 44.1 kHz configurations.")
            self.dec = generators.ChouwaGANGenerator(**dec_kwargs, **kwargs)
        else:
            raise ValueError(f"Unsupported vocoder: {vocoder_id}")

        get_console().print(
            f"[bold cyan]Vocoder:[/] {vocoder_spec['label']}",
            markup=True,
        )


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
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def enable_decoder_compile(self, mode: str = "default") -> bool:
        """Compile the selected vocoder's training forward without wrapping it."""
        if self.vocoder not in {"hifi", "chouwagan"}:
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
            spec (torch.Tensor): Target spectrogram ( linear-mag spec, or 192-bin mel when train_voc_only ).
            spec_lengths (torch.Tensor): Lengths of the target spectrograms.
            ds (torch.Tensor): Speaker embedding.
            phone (torch.Tensor, optional): Contentvec features. Unused when train_voc_only is enabled.
            phone_lengths (torch.Tensor, optional): Lengths of the contentvec features.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            pitch (torch.Tensor, optional): Quantized pitch sequence.
        """
        g = self.emb_g(ds).unsqueeze(-1)

        # Vocoder-only path
        if self.train_voc_only:
            spec_slice, ids_slice = rand_slice_segments(spec, spec_lengths, self.segment_size)

            if self.use_f0:
                pitchf_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2)

            o = self.dec(spec_slice, pitchf_slice, g=g)

            return o, ids_slice, None, None, (None, None, None, None, None, None, None)

        # Full RVC / VAE-GAN path
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if spec is not None:
            # Posterior
            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
            # Flow
            z_p = self.flow(z, spec_mask, g=g)

            # 2nd sample for KL variance reduction
            z_p2 = None
            if self.use_2_sample_kl:
                z2 = (m_q + torch.randn_like(m_q) * torch.exp(logs_q)) * spec_mask
                z_p2 = self.flow(z2, spec_mask, g=g)

            # Slicing operations
            z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
            if self.use_f0:
                pitchf_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2)


            o = self.dec(z_slice, pitchf_slice, g=g)
            return o, ids_slice, x_mask, spec_mask, (z, z_p, z_p2, m_p, logs_p, m_q, logs_q)
        else:
            get_console().print(" NONE SPEC ")
            return None, None, x_mask, None, (None, None, None, m_p, logs_p, None, None)

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
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Contentvec features. Unused when train_voc_only is enabled.
            phone_lengths (torch.Tensor): Lengths of the contentvec features.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            seed (int, optional): Seed for randomization of noise.
            spec (torch.Tensor, optional): Precomputed 192-bin mel spectrogram ( required when train_voc_only ).

        """

        # Seed handler
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Embedding
        g = self.emb_g(sid).unsqueeze(-1)

        if self.train_voc_only:
            # Vocoder-only path: spec already holds the 192-bin mel, feed it straight to the decoder.
            o = self.dec(spec, nsff0, g)

            return o, None, (None, None, None, None)

        # TextEncoder
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        # Flow
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decoder
        o = self.dec(z * x_mask, nsff0, g)

        return o, x_mask, (z, z_p, m_p, logs_p)
