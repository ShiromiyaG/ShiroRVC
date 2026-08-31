import inspect
import math
import torch
from typing import Optional, List
import random
from torch.nn.utils.parametrize import remove_parametrizations

from rvc.lib.algorithm import generators
from rvc.configs.vocoders import (
    get_architecture_id,
    get_vocoder_spec,
    normalize_vocoder,
)
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
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


def vocoder_config_from_model(model: dict) -> dict:
    """Split a config's ``model`` block into the vocoder's share of it.

    Config keys used to be tagged by prefix, and the exporter selected them with
    ``startswith("refinegan_")``.  Without the prefix there is nothing to match
    on, so the split is stated once, here, as "everything ``Synthesizer`` does
    not name in its own signature" -- the frontend, decoder and discriminator
    options, which is exactly what lands in ``**kwargs``.

    Derived from the signature rather than listed, because a hand-maintained
    list would silently start exporting -- or silently stop exporting -- a key
    the moment the constructor changed, and the failure is invisible: the
    checkpoint rebuilds with that module left at its default.
    """
    return {key: value for key, value in model.items() if key not in _SYNTHESIZER_KEYS}


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

        # Every model key beyond this constructor's own signature is vocoder
        # configuration -- the frontend's, the decoder's or the discriminator's.
        # It arrives in ``kwargs`` on a training run and in ``vocoder_config``
        # when a checkpoint is rebuilt for inference; the two never overlap, and
        # ``kwargs`` wins if they somehow do.  See ``vocoder_config_from_model``,
        # which is the exporter's side of the same split.
        vocoder_options = {**(vocoder_config or {}), **kwargs}


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
        # Each decoder names the options it understands and swallows the rest,
        # so a config carrying both decoders' keys builds either one.  Anything
        # the config pins that ``dec_kwargs`` also derives is handed over rather
        # than passed twice, which would be a duplicate keyword argument.
        decoder_config = dict(vocoder_options)
        for key in list(decoder_config):
            if key in dec_kwargs:
                dec_kwargs[key] = decoder_config.pop(key)
        generator_id = vocoder_spec["generator"]
        if generator_id == "hifi_nsf":
            self.dec = generators.HiFiGANNSFGenerator(**dec_kwargs)
        elif generator_id == "chouwagan":
            if int(sr) != 44100:
                raise ValueError(
                    f"{vocoder_spec['label']} only supports 44.1 kHz configurations."
                )
            self.dec = generators.ChouwaGANGenerator(**dec_kwargs, **decoder_config)
        elif generator_id == "refinegan":
            # Applio's RefineGAN, unchanged.  It takes neither the ResBlock
            # schedule nor the upsample kernels -- its blocks are fixed at
            # ``(3, 7, 11) x (1, 3, 5)`` and it upsamples by interpolation --
            # so ``dec_kwargs`` is filtered down to what the constructor names
            # rather than passed through.
            #
            # Nothing in this decoder is tied to a rate: the excitation reads
            # ``sample_rate`` and the trunk is built from ``upsample_rates``,
            # whose product is the hop.  So the registry decides which rates
            # ship, rather than a constant here -- ChouwaGAN keeps its own
            # check because *it* really is 44.1 kHz only.
            supported = tuple(int(rate) for rate in vocoder_spec["sample_rates"])
            if int(sr) not in supported:
                raise ValueError(
                    f"{vocoder_spec['label']} supports {supported}, not {int(sr)}."
                )
            self.dec = generators.RefineGANGenerator(
                sample_rate=int(sr),
                downsample_rates=tuple(upsample_rates[::-1]),
                upsample_rates=tuple(upsample_rates),
                upsample_initial_channel=upsample_initial_channel,
                num_mels=inter_channels,
                start_channels=int(
                    decoder_config.get("refinegan_start_channels", 16)
                ),
                leaky_relu_slope=float(
                    decoder_config.get("refinegan_leaky_relu_slope", 0.2)
                ),
                gin_channels=gin_channels,
                checkpointing=checkpointing,
            )
        else:
            raise ValueError(f"Unsupported vocoder: {vocoder_id}")

        # ``net_g`` loads non-strictly, so the id is the only thing that stops
        # a checkpoint built for a different decoder layout from loading with
        # the mismatched modules left at their init.
        self.architecture_id = str(
            vocoder_options.get(
                "architecture_id",
                get_architecture_id(vocoder_id, vocoder_options),
            )
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
            if module is not None:
                self._remove_weight_norm_from(module)

    def remove_training_modules(self) -> None:
        """Drop training-only posterior modules before exporting/inference.

        Only the *posterior* is training-only.  On the Gaussian path the flow is
        what inference runs in reverse to get ``z`` from ``z_p``, so dropping it
        here would leave the decoder with nothing to decode.
        """
        self.enc_q = None

    def enable_decoder_compile(self, mode: str = "default") -> bool:
        """Compile the selected vocoder's training forward without wrapping it."""
        if self.vocoder not in {"hifi", "chouwagan", "refinegan"}:
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

        """

        # Seed handler
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Embedding
        g = self.emb_g(sid).unsqueeze(-1)

        # TextEncoder
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        # Flow
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g)

        return o, x_mask, (z, z_p, m_p, logs_p)


#: Everything ``Synthesizer.__init__`` consumes by name.  Anything else a config
#: names is vocoder configuration; see ``vocoder_config_from_model``.
_SYNTHESIZER_KEYS = frozenset(
    name
    for name, parameter in inspect.signature(Synthesizer.__init__).parameters.items()
    if parameter.kind is not inspect.Parameter.VAR_KEYWORD and name != "self"
)
