"""The Wavehax decoder, pinned against upstream and against this fork's 441 hop.

Wavehax is the odd one out among the decoders here: it never upsamples.  The
whole network runs at the frame rate over a ``(channels, bins, frames)`` image
and reaches the waveform once, through an iSTFT.  That moves every property
worth testing onto the analysis/synthesis pair and onto the harmonic prior:

* **Length and reconstruction.**  With ``hop_length = 441`` the output length is
  ``frames * 441`` only if the framing, the window envelope and the padding
  removal agree exactly.  ``n_fft`` that is not a multiple of the hop breaks the
  envelope and colours the output on a frame period, so it is rejected rather
  than tolerated.

* **The prior.**  ``pcph_closed_form`` replaces an explicit sum over the
  harmonic comb with the Dirichlet kernel.  It is the default, so it has to be
  the *same signal* as the explicit sum, not merely a plausible one -- and it
  must stay finite where F0 is zero, which is where the closed form is singular.

* **The decoder contract.**  ``train.py`` and the ablation probe reach into the
  decoder by name and mostly without ``getattr`` guards.  Wavehax has no
  ``conv_post``; the adaptive adversarial weight has to find ``output_proj``
  instead, and it has to find a graph *leaf* there or it pins itself to its
  lower clamp for the whole run without raising anything.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip(
    "torch", reason="the generator needs torch", exc_type=ImportError
)

from rvc.configs.vocoders import (  # noqa: E402
    get_architecture_id,
    get_discriminator_id,
    uses_vits_latent,
)
from rvc.lib.algorithm.synthesizers import vocoder_config_from_model  # noqa: E402
from rvc.lib.algorithm.generators.wavehax import (  # noqa: E402
    STFT,
    WavehaxGenerator,
    generate_pcph,
    generate_pcph_closed_form,
)

RATES = (3, 3, 7, 7)
HOP = 441
LATENT = 192
SR = 44100
CONFIG = ROOT / "rvc" / "configs" / "wavehax" / "44100.json"
TRAIN_PY = ROOT / "rvc" / "train" / "train.py"


def _lift_from_train_py(*names: str):
    """Pull functions out of ``train.py``, which cannot be imported.

    Its module body parses ``sys.argv`` and it imports its siblings by bare
    name, so importing it from a test is not possible.  These functions are
    self-contained apart from each other.
    """
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    namespace: dict = {"torch": torch, "os": os}
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
    missing = wanted - set(namespace)
    assert not missing, f"train.py no longer defines {sorted(missing)}"
    return [namespace[name] for name in names]


def _generator(**overrides):
    settings = {
        "initial_channel": LATENT,
        "sr": SR,
        "upsample_rates": RATES,
        "gin_channels": 256,
        "num_blocks": 2,
    }
    settings.update(overrides)
    return WavehaxGenerator(**settings)


def _inputs(batch=2, frames=24):
    x = torch.randn(batch, LATENT, frames)
    f0 = torch.rand(batch, frames) * 300.0 + 80.0
    g = torch.randn(batch, 256, 1)
    return x, f0, g


# ---------------------------------------------------------------------------
# Length and reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [8, 24, 61])
def test_the_output_is_exactly_one_hop_per_frame(frames):
    """The rest of the pipeline slices audio as ``frames * hop``.

    A decoder that is a sample short still produces *a* waveform, and every
    loss would then compare misaligned tensors -- or fail on a shape, which is
    the lucky case.
    """
    generator = _generator()
    x, f0, g = _inputs(frames=frames)

    assert generator(x, f0, g).shape == (x.shape[0], 1, frames * HOP)


def test_the_stft_pair_reconstructs_the_signal_it_was_given():
    """Everything the decoder does happens between these two calls.

    If the framing, the window envelope or the padding removal disagreed, the
    error would show up as a periodic artefact at the frame rate rather than as
    a failure, so it is pinned directly.
    """
    torch.manual_seed(0)
    stft = STFT(n_fft=2 * HOP, hop_length=HOP)
    signal = torch.randn(2, 1, 40 * HOP)

    real, imag = stft(signal)
    reconstructed = stft.inverse(real, imag)

    assert real.shape == (2, HOP + 1, 40)
    assert reconstructed.shape == signal.shape
    assert torch.allclose(reconstructed, signal, atol=1e-5)


def test_an_n_fft_that_breaks_the_window_envelope_is_rejected():
    """A hop that does not divide ``n_fft`` leaves the Hann envelope
    non-constant, which colours the output once per frame -- audibly, and
    without any error to trace it back to."""
    with pytest.raises(ValueError, match="multiple of the hop"):
        _generator(n_fft=1000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_transforms_survive_autocast(dtype):
    """cuFFT's half-precision path only accepts power-of-two signal sizes.

    This is a hard constraint, not a precision preference, and there is no way
    to satisfy it here: ``n_fft`` must be a multiple of the 441-sample hop, and
    no power of two is.  Under autocast the output projection emits half, and
    ``torch.complex`` of two half tensors is ComplexHalf, so the *inverse*
    transform raised ``cuFFT only supports dimensions whose sizes are powers of
    two`` at the first training step.  The guard lives inside ``STFT`` rather
    than at the call site so the module is correct whoever calls it.

    Runs on CPU too, where the FFT accepts any size and the assertion that
    survives is the dtype contract: the waveform leaves in FP32.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = _generator().to(device)
    x, f0, g = _inputs()
    x, f0, g = x.to(device), f0.to(device), g.to(device)

    with torch.autocast(device_type=device, dtype=dtype):
        output = generator(x, f0, g)

    assert output.dtype is torch.float32
    assert torch.isfinite(output).all()
    assert output.shape == (x.shape[0], 1, x.shape[-1] * HOP)


def test_the_stft_pair_ignores_an_autocast_context():
    """Both directions, called directly rather than through the generator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stft = STFT(n_fft=2 * HOP, hop_length=HOP).to(device)
    signal = torch.randn(2, 1, 8 * HOP, device=device)

    with torch.autocast(device_type=device, dtype=torch.float16):
        real, imag = stft(signal)
        reconstructed = stft.inverse(real, imag)

    assert real.dtype is torch.float32 and imag.dtype is torch.float32
    assert reconstructed.dtype is torch.float32
    assert torch.allclose(reconstructed, signal, atol=1e-4)


def test_the_default_n_fft_is_twice_the_hop():
    """50% overlap, the ratio the published configs use (480 over 240)."""
    generator = _generator()

    assert generator.hop_length == HOP == math.prod(RATES)
    assert generator.n_fft == 2 * HOP
    assert generator.n_bins == HOP + 1


# ---------------------------------------------------------------------------
# The harmonic prior
# ---------------------------------------------------------------------------


def test_the_closed_form_prior_is_the_explicit_harmonic_sum():
    """``pcph_closed_form`` is the shipped default, so it has to be the same
    signal the paper's explicit sum produces, not merely a similar one.

    The Dirichlet identity is exact; the only thing that could diverge is the
    guarded division at the singular points and the phase folding, both of
    which this catches.
    """
    f0 = torch.full((1, 1, 32), 200.0)
    kwargs = dict(
        hop_length=HOP, sample_rate=SR, noise_amplitude=0.0, random_init_phase=False
    )

    explicit = generate_pcph(f0, **kwargs)
    closed_form = generate_pcph_closed_form(f0, **kwargs)

    similarity = torch.nn.functional.cosine_similarity(
        explicit.flatten(), closed_form.flatten(), dim=0
    )
    assert similarity > 0.999
    # Pseudo-*constant power*: the whole point of the envelope is that the RMS
    # does not depend on how many harmonics fit under Nyquist.
    assert closed_form.pow(2).mean().sqrt().item() == pytest.approx(0.1, rel=0.05)


def test_the_prior_is_a_comb_on_the_multiples_of_f0():
    """Wavehax inherits its pitch response from the prior rather than learning
    it, so a comb that lands off the multiples of F0 is a pitch bug in the
    vocoder.

    The envelope is deliberately flat -- every harmonic under Nyquist carries
    the same amplitude -- so the *peak* is not the fundamental and asserting
    that it is only tests the noise.  What has to hold is that the teeth are on
    the multiples and the gaps between them are empty.
    """
    frequency = 220.0
    f0 = torch.full((1, 1, 64), frequency)

    prior = generate_pcph_closed_form(
        f0, hop_length=HOP, sample_rate=SR, noise_amplitude=0.0
    )
    spectrum = torch.fft.rfft(prior[0, 0]).abs()
    resolution = SR / prior.shape[-1]

    def energy_at(hz):
        bin_index = int(round(hz / resolution))
        return spectrum[bin_index - 1 : bin_index + 2].max().item()

    teeth = [energy_at(frequency * k) for k in range(1, 9)]
    gaps = [energy_at(frequency * (k + 0.5)) for k in range(1, 9)]

    assert min(teeth) > 20.0 * max(gaps)
    # Flat envelope: no harmonic is meaningfully louder than the fundamental.
    assert max(teeth) < 2.0 * min(teeth)


def test_an_unvoiced_prior_is_finite_noise_rather_than_a_singularity():
    """The closed form divides by ``sin(theta / 2)``, which vanishes wherever
    F0 does.  Upstream guards it by clamping F0 to 1e-5, which makes the
    harmonic count 2e9 and the phase product exceed FP32 by four orders of
    magnitude; this port gives unvoiced frames one harmonic and masks them."""
    f0 = torch.zeros(1, 1, 16)
    f0[..., 8:] = 150.0

    prior = generate_pcph_closed_form(f0, hop_length=HOP, sample_rate=SR)

    assert torch.isfinite(prior).all()
    unvoiced = prior[..., : 8 * HOP]
    # Noise only, at the configured amplitude -- no harmonic energy leaks in.
    assert unvoiced.abs().max().item() < 0.1


def test_an_entirely_unvoiced_batch_short_circuits_to_noise():
    f0 = torch.zeros(2, 1, 16)

    prior = generate_pcph_closed_form(f0, hop_length=HOP, sample_rate=SR)

    assert prior.shape == (2, 1, 16 * HOP)
    assert torch.isfinite(prior).all()


# ---------------------------------------------------------------------------
# The decoder contract the trainer relies on
# ---------------------------------------------------------------------------


def test_the_trainer_finds_the_output_layer_without_a_conv_post():
    """``train.py`` used to read ``model_g.dec.conv_post`` by bare attribute
    access.  Wavehax emits a complex spectrogram and has no time-domain output
    convolution at all, so the lookup is now by list and ``output_proj`` has to
    be on it."""
    (decoder_output_layer,) = _lift_from_train_py("_decoder_output_layer")
    generator = _generator()

    assert not hasattr(generator, "conv_post")
    layer = decoder_output_layer(generator)
    assert layer is generator.output_proj
    # Real and imaginary planes.
    assert layer.out_channels == 2


def test_the_last_layer_parameter_is_a_graph_leaf_that_takes_gradient():
    """The failure this guards against is silent: if the adaptive adversarial
    weight asks for a gradient that comes back ``None``, it pins itself to its
    lower clamp for the whole run without raising anything."""
    decoder_output_layer, last_layer_parameter = _lift_from_train_py(
        "_decoder_output_layer", "_last_layer_parameter"
    )
    generator = _generator()
    x, f0, g = _inputs()

    last_layer = last_layer_parameter(decoder_output_layer(generator))
    gradient = torch.autograd.grad(
        generator(x, f0, g).square().mean(), last_layer, allow_unused=True
    )[0]

    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_the_source_injection_series_is_reported_for_wavehax_too():
    """``collect_source_path_metrics`` walks ChouwaGAN's decoder stages and
    returns nothing for a decoder that has none.  Wavehax's excitation enters
    once, at ``input_proj``, so the same excitation-versus-latent ratio is
    readable there and the run stays observable on the same axes."""
    (source_path_metrics,) = _lift_from_train_py(
        "_wavehax_source_path_metrics"
    )
    generator = _generator()

    metrics = source_path_metrics(generator)

    assert set(metrics) == {
        "Source/inject_ratio_stage_0",
        "Source/inject_ratio_output_stage",
    }
    assert all(math.isfinite(value) and value > 0.0 for value in metrics.values())


def test_the_speaker_embedding_reaches_the_output():
    """``self.cond`` is RVC's, not ChouwaGAN's and not upstream Wavehax's.

    ``hifigan_nsf`` adds ``self.cond(g)`` at its bottleneck and ``refinegan``
    copied it; upstream Wavehax has no equivalent only because it is a
    single-speaker vocoder.  A decoder here that silently dropped ``g`` would
    train perfectly well and lose speaker identity at the decoder, which is the
    kind of thing that shows up as "the vocoder sounds averaged" many hours in.
    """
    torch.manual_seed(0)
    generator = _generator(prior_type="noise").eval()
    x, f0, g = _inputs()

    assert isinstance(generator.cond, torch.nn.Conv1d)
    assert generator.cond.out_channels == generator.n_bins

    with torch.no_grad():
        torch.manual_seed(1)
        speaker_a = generator(x, f0, g=g)
        torch.manual_seed(1)
        speaker_b = generator(x, f0, g=torch.zeros_like(g))

    assert not torch.allclose(speaker_a, speaker_b, atol=1e-6)


def test_a_decoder_without_global_conditioning_still_builds():
    """``gin_channels=0`` is how the fork spells "no speaker conditioning", and
    every other decoder honours it."""
    generator = _generator(gin_channels=0)
    x, f0, _ = _inputs()

    assert generator.cond is None
    assert torch.isfinite(generator(x, f0)).all()


def test_the_template_amplitude_is_accepted_and_ignored():
    """The synthesizer passes it whenever frame energy is measured, and the
    shipped config measures it.  It belongs to ChouwaGAN's pitch template;
    Wavehax's harmonic prior sets its own amplitude from the F0 contour, so
    there is nothing here for it to scale."""
    torch.manual_seed(0)
    generator = _generator(prior_type="noise").eval()
    x, f0, g = _inputs()
    amplitude = torch.rand(x.shape[0], 1, x.shape[-1])

    with torch.no_grad():
        torch.manual_seed(1)
        plain = generator(x, f0, g=g)
        torch.manual_seed(1)
        with_amplitude = generator(x, f0, g=g, template_amplitude=amplitude)

    assert torch.equal(plain, with_amplitude)


def test_a_two_dimensional_f0_is_accepted_like_the_other_decoders():
    """The synthesizer slices ``pitchf`` to ``(batch, frames)``; upstream
    Wavehax wants ``(batch, 1, frames)``."""
    generator = _generator()
    x, f0, g = _inputs()

    assert torch.isfinite(generator(x, f0.unsqueeze(1), g)).all()


def test_the_output_survives_a_non_finite_spectrogram():
    """One overflowing bin would otherwise reach every loss through the iSTFT
    as a NaN gradient, and there is no adversarial signal in one."""
    generator = _generator()
    x, f0, g = _inputs()
    with torch.no_grad():
        generator.output_proj.bias.fill_(float("inf"))

    assert torch.isfinite(generator(x, f0, g)).all()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_shipped_config_builds_the_decoder_it_names():
    """Also the end-to-end check on unprefixed keys.

    Config keys used to be tagged ``wavehax_*`` and routed by prefix.  Now the
    decoder takes what it names and swallows the rest, which means a *typo* in
    a config key is silently ignored rather than rejected -- so the values the
    decoder actually ended up with are worth asserting, not just that it built.
    """
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]

    generator = WavehaxGenerator(
        sr=config["data"]["sample_rate"],
        initial_channel=model["inter_channels"],
        **vocoder_config_from_model(model),
        **{
            key: model[key]
            for key in ("upsample_rates", "gin_channels")
        },
    )

    assert len(generator.blocks) == model["num_blocks"]
    assert generator.blocks[0].dwconv.kernel_size == tuple(
        model["convnext_kernel_size"]
    )
    assert generator.cond.in_channels == model["gin_channels"]

    assert generator.n_fft == model["n_fft"]
    assert generator.prior_type == model["prior_type"]
    assert len(generator.blocks) == model["num_blocks"]
    # The hop the config declares and the hop the rates imply are the same
    # number in two places; a run where they disagree is unrecoverable.
    assert generator.hop_length == config["data"]["hop_length"]


def test_wavehax_shares_the_vits_frontend_but_brings_its_own_discriminator():
    """The latent frontend is shared; the discriminator is not.

    Everything the trainer does differently for ChouwaGAN -- the discrete
    prior, the R1 controller, the FM balance -- belongs to the frontend, so a
    second decoder over the same latent inherits it.  The discriminator is a
    separate axis, and Wavehax brings UnivNet's rather than ChouwaGAN's.
    Reading both from the registry is what keeps ``vocoder == "chouwagan"``
    from creeping back into the trainer.
    """
    assert uses_vits_latent("wavehax")
    assert uses_vits_latent("chouwagan")
    assert get_discriminator_id("wavehax") == "wavehax"
    assert get_discriminator_id("chouwagan") == "chouwagan"


def test_resuming_from_another_architectures_checkpoint_is_refused(tmp_path):
    """ChouwaGAN and Wavehax share a config shape and a folder layout, so
    pointing a Wavehax run at a folder holding ChouwaGAN checkpoints is one
    edited ``vocoder`` field away.

    The resume path loads the generator *non-strictly*, so nothing downstream
    would complain: the frontend weights match, the decoder stays at its init,
    and the run restarts at the old step count with a fully trained
    discriminator against a random generator.
    """
    (assert_resumable,) = _lift_from_train_py("_assert_resumable_architecture")

    class _Net(torch.nn.Module):
        def __init__(self, architecture_id):
            super().__init__()
            self.architecture_id = architecture_id

    path = tmp_path / "G_121965.pth"
    torch.save({"architecture_id": get_architecture_id("chouwagan")}, path)

    with pytest.raises(ValueError, match="Cannot resume"):
        assert_resumable(_Net(get_architecture_id("wavehax")), path)

    # The matching case, and the pre-id checkpoints, which count as stale.
    assert_resumable(_Net(get_architecture_id("chouwagan")), path) is None
    torch.save({}, path)
    with pytest.raises(ValueError, match="unknown"):
        assert_resumable(_Net(get_architecture_id("wavehax")), path)


def test_the_architecture_id_cannot_be_confused_with_refinegans():
    """``net_g`` loads checkpoints non-strictly: without a distinct id a
    ChouwaGAN checkpoint would load into a Wavehax synthesizer with the whole
    decoder silently left at its initialisation."""
    assert get_architecture_id("wavehax") != get_architecture_id("chouwagan")
    assert get_architecture_id("wavehax") == json.loads(
        CONFIG.read_text(encoding="utf-8")
    )["model"]["architecture_id"]
