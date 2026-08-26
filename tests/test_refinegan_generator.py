"""The RefineGAN decoder, pinned against the paper and against 44.1 kHz.

Three groups of properties, each of which the Applio port gets wrong or which
this fork's hop length breaks:

* **Length.** ``hop_length`` here is 441 = 3*3*7*7, so every stage rate is odd.
  The published ``padding=rate // 2`` is only exact for even rates and silently
  drops a frame per stage; the output would still be *a* waveform, just not one
  whose length matches the target.

* **Structure.** The paper downsamples with strided convolutions and ResBlocks
  and upsamples with transposed convolutions.  Applio uses a fixed sinc resample
  plus a stride-1 conv on the way down and ``nn.Upsample(mode="linear")`` on the
  way up -- fixed linear interpolation cannot synthesise anything above the
  previous stage's band, which is precisely the job.

* **Mixed precision.** Nothing about the architecture changes under autocast,
  but the phase accumulator, the intensity exponential and the F0 interpolation
  are pinned to FP32, and the output head drops non-finite activations so one
  overflow cannot reach every loss as a NaN gradient.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the generator needs torch", exc_type=ImportError)

from rvc.lib.algorithm.generators.refinegan import (  # noqa: E402
    ParallelResBlock,
    RefineGANGenerator,
    _decoder_channels,
    _kernel_schedule,
    _per_stage,
)

RATES = (3, 3, 7, 7)
HOP = 441
LATENT = 192
TRAIN_PY = ROOT / "rvc" / "train" / "train.py"


def _lift_from_train_py(name: str):
    """Pull one function out of ``train.py``, which cannot be imported.

    Its module body parses ``sys.argv`` and it imports its siblings by bare
    name, so importing it from a test is not possible. The functions themselves
    are self-contained.
    """
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {"torch": torch}
            exec(compile(ast.Module([node], []), str(TRAIN_PY), "exec"), namespace)
            return namespace[name]
    pytest.fail(f"train.py no longer defines {name}")


def _generator(**overrides) -> RefineGANGenerator:
    torch.manual_seed(0)
    kwargs = dict(
        initial_channel=LATENT,
        gin_channels=256,
        sr=44100,
        upsample_rates=RATES,
    )
    kwargs.update(overrides)
    return RefineGANGenerator(**kwargs)


def _inputs(batch=2, frames=20):
    return (
        torch.randn(batch, LATENT, frames),
        torch.rand(batch, frames) * 300 + 80,
        torch.randn(batch, 256, 1),
    )


# --------------------------------------------------------------------------
# Length


@pytest.mark.parametrize("frames", [8, 20, 41])
def test_the_output_is_exactly_hop_length_samples_per_frame(frames):
    """Odd stage rates make this the failure mode, not a formality."""
    generator = _generator()
    x, f0, g = _inputs(frames=frames)

    assert generator(x, f0, g).shape == (2, 1, frames * HOP)


@pytest.mark.parametrize("rates", [(2, 2, 8, 8), (4, 4, 4, 4), (5, 5, 7, 7)])
def test_even_and_odd_rate_schedules_are_both_exact(rates):
    """The padding rule has to reduce to the published one for even rates."""
    generator = _generator(upsample_rates=rates)
    frames = 12
    x = torch.randn(1, LATENT, frames)
    f0 = torch.full((1, frames), 200.0)

    hop = 1
    for rate in rates:
        hop *= rate
    assert generator(x, f0).shape[-1] == frames * hop


# --------------------------------------------------------------------------
# Structure


def test_the_encoder_downsamples_with_strided_convolutions_and_resblocks():
    """Applio's fixed sinc resample plus stride-1 conv is not this."""
    generator = _generator()

    assert len(generator.downsample_blocks) == len(RATES)
    for block, rate in zip(generator.downsample_blocks, reversed(RATES)):
        conv, resblock = block[0], block[1]
        assert conv.stride == (rate,)
        assert conv.kernel_size == (rate * 2,)
        # Three dilations means three residual sub-blocks, per the paper.
        assert len(resblock.convs1) == 3


def test_the_decoder_upsamples_with_transposed_convolutions():
    """Fixed linear interpolation cannot generate the band it is asked for."""
    generator = _generator()

    assert len(generator.upsample_blocks) == len(RATES)
    for upsample, rate in zip(generator.upsample_blocks, RATES):
        assert isinstance(upsample, torch.nn.ConvTranspose1d)
        assert upsample.stride == (rate,)


def test_every_decoder_stage_meets_the_skip_from_its_own_resolution():
    """The U-Net closes only if the encoder walks the rates in reverse."""
    generator = _generator()

    assert generator.downsample_rates == tuple(reversed(RATES))
    for upsample, block in zip(generator.upsample_blocks, generator.upsample_conv_blocks):
        stage = upsample.out_channels
        assert block.input_conv.weight.shape[1] == stage + stage // 2


def test_the_refinement_blocks_average_three_kernel_sizes():
    generator = _generator()

    for block in generator.upsample_conv_blocks:
        assert isinstance(block, ParallelResBlock)
        assert len(block.blocks) == 3
        assert [b[1].convs1[0].kernel_size[0] for b in block.blocks] == [3, 7, 11]


# --------------------------------------------------------------------------
# Per-stage kernels


def test_a_flat_kernel_list_still_means_the_paper_s_decoder():
    """The default must not change what the paper specifies."""
    assert _kernel_schedule((3, 7, 11), 4) == ((3, 7, 11),) * 4

    generator = _generator()
    assert [len(b.blocks) for b in generator.upsample_conv_blocks] == [3, 3, 3, 3]


def test_a_nested_kernel_list_gives_each_stage_its_own_set():
    """JSON has no tuples, so a configured schedule arrives as nested lists."""
    schedule = _kernel_schedule([[3, 7, 11], [3, 7, 11], [3, 7], [3]], 4)
    assert schedule == ((3, 7, 11), (3, 7, 11), (3, 7), (3,))

    generator = _generator(refinegan_resblock_kernel_sizes=[[3, 7, 11], [3, 7, 11], [3, 7], [3]])
    assert [len(b.blocks) for b in generator.upsample_conv_blocks] == [3, 3, 2, 1]
    # The kernels land on the stage they were configured for, in order.
    per_stage = [
        [sub[1].convs1[0].kernel_size[0] for sub in block.blocks]
        for block in generator.upsample_conv_blocks
    ]
    assert per_stage == [[3, 7, 11], [3, 7, 11], [3, 7], [3]]


def test_a_narrowed_schedule_still_produces_the_exact_output_length():
    """Narrowing must be a cost lever only, never a shape change."""
    generator = _generator(refinegan_resblock_kernel_sizes=[[3, 7, 11], [3, 7, 11], [3, 7], [3]])
    x, f0, g = _inputs()

    assert generator(x, f0, g).shape == (2, 1, 20 * HOP)


def test_a_schedule_of_the_wrong_length_is_rejected():
    """Silently applying three sets to four stages would be a real trap."""
    with pytest.raises(ValueError, match="decoder stages"):
        _kernel_schedule([[3, 7], [3, 7], [3]], 4)


def test_a_stage_cannot_be_left_with_no_kernels():
    with pytest.raises(ValueError, match="at least one kernel"):
        _kernel_schedule([[3, 7], [3, 7], [3], []], 4)


def test_dilations_are_per_stage_too():
    assert _per_stage((1, 3, 5), 4, "dilation") == ((1, 3, 5),) * 4
    assert _per_stage([[1, 3, 5], [1, 3, 5], [1, 3], [1, 3]], 4, "dilation") == (
        (1, 3, 5), (1, 3, 5), (1, 3), (1, 3)
    )
    with pytest.raises(ValueError, match="dilation schedule"):
        _per_stage([[1, 3], [1, 3]], 4, "dilation")


def test_the_default_channel_schedule_is_the_paper_s_halving():
    assert _decoder_channels(None, 512, 4) == (256, 128, 64, 32)
    assert _generator().decoder_channels == (256, 128, 64, 32)


def test_an_explicit_channel_schedule_still_meets_every_skip():
    """With the halving gone, the skip width is no longer half the stage width,
    so the concatenation has to be sized from the encoder rather than assumed."""
    generator = _generator(
        refinegan_decoder_channels=[256, 128, 96, 64],
        refinegan_resblock_kernel_sizes=[[3, 7, 11], [3, 7, 11], [11], [11]],
        refinegan_resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3], [1, 3]],
    )

    assert generator.decoder_channels == (256, 128, 96, 64)
    start = generator.template_conv.out_channels
    stages = len(RATES)
    for index, block in enumerate(generator.upsample_conv_blocks):
        skip = start * 2 ** (stages - 1 - index)
        assert block.input_conv.weight.shape[1] == generator.decoder_channels[index] + skip

    x, f0, g = _inputs()
    assert generator(x, f0, g).shape == (2, 1, 20 * HOP)


def test_a_channel_schedule_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="decoder stages"):
        _decoder_channels([256, 128, 64], 512, 4)


def test_the_shipped_config_spends_width_where_the_card_is_starved():
    """The measured lever: a conv1d reaches 26% of this GPU's throughput at 32
    channels and 79% at 256, so the paper's halving leaves the most expensive
    stage the least efficient one."""
    import json

    cfg = json.loads(
        (ROOT / "rvc" / "configs" / "refinegan" / "44100.json").read_text(encoding="utf-8")
    )
    model = cfg["model"]
    stages = len(model["upsample_rates"])

    channels = model["refinegan_decoder_channels"]
    assert len(channels) == stages
    # The two high-rate stages are the point: wider than a halving would give.
    halving = [512 // 2 ** (i + 1) for i in range(stages)]
    assert channels[2] > halving[2] and channels[3] > halving[3]
    # ...and paid for with one branch instead of three.
    kernels = model["refinegan_resblock_kernel_sizes"]
    assert len(kernels[0]) == 3 and len(kernels[1]) == 3
    assert len(kernels[2]) == 1 and len(kernels[3]) == 1
    assert len(model["refinegan_resblock_dilations"]) == stages


def test_the_decoder_exposes_everything_the_trainer_reaches_for_by_name():
    """``train.py`` touches the decoder's internals directly, not through the
    forward pass, and mostly without ``getattr`` guards.

    ``conv_post`` is the one that bit: the adaptive adversarial weight balances
    the reconstruction and adversarial gradients *at the decoder's last layer*,
    and it reads `model_g.dec.conv_post` with a bare attribute access. Naming it
    `output_conv` raised ``AttributeError`` on the first step of a real run,
    since no test drove that path. HiFi-GAN already used `conv_post`, so this is
    the fork's name for a decoder's output convolution and both vocoders owe it.
    """
    generator = _generator()

    # Read by ``_last_layer_parameter`` for the adaptive adversarial weight.
    assert isinstance(generator.conv_post, torch.nn.Conv1d)
    assert generator.conv_post.out_channels == 1
    # ``_last_layer_parameter`` reaches past weight_norm to the leaf; a missing
    # parametrisation would silently hand back a recomputed tensor and pin the
    # adaptive weight to its lower clamp forever.
    assert "weight" in generator.conv_post.parametrizations

    # Read by ``collect_source_path_metrics``.
    assert len(generator.upsample_conv_blocks) == len(RATES)


def test_the_last_layer_parameter_is_a_graph_leaf_that_takes_gradient():
    """The failure mode ``_last_layer_parameter`` exists to avoid is silent.

    Under ``weight_norm`` the ``weight`` property is recomputed on every access,
    so it is not the leaf the graph was built on and ``autograd.grad`` returns
    ``None`` for it -- which would pin the adaptive adversarial weight to its
    lower clamp for the whole run without raising anything.
    """
    last_layer_parameter = _lift_from_train_py("_last_layer_parameter")
    generator = _generator()
    x, f0, g = _inputs()
    last_layer = last_layer_parameter(generator.conv_post)

    gradient = torch.autograd.grad(
        generator(x, f0, g).square().mean(), last_layer, allow_unused=True
    )[0]

    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert gradient.abs().sum() > 0


def test_the_output_head_is_a_plain_tanh():
    """No soft limiter and no DC blocker: the paper's head, bounded by tanh."""
    generator = _generator()
    x, f0, g = _inputs()

    with torch.no_grad():
        output = generator(x, f0, g)

    assert float(output.abs().max()) < 1.0
    assert not hasattr(generator, "dc_blocker")


# --------------------------------------------------------------------------
# Conditioning


def test_the_speaker_embedding_changes_the_output():
    generator = _generator().eval()
    x, f0, _ = _inputs()

    torch.manual_seed(1)
    with torch.no_grad():
        first = generator(x, f0, torch.zeros(2, 256, 1))
    torch.manual_seed(1)
    with torch.no_grad():
        second = generator(x, f0, torch.ones(2, 256, 1))

    assert not torch.allclose(first, second)


def test_the_intensity_envelope_reaches_the_output():
    """The paper's intensity response has to survive the refinement network."""
    generator = _generator().eval()
    x, f0, g = _inputs()
    quiet = torch.full((2, 1, 20), 0.01)
    loud = torch.full((2, 1, 20), 0.9)

    with torch.no_grad():
        soft = generator(x, f0, g, template_amplitude=quiet)
        hard = generator(x, f0, g, template_amplitude=loud)

    assert not torch.allclose(soft, hard)


def test_a_missing_envelope_falls_back_to_the_nominal_amplitude():
    generator = _generator()
    x, f0, g = _inputs()

    assert generator(x, f0, g).shape == (2, 1, 20 * HOP)


# --------------------------------------------------------------------------
# Mixed precision and export


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_output_and_gradients_stay_finite(dtype):
    generator = _generator().cuda()
    x, f0, g = _inputs()
    x, f0, g = x.cuda(), f0.cuda(), g.cuda()

    with torch.autocast("cuda", dtype=dtype):
        output = generator(x, f0, g)
    assert bool(torch.isfinite(output).all())

    output.float().square().mean().backward()
    gradients = [p.grad for p in generator.parameters() if p.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(value).all()) for value in gradients)


def test_remove_weight_norm_is_idempotent_and_preserves_the_output():
    """Export runs it, and some paths run it twice."""
    generator = _generator().eval()
    x, f0, g = _inputs()

    torch.manual_seed(2)
    with torch.no_grad():
        before = generator(x, f0, g)

    generator.remove_weight_norm()
    generator.remove_weight_norm()

    torch.manual_seed(2)
    with torch.no_grad():
        after = generator(x, f0, g)

    assert torch.allclose(before, after, atol=1e-5)
