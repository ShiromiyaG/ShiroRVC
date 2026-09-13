"""Stage 0 of a staged pretrain: the decoder alone, mel -> wav.

WHAT THIS RUNS
--------------
``dec`` + the discriminator, and nothing else.  No ``enc_p``, no ``enc_q``, no
``flow``, no ``emb_g``.  The decoder reads a mel spectrogram computed from the
target audio and is judged on reproducing that audio.

That is a vocoder pretrain in the BigVGAN/HiFi-GAN sense, which
``pretrain_stage1_vocoder.py`` is *not*: stage 1 trains the autoencoder
``spec -> enc_q -> z -> dec -> audio``, where the decoder's input is a learned
192-channel ``z`` and not a mel.

HOW THIS FITS THE THREE STAGES -- READ THIS BEFORE USING IT
-----------------------------------------------------------
Stage 0 is an *initialisation* for stage 1.  It is not a stage that hands a
frozen decoder to the ones after it.

The decoder RVC trains reads ``z`` (``inter_channels``, learned); the decoder
here reads a mel (``n_mel_channels``, fixed).  Only the input projection
(``mel_conv`` on RefineGAN2, ``conv_pre`` on HiFi-GAN NSF) has a shape that
depends on which, so every other weight transfers -- the upsamplers, the
residual blocks, the excitation source, ``conv_post``.  The discriminator
transfers whole, because it only ever sees waveforms.

What does NOT work is training the decoder here and then freezing it for
stages 2 and 3.  Freezing pins the decoder's input representation, so ``z``
would have to *be* a mel: ``enc_q`` becomes a closed-form transform of the
audio, the KL stops measuring anything, and the model's ceiling drops to what
80 bins without phase carry.  If that is the architecture you want, it is a
two-stage acoustic-model + vocoder pipeline and ``enc_q`` should be deleted
rather than pinned.

So the intended order is:

    stage 0  (here)          dec + D on mel -> wav
    tools/graft_vocoder_pretrain.py     -> a pretrain G/D pair
    stage 1  --pretrain-g/-d from the graft
    stage 2, stage 3         unchanged

WHAT IT NEEDS
-------------
The same ``logs/<model>/`` a normal run uses: ``filelist.txt`` plus the audio
and f0 it names.  The phone-feature column is read and ignored, so a corpus
extracted with ``--no-embedding`` style shortcuts is fine as long as f0 is
there -- the decoder's excitation is f0-driven and cannot be trained without
it.  Speaker conditioning is off (``g=None``): a generic vocoder corpus has no
RVC speaker table, and ``cond`` is left at its initialisation for stage 1 to
train.

USAGE
-----
    python tools/pretrain_stage0_vocoder.py --model vctk-vocoder \\
        --vocoder refinegan2 --steps 200000

    # resume: same command, it continues from the newest stage0_G_*.pth
    # short diagnostic pass:
    python tools/pretrain_stage0_vocoder.py --model vctk-vocoder \\
        --vocoder refinegan2 --steps 2000 --save-every 500

Then graft the result onto a fresh RVC model and run stage 1:

    python tools/graft_vocoder_pretrain.py --model vctk-vocoder \\
        --out-g logs/pretrain_G.pth --out-d logs/pretrain_D.pth

    python tools/pretrain_stage1_vocoder.py --model my-voice \\
        --vocoder refinegan2 --total-epochs 40 \\
        --pretrain-g logs/pretrain_G.pth --pretrain-d logs/pretrain_D.pth

WHAT TO LOOK AT
---------------
``loss/g_spectral`` is the objective.  The adversarial and feature-matching
terms behave as they do in any GAN vocoder and are logged for shape, not as
targets.  A defect visible here is the decoder's: the input is the mel of the
very audio being reconstructed, so nothing upstream can be blamed for it.

    tensorboard --logdir logs/<model>/eval_stage0

Scalars land there every ``--log-every`` steps.  Every ``--preview-every``
steps it also writes the trainer's validation preview -- target and generated
mel side by side with their difference, plus both waveforms as audio --
rendered through the EMA.  The same figures and WAVs are written under
``logs/<model>/stage0/validation_samples/``.

The preview reads ``logs/reference/`` when it is there, so it is the same clip
the trainer previews in stages 1-3 and the two are comparable.  It needs only
``ref_audio.wav`` and ``ref_f0f.npy`` -- never the embedder-specific
``ref_feats.npy``, so a reference built for another embedder still works here.
Without that folder it falls back to a batch of dataset crops fixed at
startup; ``--preview-source`` forces either.

Its own log directory, not the ``eval`` the three stages share: stage 0
optimises a different objective, and the curves are not comparable to theirs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
# ``rvc/train`` modules import each other by bare name (``from mel_processing
# import ...``), which only resolves with that directory on the path -- which
# is what running ``rvc/train/train.py`` as a script normally provides.
_train_dir = os.path.join(now_dir, "rvc", "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from rvc.lib.algorithm import generators  # noqa: E402
from rvc.train.ema import WeightEMA  # noqa: E402
from rvc.train.losses import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from rvc.train.mel_processing import build_ms_mel_loss  # noqa: E402
from rvc.train.setup import (  # noqa: E402
    _inductor_cache_dir,
    get_d_model,
    normalize_san_weights,
)
from rvc.train.utils import (  # noqa: E402
    block_tensorboard_flush_on_exit,
    flush_writer,
    load_config_from_json,
    load_wav_to_torch,
    log_validation_preview,
    summarize,
    wave_to_mel,
)
from tools._staged_pretrain import (  # noqa: E402
    StageError,
    check_dataset,
    model_dir,
    resolve_vocoder,
)

CHECKPOINT_PREFIX = "stage0"


def build_decoder(config, vocoder: str, checkpointing: bool):
    """The decoder alone, built to read a mel instead of ``z``.

    ``gin_channels=0`` because no speaker embedding exists in this stage; the
    conditioning projection is created by stage 1 instead.
    """

    num_mels = int(config.data.n_mel_channels)
    if vocoder == "refinegan2":
        return generators.RefineGAN2Generator(
            sample_rate=int(config.data.sample_rate),
            upsample_rates=tuple(config.model.upsample_rates),
            upsample_initial_channel=int(config.model.upsample_initial_channel),
            num_mels=num_mels,
            start_channels=int(
                getattr(config.model, "refinegan2_start_channels", 16)
            ),
            leaky_relu_slope=float(
                getattr(config.model, "refinegan2_leaky_relu_slope", 0.2)
            ),
            gin_channels=0,
            checkpointing=checkpointing,
            source_gain=bool(
                getattr(config.model, "refinegan2_source_gain", False)
            ),
            source_noise_std=float(
                getattr(config.model, "refinegan2_source_noise_std", 0.003)
            ),
            source_harmonics=int(
                getattr(config.model, "refinegan2_source_harmonics", 0)
            ),
            source_tilt=float(
                getattr(config.model, "refinegan2_source_tilt", 1.0)
            ),
        )
    if vocoder == "hifigan_nsf":
        return generators.HiFiGANNSFGenerator(
            initial_channel=num_mels,
            resblock_kernel_sizes=config.model.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.resblock_dilation_sizes,
            upsample_rates=config.model.upsample_rates,
            upsample_initial_channel=int(config.model.upsample_initial_channel),
            upsample_kernel_sizes=config.model.upsample_kernel_sizes,
            gin_channels=0,
            sr=int(config.data.sample_rate),
            checkpointing=checkpointing,
        )
    raise StageError(
        f"Stage 0 has no decoder builder for vocoder {vocoder!r}. It builds "
        f"the generator directly rather than through Synthesizer, so a new "
        f"decoder has to be named here."
    )


def load_reference_preview(config):
    """``logs/reference`` as a one-clip preview batch, or ``None``.

    Only ``ref_audio.wav`` and ``ref_f0f.npy`` are read: the excitation needs
    the float f0 and the reconstruction target is the audio itself.  The
    embedder-specific ``ref_feats.npy`` this stage never touches, so the
    compatibility check the trainer makes on it does not apply here -- a
    reference built for a different embedder still works as a stage-0 preview.
    """

    reference_path = Path(now_dir) / "logs" / "reference"
    audio_path = reference_path / "ref_audio.wav"
    f0_path = reference_path / "ref_f0f.npy"
    if not audio_path.is_file() or not f0_path.is_file():
        return None

    from rvc.lib.utils import load_audio

    hop_length = int(config.data.hop_length)
    wave = load_audio(str(audio_path), int(config.data.sample_rate))
    pitchf = np.load(f0_path, allow_pickle=False)

    # Trim to the frames the two share, the way the trainer trims its three.
    frames = min(int(pitchf.shape[0]), int(wave.shape[0]) // hop_length)
    if frames <= 0:
        return None

    return (
        torch.FloatTensor(wave[: frames * hop_length]).view(1, 1, -1),
        torch.FloatTensor(np.asarray(pitchf[:frames], dtype=np.float32)).view(1, -1),
    )


def enable_decoder_compile(decoder, mode: str = "default") -> bool:
    """``Synthesizer.enable_decoder_compile`` for a bare decoder.

    Stage 0 builds the generator directly, so the Synthesizer's method is out
    of reach and its wrapper is replicated here: compile the *training*
    forward, keep eager when the module is in eval so the previews do not
    recompile on their own batch size, and fall back to eager for good on the
    first failure rather than taking the run down.
    """

    eager_forward = decoder.forward
    try:
        compiled_forward = torch.compile(eager_forward, dynamic=False, mode=mode)
    except Exception as error:
        print(f"  compile   not enabled ({error}); staying eager")
        return False

    failed = False

    def training_forward(*args, **kwargs):
        nonlocal failed
        if not decoder.training or failed:
            return eager_forward(*args, **kwargs)
        try:
            return compiled_forward(*args, **kwargs)
        except Exception as error:
            failed = True
            print(f"  compile   runtime failure ({error}); falling back to eager")
            return eager_forward(*args, **kwargs)

    decoder.forward = training_forward
    return True


class MelWavSegments(torch.utils.data.Dataset):
    """Fixed-length ``(audio, f0)`` segments drawn from an RVC filelist.

    Cropping in the dataset rather than the collate keeps every batch the same
    shape, which is what the mel of a segment being exactly
    ``segment_size // hop_length`` frames depends on.
    """

    def __init__(self, filelist: Path, config):
        self.sample_rate = int(config.data.sample_rate)
        self.hop_length = int(config.data.hop_length)
        self.segment_size = int(config.train.segment_size)
        self.frames = self.segment_size // self.hop_length

        self.entries = []
        for line in filelist.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            # audio | feature | f0 (coarse) | f0_voiced (float) | speaker.
            # The float f0 is what the decoder's excitation reads; the coarse
            # one belongs to ``enc_p``, which this stage does not build.
            self.entries.append((parts[0], parts[3]))
        if not self.entries:
            raise StageError(f"{filelist} yielded no usable entries.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index):
        audio_path, pitchf_path = self.entries[index]
        audio, rate = load_wav_to_torch(audio_path)
        if rate != self.sample_rate:
            raise StageError(
                f"{audio_path} is {rate} Hz; config.json says "
                f"{self.sample_rate} Hz."
            )
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        pitchf = torch.from_numpy(
            np.load(pitchf_path, allow_pickle=False)
        ).float()

        available = min(int(pitchf.shape[0]), audio.shape[-1] // self.hop_length)
        if available < self.frames:
            # Short clip: pad both to one segment. A zero f0 reads as unvoiced,
            # which is a state the excitation already has to handle.
            pad_frames = self.frames - max(available, 0)
            audio = torch.nn.functional.pad(
                audio[..., : max(available, 0) * self.hop_length],
                (0, pad_frames * self.hop_length),
            )
            pitchf = torch.nn.functional.pad(
                pitchf[: max(available, 0)], (0, pad_frames)
            )
            start = 0
        else:
            start = random.randint(0, available - self.frames)

        offset = start * self.hop_length
        return (
            audio[..., offset : offset + self.segment_size],
            pitchf[start : start + self.frames],
        )


@torch.no_grad()
def log_preview(
    writer,
    directory: Path,
    config,
    net_g,
    ema,
    batch,
    device,
    global_step: int,
    preview_every: int,
    samples: int,
):
    """Render a fixed batch and log its mels and audio, as the trainer does.

    Through the EMA when there is one, for the reason ``train.py`` gives at its
    own preview: a GAN generator's live weights swing step to step, so the
    preview would show a defect the averaged weights mostly do not have.
    """

    wav, pitchf = batch
    y = wav.to(device)
    pitchf = pitchf.to(device)
    mel = wave_to_mel(config, y, for_loss=False)

    was_training = net_g.training
    net_g.eval()
    try:
        if ema is not None:
            with ema.applied(net_g):
                y_hat = net_g(mel, pitchf, None)
        else:
            y_hat = net_g(mel, pitchf, None)
    finally:
        if was_training:
            net_g.train()

    y_hat = y_hat[..., : y.shape[-1]].float()
    generated_mel = wave_to_mel(config, y_hat, for_loss=False)

    figsize = (
        float(getattr(config.train, "validation_preview_width", 24.0)),
        float(getattr(config.train, "validation_preview_height", 5.8)),
    )
    # The shipped configs leave ``mel_fmax`` null, and the figure's frequency
    # axis needs a number to place its ticks on.
    mel_fmax = config.data.mel_fmax or config.data.sample_rate / 2

    for index in range(min(int(samples), int(y.shape[0]))):
        log_validation_preview(
            writer=writer,
            # Under ``stage0/`` so these cannot land in the same
            # ``validation_samples/epoch_XXXX`` the trainer writes for stages
            # 1-3 when both run in one folder.
            experiment_dir=str(directory / "stage0"),
            # Stage 0 counts steps, not epochs. This is the preview's ordinal,
            # which is what keeps the sample directories sorted.
            epoch=global_step // max(int(preview_every), 1),
            sample_index=index,
            global_step=global_step,
            sample_rate=int(config.data.sample_rate),
            predicted_mel=generated_mel[index],
            target_mel=mel[index],
            predicted_wave=y_hat[index],
            target_wave=y[index],
            hop_length=int(config.data.hop_length),
            mel_fmin=config.data.mel_fmin,
            mel_fmax=mel_fmax,
            dpi=getattr(config.train, "validation_preview_dpi", None),
            figsize=figsize,
        )


def latest_checkpoint(directory: Path) -> tuple[int, Path, Path] | None:
    found = {}
    for entry in directory.iterdir():
        name = entry.name
        if not name.startswith(f"{CHECKPOINT_PREFIX}_") or entry.suffix != ".pth":
            continue
        parts = name[: -len(".pth")].split("_")
        if len(parts) != 3 or parts[1] not in ("G", "D"):
            continue
        try:
            step = int(parts[2])
        except ValueError:
            continue
        found.setdefault(step, {})[parts[1]] = entry
    complete = {s: p for s, p in found.items() if set(p) == {"G", "D"}}
    if not complete:
        return None
    step = max(complete)
    return step, complete[step]["G"], complete[step]["D"]


def save(directory: Path, step: int, net_g, net_d, optim_g, optim_d, ema, config):
    payload_g = {
        "model": net_g.state_dict(),
        "step": step,
        "optimizer": optim_g.state_dict(),
        "num_mels": int(config.data.n_mel_channels),
    }
    if ema is not None:
        payload_g["ema"] = ema.state_dict()
    torch.save(payload_g, directory / f"{CHECKPOINT_PREFIX}_G_{step}.pth")
    torch.save(
        {"model": net_d.state_dict(), "step": step, "optimizer": optim_d.state_dict()},
        directory / f"{CHECKPOINT_PREFIX}_D_{step}.pth",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 0: train the decoder alone as a mel -> wav vocoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--preview-every",
        type=int,
        default=2000,
        help="Steps between mel/audio previews (0 disables them).",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=2,
        help=(
            "How many clips of the fixed preview batch to render. The "
            "reference is a single clip, so this only bites on the dataset "
            "source."
        ),
    )
    parser.add_argument(
        "--preview-source",
        choices=("auto", "reference", "dataset"),
        default="auto",
        help=(
            "Where the preview comes from. 'auto' takes logs/reference when "
            "it is there and falls back to the dataset; 'reference' requires "
            "it (default: auto)."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr-g", type=float, default=None)
    parser.add_argument("--lr-d", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=WeightEMA.DEFAULT_DECAY)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--benchmark", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the decoder's training forward (default: off).",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help=(
            "torch.compile mode. 'reduce-overhead' wants one fixed input "
            "shape per step, which this stage has by construction."
        ),
    )
    args = parser.parse_args()

    directory = model_dir(args.model)
    if not directory.is_dir():
        raise StageError(f"{directory} does not exist.")
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise StageError(
            f"{config_path} is missing. Run preprocess + extract for this "
            f"model first."
        )
    config = load_config_from_json(str(config_path))
    dataset_info = check_dataset(directory)
    vocoder = resolve_vocoder(directory, args.vocoder)

    if not torch.cuda.is_available():
        raise StageError("Stage 0 needs a CUDA device.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Stage 0 -- vocoder (mel -> wav)")
    print(f"  dir       {directory}")
    print(f"  data      {dataset_info['slices']:,} slices")
    print(f"  rate      {config.data.sample_rate} Hz")
    print(f"  vocoder   {vocoder}")
    print(f"  input     mel, {config.data.n_mel_channels} bins "
          f"(RVC trains this decoder on {config.model.inter_channels}-ch z)")
    print(f"  training  dec, discriminator")
    print(f"  held      nothing -- enc_p/enc_q/flow/emb_g are not built")

    net_g = build_decoder(config, vocoder, bool(args.checkpointing)).to(device)
    if args.compile:
        # The shared Inductor cache under logs/, as the trainer does: the
        # artefacts are a function of the code and the GPU, not of the run.
        _inductor_cache_dir()
        if enable_decoder_compile(net_g, str(args.compile_mode)):
            print(f"  compile   decoder, mode={args.compile_mode}")
    # Its own subdirectory rather than the ``eval`` the three stages share:
    # this stage optimises a different objective, and putting its curves on
    # the same axes as theirs would invite reading one as the other.
    writer = SummaryWriter(
        log_dir=str(directory / "eval_stage0"),
        flush_secs=86400,
    )
    block_tensorboard_flush_on_exit(writer)
    net_d = get_d_model(config, vocoder, bool(args.checkpointing)).to(device)

    lr_g = float(args.lr_g if args.lr_g is not None else config.train.learning_rate_g)
    lr_d = float(args.lr_d if args.lr_d is not None else config.train.learning_rate_d)
    betas = tuple(config.train.betas)
    optim_g = torch.optim.AdamW(
        net_g.parameters(), lr=lr_g, betas=betas, eps=config.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), lr=lr_d, betas=betas, eps=config.train.eps
    )

    ema = WeightEMA(net_g, decay=float(args.ema_decay)) if args.ema else None

    global_step = 0
    resume = latest_checkpoint(directory)
    if resume is not None:
        global_step, g_path, d_path = resume
        checkpoint_g = torch.load(g_path, map_location="cpu", weights_only=True)
        stored_mels = int(checkpoint_g.get("num_mels", config.data.n_mel_channels))
        if stored_mels != int(config.data.n_mel_channels):
            raise StageError(
                f"{g_path} was trained with {stored_mels} mel bins and this "
                f"config says {config.data.n_mel_channels}. The decoder's "
                f"input projection would not match."
            )
        net_g.load_state_dict(checkpoint_g["model"], strict=True)
        optim_g.load_state_dict(checkpoint_g["optimizer"])
        if ema is not None and "ema" in checkpoint_g:
            ema.load_state_dict(checkpoint_g["ema"], net_g)
        checkpoint_d = torch.load(d_path, map_location="cpu", weights_only=True)
        net_d.load_state_dict(checkpoint_d["model"], strict=True)
        optim_d.load_state_dict(checkpoint_d["optimizer"])
        print(f"  resume    step {global_step} from {g_path.name} / {d_path.name}")

    dataset = MelWavSegments(directory / "filelist.txt", config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.workers) > 0,
    )

    # Captured once and reused. The dataset crops at random on every access,
    # so drawing a fresh preview each time would compare a different segment
    # at every step and show drift that is the crop's, not the decoder's.
    preview_batch = None
    preview_origin = None
    if int(args.preview_every) > 0:
        if args.preview_source in ("auto", "reference"):
            preview_batch = load_reference_preview(config)
            if preview_batch is not None:
                preview_origin = "logs/reference"
            elif args.preview_source == "reference":
                raise StageError(
                    "--preview-source reference needs logs/reference/"
                    "ref_audio.wav and ref_f0f.npy. Create them with "
                    "tools/make_reference.py, or use --preview-source dataset."
                )
        if preview_batch is None:
            picks = [
                dataset[index]
                for index in range(min(int(args.preview_samples), len(dataset)))
            ]
            if picks:
                preview_batch = (
                    torch.stack([item[0] for item in picks]),
                    torch.stack([item[1] for item in picks]),
                )
                preview_origin = "dataset (fixed crops)"
    if preview_origin is not None:
        seconds = preview_batch[0].shape[-1] / int(config.data.sample_rate)
        print(f"  preview   {preview_origin}, {seconds:.2f}s")

    fn_spectral = build_ms_mel_loss(int(config.data.sample_rate)).to(device)
    c_mel = float(config.train.c_mel)
    branch_weights = getattr(net_d, "branch_weights", None)
    # Read off the assembled discriminator rather than the config key:
    # ``d_use_san`` is what builds the heads, ``supports_san`` is what says
    # they are actually there.
    san_active = bool(getattr(net_d, "supports_san", False))
    san_direction_weight = max(
        0.0,
        min(1.0, float(getattr(config.train, "san_direction_weight", 0.25))),
    )
    print(f"  disc      {getattr(net_d, 'version', '?')}"
          f"{'  SAN' if san_active else ''}")

    use_amp = bool(args.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n  steps     {global_step} -> {args.steps}\n")
    started = time.time()
    running = {"spectral": 0.0, "adv": 0.0, "fm": 0.0, "disc": 0.0, "n": 0}

    while global_step < int(args.steps):
        for wav, pitchf in loader:
            if global_step >= int(args.steps):
                break
            y = wav.to(device, non_blocking=True)
            pitchf = pitchf.to(device, non_blocking=True)

            # The decoder's input: the mel of the very audio it must return.
            mel = wave_to_mel(config, y, for_loss=False)

            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                y_hat = net_g(mel, pitchf, None)
            y_hat = y_hat[..., : y.shape[-1]].float()

            # Discriminator. ``combine_inputs`` runs real and fake as one
            # batch of 2B, which is the same numbers in half the launches.
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(
                    y,
                    y_hat.detach(),
                    san_training=san_active,
                    combine_inputs=True,
                )
                loss_disc = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                    san_direction_weight=san_direction_weight,
                    branch_weights=branch_weights,
                )[0]
            optim_d.zero_grad(set_to_none=True)
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1000.0)
            scaler.step(optim_d)
            if san_active:
                # The direction is only unit-norm because it is kept there;
                # an Adam step moves it off.
                normalize_san_weights(net_d)

            # Generator.
            net_d.requires_grad_(False)
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat, no_grad_real=True)
                loss_spectral = fn_spectral(y, y_hat) * c_mel
                loss_fm = (
                    feature_loss(fmap_r, fmap_g, branch_weights=branch_weights) * 2.0
                )
                # Returns a bare tensor unless ``per_branch`` is set, unlike
                # ``discriminator_loss``, which always returns a tuple.
                # This forward never sets ``san_training``, so these are plain
                # logits and ``san_direction_weight`` is inert here.
                loss_adv = generator_loss(
                    y_d_hat_g,
                    san_direction_weight=san_direction_weight,
                    use_softplus=san_active,
                    branch_weights=branch_weights,
                )
                loss_gen = loss_spectral + loss_fm + loss_adv
            optim_g.zero_grad(set_to_none=True)
            scaler.scale(loss_gen).backward()
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1000.0)
            scaler.step(optim_g)
            scaler.update()
            net_d.requires_grad_(True)

            if ema is not None:
                ema.update(net_g)

            global_step += 1
            running["spectral"] += float(loss_spectral.detach())
            running["adv"] += float(loss_adv.detach())
            running["fm"] += float(loss_fm.detach())
            running["disc"] += float(loss_disc.detach())
            running["n"] += 1

            if global_step % int(args.log_every) == 0:
                count = max(running["n"], 1)
                rate = global_step / max(time.time() - started, 1e-6)
                print(
                    f"  step {global_step:>8}  "
                    f"spectral {running['spectral'] / count:8.4f}  "
                    f"adv {running['adv'] / count:7.4f}  "
                    f"fm {running['fm'] / count:7.4f}  "
                    f"disc {running['disc'] / count:7.4f}  "
                    f"({rate:.2f} steps/s)"
                )
                summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars={
                        "loss/g_spectral": running["spectral"] / count,
                        "loss/g_adv": running["adv"] / count,
                        "loss/g_fm": running["fm"] / count,
                        "loss/d_total": running["disc"] / count,
                        "learning_rate/g": optim_g.param_groups[0]["lr"],
                        "learning_rate/d": optim_d.param_groups[0]["lr"],
                        "grad_scaler/scale": scaler.get_scale(),
                        "throughput/steps_per_second": rate,
                    },
                )
                flush_writer(writer, 0)
                running = {"spectral": 0.0, "adv": 0.0, "fm": 0.0, "disc": 0.0, "n": 0}

            if (
                preview_batch is not None
                and global_step % int(args.preview_every) == 0
            ):
                log_preview(
                    writer,
                    directory,
                    config,
                    net_g,
                    ema,
                    preview_batch,
                    device,
                    global_step,
                    int(args.preview_every),
                    int(args.preview_samples),
                )
                flush_writer(writer, 0)

            if global_step % int(args.save_every) == 0:
                save(directory, global_step, net_g, net_d, optim_g, optim_d, ema, config)
                print(f"  saved     {CHECKPOINT_PREFIX}_G_{global_step}.pth")

    save(directory, global_step, net_g, net_d, optim_g, optim_d, ema, config)
    if preview_batch is not None:
        log_preview(
            writer,
            directory,
            config,
            net_g,
            ema,
            preview_batch,
            device,
            global_step,
            int(args.preview_every),
            int(args.preview_samples),
        )
    flush_writer(writer, 0)
    print(f"\n  saved     {CHECKPOINT_PREFIX}_G_{global_step}.pth")
    print(
        f"\nNext: python tools/graft_vocoder_pretrain.py --model {directory.name} "
        f"--out-g logs/pretrain_G.pth --out-d logs/pretrain_D.pth\n"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StageError as error:
        print(f"\n[stage 0] {error}\n", file=sys.stderr)
        sys.exit(1)
