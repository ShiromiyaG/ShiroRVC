"""Option groups for the ``core.py`` command line.

Kept apart from ``core.py`` because they are several hundred lines of click
declarations and nothing else; the commands that apply them stay in core.
"""

import json
import os
from functools import lru_cache
from multiprocessing import cpu_count

import click

from rvc.configs.vocoders import (
    get_all_vocoder_sample_rates,
    get_default_vocoder,
    get_vocoder_cli_choices,
)


# Get TTS Voices -> https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list?trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4
@lru_cache(maxsize=1)  # Cache only one result since the file is static
def load_voices_data():
    with open(
        os.path.join(os.path.dirname(__file__), "lib", "extras", "tts_voices.json"),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _tts_voice_choices():
    """Voice short names accepted by the tts command."""
    return [voice["ShortName"] for voice in load_voices_data()]


def apply_options(*groups):
    """Apply one or more lists of click.option decorators to a command."""
    def decorator(function):
        for group in reversed(groups):
            for option in reversed(group):
                function = option(function)
        return function

    return decorator


# ---- Shared option groups -------------------------------------------

#: Per-command overrides for the conversion options below.  Single-file, batch
#: and TTS ship genuinely different numbers in the Gradio tabs -- TTS leans on
#: the index harder and protects consonants more, because a synthesised source
#: has no breath to preserve -- so one shared set would match none of them.
#: Anything absent here keeps the value in the list.
INFER_DEFAULTS = {"index_rate": 0.5, "clean_strength": 0.3}
BATCH_INFER_DEFAULTS = {"index_rate": 0.5, "clean_strength": 0.5, "protect": 0.3}
TTS_DEFAULTS = {
    "index_rate": 0.75,
    "clean_strength": 0.5,
    "protect": 0.5,
    "clean_audio": True,
}


def inference_options(overrides: dict | None = None) -> list:
    """The conversion knobs, built with one command's defaults applied."""
    chosen = overrides or {}

    def default(name, fallback):
        return chosen.get(name, fallback)

    return [
        click.option(
            "--pitch",
            type=click.IntRange(-24, 24),
            default=0,
            show_default=True,
            help="Set the pitch of the audio. Higher values result in a higher pitch.",
        ),
        click.option(
            "--filter_radius",
            type=click.IntRange(0, 10),
            default=3,
            show_default=True,
            help="Apply median filtering to the extracted pitch values if this value is greater than or equal to three. This can help reduce breathiness in the output audio.",
        ),
        click.option(
            "--index_rate",
            type=click.FloatRange(0, 1),
            default=default("index_rate", 0.3),
            show_default=True,
            help="Control the influence of the index file on the output. Higher values mean stronger influence. Lower values can help reduce artifacts but may result in less accurate voice cloning.",
        ),
        click.option(
            "--index_k",
            type=click.IntRange(1, 64),
            default=8,
            show_default=True,
            help="Number of index neighbours averaged per frame. Fewer keeps what is idiosyncratic about the matched frames; more averages toward the dataset's mean voice.",
        ),
        click.option(
            "--index_power",
            type=click.FloatRange(0, 8),
            default=2.0,
            show_default=True,
            help="Exponent of the inverse-distance weighting of those neighbours. 0 is a flat average, larger values approach picking only the nearest.",
        ),
        click.option(
            "--index_continuity",
            type=click.FloatRange(0, 4),
            default=0.5,
            show_default=True,
            help="Reward for neighbours that continue the frame the previous one matched, which stops the retrieval jumping between unrelated parts of the dataset. Needs an index built by this fork; ignored otherwise.",
        ),
        click.option(
            "--noise_scale",
            type=click.FloatRange(0, 1),
            default=None,
            help="Scale of the random draw from the model's prior. Lower is steadier and less breathy; unset uses the model's default (0.3 for RefineGAN2, 0.66666 otherwise).",
        ),
        click.option(
            "--volume_envelope",
            type=click.FloatRange(0, 1),
            default=1,
            show_default=True,
            help="Control the blending of the output's volume envelope. A value of 1 means the output envelope is fully used.",
        ),
        click.option(
            "--silence_gate_db",
            type=click.FloatRange(-120, 0),
            default=-60.0,
            show_default=True,
            help="Input level, in dBFS, under which the output is faded out. The content encoder gives digital silence a full-magnitude embedding that the decoder renders as hiss, so passages the input says are empty come back noisy; this gates them. -120 disables it.",
        ),
        click.option(
            "--protect",
            type=click.FloatRange(0, 0.5),
            default=default("protect", 0.33),
            show_default=True,
            help="Protect consonants and breathing sounds from artifacts. A value of 0.5 offers the strongest protection, while lower values may reduce the protection level but potentially mitigate the indexing effect.",
        ),
        click.option(
            "--f0_method",
            type=click.Choice(["crepe", "crepe-tiny", "rmvpe", "fcpe"]),
            default='rmvpe',
            show_default=True,
            help="Choose the pitch extraction algorithm for the conversion. 'rmvpe' is the default and generally recommended.",
        ),
        click.option(
            "--pth_path",
            type=str,
            required=True,
            help="Full path to the RVC model file (.pth).",
        ),
        click.option(
            "--index_path",
            type=str,
            required=True,
            help="Full path to the index file (.index).",
        ),
        click.option(
            "--split_audio",
            type=click.BOOL,
            default=False,
            show_default=True,
            help="Split the audio into smaller segments before inference. This can improve the quality of the output for longer audio files.",
        ),
        click.option(
            "--f0_autotune",
            type=click.BOOL,
            default=False,
            show_default=True,
            help="Apply a light autotune to the inferred audio. Particularly useful for singing voice conversions.",
        ),
        click.option(
            "--f0_autotune_strength",
            type=click.FloatRange(0, 1),
            default=1.0,
            show_default=True,
            help="Set the autotune strength - the more you increase it the more it will snap to the chromatic grid.",
        ),
        click.option(
            "--clean_audio",
            type=click.BOOL,
            default=default("clean_audio", False),
            show_default=True,
            help="Clean the output audio using noise reduction algorithms. Recommended for speech conversions.",
        ),
        click.option(
            "--clean_strength",
            type=click.FloatRange(0, 1),
            default=default("clean_strength", 0.7),
            show_default=True,
            help="Adjust the intensity of the audio cleaning process. Higher values result in stronger cleaning, but may lead to a more compressed sound.",
        ),
        click.option(
            "--export_format",
            type=click.Choice(["WAV", "MP3", "FLAC", "OGG", "M4A"]),
            default='WAV',
            show_default=True,
            help="Select the desired output audio format.",
        ),
        click.option(
            "--embedder_model",
            type=click.Choice(["contentvec", "spin_v2"]),
            default='contentvec',
            show_default=True,
            help="Choose the model used for generating speaker embeddings.",
        ),
        click.option(
            "--f0_file",
            type=str,
            default=None,
            help="Full path to an external F0 file (.f0). This allows you to use pre-computed pitch values for the input audio.",
        ),
    ]

FORMANT_OPTIONS = [
    click.option(
        "--formant_shifting",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Apply formant shifting to the input audio. This can help adjust the timbre of the voice.",
    ),
    click.option(
        "--formant_qfrency",
        type=float,
        default=1.0,
        show_default=True,
        help="Control the frequency of the formant shifting effect. Higher values result in a more pronounced effect.",
    ),
    click.option(
        "--formant_timbre",
        type=float,
        default=1.0,
        show_default=True,
        help="Control the timbre of the formant shifting effect. Higher values result in a more pronounced effect.",
    ),
    click.option(
        "--sid",
        type=int,
        default=0,
        show_default=True,
        help="Speaker ID for multi-speaker models.",
    ),
]

STYLE_OPTIONS = [
    click.option(
        "--style_model",
        type=str,
        default=None,
        help="Style model (.pt) whose vibrato, scoops and phrase endings replace the source's fine pitch detail. Unset converts with plain RVC.",
    ),
    click.option(
        "--style_strength",
        type=click.FloatRange(0, 1),
        default=1.0,
        show_default=True,
        help="1 generates the pitch detail from scratch; lower values stay closer to the source's own.",
    ),
    click.option(
        "--style_rate",
        type=click.FloatRange(0, 2),
        default=1.0,
        show_default=True,
        help="0 keeps the source's smoothed melody, 1 the generated style, above 1 exaggerates it.",
    ),
    click.option(
        "--style_intensity",
        type=click.FloatRange(0, 3),
        default=1.0,
        show_default=True,
        help="How far to move toward the style model's descriptors: 0 keeps the source's (the dataset mean without --style_relative), 1 the model's.",
    ),
    click.option(
        "--style_relative",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Condition each passage on the source's own descriptors shifted toward the style model's, instead of on the model's everywhere.",
    ),
    click.option(
        "--style_recenter",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Keep each note centred where the source sings it.",
    ),
    click.option(
        "--style_steps",
        type=click.IntRange(1, 256),
        default=32,
        show_default=True,
        help="Sampler steps of the style model.",
    ),
    click.option(
        "--style_cfg",
        type=click.FloatRange(1, 10),
        default=2.0,
        show_default=True,
        help="How strongly the style descriptors are followed.",
    ),
    click.option(
        "--style_descriptors",
        type=str,
        default=None,
        help='JSON offsets from the style model\'s own descriptors, in standard deviations, e.g. \'{"vibrato_extent_cents": 1.0}\'.',
    ),
]

# ---- infer ----
INFER_OWN = [
    click.option(
        "--input_path",
        type=str,
        required=True,
        help="Full path to the input audio file.",
    ),
    click.option(
        "--output_path",
        type=str,
        required=True,
        help="Full path to the output audio file.",
    ),
]

# ---- batch_infer ----
BATCH_INFER_OWN = [
    click.option(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input audio files.",
    ),
    click.option(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder for saving output audio files.",
    ),
]

# ---- tts ----
TTS_OWN = [
    click.option(
        "--tts_file",
        type=str,
        required=True,
        help="File with a text to be synthesized",
    ),
    click.option("--tts_text", type=str, required=True, help="Text to be synthesized"),
    click.option(
        "--tts_voice",
        type=click.Choice(_tts_voice_choices()),
        required=True,
        help="Voice to be used for TTS synthesis.",
    ),
    click.option(
        "--tts_rate",
        type=click.IntRange(-100, 100),
        default=0,
        show_default=True,
        help="Control the speaking rate of the TTS. Values range from -100 (slower) to 100 (faster).",
    ),
    click.option(
        "--output_tts_path",
        type=str,
        required=True,
        help="Full path to save the synthesized TTS audio.",
    ),
    click.option(
        "--output_rvc_path",
        type=str,
        required=True,
        help="Full path to save the voice-converted audio using the synthesized TTS.",
    ),
]

# ---- preprocess ----
PREPROCESS_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be trained.",
    ),
    click.option(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Target sampling rate for the audio data.",
    ),
    click.option(
        "--cpu_threads",
        type=click.IntRange(1, min(cpu_count(), 192)),
        default=4,
        show_default=True,
        help="Number of CPU threads to use for preprocessing.",
    ),
    click.option(
        "--cut_preprocess",
        type=click.Choice(["Skip", "Simple", "Automatic", "New Automatic"]),
        default='New Automatic',
        show_default=True,
        help=(
            "How to cut the dataset into segments. 'Automatic' finds silence by "
            "RMS energy; 'New Automatic' uses FireRedVAD, which decides from the "
            "audio rather than its level and needs its weights downloaded."
        ),
    ),
    click.option(
        "--process_effects",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Enable high-pass filtering during preprocessing.",
    ),
    click.option(
        "--noise_reduction",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enable noise reduction during preprocessing.",
    ),
    click.option(
        "--noise_reduction_strength",
        type=click.FloatRange(0, 1),
        default=0.7,
        show_default=True,
        help="Strength of the noise reduction filter.",
    ),
    click.option(
        "--chunk_len",
        type=click.FloatRange(0.5, 5.0),
        default=3.0,
        show_default=True,
        help="Chunk length.",
    ),
    click.option(
        "--overlap_len",
        type=click.FloatRange(0.0, 0.4),
        default=0.36,
        show_default=True,
        help="Overlap length.",
    ),
    click.option(
        "--normalization_mode",
        type=click.Choice(["none", "post_peak", "pre_peak_rvc", "pre_loudness"]),
        default='pre_peak_rvc',
        show_default=True,
        help="Normalization mode.",
    ),
    click.option(
        "--loading_resampling",
        type=click.Choice(["ffmpeg", "librosa"]),
        default='ffmpeg',
        show_default=True,
        help="Both use SoXr. ffmpeg keeps ~850 Hz more top band (flat to 15.85 kHz at 32 kHz against librosa's 15.0) for ~3 ms more filter ringing; librosa is the gentler, shorter filter.",
    ),
]

# ---- extract ----
EXTRACT_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the model."),
    click.option(
        "--f0_method",
        type=click.Choice(["crepe", "crepe-tiny", "rmvpe", "fcpe"]),
        default='rmvpe',
        show_default=True,
        help="Pitch extraction method to use.",
    ),
    click.option(
        "--cpu_threads",
        type=click.IntRange(1, min(cpu_count(), 192)),
        default=4,
        show_default=True,
        help="Number of CPU threads to use for feature extraction (optional).",
    ),
    click.option(
        "--gpu",
        type=str,
        default='-',
        show_default=True,
        help="GPU device to use for feature extraction (optional).",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Target sampling rate for the audio data.",
    ),
    click.option(
        "--vocoder_arch",
        type=click.Choice(get_vocoder_cli_choices()),
        default=get_default_vocoder(),
        show_default=True,
        help="Choose the vocoder architecture",
    ),
    click.option(
        "--embedder_model",
        type=click.Choice(["contentvec", "spin_v2"]),
        default='contentvec',
        show_default=True,
        help="Choose the model used for generating speaker embeddings.",
    ),
    click.option(
        "--include_mutes",
        type=click.IntRange(0, 10),
        default=5,
        show_default=True,
        help="Number of silent files to include.",
    ),
    click.option(
        "--feature_precision",
        type=click.Choice(["fp32", "fp16"]),
        default="fp32",
        show_default=True,
        help="Precision the extracted embeddings are stored at. fp32 doubles the feature cache on disk but keeps the retrieval index free of a quantisation floor; fp16 halves it. Either can be read back without re-extracting.",
    ),
]

# ---- train ----
TRAIN_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be trained.",
    ),
    click.option(
        "--vocoder",
        type=click.Choice(get_vocoder_cli_choices()),
        default=get_default_vocoder(),
        show_default=True,
        help="Vocoder name",
    ),
    click.option(
        "--use_checkpointing",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enables usage of checkpointing.",
    ),
    click.option(
        "--compile_vocoder",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Compile the selected vocoder decoder during training.",
    ),
    click.option(
        "--torch_compile_mode",
        type=click.Choice(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]),
        default='default',
        show_default=True,
        help="Torch compile mode used for the vocoder decoder.",
    ),
    click.option(
        "--overtrain_detector",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Holds a few whole source recordings out of training and scores them periodically. Training loss cannot see overtraining; this is the only signal that can. Auto-disables on datasets too small to give any away.",
    ),
    click.option(
        "--stop_on_overtrain",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Ends the run once held-out quality has stopped improving. Off by default: the pre-overtrain weights are exported either way, this only decides whether training keeps going.",
    ),
    click.option(
        "--epoch_save_frequency",
        type=click.IntRange(1, 100),
        default=10,
        show_default=True,
        help="Save the model every specified number of epochs.",
    ),
    click.option(
        "--save_only_latest_net_models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Save only the latest G/D files.",
    ),
    click.option(
        "--save_weight_models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Save model weights every epoch.",
    ),
    click.option(
        "--total_epoch_count",
        type=click.IntRange(1, 10000),
        default=250,
        show_default=True,
        help="Total number of epochs to train for.",
    ),
    click.option(
        "--sample_rate",
        type=click.Choice(get_all_vocoder_sample_rates()),
        required=True,
        help="Sampling rate of the training data.",
    ),
    click.option(
        "--batch_size",
        type=click.IntRange(1, 50),
        default=8,
        show_default=True,
        help="Batch size for training.",
    ),
    click.option(
        "--gpu",
        type=str,
        default='0',
        show_default=True,
        help="GPU device to use for training (e.g., '0').",
    ),
    click.option(
        "--pretrained",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Use a pretrained model for initialization.",
    ),
    click.option(
        "--custom_pretrained",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Use a custom pretrained model.",
    ),
    click.option(
        "--g_pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained generator model file.",
    ),
    click.option(
        "--d_pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained discriminator model file.",
    ),
    click.option(
        "--use_warmup",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Enables usage of warmup.",
    ),
    click.option(
        "--warmup_duration",
        type=click.IntRange(1, 999),
        default=5,
        show_default=True,
        help="Duration of warmup phase (in epochs).",
    ),
    click.option(
        "--precision",
        type=click.Choice(["fp32", "fp16", "bf16"], case_sensitive=False),
        default="fp32",
        show_default=True,
        help=(
            "Autocast dtype over FP32 master weights. fp16 adds a GradScaler; "
            "bf16 needs none and keeps losses, residual streams, the excitation "
            "and the output layer in FP32. fp32 is no autocast (with TF32 if the "
            "model config enables it)."
        ),
    ),
    click.option(
        "--cleanup",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Cleanup previous training attempt.",
    ),
    click.option(
        "--index_algorithm",
        type=click.Choice(["Auto", "Faiss", "KMeans"]),
        default='Auto',
        show_default=True,
        help="Choose the method for generating the index file.",
    ),
]

# ---- index ----
INDEX_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the model."),
    click.option(
        "--index_algorithm",
        type=click.Choice(["Auto", "Faiss", "KMeans"]),
        default='Auto',
        show_default=True,
        help="Choose the method for generating the index file.",
    ),
    click.option(
        "--index_metric",
        type=click.Choice(["l2", "cosine"]),
        default="l2",
        show_default=True,
        help="Similarity used to find neighbours. l2 reproduces what upstream RVC builds; cosine ranks by direction alone, which suits embeddings whose magnitude tracks loudness.",
    ),
    click.option(
        "--index_speaker",
        type=str,
        default="all",
        show_default=True,
        help="Build the index from one speaker's features instead of the whole dataset. 'all', or a speaker id. A per-speaker index is written as <model>_spk<id>.index alongside the full one.",
    ),
]

# ---- style_extract ----
STYLE_EXTRACT_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the style run; data goes to logs/<model_name>/style_data."),
    click.option(
        "--dataset_path",
        type=str,
        default=None,
        help="Folder of whole audio files of any length; <id>_<name> subfolders are speakers. Unset reuses the preprocessed RVC experiment logs/<model_name>.",
    ),
    click.option(
        "--embedder_model",
        type=click.Choice(["contentvec", "spin_v2"]),
        default="contentvec",
        show_default=True,
        help="Embedder of a new base (ignored with --base_path).",
    ),
    click.option(
        "--base_path",
        type=str,
        default=None,
        help="style_base.pt the data will fine-tune; its embedder, codebook and statistics are used.",
    ),
    click.option("--gpu", type=str, default="0", show_default=True, help="GPU id, or '-' for CPU."),
    click.option(
        "--recompute",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Ignore the RVC experiment's F0 and features and compute them again.",
    ),
]

# ---- style_train ----
STYLE_TRAIN_OWN = [
    click.option("--model_name", type=str, required=True, help="Name of the preprocessed and extracted experiment."),
    click.option(
        "--base_path",
        type=str,
        default=None,
        help="style_base.pt to fine-tune into <model_name>_style.pt. Unset pretrains a new base on the experiment.",
    ),
    click.option(
        "--steps",
        type=click.IntRange(0, None),
        default=0,
        show_default=True,
        help="Training steps; 0 uses the config's (200000 for a base, 5000 for a fine-tune).",
    ),
    click.option(
        "--precision",
        type=click.Choice(["bf16", "fp16", "fp32"]),
        default="bf16",
        show_default=True,
        help="Autocast dtype over FP32 weights; fp16 adds a GradScaler. TF32 is on.",
    ),
    click.option("--gpu", type=str, default="0", show_default=True, help="GPU id, or '-' for CPU."),
    click.option(
        "--speech_speakers",
        type=str,
        default=None,
        help='Speech speaker ids of a pretrain, e.g. "0-4"; "" for none. Unset keeps pretrain.yaml\'s.',
    ),
    click.option(
        "--batch_size",
        type=click.IntRange(0, None),
        default=0,
        show_default=True,
        help="Batch size; 0 uses the config's (32 for a base, 16 for a fine-tune).",
    ),
    click.option(
        "--checkpointing",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Gradient checkpointing: less VRAM, slower steps.",
    ),
]

# ---- model_information ----
MODEL_INFORMATION_OWN = [
    click.option(
        "--pth_path",
        type=str,
        required=True,
        help="Path to the .pth model file.",
    ),
]

# ---- model_blender ----
MODEL_BLENDER_OWN = [
    click.option(
        "--model_name",
        type=str,
        required=True,
        help="Name of the new fused model.",
    ),
    click.option(
        "--pth_path_1",
        type=str,
        required=True,
        help="Path to the first .pth model file.",
    ),
    click.option(
        "--pth_path_2",
        type=str,
        required=True,
        help="Path to the second .pth model file.",
    ),
    click.option(
        "--ratio",
        type=click.FloatRange(0, 1),
        default=0.5,
        show_default=True,
        help="Ratio for blending the two models (0.0 to 1.0).",
    ),
]

# ---- download ----
DOWNLOAD_OWN = [
    click.option(
        "--model_link",
        type=str,
        required=True,
        help="Direct link to the model file.",
    ),
]

# ---- prerequisites ----
PREREQUISITES_OWN = [
    click.option(
        "--pretraineds_hifigan",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download pretrained models.",
    ),
    click.option(
        "--models",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download additional models.",
    ),
    click.option(
        "--exe",
        type=click.BOOL,
        default=True,
        show_default=True,
        help="Download required executables.",
    ),
]

# ---- audio_analyzer ----
AUDIO_ANALYZER_OWN = [
    click.option(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input audio file.",
    ),
]

# ---- Commands -------------------------------------------------------
