VOCODER_COMPILE_NOT_SUPPORTED = (
    "Decoder compilation ignored: the selected vocoder does not support it."
)
VOCODER_COMPILE_NO_CUDA = (
    "Decoder compilation ignored: CUDA is unavailable."
)
VOCODER_COMPILE_ENABLED = (
    "Vocoder decoder compilation enabled with mode '{mode}'. "
    "The first training batch builds the graph."
)
VOCODER_COMPILE_ENABLE_FAILED = (
    "Vocoder decoder compilation could not be enabled:"
)
VOCODER_COMPILE_RUNTIME_FAILED = (
    "Vocoder decoder compilation failed; "
    "continuing in eager mode:"
)

DISCRIMINATOR_COMPILE_NOT_SUPPORTED = (
    "Discriminator compilation ignored: the selected vocoder does not "
    "support it."
)
DISCRIMINATOR_COMPILE_NO_CUDA = (
    "Discriminator compilation ignored: CUDA is unavailable."
)
DISCRIMINATOR_COMPILE_ENABLED = (
    "Discriminator compilation enabled with mode '{mode}'. "
    "The first training batch builds the graph."
)
DISCRIMINATOR_COMPILE_ENABLE_FAILED = (
    "Discriminator compilation could not be enabled:"
)
DISCRIMINATOR_COMPILE_RUNTIME_FAILED = (
    "Discriminator compilation failed; continuing in eager mode:"
)

FRONTEND_COMPILE_NOT_SUPPORTED = (
    "Frontend compilation ignored: this synthesizer does not support it."
)
FRONTEND_COMPILE_NO_CUDA = (
    "Frontend compilation ignored: CUDA is unavailable."
)
FRONTEND_COMPILE_ENABLED = (
    "Frontend compilation enabled with mode '{mode}'. The prior, posterior "
    "and flow build their graphs on the first batches; unlike the decoder "
    "they see variable lengths, so expect a few recompiles before the step "
    "time settles."
)
FRONTEND_COMPILE_ENABLE_FAILED = (
    "Frontend compilation could not be enabled:"
)
FRONTEND_COMPILE_RUNTIME_FAILED = (
    "Frontend compilation failed; continuing in eager mode:"
)

VOCODER_COMPILE_CLI_HELP = "Compile the selected vocoder decoder during training."
TORCH_COMPILE_MODE_CLI_HELP = "Torch compile mode used for the vocoder decoder."

TORCH_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

TENSORBOARD_VALIDATION_PREVIEW_DIR = "validation_samples"
TENSORBOARD_VALIDATION_MEL_TITLES = (
    "Mel Generated",
    "Mel Original",
    "Difference",
)
TENSORBOARD_VALIDATION_FOOTER = (
    "Sample rate: {sample_rate} Hz   |   Hop length: {hop_length}   |   n_mels: {n_mels}"
    "   |   Epoch: {epoch}   |   Global Step: {step}"
)
TENSORBOARD_VALIDATION_DIFFERENCE_LABEL = "(Mel Generated − Mel Original) in dB"
TENSORBOARD_VALIDATION_AXIS_X = "Time (s)"
TENSORBOARD_VALIDATION_AXIS_Y = "Frequency (Hz)"
TENSORBOARD_VALIDATION_DB_LABEL = "dB"
TENSORBOARD_VALIDATION_MEL_TAG = "validation_previews/mel/{sample}"
TENSORBOARD_VALIDATION_AUDIO_TAG = "validation_previews/audio/{sample}/{kind}"
TENSORBOARD_VALIDATION_SOURCE_TAG = "validation_previews/source/{sample}"
TENSORBOARD_VALIDATION_FALLBACK_NAMESPACE = "validation_previews/sample_00"
TENSORBOARD_VALIDATION_AUDIO_NAMES = {
    "generated": "generated",
    "original": "original",
}
TENSORBOARD_MEDIA_SOURCE_NAME = "source"
