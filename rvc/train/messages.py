VOCODER_COMPILE_NOT_SUPPORTED = (
    "[INIT] Decoder compilation ignored: the selected vocoder does not support it."
)
VOCODER_COMPILE_NO_CUDA = (
    "[INIT] Decoder compilation ignored: CUDA is unavailable."
)
VOCODER_COMPILE_ENABLED = (
    "[INIT] Vocoder decoder compilation enabled with mode '{mode}'. "
    "The first training batch builds the graph."
)
VOCODER_COMPILE_ENABLE_FAILED = (
    "[WARNING] Vocoder decoder compilation could not be enabled:"
)
VOCODER_COMPILE_RUNTIME_FAILED = (
    "[WARNING] Vocoder decoder compilation failed; "
    "continuing in eager mode:"
)

VOCODER_COMPILE_CLI_HELP = "Compile the selected vocoder decoder during training."
TORCH_COMPILE_MODE_CLI_HELP = "Torch compile mode used for the vocoder decoder."

TORCH_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
