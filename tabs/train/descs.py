from rvc.train.messages import TORCH_COMPILE_MODES


VOCODER_INFO_RVC = "Choose a vocoder; its available sample rates are selected automatically."

PREPROCESS_RMS_VALUE_INFO = "Target loudness for RMS normalization."

DATASET_FORMAT_INFO = "Output format for sliced audio. WAV is larger; FLAC is lossless."

RESAMPLER_INFO = "librosa: high quality. ffmpeg: alternative resampler."

SMARTCUTTER_INFO = "Automatic silence trimming with short gaps preserved."

NORMALIZATION_INFO = "Choose the loudness normalization method."

AUDIO_FILE_SLICING_INFO = "Skip existing slices, use fixed chunks, or detect silence automatically."

PITCH_EXTRACTION_INFO = "RMVPE is the recommended default. CREPE is an alternative for clean audio."

BATCH_SIZE_INFO = "Larger batches need more VRAM. Reduce the value if training runs out of memory."

SPECTRAL_LOSS_INFO = "L1 Mel is the safe default. Multi-Scale and Hybrid are alternatives."

LR_SCHEDULER_INFO = "Exponential decay is the default. Cosine annealing and none are alternatives."

KL_ANNEALING_INFO = "Cyclic KL-loss annealing. Experimental."

KL_ANNEALING_CYCLE_INFO = "Length of each KL annealing cycle, in epochs."

OPTIMIZER_INFO = "AdamW is the default. AdaBelief and RAdam are alternatives."

VOCODER_COMPILE_LABEL = "Compile vocoder decoder"

VOCODER_COMPILE_INFO = "Compile the fixed-shape training decoder for higher throughput. The first batch takes longer while the graph is built."

TORCH_COMPILE_MODE_LABEL = "Torch compile mode"

TORCH_COMPILE_MODE_INFO = "Choose the compile strategy. More aggressive modes may take longer to build and use more workspace."

TORCH_COMPILE_MODE_CHOICES = TORCH_COMPILE_MODES

BEST_STEP_LABEL = "Best in-epoch step"

BEST_STEP_INFO = "Use the best FM+Mel step for evaluation previews."
