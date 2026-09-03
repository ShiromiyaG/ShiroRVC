from rvc.lib.i18n import N_
from rvc.train.messages import TORCH_COMPILE_MODES


VOCODER_INFO_RVC = N_("Choose a vocoder; its available sample rates are selected automatically.")

PREPROCESS_RMS_VALUE_INFO = N_(
    "Target level for the normalization method. LUFS for BS.1770 loudness; "
    "dBFS for the legacy RMS mode."
)

DATASET_FORMAT_INFO = N_("Output format for sliced audio. WAV is larger; FLAC is lossless.")

RESAMPLER_INFO = N_("librosa: high quality. ffmpeg: alternative resampler.")


NORMALIZATION_INFO = N_("Choose the loudness normalization method.")

AUDIO_FILE_SLICING_INFO = N_("Skip existing slices, use fixed chunks, or detect silence automatically. New Automatic uses a neural VAD instead of an energy threshold.")

PITCH_EXTRACTION_INFO = N_("RMVPE is the recommended default. CREPE is an alternative for clean audio.")

BATCH_SIZE_INFO = N_("Larger batches need more VRAM. Reduce the value if training runs out of memory.")


LR_SCHEDULER_INFO = N_("Exponential decay is the default. Cosine annealing and none are alternatives.")

KL_ANNEALING_INFO = N_("Cyclic KL-loss annealing. Experimental.")

KL_ANNEALING_CYCLE_INFO = N_("Length of each KL annealing cycle, in epochs.")

OPTIMIZER_INFO = N_("AdamW is the default. Sched-Free AdamW needs no LR scheduler and averages its own weights. Muon orthogonalises the update for matrix-shaped layers. Lion halves the optimizer memory and uses a smaller learning rate automatically.")

VOCODER_COMPILE_LABEL = N_("Compile vocoder decoder")

VOCODER_COMPILE_INFO = N_("Compile the fixed-shape training decoder for higher throughput. The first batch takes longer while the graph is built.")

TORCH_COMPILE_MODE_LABEL = N_("Torch compile mode")

TORCH_COMPILE_MODE_INFO = N_("Choose the compile strategy. More aggressive modes may take longer to build and use more workspace.")

TORCH_COMPILE_MODE_CHOICES = TORCH_COMPILE_MODES

OVERTRAIN_DETECTOR_LABEL = N_("Overtrain detector")

OVERTRAIN_DETECTOR_INFO = N_(
    "Hold a few whole source recordings out of training and score them as it goes. "
    "Training loss keeps falling while a model overtrains, so this is the only signal "
    "that can see it. Exports the last good weights. Turns itself off on datasets too "
    "small to spare any."
)

STOP_ON_OVERTRAIN_LABEL = N_("Stop when overtrained")

STOP_ON_OVERTRAIN_INFO = N_(
    "End the run once held-out quality stops improving. The pre-overtrain model is "
    "exported either way; this only decides whether training keeps going."
)

USE_EMA_LABEL = N_("Weight averaging (EMA)")

USE_EMA_INFO = N_(
    "Export a moving average of the generator instead of one step of it. A GAN "
    "generator oscillates against its discriminator, so the average is usually better "
    "than any single step, and it makes the overtrain curve much cleaner. Costs one "
    "extra copy of the generator in VRAM."
)

INDEX_SINGLE_SPEAKER_INFO = N_(
    "On a multispeaker model, keeps one voice's articulation out of another's. "
    "Saved as <model>_spk<id>.index, beside the full one."
)
