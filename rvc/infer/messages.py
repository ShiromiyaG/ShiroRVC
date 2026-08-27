DISCRETE_INFERENCE_MODE_LABEL = "ChouwaGAN latent inference mode"
DISCRETE_INFERENCE_MODE_INFO = (
    "Deterministic uses the prior mean; stochastic samples the continuous "
    "slow and fast latent distributions used by ChouwaGAN."
)
DISCRETE_TEMPERATURE_LABEL = "ChouwaGAN latent temperature"
DISCRETE_TEMPERATURE_INFO = (
    "Higher values increase variation in stochastic continuous latent sampling."
)
#: The tag is supplied by the terminal helpers, so these carry the text only.
INFER_SEED_SPECIFIED = "Seed {seed}, {mode} mode."
#: One line rather than two: the seed is only worth printing because it is what
#: reproduces the run, and that is the same sentence as "no seed was given".
INFER_RANDOM_SEED_EXPOSED = "No seed given; using {seed} (reuse it to reproduce)."
INFER_MODE_DETERMINISTIC = "deterministic"
INFER_MODE_STOCHASTIC = "stochastic"
