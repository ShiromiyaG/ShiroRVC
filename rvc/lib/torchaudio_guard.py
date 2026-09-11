"""Import torchaudio, turning a torch/torchaudio CUDA mismatch into a fix.

The mismatch comes from installing ``requirements.txt`` into an environment
that already had a torch stack -- cloud GPU templates and Jupyter images ship
one.  pip replaces torch with the pinned build but leaves the image's
torchaudio alone, because its ``2.11.0+cu128`` already satisfies
``torchaudio==2.11.0``.  torchaudio then refuses to load, and its own message
names the problem without saying which command solves it.

Import this module before ``torchaudio`` anywhere the latter is imported.
"""

import sys
from importlib import metadata

import torch

try:
    import torchaudio  # noqa: F401
except RuntimeError as error:
    if "CUDA version" not in str(error):
        raise
    cuda = torch.version.cuda
    variant = "cu" + cuda.replace(".", "") if cuda else "cpu"
    version = metadata.version("torchaudio").split("+")[0]
    raise RuntimeError(
        f"torch {torch.__version__} and torchaudio "
        f"{metadata.version('torchaudio')} were built for different CUDA "
        "versions. Reinstall torchaudio to match torch:\n\n"
        f"    {sys.executable} -m pip install --force-reinstall --no-deps "
        f"torchaudio=={version} "
        f"--index-url https://download.pytorch.org/whl/{variant}\n"
    ) from error
