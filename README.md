<div align="center">

<img src="assets/logo.png" alt="ShiroRVC" width="220" />

# ShiroRVC

**Retrieval-based voice conversion, with a modern singing-voice vocoder.**

Train and infer through a single Gradio interface — two vocoder architectures,
four pitch extractors, three content embedders, and a training loop built for
long unattended runs.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.9-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

</div>

---

## Vocoders

| | **HiFi-GAN** | **ChouwaGAN** |
|---|---|---|
| Sample rates | 32 / 40 / 48 kHz | 44.1 kHz |
| Frontend | Original VITS (flow + posterior) | Sequential VAE, slow + fast latents |
| Generator | NSF HiFi-GAN | Anti-aliased harmonic decoder with excitation U-Net |
| Discriminator | MPD + MSD | 5 period + 3 band-split complex STFT + PQMF sub-band |

**ChouwaGAN** is the architecture this fork is built around. The frontend splits
the latent into a *slow* stream for timbre and a *fast* stream for detail, each
governed by a KL rate controller that holds the divergence at a target instead
of letting it collapse. The discriminator judges the compressed **complex** STFT
— real and imaginary parts, split into frequency sub-bands — so the adversarial
signal carries phase, which the mel reconstruction losses cannot supply. A PQMF
sub-band branch watches inter-band consistency, where upsampling aliasing shows
up. Every branch uses a SAN head with lazy R1 regularisation.

## Features

- **Inference** — single and batch conversion, with pitch-curve editing.
- **Training** — full preprocess → extract → train pipeline with live TensorBoard
  diagnostics for KL, per-module gradients and GAN balance.
- **TTS** — text to speech routed through any trained voice.
- **Voice Blender** — interpolate two models into a new one.
- **Download** — pull models from a link, drop in `.pth`/`.index` files, or
  fetch pretrained weights for a chosen vocoder and sample rate.
- **Utilities** — model export, inspection and processing tools.

<table>
<tr><td><b>Pitch extraction</b></td><td><code>rmvpe</code> · <code>crepe</code> · <code>crepe-tiny</code> · <code>fcpe</code></td></tr>
<tr><td><b>Content embedders</b></td><td><code>contentvec</code> · <code>spin_v1</code> · <code>spin_v2</code> · custom</td></tr>
<tr><td><b>Optimizers</b></td><td>AdamW · AdaBelief · RAdam · Ranger21 · Schedule-Free AdamW · Schedule-Free RAdam</td></tr>
<tr><td><b>Spectral losses</b></td><td>L1 mel · multi-scale mel · hybrid L1 + MS-STFT</td></tr>
</table>

## Installation

### Windows, prebuilt

Download `ShiroRVC-Setup-win64-<version>.zip` from the
[latest release](../../releases/latest), unzip it and run `ShiroRVC-Setup.exe`.
The wizard fetches Python, PyTorch and the application, picking a CUDA build
automatically when an NVIDIA GPU is present. It needs no administrator rights
and writes nothing outside the folder you choose. Budget about 14 GB of disk.

When it finishes, `ShiroRVC.exe` in the install folder opens the native
interface.

### From source

The installer sets up a self-contained Conda environment in `env/` — nothing is
installed system-wide, and no existing Python is touched.

<table>
<tr><th align="left">Windows</th><th align="left">Linux</th></tr>
<tr valign="top">
<td>

```bat
run-install.bat
start-gui.bat
```

</td>
<td>

```bash
chmod +x run-install.sh start-gui.sh
./run-install.sh
./start-gui.sh
```

</td>
</tr>
</table>

Required models and executables download automatically on first launch.

> **Note** — do not run either script as administrator or root. Both write into
> the project tree, and doing so leaves files your normal user cannot rewrite.

## Interfaces

There are three, over the same backend. Pick whichever suits the task; they
share `logs/`, so a model trained in one is immediately visible in the others.

| | Launch with | Notes |
| --- | --- | --- |
| **Native (Qt)** | `start-gui.bat` / `start-gui.sh` | Waveform editing, live training charts, VRAM meter, batch queue. Runs the backend out of process, so a CUDA crash does not take the window down. |
| **Browser (Gradio)** | `start-gradio.bat` / `start-gradio.sh` | The original interface. Useful over a network or through a tunnel. |
| **Command line** | `python core.py --help` | Everything is scriptable; both interfaces are built on these commands. |

The native interface lives entirely in [`gui/`](gui/README.md) and is optional:
deleting that directory leaves the Gradio app and the CLI working. Its
dependencies are separate too, in `gui/requirements-gui.txt`.

## Language

Both interfaces ship in English and Brazilian Portuguese, and start in the
language your operating system is set to display. On Windows that is the
"Windows display language" and not the format locale, so an English Windows in
Brazil gets an English interface and Brazilian number formats, which is what
each of those settings actually asks for.

The command line stays in English deliberately: the native interface parses its
output, and its messages end up quoted in bug reports.

| | How to set it |
| --- | --- |
| **Native (Qt)** | The **Language** button at the bottom of the sidebar. Remembered between sessions; it offers to restart, because every widget takes its text when the window is built. |
| **Browser (Gradio)** | *Settings → Language*, applied on the next start. Or launch with `--language pt_BR`, which overrides the stored choice for that run the way `--share` does. |
| **Either** | `RVC_LANGUAGE=pt_BR`, which sits between the flag and the stored preference. |

Never having touched the switch is not the same as having chosen English:
someone who has never opened it keeps following the operating system, while an
explicit choice of English survives switching Windows to another language.

### Translating

Catalogs are standard gettext `.po` files under `locales/`, so Poedit, Weblate
and Crowdin all work on them directly. The toolchain is stdlib-only -- no Babel
and no GNU gettext binaries to install:

```bash
python tools/i18n_tool.py extract   # sources -> locales/shiromiya.pot
python tools/i18n_tool.py update    # merge the template into every .po
python tools/i18n_tool.py compile   # .po -> .mo, which is what gets loaded
python tools/i18n_tool.py stats     # what is still untranslated
```

Adding a language is one entry in `LANGUAGES` in `rvc/lib/i18n.py`, then
`update` and `compile`. Two rules keep it working: `install()` runs before any
widget is built, and no translated string lives at module scope -- mark those
with `N_()` and call `_()` where they are used. `tests/test_i18n.py` checks the
template is current and that placeholders survive translation, because a
missing catalog falls back to English silently rather than raising.

## Monitoring a run

Drag a model folder onto `logs/run_tensorboard_in_model_folder.bat`, or pass it
as an argument on Linux:

```bash
./logs/run_tensorboard_in_model_folder.sh logs/my-model
```

The launcher uses the project's own environment and binds on all interfaces, so
the board is reachable from another machine on the network at port `25565`.

## Preparing a dataset

Place clean audio in a folder under `assets/datasets/`, then select it in the
Training tab.

- Keep recordings consistent in source, tone and loudness.
- Trim leading and trailing silence; **SmartCutter** can do this for you on
  HiFi-GAN runs.
- More clean minutes beats more noisy hours.

Preprocessing writes slices at the target rate plus 16 kHz copies. The 16 kHz
copies feed pitch and embedder extraction only — training never reads them — so
the extraction tab offers to delete them afterwards, which frees roughly a third
of the preprocessing output. Re-extracting with a different f0 method or embedder
needs them back, which means preprocessing again.

## Credits

- **[codename0og](https://github.com/codename0og/)** — SmartCutter, the learned
  silence-detection model used to trim dataset audio.
- **[dr87 / spin-for-rvc](https://github.com/dr87/spin-for-rvc)** — the `spin_v1`
  and `spin_v2` content embedders.
- [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  — the original RVC project this fork descends from.

## License

Released under the [MIT License](LICENSE).
