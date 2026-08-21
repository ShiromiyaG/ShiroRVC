#!/usr/bin/env bash
set -euo pipefail

echo "Welcome to the ShiroRVC Installer!"
echo

# Run from the repo root regardless of where the script was invoked from.
INSTALL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$INSTALL_DIR"

MINICONDA_DIR="$HOME/miniconda3"
ENV_DIR="$INSTALL_DIR/env"
MINICONDA_VERSION="py312_26.5.3-2"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.13.0"
TORCHVISION_VERSION="0.28.0"
TORCHAUDIO_VERSION="2.11.0"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"

if [ "$(id -u)" -eq 0 ]; then
    echo "Do not run the installer as root: the environment would be owned by root."
    exit 1
fi

case "$(uname -m)" in
    x86_64)          MINICONDA_ARCH="Linux-x86_64" ;;
    aarch64 | arm64) MINICONDA_ARCH="Linux-aarch64" ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

fail() {
    echo "An error occurred during installation: $1"
    exit 1
}

download() {
    # url, destination
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        fail "neither curl nor wget is available"
    fi
}

start_time=$SECONDS

install_miniconda() {
    if [ -x "$MINICONDA_DIR/bin/conda" ]; then
        echo "Miniconda already installed. Skipping installation."
        echo
        return
    fi

    echo "Miniconda not found. Starting download and installation..."
    local pinned="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-${MINICONDA_ARCH}.sh"
    local latest="https://repo.anaconda.com/miniconda/Miniconda3-latest-${MINICONDA_ARCH}.sh"

    # The pinned build mirrors run-install.bat; fall back to the rolling
    # installer when that exact build was never published for this platform.
    if ! download "$pinned" miniconda.sh; then
        echo "Pinned installer unavailable, falling back to the latest build..."
        download "$latest" miniconda.sh \
            || fail "download failed, please check your internet connection"
    fi

    bash miniconda.sh -b -p "$MINICONDA_DIR" || fail "Miniconda installation failed"
    rm -f miniconda.sh
    echo "Miniconda installation complete."
    echo
}

create_conda_env() {
    echo "Creating Conda environment..."
    "$MINICONDA_DIR/bin/conda" create -y -k --prefix "$ENV_DIR" "python=$PYTHON_VERSION" \
        || fail "could not create the Conda environment"
    echo "Conda environment created successfully."
    echo

    echo "Installing uv package installer..."
    "$ENV_DIR/bin/python" -m pip install uv || fail "could not install uv"
    echo "uv installation complete."
    echo
}

install_dependencies() {
    echo "Installing dependencies..."

    # `--python` pins every install to the project env, which avoids having to
    # activate Conda inside a non-interactive shell.
    local uv=("$ENV_DIR/bin/python" -m uv pip install --python "$ENV_DIR/bin/python")

    "${uv[@]}" --upgrade setuptools || fail "setuptools"
    "${uv[@]}" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION" --upgrade --index-url "$PYTORCH_INDEX_URL" \
        || fail "PyTorch"
    "${uv[@]}" -r "$INSTALL_DIR/requirements.txt" || fail "requirements.txt"

    # Translation catalogs: gettext falls back to English without raising
    # when a .mo is missing, so a skipped build looks like working software.
    "$ENV_DIR/bin/python" "$INSTALL_DIR/tools/i18n_tool.py" compile || true

    echo "Dependencies installation complete."
    echo
}

install_miniconda
create_conda_env
install_dependencies

elapsed=$(( SECONDS - start_time ))
printf 'Installation time: %d hours, %d minutes, %d seconds.\n' \
    $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )) $(( elapsed % 60 ))
echo

chmod +x "$INSTALL_DIR/start.sh" "$INSTALL_DIR/logs/run_tensorboard_in_model_folder.sh" 2>/dev/null || true

echo "ShiroRVC has been installed successfully!"
echo "To start ShiroRVC, please run './start.sh'."
echo
