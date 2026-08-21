#!/bin/sh
# Launcher for the Qt interface.
set -e

INSTALL_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$INSTALL_DIR"

if [ -x "$INSTALL_DIR/env/bin/python" ]; then
    PYTHON="$INSTALL_DIR/env/bin/python"
elif [ -x "$INSTALL_DIR/.venv/bin/python" ]; then
    PYTHON="$INSTALL_DIR/.venv/bin/python"
else
    PYTHON=$(command -v python3 || command -v python)
fi

if [ -z "$PYTHON" ]; then
    echo "No Python interpreter found. Run run-install.sh first." >&2
    exit 1
fi

if ! "$PYTHON" -c "import PySide6" >/dev/null 2>&1; then
    echo "Installing the Qt interface dependencies..."
    "$PYTHON" -m pip install -r "$INSTALL_DIR/gui/requirements-gui.txt"
fi

exec "$PYTHON" -m gui "$@"
