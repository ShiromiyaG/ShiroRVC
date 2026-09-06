#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where the script was invoked from.
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Counterpart of the System32 guard in start.bat: the fork writes into its own
# directory tree, and doing that as root leaves files the normal user cannot
# rewrite on the next run.
if [ "$(id -u)" -eq 0 ]; then
    printf '\033[0;31m'
    echo "The fork shouldn't be run as root."
    printf '\033[0m'
    exit 1
fi

# Checking the interpreter rather than just the directory also catches an
# environment that exists but was never finished.
if [ ! -x env/bin/python ]; then
    echo "Please run './run-install.sh' first to set up the environment."
    exit 1
fi

# "$@" forwards anything the caller added -- --language pt_BR, --share, --port.
exec env/bin/python app.py --open "$@"
