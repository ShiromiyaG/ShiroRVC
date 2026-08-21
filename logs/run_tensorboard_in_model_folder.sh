#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root from this script's own location, so the launcher works
# no matter what the current directory is when it runs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PORT=25565
ADDRESS="http://localhost:$PORT"

# Accept the log directory as an argument; fall back to a prompt.
LOGDIR="${1:-}"
if [ -z "$LOGDIR" ]; then
    # `|| true` so a closed stdin reaches the message below instead of having
    # `set -e` abort on the failed read.
    read -rp "Key in model saved directory: " LOGDIR || true
fi
if [ -z "$LOGDIR" ]; then
    echo "No directory provided. Exiting."
    exit 1
fi
if [ ! -d "$LOGDIR" ]; then
    echo "Directory not found: $LOGDIR"
    exit 1
fi

# The project ships its own Conda env, so a bare `tensorboard` would resolve to
# whatever happens to be on PATH -- a different interpreter, or nothing at all.
PYTHON="$REPO_ROOT/env/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Environment not found at $REPO_ROOT/env"
    echo "Please run './run-install.sh' first to set up the environment."
    exit 1
fi

echo "Log directory:       $LOGDIR"
echo "TensorBoard address: $ADDRESS"
echo

# Bind on 0.0.0.0 so the board is also reachable from other machines, while the
# browser gets a host name that always resolves.
"$PYTHON" -m tensorboard.main --logdir="$LOGDIR" --host=0.0.0.0 --port="$PORT" &
TB_PID=$!

# Registered before the wait so an early Ctrl+C still takes TensorBoard down.
cleanup() {
    trap - EXIT INT TERM
    kill "$TB_PID" 2>/dev/null || true
    wait "$TB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$ADDRESS" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open "$ADDRESS" >/dev/null 2>&1 || true
else
    echo "Could not detect a web browser. Please open $ADDRESS manually."
fi

echo "TensorBoard is running with PID $TB_PID."
echo "Press Ctrl+C to stop TensorBoard and exit."

# `wait` returns non-zero when interrupted, which must not trip `set -e`.
wait "$TB_PID" || true
