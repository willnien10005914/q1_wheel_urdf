#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/isaac_env.sh"
export PYTHONPATH="$ROOT:$ROOT/q1_backflip:${PYTHONPATH:-}"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u q1_backflip/evaluate.py --headless "$@"
