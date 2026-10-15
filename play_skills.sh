#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
export WHEEL_HUMANOID_ROOT="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/play_native_web.py --real-time "$@"
