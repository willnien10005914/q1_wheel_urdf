#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
export WHEEL_HUMANOID_ROOT="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/q1_backflip_v2:${PYTHONPATH:-}"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/validate_web_backflip.py --headless "$@"
