#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
export WHEEL_HUMANOID_ROOT="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
PLAYER="scripts/reinforcement_learning/rsl_rl/play_reference_web.py"
if [[ "${1:-}" == --legacy-native ]]; then
  shift
  PLAYER="scripts/reinforcement_learning/rsl_rl/play_native_web.py"
fi
exec "$ISAAC_PYTHON" -u "$PLAYER" --real-time "$@"
