#!/usr/bin/env bash
# View a native 23-action inertia PPO checkpoint in Isaac Sim (GUI by default).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
export WHEEL_HUMANOID_ROOT="$ROOT"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/play_inertia.py "$@"
