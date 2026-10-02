#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/evaluate_recovery_v3.py --headless "$@"
