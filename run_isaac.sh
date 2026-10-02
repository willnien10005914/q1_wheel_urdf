#!/bin/bash
# Run any python script inside the Isaac Lab environment with the wheel_humanoid_lab extension on the path.
#   ./run_isaac.sh tests/test_skate_contract.py --num_envs 4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="${WHEEL_HUMANOID_ROOT:-$ROOT}"
cd "$ROOT"
exec python -u "$@"
