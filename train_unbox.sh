#!/bin/bash
# Train the open-box sit-up: supine → tuck → yoga press → knee-roller kneel.
# Stand-up is a different PPO (./train_posture.sh).
#   ./train_unbox.sh
#   NUM_ENVS=64 MAX_ITERS=3 ./train_unbox.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-8000}"
RUN_NAME="${RUN_NAME:-unbox_${NUM_ENVS}envs}"

COMMON=(
  --task Isaac-Q1-Unbox-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

case "$MODE" in
  gui) exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
