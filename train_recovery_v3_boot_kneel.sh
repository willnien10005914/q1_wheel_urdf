#!/bin/bash
# Train kneel→stand only (operator-assisted boot). No gripper / no floor plant.
#
#   ./train_recovery_v3_boot_kneel.sh                 # default 256×500 probe
#   NUM_ENVS=4096 MAX_ITERS=5000 ./train_recovery_v3_boot_kneel.sh  # large
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"

TASK=Isaac-Q1-RecoveryV3-BootKneel-v0
export RECOVERY_MODE=boot_kneel

LAUNCH="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERS="${MAX_ITERS:-500}"
RUN_NAME="${RUN_NAME:-v3_boot_kneel_${NUM_ENVS}e_${MAX_ITERS}it}"

echo "[INFO] Boot-kneel PPO: task=$TASK envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
echo "[INFO] ISAAC_PYTHON=$ISAAC_PYTHON ISAACLAB_PATH=$ISAACLAB_PATH"

COMMON=(
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

case "$LAUNCH" in
  gui) exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
