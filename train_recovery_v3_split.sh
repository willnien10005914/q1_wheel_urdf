#!/bin/bash
# Train an independent recovery PPO for ONE floor orientation.
#
#   ./train_recovery_v3_supine.sh          # 正躺 only
#   ./train_recovery_v3_prone.sh           # 趴躺 only
#   NUM_ENVS=256 MAX_ITERS=50 ./train_recovery_v3_supine.sh   # smoke
#
# Defaults: NUM_ENVS=4096  MAX_ITERS=15000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE_NAME="${RECOVERY_SPLIT_MODE:-}"
if [[ -z "$MODE_NAME" ]]; then
  case "$(basename "$0")" in
    *supine*) MODE_NAME=supine ;;
    *prone*)  MODE_NAME=prone ;;
    *) echo "Set RECOVERY_SPLIT_MODE=supine|prone or use the dedicated scripts."; exit 1 ;;
  esac
fi
case "$MODE_NAME" in
  supine) TASK=Isaac-Q1-RecoveryV3-Supine-v0; DEFAULT_RUN=v3_supine_feet_elbow ;;
  prone)  TASK=Isaac-Q1-RecoveryV3-Prone-v0;  DEFAULT_RUN=v3_prone_feet_elbow ;;
  *) echo "Unknown mode '$MODE_NAME'"; exit 1 ;;
esac

cd "$ROOT"
# Use Kit-bundled Isaac Sim python (Lab venv may be incomplete on this machine).
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"
export RECOVERY_MODE="$MODE_NAME"

LAUNCH="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-6144}"
MAX_ITERS="${MAX_ITERS:-20000}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN}_${NUM_ENVS}envs_${MAX_ITERS}it}"

echo "[INFO] Split recovery PPO: mode=$MODE_NAME task=$TASK envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
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
