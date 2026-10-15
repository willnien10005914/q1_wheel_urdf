#!/bin/bash
# Mixed supine+prone recovery PPO (legacy). Prefer the split scripts:
#   ./train_recovery_v3_supine.sh   # 正躺-only independent PPO
#   ./train_recovery_v3_prone.sh    # 趴躺-only independent PPO
# See docs/reference/review_recovery_v3/SPLIT_MODE_TRAINING.md
#
#   ./train_recovery_v3.sh
#   NUM_ENVS=64 MAX_ITERS=3 ./train_recovery_v3.sh
#   RECOVERY_MODE=supine NUM_ENVS=256 MAX_ITERS=100 ./train_recovery_v3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERS="${MAX_ITERS:-300}"
if (( MAX_ITERS > 300 )); then
  if [[ "${ALLOW_REDESIGN:-0}" == "1" ]]; then
    echo "[INFO] ALLOW_REDESIGN=1: launching past the previous full-budget gate after controller redesign."
  else
    python tools/recovery_v3/check_gate.py --check
  fi
fi
RUN_NAME="${RUN_NAME:-recovery_v3_${NUM_ENVS}envs}"

COMMON=(
  --task Isaac-Q1-RecoveryV3-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

case "$MODE" in
  gui) exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
