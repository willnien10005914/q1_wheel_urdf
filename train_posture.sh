#!/bin/bash
# Train the Q1 posture policy (kneel on four wheels <-> stand up) with Isaac Lab + RSL-RL PPO.
# Separate network from the skate PPO; play.py switches between them (K / U / web buttons).
#   ./train_posture.sh                          # headless, 4096 envs, 4k iters (~2-3 h)
#   NUM_ENVS=64 MAX_ITERS=3 ./train_posture.sh  # smoke test
#   RESUME=1 CHECKPOINT=logs/rsl_rl/q1_posture_cubemars/<run>/model_1500.pt ./train_posture.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-4000}"
RUN_NAME="${RUN_NAME:-posture_${NUM_ENVS}envs_drive}"
WARM_START="${WARM_START:-$ROOT/checkpoints/q1_posture_ppo.pt}"
export RESET_NOISE_STD="${RESET_NOISE_STD:-0.35}"

COMMON=(
  --task Isaac-Q1-Posture-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)
if [[ "${RESUME:-}" == "1" || "${RESUME:-}" == "true" ]]; then
  COMMON+=(--resume --checkpoint "${CHECKPOINT:?set CHECKPOINT=path/to/model.pt}")
elif [[ "$WARM_START" != "none" && -f "$WARM_START" ]]; then
  echo "[train_posture] warm start from $WARM_START"
  COMMON+=(--resume --checkpoint "$WARM_START")
fi

case "$MODE" in
  gui) exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
