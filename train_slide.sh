#!/bin/bash
# Fine-tune the foreaft slide PPO so the passing (rear) foot micro-lifts on each swap.
#   ./train_slide.sh                          # 8192 envs, +10k iters from q1_slide_ppo.pt
#   NUM_ENVS=64 MAX_ITERS=5 ./train_slide.sh  # smoke test
#   WARM_START=none ./train_slide.sh          # train from scratch
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-10000}"
RUN_NAME="${RUN_NAME:-slide_${NUM_ENVS}envs_passlift}"
WARM_START="${WARM_START:-$ROOT/checkpoints/q1_slide_ppo.pt}"
export RESET_NOISE_STD="${RESET_NOISE_STD:-0.35}"

COMMON=(
  --task Isaac-Q1-Slide-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)
if [[ "$WARM_START" != "none" && -f "$WARM_START" ]]; then
  echo "[train_slide] warm start from $WARM_START"
  COMMON+=(--resume --checkpoint "$WARM_START")
fi

case "$MODE" in
  gui) exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
