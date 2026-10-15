#!/bin/bash
# Train dual-wheel standing skate (AgiBot X2 style) with Isaac Lab + RSL-RL PPO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then
  shift || true
fi

NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERS="${MAX_ITERS:-3000}"
RUN_NAME="${RUN_NAME:-stand_skate_x2_${NUM_ENVS}envs}"

COMMON=(
  --task Isaac-WheelHumanoid-Skateboard-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

if [[ "${RESUME:-}" == "1" || "${RESUME:-}" == "true" ]]; then
  CKPT="${CHECKPOINT:-$ROOT/checkpoints/skateboard_ppo.pt}"
  COMMON+=(--resume --checkpoint "$CKPT")
fi

case "$MODE" in
  gui)
    exec "$ISAAC_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@"
    ;;
  headless|*)
    exec "$ISAAC_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@"
    ;;
esac
