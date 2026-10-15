#!/bin/bash
# Play the skateboard PPO policy in Isaac Sim with WASD keyboard control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${Q1_LEGACY_PLAY:-0} != 1 ]]; then
  exec bash "$ROOT/play_skills.sh" "$@"
fi
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

TASK="${TASK:-Isaac-Q1-Skate-Play-v0}"
CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_skate_ppo.pt}"
NUM_ENVS="${NUM_ENVS:-1}"

ARGS=(
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --real-time
)

if [[ -f "$CKPT" ]]; then
  ARGS+=(--checkpoint "$CKPT")
fi

exec "$ISAAC_PYTHON" scripts/reinforcement_learning/rsl_rl/play.py "${ARGS[@]}" "$@"
