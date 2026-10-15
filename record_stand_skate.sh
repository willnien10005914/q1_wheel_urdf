#!/bin/bash
# Headless play + RGB video of the standing two-wheel skate policy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

CKPT="${CHECKPOINT:-$ROOT/checkpoints/skateboard_ppo.pt}"
STEPS="${STEPS:-500}"
VX="${VX:-1.6}"

exec "$ISAAC_PYTHON" scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-WheelHumanoid-Skateboard-Play-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length "$STEPS" \
  --enable_cameras \
  --no-keyboard \
  --checkpoint "$CKPT" \
  --cmd_vx "$VX" \
  --cmd_vy 0.0 \
  --cmd_yaw 0.0 \
  --max_steps "$STEPS" \
  "$@"
