#!/usr/bin/env bash
# Native BU10 wheeled asset: manager-based PPO, separate from the 24-axis web skills.
# NUM_ENVS=64 MAX_ITERS=10 ./train_inertia.sh --no_publish_checkpoint
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
MODE="${1:-headless}"
if [[ "${1:-}" == gui || "${1:-}" == headless ]]; then shift; fi
COMMON=(
  --task Isaac-Q1-Inertia-Skate-v0
  --num_envs "${NUM_ENVS:-1024}"
  --max_iterations "${MAX_ITERS:-20000}"
  --seed "${SEED:-42}"
  --run_name "${RUN_NAME:-inertia_native23}"
)
if [[ "${RESUME:-0}" == 1 ]]; then
  COMMON+=(--resume --checkpoint "${CHECKPOINT:?set CHECKPOINT to a native inertia checkpoint}")
fi
if [[ "$MODE" != gui ]]; then COMMON+=(--headless); fi
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@"
