#!/usr/bin/env bash
# All active Q1 skill training uses the native CAD URDF and Isaac Lab managers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
SKILL="${1:-walk}"
if [[ $# -gt 0 ]]; then shift; fi
case "$SKILL" in
  stand|standard) TASK=Stand ;;
  walk|skate) TASK=Walk ;;
  standup|posture|unbox) TASK=Standup ;;
  backflip) TASK=Backflip ;;
  slide|sliding) TASK=Slide ;;
  *) echo "Skill must be stand, walk, standup, backflip or slide." >&2; exit 2 ;;
esac
MODE="${1:-headless}"
if [[ "${1:-}" == gui || "${1:-}" == headless ]]; then shift; fi
ARGS=(--task "Isaac-Q1-Native-${TASK}-v0" --num_envs "${NUM_ENVS:-1024}"
      --max_iterations "${MAX_ITERS:-20000}" --seed "${SEED:-42}"
      --experiment_name "q1_native_${SKILL}" --run_name "${RUN_NAME:-native_${SKILL}}")
if [[ "$MODE" != gui ]]; then ARGS+=(--headless); fi
if [[ "${RESUME:-0}" == 1 ]]; then ARGS+=(--resume --checkpoint "${CHECKPOINT:?set a native checkpoint}" --resume_next_iteration); fi
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py "${ARGS[@]}" "$@"
