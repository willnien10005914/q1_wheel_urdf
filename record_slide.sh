#!/bin/bash
# Headless play + RGB video of Isaac-Q1-Slide (L/R fore-aft stance + micro-lift).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_slide_ppo.pt}"
STEPS="${STEPS:-800}"
PROFILE="${PROFILE:-0:0.0,50:0.4,200:0.8,400:1.0}"
OUT="${OUT:-$ROOT/docs/results/q1_slide_foreaft_live.mp4}"

"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Q1-Slide-Play-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length "$STEPS" \
  --enable_cameras \
  --no-keyboard \
  --checkpoint "$CKPT" \
  --cmd_profile "$PROFILE" \
  --cmd_vy 0.0 \
  --hold_heading "${HEADING:-0.0}" \
  --max_steps "$STEPS" \
  "$@"

SRC="$(dirname "$CKPT")/videos/play/rl-video-step-0.mp4"
if [[ -f "$SRC" ]]; then
  mkdir -p "$(dirname "$OUT")"
  cp "$SRC" "$OUT"
  echo "[INFO] Video saved to: $OUT"
else
  echo "[WARN] Expected video missing: $SRC" >&2
  exit 1
fi
