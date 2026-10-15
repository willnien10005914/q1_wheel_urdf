#!/bin/bash
# Headless RGB of the trained unbox PPO (supine → kneel).
#   ./record_unbox.sh
#   CHECKPOINT=logs/rsl_rl/q1_unbox_cubemars/<run>/model_7999.pt STEPS=1000 ./record_unbox.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_unbox_ppo.pt}"
STEPS="${STEPS:-1000}"
OUT="${OUT:-$ROOT/checkpoints/videos/q1_unbox_$(basename "${CKPT%.pt}").mp4}"
mkdir -p "$(dirname "$OUT")"

"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Q1-Unbox-Play-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length "$STEPS" \
  --enable_cameras \
  --no-keyboard \
  --no-web \
  --checkpoint "$CKPT" \
  --max_steps "$STEPS" \
  --seed 42 \
  'env.viewer.eye=[2.0,-2.7,1.3]' \
  'env.viewer.lookat=[0.0,0.0,0.35]' \
  "$@"

SRC="$(dirname "$CKPT")/videos/play/rl-video-step-0.mp4"
if [[ ! -f "$SRC" ]]; then
  SRC="$(find "$ROOT/checkpoints" "$ROOT/logs" -name 'rl-video-step-0.mp4' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}')"
fi
if [[ -n "${SRC:-}" && -f "$SRC" ]]; then
  cp "$SRC" "$OUT"
  echo "[INFO] Video saved to: $OUT"
else
  echo "[WARN] Could not locate rl-video-step-0.mp4"
  find "$ROOT/checkpoints" "$ROOT/logs" -name 'rl-video-step-0.mp4' 2>/dev/null | head -20 || true
fi
