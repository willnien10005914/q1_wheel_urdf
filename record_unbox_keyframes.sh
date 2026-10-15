#!/bin/bash
# Playback MediaPipe/web unbox keyframes in Isaac Sim and write an RGB mp4.
#   ./record_unbox_keyframes.sh
#   KEYFRAMES=docs/reference/a3_unbox_ref_web_recording.json ./record_unbox_keyframes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"

KF="${KEYFRAMES:-$ROOT/docs/reference/a3_unbox_ref_web_recording.json}"
OUT="${OUT:-$ROOT/checkpoints/videos/q1_unbox_keyframes.mp4}"
mkdir -p "$(dirname "$OUT")"

"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/playback_unbox_keyframes.py \
  --keyframes "$KF" \
  --headless \
  --video \
  --enable_cameras \
  --out "$OUT" \
  "$@"

echo "[INFO] Done: $OUT"
