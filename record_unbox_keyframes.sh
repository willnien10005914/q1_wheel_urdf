#!/bin/bash
# Playback MediaPipe/web unbox keyframes in Isaac Sim and write an RGB mp4.
#   ./record_unbox_keyframes.sh
#   KEYFRAMES=docs/reference/a3_unbox_ref_web_recording.json ./record_unbox_keyframes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/isaac/env_isaaclab/bin:$HOME/.local/bin:$PATH"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac/IsaacLab}"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="${WHEEL_HUMANOID_ROOT:-$ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export DISPLAY="${DISPLAY:-:0}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only

cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"

KF="${KEYFRAMES:-$ROOT/docs/reference/a3_unbox_ref_web_recording.json}"
OUT="${OUT:-$ROOT/checkpoints/videos/q1_unbox_keyframes.mp4}"
mkdir -p "$(dirname "$OUT")"

python -u scripts/reinforcement_learning/rsl_rl/playback_unbox_keyframes.py \
  --keyframes "$KF" \
  --headless \
  --video \
  --enable_cameras \
  --out "$OUT" \
  "$@"

echo "[INFO] Done: $OUT"
