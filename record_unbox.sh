#!/bin/bash
# Headless RGB of the trained unbox PPO (supine → kneel).
#   ./record_unbox.sh
#   CHECKPOINT=logs/rsl_rl/q1_unbox_cubemars/<run>/model_7999.pt STEPS=1000 ./record_unbox.sh
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

CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_unbox_ppo.pt}"
STEPS="${STEPS:-1000}"
OUT="${OUT:-$ROOT/checkpoints/videos/q1_unbox_$(basename "${CKPT%.pt}").mp4}"
mkdir -p "$(dirname "$OUT")"

python -u scripts/reinforcement_learning/rsl_rl/play.py \
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
