#!/bin/bash
# Headless play + RGB video of the Isaac-Q1-Skate policy (CubeMars plant).
#
# Command profile (policy steps @ 50 Hz): coast 2 s -> glide 0.6 m/s -> skate 1.2 m/s.
#   CHECKPOINT=logs/rsl_rl/q1_skate/<run>/model_3000.pt ./record_skate.sh
#   STEPS=600 PROFILE="0:0.0,100:0.6,300:1.2" ./record_skate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/isaac/env_isaaclab/bin:$HOME/.local/bin:$PATH"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac/IsaacLab}"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="${WHEEL_HUMANOID_ROOT:-$ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export DISPLAY="${DISPLAY:-:0}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"

CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_skate_ppo.pt}"
STEPS="${STEPS:-600}"
PROFILE="${PROFILE:-0:0.0,100:0.6,300:1.2}"
OUT="${OUT:-$ROOT/checkpoints/videos/q1_skate_$(basename "${CKPT%.pt}").mp4}"

python -u scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Q1-Skate-Play-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length "$STEPS" \
  --enable_cameras \
  --no-keyboard \
  --checkpoint "$CKPT" \
  --cmd_profile "$PROFILE" \
  --cmd_vy 0.0 \
  --cmd_yaw 0.0 \
  --max_steps "$STEPS" \
  "$@"

SRC="$(dirname "$CKPT")/videos/play/rl-video-step-0.mp4"
if [[ -f "$SRC" ]]; then
  mkdir -p "$(dirname "$OUT")"
  cp "$SRC" "$OUT"
  echo "[INFO] Video saved to: $OUT"
fi
