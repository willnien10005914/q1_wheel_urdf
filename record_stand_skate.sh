#!/bin/bash
# Headless play + RGB video of the standing two-wheel skate policy.
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

CKPT="${CHECKPOINT:-$ROOT/checkpoints/skateboard_ppo.pt}"
STEPS="${STEPS:-500}"
VX="${VX:-1.6}"

exec python scripts/reinforcement_learning/rsl_rl/play.py \
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
