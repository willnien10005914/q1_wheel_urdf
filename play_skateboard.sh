#!/bin/bash
# Play the skateboard PPO policy in Isaac Sim with WASD keyboard control.
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
python -m pip install -e source/wheel_humanoid_lab -q

TASK="${TASK:-Isaac-Q1-Skate-Play-v0}"
CKPT="${CHECKPOINT:-$ROOT/checkpoints/q1_skate_ppo.pt}"
NUM_ENVS="${NUM_ENVS:-1}"

ARGS=(
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --real-time
)

if [[ -f "$CKPT" ]]; then
  ARGS+=(--checkpoint "$CKPT")
fi

exec python scripts/reinforcement_learning/rsl_rl/play.py "${ARGS[@]}" "$@"
