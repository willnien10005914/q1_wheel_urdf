#!/bin/bash
# Train dual-wheel standing skate (AgiBot X2 style) with Isaac Lab + RSL-RL PPO.
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

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then
  shift || true
fi

NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERS="${MAX_ITERS:-3000}"
RUN_NAME="${RUN_NAME:-stand_skate_x2_${NUM_ENVS}envs}"

COMMON=(
  --task Isaac-WheelHumanoid-Skateboard-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

if [[ "${RESUME:-}" == "1" || "${RESUME:-}" == "true" ]]; then
  CKPT="${CHECKPOINT:-$ROOT/checkpoints/skateboard_ppo.pt}"
  COMMON+=(--resume --checkpoint "$CKPT")
fi

case "$MODE" in
  gui)
    exec python scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@"
    ;;
  headless|*)
    exec python scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@"
    ;;
esac
