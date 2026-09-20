#!/bin/bash
# Train the Q1 Wheel inline-skate policy (CubeMars plant) with Isaac Lab + RSL-RL PPO.
#   ./train_skate.sh                          # headless, 4096 envs, 20k iters
#   NUM_ENVS=64 MAX_ITERS=5 ./train_skate.sh  # smoke test
#   RESUME=1 CHECKPOINT=logs/rsl_rl/q1_skate_cubemars/<run>/model_1500.pt START_ITER=1500 ./train_skate.sh
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
python -m pip install -e source/wheel_humanoid_lab -q

MODE="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-20000}"
RUN_NAME="${RUN_NAME:-skate_${NUM_ENVS}envs}"
START_ITER="${START_ITER:-0}"   # curriculum iteration offset when resuming

COMMON=(
  --task Isaac-Q1-Skate-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
  "env.curriculum.stage.params.start_iter=$START_ITER"
  "env.curriculum.action_rate.params.start_iter=$START_ITER"
)
if [[ "${RESUME:-}" == "1" || "${RESUME:-}" == "true" ]]; then
  COMMON+=(--resume --checkpoint "${CHECKPOINT:?set CHECKPOINT=path/to/model.pt}")
fi

case "$MODE" in
  gui) exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
