#!/bin/bash
# Fine-tune the skate PPO into the slide / push-skate policy (alternating leg lifts, random wander).
# Warm-starts from checkpoints/q1_skate_ppo.pt so balance is kept; only the stride rewards are new.
#   ./train_slide.sh                          # headless, 4096 envs, +3k iters (~2 h)
#   NUM_ENVS=64 MAX_ITERS=3 ./train_slide.sh  # smoke test
#   WARM_START=logs/rsl_rl/q1_skate_cubemars/<run>/model_19999.pt ./train_slide.sh
#   WARM_START=none ./train_slide.sh          # train from scratch
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
MAX_ITERS="${MAX_ITERS:-3000}"
RUN_NAME="${RUN_NAME:-slide_${NUM_ENVS}envs}"
WARM_START="${WARM_START:-$ROOT/checkpoints/q1_skate_ppo.pt}"

COMMON=(
  --task Isaac-Q1-Slide-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)
if [[ "$WARM_START" != "none" && -f "$WARM_START" ]]; then
  echo "[train_slide] warm start from $WARM_START"
  COMMON+=(--resume --checkpoint "$WARM_START")
fi

case "$MODE" in
  gui) exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
