#!/bin/bash
# Mixed supine+prone recovery PPO (legacy). Prefer the split scripts:
#   ./train_recovery_v3_supine.sh   # 正躺-only independent PPO
#   ./train_recovery_v3_prone.sh    # 趴躺-only independent PPO
# See docs/reference/review_recovery_v3/SPLIT_MODE_TRAINING.md
#
#   ./train_recovery_v3.sh
#   NUM_ENVS=64 MAX_ITERS=3 ./train_recovery_v3.sh
#   RECOVERY_MODE=supine NUM_ENVS=256 MAX_ITERS=100 ./train_recovery_v3.sh
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

NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERS="${MAX_ITERS:-300}"
if (( MAX_ITERS > 300 )); then
  if [[ "${ALLOW_REDESIGN:-0}" == "1" ]]; then
    echo "[INFO] ALLOW_REDESIGN=1: launching past the previous full-budget gate after controller redesign."
  else
    python tools/recovery_v3/check_gate.py --check
  fi
fi
RUN_NAME="${RUN_NAME:-recovery_v3_${NUM_ENVS}envs}"

COMMON=(
  --task Isaac-Q1-RecoveryV3-v0
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

case "$MODE" in
  gui) exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
