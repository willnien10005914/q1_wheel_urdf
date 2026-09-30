#!/bin/bash
# Train an independent recovery PPO for ONE floor orientation.
#
#   ./train_recovery_v3_supine.sh          # 正躺 only, full budget defaults
#   ./train_recovery_v3_prone.sh           # 趴躺 only
#   NUM_ENVS=256 MAX_ITERS=50 ./train_recovery_v3_supine.sh   # smoke
#
# Defaults (see docs/reference/review_recovery_v3/SPLIT_MODE_TRAINING.md):
#   NUM_ENVS=6144  (4096 left ~56% GPU; bump for better util, fall back if OOM)
#   MAX_ITERS=12000 (prior 20k runs flattened ~10–12k)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE_NAME="${RECOVERY_SPLIT_MODE:-}"
if [[ -z "$MODE_NAME" ]]; then
  case "$(basename "$0")" in
    *supine*) MODE_NAME=supine ;;
    *prone*)  MODE_NAME=prone ;;
    *) echo "Set RECOVERY_SPLIT_MODE=supine|prone or use the dedicated scripts."; exit 1 ;;
  esac
fi
case "$MODE_NAME" in
  supine) TASK=Isaac-Q1-RecoveryV3-Supine-v0; DEFAULT_RUN=v3_supine_only ;;
  prone)  TASK=Isaac-Q1-RecoveryV3-Prone-v0;  DEFAULT_RUN=v3_prone_only ;;
  *) echo "Unknown mode '$MODE_NAME'"; exit 1 ;;
esac

export PATH="$HOME/isaac/env_isaaclab/bin:$HOME/.local/bin:$PATH"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac/IsaacLab}"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="${WHEEL_HUMANOID_ROOT:-$ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export DISPLAY="${DISPLAY:-:0}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
export RECOVERY_MODE="$MODE_NAME"

cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"
python -m pip install -e source/wheel_humanoid_lab -q

LAUNCH="${1:-headless}"
if [[ "${1:-}" == "gui" || "${1:-}" == "headless" ]]; then shift || true; fi

NUM_ENVS="${NUM_ENVS:-6144}"
MAX_ITERS="${MAX_ITERS:-12000}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN}_${NUM_ENVS}envs}"
ALLOW_REDESIGN="${ALLOW_REDESIGN:-1}"

echo "[INFO] Split recovery PPO: mode=$MODE_NAME task=$TASK envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
if (( NUM_ENVS > 4096 )); then
  echo "[INFO] NUM_ENVS=$NUM_ENVS > 4096 — if PhysX OOMs, retry with NUM_ENVS=4096."
fi

COMMON=(
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERS"
  --seed 42
  --run_name "$RUN_NAME"
)

case "$LAUNCH" in
  gui) exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" "$@" ;;
  *)   exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${COMMON[@]}" --headless "$@" ;;
esac
