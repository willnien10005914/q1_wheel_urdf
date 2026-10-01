#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
python -m pip install -e source/wheel_humanoid_lab -q
# rsl_rl learn() runs start_iter + max_iterations; from 3200 need 6800 more to hit 10000.
exec python -u scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Q1-RecoveryV3-Supine-v0 \
  --num_envs 6144 \
  --max_iterations 6800 \
  --seed 42 \
  --run_name v3_supine_resume_to_10k \
  --resume \
  --load_run 2026-10-01_09-33-18_v3_supine_only_6144envs_10k \
  --checkpoint model_3200.pt \
  --headless
