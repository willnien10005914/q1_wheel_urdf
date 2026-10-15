#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/isaac_env.sh"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
# rsl_rl learn() runs start_iter + max_iterations; from 3200 need 6800 more to hit 10000.
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Q1-RecoveryV3-Supine-v0 \
  --num_envs 6144 \
  --max_iterations 6800 \
  --seed 42 \
  --run_name v3_supine_resume_to_10k \
  --resume \
  --load_run 2026-10-01_09-33-18_v3_supine_only_6144envs_10k \
  --checkpoint model_3200.pt \
  --headless
