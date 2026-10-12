#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/isaac_env.sh
export WHEEL_HUMANOID_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/q1_backflip_v2:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=${DISPLAY:-:0}
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
exec "$ISAAC_PYTHON" -u q1_backflip_v2/search.py --headless "$@"
