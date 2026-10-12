#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WHEEL_HUMANOID_ROOT=/home/will/projects/q1_wheel_urdf
export PYTHONPATH="$PWD/q1_backflip:/home/will/projects/q1_wheel_urdf/source/wheel_humanoid_lab:/home/will/IsaacLab/source/isaaclab:/home/will/IsaacLab/source/isaaclab_assets:/home/will/IsaacLab/source/isaaclab_tasks:/home/will/IsaacLab/source/isaaclab_rl:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=${DISPLAY:-:0}
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
exec /home/will/isaacsim/python.sh -u q1_backflip/train.py --headless "$@"
