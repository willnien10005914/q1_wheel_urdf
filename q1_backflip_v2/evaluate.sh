#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/isaac_env.sh
export WHEEL_HUMANOID_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/q1_backflip_v2:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
exec "$ISAAC_PYTHON" -u q1_backflip_v2/evaluate.py --headless "$@"
