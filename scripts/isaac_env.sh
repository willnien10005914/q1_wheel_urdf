# Shared Isaac Lab / Isaac Sim path resolution for Q1 scripts.
# This machine uses binary Isaac Sim at ~/isaacsim + IsaacLab at ~/IsaacLab.
# The Lab venv (~/IsaacLab/env_isaaclab) may lack torch/isaacsim; prefer _isaac_sim/python.sh.

_q1_pick_isaaclab_path() {
  local cand
  for cand in \
    "${ISAACLAB_PATH:-}" \
    "${HOME}/IsaacLab" \
    "${HOME}/isaac/IsaacLab"
  do
    [[ -n "$cand" && -d "${cand}/source/isaaclab" ]] || continue
    echo "$cand"
    return 0
  done
  return 1
}

_q1_pick_isaac_sim() {
  local cand
  for cand in \
    "${ISAAC_SIM_PATH:-}" \
    "${HOME}/isaacsim" \
    "${HOME}/isaac/isaacsim"
  do
    [[ -n "$cand" && -x "${cand}/python.sh" ]] || continue
    echo "$cand"
    return 0
  done
  return 1
}

ISAACLAB_PATH="$(_q1_pick_isaaclab_path)" || {
  echo "ERROR: Isaac Lab tree not found. Set ISAACLAB_PATH (expected ~/IsaacLab)." >&2
  return 1 2>/dev/null || exit 1
}
ISAAC_SIM_PATH="$(_q1_pick_isaac_sim)" || {
  echo "ERROR: Isaac Sim not found. Set ISAAC_SIM_PATH (expected ~/isaacsim)." >&2
  return 1 2>/dev/null || exit 1
}

# Ensure Lab can find the binary sim.
if [[ ! -e "${ISAACLAB_PATH}/_isaac_sim" ]]; then
  ln -sfn "${ISAAC_SIM_PATH}" "${ISAACLAB_PATH}/_isaac_sim"
  echo "[isaac_env] linked ${ISAACLAB_PATH}/_isaac_sim -> ${ISAAC_SIM_PATH}"
fi

# Optional pip/edit target (may be incomplete); do NOT activate it for runtime.
ISAAC_VENV="${ISAAC_VENV:-${ISAACLAB_PATH}/env_isaaclab}"

# Runtime python: Kit-bundled interpreter via Isaac Sim wrapper.
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_PATH}/python.sh}"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "ERROR: ISAAC_PYTHON not executable: $ISAAC_PYTHON" >&2
  return 1 2>/dev/null || exit 1
fi

# Put Isaac Lab + Q1 extension on path for the Kit python.
_Q1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ISAAC_SRC_PATH=""
for _pkg in isaaclab isaaclab_tasks isaaclab_rl isaaclab_assets isaaclab_mimic isaaclab_contrib; do
  if [[ -d "${ISAACLAB_PATH}/source/${_pkg}" ]]; then
    _ISAAC_SRC_PATH="${_ISAAC_SRC_PATH:+${_ISAAC_SRC_PATH}:}${ISAACLAB_PATH}/source/${_pkg}"
  fi
done
export PYTHONPATH="${_Q1_ROOT}/source/wheel_humanoid_lab:${_ISAAC_SRC_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export WHEEL_HUMANOID_ROOT="${WHEEL_HUMANOID_ROOT:-${_Q1_ROOT}}"
export ISAACLAB_PATH ISAAC_SIM_PATH ISAAC_VENV ISAAC_PYTHON
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export DISPLAY="${DISPLAY:-:0}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"

# Avoid accidentally preferring a broken Lab venv inside isaaclab.sh helpers.
unset VIRTUAL_ENV

# Convenience: `python` in scripts means the Isaac Sim wrapper.
python() { "$ISAAC_PYTHON" "$@"; }
export -f python
