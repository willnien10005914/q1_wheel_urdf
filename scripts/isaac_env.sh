# Shared Isaac Lab path resolution for Q1 scripts.
# Prefer machine layout: ~/IsaacLab/{env_isaaclab,}  (also supports legacy ~/isaac/...).
# Override with ISAAC_VENV / ISAACLAB_PATH.

_q1_pick_isaac_venv() {
  local cand
  for cand in \
    "${ISAAC_VENV:-}" \
    "${HOME}/IsaacLab/env_isaaclab" \
    "${HOME}/isaac/env_isaaclab" \
    "${HOME}/isaac/IsaacLab/env_isaaclab"
  do
    [[ -n "$cand" && -x "${cand}/bin/python" ]] || continue
    echo "$cand"
    return 0
  done
  return 1
}

_q1_pick_isaaclab_path() {
  local cand
  for cand in \
    "${ISAACLAB_PATH:-}" \
    "${HOME}/IsaacLab" \
    "${HOME}/isaac/IsaacLab"
  do
    [[ -n "$cand" && -d "${cand}/source" ]] || continue
    echo "$cand"
    return 0
  done
  # Fall back to parent of venv if it looks like IsaacLab.
  if [[ -n "${ISAAC_VENV:-}" && -d "$(dirname "$ISAAC_VENV")/source" ]]; then
    dirname "$ISAAC_VENV"
    return 0
  fi
  return 1
}

ISAAC_VENV="$(_q1_pick_isaac_venv)" || {
  echo "ERROR: Isaac Lab venv not found. Set ISAAC_VENV (expected ~/IsaacLab/env_isaaclab)." >&2
  return 1 2>/dev/null || exit 1
}
ISAACLAB_PATH="$(_q1_pick_isaaclab_path)" || {
  echo "ERROR: Isaac Lab tree not found. Set ISAACLAB_PATH (expected ~/IsaacLab)." >&2
  return 1 2>/dev/null || exit 1
}

# Binary Isaac Sim install (needed for Kit / cameras). Prefer existing symlink.
if [[ ! -e "${ISAACLAB_PATH}/_isaac_sim" ]]; then
  for cand in "${HOME}/isaacsim" "${HOME}/isaac/isaacsim" "/home/will/isaacsim"; do
    if [[ -x "${cand}/python.sh" ]]; then
      ln -sfn "$cand" "${ISAACLAB_PATH}/_isaac_sim"
      echo "[isaac_env] linked ${ISAACLAB_PATH}/_isaac_sim -> $cand"
      break
    fi
  done
fi

export ISAAC_VENV ISAACLAB_PATH
export PATH="${ISAAC_VENV}/bin:${HOME}/.local/bin:${PATH:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export DISPLAY="${DISPLAY:-:0}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"

# shellcheck disable=SC1091
source "${ISAAC_VENV}/bin/activate"
