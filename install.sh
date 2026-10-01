#!/usr/bin/env bash
# Install Q1 Wheel URDF Python tooling with uv (+ optional Isaac Lab editable install).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TOOLS_ONLY=0
SKIP_ISAAC=0
for arg in "$@"; do
  case "$arg" in
    --tools-only) TOOLS_ONLY=1 ; SKIP_ISAAC=1 ;;
    --skip-isaac) SKIP_ISAAC=1 ;;
    -h|--help)
      cat <<EOF
Usage: ./install.sh [--tools-only] [--skip-isaac]

  --tools-only   Only create .venv via uv (web / MediaPipe tools). Skip Isaac Lab.
  --skip-isaac   Same Isaac skip, but still sync the default uv env.
EOF
      exit 0
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing required command: $1" >&2
    exit 1
  fi
}

echo "==> q1_wheel_urdf install"
echo "    root: $ROOT"

need_cmd curl
need_cmd python3

# --- uv -----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
need_cmd uv
echo "    uv: $(uv --version)"

# --- project venv (web UI, MediaPipe helpers, wheel_humanoid_lab meta) ---
echo "==> uv sync (default + tools extras)"
if [[ "$TOOLS_ONLY" -eq 1 ]]; then
  uv sync --extra tools
else
  uv sync --extra tools --extra isaac-meta
fi

VENV_PY="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: uv sync did not create $VENV_PY" >&2
  exit 1
fi
echo "    venv python: $VENV_PY"

# Make the lab package importable from the uv venv (non-Isaac tooling).
# Isaac Sim play/train still use env_isaaclab — do not target $VIRTUAL_ENV here.
uv pip install --python "$VENV_PY" -e "$ROOT/source/wheel_humanoid_lab"

# --- Isaac Lab editable install ----------------------------------------
ISAAC_VENV="${ISAAC_VENV:-$HOME/isaac/env_isaaclab}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac/IsaacLab}"

if [[ "$SKIP_ISAAC" -eq 0 ]]; then
  if [[ -x "$ISAAC_VENV/bin/python" ]]; then
    echo "==> Isaac Lab venv: $ISAAC_VENV"
    echo "    installing wheel_humanoid_lab editable"
    "$ISAAC_VENV/bin/python" -m pip install -e "$ROOT/source/wheel_humanoid_lab" -q
    if [[ -d "$ISAACLAB_PATH" ]]; then
      echo "    ISAACLAB_PATH=$ISAACLAB_PATH"
    else
      echo "NOTE: Isaac Lab tree not found at $ISAACLAB_PATH"
      echo "      Set ISAACLAB_PATH if your install lives elsewhere."
    fi
  else
    echo "NOTE: Isaac venv not found at $ISAAC_VENV"
    echo "      Install Isaac Lab first, or re-run with --skip-isaac / --tools-only."
    echo "      Play/train scripts expect: $ISAAC_VENV and $ISAACLAB_PATH"
  fi
else
  echo "==> skipping Isaac Lab editable install"
fi

# --- MediaPipe model hint ----------------------------------------------
MODEL_DIR="$ROOT/tools/models"
MODEL="$MODEL_DIR/pose_landmarker_heavy.task"
if [[ ! -f "$MODEL" ]]; then
  echo
  echo "NOTE: MediaPipe pose model not present (optional, for unbox extract):"
  echo "  mkdir -p $MODEL_DIR"
  echo "  # download pose_landmarker_heavy.task from Google MediaPipe model cards"
  echo "  # into $MODEL"
fi

echo
echo "==> install complete"
echo
echo "Quick start:"
echo "  Web only:     uv run python web/serve.py"
echo "  Isaac play:   ./play_skateboard.sh          # http://127.0.0.1:8766/web/"
echo "  Stop Isaac:   ./stop_isaac.sh"
echo "  Train skate:  ./train_skate.sh"
echo "  Recovery:     ALLOW_REDESIGN=1 ./train_recovery_v3_supine.sh"
echo "                ALLOW_REDESIGN=1 ./train_recovery_v3_prone.sh"
echo
echo "Activate uv venv:  source $ROOT/.venv/bin/activate"
echo "README (中文):     $ROOT/README.md"
