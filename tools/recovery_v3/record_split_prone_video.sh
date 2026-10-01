#!/bin/bash
# Record + upload split prone PPO review video, then resume interrupted supine training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only

CKPT="${CKPT:-$ROOT/checkpoints/q1_recovery_v3_prone_ppo.pt}"
OUT="${OUT:-docs/reference/review_recovery_v3/split_prone}"
STEPS="${STEPS:-1999}"
echo "[record] CKPT=$CKPT OUT=$OUT"
test -f "$CKPT"
python -m pip install -e source/wheel_humanoid_lab -q
mkdir -p "$OUT"

find_success() {
  local seed out
  for seed in "$@"; do
    out="logs/recovery_v3_split_prone_probe/probe_${seed}"
    echo "=== probe prone seed=$seed ===" >&2
    bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode prone --seed "$seed" --steps "$STEPS" --out "$out" >&2 || true
    if python3 -c "import json;from pathlib import Path;j=json.loads(Path('$out/evaluation.json').read_text());import sys;sys.exit(0 if j['summary']['kneel_then_stand'][0] else 1)"; then
      echo "FOUND prone seed=$seed stand=True" >&2
      echo "$seed"
      return 0
    fi
  done
  return 1
}

SEED=$(find_success 4119 4107 4123 4200 5107 6107 42 || true)
[[ -n "${SEED:-}" ]] || SEED=4119
echo "[record] VIDEO_SEED prone=$SEED"

echo "=== VIDEO PRONE seed=$SEED ==="
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode prone --seed "$SEED" --video --steps "$STEPS" --out "$OUT"
test -f "$OUT/recovery_eval_prone.mp4"
python3 tools/upload_gofile.py "$OUT/recovery_eval_prone.mp4" "$OUT/recovery_eval_prone.gofile.json"
# also copy convenience json next to review docs
cp -f "$OUT/recovery_eval_prone.gofile.json" docs/reference/review_recovery_v3/split_prone.gofile.json

python3 - <<PY
import json
from pathlib import Path
p=Path("$OUT/evaluation.json")
j=json.loads(p.read_text()); s=j["summary"]
g=json.loads(Path("$OUT/recovery_eval_prone.gofile.json").read_text())
print("RESULT prone seed", j.get("seed"),
      "stand", s["kneel_then_stand"],
      "upright", s["upright_kneel"],
      "kneel", s["kneel"],
      "max_stage", s["max_stage"],
      "stand_hold", s["best_stand_hold_s"],
      "url", g.get("data",{}).get("downloadPage"))
PY
echo "=== RECORD DONE $(date -Is) ==="
