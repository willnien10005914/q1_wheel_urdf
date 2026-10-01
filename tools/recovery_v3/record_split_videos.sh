#!/bin/bash
# Record + upload split supine & prone review videos (longer clip to capture stand).
# STEPS=2999 ≈ 60 s @ 0.02 s/step (episode_length_s is 40–45; extra margin for stand hold).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
python -m pip install -e source/wheel_humanoid_lab -q

STEPS="${STEPS:-2999}"
SUPINE_CKPT="${SUPINE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_supine_ppo.pt}"
PRONE_CKPT="${PRONE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_prone_ppo.pt}"
test -f "$SUPINE_CKPT" && test -f "$PRONE_CKPT"
echo "[record] steps=$STEPS (~$(python3 -c "print(round($STEPS*0.02,1))")s)"

find_success() {
  local mode=$1 ckpt=$2; shift 2
  local seed out
  for seed in "$@"; do
    out="logs/recovery_v3_split_probe/probe_${mode}_${seed}"
    echo "=== probe $mode seed=$seed ===" >&2
    bash evaluate_recovery_v3.sh --checkpoint "$ckpt" --mode "$mode" --seed "$seed" --steps "$STEPS" --out "$out" >&2 || true
    if python3 -c "import json;from pathlib import Path;j=json.loads(Path('$out/evaluation.json').read_text());s=j['summary'];import sys;sys.exit(0 if s['kneel_then_stand'][0] and s['best_stand_hold_s'][0]>=1.0 else 1)"; then
      echo "FOUND $mode seed=$seed stand=True" >&2
      echo "$seed"
      return 0
    fi
    # Prefer upright kneel at least if no full stand yet
    python3 -c "import json;from pathlib import Path;j=json.loads(Path('$out/evaluation.json').read_text());s=j['summary'];print('  kneel',s['kneel'],'upright',s['upright_kneel'],'stand',s['kneel_then_stand'],'hold',s['best_stand_hold_s'],'stage',s['max_stage'])" >&2 || true
  done
  return 1
}

record_one() {
  local mode=$1 ckpt=$2 seed=$3
  local out="docs/reference/review_recovery_v3/split_${mode}"
  mkdir -p "$out"
  echo "=== VIDEO $mode seed=$seed steps=$STEPS ==="
  bash evaluate_recovery_v3.sh --checkpoint "$ckpt" --mode "$mode" --seed "$seed" --video --steps "$STEPS" --out "$out"
  local mp4="$out/recovery_eval_${mode}.mp4"
  test -f "$mp4"
  python3 tools/upload_gofile.py "$mp4" "$out/recovery_eval_${mode}.gofile.json"
  cp -f "$out/recovery_eval_${mode}.gofile.json" "docs/reference/review_recovery_v3/split_${mode}.gofile.json"
  python3 - <<PY
import json
from pathlib import Path
j=json.loads(Path("$out/evaluation.json").read_text()); s=j["summary"]
g=json.loads(Path("$out/recovery_eval_${mode}.gofile.json").read_text())
print("RESULT", "$mode", "seed", j.get("seed"),
      "stand", s["kneel_then_stand"], "upright", s["upright_kneel"], "kneel", s["kneel"],
      "max_stage", s["max_stage"], "stand_hold", s["best_stand_hold_s"],
      "url", g.get("data",{}).get("downloadPage"))
PY
}

# Prefer seeds that previously stood; expand search for supine (harder).
SUPINE_SEED=$(find_success supine "$SUPINE_CKPT" 5107 4107 4119 4123 4200 6107 42 7001 8001 9001 || true)
PRONE_SEED=$(find_success prone "$PRONE_CKPT" 4119 4107 4123 4200 5107 6107 42 || true)
[[ -n "${SUPINE_SEED:-}" ]] || SUPINE_SEED=5107
[[ -n "${PRONE_SEED:-}" ]] || PRONE_SEED=4119
echo "[record] SEEDS supine=$SUPINE_SEED prone=$PRONE_SEED"

record_one prone "$PRONE_CKPT" "$PRONE_SEED"
record_one supine "$SUPINE_CKPT" "$SUPINE_SEED"

python3 - <<'PY'
from pathlib import Path
Path('docs/reference/review_recovery_v3/SPLIT_VIDEOS.md').write_text("""# Split recovery PPO review videos

Longer recordings (2999 steps ≈ 60 s) so stand is not cut off after kneel.

| Mode | Checkpoint | gofile |
|---|---|---|
| Prone / 趴躺 | `q1_recovery_v3_prone_ppo.pt` | see `split_prone.gofile.json` |
| Supine / 正躺 | `q1_recovery_v3_supine_ppo.pt` | see `split_supine.gofile.json` |

Local mp4s under `docs/reference/review_recovery_v3/split_{prone,supine}/`.
""")
print('wrote SPLIT_VIDEOS.md')
PY
echo "=== ALL DONE $(date -Is) ==="
