#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$HOME/isaac/env_isaaclab/bin/activate"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only

TRAIN_LOG="$ROOT/logs/train_recovery_v3_smooth_arms.log"
TRAIN_PID_FILE="$ROOT/logs/train_recovery_v3_smooth_arms.pid"
RUN_DIR="$ROOT/logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_20-43-53_v3_smooth_arms_full_4096"

echo "=== WAIT TRAIN $(date -Is) ==="
if [[ -f "$TRAIN_PID_FILE" ]]; then
  TRAIN_PID=$(cat "$TRAIN_PID_FILE")
  while ps -p "$TRAIN_PID" >/dev/null 2>&1; do
    iter=$(rg -o 'Learning iteration [0-9]+/20000' "$TRAIN_LOG" | tail -1 || true)
    echo "waiting $iter $(date +%H:%M:%S)"
    sleep 120
  done
fi
for i in $(seq 1 60); do
  rg -q 'Training time:' "$TRAIN_LOG" && break
  sleep 5
done
echo "=== TRAIN DONE MARKERS ==="
rg -n 'Training time:|Training checkpoint:.*model_19999|Traceback|out of memory' "$TRAIN_LOG" | tail -15 || true

CKPT="$RUN_DIR/model_19999.pt"
[[ -f "$CKPT" ]] || CKPT=$(ls -1t "$RUN_DIR"/model_*.pt | head -1)
echo "USING CKPT=$CKPT"
test -f "$CKPT"

python -m pip install -e source/wheel_humanoid_lab -q
mkdir -p logs/recovery_v3_smooth_arms_ppo_evaluation \
  docs/reference/review_recovery_v3/smooth_arms_supine \
  docs/reference/review_recovery_v3/smooth_arms_prone

for SEED in 4107 5107 6107; do
  echo "=== EVAL seed=$SEED $(date -Is) ==="
  bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode both --num_envs 32 --seed "$SEED" --steps 1999 \
    --out "logs/recovery_v3_smooth_arms_ppo_evaluation/seed_${SEED}"
done

python3 - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
root=Path('logs/recovery_v3_smooth_arms_ppo_evaluation')
rows=[]
for p in sorted(root.glob('seed_*/evaluation.json')):
 j=json.loads(p.read_text()); s=j['summary']
 for mode,name in enumerate(['supine','prone']):
  idx=[i for i,m in enumerate(s['mode']) if m==mode]
  def rate(k):
   vals=[bool(s[k][i]) for i in idx]; return sum(vals), len(vals)
  rows.append((j.get('seed'), name, rate('kneel'), rate('upright_kneel'), rate('kneel_then_stand')))
agg=defaultdict(lambda:[0,0,0,0,0,0]); lines=[]
for seed,name,k,u,st in rows:
 lines.append(f'seed={seed} {name}: kneel {k[0]}/{k[1]} upright {u[0]}/{u[1]} stand {st[0]}/{st[1]}')
 for i,r in enumerate((k,u,st)):
  agg[name][2*i]+=r[0]; agg[name][2*i+1]+=r[1]
lines.append('---AGGREGATE---')
for name in ['supine','prone']:
 a=agg[name]; lines.append(f'{name}: kneel {a[0]}/{a[1]} upright {a[2]}/{a[3]} stand {a[4]}/{a[5]}')
text='\n'.join(lines); (root/'summary.txt').write_text(text+'\n'); print(text)
PY

find_success() {
  local mode=$1; shift
  local seed
  for seed in "$@"; do
    local out="logs/recovery_v3_smooth_arms_ppo_evaluation/probe_${mode}_${seed}"
    echo "=== probe $mode seed=$seed ==="
    bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode "$mode" --seed "$seed" --steps 1999 --out "$out"
    if python3 -c "import json;from pathlib import Path;j=json.loads(Path('$out/evaluation.json').read_text());import sys;sys.exit(0 if j['summary']['kneel_then_stand'][0] else 1)"; then
      echo "$seed"; return 0
    fi
  done
  return 1
}

SUPINE_SEED=$(find_success supine 4107 4119 4123 4200 5107 6107 || echo 4107)
PRONE_SEED=$(find_success prone 4119 4107 4123 4200 4300 5107 6107 || echo 4107)
echo "VIDEO_SEEDS supine=$SUPINE_SEED prone=$PRONE_SEED"

echo "=== VIDEO SUPINE seed=$SUPINE_SEED ==="
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode supine --seed "$SUPINE_SEED" --video --steps 1999 \
  --out docs/reference/review_recovery_v3/smooth_arms_supine
echo "=== VIDEO PRONE seed=$PRONE_SEED ==="
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode prone --seed "$PRONE_SEED" --video --steps 1999 \
  --out docs/reference/review_recovery_v3/smooth_arms_prone

echo "=== UPLOAD ==="
python3 tools/upload_gofile.py docs/reference/review_recovery_v3/smooth_arms_supine/recovery_eval_supine.mp4 docs/reference/review_recovery_v3/smooth_arms_supine.gofile.json
python3 tools/upload_gofile.py docs/reference/review_recovery_v3/smooth_arms_prone/recovery_eval_prone.mp4 docs/reference/review_recovery_v3/smooth_arms_prone.gofile.json
echo "=== ALL DONE $(date -Is) ==="
