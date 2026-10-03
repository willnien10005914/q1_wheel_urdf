#!/bin/bash
# Small-batch recovery probe: train briefly, print plant/kneel/stand rates.
# Usage: MODE=prone NUM_ENVS=256 MAX_ITERS=150 ./tools/probe_recovery_small.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"

MODE="${MODE:-prone}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERS="${MAX_ITERS:-150}"
TAG="${TAG:-probe}"
RUN_NAME="v3_${MODE}_feet_restore_${TAG}_${NUM_ENVS}e_${MAX_ITERS}it"
LOG="logs/probe_${MODE}_${TAG}_${NUM_ENVS}e_${MAX_ITERS}it.log"
mkdir -p logs

echo "[probe] mode=$MODE envs=$NUM_ENVS iters=$MAX_ITERS run=$RUN_NAME"
NUM_ENVS="$NUM_ENVS" MAX_ITERS="$MAX_ITERS" RUN_NAME="$RUN_NAME" \
  "./train_recovery_v3_${MODE}.sh" headless 2>&1 | tee "$LOG"

python3 - "$LOG" "$MODE" <<'PY'
import re,sys
from pathlib import Path
log=Path(sys.argv[1]).read_text(errors='ignore'); mode=sys.argv[2]
# last curriculum block
keys=[
 f'{mode}_plant_success',f'{mode}_kneel_success',f'{mode}_upright_kneel_success',
 f'{mode}_kneel_then_stand_success','stage_cap','Mean reward'
]
idx=max((i for i,l in enumerate(log.splitlines()) if 'Learning iteration' in l), default=-1)
block='\n'.join(log.splitlines()[idx:idx+120]) if idx>=0 else log[-8000:]
print('===== PROBE SUMMARY', mode, '=====')
for k in keys:
  for l in block.splitlines():
    if k in l:
      print(l.strip()); break
# also max plant across all iters
plants=[float(m.group(1)) for m in re.finditer(rf'{mode}_plant_success:\s*([0-9.]+)', log)]
kneels=[float(m.group(1)) for m in re.finditer(rf'{mode}_kneel_success:\s*([0-9.]+)', log)]
stands=[float(m.group(1)) for m in re.finditer(rf'{mode}_kneel_then_stand_success:\s*([0-9.]+)', log)]
caps=[float(m.group(1)) for m in re.finditer(r'stage_cap:\s*([0-9.]+)', log)]
print(f'max_plant={max(plants) if plants else None} max_kneel={max(kneels) if kneels else None} max_stand={max(stands) if stands else None} max_cap={max(caps) if caps else None}')
# gate: plant>0.05 or kneel>0 or cap>=4
ok=(max(plants or [0])>=0.05) or (max(kneels or [0])>0) or (max(caps or [0])>=4)
print('PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
PY
