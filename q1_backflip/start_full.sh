#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json,pathlib
base=pathlib.Path('logs/q1_backflip')
for name in ['probe256_100','capacity6144_flip3']:
 matches=sorted(base.glob('*_'+name+'/completed.json'))
 if not matches:raise SystemExit('missing gate: '+name)
 d=json.loads(matches[-1].read_text())
 if not d['finite']:raise SystemExit('nonfinite gate: '+name)
print('Numeric and capacity gates passed. No successful backflip yet.')
PY
if [[ -f full20k.pid ]]; then
  read -r existing_pid < full20k.pid
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Training already running: $existing_pid"
    exit 0
  fi
fi
nohup setsid bash q1_backflip/launch.sh --num_envs 6144 --max_iterations 20000 --resume logs/q1_backflip/2026-10-08_19-18-11_probe256_100/model_99.pt --warm_start --run_name full6144_20k_wheels > full6144_20k.log 2>&1 < /dev/null &
printf '%s\n' "$!" > full20k.pid
cat full20k.pid
