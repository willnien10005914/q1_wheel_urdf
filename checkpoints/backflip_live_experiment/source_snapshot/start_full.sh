#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f v2_full20k.pid ]]; then
  read -r existing_pid < v2_full20k.pid
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Q1 v2 full pipeline already running: $existing_pid"
    exit 0
  fi
fi
nohup setsid python3 -u q1_backflip_v2/full_pipeline.py "$@" > v2_full_pipeline.log 2>&1 < /dev/null &
printf '%s\n' "$!" > v2_full20k.pid
cat v2_full20k.pid
