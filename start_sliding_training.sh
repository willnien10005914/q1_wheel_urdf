#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-native-slide-training.service; then
  echo 'Sliding follow-up queue is already active.'; exit 0
fi
exec systemd-run --user --unit=q1-native-slide-training --collect \
  --description="Q1 sliding PPO after current native queue" \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=RestartPreventExitStatus=78 --property=StartLimitIntervalSec=300 \
  --property=StartLimitBurst=3 --property=KillMode=control-group \
  --property=Nice=5 --working-directory="$ROOT" \
  /usr/bin/python3 -u "$ROOT/tools/queue_native_slide.py"
