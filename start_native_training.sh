#!/usr/bin/env bash
# Run the persistent sequential queue; --foreground also works without systemd.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --foreground ]]; then
  shift
  exec python3 -u "$ROOT/tools/native_training_queue.py" "$@"
fi
if systemctl --user is-active --quiet q1-native-training.service; then
  echo "q1-native-training is already running."
  exec python3 "$ROOT/tools/native_training_queue.py" --status
fi
exec systemd-run --user --unit=q1-native-training --collect \
  --description="Q1 native manager sequential PPO training" \
  --property=Restart=on-failure --property=RestartSec=60 --property=Nice=5 \
  --property=RestartPreventExitStatus=78 \
  --property=StartLimitIntervalSec=300 --property=StartLimitBurst=3 \
  --property=KillMode=control-group --working-directory="$ROOT" \
  /usr/bin/python3 -u "$ROOT/tools/native_training_queue.py" "$@"
