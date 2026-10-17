#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-native-deadline.service; then
  echo 'Deadline campaign is already running.'; exit 0
fi
exec systemd-run --user --unit=q1-native-deadline --collect \
  --description='Q1 deadline training, validation and GitHub delivery before Oct 19 reboot' \
  --property=KillMode=control-group --property=Nice=5 --working-directory="$ROOT" \
  /usr/bin/python3 -u "$ROOT/tools/deadline_campaign.py"
