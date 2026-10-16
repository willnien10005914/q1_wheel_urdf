#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-native-web.service; then
  echo "Q1 native manager Web is already running; check its listener with: ss -ltn 'sport = :8875'"
  exit 0
fi
exec systemd-run --user --unit=q1-native-web --collect \
  --description="Q1 native CAD manager Web player" \
  --property=Restart=on-failure --property=RestartSec=10 --property=Nice=5 \
  --property=StartLimitIntervalSec=300 --property=StartLimitBurst=3 \
  --property=KillMode=control-group --working-directory="$ROOT" \
  "$ROOT/play_skateboard.sh" --headless --no-browser --no-keyboard --web-port 8875 --web-host "${Q1_WEB_HOST:-127.0.0.1}" "$@"
