#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --foreground ]]; then
  shift
  exec python3 -u "$ROOT/tools/reference_campaign.py" "$@"
fi
if [[ "${1:-}" == --install-service ]]; then
  python3 - "$ROOT" <<'INSTALL'
from pathlib import Path
import os,sys
root=Path(sys.argv[1]).resolve()
q1_config_home=Path(os.environ.get('XDG_CONFIG_HOME',str(Path.home()/'.config')))
target=q1_config_home/'systemd/user/q1-wheel-reference-training.service'
marker='# Managed by Q1 wheel reference campaign launcher'
if target.exists() and not target.read_text().startswith(marker):
    raise SystemExit('Existing service is not owned by this launcher; refusing overwrite')
target.parent.mkdir(parents=True,exist_ok=True)
text=(root/'config/systemd/q1-wheel-reference-training.service.in').read_text().replace('@ROOT@',str(root))
tmp=target.with_suffix('.tmp');tmp.write_text(text);tmp.replace(target)
print('Installed',target)
INSTALL
  systemctl --user daemon-reload
  exec systemctl --user enable --now q1-wheel-reference-training.service
fi
if [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/q1-wheel-reference-training.service" ]]; then
  exec systemctl --user start q1-wheel-reference-training.service
fi
if systemctl --user is-active --quiet q1-wheel-reference-training.service; then
  echo 'Video-reference campaign is already running or waiting.'
  exec python3 "$ROOT/tools/reference_campaign.py" --status
fi
exec systemd-run --user --unit=q1-wheel-reference-training --collect \
  --description='Q1 video-reference PPO after q1_leg handover; recording and GitHub/GoFile delivery' \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=RestartPreventExitStatus=78 --property=StartLimitIntervalSec=300 \
  --property=StartLimitBurst=3 --property=KillMode=control-group --property=TimeoutStopSec=90 \
  --property=Nice=5 --working-directory="$ROOT" \
  /usr/bin/python3 -u "$ROOT/tools/reference_campaign.py" "$@"
