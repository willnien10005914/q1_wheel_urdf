#!/usr/bin/env bash
# Native URDF + ManagerBasedRLEnv Web is the default player.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/play_skills.sh" "$@"
