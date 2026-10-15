#!/usr/bin/env bash
# Current native manager entry point.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/play_skills.sh"  "$@"
