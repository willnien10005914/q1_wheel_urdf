#!/bin/bash
# Alias for ./train_unbox.sh (supine → kneel). Stand-up is ./train_posture.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/train_unbox.sh" "$@"
