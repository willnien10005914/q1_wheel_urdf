#!/usr/bin/env bash
# Native URDF manager backflip training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/train_backflip.sh" "$@"
