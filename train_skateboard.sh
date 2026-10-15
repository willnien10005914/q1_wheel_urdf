#!/usr/bin/env bash
# Native URDF manager training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/train_native.sh" walk "$@"
