#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/web"
npx --yes esbuild app.js --bundle --format=esm --outfile=app.bundle.js
echo "wrote $ROOT/web/app.bundle.js ($(wc -c < app.bundle.js) bytes)"
