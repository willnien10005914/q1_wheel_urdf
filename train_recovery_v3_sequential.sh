#!/bin/bash
# Sequential split recovery training: prone → supine.
#   NUM_ENVS=6144 MAX_ITERS=10000 ./train_recovery_v3_sequential.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-6144}"
MAX_ITERS="${MAX_ITERS:-10000}"
ALLOW_REDESIGN="${ALLOW_REDESIGN:-1}"
export NUM_ENVS MAX_ITERS ALLOW_REDESIGN

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
PRONE_LOG="logs/train_recovery_v3_prone_split_${STAMP}.log"
SUPINE_LOG="logs/train_recovery_v3_supine_split_${STAMP}.log"
SEQ_LOG="logs/train_recovery_v3_sequential_${STAMP}.log"

{
  echo "[SEQ] start $(date -Is) NUM_ENVS=$NUM_ENVS MAX_ITERS=$MAX_ITERS"
  echo "[SEQ] === 1/2 prone ==="
  RUN_NAME="v3_prone_only_${NUM_ENVS}envs_10k" \
    ./train_recovery_v3_prone.sh headless 2>&1 | tee "$PRONE_LOG"
  echo "[SEQ] prone finished $(date -Is) exit=${PIPESTATUS[0]}"

  echo "[SEQ] === 2/2 supine ==="
  RUN_NAME="v3_supine_only_${NUM_ENVS}envs_10k" \
    ./train_recovery_v3_supine.sh headless 2>&1 | tee "$SUPINE_LOG"
  echo "[SEQ] supine finished $(date -Is) exit=${PIPESTATUS[0]}"
  echo "[SEQ] all done $(date -Is)"
} 2>&1 | tee "$SEQ_LOG"
