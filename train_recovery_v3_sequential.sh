#!/bin/bash
# Sequential split recovery: 正躺 → 趴躺.
# Defaults for full run after small-batch gate passes:
#   NUM_ENVS=6144 PRONE_NUM_ENVS=6144 MAX_ITERS=20000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-6144}"
PRONE_NUM_ENVS="${PRONE_NUM_ENVS:-6144}"
MAX_ITERS="${MAX_ITERS:-20000}"
export MAX_ITERS

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SUPINE_LOG="logs/train_recovery_v3_supine_feet_${STAMP}.log"
PRONE_LOG="logs/train_recovery_v3_prone_feet_${STAMP}.log"
SEQ_LOG="logs/train_recovery_v3_sequential_feet_${STAMP}.log"

{
  echo "[SEQ] start $(date -Is) SUPINE_ENVS=$NUM_ENVS PRONE_ENVS=$PRONE_NUM_ENVS MAX_ITERS=$MAX_ITERS"
  echo "[SEQ] feet/knee/wheels/waist/hips restored; elbow-preferential plant; soft gripper tax"
  echo "[SEQ] === 1/2 supine (正躺) envs=$NUM_ENVS ==="
  NUM_ENVS="$NUM_ENVS" RUN_NAME="v3_supine_feet_elbow_${NUM_ENVS}envs_${MAX_ITERS}it" \
    ./train_recovery_v3_supine.sh headless 2>&1 | tee "$SUPINE_LOG"
  echo "[SEQ] supine finished $(date -Is) exit=${PIPESTATUS[0]}"

  echo "[SEQ] === 2/2 prone (趴躺) envs=$PRONE_NUM_ENVS ==="
  NUM_ENVS="$PRONE_NUM_ENVS" RUN_NAME="v3_prone_feet_elbow_${PRONE_NUM_ENVS}envs_${MAX_ITERS}it" \
    ./train_recovery_v3_prone.sh headless 2>&1 | tee "$PRONE_LOG"
  echo "[SEQ] prone finished $(date -Is) exit=${PIPESTATUS[0]}"
  echo "[SEQ] all done $(date -Is)"
} 2>&1 | tee "$SEQ_LOG"
