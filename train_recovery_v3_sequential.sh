#!/bin/bash
# Sequential split recovery training: 正躺(supine) → 趴躺(prone).
#   NUM_ENVS=4096 MAX_ITERS=15000 ./train_recovery_v3_sequential.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-15000}"
export NUM_ENVS MAX_ITERS

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SUPINE_LOG="logs/train_recovery_v3_supine_elbow_${STAMP}.log"
PRONE_LOG="logs/train_recovery_v3_prone_elbow_${STAMP}.log"
SEQ_LOG="logs/train_recovery_v3_sequential_elbow_${STAMP}.log"

{
  echo "[SEQ] start $(date -Is) NUM_ENVS=$NUM_ENVS MAX_ITERS=$MAX_ITERS"
  echo "[SEQ] redesign: elbow-only assist, arm calm, waist assist, kneel wheel mute / no yaw-spin"
  echo "[SEQ] === 1/2 supine (正躺) ==="
  RUN_NAME="v3_supine_elbow_waist_${NUM_ENVS}envs_${MAX_ITERS}it" \
    ./train_recovery_v3_supine.sh headless 2>&1 | tee "$SUPINE_LOG"
  echo "[SEQ] supine finished $(date -Is) exit=${PIPESTATUS[0]}"

  echo "[SEQ] === 2/2 prone (趴躺) ==="
  RUN_NAME="v3_prone_elbow_waist_${NUM_ENVS}envs_${MAX_ITERS}it" \
    ./train_recovery_v3_prone.sh headless 2>&1 | tee "$PRONE_LOG"
  echo "[SEQ] prone finished $(date -Is) exit=${PIPESTATUS[0]}"
  echo "[SEQ] all done $(date -Is)"
} 2>&1 | tee "$SEQ_LOG"
