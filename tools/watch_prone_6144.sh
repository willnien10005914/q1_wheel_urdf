#!/bin/bash
# After current supine (4096) finishes, ensure prone runs with 6144 envs.
# The in-flight sequential job still has NUM_ENVS=4096 exported; this watcher
# intercepts a 4096 prone launch (or a gap with no train) and starts 6144.
set -uo pipefail
ROOT=/home/will/projects/q1_wheel_urdf
cd "$ROOT"
LOG=logs/train_recovery_v3_prone_6144_watch.log
mkdir -p logs
exec >>"$LOG" 2>&1
echo "[watch] start $(date -Is) — wait for supine end, launch prone NUM_ENVS=6144"

supine_alive() { pgrep -f 'train.py.*RecoveryV3-Supine' >/dev/null; }
prone_4096() { pgrep -f 'train.py.*RecoveryV3-Prone.*--num_envs 4096' >/dev/null; }
prone_any() { pgrep -f 'train.py.*RecoveryV3-Prone' >/dev/null; }
prone_6144() { pgrep -f 'train.py.*RecoveryV3-Prone.*--num_envs 6144' >/dev/null; }

while supine_alive; do
  echo "[watch] $(date -Is) supine still running"
  sleep 30
done
echo "[watch] $(date -Is) supine gone"

# Give sequential a moment to spawn prone-4096 if it will.
for i in 1 2 3 4 5 6 7 8 9 10; do
  if prone_4096; then
    echo "[watch] $(date -Is) caught prone-4096 — stopping it and sequential wrapper"
    pkill -f 'train.py.*RecoveryV3-Prone' || true
    pkill -f 'train_recovery_v3_sequential' || true
    sleep 8
    break
  fi
  if prone_6144; then
    echo "[watch] $(date -Is) prone-6144 already running — nothing to do"
    exit 0
  fi
  if prone_any; then
    echo "[watch] $(date -Is) prone running (checking args)"
    pgrep -af 'RecoveryV3-Prone' || true
  fi
  sleep 3
done

if prone_6144; then
  echo "[watch] $(date -Is) prone-6144 already running"
  exit 0
fi

# Stop any leftover sequential so it cannot respawn 4096 prone.
pkill -f 'train_recovery_v3_sequential' || true
pkill -f 'train.py.*RecoveryV3-Prone.*--num_envs 4096' || true
sleep 5

if prone_any; then
  echo "[watch] $(date -Is) unexpected prone still alive; refusing to double-launch"
  pgrep -af 'RecoveryV3-Prone' || true
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
PRONE_LOG="logs/train_recovery_v3_prone_elbow_${STAMP}.log"
echo "[watch] $(date -Is) launching prone NUM_ENVS=6144 MAX_ITERS=15000 log=$PRONE_LOG"
NUM_ENVS=6144 MAX_ITERS=15000 \
  RUN_NAME="v3_prone_elbow_waist_6144envs_15000it" \
  ./train_recovery_v3_prone.sh headless 2>&1 | tee "$PRONE_LOG"
echo "[watch] $(date -Is) prone finished exit=${PIPESTATUS[0]}"
