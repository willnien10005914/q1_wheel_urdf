#!/bin/bash
# Train the two mode policies one after the other (single GPU): posture (kneel <-> stand), then
# slide (push-skate, warm-started from the skate PPO). Logs go to logs/train_posture.log and
# logs/train_slide.log; final weights land in checkpoints/q1_posture_ppo.pt / q1_slide_ppo.pt.
#   nohup ./train_modes.sh > logs/train_modes.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

echo "[train_modes] $(date '+%F %T') posture start"
./train_posture.sh > logs/train_posture.log 2>&1
echo "[train_modes] $(date '+%F %T') posture done (exit $?)"

echo "[train_modes] $(date '+%F %T') slide start"
./train_slide.sh > logs/train_slide.log 2>&1
echo "[train_modes] $(date '+%F %T') slide done (exit $?)"
