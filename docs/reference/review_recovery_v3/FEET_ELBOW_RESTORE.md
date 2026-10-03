# Recovery restore: feet/knee/wheels/waist/hips + elbow-only plant

## What went wrong
The previous elbow redesign hard-banned distal contact for plant and muted kneel
wheels. Every env stayed at **stage 0** (no foot/knee roller usage).

## Restore goals
1. Keep **foot active wheels + knee passive rollers + waist + hip_pitch (AKE90)** as in the last standing policy.
2. Prefer **elbow** floor assist; soft-tax grippers/wrists (no hard plant ban).
3. Kneel wheel residual **±1.5** (help plant, limit spin) — not zero.
4. Small-batch probe until plant/kneel rates rise, then **6144 × 20000** supine → prone.

## Small-batch gate
`MODE=prone NUM_ENVS=256 MAX_ITERS=150 ./tools/probe_recovery_small.sh`

Pass if any of: `plant≥5%`, `kneel>0`, or `stage_cap≥4`.

## Large batch (only after pass)
```bash
NUM_ENVS=6144 MAX_ITERS=20000 PRONE_NUM_ENVS=6144 ./train_recovery_v3_sequential.sh
```
