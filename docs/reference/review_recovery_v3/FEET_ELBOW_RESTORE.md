# Recovery restore: feet/knee/wheels/waist/hips + elbow-preferential plant

## Supine 6144×20k (done)
- Peak stand **79% at iter 8000** → collapsed to **34%** by kneel+lift farming.
- Eval 16 envs: peak **12/16 stood**, final **0/16** (stage-3 kneel).
- Videos: [peak8000](https://gofile.io/d/46SQWJMx) · [final19999](https://gofile.io/d/clJljFUy)
- Deploy ckpt: `checkpoints/q1_recovery_v3_supine_ppo.pt` (= peak8000)

## Anti-farm fix (for prone + future)
- Fade kneel after latch; cut lift once knelt
- `kneel_linger` penalty; higher stand weight

## Prone
Restarted `6144 × 20000` with anti-farm rewards after aborting the ~66-iter early start.
