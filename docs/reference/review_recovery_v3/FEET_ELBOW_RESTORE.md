# Recovery restore: feet/knee/wheels/waist/hips + elbow-preferential plant

## Root cause
Elbow-only hard plant + kneel wheel mute left every env at **stage 0**.

## Fixes
- Restore hand plant gate + **leg escape** (prone/supine) so foot wheels / knee rollers advance stages.
- Kneel wheel residual **±1.5** (help, not spin).
- Soft **gripper excess** tax (prefer forearm, don't freeze plant).
- Sticky stage-0, shorter early blends, hip/waist/lift rewards.

## Small-batch gates (512 envs × 600 iters)

| Mode | Plant | Kneel | Stand | Cap |
|------|------:|------:|------:|----:|
| Prone (趴躺) | 99.8% | 80.2% | 79.1% | 4 |
| Supine (正躺) | 99.5% | 95.8% | 35.0% | (rising) |

Supine stand still climbing at 600 iters; large batch should finish it.

## Large batch
```bash
NUM_ENVS=6144 PRONE_NUM_ENVS=6144 MAX_ITERS=20000 ./train_recovery_v3_sequential.sh
```
