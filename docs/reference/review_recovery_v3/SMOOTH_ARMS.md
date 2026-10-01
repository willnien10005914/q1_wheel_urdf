# Recovery v3 redesign: slower floor→kneel + anti-flail arms

## Why standing arms looked random

The reviewed videos used `model_19999.pt` from **4096 envs × 20000 iters** (not 200000).
Late training `Mean action noise std` grew to ~1.15, while standing only weakly penalized
`action_rate`/`action_size`. Residual PPO could thrash unloaded arms after kneel/stand.

## Why floor→kneel looked like a fast-forward

Motor-prior stage blends were planned for ~2.0 s, but the controller advanced as soon as
`elapsed >= 1.5` and a 0.25 s contact hold passed. That cut each blend at ~75% and jumped
to the next keyframe. Example old supine timeline (seed 4107 PPO): 0 → 1.5 → 3.1 → 4.6 → 7.1 s.

## Changes (this redesign)

1. Longer early `transition_duration_s` (3.5 s for plant / wheel / press).
2. Advance only after `elapsed >= duration[mode, stage]` (finish the blend).
3. New rewards: `arm_assist` (push while rising) and `arm_calm` (penalize unloaded thrash).
4. Standing reward now multiplies by a quiet-arm factor.
5. Stronger global `action_rate` / `action_size` penalties; lower PPO entropy / desired KL.
6. `ContactPositionAction` shrinks/blends arm residuals toward the motor prior once arms are unloaded at stage ≥ 3.
7. Episode length 40 s so the slower kneel still leaves room to stand.

## Prior-only smoke (no PPO)

`logs/recovery_v3_smooth_prior_smoke/supine/` seed 4107:

```
t=0.00s stage=0
t=3.50s stage=1
t=7.10s stage=2
t=10.60s stage=3
t=27.10s stage=4
```

## Retrain

Previous full-budget gate hashes are intentionally stale after this redesign.

```bash
ALLOW_REDESIGN=1 NUM_ENVS=4096 MAX_ITERS=20000 RUN_NAME=v3_smooth_arms_full_4096 ./train_recovery_v3.sh headless
```
