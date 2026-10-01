# Recovery v3 smooth-arms full PPO — results

**Training finished**: 4096 envs × 20000 iters, run `v3_smooth_arms_full_4096`.
Checkpoint: `logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_20-43-53_v3_smooth_arms_full_4096/model_19999.pt`
Wall time: ~18.8 h (`Training time: 67525.96 seconds`). Action noise std ended ~0.17 (vs ~1.15 on the prior full run).

## Cumulative training rates (last iter; include curriculum — not final eval)

| Mode | kneel | upright kneel | kneel→stand |
|---|---:|---:|---:|
| Supine | ~98.7% | ~66%* | ~63%* |
| Prone | ~97.8% | ~97.6% | ~96% |

\*Exact last-iter values are in `logs/train_recovery_v3_smooth_arms.log`.

## Independent deterministic eval (3 seeds × 32 envs)

| Floor start | kneel | upright kneel | kneel→stand |
|---|---:|---:|---:|
| Supine / 正躺 | 43/48 | **2/48** | **2/48** |
| Prone / 反躺 | 48/48 | **48/48** | **48/48** |

```
seed=4107 supine: kneel 15/16 upright 0/16 stand 0/16
seed=4107 prone: kneel 16/16 upright 16/16 stand 16/16
seed=5107 supine: kneel 16/16 upright 2/16 stand 2/16
seed=5107 prone: kneel 16/16 upright 16/16 stand 16/16
seed=6107 supine: kneel 12/16 upright 0/16 stand 0/16
seed=6107 prone: kneel 16/16 upright 16/16 stand 16/16
---AGGREGATE---
supine: kneel 43/48 upright 2/48 stand 2/48
prone: kneel 48/48 upright 48/48 stand 48/48
```

**Reading:** prone is strong after the redesign. Supine often sticks around supported-press / fails upright kneel under independent seeds — the slower floor→kneel + arm-calm changes helped prone a lot but hurt supine upright transfer. Do not mix these with cumulative training rates.

## Isaac review videos

| Mode | Seed | Stood? | gofile |
|---|---:|---|---|
| Supine / 正躺 | 5107 | no (kneel / stage 2) | https://gofile.io/d/mGDNYrWk |
| Prone / 反躺 | 4119 | yes (~21.7 s hold) | https://gofile.io/d/XbTdcTCE |

Local: `docs/reference/review_recovery_v3/smooth_arms_supine/recovery_eval_supine.mp4`, `.../smooth_arms_prone/recovery_eval_prone.mp4`.

See also `SMOOTH_ARMS.md` for the redesign rationale.
