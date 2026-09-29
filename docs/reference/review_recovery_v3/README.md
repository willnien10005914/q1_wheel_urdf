# Q1 physical recovery v3 — Cursor handoff

**Full-budget PPO finished** (4096 envs × 20000 iters). These are simulation results, not an approved hardware demonstration.

## Full-budget training

| Field | Value |
|---|---|
| Command | `NUM_ENVS=4096 MAX_ITERS=20000 RUN_NAME=v3_both_stand_full_4096 ./train_recovery_v3.sh headless` |
| Log | `logs/train_recovery_v3_full.log` |
| Run dir | `logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-24-17_v3_both_stand_full_4096` |
| Final checkpoint | `.../model_19999.pt` (also `checkpoints/q1_recovery_contact_v3_ppo.pt`) |
| Wall time | ~19.3 h (`Training time: 69693.41 seconds`) |
| Seed | 42 (fresh; did not resume the 100-iter short checkpoint) |

Cumulative training rates at the last iteration (include early kneel-only curriculum; **do not** treat as final policy success):

| Floor start | upright kneel | kneel→stand |
|---|---:|---:|
| Supine | 94.0% | 85.4% |
| Prone | 98.0% | 93.4% |

## Independent deterministic PPO evaluation (post-train)

3 seeds × 32 envs (`mode=both`, 16 supine + 16 prone each): seeds **4107 / 5107 / 6107**.

| Floor start | Four-wheel assisted kneel | Upright kneel | Kneel then stand ≥1 s |
|---|---:|---:|---:|
| Supine / 仰躺（正躺） | 48/48 | 48/48 | **47/48** |
| Prone / 趴姿（反躺） | 48/48 | 48/48 | **9/48** |

Per-seed detail:

```
seed=4107 supine: kneel 16/16 upright 16/16 stand 16/16 unbroken 16/16
seed=4107 prone: kneel 16/16 upright 16/16 stand 5/16 unbroken 16/16
seed=5107 supine: kneel 16/16 upright 16/16 stand 16/16 unbroken 16/16
seed=5107 prone: kneel 16/16 upright 16/16 stand 2/16 unbroken 16/16
seed=6107 supine: kneel 16/16 upright 16/16 stand 15/16 unbroken 16/16
seed=6107 prone: kneel 16/16 upright 16/16 stand 2/16 unbroken 16/16
---AGGREGATE---
supine: kneel 48/48 upright 48/48 stand 47/48
prone: kneel 48/48 upright 48/48 stand 9/48
```

Artifacts: `logs/recovery_v3_full_ppo_evaluation/` and `INDEX.json` in this folder.

**Reading:** supine stand is strong; prone still reaches upright kneel reliably but often fails the stand transfer under independent seeds. Training cumulative prone-stand (~93%) overstates the final policy's deterministic prone-stand rate — keep these metrics separate. Short-run gate had prone stand 16/16 on `model_99.pt`; full-budget `model_19999.pt` regressed prone stand.

## Isaac Sim review videos

| Mode | Seed | Local mp4 | gofile | Stood? |
|---|---:|---|---|---|
| Supine / 正躺 | 4107 | `supine_full/recovery_eval_supine.mp4` | https://gofile.io/d/u4JTNiel | yes (~21 s hold) |
| Prone / 反躺（典型失败） | 4107 | `prone_full/recovery_eval_prone.mp4` | https://gofile.io/d/dkrVZrC8 | no (upright kneel only) |
| Prone / 反躺（成功示范） | 4119 | `prone_full_success/recovery_eval_prone.mp4` | https://gofile.io/d/8WW49xzs | yes (~22 s hold) |

```bash
CKPT=logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-24-17_v3_both_stand_full_4096/model_19999.pt
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode both --num_envs 32 --seed 4107 --out logs/recovery_v3_full_ppo_evaluation/seed_4107 --steps 1499
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode supine --seed 4107 --video --out docs/reference/review_recovery_v3/supine_full --steps 1499
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode prone --seed 4107 --video --out docs/reference/review_recovery_v3/prone_full --steps 1499
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode prone --seed 4119 --video --out docs/reference/review_recovery_v3/prone_full_success --steps 1499
```

Controller remains **residual PPO + contact-gated motor priors + wheel balance feedback**. Match each `.pt` to its run `controller_snapshot/` / `controller_manifest.json`.

## Short-run gate evidence (provenance only)

Short checkpoint `.../2026-09-29_00-04-54_v3_both_stand_short/model_99.pt` previously evaluated seed 4107 as supine stand 15/16, prone stand 16/16. That gate authorized the full budget; it is not the full-run policy result above.
