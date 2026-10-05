# Boot / unbox pose analysis (no gripper)

## Verdict

**Train only operator-assisted kneel → stand** (`Isaac-Q1-RecoveryV3-BootKneel-v0`).

Do **not** dual-train supine+prone floor plant for the product boot path.

## Why this wins

| Init | Gripper needed? | Power-button OK? | Train difficulty | Notes |
|------|-----------------|------------------|------------------|-------|
| Supine floor → stand | Elbow plant often; gripper re-touch after kneel | Button under back — awkward for long storage | High (peak kneel→stand fragile; armpark stand 0%) | Keep only as optional full recovery |
| Prone floor → stand | Less arm plant; still floor contact stages | Back button accessible while packed | Medium–high | Good for storage, still hard RL |
| Arms-to-head on floor | Helps keep tips clear | Same as orientation | Marginal alone | Useful **pre-pose**, not a full strategy |
| **Assisted kneel (human sits up)** | **No** — arms parked off floor | Unbox prone, press while supporting, sit into kneel | **Lowest** | Policy only learns kneel→stand on wheels/rollers/hips/waist |

## Product procedure (recommended)

1. **Store / ship prone** (belly down) so the back power button stays reachable after long storage.
2. Unbox; **manually park arms** toward head/chest (shoulder pitch ≈ 0.35, elbow ≈ −1.1 — same as kneel prior; grippers off floor).
3. Operator **sits the torso up** into kneel: knee rollers + foot wheels planted, hips/waist near kneel priors (`KNEEL_*`).
4. Power on (or power while supporting, then release into kneel).
5. Policy runs **kneel → stand only** — no gripper plant.

## Evidence from this repo

- Supine feet-restore: peak stand ~79% @8k then farmed down; videos still show post-kneel arm/gripper re-touch.
- Supine armpark hard lock: kneel ~27%, **stand 0%** (arms frozen before a viable rise).
- Prone often reaches stand more cleanly once knelt, but full floor curriculum is still expensive.
- Motor priors already park arms at pitch 0.35 / elbow −1.1 at kneel — matches “hands toward head” assist.

## Probe → scale plan

```bash
# small-batch probe
NUM_ENVS=256 MAX_ITERS=500 RUN_NAME=v3_boot_kneel_probe ./train_recovery_v3_boot_kneel.sh

# if cumulative kneel_then_stand_success ≳ 0.4 by ~300–500 iters:
NUM_ENVS=4096 MAX_ITERS=5000 RUN_NAME=v3_boot_kneel_full ./train_recovery_v3_boot_kneel.sh
```

Gate: stand reward > 0 and `supine_kneel_then_stand_success` rising (boot mode reports under supine_* diagnostics because it reuses mode=0 sit-back stand priors).

## Probe result (2026-10-05)

Run: `v3_boot_kneel_probe_256e_500it` → `logs/rsl_rl/q1_recovery_v3_boot_kneel/2026-10-05_18-28-15_v3_boot_kneel_probe_256e_500it`  
Stable ckpt: `checkpoints/q1_recovery_v3_boot_kneel_ppo.pt`

| Iter | Stand reward | Cumulative kneel→stand |
|------|--------------|------------------------|
| ~100 | ~8–12 | **~0.79** |
| ~300 | ~13–16 | **~0.83** |
| **499 (final)** | — | **0.836** |

Gate **passed**. Scaled next: `4096×5000` full boot-kneel train (single mode only).

## Full train + demo (2026-10-05)

- Run: `v3_boot_kneel_full_4096e_5k` → final cumulative kneel→stand **0.943**
- Ckpt: `checkpoints/q1_recovery_v3_boot_kneel_ppo.pt` (= `model_4999.pt`)
- Demo (1 env, stop after stand ~4.8s): https://gofile.io/d/bAjDlQFs  
  Local: `docs/reference/review_recovery_v3/live_boot_kneel/recovery_eval_boot_kneel.mp4`
