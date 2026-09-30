# Split recovery: independent PPOs for 正躺 / 趴躺

**Goal:** stop training one mixed policy for both floor starts. Prior `v3_smooth_arms_full_4096`
(4096 envs × 20k iters, both modes) left **supine kneel→stand at 2/48** while **prone was 48/48**.
`v3_both_stand_full_4096` had the opposite asymmetry (prone stand 9/48). One network fighting
two geometries is the failure mode we are removing.

## What changes

| Item | Before (mixed v3) | Split plan |
|---|---|---|
| Gym task | `Isaac-Q1-RecoveryV3-v0` (`mode=-1`, env_id%2) | `…-Supine-v0` / `…-Prone-v0` |
| Checkpoint | `q1_recovery_contact_v3_ppo.pt` | `q1_recovery_v3_supine_ppo.pt` / `…_prone_ppo.pt` |
| Stand unlock | needs **both** modes ≥3 kneels | unlock when **active** mode(s) ≥3 kneels |
| `num_envs` | 4096 (~56% GPU on RTX 5080 Laptop) | **6144** default (fallback 4096 on OOM) |
| `max_iterations` | 20000 (reward flat ~10–12k) | **12000** |
| Foot stacking | self-collision **off** globally | soft `foot_apart` reward + no PhysX self-collision flip |
| 內八 (pigeon-toe) | PPO hip_roll residuals free | 75% prior blend on hip_roll + `hip_square` penalty |
| Supine stand | stand weight 10, often stuck at kneel | stand **14**, kneel **6**, episode 45 s |

## Launch (from `q1_wheel/wheel_humanoid_urdf`)

```bash
# Smoke (minutes)
NUM_ENVS=256 MAX_ITERS=50 ./train_recovery_v3_supine.sh
NUM_ENVS=256 MAX_ITERS=50 ./train_recovery_v3_prone.sh

# Full budget (serial — one job at a time on this GPU)
ALLOW_REDESIGN=1 ./train_recovery_v3_supine.sh
ALLOW_REDESIGN=1 ./train_recovery_v3_prone.sh

# If 6144 OOMs:
NUM_ENVS=4096 ALLOW_REDESIGN=1 ./train_recovery_v3_supine.sh
```

Logs land under:

- `logs/rsl_rl/q1_recovery_v3_supine/<timestamp>_v3_supine_only_*`
- `logs/rsl_rl/q1_recovery_v3_prone/<timestamp>_v3_prone_only_*`

Final convenience copies: `checkpoints/q1_recovery_v3_{supine,prone}_ppo.pt`.

## Eval

```bash
./evaluate_recovery_v3.sh --mode supine --checkpoint checkpoints/q1_recovery_v3_supine_ppo.pt \
  --num_envs 16 --seed 4107 --out docs/reference/review_recovery_v3/split_supine
./evaluate_recovery_v3.sh --mode prone --checkpoint checkpoints/q1_recovery_v3_prone_ppo.pt \
  --num_envs 16 --seed 4119 --out docs/reference/review_recovery_v3/split_prone
```

Acceptance (same bar as prior independent evals): **≥14/16 kneel→stand** per mode across 3 seeds.

## Play / web

`play_recovery_bridge` loads split checkpoints when present and picks by lie mode
(正躺 → supine PPO, 趴躺 → prone PPO). Override with `RECOVERY_CHECKPOINT=…`.

## Foot collision note

Global `enabled_self_collisions=False` is intentional (waist↔torso / roller↔shin overlaps).
We do **not** turn it on for this round. Instead:

1. Reward L–R wheel separation (`foot_apart`, weight 2–3).
2. Penalize adduct hip_roll (`hip_square`, weight −3).
3. Damp hip_roll residuals toward motor prior (0) at 75%.

If feet still stack after split training, next step is filtered L↔R wheel-only collision
(requires enabling self-collisions **and** filtering the known bad pairs).

## Why single-mode should help supine stand

Smooth-arms redesign slowed floor→kneel and added arm-calm; that helped prone but
starved supine upright transfer under independent seeds. A supine-only PPO:

- never spends capacity on prone contact geometry,
- unlocks stage-4 stand from its own kneel curriculum,
- gets a higher stand / lower kneel reward ratio so kneel is not a local optimum.

## Suggested order

1. Smoke both scripts (50 iters) — confirm `cap` reaches 4 and no TensorDict crash.
2. Train **prone** first (already strong baseline; validates pipeline + foot rewards).
3. Train **supine** with the same budget; if still kneel-stuck, raise `stand` further or
   shorten upright preposition hold in `motor_priors.json` for supine only (separate follow-up).
4. Wire play + re-eval; keep mixed `q1_recovery_contact_v3_ppo.pt` only as fallback.
