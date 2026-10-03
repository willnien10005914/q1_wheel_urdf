# Elbow-only + waist-assist recovery retrain

Retrain independent 正躺 / 趴躺 PPOs with contact and spin constraints from hardware review.

## Goals

1. **Steady arms** through kneel→stand (no thrash / shake).
2. **Elbow-only floor assist** — wrists/grippers must not jam the floor (gripper cracks).
3. **Waist joints help** torso rise (pitch effort rewarded; yaw thrash discouraged).
4. **No whole-body spin** from kneel→stand (wheels muted at kneel; yaw-rate + L/R differential taxed).

## Budget

| Item | Value |
|---|---|
| `num_envs` | **4096** supine / **6144** prone |
| `max_iterations` | **15000** |
| Order | **supine → prone** (serial on one GPU) |
| Tasks | `Isaac-Q1-RecoveryV3-Supine-v0` / `…-Prone-v0` |
| Run names | `v3_supine_elbow_waist_4096envs_15000it`, `v3_prone_elbow_waist_6144envs_15000it` |

## MDP changes (summary)

| Change | Where |
|---|---|
| `elbow_force` / `distal_force` split | `mdp.py` State |
| Plant / assist use elbows only | rewards `plant`, `arm_assist` |
| `gripper_floor` penalty (−6) | distal contact + low distal height |
| Stronger `arm_calm` (−4) from stage≥2 | reward + arm prior blend 0.7→0.95 |
| Wrist/gripper actions locked to prior | ContactPositionAction |
| `waist_assist` (+3) | waist pitch help × torso upright |
| Stage-3 wheel residual **zero**; sit-back mute | ContactWheelVelocityAction |
| `yaw_spin` (−4) | yaw rate + wheel differential |

Obs adds `distal_force` (2) next to elbow force → **fresh train required** (old ckpts incompatible).

## Launch

```bash
# Smoke
NUM_ENVS=64 MAX_ITERS=5 ./train_recovery_v3_supine.sh

# Full sequential (supine 4096 → prone 6144)
NUM_ENVS=4096 PRONE_NUM_ENVS=6144 MAX_ITERS=15000 ./train_recovery_v3_sequential.sh
```

Logs: `logs/train_recovery_v3_sequential_elbow_*.log`  
Checkpoints: `checkpoints/q1_recovery_v3_{supine,prone}_ppo.pt` (copied at end of each run).
