# Q1 physical recovery v3 — Cursor handoff

Both-mode short-run acceptance passed; **full-budget PPO is running**. These are simulation results, not an approved hardware demonstration. Video recording/upload remains assigned to Cursor after the full run finishes.

## Full-budget training (launched)

| Field | Value |
|---|---|
| Command | `NUM_ENVS=4096 MAX_ITERS=20000 RUN_NAME=v3_both_stand_full_4096 ./train_recovery_v3.sh headless` |
| Gate | `logs/recovery_v3_gate.json` (`allow_full_budget: true`, schema 3) |
| Log | `logs/train_recovery_v3_full.log` |
| PID file | `logs/train_recovery_v3_full.pid` |
| Run dir | `logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-24-17_v3_both_stand_full_4096` |
| Hardware | RTX 5080 Laptop 16 GB (~5.8 GB used at 4096 envs; ~3 s/iter; ETA ~17 h) |
| Seed | 42 (fresh run; does not resume the 100-iter short checkpoint) |

Short-run evidence that opened the gate remains below. Do not mix cumulative training curriculum rates with the independent deterministic evaluation.

## Latest deterministic PPO evaluation

| Floor start | Four-wheel assisted kneel | Upright kneel, arms unloaded | Kneel then stand ≥1 s |
|---|---:|---:|---:|
| Supine / 仰躺 | 16/16 | 15/16 | 15/16 |
| Prone / 趴姿 | 16/16 | 16/16 | 16/16 |

Evaluation: `logs/recovery_v3_both_stand_ppo_evaluation/evaluation.json` (seed 4107). Checkpoint: `/home/testpc-3/projects/q1_wheel/wheel_humanoid_urdf/logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-04-54_v3_both_stand_short/model_99.pt`.
Training log: `logs/train_recovery_v3_both_stand.log`. Training rates count completed episodes cumulatively, including the initial kneel-only curriculum; use the independent evaluation for the final policy's success rate.

The controller is **residual PPO plus contact-gated motor priors and wheel balance feedback**. A successful zero-action motor-prior rollout is not proof that PPO learned recovery unaided. Old `.pt` files require their own controller snapshot: source and priors are frozen in each new run's `controller_snapshot/` and `controller_manifest.json`.

## What changed

* Real PhysX/CubeMars motor control, with root/joint state writes only at reset. Stage changes require measured sustained support.
* Capture the successful kneel command before interpolation pushes it past the stable posture. Separate four-wheel assisted kneel from torso-upright, unloaded-arm kneel.
* Supine first unloads arms through a separate leg adjustment, then raises the torso. Before standing it sits back to move the pelvis toward the foot wheels; wheel balance feedback starts with leg extension.
* Shift mass before extending the legs. Standing wheel control uses pitch, pitch rate and forward velocity feedback, with PPO residuals. Combined wheel target is limited to ±25 rad/s; current/torque/back-EMF plant limits remain active. The earlier recovery-only ±8 rad/s target limit remains the residual bound.
* Full-budget gate requires both assisted and upright kneel in both modes, training and deterministic evaluation, validated standing priors enabled in both modes, and matching current source hashes. Old gate schema cannot authorize a new run.
* Evaluations report first-episode validity, actual joint angles, joint targets, wheel velocity targets, torso orientation, contacts, torque/current, saturation and hold times. A reset cannot combine partial successes into recovery evidence.

Assisted kneel: all four wheel/roller vertical forces >5 N, pelvis height 0.32–0.65 m, pelvis upright >0.7, speed <0.6 m/s for 0.4 s. Upright kneel adds torso upright >0.85 and each arm force <30 N for 0.5 s. Standing requires wheel support, pelvis height >0.72 m, pelvis and torso upright >0.9, arm forces <30 N and speed <0.5 m/s for 1 s after kneeling. Contacts are EMA-filtered at 50 Hz.

## Web review and recording

Latest exported Web trace: `logs/recovery_v3_both_stand_ppo_review/evaluation.json`; controller: `ppo`. This is actual physical telemetry, interpolated for display. Stage labels describe the observed phase, not acceptance by themselves. Initial FK proposals (`?version=3` without `physics=1`) remain geometry-only and are not the reviewed physical motion.

Forward port 8765 in Cursor Remote and open:
`http://localhost:8765/web/getup_review.html?version=3&physics=1`

If needed, start `python3 web/serve.py --port 8765 --no-browser` from the repo root.

Cursor can record each mode separately with the current model and controller:

```bash
cd ~/projects/q1_wheel/wheel_humanoid_urdf
bash evaluate_recovery_v3.sh --checkpoint /home/testpc-3/projects/q1_wheel/wheel_humanoid_urdf/logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-04-54_v3_both_stand_short/model_99.pt --mode supine --seed 4107 --video --out docs/reference/review_recovery_v3/supine_latest --steps 1499
bash evaluate_recovery_v3.sh --checkpoint /home/testpc-3/projects/q1_wheel/wheel_humanoid_urdf/logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_00-04-54_v3_both_stand_short/model_99.pt --mode prone --seed 4107 --video --out docs/reference/review_recovery_v3/prone_latest --steps 1499
```

Outputs are `recovery_eval_supine.mp4` / `recovery_eval_prone.mp4` plus separate `evaluation.json`. Use `--hold-kneel` to inspect the kneeling phase alone. Omit `--checkpoint` to inspect the motor prior without PPO. One robot per video; multi-env spacing is 5 m. No video was recorded/uploaded by Codex.

## Evidence and limits

* `docs/reference/getup_candidates_v3/motor_priors.json`: motor priors, staged transfer and balance parameters; still candidates pending user review.
* `motor_prior_knots.csv`: desired stage targets; `physics_stage_contacts.csv`: measured transition contacts.
* `physics_joint_trajectory.csv`: actual sampled URDF joint angles, position targets and wheel velocity targets (radians/rad/s), without visualization interpolation. These are review candidates, not user-approved ground truth.
* `logs/recovery_v3_stand_velocity_validation.json`: repeated prone floor-to-stand candidate trials.
* `logs/recovery_v3_upright_preposition_validation.json`: repeated supine two-phase upright-kneel trials.
* `logs/recovery_v3_stand_sit_back_validation.json`: repeated supine floor-to-stand trials.
* `logs/recovery_v3_both_stand_validation/evaluation.json`: integrated zero-action motor-prior control, 16/16 floor→kneel→stand in each mode (seed 3107).
* `logs/recovery_v3_gate.json`: current acceptance decision.
* `ROOT_CAUSE.md`: v2 failures and v3 changes.

Run `python tools/recovery_v3/test_contact_gates.py` in the Isaac environment for contact/balance regressions. Full training has not been launched. Self-collisions and continuous thermal derating remain limitations of the inherited plant; these results do not establish hardware safety or robustness to randomized mass/COM.
