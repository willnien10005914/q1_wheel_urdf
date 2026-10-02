# Stand-cut recovery re-record

Kneel wheels clamped to ±1.5 rad/s at stage 3; arms calmed from kneel; clip ends after stand (~0.8s hold + 1.5s tail) when stand succeeds.

| 動作 | PPO | gofile | notes |
|---|---|---|---|
| 正躺 supine | `checkpoints/q1_recovery_v3_supine_ppo.pt` | https://gofile.io/d/zvDiT2I6 | seed=5107; reaches kneel (stage 3) but **no upright/stand** on 1-env and 16-env sweeps (0/16×5+ seeds). Split supine PPO needs retrain for kneel→stand. Clip is full 2000 steps (~40s). |
| 趴躺 prone | `checkpoints/q1_recovery_v3_prone_ppo.pt` | https://gofile.io/d/Xohblebj | seed=4119; **stand success** at ~22.1s; stopped early (ran_steps=1179, hold≈2.32s). |

Local mp4:

- `docs/reference/review_recovery_v3/live_supine/recovery_eval_supine.mp4`
- `docs/reference/review_recovery_v3/live_prone/recovery_eval_prone.mp4`

## FAQ (this re-record)

**每次模擬錄影動作會不一樣嗎？**  
會。Eval 用 deterministic mean action，但不同 `--seed` 會改變初始姿態 / domain randomization，軌跡就不同。同一 seed 大致可重現，PhysX/GPU 仍可能有微小差異。

**起身時手會抖，需要重新訓練嗎？**  
若要讓手臂明顯變穩，**需要再訓**（或加 inference 端手臂指令平滑）。`arm_calm` 只是 reward shaping，已載入的 PPO 權重不會因為改 reward 就變乖；目前只靠 eval 端 kneel 輪速 clamp + 既有 arm_calm 權重。Prone 已能站起；抖手若仍明顯，下一步是對 prone/supine 做 fine-tune（提高 `arm_calm` / stand calm 項）而不是只改錄影參數。

**Prone kneel 輪子轉圈？**  
Stage 3 輪速 residual 已 clamp 到 ±1.5 rad/s（floor / stand 仍可用更大 residual）。
