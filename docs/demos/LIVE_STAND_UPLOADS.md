# Stand-cut recovery re-record

Kneel wheels clamped to ±1.5 rad/s; arms calmed from kneel; clip ends after stand (~0.8s hold + 1.5s tail).

| 動作 | PPO | gofile | notes |
|---|---|---|---|
| 正躺 supine | `checkpoints/q1_recovery_v3_supine_ppo.pt` | https://gofile.io/d/zvDiT2I6 | seed=5107 stage=[3] stand=[False] hold=[0.0] steps=2000 — wide seed re-sweep in progress for a stand-success clip |
| 趴躺 prone | `checkpoints/q1_recovery_v3_prone_ppo.pt` | https://gofile.io/d/Xohblebj | seed=4119 stage=[4] stand=[True] hold≈2.32s steps=1179 stopped_after_stand |

Local mp4:

- `docs/reference/review_recovery_v3/live_supine/recovery_eval_supine.mp4`
- `docs/reference/review_recovery_v3/live_prone/recovery_eval_prone.mp4`
