# Split recovery PPO review videos

Training finished (6144 envs × 10000 iters each). Recordings use **2999 steps ≈ 60 s** so stand is not cut off after kneel.

| Mode | Seed | Stand hold | gofile |
|---|---:|---:|---|
| Prone / 趴躺 | 4119 | ~6.2 s | https://gofile.io/d/hwX7CfTN |
| Supine / 正躺 | 5107 | ~15.3 s | https://gofile.io/d/z9w7axZf |

Checkpoints: `checkpoints/q1_recovery_v3_{prone,supine}_ppo.pt`  
Local: `docs/reference/review_recovery_v3/split_{prone,supine}/recovery_eval_*.mp4`
