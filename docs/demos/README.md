# Q1 訓練示範影片 ↔ PPO

本頁整理**最後有效**策略的 Isaac Sim 錄影，以及對應的 checkpoint 路徑。影片已進倉庫；gofile 為外連備份（可能過期，以 repo 內 mp4 為準）。

## 對照表

| 動作 | 示範影片（repo） | gofile | PPO checkpoint | Gym task |
|---|---|---|---|---|
| **正躺起身** supine | [`review_recovery_v3/split_supine/recovery_eval_supine.mp4`](../reference/review_recovery_v3/split_supine/recovery_eval_supine.mp4) | [z9w7axZf](https://gofile.io/d/z9w7axZf) | `checkpoints/q1_recovery_v3_supine_ppo.pt` | `Isaac-Q1-RecoveryV3-Supine-v0` |
| **趴躺起身** prone | [`review_recovery_v3/split_prone/recovery_eval_prone.mp4`](../reference/review_recovery_v3/split_prone/recovery_eval_prone.mp4) | [hwX7CfTN](https://gofile.io/d/hwX7CfTN) | `checkpoints/q1_recovery_v3_prone_ppo.pt` | `Isaac-Q1-RecoveryV3-Prone-v0` |
| **站立滑行** skate | [`results/q1_skate_cubemars_iter3800.mp4`](../results/q1_skate_cubemars_iter3800.mp4) | [Oh6jh3Qd](https://gofile.io/d/Oh6jh3Qd) | `checkpoints/q1_skate_ppo.pt` | `Isaac-Q1-Skate-v0` |
| **側滑 · 左右腳前後** slide foreaft | [`results/q1_slide_foreaft_iter32000.mp4`](../results/q1_slide_foreaft_iter32000.mp4) | [FEl7B5IT](https://gofile.io/d/FEl7B5IT) | `checkpoints/q1_slide_ppo.pt` | `Isaac-Q1-Slide-v0` |
| **側滑 · 抬腳通過** slide passlift | [`results/q1_slide_passlift_iter42000.mp4`](../results/q1_slide_passlift_iter42000.mp4) | [pNhvqoqe](https://gofile.io/d/pNhvqoqe) | 同上 `q1_slide_ppo.pt` | 同上 |
| **側滑 · 跨步** slide stride | [`results/q1_slide_stride_iter32000.mp4`](../results/q1_slide_stride_iter32000.mp4) | [PjnkwYEl](https://gofile.io/d/PjnkwYEl) | 同上 `q1_slide_ppo.pt` | 同上 |

### Raw GitHub（本分支推送後可播）

將 `<branch>` 換成實際分支名（例：`main` 或 `cursor/demo-videos-ppo-7d44`）：

- 正躺：`https://github.com/willnien10005914/q1_wheel_urdf/raw/<branch>/docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.mp4`
- 趴躺：`https://github.com/willnien10005914/q1_wheel_urdf/raw/<branch>/docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.mp4`
- 滑行：`https://github.com/willnien10005914/q1_wheel_urdf/raw/<branch>/docs/results/q1_skate_cubemars_iter3800.mp4`
- 左右腳前後：`https://github.com/willnien10005914/q1_wheel_urdf/raw/<branch>/docs/results/q1_slide_foreaft_iter32000.mp4`

## 播放設定（Isaac）

前提見根目錄 [`README.md`](../../README.md)：Isaac Lab 2.3、`~/isaac/env_isaaclab`、本倉 `./install.sh`。

```bash
# 網頁 + Isaac（載入 skate；按鈕可切 slide / recovery）
./play_skateboard.sh
# http://127.0.0.1:8766/web/
# 建議：正躺或趴躺 → Recovery PPO 起身 → Walk / Slide → WASD

# 僅重錄 skate（headless）
./record_skate.sh

# 重錄 split recovery（需 Isaac）
./evaluate_recovery_v3.sh --mode supine --checkpoint checkpoints/q1_recovery_v3_supine_ppo.pt \
  --seed 5107 --video --steps 2999 --out docs/reference/review_recovery_v3/split_supine
./evaluate_recovery_v3.sh --mode prone --checkpoint checkpoints/q1_recovery_v3_prone_ppo.pt \
  --seed 4119 --video --steps 2999 --out docs/reference/review_recovery_v3/split_prone
```

網頁按鈕與策略對應：正躺／趴躺只擺姿；**Recovery PPO 起身**依上次躺姿選 `…_supine_ppo.pt` 或 `…_prone_ppo.pt`；**Walk / skate** → `q1_skate_ppo.pt`；**Slide mode** → `q1_slide_ppo.pt`。

## 備註

- Recovery split 細節與評測：[`../reference/review_recovery_v3/SPLIT_VIDEOS.md`](../reference/review_recovery_v3/SPLIT_VIDEOS.md)
- Skate 訓練結果說明（英文）：根目錄 [`README.en.md`](../../README.en.md) Results 一節
- 淘汰權重勿用：`q1_getup_ppo.pt`、混合 `q1_recovery_contact_v3_ppo.pt`、`skateboard_ppo.pt`
