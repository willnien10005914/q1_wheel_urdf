# Spark 與新版輪式 URDF 整合（2026-10-15）

以 GitHub `main` 的 `5ded8ed` 為基準，整合本機 Spark 分支 `9ba86a7` 的執行需求，保留最新 Web 起身、後翻、跪姿及滑行控制。新版 BU10 輪式 URDF 使用獨立的 Manager-based 任務與模型，既有技能仍使用原先已訓練的 24 軸機器人。

## 啟動與平台

```bash
./play_skateboard.sh
# 若 8765 已被其他程式使用：
./play_skateboard.sh --web-port 8875
```

共用 `scripts/isaac_env.sh` 支援 binary Isaac Sim 的 `python.sh` 和 pip 安裝的 Isaac Python。可設定 `ISAAC_PYTHON`、`ISAAC_SIM_PATH` 或 `ISAAC_VENV` 明確選擇 runtime；`ISAACLAB_PATH` 指定 Lab 原始碼目錄。保留有效的 `DISPLAY`，否則偵測桌機 `:0` 或 Spark `:1` 等可用顯示器。PRIME offload 改為選用 `Q1_PRIME_OFFLOAD=1`。

| 平台 | 本次驗證 |
| --- | --- |
| DGX Spark / GB10 / aarch64 | CUDA 核心、PhysX、Manager PPO、實際 Isaac X11 視窗、Web API 已執行 |
| RTX 4090 / x86_64 | 共用可攜式啟動器與 RSL-RL 相容程式；本次沒有此 GPU 的實機 |
| RTX 5080、5090 / x86_64 | 同上；需使用支援各 GPU 的 Isaac／PyTorch CUDA 安裝 |

實測軟體：Isaac Sim 5.1.0、Lab tree `VERSION=2.3.2`（套件 metadata `isaaclab=0.54.4`）、RSL-RL 5.0.1、PyTorch 2.9.0+cu130、Python 3.11.15、驅動 580.173.02。沒有修改全域 Isaac Lab 或替換共享環境套件。平台安裝請對照 [NVIDIA Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) 與 [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) 的對應版本。

RSL-RL 3.0.1 與 5.0.1 的真實 CPU runner 已驗證雙向載入 MLP checkpoint；包含正規化開／關與 scalar／log std，轉換前後推論誤差皆為 0。轉換寫入暫存快取，原始權重保持不變。可用 `tests/test_runtime_compat.py` 重現。

## 既有動作回歸

原版與整合版使用同一 GB10、checkpoint、固定種子及指令，依序執行模擬。

| 測試 | 結果 |
| --- | --- |
| Manager 滑行：seed 4107，vx=0.4，300 步／6 秒 | 原版與整合版的觀測、動作、獎勵、位置、高度、重力、指令逐值相同；RMSE=0；終止次數 0 |
| 標準跪姿 → Unbox 起身：4 個環境，1250 步 | 原版 4/4、整合版 4/4，完整紀錄相同，結束仍穩定；站穩累積約 21.1 秒 |
| Web 站姿 → 後翻：seeds 4107、5107 | 原版 2/2、整合版 2/2；沒有非法提前起跳；落地收臂 hold 2.02 秒 |
| 既有 skate 契約 | 24 action、92 actor observation、26 joints、CubeMars 力矩／電流限制及 50 步隨機動作通過 |
| Spark GUI／Web | `:1` 有實際 Isaac Sim 5.1 視窗；`GET /web/` 200；`POST /api/pose` 的 kneel 指令進入穩定跪姿 |

這些是有限步數的回歸結果，沒有把少量種子的成功率當作大規模統計。正式 Web checkpoint 沒有被小訓練覆寫。

## 小 PPO 驗證

所有本次訓練透過 `scripts/reinforcement_learning/rsl_rl/train.py`，此入口會檢查 `ManagerBasedRLEnvCfg`，拒絕 Direct 環境。`--no_publish_checkpoint`（或 `--no-publish`）只保存該次 log 下的模型。

第一輪：64 env × 24 steps × 10 iterations，延續 skate 正式模型，learning rate `1e-5`、noise std `0.03`，約 12.17 秒。相同參考觀測的 action RMSE 為 0.005228，但獨立 6 秒 rollout 的速度由約 0.343 降至 0.176 m/s，嚴格軌跡比對未通過，因此保留為實驗結果。量測工具同時比較相同輸入的策略輸出及實際閉迴路動作。

更保守的重跑同樣更新 10 次，約 11.59 秒，使用 learning rate `1e-6` 與 `--freeze_obs_normalization`；此旗標保留既有 mean/std 和正規化運算，僅停止累積新的統計。實際檢查 actor／critic 的統計完全未變，網路權重確實更新。相同觀測 action RMSE `0.002628`；獨立 6 秒測試沒有跌倒終止，平均前進速度 `0.3431 → 0.3098 m/s`（約 -9.7%）、平均獎勵約 -3.4%、高度 RMSE `0.00223 m`。維持平衡與前進，但嚴格逐步觀測／動作／獎勵門檻仍未全過，沒有宣稱等同原策略。

保守驗證模型為 `checkpoints/spark_smoke/model_20008.pt`，既有 Web 仍使用正式模型。詳細結果、模型雜湊與固定驗證門檻記錄於 [機器可讀驗證報告](spark_inertia_validation.json)。可重跑：

```bash
RESET_NOISE_STD=0.03 ./run_isaac.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Isaac-Q1-Skate-v0 --num_envs 64 --max_iterations 10 \
  --seed 42 --run_name integration_legacy_conservative --resume \
  --checkpoint checkpoints/q1_skate_ppo.pt --no-publish --freeze_obs_normalization \
  agent.algorithm.learning_rate=0.000001 agent.algorithm.schedule=fixed \
  env.curriculum.stage.params.start_iter=20000 \
  env.curriculum.action_rate.params.start_iter=20000
```

## 新版 URDF 與訓練

來源是 `~/will/q1_inertia/BU10_robot_Assets.zip` 內的 `urdf/quanta_robot_wheeled.urdf`。來源檔、整合檔與 28 個 STL 的 SHA-256 見 [資產 manifest](../../config/q1_inertia_manifest.json)。只調整 mesh 相對路徑，保留原始名稱、軸向、限位、質量、質心、完整慣量及 mimic 宣告。`qn1.py` 和 legged URDF 沒有加入訓練。

新版是 41.052262573 kg、28 links、27 movable joints，其中 4 個 mimic、23 個獨立控制軸；來源沒有頸部／頭部。既有 checkpoint 是 24 action／92 observation，不能直接套用。新增任務 `Isaac-Q1-Inertia-Skate-v0` 使用原生 21 個 position action＋2 個 wheel velocity action，actor observation 78、critic 81，物理 200 Hz、策略 50 Hz。

已逐連桿驗證 USD／PhysX 實際質量、質心與完整慣量矩陣，包含主軸旋轉重建；最大慣量絕對誤差約 `9.2e-9 kg m²`。專用 converter 明確要求匯入 URDF inertia tensor，設定也列入 USD cache hash。4 個 mimic 由零維度 action term 在每個物理子步施加 follower target；有限增益 servo 是連桿的近似，追蹤誤差會存在。自碰撞關閉，CAD mesh 使用 convex hull collider。

```bash
# 小訓練，保留正式模型：
NUM_ENVS=64 MAX_ITERS=20 ./train_inertia.sh --no_publish_checkpoint

# 查看這次 native smoke checkpoint（GUI；尚未收斂）：
./play_inertia.sh --real-time
# 或有限步數的無頭播放驗證：
./play_inertia.sh --headless --steps 20

# 較長訓練（模型仍是 native 23 軸）：
NUM_ENVS=1024 MAX_ITERS=20000 ./train_inertia.sh

# 檢查實際 PhysX 的來源慣量與 mimic：
./run_isaac.sh tests/test_inertia_contract.py --headless --num_envs 2
```

初次 native PPO：64 env × 24 steps × 20 iterations，共 30,720 transitions，約 21.59 秒；沒有 nonfinite termination，已保存 `checkpoints/inertia_native_smoke/model_19.pt`。獨立播放入口只使用新版 Manager 任務，20 步重播正常。20 次更新尚未收斂，仍會跌倒，不能視為新資產已學會穩定滑行／起身／後翻。

完整對照工具與指令見 [validation tools](../../tools/README_validation.md)。原始長軌跡和訓練 log 放在本機 `outputs/integration/`、`logs/rsl_rl/`；正式提交包含驗證摘要及獨立 smoke checkpoint，便於檢視與重跑。
