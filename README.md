# Q1 / BU10 輪式人形：新版 URDF、Manager-based PPO 與 Web

目前主要 Web、播放與技能訓練均使用 `urdf/q1_inertia/quanta_robot_wheeled.urdf`，由 Isaac Lab `ManagerBasedRLEnv` 執行。來源為 `~/will/q1_inertia/BU10_robot_Assets.zip` 的輪式模型；總重約 **41.0523 kg**，23 個主動關節加 4 個 mimic 關節。沒有虛擬頸部軸，也沒有使用 legged `qn1.py`。

新的共同策略契約為 **23 action / 90 actor observation / 90 critic observation**，物理 200 Hz、策略 50 Hz。Stand、Walk、Standup、Backflip 與 Web 共用相同資產、馬達模型、manager action 及 observation。舊 24 維 PPO 不會自動套用到新機器人。

## Web

```bash
./play_skateboard.sh
# http://127.0.0.1:8875/web/
```

頁面顯示原生 27 關節模型，4 個 mimic 為唯讀。可選 Stand、Walk、Kneel、Stand up、Backflip；W/S 前後、A/D 轉向、空白停止、R 重設。動作切換只改控制器目標，不寫入模擬姿態；只有明確 Reset 會重設機器人。

策略尚未訓練完成時，頁面會顯示平衡／姿態控制或訓練中的 PPO。後空翻必須已有原生 checkpoint 才能使用。預覽模型和通過驗證的模型分別標示，訓練更新不代表技能已成功。

無頭常駐 Web：

```bash
./start_native_web.sh
systemctl --user status q1-native-web
systemctl --user stop q1-native-web
```

## 大量訓練佇列

```bash
./start_native_training.sh
python3 tools/native_training_queue.py --status
journalctl --user -u q1-native-training -f
# 中止並保留進度：
systemctl --user stop q1-native-training
# 再啟動時從本次有效 checkpoint 續跑：
./start_native_training.sh
```

沒有 user systemd 的環境可用 `./start_native_training.sh --foreground`。

預設計畫見 [config/native_training_queue.json](config/native_training_queue.json)：

| 順序 | 任務 | 平行環境 | PPO updates |
| --- | --- | ---: | ---: |
| 1 | Stand 站立平衡 | 1024 | 5,000 |
| 2 | Walk 雙輪行進與轉向 | 1024 | 20,000 |
| 3 | Standup 跪姿起身 | 1024 | 20,000 |
| 4 | Backflip 後空翻 | 1024 | 40,000 |

共 85,000 次 PPO 更新，名義約 **20.89 億 transitions**。每 1000 updates 分段續跑並驗證，訓練和驗證依序使用 GPU。Walk、Standup、Backflip 從原生 Stand 模型 warm-start，使用新的任務迭代計數；不載入舊 24 軸模型。

`outputs/native_manager/queue/status.json` 保存階段、進度、PID、log、驗證狀態與來源 SHA-256。保存中的不完整 checkpoint 不會被選用。若來源控制器改變，既有實驗會停止並要求新的 state 路徑，避免無聲混用不同物理設定。GPU 記憶體不足時會降低環境數並有限次重試。

完成更新預算但沒有通過驗證會標為 `completed_with_unvalidated_skills`，不會冒充技能完成。後翻驗證要求實際離地、完整後轉與穩定落地，沒有 root pose／速度輔助。

## Checkpoints 與進度

- `checkpoints/native_manager/<skill>/latest.pt`：本次訓練預覽，可尚未通過動作驗證。
- `checkpoints/native_manager/<skill>/model.pt`：通過獨立驗證的較佳模型。
- `validated.json` / `latest.json`：來源、迭代、雜湊與驗證紀錄。
- `logs/rsl_rl/q1_native_*`：原始模型、TensorBoard 與設定快照。
- `logs/native_manager_queue`：每段訓練與驗證 log。

Web 自動檢查新的原生模型，優先使用通過驗證的版本。大量訓練模型與 log 保存在本機，不會把每次中間權重寫入 Git。

## 個別訓練

```bash
NUM_ENVS=1024 MAX_ITERS=20000 ./train_native.sh walk
./train_stand.sh
./train_backflip.sh
./train_posture.sh       # native Standup
./train_skate.sh         # native Walk
```

需要隔離測試、不更新預覽檔時加 `--no-publish`。`RESUME=1 CHECKPOINT=/path/model.pt` 可續跑同一個原生任務；跨任務 warm-start 使用 `--reset_iteration`，佇列會自動處理。

主要 train.py 會檢查 Manager 環境和新版資產路徑；舊 URDF 任務會被拒絕。歷史 Direct 後翻程式仍保留供查閱，其 `launch.sh` 已改成新版 manager 入口。尚未遷移的舊 recovery／MediaPipe 工具不屬於這個新技能佇列。

## Runtime 與物理

支援 pip Isaac Sim Python 及 binary `python.sh`，也會偵測有效的 X display（包含 Spark `:1`）。可設定 `ISAAC_PYTHON`、`ISAACLAB_PATH`、`ISAAC_SIM_PATH` 或 `ISAAC_VENV` 指定既有安裝；不會自動替換你的 CUDA／Isaac 套件。

主要馬達為 CubeMars **explicit actuator**（電壓／電流限制、齒輪、摩擦、延遲），mimic 使用 implicit servo；Direct／Manager 是環境組織方式，與 actuator 的 explicit／implicit 分類不同。慣量 converter 明確要求匯入來源 tensor，來源與 PhysX 已逐連桿核對。mimic servo 與凸包碰撞為模擬近似，自碰撞目前關閉。

DGX Spark / GB10 已實測；RTX 4090／5080／5090 使用同一可攜式程式入口，但仍需平台相符的 Isaac Sim／PyTorch CUDA 安裝，本機沒有這三張卡可做實機驗證。

## 驗證

```bash
./run_isaac.sh tools/check_runtime.py --require-cuda
./run_isaac.sh tests/test_inertia_contract.py --headless --num_envs 2
./run_isaac.sh tests/test_native_manager_skills.py --skill Stand --steps 200 --out outputs/native_stand.json
python3 tests/test_native_queue.py
```

[新版 manager 遷移與實測紀錄](docs/training/native_manager_2026-10-15.md)；[前一版 Spark／慣量整合歷史](docs/training/spark_inertia_2026-10-15.md)。歷史 checkpoint 保留，但不是新版 Web 的 fallback。
