# Q1 / BU10 輪式人形：新版 URDF、Manager-based PPO 與 Web

目前主要 Web、播放與技能訓練均使用 `urdf/q1_inertia/quanta_robot_wheeled.urdf`，由 Isaac Lab `ManagerBasedRLEnv` 執行。來源為 `~/will/q1_inertia/BU10_robot_Assets.zip` 的輪式模型；總重約 **41.0523 kg**，23 個主動關節加 4 個 mimic 關節。沒有虛擬頸部軸，也沒有使用 legged `qn1.py`。

新的共同策略契約為 **23 action / 90 actor observation / 90 critic observation**，物理 200 Hz、策略 50 Hz。Stand、Walk、Standup、Backflip 與 Web 共用相同資產、馬達模型、manager action 及 observation。舊 24 維 PPO 不會自動套用到新機器人。

## 本次修改與新舊 URDF 比較

本次採用 `~/will/q1_inertia/BU10_robot_Assets.zip` 內的新版 CAD 輪式 URDF。**舊版已經有 inertia；這次改用新版來源資料，沒有自行估算或增加慣量數值。** 新版 URDF 只修改 mesh 相對路徑，來源的 link、joint、質量、質心、慣量、軸向、限位與 mimic 標籤均保留。來源檔案與整合後檔案的 SHA-256 見 [q1_inertia_manifest.json](config/q1_inertia_manifest.json)。

| 項目 | 舊版 `urdf/wheel_humanoid.urdf` | 新版 `urdf/q1_inertia/quanta_robot_wheeled.urdf` |
| --- | ---: | ---: |
| 總質量 | 40.000 kg | 41.0523 kg（增加約 2.63%） |
| 含 inertial 的 link | 28 | 28 |
| 主動關節／PPO action | 24 | 23 |
| 單個輪子質量 | 1.2439 kg | 1.6388 kg（增加約 31.7%） |
| 輪子 `Iyy` | 0.007734 kg·m² | 0.00913136 kg·m²（增加約 18.1%） |
| 左髖 Pitch 軸向 | `0 1 0` | `0 -1 0` |
| 左髖 Pitch 限位 | −1.8～1.2 rad | −1.5708～1.5708 rad |
| 主要 Web 執行環境 | 舊 Direct 環境 | `ManagerBasedRLEnv` |

### Inertia 在 URDF 的位置

慣性資料放在**每個 `<link>` 的 `<inertial>` 區塊**，不是放在 `<joint>`。例如新版 [髖部 center_hip_link](urdf/q1_inertia/quanta_robot_wheeled.urdf)：

```xml
<inertial>
  <origin rpy="0 0 0"
          xyz="-0.00014023645 -0.00012768238 -0.1029538"/>
  <mass value="5.0857092"/>
  <inertia ixx="0.027647085" ixy="4.5466489e-06"
           ixz="-1.3156818e-05" iyy="0.016804141"
           iyz="-4.5982075e-05" izz="0.019424373"/>
</inertial>
```

- `origin xyz`：質心相對於該 link 座標的位置，單位 m；`rpy` 指定慣量座標方向，單位 rad。這不是整台機器人的世界座標質心。
- `mass`：該 link 的質量，單位 kg。
- `inertia`：質心處、以 inertial 座標表示的對稱慣量矩陣；六個獨立分量為 `ixx / iyy / izz / ixy / ixz / iyz`，單位 kg·m²。

### 程式整合與訓練影響

1. 新增來源 URDF、28 個使用中的 mesh 與來源雜湊紀錄，保留舊資產供歷史版本查閱。
2. 在 [q1_inertia.py](source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_inertia.py) 明確設定 `import_inertia_tensor=True`；此設定也參與 USD converter cache hash，避免沿用不同慣量匯入設定的舊快取。
3. Web、播放和 Stand／Walk／Standup／Backflip 訓練統一使用新資產、23-action／90-observation 契約及 managers；調整原生關節名稱、方向、限位與 4 個 mimic 的對應。
4. 保留 CubeMars explicit actuator 的電壓／電流／力矩限制，mimic 使用 implicit servo。**Direct／Manager 是環境架構，explicit／implicit 是 actuator 分類，兩者不同。**
5. 共用 Spark／RTX 啟動入口，保留有效 display 偵測，補上 Linux ARM 子程序所需的 OpenMP 預載；不修改 x86 的預載設定。
6. 新增依序執行的 PPO 佇列、checkpoint 保存／續跑、獨立 seed 驗證及 Web 模型更新；預覽與已通過驗證的模型分開發布。

質量、質心與旋轉慣量改變會影響平衡、輪子加速、推蹬、空中旋轉及落地反應，因此需要重新訓練。此次也改變了 action 維度、關節方向與限位，舊 24 維 PPO 無法直接載入新版。Web 顯示新版模型與模擬狀態，物理慣量由 Isaac Sim 使用。新版來源資料的保留不等於已完成實機辨識；碰撞與 mimic 的模擬近似見下方 Runtime 與物理。

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

連續失敗超過重試預算會以 exit 78 結束，systemd 不會重啟該失敗實驗；重新執行也會保留重試上限。檢查 log、修正原因後，使用新的 `--state` 路徑開始新實驗。Web 服務啟動亦有次數限制。

完成更新預算但沒有通過驗證會標為 `completed_with_unvalidated_skills`，不會冒充技能完成。後翻驗證要求實際離地、完整後轉與穩定落地，沒有 root pose／速度輔助。

### 訓練狀況快照：2026-10-16 03:51 UTC

這是查核當下的紀錄，**不是即時進度**；最新狀況請執行上方 `--status` 並查看訓練 log。正式實驗 `20261015T232705` 在 `~/will/q1_wheel_urdf` 運行，使用 1024 env；訓練與 Web 兩個 user service 均為 active/running，服務重啟次數皆為 0。

| 任務 | 查核狀況 | 更新預算 |
| --- | --- | ---: |
| Stand | log 已到 iteration 4594（零起算，約完成 4595 次更新）；已保存 `model_4500.pt` | 5,000 |
| Walk | 等待 Stand 完成 | 20,000 |
| Standup | 等待 Walk 完成 | 20,000 |
| Backflip | 等待 Standup 完成 | 40,000 |

Stand 的 1000／2000／3000／4000-update 段落均完成正式驗證。最近 4000-update 模型（`model_3999.pt`）在 seed 9127、16 env、500 steps／10 秒的 first-trial 檢查：**16/16 成功、0 跌倒、0 非法接觸、0 nonfinite，upright fraction 1.0**；速度誤差約 0.00712 m/s、yaw 誤差約 0.0405 rad/s。這是本次模擬條件下的站立驗證，未涵蓋任意外力或實機情境。

Web 已載入該已驗證 Stand 模型，SHA-256 與驗證報告一致：`4881b97c74b7d0d759a5b081376fd1dd84a4b952c391cb4e6fcb3e803f9df595`。後續三項正式大量訓練尚未開始，不能據此宣稱 Walk／Standup／Backflip 已收斂。完整驗證檔位於本機 `outputs/native_manager/queue/stand_00{1000,2000,3000,4000}_validation.json`。

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

Linux ARM 入口會依 Isaac Sim 的需求，為子程序預載系統與 PyTorch 的 `libgomp`；保留既有 `LD_PRELOAD` 且不重複加入，讓 user systemd 與互動式啟動一致。x86 RTX 不加 ARM 預載設定。

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
