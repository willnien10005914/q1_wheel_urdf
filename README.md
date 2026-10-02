# Q1 Wheel（BI2 Wheel）URDF + Isaac Lab PPO

廣達 Q1 / BI2 輪式人形：約 1.4 m、**40 kg**、24 顆 CubeMars 準直驅關節、雙 Ø200 mm 驅動輪、骨盆 Xsens MTi-630 IMU。在 Isaac Lab（Isaac Sim 5.1 / PhysX、RSL-RL PPO）訓練站立滑行、側滑、蹲跪／站起，以及正躺／趴躺起身。

英文版說明見 [`README.en.md`](README.en.md)。

---

## 快速開始（別台 Ubuntu）

### 1. 系統前提

- Ubuntu 22.04 / 24.04，NVIDIA 驅動可用
- 已安裝 [Isaac Lab 2.3](https://isaac-sim.github.io/IsaacLab/)（預設路徑 `~/isaac/IsaacLab`，venv `~/isaac/env_isaaclab`）
- Git、curl、Python ≥ 3.10

### 2. 一鍵安裝（uv）

```bash
git clone git@github.com:willnien10005914/q1_wheel_urdf.git
cd q1_wheel_urdf
./install.sh
```

`install.sh` 會：

1. 安裝 [uv](https://github.com/astral-sh/uv)（若尚未安裝）
2. 用 uv 建立 `.venv`，安裝本倉庫 Python 套件（`wheel_humanoid_lab`、網頁／工具依賴）
3. 在 Isaac Lab venv 內可編輯安裝 `source/wheel_humanoid_lab`（若偵測到 `~/isaac/env_isaaclab`）

僅裝 Python 工具、不碰 Isaac：

```bash
./install.sh --tools-only
```

### 3. 播放（Isaac Sim + 網頁）

```bash
./play_skateboard.sh
# 瀏覽器：http://127.0.0.1:8766/web/
# 結束：./stop_isaac.sh
```

僅開網頁（無 Isaac、無 PPO）：

```bash
uv run python web/serve.py
# http://127.0.0.1:8765/web/
```

---

## 已訓練 PPO（最後有效 checkpoint）

倉庫只保留各動作**最後有效**的權重（其餘實驗 ckpt 不追蹤）。

| 動作 | Checkpoint | Gym task | 說明 |
|---|---|---|---|
| **Walk / 滑行** | `checkpoints/q1_skate_ppo.pt` | `Isaac-Q1-Skate-v0` | 雙輪站立、前傾、手臂後擺、WASD 追蹤 twist |
| **Slide / 側滑** | `checkpoints/q1_slide_ppo.pt` | `Isaac-Q1-Slide-v0` | 前後腳交替微抬腳滑行（X2 風格） |
| **Stand / 站起** | `checkpoints/q1_posture_ppo.pt` | `Isaac-Q1-Posture-v0` | 四輪跪姿 ↔ 站立（再切到 skate） |
| **Sit / 跪坐** | 同上 `q1_posture_ppo.pt`（Kneel） | `Isaac-Q1-Posture-v0` | 從站立蹲成四輪跪姿 |
| **地板 → 跪**（開箱鏈） | `checkpoints/q1_unbox_ppo.pt` | `Isaac-Q1-Unbox-v0` | 正躺開箱 → 仰臥起坐 → 穩定跪姿 |
| **正躺起身** | `checkpoints/q1_recovery_v3_supine_ppo.pt` | `Isaac-Q1-RecoveryV3-Supine-v0` | 僅正躺：地板 → 跪 → 站 |
| **趴躺起身** | `checkpoints/q1_recovery_v3_prone_ppo.pt` | `Isaac-Q1-RecoveryV3-Prone-v0` | 僅趴躺：地板 → 跪 → 站 |

ONNX / TorchScript（skate，含 observation normalizer）：

- `checkpoints/exported/q1_skate_policy.onnx`（輸入 `obs[1,92]` → `actions[1,24]`）
- `checkpoints/exported/q1_skate_policy.pt`

**不建議使用（已淘汰，不在倉庫追蹤）：** `skateboard_ppo.pt`（舊 ideal-PD）、`q1_getup_ppo.pt`、`q1_recovery_reference_v2_ppo.pt`、混合 `q1_recovery_contact_v3_ppo.pt`（已由分拆 supine/prone 取代）。

### 示範影片（已上傳）↔ PPO 位置

完整索引見 **[`docs/demos/`](docs/demos/README.md)**。重點如下：

| 動作 | 影片 | PPO |
|------|------|-----|
| 正躺起身 | [`split_supine/recovery_eval_supine.mp4`](docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.mp4) · [gofile](https://gofile.io/d/z9w7axZf) | `checkpoints/q1_recovery_v3_supine_ppo.pt` |
| 趴躺起身 | [`split_prone/recovery_eval_prone.mp4`](docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.mp4) · [gofile](https://gofile.io/d/hwX7CfTN) | `checkpoints/q1_recovery_v3_prone_ppo.pt` |
| 站立滑行 skate | [`q1_skate_cubemars_iter3800.mp4`](docs/results/q1_skate_cubemars_iter3800.mp4) · [gofile](https://gofile.io/d/Oh6jh3Qd) | `checkpoints/q1_skate_ppo.pt` |
| 側滑 · **左右腳前後** | [`q1_slide_foreaft_iter32000.mp4`](docs/results/q1_slide_foreaft_iter32000.mp4) · [gofile](https://gofile.io/d/FEl7B5IT) | `checkpoints/q1_slide_ppo.pt` |
| 側滑 · 抬腳 / 跨步 | [`passlift`](docs/results/q1_slide_passlift_iter42000.mp4) · [`stride`](docs/results/q1_slide_stride_iter32000.mp4) | 同上 `q1_slide_ppo.pt` |

Raw（`main`）：

- https://github.com/willnien10005914/q1_wheel_urdf/raw/main/docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.mp4
- https://github.com/willnien10005914/q1_wheel_urdf/raw/main/docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.mp4
- https://github.com/willnien10005914/q1_wheel_urdf/raw/main/docs/results/q1_skate_cubemars_iter3800.mp4
- https://github.com/willnien10005914/q1_wheel_urdf/raw/main/docs/results/q1_slide_foreaft_iter32000.mp4

### 各 PPO 細節

| PPO | 觀測 / 動作 | 訓練重點 | 播放行為 |
|---|---|---|---|
| **skate** | 92-D obs、24-D action；CubeMars 致動器 + DR | 課程 stand→glide→swizzle→lean→micro-lift | 預設模式；WASD 控 `vx` / yaw |
| **slide** | 同 skate 契約 | 前後腳交換、50–220 ms 抬腳獎勵 | 自動 wander 或 WASD |
| **posture** | 同契約；較寬腿部 clip | kneel↔stand 姿態轉移 | Kneel / Stand 按鈕 |
| **unbox** | 同契約 + MediaPipe 關鍵幀 prior | 正躺開箱到穩定四輪跪 | API `unbox`（網頁無獨立按鈕） |
| **recovery supine** | recovery_v3 MDP；接觸閘門 | 6144 envs × ~10k iters，僅 mode=正躺 | 先按正躺，再 Recovery |
| **recovery prone** | 同上 | 同上，僅 mode=趴躺 | 先按趴躺，再 Recovery |

開箱鏈（可選）：正躺 → **unbox** 跪 → **posture** 站 → skate/slide。  
地板起身鏈（網頁主流程）：正躺／趴躺 → **Recovery PPO** → 站穩後切 Walk / Slide。

---

## 網頁按鈕 ↔ 功能

`./play_skateboard.sh` 會啟動 Isaac Sim，並在 **8766** 提供 `/api/pose`。按鈕對應如下：

| 按鈕 | `POST /api/pose` | 載入的策略 | 行為 |
|---|---|---|---|
| **Kneel (4 wheels)** | `kneel` | `q1_posture_ppo.pt`（target=跪） | 蹲成四輪跪姿 |
| **Stand up** | `stand` | posture → 完成後切 `q1_skate_ppo.pt` | 從跪站起，再交給滑行策略 |
| **Slide mode** | `slide` | `q1_slide_ppo.pt` | 側滑／交替抬腳模式 |
| **Walk / skate** | `skate` | `q1_skate_ppo.pt` | 站立滑行；鍵盤 WASD |
| **正躺 supine** | `supine` | （尚不跑 PPO） | 放到正躺並**凍結**姿態 |
| **趴躺 prone** | `prone` | （尚不跑 PPO） | 放到趴躺並**凍結**姿態 |
| **Recovery PPO 起身** | `recovery` | 依上次躺姿選 `…_supine_ppo.pt` 或 `…_prone_ppo.pt` | 接觸閘門起身到站立 |

建議流程：先選 **正躺** 或 **趴躺** → 再按 **Recovery PPO 起身** → 站穩後按 **Walk / skate** 或 **Slide mode**，用 WASD 操控。

其餘：

| 控制 | 說明 |
|---|---|
| Zero / T-pose / Reach | 僅本機網頁預設姿，不送 Isaac |
| Iso / Front / Side / Top | 相機視角 |
| 滑桿拖曳 | 覆蓋該馬達到 sim；**Follow PPO** 交還策略 |
| Load A3 keyframes… | MediaPipe 開箱參考軌跡編輯（見下方） |
| 鍵盤 W/S | 前進／後退 |
| A/D、Q/E | 轉向 |
| Space / X / L | 停止 |
| R | 重置 episode |
| K / U | 快捷 Kneel / Stand |
| P | 切換 Slide |
| G | 躺下（supine） |

---

## 訓練指令

```bash
# 站立滑行
./train_skate.sh

# 側滑
./train_slide.sh

# 跪 ↔ 站
./train_posture.sh

# 開箱：正躺 → 跪
./train_unbox.sh

# 分拆起身（建議一次只跑一個）
ALLOW_REDESIGN=1 ./train_recovery_v3_supine.sh
ALLOW_REDESIGN=1 ./train_recovery_v3_prone.sh
# 或連續：./train_recovery_v3_sequential.sh
```

環境變數常用：`NUM_ENVS`、`MAX_ITERS`、`RESUME=1`、`CHECKPOINT=...`。

訓練日誌在 `logs/`（已 gitignore）；結束後有效權重請複製到上表 `checkpoints/` 路徑。

---

## 目錄結構（精簡）

```
config/q1_wheel_components.yaml     硬體表（馬達 SKU、關節對應、質量）
urdf/                               結構 URDF + 網頁輕量 URDF
meshes/                             視覺／碰撞 STL
source/wheel_humanoid_lab/          Isaac Lab 擴充（skate / slide / posture / unbox / recovery_v3）
scripts/reinforcement_learning/     train / play / web bridge
web/                                THREE.js 關節 UI
checkpoints/                        最後有效 PPO + exported ONNX
tools/                              URDF 建置、MediaPipe、recovery 工具
docs/                               馬達規格、參考姿態、評測紀錄
docs/demos/                         示範影片 ↔ PPO 對照（正躺／趴躺／滑行／側滑）
install.sh / pyproject.toml         uv 安裝入口
```

---

## MediaPipe 開箱參考（選用）

```bash
# 需 mediapipe（install.sh 的 tools extra）
uv run python tools/unbox_pose_extract.py \
  --video docs/reference/a3_unbox_ref.mp4 \
  --model tools/models/pose_landmarker_heavy.task \
  --out docs/reference --overlay

uv run python web/serve.py   # 編輯馬達角 → Save for RL
uv run python tools/apply_unbox_recording.py \
  --recording docs/reference/a3_unbox_ref_web_recording.json
./train_unbox.sh
```

詳見 [`docs/reference/README_unbox.md`](docs/reference/README_unbox.md)。

---

## 觀測／動作契約（skate 系）

- **觀測 92-D**：IMU gyro 3 + projected gravity 3 + joint pos 24 + joint vel 24 + last action 24 + twist 3 + head 4 + body 6 + arm_style 1
- **動作 24-D**：22 位置目標 + 2 輪速（×25 rad/s）
- 控制 50 Hz，PhysX / CAN 200 Hz

合約單元測試：

```bash
./run_isaac.sh tests/test_skate_contract.py
```

---

## 授權與注意

- 本倉庫含 URDF、訓練腳本與最後有效 PPO；Isaac Sim / Isaac Lab 需另行安裝並遵守其授權。
- GPU 訓練建議單卡一次一 job；`stop_isaac.sh` 可清殘留 Kit／佔用埠。
