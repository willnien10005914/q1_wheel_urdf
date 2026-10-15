# Native URDF / Manager Web 與大量訓練

本次把主要 Web、播放及技能訓練遷移至新版 BU10 輪式 URDF。預設 `play_skateboard.sh → play_skills.sh → play_native_web.py` 建立真正的 `ManagerBasedRLEnv`；不使用舊的 Direct Web 環境，也不補虛擬頸部或載入舊 24-action PPO。

## 共同契約

- 來源：`urdf/q1_inertia/quanta_robot_wheeled.urdf`，質量約 41.0523 kg。
- 23 主動關節、4 mimic，action 23、actor observation 90、critic observation 90。
- 物理 200 Hz、策略 50 Hz。
- Stand、Walk、Standup、Backflip、Web 共用資產、motor plant、command/action/observation managers。
- 姿態與平衡先驗只產生馬達 targets，經 CubeMars 電流／電壓／力矩限制作用；沒有 root 外力、速度注入或逐步姿態寫入。
- 只有 reset event 可以寫入起始姿態。技能切換已檢查 root/joint state 完全不變。
- 新 URDF 的原始軸向、限位、質量／質心／慣量保持原樣。手臂與腰部限位不同，因此後空翻需要新訓練。

原生技能 phase 僅用於起身／下跪／後翻；持續站立與行進的 phase 固定為零，避免長時間播放超出短 episode 的正規化分布。兩輪的正負方向依新 URDF 處理。Mimic 使用有限增益 servo、碰撞使用凸包、自碰撞關閉，仍是模擬近似。

## 已完成的檢查

| 檢查 | 結果與範圍 |
| --- | --- |
| 四技能 PPO 預跑 | Stand、Walk、Standup、Backflip 各 64 env、3 updates，能收集 rollout、更新網路、保存 checkpoint |
| 佇列完整預演 | 四階段依序訓練／驗證，後三階段從原生 Stand warm-start；沒有載入舊模型 |
| 動作驗證 | 預演僅 Stand 通過短時間驗證；其餘三個技能尚未收斂，沒有發布為已驗證模型 |
| 最終站立先驗 | Native Web manager：2 env、2000 steps（40 s），upright fraction 1.0、0 termination、0 nonfinite、最低 root 高度 0.8368 m |
| 真實 Web / 瀏覽器 | 實際新 URDF/STL、27 sliders、4 mimic、23/90 API 契約、無 browser page error |
| 佇列回歸測試 | 6 tests：迭代排序、不完整／新儲存檔排除、預覽與驗證模型分開、驗證中斷後不重做 PPO、續跑不重複儲存的迭代索引、失敗重試耗盡後服務重啟也不重跑 |
| RSL-RL / CUDA | GB10 CUDA 運算、manager registration audit 與 RSL-RL CPU 相容測試通過 |

40 秒的穩定結果是零殘差的平衡先驗測試，不表示新 PPO 已完成大量訓練。瀏覽器測試驗證模型、API 與操作路徑，也不代表早期 PPO 的動作成功。原始紀錄在本機 `outputs/native_manager/`，摘要見 [native_manager_validation.json](native_manager_validation.json)。

## 正式佇列

使用 1024 個平行環境、每次 24 policy steps：

| 順序 | task | updates | 名義 transitions |
| --- | --- | ---: | ---: |
| 1 | Isaac-Q1-Native-Stand-v0 | 5,000 | 122,880,000 |
| 2 | Isaac-Q1-Native-Walk-v0 | 20,000 | 491,520,000 |
| 3 | Isaac-Q1-Native-Standup-v0 | 20,000 | 491,520,000 |
| 4 | Isaac-Q1-Native-Backflip-v0 | 40,000 | 983,040,000 |

總計 85,000 updates、2,088,960,000 transitions；這是排定的預算，不是已完成的樣本數。每 1000 updates 保存一段並在不同 seed 9127、16 env 進行 first-trial 驗證。GPU OOM 時會減少環境數，因此實際 transitions 可能較少。

Standup 從跪姿起身；這裡將使用者的「standard」按 Stand／Standup 處理。Walk 指雙輪行進與轉向。Backflip 有單次下蹲／推蹬、實際離地後收腿、旋轉與落地獎勵，沒有強制翻轉 root。舊正反躺 recovery 與 MediaPipe 工具仍屬歷史版本，未冒充新 URDF 的已訓練技能。

## 保存、續跑與驗證

`tools/native_training_queue.py` 使用單一鎖、獨立 process group、原子 JSON／模型發布。來源 controller/URDF/馬達設定 SHA-256 會固定；來源或計畫變更需使用新的 state 路徑。中斷後選取有效的完整 checkpoint，完成更新但尚未驗證的段落會先補驗證。

真實 RSL-RL 的兩個單次更新段也已走完：model_0 → model_1、completed_iterations=2；避免最後只剩一個 update 時重複同一索引。

- `latest.pt`：訓練預覽，Web 標示「訓練中 PPO」。
- `model.pt`：通過驗證且分數較佳的版本；保留舊的已驗證版本直到新的更佳模型出現。
- 完成預算但未通過動作檢查：`completed_with_unvalidated_skills`。
- 訓練／驗證程式異常：記錄 failed 狀態與 log，有限重試，不會標為成功。

Standup 要求達到穩定站立；Walk 檢查姿態、跌倒與速度追蹤；Backflip 要求實際離地、完整後轉和至少一秒穩定落地，並排除非法身體接觸。所有 rollout 都透過 managers 執行。

```bash
./start_native_training.sh
./start_native_web.sh
python3 tools/native_training_queue.py --status
systemctl --user status q1-native-training q1-native-web
journalctl --user -u q1-native-training -f
# 中止仍保留已保存模型：
systemctl --user stop q1-native-training
```

Web：http://127.0.0.1:8875/web/。訓練與 Web 使用相同原生任務；Web 僅一個環境，會顯示本機佇列進度與模型驗證狀態。Git 保留程式、計畫與驗證摘要，大量中間模型留在本機 `checkpoints/native_manager/`、`logs/`。

## 常駐啟動修正

2026-10-15 23:24 UTC 檢查發現，16:14 啟動的 user systemd 服務缺少互動式 shell 的 ARM `LD_PRELOAD`。Isaac Sim 在 bootstrap 即退出，因此該次正式佇列完成 updates 為 **0**；先前的互動式短測試結果仍有效。服務重啟繞過了 supervisor 的重試上限，形成重啟循環。已停止這兩個 Q1 服務，原失敗狀態與 log 留在本機供追查。

共同入口現在在 Linux ARM 自行偵測並預載 Isaac Sim checker 所需的系統／PyTorch OpenMP library，保留既有值、不重複加入，也不修改系統或 venv。清除 `LD_PRELOAD` 後實測 `import isaacsim` 成功；模擬 x86 分支不加入預載，既有預載亦保留。

Supervisor 的耗盡重試與設定錯誤改用 terminal exit 78，systemd 設定 `RestartPreventExitStatus=78`；重啟 supervisor 也會先檢查已耗盡預算。兩個服務另外限制五分鐘內最多三次啟動。只有成功完成訓練段落才清零連續失敗計數。

修正後的新正式實驗為 `20261015T232705`，在 `~/will/q1_wheel_urdf` 執行。23:34 UTC 已確認 1024 env 的 Stand PPO 持續更新，checkpoint 從 `model_0.pt` 推進到 `model_100.pt`（含 101 次更新），所有模型與 optimizer tensor 均有限值；Web 自動載入與發布檔相同 SHA-256 的 `model_100`。兩個 user service 均 active/running、重啟次數 0，真實瀏覽器測試無 page error。此時仍是未完成的早期訓練，正式動作驗證依原計畫在每 1000 updates 後執行，尚未宣稱任何大量訓練技能已收斂。摘要與來源雜湊收錄於 validation JSON 的 `systemd_recovery`。
