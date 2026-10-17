# 新 URDF 左右交替滑行（2026-10-17）

新增獨立的 Sliding，不把既有的雙輪行進 Walk 當作交替滑步。舊版 Slide 是 24-action 模型，仍保留；新版以原生 URDF 23-action 任務重新訓練。

## 動作與獎勵

目標：一腳在地面支撐、相對身體向後推，另一腳短暫抬起向前回收，落地後交換角色。週期 1.4 秒，單腳 66% 支撐；前後參考行程約 18 cm、回收抬高目標 3.5 cm，依 curriculum 漸增並以 1.8 秒平順進入。參考逆運動學已逐一對照 URDF 左／右腿完整關節鏈。所有動作只產生馬達 targets／effort，經既有 CubeMars plant；不注入 root 外力，不寫入機器人姿勢或速度。

獎勵依實際物理量而非 target 判定：

- 貼地向後推：腳輪法向接觸 >60 N、腳相對軀幹向後、地面切向反力朝前、身體確實前進。向前反力以水平 heading 計算，排除傾身把垂直支撐力誤當推力。
- 抬腳向前回收：回收腳接觸 <5 N、相對軀幹向前、另一腳確實承重；抬得太高、長時間浮空、雙腳離地均受罰。
- 完整循環：累積支撐向後行程 >6 cm、正向推進 impulse >0.5 N·s、支撐承重比例 >60%，接著真實離地回收向前 >6 cm，最後落地。回收時雙腳同時失去支撐達 40 ms 不計次。只有抬腳、原地換腳或單靠輪子前進，不能取得完整循環分。

這是獎勵／參考動作設計，不能當作策略已學會該動作的證明。

## 排程

`config/native_slide_training_6144.json`：6144 env、20,000 updates，每 1000 updates 以 seed 9127、64 env、1000 steps（20 秒）做 first-trial 評估，包含第 10k／20k 里程碑。合格要求至少 80% 試驗的左右腳各完成 3 次有效推進／回收、跌倒率不超過 10%、upright fraction >=95%、平均速度誤差 <0.2 m/s、hop penalty <0.1、轉向誤差 <0.2 rad/s、結束時已停穩、無非有限值。正式評估包含 2 秒站立、前進、左右轉、最後 4 秒停止；不是只檢查輪子能動。

以獨立的 `q1-native-slide-training.service` 等待目前 Stand→Walk→Standup→Backflip 完成後開始；不修改已固定的四階段 plan／controller hashes，不丟棄目前訓練進度。上游失敗會明確記為 blocked_upstream_failed；上游暫停則繼續等待。新任務有獨立狀態、來源雜湊與有限重試；OOM 時沿用減少環境數的保護。

Sliding 的行進距離界限為 12 m，避免正常 20 秒滑行被原站立用的 4 m 界限誤判終止；這只作用於新 Sliding 任務，不修改正在跑的四個任務。

觀測形狀仍為 90，但加入滑行週期特徵，模型契約為 `q1-native-slide-v1-23x90`，不混載普通 Stand／Walk 模型或舊 24 軸權重。本任務從新 actor 開始，避免沿用 Stand 正規化中未出現的週期通道。

```sh
cd ~/projects/q1_wheel_urdf
bash start_sliding_training.sh
systemctl --user status q1-native-slide-training
cat outputs/native_slide_queue/waiting.json
# 開始大量訓練後：
python3 tools/native_training_queue.py --status --state outputs/native_slide_queue/status.json
```

## 網頁整合

新增「交替滑行 Sliding」，與「雙輪行進 Walk」分開。介面顯示排隊／訓練狀態；沒有已驗證新契約模型時按鈕停用並顯示「等待模型」或「訓練中，待驗證」，API／鍵盤也不能繞過。合格模型由佇列發布至 checkpoints/native_manager/slide/model.pt 後，網頁會自動載入並啟用按鈕。執行時可查看左右有效推進次數，保留統一的按下動畫。

啟動後預設前進 0.4 m/s；WASD 的 Sliding 速度限制在 0–0.6 m/s、yaw ±0.15 rad/s，S／Space 停止，避免送入未訓練的倒退指令。切換動作先處理舊鍵盤 stop 指令，防止按 Sliding 後被舊零速訊息立即抵消。

## 驗證範圍

- CPU reward／URDF FK：6 tests 通過，包含原地抬腳、雙腳跳起與錯向推力不計為有效循環。
- 模型閘門：4 tests 通過，拒絕錯誤契約與未驗證預覽、正確模型可切換指令且不寫入 pose、按鍵速度受限。
- 佇列：9 tests 通過，額外 Sliding 來源不改原四階段來源字典。
- 真實瀏覽器／真實 native player：新增按鈕、等待文字、停用、鍵盤拒絕、站立仍穩定；0 page errors，桌機與窄畫面截圖附於此目錄。
- 最終版小批 PPO：64 env × 3 updates，能收集 rollout、更新與儲存模型，獨立驗證沒有 NaN/Inf；有效滑步 0/8，8/8 失敗，因此沒有發布為可用滑行模型。這是小批流程驗證，不是動作成功。

本機原始資料：outputs/native_slide_smoke_v3/、logs/rsl_rl/q1_native_slide/；摘要見 summary.json，首次正式排隊紀錄見 queue_started.json。
