# 10/19 09:00 台灣時間重開機前的限時訓練

這是限時實驗，不能保證所有動作收斂成功。維持 6,144 envs、24 steps/update、現有 PPO learning rate 3e-4 / entropy 0.002，不更動新版 URDF 或正在使用的馬達模型。原 10k/20k 分段改為每 1,000 iterations 以 64 envs 驗證，保留最高分的已通過模型與獨立最新 preview。

| 順序 | 動作 | 新預算 | 單項時間上限 |
|---|---|---:|---:|
| 1 | Stand | 至少已完成 4,000；保留進度與最佳模型，不追加 | — |
| 2 | Walk | 4,000 | 5 小時 |
| 3 | Standup / Unbox | 5,000 | 6 小時 |
| 4 | 交替 Sliding | 5,000 | 6 小時 |
| 5 | Backflip | 14,000 | 14 小時 |

後續合計 28k updates，按目前約 3.2 秒/update，純訓練約 25 小時；啟動及驗證另計，實際速度可能不同。原實驗 state 與 checkpoint 保留，新計畫使用 `outputs/native_deadline_20261019/status.json`，延續相同 run_id，因此能找回已完成的 Stand checkpoint；計畫及 source hash 重新記錄，不覆寫舊實驗的固定計畫。

## 截止與交付

- 台灣時間 10/19 04:00 停止追加訓練，若到單項上限則提早保存並換下一項。每項最多一次重試，耗盡後明確記錄失敗並繼續。
- 截止時只使用完整 checkpoint；中斷中的未儲存更新不冒充已保存。可驗證的最新 checkpoint 仍會評估，未通過者只保留 preview。
- 結束後自動跑 queue / deadline / regression / reward / Web gate / RSL 相容性 / policy 與 controls 檢查、五項 64-env 定量驗證、實際 Web 各按鈕請求、真實 Chrome 桌面與窄螢幕測試，以及 Kneel→Unbox 控制器影片。程式檢查通過不等於動作成功，兩者分開記錄。
- 10/19 08:00 前結束新增測試，預留上傳；08:30 為交付截止，預留半小時給 09:00 重開機。
- 在獨立 Git worktree 整理各項 best / latest 模型、metadata、SHA-256、驗證與失敗報告並推送 GitHub `main`；不提交使用者的其他變更。交付結果記於 `outputs/native_deadline_20261019/final_review/delivery.json`。
- 網路／GitHub 或模擬器故障可能使交付失敗，報告會保留本機，不宣稱上傳成功。

## 尚無法在新版 URDF 訓練的項目

Kneel 沒有獨立原生 PPO 任務，會執行控制器檢查與 Kneel→Unbox 串接；四種正／反躺 Elbow / Gripper Recovery 仍是舊 118-observation / 24-action 契約，無新版 90 / 23 任務，列為未遷移並確認 Web 拒絕，不把舊模型當新模型訓練完成。原本所有舊模型保留。

## 操作

`./start_deadline_training.sh` 啟動已初始化的新 state；原主排程與 Sliding waiter 必須先停止，避免同 GPU 重複訓練。`systemctl --user status q1-native-deadline.service` 檢查服務。重啟網頁 player 將優先顯示新 state。最終模型備份在 `checkpoints/native_deadline_20261019/`；Web 仍讀取 `checkpoints/native_manager/` 的正式發布模型。
