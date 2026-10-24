# 指定影片 15–23 秒 → 輪型 Q1 訓練與自動交付

來源：[日刊工業影片 iPItgmg4RBw](https://youtu.be/iPItgmg4RBw)，僅使用指定的 15–23 秒。已下載正確來源、裁切 720p 本機片段，以 MediaPipe Pose Landmarker heavy 0.10.35、CPU/XNNPACK、軟體 GL/Mesa llvmpipe 抽取 121 個取樣影格，120 個實際偵測成功（99.17%），關鍵腿／軀幹點 visibility 中位數約 0.633。`mediapipe_subject_review.jpg` 已人工檢視為唯一主體 CUE7，不是背景人物。原影片與完整下載 cache 沒有提交 GitHub；衍生 landmarks、軌跡、SHA 與投影方法在 `data/reference/native_unbox_20261024/`。

## Retargeting 與限制

保留完整八秒的非單調過程：初始低蹲、後仰再下沉、最後起立。蹲下以同一時間軌跡倒放，沒有用直線手寫起身冒充影片。姿態含 23 個獨立關節；膝蓋 mimic 仍按原 URDF 跟隨。雙輪高度使用完整 URDF FK 與 root pitch 計算，保留 2 mm 起始間隙。來源是單一相機的機器人影片，MediaPipe 並不是精確的馬達量測；下列形態／可達性投影都有記錄，不能把輸出當原機器人的真實 encoder 角度。

- 左右腿依偵測 confidence 做矢狀面對稱投影，避免遮擋腿造成一輪懸空；原始左右 landmarks 與角度保留。
- 新版腰 pitch 的 lower limit 是 0，不能向後彎。影片的後仰改以 root pitch（最多 -0.6 rad）與髖關節耦合表示，起始方向只在 reset 寫入，之後靠原生馬達力矩與 PPO 學習，沒有外力支撐、飛行輔助或動作中 teleport。
- 最後 0.8 秒將站立膝角平滑投影到最多 0.40 rad，保留微彎。起始骨盆高度約 0.667 m，終點約 0.876 m，依 Q1 的腿長與輪徑計算。
- 手腕、gripper 及單鏡頭無法可信辨識的自由度使用中立值。肩膀 pitch 的軸向與髖不同，已按真實 URDF 的左右 +Y/-Y 重新對應。
- `approved_for_training` 代表來源、confidence、關節限制及幾何檢查通過；**不代表動態穩定或動作已學成**。

## 排程

| 順序 | 任務 | envs | 正式更新次數 |
|---|---|---:|---:|
| 1 | 影片起立 Standup | 6,144 | 10,000 |
| 2 | 反向蹲下 Kneel | 6,144 | 10,000 |
| 3 | 微彎站姿雙輪行進 Walk | 6,144 | 10,000 |
| 4 | 後空翻 Backflip | 6,144 | 10,000 |
| 5 | 左右交替貼地推進／微抬回收 Skate | 6,144 | 10,000 |

起立與蹲下使用影片 reference；後空翻使用同一微彎站姿作起點、獨立馬達推蹬／收腿／落地課程，不能把沒有後翻的原影片說成 flip mocap。Skate 使用真正接觸負載、向後推蹬、懸空向前回收及完整循環 reward，沒有只抬腳就算成功。正反躺 Elbow/Gripper Recovery 不重訓。

每 1,000 次更新以 64-env first trial 檢查，不放寬成功門檻，只發布通過且分數改善的 best；latest preview 另存。新模型契約為 `q1-native-video-v1-23x90`，需要 ref/URDF/weights SHA 與技能名稱一致，不能直接把舊 90/23 形狀相同的模型當影片策略。

## GPU 交接與小批量

本機只有單張 16 GB RTX 4090 Laptop GPU。偵查時 Q1 leg 10240-env 程序占約 7 GB、GPU utilization 約 74–81%；沒有獨立 GPU 或運算／記憶體隔離，無法承諾同跑 6144-env 不拖慢上游，所以不做競爭 GPU 的容量試驗。MediaPipe 與程式準備以最多兩條 CPU 執行緒並行。

監控整個 `q1-leg-upper-training.service` 與它公布的最新同專案／同 robot run，不只看某個 chunk PID。最初偵查的 run 是 `20261022T204705_454380Z`；上游若中斷後另起新 run，會驗證它仍位於原訓練目錄並追蹤該 run，而不是永遠卡在已被替換的舊 state。必須全域及該 run 的 state 都是 passed / complete / budget_exhausted、服務已停、沒有 leg trainer/supervisor、GPU compute consumers 清空連續 60 秒，才取得上游原有 training flock 及共享 GPU flock，重新核對後交接。interrupted / runtime_error 是可恢復的中斷，**不算完成**；狀態不明或 GPU 查詢失敗也繼續等待，不殺掉別人的程序。

交接後先跑每任務 64 envs / 3 updates、數值／載入／驗證流程測試與 Web 蹲姿開機 smoke；通過管線檢查才啟動正式 6144-env。小批量不宣稱已收斂。若 6144 容量失敗，不悄悄縮小 envs 冒稱完成。

## 開機姿態與網頁

`./play_skateboard.sh` / `./play_skills.sh` 預設使用新 Reference-Web，從影片起始蹲姿開始，等待使用者按起立。只有符合新契約並 validated 的模型會開放正式按鈕。`--allow-preview` 是明示實驗模式；預設不執行失敗模型。`./play_skills.sh --legacy-native` 可開原播放器與原模型。不要在上游訓練中另開 Isaac viewer；viewer 也會用 GPU。

## 完成後

自動重新驗證並錄製一鏡到底「初始蹲姿→起立→膝微彎前滑→停穩→後空翻」，另錄交替 skating。過程不 reset 隱藏跌倒；若只能載入 preview、模型缺少或曾失敗，影片與報告會明確標示，不能冒充成功。影片上傳 GoFile；只保存公開下載頁、檔名、大小、SHA，不提交管理 token。模型、驗證、影格與公開影片連結在隔離 Git worktree 提交並推 GitHub `main`，不混入使用者其他修改。

監控啟動：`./start_reference_training.sh`；狀態：`python3 tools/reference_campaign.py --status`。本次部署使用持續的使用者層級 systemd 服務（`./start_reference_training.sh --install-service`），重開機後仍會等待上游完成；使用者已確認保留 Q1 leg 優先順序，interrupted 時也不搶先啟動；CPU watcher 與 state 可續用，訓練來源／reference hash 改動則須建立新的實驗 state，不能覆蓋舊試驗。
