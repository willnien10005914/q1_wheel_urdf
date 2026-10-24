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

起立與蹲下使用影片 reference；蹲下保留完整姿態順序，另外重新配置每段時間，將原 8 秒延長至約 **20.322 秒**，限制 reference 高度速度 ≤0.16 m/s、關節速度 ≤0.8 rad/s。後空翻使用同一微彎站姿作起點、獨立馬達推蹬／收腿／落地課程，不能把沒有後翻的原影片說成 flip mocap。Skate 使用真正接觸負載、向後推蹬、懸空向前回收及完整循環 reward，沒有只抬腳就算成功。正反躺 Elbow/Gripper Recovery 不重訓。

慢蹲使用世界垂直速度與雙側主輪實際接觸力驗證，初始接觸穩定寬限為 0.2 秒。持續離地超過 0.06 秒、失去雙輪接觸超過 0.12 秒、下降速度超過 0.38 m/s 持續 0.1 秒，或超出目標高度 3 cm 且向上速度超過 0.24 m/s 持續 0.08 秒，會鎖定違規並終止該次 trial。終點須雙輪接觸比例 ≥95% 且穩定保持 1.5 秒；64 個首次測試中只要有慢蹲品質違規，就不發布為通過。kneel episode 為 36 秒，每次驗證 1,300 steps，覆蓋完整下降與最後停留。這些是訓練和驗收條件，實際動作仍須 GPU 測試確認。

每 1,000 次更新以 64-env first trial 檢查，不放寬成功門檻，只發布通過且分數改善的 best；latest preview 另存。新模型契約為 `q1-native-video-v1-23x90`，需要 ref/URDF/weights SHA 與技能名稱一致，不能直接把舊 90/23 形狀相同的模型當影片策略。

## GPU 交接與小批量

本機只有單張 16 GB RTX 4090 Laptop GPU。偵查時 Q1 leg 10240-env 程序占約 7 GB、GPU utilization 約 74–81%；沒有獨立 GPU 或運算／記憶體隔離，無法承諾同跑 6144-env 不拖慢上游，所以不做競爭 GPU 的容量試驗。MediaPipe 與程式準備以最多兩條 CPU 執行緒並行。

監控整個 `q1-leg-upper-training.service` 與它公布的最新同專案／同 robot run，不只看某個 chunk PID。最初偵查的 run 是 `20261022T204705_454380Z`；上游若中斷後另起新 run，會驗證它仍位於原訓練目錄並追蹤該 run，而不是永遠卡在已被替換的舊 state。必須全域及該 run 的 state 都是 passed / complete / budget_exhausted、服務已停、沒有 leg trainer/supervisor、GPU compute consumers 清空連續 60 秒，才取得上游原有 training flock 及共享 GPU flock，重新核對後交接。interrupted / runtime_error 是可恢復的中斷，**不算完成**；狀態不明或 GPU 查詢失敗也繼續等待，不殺掉別人的程序。

交接後先跑每任務 64 envs / 3 updates、數值／載入／驗證流程測試與 Web 蹲姿開機 smoke；通過管線檢查才啟動正式 6144-env。小批量不宣稱已收斂。若 6144 容量失敗，不悄悄縮小 envs 冒稱完成。

## 開機姿態與網頁

`./play_skateboard.sh` / `./play_skills.sh` 預設使用新 Reference-Web，從影片起始蹲姿開始，等待使用者按起立。只有符合新契約並 validated 的模型會開放正式按鈕。`--allow-preview` 是明示實驗模式；預設不執行失敗模型。`./play_skills.sh --legacy-native` 可開原播放器與原模型。不要在上游訓練中另開 Isaac viewer；viewer 也會用 GPU。

Web 起立、慢蹲或後翻未完成時不接受中途切動作；蹲姿須先完成起立，慢蹲／後翻須先停穩。起立完成後自動進入站姿保持，可接 WASD，站立時再按起立不會重跑蹲姿。按鈕共用按下動畫、顯示停用原因，拒絕訊息保留六秒。舊正反躺模型仍完整保存，但尚未通過這次 URDF／影片策略契約驗證，新頁暫停使用；`--legacy-native` 是原生平衡播放器，不是已驗證的正反躺入口。

## 完成後

自動以新 seed 10424、64 個首次 trial 重新驗證；最後驗證未過會撤銷 Web 正式模型資格並保留權重。各模型通過後，再以實際 Reference-Web runtime 跑連續起立、前滑、停穩、慢蹲、再起立、後翻及 skating 的按鈕／控制檢查；整合失敗同樣停用正式入口。

錄製五個獨立動作影片 `standup`、`slow_kneel`、`forward_glide`、`backflip`、`skating`，另保留一鏡到底「初始蹲姿→起立→膝微彎前滑→停穩→後空翻」與交替 skating，合計七片。每片都從開機蹲姿實際起立，每片開頭僅一次 reset，切換動作不改寫 root／關節物理狀態。慢蹲片先完整起立，14 秒開始慢蹲，涵蓋 20.322 秒下降及 4 秒停留。記錄雙輪接觸力、四支撐合力、世界垂直速度與慢蹲違規。若只能載入 preview、模型缺少或曾失敗，影片與報告會明確標示，不能冒充成功。影片上傳 GoFile；只保存公開下載頁、檔名、大小、SHA，不提交管理 token。模型、驗證、影格與公開影片連結在隔離 Git worktree 提交並推 GitHub `main`，不混入使用者其他修改。

監控啟動：`./start_reference_training.sh`；狀態：`python3 tools/reference_campaign.py --status`。本次部署使用持續的使用者層級 systemd 服務（`./start_reference_training.sh --install-service`），重開機後仍會等待上游完成；使用者已確認保留 Q1 leg 優先順序，interrupted 時也不搶先啟動；CPU watcher 與 state 可續用，訓練來源／reference hash 改動則須建立新的實驗 state，不能覆蓋舊試驗。
