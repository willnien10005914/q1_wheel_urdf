# 新 URDF 逐動作實測與 GoFile 影片（2026-10-17）

[96 秒全部動作總覽](https://gofile.io/d/vRNj4vCR)。所有影片來自新版 `urdf/q1_inertia/quanta_robot_wheeled.urdf`、23 action／90 observation 的 ManagerBasedRLEnv，沒有代用舊 URDF 成功影片。失敗後持續錄製，沒有自動 reset 隱藏跌倒。影片及數據以單一種子／單次嘗試為主，不是統計成功率。

## 有沒有重新訓練

有：6144 env 的 Stand 正式更新已完成 3000／5000 次。三次不同 checkpoint 的 64-env first-trial 驗證：第 1000 次 64/64、0 failure；第 2000 次 45/64、19 failure；第 3000 次 60/64、5 failure。只保留分數最高的第 1000 次作為目前已驗證 Stand 模型，SHA-256 `1f171c4f2babce0e8930ade2489e5085339a35223fd95fa61a29de26a50e86a3`。備份於 `checkpoints/native_stand_review_20261017/`，不是最終訓練完成版本。

Walk、Standup、Backflip 還沒有輪到正式大量訓練；影片中的 task PPO 是先前 64-env、3 updates 的 smoke 模型，畫面已標明。控制器影片使用零殘差與原生馬達控制，不能算成該技能已重訓完成。

## 每個可執行動作的影片

| 總覽起點 | 測試 | 結果 | 個別影片 |
|---|---|---|---|
| 00:00 | Stand | 第 1000 次 PPO，12 秒站穩，無跌倒 | [GoFile](https://gofile.io/d/cm1HvetT) |
| 00:12 | 行進控制器 | 14 秒短測不倒；無正式 Walk PPO，速度仍有誤差 | [GoFile](https://gofile.io/d/jW2s59L8) |
| 00:26 | 行進小批 PPO | 只有 3 次更新；本片有暖身，短測不倒不等於正式通過 | [GoFile](https://gofile.io/d/pecFvhry) |
| 00:38 | Kneel | 失敗，約 6.20 秒非法接觸／失衡，最終倒地 | [GoFile](https://gofile.io/d/Jv6LwMKG) |
| 00:48 | 起立控制器 | 失敗，約 2.26 秒失衡，無正式起立 PPO | [GoFile](https://gofile.io/d/1LX2ojsi) |
| 01:00 | 起立小批 PPO | 失敗；只有 3 次更新 | [GoFile](https://gofile.io/d/Uv90L7eJ) |
| 01:12 | 後翻小批 PPO | 失敗；只有 3 次更新，約 1.98 秒非法接觸 | [GoFile](https://gofile.io/d/o9h6ydn8) |
| 01:19 | 下跪→起立控制器 | 失敗；下跪已失衡，後續起立未成功 | [GoFile](https://gofile.io/d/Dnwb31Ej) |

行進影片先等待 2 秒，再前進、左右轉、倒退。抽樣平均絕對速度誤差約 0.164 m/s（控制器）／0.159 m/s（小批 PPO）；沒有量化轉向追蹤成功率。先前「從第一步即持續下達 0.4 m/s」的 8-env smoke 評估仍然失敗，因此沒有把這段暖身後的單片結果升格成正式 Walk 驗證。

Kneel 不能用泛用的 standing success latch 判定。其原始 native failure 高度門檻亦按站立設定；錄影工具對 Kneel 使用低跪姿高度門檻，仍保留非法接觸／失衡檢查。即使短暫達到低姿態，後來倒地仍標失敗。

## 實際網頁控制器複驗

另外以 `play_skills.sh → play_native_web.py`、相同 SHA-256 的正式 Stand 模型快照跑 850 steps：Reset、Stand、Kneel、Unbox 皆接受請求，但下跪→起立最終倒地，root z 約 0.136 m。Backflip 因沒有正式原生模型而拒絕；Slide、四種 Recovery 也拒絕。詳見 `actual_web_sequence.json`。這與一般控制器影片的暖身條件不同，不能混同為同一條軌跡。

## 尚不能執行的舊動作

Slide 的舊模型為 92 observation／24 action；正躺 Elbow、反躺 Elbow、正躺 Gripper、反躺 Gripper 均為 118／24。五個模型逐一載入檢查並記錄 SHA-256，均不符合新版 90／23 契約，也沒有對應新版任務。因此這五項只完成相容性檢查，沒有在新 URDF 執行動作，也沒有冒充它們的成功影片；需先遷移。

## 訓練卡住與恢復

續跑 3000→4000 段落在「Starting the simulation」停滯約 6 小時，子程序 CPU 約 100%、GPU 空閒；服務的 running heartbeat 不代表訓練有進展。停止卡住程序保留所有已儲存 checkpoint。

新增 watchdog：子程序 600 秒無 log 輸出即終止並進入現有有限重試預算，耗盡仍 terminal exit 78，不無限重啟。包含強制終止卡住程序的測試，queue 7 tests、manager regression 8 tests 通過。工具變更不修改固定的 URDF／controller／motor 來源雜湊。恢復進度另記 `training_resumed.json`。

原始影片與完整 trace 在本機 `outputs/native_review_20261017/`；此目錄保存摘要、圖片及分享連結。GoFile API 對九次上傳均回報成功；此主機直接開分享頁遇到自簽憑證錯誤，未繞過 TLS 驗證，不能宣稱已由本機瀏覽器播放遠端影片。96 秒本機合成檔已完整解碼檢查。
