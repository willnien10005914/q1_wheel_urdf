# 膝蓋方向、零位與不穩原因核對（2026-10-19）

結論：不能只歸因於訓練不夠。新版左腿軸向確實改了，現有 reference 已做符號轉換；沒有發現「左右膝蓋忘了轉正負號」的證據。但找到錯用起立 actor、初始化觸地高度不符及網頁速度超過驗證條件三項具體問題。

## 1. 新舊關節座標不同，Zero 不是校正

| 關節 | 舊模型 axis | 新 CAD axis | 相同彎曲姿勢的舊→新角度 |
|---|---|---|---|
| 左 hip pitch | +Y | -Y | -0.35 → +0.35 rad |
| 左 knee | +Y | -Y | +0.70 → -0.70 rad |
| 右 hip pitch | +Y | +Y | -0.35 → -0.35 rad |
| 右 knee | +Y | +Y | +0.70 → +0.70 rad |

新版左膝範圍 [-2.5, 0]、右膝 [0, 2.5]。Native `pose_values` 左膝 -2.36、右膝 +2.36 是鏡射後的深屈膝，不能用舊版「兩膝都正角度」理解。URDF 零位是模型關節座標的參考，不等於平衡站姿；網頁 Zero/Zero all 會下達零角度，不會重新定義 URDF axis、origin 或馬達 encoder offset。此診斷針對模擬，沒有實體 encoder 校正資料。

`geometry.json` 記錄實際 URDF SHA、全鏈 FK、左右輪中心、膝軸位置及全身 CoM。`pose_geometry.png` 是前向運動學圖，+X 為前；不是動態成功示範。此次沒有更改 URDF 軸向，也沒有把膝蓋一律設零。

## 2. 下跪誤用起立 PPO

舊 Web 在 kneel mode 使用 standup actor，而 standup 的訓練命令是起身。8-env 對照測試中，先站立暖身 2 秒、剛切換下跪（時間軸 t=2.0）時，該 actor 的左／右膝 action 約 +2.02／-2.38，讓屈膝目標往伸直方向偏離；這是錯誤技能／輸入分布，不能以增加同一套訓練解決。

從同一站立暖身切換：錯用 actor 的 8/8 在約 0.42–0.48 秒後首次失敗；零殘差的下跪控制器仍有 8/8 失敗，但約在 2.4–2.68 秒後。這只能證明錯用 actor 惡化過程，不能宣稱移除它已學會下跪。

修正：kneel 不再借用 standup actor；在沒有獨立 kneel policy 時，使用零 PPO 殘差的原生 reference。Web 不再把 kneel 標成起立模型，也把未通過的 preview 顯示為「未通過驗證 PPO」。

## 3. 起立 reset 把輪子埋地約 22 mm

既有 kneel reference 的實際 CAD FK 需要骨盆高約 **0.471953 m**，輪底才在地面；原訓練 reset 卻硬寫 0.450 m。因此直立 root、相同關節角的輪底約在地面下 21.95 mm。接觸解算可能引入跳動，這不是膝蓋零位校正。

修正：reset 高度從現有 URDF 腿鏈 origin、當下 pose angle 及輪半徑計算，留 2 mm 間隙。Kneel reset 改為 **0.473953 m**；Stand 維持 **0.848686 m**，沒有改站立基準角度。3 項測試使用完整 URDF FK 確認左右輪都有 2 mm 間隙。

這只修正輪子初始化穿地，不保證 calf 接觸、全身 CoM 或起立控制已平衡。原 kneel FK 的 CoM 位於輪中心前方約 0.116 m，而 Stand 約 0.021 m；這是靜態幾何線索，不能單獨判定輪式機器人的動態穩定性。修正高度後，起立控制器與舊起立 PPO 仍各有 8/8 失敗，物理複測見 `after_reset_height_fix.json`；仍須改善 reference／支撐控制及重新訓練，不能直接把舊失敗模型標記通過。

## 4. W 的 1.2 m/s 並不是已通過的 0.4 m/s 條件

既有正式 Walk 驗證只有固定 +0.4 m/s、單一 seed、10 秒 first trial，沒有涵蓋所有 Web 操作。瀏覽器與 Isaac WASD 下達 +1.2 m/s，舊後端直接接受。

本次同 seed、8-env、先站 2 秒再切換、共 12 秒、沒有中途 reset 的小測：

| 指令 | 曾跌倒／失敗的環境 |
|---|---:|
| +0.4 m/s | 0/8 |
| +1.2 m/s | 7/8 |

修正 Web 後端前進上限為 +0.4 m/s，倒退 -0.2、轉向 ±0.15 以降低突變。只有 +0.4 前進在上述小測通過；這不等於倒退、轉向、急停都已通過。真正交替 Sliding 是另一個技能，原 4k 模型仍沒有有效左右推蹬循環，仍禁用。

## 5. 後翻及下一步

12k 後翻原始 64-env 驗證為 0/64；本次無 PPO 的原生後翻控制器與原 preview 也都 8/8 曾失敗。因此需先校對負載／推蹬／收腿／落地的 reference、質心與接觸控制，不能保證多跑 iterations 就會成功。輪子初始化修正也沒有解決後翻軌跡。

本次未啟動新 PPO 訓練，沒有修改既有 checkpoint。後續應先通過穩定下跪保持及起身、原地起跳的基準控制，再按新基準重訓與驗證後翻。既有模型的舊實驗報告保留，不改成成功。

## 複現與限制

- `tools/audit_native_pose_geometry.py --out <json>`：CPU / numpy CAD 幾何，不啟動 Isaac。
- `./run_isaac.sh tools/probe_native_calibration.py --out <json>`：8-env 原生馬達／PPO 對照，沒有外力支撐或隱藏 reset。
- `motor_probes.json` 是修正前程式（基於 `288c28f`）的測試；初始 height 0.45 與 trace 可核對。以新程式重跑會使用已修正的 reset，不能混作同一條實驗。
- 小測統計是單一 seed／8 個平行環境，目的在重現與隔離問題，不是通用成功率。
