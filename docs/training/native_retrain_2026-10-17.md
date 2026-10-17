# 2026-10-17 RTX 4090 新 URDF PPO 重訓

來源：GitHub main `358c668`。本機 RTX 4090 Laptop 16 GiB，系統 RAM 約 30 GiB。此機驗證 x86/CUDA 執行環境，不代表在本機重新驗證 Spark ARM 硬體。

新版原生契約是 23 action、90 actor/critic observation。舊版 24 action PPO 不直接移植；保留所有舊 checkpoint。新版 URDF 的 Inertia-Skate 基礎任務為 23/78/81，與四技能 23/90/90 亦不同，不混用 checkpoint。

## 執行順序

1. CUDA runtime doctor、來源 URDF 對 PhysX 質量／質心／完整慣量／mimic 實測。
2. 四個原生任務各 64 env × 3 updates，接著在 seed 9127、8 env、500 steps 做首次嘗試評估。小批任務不要求已收斂，但必須正常更新、儲存與完成評估，不得有非有限值。
3. 同一 Stand 任務比較 1024／3072／6144 env，每組 10 updates，記錄 GPU 峰值及訓練吞吐量。只跑一個模擬程序。
4. 依量測選正式批量，使用持久化單 GPU 佇列。Stand → Walk → Standup → Backflip，後續技能由同契約 Stand warm-start。每段獨立驗證，通過且分數更佳才發布 model.pt；latest.pt 只代表訓練預覽。

預算沿用新版正式計畫：Stand 5000、Walk 20000、Standup 20000、Backflip 40000 updates；每 1000 updates 驗證。環境數增加會增加總 transitions，不保證後翻或起立成功。

## 尚未遷移的項目

新版目前只有以上四個 PPO 訓練任務；Kneel 是同一控制器的下降模式，沒有獨立 Kneel PPO。舊版 Slide 及正反躺 Elbow／Gripper 四種 Recovery 尚未提供新 URDF 訓練／驗證契約。因此本輪不能把它們列成「新 URDF 已排訓」或用舊 24-action 模型代替。後續需先遷移重置姿勢、原生軸向、接觸階段及完成條件，再各跑小批驗證後加入正式佇列。

## 記錄

原始 log、完整 state 與容量取樣在本機 outputs/native_20261017/、logs/native_manager_queue/。提交摘要會記錄實測結果，不把開始訓練當作已完成。

## 本機驗證結果與正式設定

- CUDA runtime doctor 通過，PyTorch 2.7.0+cu128、RSL-RL 5.0.1；CPU checkpoint 相容性 3 tests、queue 6 tests、manager regression 8 tests 皆通過。
- PhysX 來源契約通過：41.052265167 kg、最大慣量誤差 9.18e-9 kg m²、4 個 mimic；64 個物理步觀測有限。Mimic 有限增益追蹤誤差最大約 0.076 rad，不是理想剛性連桿。
- 四個 64-env PPO 均實際完成 3 updates、儲存 checkpoint，所有模型／optimizer tensor 有限。Stand 評估 8/8 成功、0 跌倒；Walk 1/8 曾達站穩條件但 8/8 最後失敗且速度追蹤未達標；Standup、Backflip 0/8。後翻有 6/8 離地，但均非法接觸失敗。這是 pipeline smoke，不把未收斂模型當成功。
- 每個容量測試均 10 updates，4 組 checkpoint／optimizer tensor 皆有限。摘要與 SHA-256 見 native_retrain_20261017/。

| envs | GPU 峰值 MiB | 穩態秒/update | 約 transitions/s |
| ---: | ---: | ---: | ---: |
| 1024 | 3180 | 1.665 | 14760 |
| 3072 | 4150 | 2.270 | 32479 |
| 6144 | 5496 | 3.260 | 45232 |
| 12288 | 8188 | 5.345 | 55175 |

採 6144：12288 雖放得下，但在相同更新次數下總時程約增 64%，吞吐量只增加約 22%。正式設定 config/native_training_4090_20261017.json：6144 env、chunk 1000、evaluation 64 env × 500 steps、訓練 seed 42、評估 seed 9127。預算共 85,000 updates、名義 12,533,760,000 transitions。這是預算，非已完成量。

正式訓練在 ~/projects/q1_wheel_urdf，使用 q1-native-training user service；checkpoint 在 checkpoints/native_manager，完整模型／來源雜湊在 outputs/native_manager/queue/status.json。原始 GitHub 更新及既有模型保持原樣。本輪只新增實驗計畫與驗證紀錄。

```sh
cd ~/projects/q1_wheel_urdf
./start_native_training.sh --config config/native_training_4090_20261017.json
python3 tools/native_training_queue.py --status
systemctl --user status q1-native-training
# 需要停止時保留已儲存 checkpoint：
systemctl --user stop q1-native-training
```

短測推估 Stand 約 4–5 小時，完整四階段約 3–5 天；後三技能接觸、GPU 溫度與後續迭代速度可能不同，需以實際 log 修正。每 1000 更新評估，latest.pt 與已驗證 model.pt 分開；完成預算但沒通過會記為 completed_with_unvalidated_skills。尚未遷移的 Slide／四種 Recovery 不在此四階段預算內，也沒有宣稱已開始新 URDF 大訓練。
