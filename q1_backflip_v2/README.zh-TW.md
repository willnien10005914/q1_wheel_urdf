# Q1 單次蓄力後翻實驗

使用原 Q1 URDF、質量及 CubeMars 電壓／電流／力矩限制。程式與模型留在獨立工作目錄；目前沒有接入原專案的網頁。

動作從 knee 0.30 rad 開始，先保持 0.4 秒，再做一次平滑下蹲與蹬地。收腿由真實離地及全身質心上升觸發。蓄力前離地視為失敗；禁止身體非法接地；不注入外力或改寫飛行中的 root pose。

`env.py` 是單次蓄力模型，`compact_env.py` 另外控制空中的肩膀、手肘與伸腿角度。位置目標受 URDF 限位及速度限制，速度前饋只取平滑參考動作，避免把 PPO 探索雜訊轉成高頻速度命令。

`search.py --replicas 4` 會讓每個候選參數經過四次不同馬達隨機化，再用平均分數減去一半標準差選擇候選。`--landing_goal` 固定已搜尋出的起跳參數，只搜尋空中姿勢、收腿延遲、伸腿角度及輪子／跪姿落地選擇。

訓練與測試：

```bash
bash q1_backflip_v2/launch.sh --num_envs 64 --max_iterations 10 --teacher q1_backflip_v2/teacher_landing.json --run_name landing_smoke --save_interval 5
bash q1_backflip_v2/launch.sh --num_envs 6144 --max_iterations 20000 --teacher q1_backflip_v2/teacher_landing.json --run_name landing6144_20k --save_interval 200
bash q1_backflip_v2/evaluate.sh --checkpoint logs/q1_backflip_v2/<run>/model_19999.pt --out artifacts/v2/<evaluation>
```

成功需要真實起跳、足夠的空中後翻轉角、回到接近一圈的姿態、合法支撐、角速度與平移速度足夠低，且連續保持平衡兩秒。影片或獎勵上升本身不能證明成功。測試保存模型及環境 SHA256、提前小跳、碰地部位、骨盆高度與邊界終止資訊。

目前仍採用原本實驗的「不啟用自碰撞」設定；正式接入網頁前需另外驗證。新收腿模型的骨盆高度終止值是 0.04 m，因原 0.18 m 會在合法離地期間提前結束；全身非法接觸判定仍保留。

截至初步驗證：512 個環境／500 次 PPO 參數有限、96 次測試無提前小跳但無成功落地。之後的收腿候選在獨立 96 次測試中有 76 次空中轉角至少 2π，仍為 0 次穩定落地；大腿先接地與落地後位移是下一階段改善目標。詳細原始結果在 `artifacts/v2/compact_robust_eval96/evaluation.json`。

後續加入 `landing_env.py`：真實落地後，以量測的軀幹 pitch 修正跪姿 hip pitch，仍由原馬達模型出力。獨立 96 次驗證已有 25 次成功（連續平衡至少兩秒），最長 4.02 秒，提前小跳 0 次。這是 GPU 動作搜尋搭配姿態回授的候選；PPO 模型仍需完整訓練。成功影片用相同 seed 4107、32 個物理環境、env 0 重現，平衡 3.56 秒；其餘環境只在渲染器隱藏，物理未改變。

使用者要求顯存足夠且有效時把環境數加倍。12288 個環境的 30 次訓練通過，顯存峰值 9138 MiB、最低可用 6816 MiB。相同第 10–29 次迭代，6144 與 12288 每秒樣本均值分別為 54291.95 與 67866.30，增加約 25%。迭代耗時由 2.716 秒增為 4.3465 秒，因此同樣 20k 會有約兩倍樣本、較長訓練時間，並非把 20k 跑得更快。

正式長跑已切換成 12288 個環境，從短測 `model_29.pt` 繼續至總計 20000 次迭代，PPO 初始探索標準差 0.05。`v2_full_pipeline.json` 記錄進度；`v2_landing12288_20k.log` 是訓練日誌；結束後自動做 96 次独立測試、錄影與上傳。大環境短測與詳細比較保存於 `artifacts/v2/envs_benchmark/`。

已驗證成功的候選影片：https://gofile.io/d/Y4vvTa2R
