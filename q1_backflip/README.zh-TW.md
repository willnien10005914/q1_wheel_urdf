# Q1 獨立後空翻訓練

此套件使用原 Q1 URDF／CubeMars actuator，輸出、USD cache、checkpoint 均存放於目前 workspace，不改動原 Q1 專案。G1 repository 保持獨立，僅將其參考關節動作做 Q1 限制內的初步映射與重定時；這不是已驗證的動力學 retargeting 成果。

正式訓練：6144 environments × 20000 PPO iterations，seed 42，50 Hz policy / 200 Hz physics。以 256×100 基礎模型 warm start，正式 iteration 從 0 開始，完整跑 20k。前 300 updates 訓練微彎站姿，300～1000 漸進混入翻轉參考與 reward，之後完整翻轉／落地目標。起始 knee=0.30 rad、hip=−0.15 rad；pelvis height 用 URDF FK 算得約 0.8813 m。

目前只啟動雙輪微彎落地版本；前後错腳跪姿候選已留 `--landing kneel`，尚未獨立訓練或驗證。自碰撞與原 Q1 設定一致關閉，最終驗收仍需單獨檢查。無施加外力、無每步改寫 root、膝 roller 僅跟隨膝關節 mimic。嚴格成功需要實際上升離地、後轉完整一圈、雙輪支撐、姿態與速度穩定保持 2 秒且無非法接觸。

已跑的小批量：64×10（回合／數值）、256×100（PPO 參數／學習）、64×20（翻轉階段）、6144×3（容量與修正後的翻轉程式路徑）。均能完成；早期尚無成功後空翻。這些檢查驗證可執行與數值有限，不能當成動作成功率驗收。

查看狀態：

```bash
tail -f full6144_20k.log
cat full20k.pid
```

Run 目錄位於 `logs/q1_backflip/*_full6144_20k_wheels/`，包含 `manifest.json`、`source_snapshot/`、`env.yaml`、`agent.yaml`、TensorBoard events 與每 200 iterations 的 `model_*.pt`。`running.json` 記錄 PID 與預算；正常完成後另有 `completed.json`。異常退出應以日誌及程序存在性判定，不能只看 `running.json`。

重啟入口 `bash q1_backflip/start_full.sh` 有數值 gate 及 PID 檢查，但從 probe 重新 warm start。中斷續訓應明確使用現有最新 checkpoint：

```bash
bash q1_backflip/launch.sh --num_envs 6144 --max_iterations 20000 \
  --resume /absolute/path/to/model_XXXX.pt --run_name resumed6144_20k
```

`--max_iterations` 是目標總 iterations；resume 從已保存 iteration 的下一個開始。不要在已有長跑程序時重複啟動。網頁 Backflip 尚未接入，應待獨立評估及 controller handoff 驗證後再加入。
