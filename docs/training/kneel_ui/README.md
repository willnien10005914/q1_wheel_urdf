# 動作按鈕整理（2026-10-12）

Stand up 原本從跪姿轉交 Unbox，同一功能合併為「起立（Unbox PPO）」。Recovery PPO 原本依躺姿選同一個 Gripper 模型，已移除重複入口，保留四個明確的 Elbow／Gripper 按鈕。舊 API 名稱保留相容。

所有按鈕共用短暫按下動畫；送出、準備、執行與完成狀態分開。完成後沒有持續的選取框，鍵盤焦點另外使用 focus-visible 樣式。Chrome 瀏覽器以明確標記的 UI fixture 檢查單一起立按鈕、發送 unbox 指令、各類按鈕動畫、執行／完成狀態及窄畫面捲動；無 JavaScript 錯誤。browser_result.json、executing.png、mobile.png 為 UI 驗證，不是機器人動作成功證據。

tests/verify_button_feedback.cjs 可重跑，需安裝 Playwright，或以 PLAYWRIGHT_PACKAGE 指定現有套件。Q1_UI_TEST_URL 預設 http://127.0.0.1:8789；先從 repo 根目錄啟動本機 HTTP 伺服器。測試會攔截 API，不操作使用者的模擬器。

## Kneel 與 Unbox 實際物理驗證

Kneel 改為從當前角度限速下降，只限制膝、髖前後與腰前後角度；保留 PPO 的左右平衡。站立輪控制持續至膝輪真正接觸地面，再漸變交給姿態 PPO；站立重力補償逐步退場。完成需確認跪姿角度、低速度及至少三個輪接觸，避免把半蹲誤認為完成。

seed 4107、每批 4 個環境，從靜止站立及 Skate 停止後各測一次，共 8 個案例全部完成 Kneel。以 5 N 門檻檢查當前接觸及接觸歷史，雙腳底輪同時失去接觸的幀數均為 0；詳見 motion_result.json。這是此批驗證結果，不代表任意初始狀態的成功率。

新的平順跪下會留下與 Unbox 訓練起始姿態不同的手臂角度，直接交接曾失敗。現在標準 Unbox 先以馬達平順移動肩、肘、腰至訓練角度，並僅對這些關節加重力補償；確認實際角度及速度後才啟動原 PPO，網頁顯示「準備中」。準備超時顯示失敗，需 Reset。後翻落地的實驗 Unbox 分支維持原流程。未改模型權重、URDF 或訓練快照，Kneel／Unbox 交接未寫入根部或關節模擬狀態。

[kneel_unbox.mp4](kneel_unbox.mp4) 為連續物理模擬：3 秒按 Kneel、7 秒按 Unbox；完成起立後持續平衡至 22 秒，最終高度 0.8402 m、狀態 unbox_hold。frame_0900.png 為 18 秒站立畫面。錄製命令：

```sh
bash play_skills.sh --headless --no-browser --web-port 8791 --steps 1100 --sequence 0:reset,150:kneel,350:unbox --out /tmp/q1_kneel_unbox.json --video-path docs/training/kneel_ui/kneel_unbox.mp4
```

單次動作在準備／執行時禁用重複點擊；完成後恢復一般外觀。瀏覽器 fixture 也檢查此行為和 Unbox 準備狀態。
