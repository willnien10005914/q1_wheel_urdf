# 動作按鈕整理（2026-10-12）

Stand up 原本從跪姿轉交 Unbox，同一功能合併為「起立（Unbox PPO）」。Recovery PPO 原本依躺姿選同一個 Gripper 模型，已移除重複入口，保留四個明確的 Elbow／Gripper 按鈕。舊 API 名稱保留相容。

所有按鈕共用短暫按下動畫；送出、準備、執行與完成狀態分開。完成後沒有持續的選取框，鍵盤焦點另外使用 focus-visible 樣式。Chrome 瀏覽器以明確標記的 UI fixture 檢查單一起立按鈕、發送 unbox 指令、各類按鈕動畫、執行／完成狀態及窄畫面捲動；無 JavaScript 錯誤。browser_result.json、executing.png、mobile.png 為 UI 驗證，不是機器人動作成功證據。

tests/verify_button_feedback.cjs 可重跑，需安裝 Playwright，或以 PLAYWRIGHT_PACKAGE 指定現有套件。Q1_UI_TEST_URL 預設 http://127.0.0.1:8789；先從 repo 根目錄啟動本機 HTTP 伺服器。測試會攔截 API，不操作使用者的模擬器。
