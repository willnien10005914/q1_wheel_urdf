# 網頁與鍵盤修正驗證（2026-10-12）

修正舊網頁固定連接 8766、右側滑桿被按鈕擠出畫面、新播放器未訂閱 Isaac 鍵盤的問題。新版使用同源 API、可捲动側欄與行動版頁面，並在站立待命時接受位置／輪速覆蓋。

Chrome 無頭瀏覽器連接獨立 Isaac 後端 8787，確認 26 個滑桿、390×844 畫面捲動、肘部目標實際作用、Release all、正反躺與 recovery 按鈕派送、Backflip 請求、W/A 指令與放開停止。這是控制連線驗證，不代表每個實驗 recovery 已成功起身。結果見 browser_result.json，截圖見 desktop.png 與 mobile.png。

另用真正有視窗的 Isaac 在 8788 執行 5 步，確認鍵盤訂閱與正常關閉；鍵盤事件與技能保護由 tests/test_skill_controls.py 驗證。此次沒有更換模型或重訓。
