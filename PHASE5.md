> 本文件保留 Phase 5 完成時的雙題驗收紀錄。最新三題範圍見 README.md 與 PROBLEM3.md；下方驗收 JSON 由目前三題驗證更新。

# Phase 5 — 最終 QA、GitHub 文件與 CRISP-DM 報告

本階段延續既有實作，正式題目僅 Student Performance 與 Appliances Energy。不 commit／push／部署。

## 審查與修正

- 逐一檢查 Python 語法／匯入、正式原始資料、模型、CV、預測、診斷、PNG、dashboard assets 及文件。
- 舊 Phase 2/3 驗證仍強制前端是 Phase 2 snapshot，與 Phase 4 衝突；改為預設驗證當前 backend，
  嚴格歷史 hash 另用 --check-historical-frontend。報告如實標示 frontend_unchanged=false，不偽稱沒變。
- 修正 Phase 3 驗證的 seed／gap 由 run configuration 取得，避免資料專屬 hardcoding。
- 歷史 browser scripts 指向其歷史資料／快照，不把舊題放回正式 dropdown。
- 修正 Phase 2 Adjusted R² 的名義欄數計算，改用 training-fitted affine rank，與 Phase 3／dashboard 統一；補上 aliasing 與跨階段一致性測試。此修正不改選擇或預測。
- requirements-dev 補 Pillow；requirements-lock.txt 與 reports/phase5/environment.json 記錄實際版本。
- 新增 validate_project.py，一次執行 FAST、診斷、dashboard、文件、unit／integration／browser QA，
  並保存每個命令 return code、wall time 與 log。
- 新增 report_preparation.py 與 docs/README.template.md，以真實 run／診斷產物產生 README、
  CRISP-DM 報告與 26 圖索引，拒絕過期 run hash 或缺圖。
- 原始／歷史資料 hash 受保護，兩題結果與圖表路徑隔離。

## 實際執行與可重現入口

    python validate_project.py

本輪已明確重跑兩題完整 FAST，再用 --reuse-training 完成下游整合驗收；
這個旗標不冒稱又訓練一次。Pipeline 原始時間由兩題最新 run.json 記錄。
FULL 沒有執行，不宣稱 FAST 與 FULL 等價；完整版入口及資源限制見 README。

精確通過／失敗數與實際時間，以 [validation.json](reports/phase5/validation.json) 為準。
涵蓋 Baseline＋五選擇器、fold-local 前處理／選擇、holdout 隔離、時間序、
原始／encoded counts、12 個正式模型、實際預測與區間、26 張 300 dpi 圖、
兩題互動／CSV／情境／theme／四種 viewport／相對 assets／缺資料空狀態。
[環境](reports/phase5/environment.json) 與 [檔案 inventory](reports/phase5/audit_inventory.json) 可追溯審查範圍。

## 文件交付與已揭露限制

- [README](README.md)：安裝、來源、結構、FAST/FULL、方法、評估、限制、真實結果、部署。
- [CRISP-DM 繁中報告](reports/CRISP_DM_REPORT.md)：兩題六階段、train/CV/test、五假設、
  rank-aware adjusted R²、CI/PI、全部真實圖表、跨單位比較限制。
- [圖表索引](reports/FIGURE_INDEX.md)：全部 26 檔的圖號／檔名／用途／章節。
- [AI Assistance 待補](reports/AI_ASSISTANCE.md)：真實 ChatGPT 對話 PDF、
  NotebookLM 至少 100 中文字來源摘要與最終 PDF 排版／匯出。

指定 Student Kaggle archive 實際為 Portuguese，正式 Mathematics 使用已追溯 UCI 原檔；
不能聲稱兩份指定 Kaggle 都符合預期。early-30 明確排除歷史 G1/G2。
能源保留 lights，date 為 metadata，不假稱同期感測估計就是提前預報。
學生少樣本、能源自相關／異質變異／非常態，以及 selection uncertainty 均限制推論與區間覆蓋。

GitHub Pages 僅驗證靜態相容性與本機 nested path，沒有宣稱本輪新變更已公開部署。
待使用者檢查實際 AI 附件、補身份／課程資訊與最終 PDF；本輪不 commit 或 push。
