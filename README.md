# AIoT HW2 — Multiple Linear Regression 與 Feature Selection

本專案以 Student Performance 與 Appliances Energy 兩題，研究五種特徵選擇對線性迴歸的影響。
沿用既有 Python Pipeline 與 DataLab 互動儀表板；所有指標、係數、預測、區間及 PNG 都來自真實執行。
正式範圍只有這兩題，不提供任意資料上傳。

## 作業目標與完成範圍

- 原始輸入數符合 14–39，區分 original、encoded、selected 與有效設計 rank。
- Baseline 加 Correlation、Mutual Information、RFE、RFECV、LASSO，最終 predictor 全為 LinearRegression。
- 完整 Pipeline 與 training-only nested CV；保留獨立 test，不以 test 選模型。
- 五項迴歸假設、95% mean CI／individual PI、報告圖表、兩題靜態互動介面及自動驗證。
- 本次實際執行 FAST。FULL 尚未執行；AI 附件與最終 PDF 仍須人工完成。

## 資料集、目標與特徵數

| 題目 | 樣本 | CSV 欄位 | 原始輸入 | 完整編碼 | 目標 | 單位 | Train / test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 學生表現 | 395 | 33 | 30 | 56 | G3 | grade (0-20) | 316 / 79 |
| 家電能源 | 19735 | 29 | 25 | 25 | Appliances | Wh | 15788 / 3947 |

Student 的 G3 是 0–20 數學期末成績。原始 student-mat.csv 有 33 欄、32 個非目標欄位；
本專案明確排除歷史 G1/G2，採 early-30。加入 G1/G2 會改變實際可預測時點，不能混稱為同一問題。

Energy 的 Appliances 單位 Wh；排除目標、rv1、rv2，date 另存時間 metadata，保留 lights。
完整欄位、逐欄驗證、來源 hash 與欄位定義見 [資料說明](datasets/README.md)。

### 資料來源差異與取得方式

指定 [Student Kaggle](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set) 的已下載 archive
只有 student-por.csv（649 筆），不能冒充數學科。
真正 Mathematics 檔來自 [UCI Student](https://archive.ics.uci.edu/dataset/320/student+performance)：
下載原始 ZIP、再解壓內層 student.zip，放置 student-mat.csv 於 datasets/student_performance/。

[Energy Kaggle 副本](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)
內原名 KAG_energydata_complete.csv，放置為 datasets/appliances_energy/energydata_complete.csv；
原始參考是 [UCI Appliances](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)。
兩題 CSV 已在儲存庫，source.json／student-mat.source.json 保留來源與 hash。
網路或認證受限時請從來源頁下載真實檔，不生成替代資料或提交憑證。
**交付是兩個真實驗證資料集，但不宣稱兩份原檔都從指定 Kaggle 下載；學生來源差異仍須教師接受。**

## 專案結構

    datasets/                      schema、loader、ColumnTransformer、split 與原始 CSV
    feature_selection.py           五種選擇器與快取排名
    regression_pipeline.py         training-only nested CV、FAST/FULL
    phase2_experiment.py            訓練、模型與 CSV/JSON
    regression_diagnostics.py      rank、VIF、假設、CI/PI
    phase3_evaluation.py            指標、26 張 PNG 與診斷表
    dashboard_integration.py        已擬合模型的靜態推論 descriptor
    dashboard_export.py / main.py   正式 dashboard export
    dashboard/dist/                原有 DataLab HTML/CSS/JS 與真實 data bundle
    report_preparation.py           從實際產物產生 README、CRISP-DM 與圖表索引
    validate_project.py             完整端到端驗證與命令紀錄
    tests/                         unit、backend integration 與 Edge browser QA
    results/<dataset>/FAST/         正式模型、CV、預測與 runtime
    reports/figures/<dataset>/FAST/ 300 dpi PNG
    reports/tables/<dataset>/FAST/  metrics、VIF、assumptions、interval rows
    reports/phase5/                 最終驗收、環境與 audit
    historical/                    歷史參考，不是正式題目

## 安裝與環境

建議 Python 3.12、四核心桌機與足夠記憶體；本次實際版本記錄於
[environment.json](reports/phase5/environment.json)。
一般安裝：

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt

requirements.txt 是 runtime；requirements-dev.txt 另含 Playwright／Pillow；
requirements-lock.txt 記錄本次已驗證的直接套件精確版本，不冒稱跨平台完整 transitive lock。
重現同環境可改用 pip install -r requirements-lock.txt。
Browser QA 預設使用已安裝的 Microsoft Edge（channel=msedge）；請先安裝 Edge。
只做 Python 建模不需要瀏覽器。模型 joblib 只載入可信專案產物，換套件版本後應重新訓練，不保證跨版本 pickle 相容。

## FAST／FULL 執行

    python -m datasets
    python phase2_experiment.py --mode FAST --jobs 2
    python phase3_evaluation.py --mode FAST
    python dashboard_export.py --mode FAST
    python report_preparation.py --mode FAST

main.py --mode FAST 等同正式 dashboard export，不執行歷史題目。
完整驗證（含重新訓練、圖表、匯出、文件與 browser QA）：

    python validate_project.py

已完成新一輪訓練時，可明確重用模型：

    python validate_project.py --reuse-training

FULL 是更徹底的 training-only 探索，可能較耗時：

    python phase2_experiment.py --mode FULL --jobs 2
    python phase3_evaluation.py --mode FULL
    python dashboard_export.py --mode FULL
    python report_preparation.py --mode FULL

| 設定 | FAST | FULL |
| --- | --- | --- |
| inner / outer folds | 2 / 2 | 4 / 5 |
| k | 1、5、10、20、all（可行範圍） | 全部可行 k |
| LASSO alpha fraction | 0.01、0.1 | 0.0001、0.001、0.01、0.1、1 |
| 用途 | 開發／除錯證據 | 更徹底最終評估 |

FAST 與 FULL 統計證據不等價，本次沒有執行 FULL。
FULL artifacts 必須真實存在才能產圖／匯出，不以 FAST 假冒。
--jobs 限 1 或 2，native thread=1，outer 不再平行；.cache/phase2/ 僅快取相同訓練 fold。
--no-cache 可停用；cache hit 與 cold runtime 不可混為效能比較。

## 前處理、切分與洩漏防治

數值 median imputation＋StandardScaler；學生 nominal 類別 most-frequent imputation＋
OneHotEncoder(handle_unknown="ignore")。
所有 transformer、ranking、selection 均在 training fold 內 fit，未知類別以零 indicator 處理。

學生 seed=42，316 train / 79 test，inner/outer shuffled KFold。
能源保留時間序，15,788 train / 3,947 test，inner/outer expanding TimeSeriesSplit，預設 gap=0；
可用 --gap 指定列數 embargo。date 不進模型，不用未來資料補過去值。
內層調參、外層評估選擇流程；全訓練 inner CV 決定正式方法，test 只在選擇固定後評估。
測試集已被觀察，後續不可反覆依 test 改設計。

## 五種 Feature Selection

- Correlation：訓練絕對 Pearson 排名。
- Mutual Information：互資訊排名，二元 indicators 視為 discrete，固定隨機種子。
- RFE：標準化線性係數遞迴消除，快取完整路徑避免逐 k 重算。
- RFECV：完整 raw-input Pipeline CV 在 RFE 路徑選 k；可能與 tuned RFE 相同。
- LASSO：僅選非零 support，之後以 LinearRegression 重擬合；空 support 有明示一欄 fallback。

## 評估與五項回歸假設

MAE／MSE／RMSE／R² 分開記錄 train、CV、test。CV std 是 fold RMSE 標準差，不是 CI。
Adjusted R² 採含截距的有效 affine rank，自由度不足為空；test adjusted R² 僅為描述性懲罰。
Phase 5 已將 Phase 2／Phase 3 的 Adjusted R² 統一為相同 training affine rank；歷史名義欄數版本不再用於正式結果。

五項證據包括 residual-vs-fitted 與分箱趨勢（線性）、DW／ACF／能源 Ljung-Box（獨立性）、
Koenker-Breusch-Pagan（等變異）、直方圖／Q-Q／Shapiro 或 Jarque-Bera（常態性）、
VIF／rank／condition（共線性）。
完整 one-hot 可能 alias；不可識別係數不提供假 SE/CI。
不自動宣稱假設成立；統計顯著也不是因果。

95% mean CI 與 individual PI 使用選定設計的 statsmodels OLS 等價滿秩基底。
學生 HC3、能源 HAC 另供漸近 robust mean CI，並非 robust individual PI。
超出 affine span 不輸出可識別區間；未校正 selection uncertainty，不保證覆蓋，不任意裁切上下界。

## 實際 FAST 結果

| 題目 | 方法 | encoded k | Inner CV RMSE | CV std | Test RMSE | Test R² |
| --- | --- | --- | --- | --- | --- | --- |
| 學生表現 | Correlation | 1 | 4.2798 | 0.0346 | 4.3446 | 0.0795 |
| 學生表現 | RFE | 1 | 4.2798 | 0.0346 | 4.3446 | 0.0795 |
| 學生表現 | RFECV | 1 | 4.2798 | 0.0346 | 4.3446 | 0.0795 |
| 學生表現 | Mutual Information | 10 | 4.3164 | 0.0908 | 4.3185 | 0.0905 |
| 學生表現 | LASSO | 3 | 4.6109 | 0.0384 | 4.3107 | 0.0938 |
| 學生表現 | Baseline | 56 | 4.7876 | 0.0924 | 4.1957 | 0.1415 |
| 家電能源 | Mutual Information | 10 | 96.0286 | 1.0752 | 93.1965 | -0.0480 |
| 家電能源 | RFE | 10 | 98.2936 | 1.6599 | 87.0556 | 0.0856 |
| 家電能源 | RFECV | 10 | 98.2936 | 1.6599 | 87.0556 | 0.0856 |
| 家電能源 | Correlation | 5 | 98.4382 | 0.0480 | 94.7469 | -0.0831 |
| 家電能源 | LASSO | 4 | 99.2578 | 0.6517 | 94.7312 | -0.0828 |
| 家電能源 | Baseline | 25 | 100.4710 | 2.8834 | 86.7329 | 0.0924 |

正式學生推薦 Correlation，能源推薦 Mutual Information，依 training CV 決定。
Baseline 測試結果可能更好，並未據此重選。負 test R² 保留，不隱藏失敗。
兩題 RMSE 單位不同（grade、Wh），不能直接以大小比較模型品質。
完整 train/test MAE、MSE、rank-adjusted R² 與 nested CV 見
[CRISP-DM 報告](reports/CRISP_DM_REPORT.md)及 reports/tables/。

## 實際時間

| 題目 | Pipeline 秒 | 診斷／圖表秒 | 上述合計秒 | cache namespace 已存在 |
| --- | --- | --- | --- | --- |
| 學生表現 | 15.58 | 7.39 | 22.98 | False |
| 家電能源 | 19.18 | 10.45 | 29.62 | False |

Pipeline 包含載入／CV／評估／匯出，不含 Python 匯入及程序啟閉。
診斷包含 13 圖／題。上述合計不含共用 dashboard、browser QA 或文件產生；
共用命令 wall time 與精確測試數見 [驗收 JSON](reports/phase5/validation.json)。

## 儀表板與本機執行

保留既有雙題、六模型、k、CV 曲線、係數／統計、殘差排序／分頁、數值與類別預測、
95% CI/PI、外插警示、情境比較、CSV、深色與響應式介面。切題隔離全部資料與情境。
全部前端數值由 Python 真實產物提供，與 sklearn／statsmodels 逐筆核對。
靜態架構不需要後端 API 或前端 build。

    python -m http.server 8000 --directory dashboard/dist

開啟 http://localhost:8000 ，也可直接開 dashboard/dist/index.html。
相對資產支援 project-subpath hosting；未載入或不相容的資料有明確空狀態。

## 測試與驗收

    python -m unittest discover -s tests -p "test_*.py" -v
    python tests/verify_phase2_results.py
    python tests/verify_phase3_results.py
    python tests/verify_official_dashboard.py

最終通過／失敗數、模型／預測／區間核對數、資產 SHA、四種 viewport 與命令時間，
以 [Phase 5 驗收](reports/phase5/validation.json) 為準。
歷史的 frontend-byte preservation 已改為 --check-historical-frontend 顯式選項，
不會把 Phase 4 正常更新判成後端失敗，也不偽稱前端沒變。
歷史驗證檔不屬正式雙題驗收，不對舊題重跑模型。

## GitHub Pages 發布

既有 .github/workflows/static.yml 上傳 dashboard/dist，使用相對資產。
到 Settings > Pages > Source 選 GitHub Actions；先本機驗證，再由使用者授權 commit/push 到 main，
或對已存在 commit 手動 workflow_dispatch。完成後查看 Actions、deployment URL 與公開頁實際內容。
本輪不 commit/push，也不因本機相容性通過便宣稱新內容已部署。

## 報告與 AI 使用附件

- [繁體中文 CRISP-DM 稿](reports/CRISP_DM_REPORT.md)：六階段、真實數值、26 圖及限制。
- [圖表索引](reports/FIGURE_INDEX.md)／[CSV 索引](reports/figure_index.csv)。
- [AI Assistance 待補欄](reports/AI_ASSISTANCE.md)：真實 ChatGPT 對話 PDF、
  NotebookLM 至少 100 中文字的來源導向摘要、最終 PDF 匯出。
- [Phase 5 審查紀錄](PHASE5.md)。
- 較早階段文件與歷史實驗保留為參考；正式目前範圍以本文件為準。

## 已知限制

FAST folds 少；小學生樣本／類別多、能源自相關／異質變異與時段轉移，
使 OLS 假設、選擇後推論及區間覆蓋受到限制。學生 Kaggle subject 差異已揭露；
不虛稱兩個指定 Kaggle 原檔均符合預期。FULL、NotebookLM、真實對話 PDF 與最終 PDF 排版尚待完成。
公開部署狀態須另驗證。無任意上傳、無自動因果結論。

## 參考資料

[UCI Student](https://archive.ics.uci.edu/dataset/320/student+performance)、
[UCI Energy](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)、
[sklearn 洩漏防治](https://scikit-learn.org/stable/common_pitfalls.html)、
[statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)、
[Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
完整來源見 CRISP-DM 報告。
