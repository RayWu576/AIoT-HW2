# Problem 3 — Car Price Prediction

在保留 Student Performance（Problem 1）及 Appliances Energy（Problem 2）的基礎上，新增正式第三題。沿用現有 Pipeline、五選擇器、診斷／報告產物及 DataLab；沒有重新建立介面。

## 已驗證資料與前處理

- Kaggle：https://www.kaggle.com/datasets/hellbuoy/car-price-prediction 。
- datasets/car_price/CarPrice_Assignment.csv 為既有真實來源檔，SHA-256 已對照 source.json。
- 205 筆、26 欄；排除 car_ID、price，保留 **24 個原始候選輸入**（14 numeric、10 categorical），目標 price（USD）。
- CarName 有 147 個不同車名，以 deterministic manufacturer transformer 取得 22 個廠牌；修正 maxda、porcshce、toyouta、vokswagen、vw 等固定拼字。
- 所有 imputer、scaler、encoder、ranking／selection 在訓練 fold 內擬合；OneHotEncoder(handle_unknown="ignore")。
- seed=42，164／41 shuffled holdout，inner／outer 各 2 folds（FAST）。本輪完整訓練編碼寬度為 **73**，不與原始 24 混稱。
- FULL grid 上界也採訓練部分的廠牌數，避免以 147 個原始車名誤設 k、產生大量等效重複候選。

## 實際模型與統計限制

Baseline、Correlation、Mutual Information、RFE、RFECV、LASSO 全部執行，最終 predictor 都是 LinearRegression。
Training CV 推薦 LASSO support，k=38，inner CV RMSE=3183.2454 USD；test RMSE=2869.8225 USD，R²=0.8957。
LASSO 只作選特徵，並非最終預測器。Baseline 的 test 雖較佳，不據此改選。
重複 FAST 的選擇、CV 與 test 指標一致，參照 results/car_price/FAST/reproducibility_reference.json。

含截距有效訓練 rank 為 37，少於 39 個名義設計欄，需處理 aliasing。Test 只有 41 筆，
rank-adjusted test R² 的剩餘描述性分母僅 4；不可與一般重新擬合的 adjusted R² 推論混稱。
Baseline 的 training rank=60 大於 test 樣本數，因此 test adjusted R² 為空值。
稀疏類別可能出現 leverage=1，HC3 mean CI 會誠實標示無法估計；未提供偽造值。
經典 OLS mean CI／individual PI 仍須依假設及 estimability 判讀，不代表當前市場估價保證。

## 儀表板與報告

第三題切換更新所有模型、k、CV 曲線、USD 指標、係數、五項中文診斷卡片、24 原始欄位表單、
廠牌 dropdown、CI／PI、外插警示與 CSV，並清空其他題情境。保留放大的字體與深／淺色手機介面。
前端匯出 exact training-fitted imputation／scaling／one-hot，含 manufacturer lexical aliases；
以所有候選模型實際預測及缺值／未知類別／vw 別名測試核對 Python。

新增 13 張 300 dpi PNG 於 reports/figures/car_price/FAST/，
三題共 39 張；README、CRISP-DM、圖表索引均從真實產物更新。
詳見 [完整報告](reports/CRISP_DM_REPORT.md) 與 [驗收 JSON](reports/phase5/validation.json)。

## 執行

    python phase2_experiment.py --dataset car_price --mode FAST --jobs 2
    python phase3_evaluation.py --dataset car_price --mode FAST
    python dashboard_export.py --mode FAST
    python report_preparation.py --mode FAST

正式 CLI 的 all 現在包含三題，單題執行摘要保留其他題的既有真實紀錄。
validate_project.py --reuse-training 可驗證現有三題而不重訓；
FULL 可選但本輪未執行。沒有 commit、push 或部署。
