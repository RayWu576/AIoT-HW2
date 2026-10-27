# AIoT HW2 專題報告：多元線性迴歸與特徵選擇
## Student Performance、Appliances Energy 與 Car Price Prediction

## 摘要

本研究以學生學業表現、家電能源消耗與汽車價格三個真實資料集，探討 Multiple Linear Regression（MLR）搭配特徵選擇的預測表現、可解釋性與統計限制。三題分別保留 30、25、24 個原始輸入，皆符合 14–39 個原始特徵的作業要求。模型包含全特徵 Baseline 與 Correlation、Mutual Information、RFE、RFECV、LASSO 五種選擇方式，最終預測器均為 sklearn LinearRegression。

本次以 {{MODE}} 模式實際執行，透過完整 Pipeline 與 training-only nested cross-validation 防止資料洩漏。依訓練 CV 選定的學生、能源及汽車模型，測試 R² 分別約為 0.0795、−0.0480 與 0.8957；這些結果反映不同預測情境的難度，而非僅以單一數字宣告成功。研究另檢查五項回歸假設、建立有明確限制的信賴／預測區間，並將真實產物整合至 DataLab 靜態互動儀表板。FULL 尚未執行，不能把 FAST 視為等價的最終統計證據。

## 一、研究動機與問題定義（Business Understanding）

多元線性迴歸可以用係數呈現輸入與目標的條件關聯，但加入更多變數不一定改善泛化，也可能產生過度擬合或共線性。本研究關注以下問題：

1. 特徵選擇能否在較少輸入下，維持合理的預測表現？
2. 訓練 CV 推薦的方法與 held-out test 表現是否一致？
3. 類別編碼、樣本數及時間相依性如何影響係數與統計推論？
4. 如何把 Python 的真實模型結果，以可驗證的方式部署到瀏覽器？

三題的應用情境不同：學生題研究不使用前期成績的早期學習支持；能源題利用同期感測資料估計耗能，並非事先取得未來感測值的負載預報；汽車題利用規格與廠牌估計資料中的價格，不是當前市場報價。這些預測關聯皆不代表因果效果。

## 二、資料來源與特性（Data Understanding）

{{OVERVIEW}}

「原始輸入」是排除目標及指定無效欄位後的候選欄位數；「完整編碼」是此訓練切分擬合後的矩陣寬度。編碼維度會隨 training fold 的類別詞彙改變，不能直接以原始特徵數代替。

### 2.1 Student Performance

正式實驗使用 Mathematics 的 student-mat.csv，共 395 筆、33 個 CSV 欄位。G3 是 0–20 的數學期末成績。原檔有 32 個非目標欄位，本研究明確排除歷史 G1／G2，形成 30 個原始輸入的 early-30 情境。G1／G2 分別為前期學業成績；加入後會改變可取得資訊與預測時點，不應混稱為同一個早期預測問題。

指定 [Kaggle 來源](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set) 的已下載 archive 實際只有 student-por.csv（649 筆）。因此使用可追溯的 [UCI Mathematics 原檔](https://archive.ics.uci.edu/dataset/320/student+performance)，並保留來源與 SHA-256；未將葡萄牙文科檔改名冒充數學科。「葡萄牙中學生」描述資料來源背景，不等於模型使用的是 Portuguese 科目檔。

### 2.2 Appliances Energy

[能源 Kaggle 副本](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction) 對應 [UCI 原始資料](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)，共 19,735 筆、29 欄，目標 Appliances 單位 Wh。排除 rv1、rv2 與目標；date 作時間 metadata，保留 lights，最後得到 25 個原始數值輸入。觀測順序保留，不使用未來觀測補過去值。

### 2.3 Car Price

[Car Price Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) 的真實 CarPrice_Assignment.csv 有 205 筆、26 欄。排除 car_ID 與目標 price 後，保留 24 個原始候選輸入，目標單位 USD。

CarName 有 147 個不同車名，先以確定性規則擷取第一個 token 作廠牌，修正固定拼字後得到 22 個廠牌，避免直接對高基數完整車名編碼。本次訓練切分的完整編碼寬度為 73。詳細 schema、欄位定義及取得方式見 [datasets/README.md](datasets/README.md)。

## 三、資料清理與前處理（Data Preparation）

loader 逐欄檢查 schema、筆數、目標數值與無限值。能源另檢查時間戳完整、唯一且遞增，不靜默排序；缺檔或版本不符會明確報錯，不產生替代資料。

數值欄位在每個 training fold 內以中位數補值，再使用 StandardScaler。名目類別以該 fold 眾數補值，接著使用 OneHotEncoder(handle_unknown="ignore")；未知類別對應全零 indicator。汽車的廠牌擷取也保留在 Pipeline 中。雖然目前來源檔沒有需要補值的原始輸入，仍保留可重用的 imputer。

| 題目 | Holdout | 訓練期 CV | 預測情境 |
| --- | --- | --- | --- |
| 學生 | seed=42，316／79 shuffled 80/20 | shuffled KFold | 不使用 G1／G2 的 early-30 |
| 能源 | 前 15,788／後 3,947 筆 | expanding TimeSeriesSplit，gap=0 | 按時間評估同期耗能估計 |
| 汽車 | seed=42，164／41 shuffled 80/20 | shuffled KFold | 規格與廠牌的價格估計 |

補值、縮放、編碼、排名與選擇都在完整 Pipeline 的 training fold 內擬合，不能先對全資料 fit 再 CV。測試集不參與特徵數、alpha 或方法選擇。

## 四、多元線性迴歸與特徵選擇（Modeling）

模型形式為：

$$
\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_j z_j
$$

其中 $z_j$ 是訓練流程產生的選定編碼欄位，$p$ 是選定矩陣的欄數。使用最小平方法估計係數；若矩陣有 aliasing，係數可能不是逐欄唯一可識別，必須另檢查有效 rank。

| 方法 | 本專案實作 |
| --- | --- |
| Baseline | 保留全部編碼欄位，擬合 LinearRegression |
| Correlation | 依 training 絕對 Pearson correlation 排名，再選 k |
| Mutual Information | 依互資訊排名，二元 indicators 視為 discrete，固定隨機種子 |
| RFE | 沿標準化線性係數遞迴消除，快取完整路徑 |
| RFECV | 在完整 raw-input Pipeline CV 中選 RFE 路徑上的 k；可能與 tuned RFE 相同 |
| LASSO | 以非零 support 選特徵，再重新擬合 LinearRegression；空 support 有明示的一欄 fallback |

**LASSO 是選擇器，不是本專案最終預測器；VIF 是事後診斷，不是目前選擇流程中的刪欄規則。**

內層 CV 調整 k／alpha，外層評估完整選擇流程；全訓練部分再依 inner CV 固定正式推薦方法，最後才評估 test。Inner CV RMSE 是調參統計，per-method nested CV 與包含方法選擇的 workflow nested CV 另有輸出，不能混為同一分數。

| 設定 | FAST | FULL |
| --- | --- | --- |
| inner／outer folds | 2／2 | 4／5 |
| k | 1、5、10、20、all，依可行寬度 | 全部可行 k |
| LASSO alpha fraction | 0.01、0.1 | 0.0001、0.001、0.01、0.1、1 |
| 本次狀態 | 已實際執行 | 尚未執行 |

最多兩個 joblib workers，每個 native thread=1，outer loops 不再平行。cache 僅重用同一 training fold 的相同計算，cold 與 cache-hit runtime 不應直接作效能比較。

## 五、模型評估與結果（Evaluation）

### 5.1 評估指標

MAE 為平均絕對誤差；MSE 為平均平方誤差；RMSE 為 MSE 平方根，保留目標單位。$R^2=1-SSE/SST$ 可為負，表示相對該評估樣本平均值的描述性參考沒有改善。

Adjusted R² 使用含截距的有效 training affine rank $r$：

$$
R^2_{\mathrm{adj}}=1-\frac{SSE/(n-r)}{SST/(n-1)}
$$

若自由度不足或目標為常數，輸出空值。Test adjusted R² 只是描述性的複雜度懲罰，不是在 test 重新擬合模型的推論。CV std 是 fold RMSE 標準差，不是平均數標準誤或信賴區間。

### 5.2 Baseline 與五種選擇器：實際 {{MODE}} 結果

{{RESULTS}}

推薦方法只依 training CV。即使 Baseline 的 test 結果較好，也不據此重選方法。三題 RMSE 分別是 grade、Wh、USD，**不能直接比較數字大小判斷哪題模型較佳**。

### 5.3 正式推薦模型的 Train／Test 指標

{{SELECTED_METRICS}}

學生 Correlation 的 Test RMSE 約 4.3446 分、R² 約 0.0795，顯示不使用歷史成績時，背景與學習資訊的預測訊號有限。能源 Mutual Information 的 Test R² 約 −0.0480，保留其不佳結果，不以已觀察的 test 重調方法。汽車 LASSO support 的 Test RMSE 約 2869.8225 USD、R² 約 0.8957，但只有 41 筆 test，不能將此分數視為對所有市場或車型的保證。

### 5.4 五項回歸假設與殘差診斷

線性以 residual-vs-fitted 與分箱均值檢查；獨立性使用 DW／ACF，能源另有 Ljung–Box；等變異使用 Koenker–Breusch–Pagan；常態性使用直方圖、Q-Q、Shapiro-Wilk 或大樣本 Jarque–Bera；共線性則檢查 VIF、rank 與 condition number。

下表來自各題 training-CV 選定模型的實際診斷。未觸發篩檢不等於假設已證明成立，顯著檢定也不是因果證據。

{{DIAGNOSTICS}}

學生與汽車原始列序不是時間，DW 只能作描述，仍可能有學校或廠牌群組相依。能源殘差的明顯時間相依與異質變異限制傳統 OLS 推論。大樣本常態檢定對微小偏離也敏感，因此須與圖形共同判讀。

### 5.5 信賴區間與預測區間

95% mean CI 描述條件平均 $E[Y|X]$ 的不確定性；individual PI 額外納入個別誤差變異，通常更寬。使用選定 sklearn 模型的相同設計，透過 centered SVD 的等價滿秩基底，以 statsmodels OLS 計算 t 區間。學生／汽車另有 HC3，能源有 HAC（24 lag）漸近 mean CI；robust mean CI 不是 robust individual PI。

完整 one-hot 加截距可能產生 aliasing；不可識別係數不提供假的個別 SE／CI，超出 training affine span 也不提供可識別區間。稀疏類別若產生 leverage=1，HC3 可能無法估計。區間沒有校正選模不確定性，未證明 nominal coverage，也不任意裁切負 Wh／USD 或超過成績範圍的上下界。

汽車推薦模型的 training rank=37，而 test 只有 41 筆，test adjusted R² 的分母僅 4，解釋上尤其要保守。完整數值見 [三題 CRISP-DM 報告](reports/CRISP_DM_REPORT.md) 及 reports/tables/。

### 5.6 實際值與預測值圖

![學生測試集：實際與預測](reports/figures/student_performance/FAST/heldout_actual_vs_predicted.png)

圖一：學生模型的預測變動有限，無法充分區分全部期末成績；圖中離對角線較遠的點代表誤差較大。

![能源測試集：實際與預測](reports/figures/appliances_energy/FAST/heldout_actual_vs_predicted.png)

圖二：能源高耗能觀測常位於理想對角線下方，顯示模型對部分耗能峰值低估；此現象應與負 Test R²、殘差右尾及時間相依一起解讀。

![汽車測試集：實際與預測](reports/figures/car_price/FAST/heldout_actual_vs_predicted.png)

圖三：汽車多數點較接近對角線，但仍有明顯偏離。圖標題的 LASSO 指特徵選擇方法，最終 predictor 仍為 LinearRegression。

全部 **39 張 300 dpi PNG** 的圖號、檔名與用途見 [圖表索引](reports/FIGURE_INDEX.md)。

### 5.7 實際執行時間與驗證

{{TIMING}}

Pipeline 包含載入、nested／final CV、評估與模型／CSV 匯出，不含 Python 匯入或程序啟閉。診斷包含各題 13 張圖。上述合計不含共用 dashboard export、browser QA 與文件產生，也不假稱三題來自同一次 cold run。

{{VALIDATION}}

## 六、互動儀表板與部署（Deployment）

沿用既有 DataLab，提供三題／六模型／k 切換、CV 曲線、係數與統計、殘差排序／分頁、數值與類別表單、95% CI／PI、外插警示、情境比較、CSV、深色模式與響應式版面。五項假設以中文診斷卡片呈現，完整統計可展開；切題清空情境並更新單位及表單。

瀏覽器使用 Python 匯出的 training-fitted imputation、scaling、one-hot、manufacturer aliases 與精確係數順序，透過自動測試核對實際預測及區間。靜態架構不需要後端 API 或前端 build，也不提供任意資料上傳。

本機開啟：

    python -m http.server 8000 --directory dashboard/dist

瀏覽 http://localhost:8000 ，亦可直接開 dashboard/dist/index.html。相對資產支援 project-subpath hosting。GitHub Pages 使用既有 .github/workflows/static.yml 上傳 dashboard/dist；設定 Source 為 GitHub Actions，由授權的 main push 或 workflow_dispatch 發布，再檢查 Actions 與公開頁。此文件不將本機通過冒稱為最新公開部署已驗證。

{{NOTEBOOKLM}}

## 七、討論、限制與後續研究

三題展現不同限制：學生少樣本及 early-30 的弱預測訊號；能源的自相關、異質變異與時段轉移；汽車的稀疏類別、共線性與小型 test。原始特徵數、編碼欄數及有效 rank 必須分開報告。

本次只執行 FAST。未來可先規劃 training-only FULL、時間 gap 敏感度、group-aware 驗證，或預先登記的 RFE＋VIF、Ridge、非線性交互作用與其他模型比較。這些屬後續研究，不是已執行成果；已觀察的 test 不應反覆用來調整設計。

## 八、結論

本專案完成三題可追溯的 MLR 與五種特徵選擇實作，將訓練、CV、test、假設診斷與區間分開報告，並以同一組真實產物支援靜態互動介面。實驗沒有證明「特徵越少一定越好」或「所有假設都成立」；預測表現與係數解釋應在資料、切分及統計限制下評估。NotebookLM 摘要提供研究對照，不能取代本專案實際執行的數值證據。

## 九、專案結構、環境與重現方式

    datasets/                       schema、loader、前處理、split 與真實 CSV
    feature_selection.py            五種特徵選擇與快取排名
    regression_pipeline.py          training-only nested CV、FAST／FULL
    phase2_experiment.py             訓練、模型與 CSV／JSON
    regression_diagnostics.py        有效 rank、VIF、假設與 CI／PI
    phase3_evaluation.py             三題指標、診斷及 39 張 PNG
    dashboard_integration.py         靜態精確推論 descriptor
    dashboard/dist/                 既有 DataLab 與真實 data bundle
    docs/README.template.md          本報告模板
    docs/NOTEBOOKLM_RESEARCH.md      使用者提供的摘要與研究對照
    report_preparation.py            由真實產物產生 README／報告
    validate_project.py              端到端驗收
    tests/                          unit、integration、browser QA
    results/<dataset>/FAST/          模型、CV、預測與 runtime
    reports/                        完整報告、PNG、表格及驗收紀錄

建議 Python 3.12。本次實際套件版本見 [environment.json](reports/phase5/environment.json)；requirements-lock.txt 是已驗證的直接套件精確版本，不冒稱跨平台完整 transitive lock。Browser QA 預設使用已安裝的 Microsoft Edge。

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt

執行三題 FAST：

    python phase2_experiment.py --mode FAST --jobs 2
    python phase3_evaluation.py --mode FAST
    python dashboard_export.py --mode FAST
    python report_preparation.py --mode FAST

只重跑 Problem 3 時，在前兩個指令加入 --dataset car_price。單題摘要保留其他題的真實既有紀錄。FULL 可將 --mode FAST 改為 --mode FULL，但須先產生真實 FULL artifacts，不能以 FAST 假冒。

驗證：

    python validate_project.py --reuse-training
    python -m unittest discover -s tests -p "test_*.py" -v
    python tests/verify_official_dashboard.py

validate_project.py 不加 --reuse-training 時會重訓三題。原始缺檔時請按 [資料說明](datasets/README.md) 取得真實 CSV；不生成替代資料。模型 joblib 僅載入可信產物，換套件版本應重訓。

ChatGPT 真實對話 PDF、NotebookLM 來源引用／使用日期、姓名／學號／課程資訊及最終 PDF 排版仍待補；已收到的 NotebookLM 摘要不會再標成未填。詳見 [AI Assistance 紀錄](reports/AI_ASSISTANCE.md)。

## 十、參考資料

1. [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)。
2. [指定 Student Kaggle archive](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)。
3. [UCI Appliances Energy](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)。
4. [Appliances Kaggle 副本](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)。
5. [Car Price Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)。
6. [scikit-learn：資料洩漏與 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)。
7. [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)。
8. [GitHub Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
9. NotebookLM 研究摘要：使用者提供；原始來源與引用位置待補。
10. [GitHub 原始碼](https://github.com/RayWu576/AIoT-HW2)。
