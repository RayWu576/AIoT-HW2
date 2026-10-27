# Official regression datasets: Student Performance, Appliances Energy and Car Price

Feature definitions live in config.py and are shared by loading, preprocessing, experiments and dashboard export. Raw CSVs are authentic downloaded data; malformed versions are rejected rather than fabricated or silently substituted.

## Problem 1 - Student Performance (unchanged 30-input early prediction)

Requested Kaggle: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

Verified archive contains student-por.csv only: 649 rows, 33 comma-delimited columns. It is preserved for provenance, never used as Mathematics. The authentic student-mat.csv was obtained separately from UCI: https://archive.ics.uci.edu/dataset/320/student+performance using https://archive.ics.uci.edu/static/public/320/student+performance.zip (extract the nested student.zip).

The Mathematics file has 395 rows and 33 semicolon-delimited columns: 32 predictors plus G3. The final official definition explicitly excludes G1 and G2, leaving **30 original predictors**, with 13 numeric and 17 categorical. G1/G2 are first/second-period historical academic grades; including them would require those grades at prediction time and describes a later prediction use case. The official early model keeps them out. The full 32-input option remains available only by explicit `include_prior_grades=True` for reference.

Actual CSV columns, in source order:

```text
school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
reason, guardian, traveltime, studytime, failures, schoolsup, famsup,
paid, activities, nursery, higher, internet, romantic, famrel, freetime,
goout, Dalc, Walc, health, absences, G1, G2, G3
```

Target: numeric G3 (grade 0-20). Categorical: school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic. Other official inputs are numeric; ordinal codes retain their numeric ordering. The current training split has 56 encoded columns, distinct from its 30 original inputs.

## Problem 2 - Appliances Energy

OFFICIAL_DATASETS and PHASE2_DATASETS both contain Student Performance, Appliances Energy and Car Price. Car Price is now official Problem 3 in OFFICIAL_DATASETS; LEGACY_DATASETS only retains the earlier seed=0 runner for reference.

Appliances Energy is restored at appliances_energy/energydata_complete.csv from the authentic, hash-verified historical Kaggle download. Source: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction ; reference: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction . Its archive filename was KAG_energydata_complete.csv. There are 19,735 rows and 29 raw columns:

```text
date, Appliances, lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9, T_out,
Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint, rv1, rv2
```

Exclude target Appliances (Wh), date and rv1/rv2. Exactly 25 numeric inputs remain, including lights. Dates are separate aligned metadata; the loader rejects invalid/missing/duplicate/descending timestamps instead of sorting observations silently. load_appliances_energy() uses the same strict loader and numeric preprocessing factory. Phase 2 holdout uses first 80%/last 20% with TimeSeriesSplit at both CV levels. Student holdout and shuffled CV use random_state=42 in Phase 2. See PHASE2.md for modes, leakage protections and actual execution evidence.

## Shared current Pipeline and evaluation contract

```python
from datasets import load_student_performance, load_appliances_energy, make_preprocessor
student = load_student_performance()
energy = load_appliances_energy()
preprocessor = make_preprocessor(student.config)  # unfitted; fit inside CV
```

Numeric predictors use fold-training median imputation and StandardScaler. Student nominal predictors use mode imputation and OneHotEncoder(handle_unknown="ignore", sparse_output=False). Original feature counts are 30 and 25; encoded widths are fold-dependent (full training 56 and 25). No target enters X. Phase 2 uses seed 42 for Student's 316/79 shuffled holdout/KFold; Energy's 15,788/3,947 holdout and both CV levels preserve chronology. The static exporter reuses these fitted artifacts, exact feature ordering and recorded CV results. See ../PHASE4.md for browser inference and verification.

Acquire Energy from the linked Kaggle page, extract KAG_energydata_complete.csv and save it as datasets/appliances_energy/energydata_complete.csv. Source provenance/hashes are preserved in each dataset directory. Missing files raise explicit FileNotFoundError; schema errors reject incorrect counts, headers, invalid numeric values and dates. Files are never fabricated.


### Problem 3 — Car Price Prediction

來源：[hellbuoy Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)。
真實 CarPrice_Assignment.csv 已核對 SHA-256：205 筆、26 欄；排除 car_ID 與 price 後有 24 個原始候選輸入（14 數值、10 類別），目標 price 單位 USD。
CarName 有 147 個原始車名，Pipeline 以第一個 token 擷取廠牌並修正固定拼字，得到 22 個廠牌；
不是把原始車名直接 one-hot。未知廠牌以 handle_unknown="ignore" 處理。
完整訓練切分產生 73 個編碼欄位，與 24 個原始輸入不同。

seed=42，164 train／41 test；inner／outer shuffled KFold。
與其他題相同的 Baseline＋五選擇器，最終都為 LinearRegression。
本輪 FAST 以 training CV 選 LASSO support（38 encoded），Test RMSE 2869.8225 USD、R² 0.8957。
LASSO 僅選特徵；不以 test 較佳的 Baseline 重選。205 筆與稀疏廠牌限制推論及外推，不是當前市場估價保證。

只重跑第三題：

    python phase2_experiment.py --dataset car_price --mode FAST --jobs 2
    python phase3_evaluation.py --dataset car_price --mode FAST
    python dashboard_export.py --mode FAST
    python report_preparation.py --mode FAST

單題訓練摘要會保留其他題的真實既有紀錄，三題結果分開存放。
