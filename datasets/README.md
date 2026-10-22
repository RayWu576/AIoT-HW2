# Official regression datasets

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

## Problem 2 - Car Price Prediction

Kaggle: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Downloaded public archive: https://www.kaggle.com/api/v1/datasets/download/hellbuoy/car-price-prediction

Verified CarPrice_Assignment.csv: **205 observations, 26 columns**, no missing values in this copy. Exclude the identifier car_ID and target price from X, leaving **24 original candidate inputs**: 14 numerical and 10 categorical. Target unit: **USD**. Identifiers must be unique/nonmissing but are never predictors. Source URL, column list, row count and SHA-256 are recorded in car_price/source.json.

Actual complete CSV schema:

```text
car_ID, symboling, CarName, fueltype, aspiration, doornumber, carbody,
drivewheel, enginelocation, wheelbase, carlength, carwidth, carheight,
curbweight, enginetype, cylindernumber, enginesize, fuelsystem,
boreratio, stroke, compressionratio, horsepower, peakrpm, citympg,
highwaympg, price
```

Numeric inputs: symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg.

Categorical inputs: CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem. Number words in door/cylinder fields remain nominal categories; no assumptions about equally spaced effects are imposed.

### CarName decision

CarName contains 147 distinct vehicle names among 205 observations. Encoding all names risks sparse model-specific indicators and overfitting. The Pipeline replaces its values with the first whitespace-separated manufacturer token, lowercases it, and applies fixed lexical corrections:

| Token | Corrected brand |
| --- | --- |
| maxda | mazda |
| porcshce | porsche |
| toyouta | toyota |
| vokswagen, vw | volkswagen |
| Nissan | nissan (lowercasing) |

This produces 22 brands in the complete authentic file. The mapping uses no target values or learned global frequencies. Brand imputation and vocabulary fitting still occur only inside each training fold. Unknown manufacturers receive all-zero one-hot indicators via handle_unknown="ignore". CarName remains one of the **24 original candidate fields**; manufacturer extraction does not inflate that count. The current 164-observation training set produces **71 encoded columns** in total; fold vocabularies and encoded counts can differ.

## Acquisition

Place files at student_performance/student-mat.csv and car_price/CarPrice_Assignment.csv under this directory. Both required files are included locally. To reacquire the car file with an authenticated Kaggle CLI:

```powershell
kaggle datasets download -d hellbuoy/car-price-prediction -p datasets/car_price --unzip
```

Or download the archive from the Kaggle page/public endpoint above. Obtain Mathematics from the UCI nested archive as documented; the specified student Kaggle copy supplies a different subject. If authentication or networking blocks acquisition, download through the source page with your own account and place the authentic CSV at the documented path. Never commit Kaggle credentials. Missing files produce a clear FileNotFoundError; wrong headers, duplicate headers, wrong row counts, invalid numerics/infinity or missing targets raise DatasetSchemaError. Missing predictors are retained for fold-local imputation.

## Pipeline and split contract

```python
from datasets import load_student_performance, load_car_price, make_preprocessor, holdout_indices, make_cv
student = load_student_performance()  # official 30 original inputs
car = load_car_price()               # 24 original candidate inputs
train, test = holdout_indices(car)   # seeded 164/41 random split
cv = make_cv(car.config, n_splits=4)  # apply only to raw training rows
preprocessor = make_preprocessor(car.config)  # unfitted ColumnTransformer
```

Numeric missing values use training-fold medians, followed by StandardScaler. Nominal features use training-fold mode imputation and OneHotEncoder(handle_unknown="ignore", sparse_output=False). Entirely missing training columns are retained. CarName manufacturer extraction is a cloneable sklearn transformer inside the categorical Pipeline. No preprocessing is fitted before GridSearchCV. Every candidate uses preprocessing - selector - LinearRegression; all learning occurs on that fold's training rows.

Both official datasets use reproducible random holdout/KFold rather than a temporal split. All index lists and random_state=0 are recorded under results/<dataset_id>/metadata.json. Original feature counts (30, 24) satisfy 14-39; feature_counts() reports encoded width separately after fitting. For method semantics, search bounds, actual execution and validation commands, see the repository README.


## Phase 2 backend scope: restored Appliances Energy

The latest Phase 2 request uses Student Performance (the same early-30 definition) and Appliances Energy. Car Price remains available for the previously completed workflow and unchanged dashboard. PHASE2_DATASETS and the independent phase2_experiment.py entry point make this scope explicit.

Appliances Energy is restored at appliances_energy/energydata_complete.csv from the authentic, hash-verified historical Kaggle download. Source: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction ; reference: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction . Its archive filename was KAG_energydata_complete.csv. There are 19,735 rows and 29 raw columns:

```text
date, Appliances, lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9, T_out,
Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint, rv1, rv2
```

Exclude target Appliances (Wh), date and rv1/rv2. Exactly 25 numeric inputs remain, including lights. Dates are separate aligned metadata; the loader rejects invalid/missing/duplicate/descending timestamps instead of sorting observations silently. load_appliances_energy() uses the same strict loader and numeric preprocessing factory. Phase 2 holdout uses first 80%/last 20% with TimeSeriesSplit at both CV levels. Student holdout and shuffled CV use random_state=42 in Phase 2. See PHASE2.md for modes, leakage protections and actual execution evidence.
