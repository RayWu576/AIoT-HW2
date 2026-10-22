# Phase 1 regression datasets

The official assignment problems are Student Performance (Mathematics) and Appliances Energy Prediction. The startup/Boston experiments, their five selectors, saved results and dashboard remain historical references. No dashboard migration or full experiment training occurs in Phase 1.

## Verified acquisition and schema discrepancies (2026-10-21)

Student Kaggle: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

The downloaded Kaggle archive contains **only student-por.csv: 649 rows, 33 columns, comma-delimited**. It does not contain student-mat.csv. This Portuguese-language file is retained for provenance and is never substituted for Mathematics. The original Mathematics CSV was separately downloaded from UCI: https://archive.ics.uci.edu/dataset/320/student+performance (download https://archive.ics.uci.edu/static/public/320/student+performance.zip ; extract student.zip, then student-mat.csv).

**Actual Mathematics: 395 rows, 33 columns = 32 predictors + G3 (semicolon-delimited).** The requested combination of 30 predictors while retaining G1 and G2 and every original predictor is inconsistent with the authentic CSV. No authentic column has been silently removed. Two explicit profiles are available; no default is chosen until the assignment definition is resolved:

| Profile | Original inputs | Numeric | Categorical | Exclusions |
| --- | ---: | ---: | ---: | --- |
| early-30 | 30 | 13 | 17 | G3 target; G1/G2 excluded explicitly |
| prior-grades-32 | 32 | 15 | 17 | G3 target only |

G1 and G2 are historical first/second period academic grades. Including them supports prediction after these grades are available; it does not describe an early intervention model before grades are issued. Both profiles meet the 14-39 requirement. The 30-input profile is a selected use-case definition, not a claim that the original file has only 30 predictors. G3 is a numeric final grade (0-20).

All actual student columns, in source order:

```text
school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
reason, guardian, traveltime, studytime, failures, schoolsup, famsup,
paid, activities, nursery, higher, internet, romantic, famrel, freetime,
goout, Dalc, Walc, health, absences, G1, G2, G3
```

Nominal/binary predictors (OneHotEncoder): school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic. Remaining predictors are numeric, including ordinal codes treated as numeric. These definitions live only in config.py.

Energy Kaggle copy: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction
Original reference: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

The Kaggle archive contains **KAG_energydata_complete.csv**, 19,735 rows and 29 columns. It is stored byte-for-byte as energydata_complete.csv; the archive filename and SHA-256 are recorded in source.json. No columns or observations were removed from the raw file.

All actual energy columns:

```text
date, Appliances, lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9, T_out,
Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint, rv1, rv2
```

The 25 numeric inputs are **lights**, T1/RH_1 through T9/RH_9, T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint. Appliances is the numeric target, in **Wh**, date is preserved separately as temporal_metadata, and random variables rv1/rv2 are excluded. Input order is preserved. Invalid, missing, repeated or descending dates cause an error; the loader never silently sorts, shuffles, deduplicates or drops rows.

## Reacquiring files

Public Kaggle download endpoints used successfully without authentication:

- https://www.kaggle.com/api/v1/datasets/download/larsen0966/student-performance-data-set
- https://www.kaggle.com/api/v1/datasets/download/loveall/appliances-energy-prediction

Alternatively, with the Kaggle CLI configured locally (never commit credentials):

```powershell
kaggle datasets download -d larsen0966/student-performance-data-set -p datasets/student_performance --unzip
kaggle datasets download -d loveall/appliances-energy-prediction -p datasets/appliances_energy --unzip
```

Rename the energy archive CSV to energydata_complete.csv. Obtain student-mat.csv from the UCI nested archive as explained above, and place it in datasets/student_performance/. If Kaggle requires login or network access is blocked, download through its dataset page using your own account. Missing files raise FileNotFoundError with acquisition guidance. Alternate schemas/subjects are rejected rather than replaced. source.json and student-mat.source.json record actual downloaded column lists, rows and hashes.

## Safe usage and Phase 2 boundary

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from datasets import (load_student_performance, load_appliances_energy,
                      make_preprocessor, holdout_indices, make_cv, feature_counts)

student = load_student_performance(include_prior_grades=False)  # explicit early-30
full = load_student_performance(include_prior_grades=True)      # authentic 32
energy = load_appliances_energy()
train, test = holdout_indices(energy, test_size=0.2, gap=0)
cv = make_cv(energy.config, n_splits=5, gap=0)
pipeline = Pipeline([
    ('preprocessor', make_preprocessor(energy.config)),
    ('regressor', LinearRegression()),
])
# Phase 2: run CV on raw energy.X.iloc[train], with this whole pipeline.
# Fit on train only; transform/predict on test only after model selection.
```

Numeric missing values use training medians, categorical missing values use the training mode; all-missing training columns are retained. Nominal encoding uses OneHotEncoder(handle_unknown="ignore"). Numeric scaling, imputation and encoding are unfitted sklearn Pipeline/ColumnTransformer objects. Fit them separately inside every CV training fold; never preprocess the entire dataset before CV. There is no forward interpolation, backfill, global-statistic imputation or target imputation. Invalid numeric strings and infinity are rejected even in excluded numeric columns.

Original counts describe raw predictor columns. Encoded model width is learned from each training fold: call feature_counts(config, fitted_preprocessor) and get_feature_names_out() after fitting. Never assume encoded counts equal 30/32/25.

Energy holdout is the last 20% (15,788 train / 3,947 test at gap=0). CV uses expanding TimeSeriesSplit; optional gap is measured in rows (original sampling interval is 10 minutes). Student uses seeded shuffled holdout/KFold. Dates remain aligned with X/y by source row index. Apply CV only within the training portion. This prepares splitting and preprocessing for Phase 2; it does not yet integrate the historical selectors into temporal nested model selection. The historical feature_selection.py uses shuffled internal folds and must not be plugged into energy experiments unchanged. Phase 2 must also refit preprocessing inside each inner selection fold, not just an outer pipeline. The existing five method implementations remain preserved.

Validate without training full experiments:

```powershell
python -m datasets --student-profile early-30
python -m datasets --student-profile prior-grades-32
python -m unittest discover -s tests -p test_datasets.py -v
```

Tests require authentic local files and intentionally fail clearly when they are missing. Mutated temporary copies test malformed schema, missingness and unseen categories; no fabricated dataset is presented as assignment data. Only a small 240-observation baseline CV smoke test is trained. Historical browser verification tests are separate and unchanged.
