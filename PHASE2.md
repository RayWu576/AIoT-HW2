> Historical Phase 2 record. The current official dashboard now uses Student/Energy; see README.md and PHASE5.md. Old frontend-byte preservation applies only to historical replay.

# Phase 2: leakage-safe Multiple Linear Regression

This backend phase follows the latest request: Student Performance (30 original inputs) and Appliances Energy (25 original inputs including lights). The existing Car Price implementation and dashboard remain intact. No file under dashboard/dist is written by the Phase 2 runner. Dataset IDs are explicitly separated: PHASE2_DATASETS drives this runner; OFFICIAL_DATASETS still describes the preserved student/car dashboard.

## Run

```powershell
python phase2_experiment.py --mode FAST --jobs 2
python phase2_experiment.py --mode FULL --jobs 2
python phase2_experiment.py --dataset appliances_energy --mode FAST --jobs 1
# Optional temporal embargo measured in rows (energy only)
python phase2_experiment.py --dataset appliances_energy --mode FULL --gap 6
# Disable persistent fitting cache
python phase2_experiment.py --mode FAST --no-cache
```

Both datasets use final LinearRegression for Baseline, Correlation, Mutual Information, RFE, RFECV and LASSO. LASSO only selects nonzero features; an unregularized LinearRegression is refitted on that support. If LASSO selects none, the largest absolute coefficient (stable first-column tie) supplies one predictor.

## Data and preprocessing

- Student: authentic student-mat.csv, 395 rows, 33 raw columns; official 30 predictors exclude G3/G1/G2. G1/G2 are historical grades and remain excluded from this early prediction problem. Fixed random_state=42, shuffled 80/20 holdout: 316 train / 79 test. Inner and outer validation use seeded shuffled KFold.
- Energy: authentic Kaggle copy restored byte-for-byte from the historical archive, 19,735 rows and 29 columns. Exclude Appliances (target in Wh), rv1, rv2 and date. Keep all 25 numeric inputs, including lights. Date remains separate ordered metadata. First 15,788 rows train; last 3,947 test. Both CV levels use expanding TimeSeriesSplit, never shuffled. Optional gap applies at the holdout boundary and both CV levels; default gap=0.
- Existing ColumnTransformer/Pipeline preprocessing is reused. Numeric median imputation and StandardScaler, plus student categorical mode imputation and OneHotEncoder(handle_unknown="ignore"), are fitted only on each fold's training rows. No global preprocessing is fitted before validation. Missing targets, infinities, malformed schema and unordered dates remain errors.
- Original input count and training-fold encoded width are distinct. The current full student training vocabulary produces 56 encoded columns; energy has 25. Selected encoded names map explicitly to original fields, preserving actual coefficient order.

Energy source: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction ; reference: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction . The archive file KAG_energydata_complete.csv is stored as energydata_complete.csv. SHA-256 is recorded in datasets/appliances_energy/source.json and each run.json. Student acquisition discrepancies remain documented in datasets/README.md.

## Nested evaluation and model selection

fit_training() accepts only raw training X/y, never the held-out test set.

1. Each outer training partition gets its own full-Pipeline inner GridSearchCV. Imputation, scaling, category fitting and ranking are fitted independently inside each inner fold.
2. For each method, the best inner-CV pipeline is evaluated on the outer validation partition. Per-method Nested CV RMSE/std summarize these outer scores.
3. Each outer fold also chooses its method strictly from inner scores; its validation score contributes to the workflow nested estimate of the entire selection rule.
4. After nested evaluation, all methods are tuned again using only the full training portion. The final method minimizes this final inner best CV RMSE, with a stable method-order tie break. This is the same selection rule evaluated by the nested workflow. Saved best_cv_rmse/cv_std are tuning statistics, distinct from per-method nested scores and workflow nested scores.
5. Only after this selection is fixed does evaluate_holdout() receive X_test/y_test. It cannot fit or change a model. Train/test metrics are MAE, MSE, RMSE, R2 and adjusted R2 (null if degrees of freedom are insufficient).

Held-out test performance may favor another method; it never changes the chosen model. Do not use these test results to tune the next experiment repeatedly.

RFE computes the complete elimination path once per training fold. RFECV searches k on this RFE path using CV of the entire raw-input Pipeline, so preprocessing is refitted at the correct level. This implements recursive elimination with CV without feeding a globally encoded matrix to sklearn RFECV. Tuned RFE and RFECV can be identical; their paths are cached/shared rather than fitted redundantly. Existing historical selector classes are preserved.

## FAST versus FULL

| Setting | FAST | FULL |
| --- | --- | --- |
| Outer folds | 2 | 5 |
| Inner folds | 2 | 4 |
| k grid, non-LASSO selectors | 1, 5, 10, 20 below encoded width, plus all | Every integer from 1 through training-partition encoded-width bound |
| LASSO alpha fractions of fold target std | 0.01, 0.1 | 0.0001, 0.001, 0.01, 0.1, 1.0 |
| Purpose | Development and debugging | More thorough final evaluation |

FULL grid bounds use raw categories only in that search's training partition. Each validation fold still learns its own encoder; k is capped to its actual encoded width. No fitted vocabulary or statistics from the partition-wide bound are passed into the fold. k='all' is fold-adaptive.

FAST and FULL do not provide equivalent statistical evidence. FAST's two outer folds have high uncertainty. FULL is implemented and its grid/strategy is tested, but FULL experiments have not been executed in this task. More folds do not guarantee better held-out results.

## Four-core resource control

Inner search supports jobs=1 or jobs=2 only (default 2). Outer folds/methods run sequentially; BLAS/OpenMP threads are limited to one in training and each worker. No nested pools or unbounded n_jobs=-1 are used. Pipeline joblib caching reuses preprocessing and expensive ranking for identical fold training X/y across candidate k values. The k-only prefix selection and final LinearRegression remain cheap.

Cache paths are .cache/phase2/<dataset>/<mode>/<source-and-package-hash>/ and ignored by git. Code and sklearn/numpy/joblib changes invalidate the namespace, avoiding stale fitted custom estimators. Saved pipelines have memory=None and never rely on the absolute cache path. The two cold namespaces from validation use about 185 MB total on disk; cached arrays are small relative to 28 GB RAM. --no-cache is available for comparison. Cached and uncached fit timings differ; reported timings include any namespace hits and exclude Python import/startup/teardown.

## Artifacts

Separate output directories are results/student_performance/FAST/ and results/appliances_energy/FAST/; FULL writes corresponding FULL/ subdirectories. Prior result files remain untouched.

- run.json and dashboard_payload.json: structured schema, counts, split indices, source hash, mode/seed/versions, CV scores, train/test metrics, runtimes, selected features, coefficients and intercepts.
- model_comparison.csv: all six methods and CV/train/test metrics, ranked by final training inner CV.
- selected_features.json: encoded names, original names and their mapping.
- coefficients.csv: intercept plus coefficients in model input order. Numeric coefficients refer to standardized inputs; category coefficients refer to training one-hot indicators.
- train_predictions.csv and test_predictions.csv: all six real model predictions, actual targets, source row indices, and energy timestamps.
- models/*.joblib: six independently loadable fitted sklearn Pipelines; best_model.joblib and best_model.json identify the training-selected winner.
- cv_searches/*.csv: final training-only search grids and fold scores; outer_folds.json records nested scores, selected parameters/counts and source row boundaries.
- results/phase2_FAST_summary.json and phase2_FAST_initial_summary.json: the two actual FAST executions; phase2_validation.json records integration validation.

## Actual FAST results

The most recent run used jobs=2 and new cache namespaces (no preexisting namespace). Per-method timings depend on shared cache order, so they are not fair standalone algorithm benchmarks.

### student_performance

Runtime: 16.94 seconds. Training-selected best: Correlation. Workflow nested CV RMSE: 4.2745 +/- 0.0400.

| Method | Selected encoded | Best inner CV RMSE | Inner std | Nested CV RMSE | Nested std | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 56 | 4.7876 | 0.0924 | 4.7876 | 0.0924 | 4.1957 | 0.1415 |
| Correlation | 1 | 4.2798 | 0.0346 | 4.2745 | 0.0400 | 4.3446 | 0.0795 |
| Mutual Information | 10 | 4.3164 | 0.0908 | 4.4027 | 0.1771 | 4.3185 | 0.0905 |
| RFE | 1 | 4.2798 | 0.0346 | 4.2798 | 0.0346 | 4.3446 | 0.0795 |
| RFECV | 1 | 4.2798 | 0.0346 | 4.2798 | 0.0346 | 4.3446 | 0.0795 |
| LASSO | 3 | 4.6109 | 0.0384 | 4.6109 | 0.0384 | 4.3107 | 0.0938 |

### appliances_energy

Runtime: 21.96 seconds. Training-selected best: Mutual Information. Workflow nested CV RMSE: 98.9886 +/- 0.5984.

| Method | Selected encoded | Best inner CV RMSE | Inner std | Nested CV RMSE | Nested std | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 25 | 100.4710 | 2.8834 | 100.4710 | 2.8834 | 86.7329 | 0.0924 |
| Correlation | 5 | 98.4382 | 0.0480 | 98.9886 | 0.5984 | 94.7469 | -0.0831 |
| Mutual Information | 10 | 96.0286 | 1.0752 | 96.1827 | 1.2294 | 93.1965 | -0.0480 |
| RFE | 10 | 98.2936 | 1.6599 | 100.0206 | 0.0670 | 87.0556 | 0.0856 |
| RFECV | 10 | 98.2936 | 1.6599 | 100.0206 | 0.0670 | 87.0556 | 0.0856 |
| LASSO | 4 | 99.2578 | 0.6517 | 99.2578 | 0.6517 | 94.7312 | -0.0828 |

Latest total inside CLI main: 38.94 seconds. Initial total before cache namespace hardening: 32.61 seconds. Both runs reproduce selected methods/counts and performance metrics. Run timings exclude Python imports/process teardown; per-dataset timings cover loading, training, evaluation and model/CSV export, excluding final JSON serialization.

Energy's training-selected MI pipeline has negative test R2. Baseline has better held-out metrics in both problems in this FAST run, but the held-out set was not used to override the training-selected model. These observations motivate stronger training-only validation, not test-driven model selection.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python tests/verify_phase2_results.py
```

Unit tests cover all six methods, final LinearRegression, authentic schemas, seed=42 splitting, both temporal CV levels, missing/unknown categories, explicit fold-level preprocessing/selection spies, cache reuse without cross-fold leakage, cached RFE equivalence, coefficient/name order, finite predictions, holdout evaluation that cannot refit, reproducibility, and the FULL search configuration. The integration test reloads all 12 full-size FAST models, verifies all stored train/test predictions and metrics, checks both actual runs reproduce, and compares every frontend file hash with the pre-Phase-2 snapshot. No frontend/browser migration is part of this phase.

## Recommended Phase 3 work

Run FULL to assess selection stability with more training folds and a complete k grid. Investigate residuals, influential observations, multicollinearity and variation of selected original/encoded features across folds. For energy, assess temporal distribution shift and choose any embargo/forecast horizon from domain requirements before tuning. Any lag or rolling features must use past observations only and keep the test period untouched. For student, report the limited predictive information of the early-30 profile and preserve G1/G2 exclusion. Keep the final estimator LinearRegression and distinguish development tuning scores, nested estimates and final holdout performance.
