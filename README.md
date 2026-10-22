# Phase 3: evaluation and diagnostics

Saved Student Performance and Appliances Energy FAST models now have rank-aware evaluation, five-assumption diagnostics, 95% mean/individual intervals, and 13 report-ready PNGs each. No Phase 2 selection or frontend files are changed.

```powershell
python phase3_evaluation.py --mode FAST
python -m unittest discover -s tests -p "test_*.py" -v
python tests/verify_phase3_results.py
```

See [Phase 3 methodology, actual findings and statistical limitations](PHASE3.md). Figures are in `reports/figures/<dataset>/FAST/`; CSV/JSON tables and interval rows are in `reports/tables/<dataset>/FAST/`. FULL reports require existing real FULL model artifacts.

---

# Current task: Phase 2 backend

The latest Phase 2 request specifies **Student Performance (30 original inputs)** and **Appliances Energy (25 original inputs)**. The separate backend runner below implements leakage-safe nested CV, FAST/FULL modes and saved model/metric artifacts. The previously completed Car Price implementation, prior results and frontend remain preserved. The student/car dashboard described later is unchanged by Phase 2.

```powershell
python phase2_experiment.py --mode FAST --jobs 2
python phase2_experiment.py --mode FULL --jobs 2
python -m unittest discover -s tests -p "test_*.py" -v
python tests/verify_phase2_results.py
```

See [Phase 2 methods, actual timings, all six-method performance tables and artifact schema](PHASE2.md). Results are under `results/student_performance/FAST/` and `results/appliances_energy/FAST/`; FULL uses separate `FULL/` directories. Phase 2 does not use `python main.py`, which remains the earlier student/car/dashboard workflow.

---

# AIoT HW2 - Student Performance and Car Price

The two official regression problems use the same five feature selection methods and LinearRegression, with separate datasets, results and interactive dashboard state.

| Problem | Observations | Raw CSV columns | Original candidate inputs | Encoded inputs in this training split | Target |
| --- | ---: | ---: | ---: | ---: | --- |
| Student Performance (Mathematics) | 395 | 33 | 30 | 56 | G3, grade 0-20 |
| Car Price Prediction | 205 | 26 | 24 | 71 | price, USD |

Counts of encoded columns depend on the training fold. Student excludes G1/G2 for early prediction. Car excludes car_ID and derives a manufacturer category from CarName inside its Pipeline. Both original counts meet the assignment range of 14-39. See [complete schemas, acquisition and preprocessing decisions](datasets/README.md).

## Run

```powershell
pip install -r requirements.txt
python -m datasets
python main.py
# Optional: regenerate one problem, retaining the other official payload
python main.py --dataset car_price
# Rebuild presentation from the recorded CV choices, without repeating CV
python main.py --export-only
```

Open `dashboard/dist/index.html` directly, or serve it locally:

```powershell
python -m http.server 8000 --directory dashboard/dist
```

The dashboard switches only between Student Performance and Car Price. It keeps the existing five-method comparison, charts, scenarios, coefficient table, residual pagination, CSV export, theme and mobile layout. Inputs are original numeric fields and categorical dropdowns; the browser applies exported training-fitted imputation, scaling, manufacturer normalization and one-hot encoding before evaluating the model equation. No upload interface is implemented.

## Evaluation and feature selection

Both problems use 80% training / 20% held-out testing with random_state=0: student 316/79, car 164/41. Four-fold shuffled training CV chooses the feature count, LASSO alpha and recommended method. Each candidate is a complete raw-input Pipeline, so every fold independently fits missing-value handling, scaling, categories and feature selection. Test observations never enter any selection step. Model comparison is ordered by training CV RMSE; test curves are descriptive and do not choose the recommendation.

- Correlation: absolute Pearson ranking.
- Mutual Information: regression MI, with binary encoded features marked discrete.
- RFE: scaled LinearRegression elimination, with k chosen using full-Pipeline CV.
- RFECV: full-Pipeline CV over the RFE elimination path, refitting preprocessing in every fold. This replaces calling RFECV on a globally encoded matrix; its bounded search can match tuned RFE.
- LASSO: alpha chosen by training CV, nonzero coefficient support selected, followed by LinearRegression. Fixed-k dashboard curves retain the historical standardized coefficient-ranking interpretation; the automatic support model is also included.

The predeclared fixed-k search is 1-20 encoded features, with LASSO alpha fractions 0.001/0.01/0.1 of the training-fold target standard deviation. Automatic LASSO support can exceed 20 (35 in the current car run). Baseline keeps all encoded inputs. CV scores are tuning scores, not unbiased nested-CV performance estimates. The held-out test set provides final assessment. Only 205 car observations and 41 test cases are available: do not interpret one split as robust proof of generalization or tune repeatedly against its test scores.

Full one-hot sets can be rank-deficient. Predictions remain defined, but unidentifiable OLS diagnostics are shown as -. OLS p-values are exploratory after selection. Adjusted R- is absent when there are insufficient test degrees of freedom; percentage error is absent for G3=0.

## Validation

```powershell
python -m unittest discover -s tests -p test_datasets.py -v
pip install -r requirements-dev.txt
python tests/verify_official_dashboard.py
```

Browser validation uses headless Microsoft Edge via Playwright. It checks every exported model on every held-out row, compares all ten official Python pipelines on missing/unseen-category cases, and exercises switching, controls, units, CSV, scenarios, pagination and mobile overflow. Screenshots are written to `dashboard/qa/`; the report is `results/official_validation.json`.

## Files and artifacts

- `datasets/config.py`: sole feature definitions and official registry.
- `datasets/loading.py`, `preprocessing.py`, `splitting.py`: validated local data and reusable sklearn components.
- `feature_selection.py`: original implementations plus fold-local official adapters.
- `regression_experiment.py`, `main.py`: repeatable real model execution.
- `dashboard_export.py`: fitted equations and preprocessing descriptors.
- `dashboard/dist/prediction.js`, `app.js`: browser preprocessing/prediction and existing UI.
- `results/student_performance/`, `results/car_price/`: CV search records, comparison, selected features, actual test predictions, fitted pipelines and split metadata.
- `datasets/*/source.json`: downloaded-file SHA-256, schema and source. Mathematics provenance is `student-mat.source.json`.

## Historical work

Previous startup/Boston code and results remain preserved. The old main script is `historical/startup_main.py`; `housing_experiment.py`, original selectors and historical verification scripts are retained. The earlier dashboard snapshot is `historical/dashboard/dist/`. Superseded dataset artifacts and prior documentation are under `historical/` and are not part of the official registry or current dashboard. Historical browser scripts target their old snapshot/data contract; use the official validation command above for this application. Running historical exporters can overwrite dashboard data, so regenerate the official dashboard afterward if needed.

No commits or pushes are performed by these commands.
