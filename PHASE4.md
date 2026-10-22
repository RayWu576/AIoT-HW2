# Phase 4 - Existing DataLab integration

Exactly two official problems use the saved Phase 2/3 FAST artifacts. The existing dashboard layout, dropdowns, feature controls, chart comparison, prediction forms, scenario comparison, coefficient/statistics tables, residual sorting/pagination, CSV export, dark mode and responsive interface remain. Added overview, six-model comparison including Baseline, train/test metric scope, validation-only curves, five diagnostics, source/field definitions and statistically justified intervals.

## Data and inference contract

`dashboard_integration.py` validates the raw source and Phase 3 source-run SHA256, loads saved final pipelines, and exports `dashboard/dist/data.json` plus equivalent `data.js` (schema 5). It does not repeat CV or change previous fitted models. Alternate recorded candidates reuse the training-fitted transformer/ranking and fit LinearRegression only on training rows; LASSO alternate alpha fits its support on training rows. Only recorded CV candidates are offered; duplicate final LASSO support counts expose the candidate with best recorded training CV. These are exploratory candidates, never test-selected recommendations. Private candidate joblib files live in `results/dashboard/FAST/<dataset>/models/` for reproducibility and QA; they are not downloaded by the browser.

The static descriptor contains exact median imputation, numeric StandardScaler constants, categorical most-frequent fills, fitted one-hot categories, transformed names and coefficient order. Unknown categories produce all-zero indicators exactly as sklearn handle_unknown=ignore. No approximation or invented coefficient is used. UI validation rejects missing/invalid values; direct inference still verifies missing-value preprocessing. Physical bounds and observed training bounds are distinct. Out-of-range values or high leverage warn rather than silently clamp. No output clipping is applied.

Prediction intervals use the selected model's training-only rank-reduced SVD design and statsmodels OLS covariance, error variance, residual df and t critical value. Individual 95% PI is separate from conditional-mean CI. HC3 (student) or HAC with 24 lags (energy) supplies an asymptotic robust mean CI only. Outside the training affine span, intervals are explicitly unavailable. One-hot aliasing suppresses individual coefficient inference where unidentified. Classical PI assumes iid homoscedastic normal errors, may have poor coverage under dependence/heteroscedasticity, and does not account for selection uncertainty. These are approximate post-selection pointwise intervals, not guarantees or causal effects.

The 30-input student early-prediction profile excludes historical grades G1/G2. Including them describes a later prediction scenario. Energy preserves chronology in both splits/CV; contemporaneous sensors do not establish an ahead-of-time forecasting capability. FAST two-fold evidence remains distinct from FULL.

## Diagnostics and UI state

Actual test predictions feed metrics, scatter, residual table and CSV. Full training observations feed assumption tests; training scatter is deterministically sampled (including extremes), Q-Q uses up to 201 quantiles, and histogram densities distinguish train from test. CV curves use recorded inner-fold RMSE and reconstructed validation R2, never held-out selection. Final-method nested CV is displayed separately. Original, represented-original, selected-encoded and design-rank counts are distinct.

Switching dataset rebuilds every input/control/chart/unit/table and clears scenarios, residual page, sort and metric scope. Models and k values are keyed by dataset, method and recorded candidate. CSV includes dataset/run/units, train/test/CV metrics, inputs, intervals, coefficients and every test residual. Its filename identifies active dataset and model. Theme persists independently.

## Verification and local use

Run `python -m unittest discover -s tests -p "test_*.py" -v` and `python tests/verify_official_dashboard.py` (install Playwright; installed Edge required). Browser QA checks all candidate held-out predictions against actual candidate sklearn pipelines, missing/unseen categorical cases, official model intervals against Phase 3 statsmodels calculations, both problem switches, six methods and every k, charts, metric scope, invalid inputs, scenario clearing, CSV, dark-mode persistence, direct-file and nested-path HTTP loading, missing-bundle empty state and 1440/768/390/320 px layouts. Actual machine-readable results are in `reports/phase4_validation.json`; desktop/mobile captures are under reports/phase4_*.png. Historical verify_phase2/3_results scripts deliberately include old frontend-byte fingerprints and are not the Phase 4 frontend acceptance test.

Regenerate with `python dashboard_export.py --mode FAST` (or `python main.py --mode FAST`). Serve with `python -m http.server 8000 --directory dashboard/dist`; open http://localhost:8000. All runtime assets use relative links. GitHub Pages uses the existing workflow and dashboard/dist artifact. Public deployment was not triggered or verified. No commit/push is authorized for this phase.

## Recorded verification

47 Python unit tests passed (43 existing plus 4 Phase 4 artifact-contract tests). Browser validation passed for 46 real candidate pipelines, 92,598 held-out predictions, 138 missing/unknown-category cases and 24,156 official-model interval rows. Maximum absolute differences were 3.35e-9 (predictions) and 3.44e-9 (intervals), below the declared tolerances. Six methods, all available k, chart/metric switching, CSV, scenario isolation, theme persistence, empty state and four viewport widths passed with zero JavaScript runtime errors. See the JSON report for final asset hashes and runtime. Screenshots were visually inspected at readable crops for both layouts.

Historical exporters now write only to results/historical_student_car or results/historical_boston_startups; invoking an old experiment cannot replace active official dashboard data. Historical documentation may mention older output commands; it is explicitly not the current interface contract.
