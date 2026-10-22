# Phase 3: model evaluation, diagnostics and report figures

## Reproduce without repeating Phase 2 training

```powershell
python phase3_evaluation.py --mode FAST
python phase3_evaluation.py --dataset appliances_energy --mode FAST
python -m unittest discover -s tests -p "test_*.py" -v
python tests/verify_phase2_results.py
python tests/verify_phase3_results.py
```

FULL reports can be generated with --mode FULL only after real Phase 2 FULL artifacts exist. This task evaluated saved FAST artifacts. It does not invent FULL results, repeat training searches, change selected models, or modify frontend files. Statsmodels auxiliary OLS fits below are diagnostic equivalents of the fixed selected regressions, not new feature/method selection.

## Output layout

- reports/figures/<dataset_id>/FAST/: 13 descriptive 300-dpi PNGs for each dataset (26 total).
- reports/tables/<dataset_id>/FAST/evaluation_metrics.csv: separate train/test rows for all six final fitted pipelines, with rank-aware adjusted R2.
- cross_validation_metrics.csv: inner tuning and outer nested CV RMSE/std, taken from Phase 2 records; kept separate from training/test metrics.
- training_cv_feature_count_curves.csv: real recorded CV RMSE/std plus algebraically reconstructed fold validation R2, with explicit count semantics.
- <method>_vif.csv and <method>_encoded_correlation.csv: every final model's transformed training inputs. VIF excludes the intercept; constant/perfectly aliased columns have explicit statuses rather than misleading zeros.
- <method>_test_intervals.csv: source row, actual target, sklearn prediction, estimability mask, classical 95% conditional-mean CI, classical 95% individual PI, and HC3/HAC approximate mean CI. Energy retains date ordering.
- coefficient_identifiability.csv: saved sklearn intercept/coefficients and per-coefficient identifiability flags.
- assumption_summary.csv, interval_summary.csv and diagnostics.json: all six models' statistics, interpretations, limitations, descriptive PI coverage and source model/run hashes.
- reports/<dataset_id>_FAST_diagnostics.md: report-ready selected-model narrative and all-method held-out table.
- reports/phase3_FAST_manifest.json and phase3_validation.json: run configuration and generated-artifact validation.

No dataset overwrites the other's outputs. FAST and FULL use separate subdirectories. Plots focus on the training-selected model; evaluation/diagnostics/interval tables cover all 12 saved final models.

## Figures (each dataset)

1. target_distribution_train_test.png
2. training_original_numeric_correlation_heatmap.png
3. feature_count_vs_inner_cv_rmse.png
4. feature_count_vs_inner_validation_r2.png
5. baseline_vs_feature_selection_rmse.png
6. heldout_actual_vs_predicted.png
7. training_residuals_vs_fitted.png
8. residual_histogram_train_test.png
9. training_residual_normal_qq.png
10. selected_feature_coefficients.png
11. selected_encoded_correlation_heatmap.png
12. training_residual_autocorrelation.png
13. heldout_mean_confidence_and_individual_prediction_intervals.png

Units are grade (0-20) for G3 and Wh for Appliances. Original numeric correlation plots include the target but exclude nominal fields; transformed model correlations are also exported. Coefficients are predictive associations per encoded input unit, not causal effects. Numeric inputs are standardized; categorical inputs are 0/1 indicators. The energy interval figure shows the first 288 chronological test observations for readability; all test rows remain in CSV.

## Rank-aware evaluation

Let Z be the selected transformed feature matrix, A=[1,Z], and r its effective affine rank including the intercept. Training residual df is n_train-r, not n_train minus the original input count or the nominal encoded width. We compute

```
R2 = 1 - SSE/SST
adjusted_R2 = 1 - (SSE/(n-r)) / (SST/(n-1))
```

when n>r, n>1 and SST>0. Otherwise the adjusted value is absent; constant-target R2 is also absent. Evaluation tables record both training fitted rank and evaluation-matrix rank. Held-out adjusted R2 uses r_train as a descriptive fitted-complexity penalty; it is not a test-set OLS refit or an unbiased inferential statistic. The effective training rank can differ from the evaluation rank.

Phase 3 supersedes Phase 2's nominal-count adjusted R2 only. MAE, MSE, RMSE, R2 and all predictions are verified unchanged. In particular, the student Baseline has 56 encoded inputs but rank 40/57 including intercept; the MI model has rank 10/11. Treating all 57/11 columns as independent would give incorrect residual df.

## Classical assumptions: evidence, not automatic acceptance

- Linearity: training residual scatter and grouped/bin means. Distinct fitted values are grouped directly when there are at most ten levels, avoiding quantile bins hiding rare student failure levels. Bin-mean span >0.5 residual SD is an explicitly declared visual screening heuristic, not a significance test; a smaller span does not establish linearity.
- Independence: energy residuals retain chronological order for DW, ACF and Ljung-Box at lags 1/6/12/24 (10-minute samples). ACF bands are approximate white-noise reference bands. Ljung-Box uses model_df=0 as an exploratory post-regression diagnostic, not an ARMA-adjusted formal guarantee. Student residuals use source row order only: no temporal ordering is justified, so DW cannot establish independence or exclude school/group dependence.
- Homoscedasticity: Koenker-Breusch-Pagan on an intercept-containing full-rank basis spanning the selected design, with auxiliary df r-1. Degenerate residuals or inadequate df yield an unavailable test. Nonrejection does not prove constant variance; temporal dependence weakens nominal calibration. Statsmodels explicitly notes the full-rank requirement for the test's df: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html .
- Normality: student uses Shapiro-Wilk on all 316 training residuals; energy uses large-sample Jarque-Bera on all 15,788. Shapiro p-values are not used above 5,000 observations because of the documented limitation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html . Histograms, Q-Q, skew and kurtosis supplement tests. Reported p=0 means numerical underflow, not a literal zero probability. Fitted/selected and dependent residuals weaken iid test assumptions; large n also makes tests highly sensitive.
- Multicollinearity: rank, raw/effective condition numbers, VIF and correlation matrices. Full one-hot sets can be exactly aliased even while predictions remain defined. VIF >10 is only a screening convention. Individual coefficients participating in null directions are not uniquely identified. Constant predictors have undefined VIF; perfect collinearity has infinite VIF, represented by null plus its explicit status.

All diagnostic tests are exploratory after feature/method selection. No multiplicity adjustment is claimed. Statistical associations/tests do not establish causation.

## Mean confidence intervals versus individual prediction intervals

A centered SVD supplies a full-rank orthogonal basis for the same selected affine regression space. Statsmodels OLS uses this basis and n_train-r residual df. Its mean predictions are numerically checked against the saved sklearn Pipeline on every evaluated test row. This resolves nonunique dummy coefficients without pretending all coefficients are identifiable.

Classical pointwise 95% mean CI estimates uncertainty in E[Y|X]. Classical 95% individual PI adds the fitted residual noise variance for one new Y. They use OLS covariance and t quantiles, not arbitrary percentages. A test row outside the training affine span has a valid sklearn convention-based point prediction but an unidentified conditional mean under the unrestricted aliased model; its intervals are masked as nonestimable. Saturated or zero-error-variance fits yield explicit unavailable intervals. Statsmodels prediction definitions: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html .

Additional approximate mean CIs use HC3 for student and HAC for energy, with a declared 24-lag bandwidth (4 hours at 10-minute sampling). These use asymptotic normal quantiles, not a robust individual PI. HC3 may be unavailable at unit leverage; HAC assumes appropriate weak dependence and bandwidth. Robust covariance references: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html .

All intervals condition on the selected model and omit feature/method-selection uncertainty. Classical PI is not made valid under heteroscedasticity or temporal dependence by merely changing mean-covariance estimates. Neither classical nor HAC intervals are a forecast guarantee. G3 bounds [0,20] and nonnegative Wh are not imposed by clipping, because that would silently change nominal interval behavior. Empirical held-out PI coverage is reported descriptively and never used to tune or validate mean-CI coverage using individual outcomes.

## Feature-count validation figures without refitting

For each saved inner validation fold, Phase 2 recorded actual RMSE. Its exact validation n and target SST are reconstructed with the same seed/gap/split configuration. Therefore

```
validation_R2 = 1 - n_validation * recorded_RMSE**2 / SST_validation
```

recovers the actual fold R2 (up to serialized numerical precision), without inventing scores or accessing held-out test data for selection. Curves show the mean and std of these fold values. Sparse FAST k values are not exhaustive evidence. For k=all the x-coordinate uses final training encoded width and explicitly warns that fold widths may differ. LASSO shows only its recorded best-alpha point at final-refit support size; its fold support counts vary and no unsupported full LASSO k path is drawn. RFE/RFECV overlap when their searches coincide.

## Actual selected-model findings

### student_performance: Correlation

Held-out MAE=3.4755, MSE=18.8756, RMSE=4.3446, R2=0.0795, rank-adjusted R2=0.0675. Selected encoded count=1; fitted rank=2. Inner CV RMSE=4.2798 +/- 0.0346; outer nested RMSE=4.2745 +/- 0.0400.

- linearity: Bin-mean residual span is 0.373 residual SD; inspect the plotted trend.
- independence: DW=2.0610; lag-1 ACF=-0.0332.
- homoscedasticity: LM=1.4912, p=0.222; constant variance not rejected.
- normality: p=5.32e-09, skew=-0.725, kurtosis=3.637; normality rejected.
- multicollinearity: Design rank 2/2; 0 perfectly collinear columns; 0 finite VIF values above 10.

DW is descriptive only for students. The early-30 problem excludes historical G1/G2; adding them would change when prediction can be made and is not done here. The few discrete fitted levels limit smooth-pattern assessment.

### appliances_energy: Mutual Information

Held-out MAE=52.6493, MSE=8685.5968, RMSE=93.1965, R2=-0.0480, rank-adjusted R2=-0.0506. Selected encoded count=10; fitted rank=11. Inner CV RMSE=96.0286 +/- 1.0752; outer nested RMSE=96.1827 +/- 1.2294.

- linearity: Bin-mean residual span is 0.357 residual SD; inspect the plotted trend.
- independence: DW=0.5318; lag-1 ACF=0.7340.
- homoscedasticity: LM=179.3475, p=3.2e-33; evidence against constant variance.
- normality: p=below floating-point resolution, skew=3.355, kurtosis=16.913; normality rejected.
- multicollinearity: Design rank 11/11; 0 perfectly collinear columns; 0 finite VIF values above 10.

Ljung-Box at tested lags rejects zero autocorrelation (minimum stored p=0; zero reflects numerical underflow). The inspected residual figure shows middle-bin negative means with positive means toward both ends, a possible curvature cue, alongside a pronounced positive tail. These observations and BP/JB evidence argue against treating conventional iid-normal homoscedastic OLS intervals as calibrated. The negative test R2 is retained; no test-driven model switch was made.

## Validation and scope

Numerical unit tests compare rank-adjusted training R2 and classical/HC3 intervals with independently constructed statsmodels designs, check HAC bounds, singular/out-of-span cases, saturated fits, constant targets, zero variance and VIF status handling. Integration tests verify all actual train/test metrics, unchanged saved predictions, all 24,156 interval rows, CV R2 reconstruction, 26 readable high-resolution PNGs, dataset-specific paths and metadata, and unchanged frontend hashes.

This phase does not perform Phase 4 frontend work, change model selection, run FULL training, commit or push. Report limitations rather than claiming assumptions are satisfied. Subsequent model changes would require a new training-only evaluation protocol rather than tuning against this held-out report.
