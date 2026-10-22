# Appliances Energy Prediction: Phase 3 (FAST)

Training-selected method: Mutual Information. No new model search or frontend changes.

## Held-out results (all saved final pipelines)

| Method | Encoded count | Effective training rank | RMSE | R2 | Adjusted R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutual Information | 10 | 11 | 93.1965 | -0.0480 | -0.0506 |
| RFE | 10 | 11 | 87.0556 | 0.0856 | 0.0833 |
| RFECV | 10 | 11 | 87.0556 | 0.0856 | 0.0833 |
| Correlation | 5 | 6 | 94.7469 | -0.0831 | -0.0845 |
| LASSO | 4 | 5 | 94.7312 | -0.0828 | -0.0839 |
| Baseline | 25 | 26 | 86.7329 | 0.0924 | 0.0866 |

Training effective affine design rank (including intercept) defines model complexity. Held-out adjusted R2 uses n_test-rank_train as a descriptive penalty, not refitted-test inference; null for n<=rank or constant target.

## Selected-model assumption evidence

### linearity: Binned residual means vs fitted (descriptive)

Bin-mean residual span is 0.357 residual SD; inspect the plotted trend.

A span >0.5 SD is a declared visual screening heuristic, not a significance test. A small span does not establish linearity.

### independence: Durbin-Watson and Ljung-Box

DW=0.5318; lag-1 ACF=0.7340.

Energy residuals retain chronological order; Ljung-Box model_df=0 is exploratory after regression/selection, with multiple correlated lag tests.

### homoscedasticity: Koenker-Breusch-Pagan on full-rank equivalent design

LM=179.3475, p=3.2e-33; evidence against constant variance.

Rank-reduced auxiliary design includes an intercept. Nonrejection is not proof; temporal dependence and selection can invalidate nominal p-values.

### normality: Jarque-Bera (large-sample)

p=below floating-point resolution, skew=3.355, kurtosis=16.913; normality rejected.

Shapiro p-values are not used beyond 5000 observations. Large-sample tests detect small departures; fitted/selected and serially dependent residuals are not iid raw samples. Inspect histogram and Q-Q.

### multicollinearity: Design rank, condition number and VIF (intercept excluded)

Design rank 11/11; 0 perfectly collinear columns; 0 finite VIF values above 10.

VIF >10 is a screening convention, not a universal cutoff. Full one-hot sets create exact aliasing; null VIF plus status denotes infinity/undefined, not zero. Individual aliased coefficients are not unique.

## Interval interpretation

Classical mean CI and individual PI are pointwise working-model intervals. HC3/HAC mean intervals are asymptotic approximations. None correct selection uncertainty or ensure temporal/heteroscedastic PI coverage.
Mean CI concerns E[Y|X]; individual PI adds residual variability for one new Y. Nominal coverage is not asserted. Outside-span rows have no estimable interval. Bounds are not clipped; grades outside 0-20 or negative Wh bounds expose the linear working model limitation.

Energy residuals are chronological. HAC uses a declared 24-lag bandwidth (4 hours at 10-minute sampling), not a guarantee of forecast uncertainty. Serial dependence weakens conventional OLS and BP/normality-test assumptions.

Feature-count validation R2 is reconstructed from saved fold RMSE and exact fold target SST; no test data, model refitting or fabricated validation scores. LASSO displays only its best-alpha point at final-refit support size, not a fabricated k path.

All tests are exploratory after feature/method selection; they are not causal inference or simultaneous guarantees. FAST has only two outer folds; FULL was not rerun.

See PHASE3.md for methodology, statistical references and output definitions.
