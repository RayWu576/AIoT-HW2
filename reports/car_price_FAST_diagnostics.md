# Car Price Prediction: Phase 3 (FAST)

Training-selected method: LASSO. No new model search or frontend changes.

## Held-out results (all saved final pipelines)

| Method | Encoded count | Effective training rank | RMSE | R2 | Adjusted R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LASSO | 38 | 37 | 2869.8225 | 0.8957 | -0.0433 |
| Correlation | 5 | 6 | 3816.5329 | 0.8155 | 0.7891 |
| Mutual Information | 5 | 6 | 3811.5904 | 0.8160 | 0.7897 |
| RFE | 5 | 6 | 3093.6381 | 0.8788 | 0.8614 |
| RFECV | 5 | 6 | 3093.6381 | 0.8788 | 0.8614 |
| Baseline | 73 | 60 | 2794.4080 | 0.9011 | nan |

Training effective affine design rank (including intercept) defines model complexity. Held-out adjusted R2 uses n_test-rank_train as a descriptive penalty, not refitted-test inference; null for n<=rank or constant target.

## Selected-model assumption evidence

### linearity: Binned residual means vs fitted (descriptive)

Bin-mean residual span is 0.736 residual SD; inspect the plotted trend.

A span >0.5 SD is a declared visual screening heuristic, not a significance test. A small span does not establish linearity.

### independence: Durbin-Watson (source row order; descriptive only)

DW=1.7990; lag-1 ACF=0.0902.

These observations have no justified temporal sampling order. DW cannot establish independence; group dependence remains possible.

### homoscedasticity: Koenker-Breusch-Pagan on full-rank equivalent design

LM=82.4945, p=1.65e-05; evidence against constant variance.

Rank-reduced auxiliary design includes an intercept. Nonrejection is not proof; temporal dependence and selection can invalidate nominal p-values.

### normality: Shapiro-Wilk

p=1.53e-05, skew=0.645, kurtosis=5.823; normality rejected.

Shapiro p-values are not used beyond 5000 observations. Large-sample tests detect small departures; fitted/selected and serially dependent residuals are not iid raw samples. Inspect histogram and Q-Q.

### multicollinearity: Design rank, condition number and VIF (intercept excluded)

Design rank 37/39; 4 perfectly collinear columns; 5 finite VIF values above 10.

VIF >10 is a screening convention, not a universal cutoff. Full one-hot sets create exact aliasing; null VIF plus status denotes infinity/undefined, not zero. Individual aliased coefficients are not unique.

## Interval interpretation

Classical mean CI and individual PI are pointwise working-model intervals. HC3/HAC mean intervals are asymptotic approximations. None correct selection uncertainty or ensure temporal/heteroscedastic PI coverage.
Mean CI concerns E[Y|X]; individual PI adds residual variability for one new Y. Nominal coverage is not asserted. Outside-span rows have no estimable interval. Bounds are not clipped; grades outside 0-20 or negative Wh bounds expose the linear working model limitation.

Manufacturer is derived from CarName inside training-fold preprocessing; small-sample and brand-group dependence limit inference.

Feature-count validation R2 is reconstructed from saved fold RMSE and exact fold target SST; no test data, model refitting or fabricated validation scores. LASSO displays only its best-alpha point at final-refit support size, not a fabricated k path.

All tests are exploratory after feature/method selection; they are not causal inference or simultaneous guarantees. FAST has only two outer folds; FULL was not rerun.

See PHASE3.md for methodology, statistical references and output definitions.
