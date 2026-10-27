# Student Performance (Mathematics): Phase 3 (FAST)

Training-selected method: Correlation. No new model search or frontend changes.

## Held-out results (all saved final pipelines)

| Method | Encoded count | Effective training rank | RMSE | R2 | Adjusted R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correlation | 1 | 2 | 4.3446 | 0.0795 | 0.0675 |
| RFE | 1 | 2 | 4.3446 | 0.0795 | 0.0675 |
| RFECV | 1 | 2 | 4.3446 | 0.0795 | 0.0675 |
| Mutual Information | 10 | 10 | 4.3185 | 0.0905 | -0.0281 |
| LASSO | 3 | 4 | 4.3107 | 0.0938 | 0.0575 |
| Baseline | 56 | 40 | 4.1957 | 0.1415 | -0.7170 |

Training effective affine design rank (including intercept) defines model complexity. Held-out adjusted R2 uses n_test-rank_train as a descriptive penalty, not refitted-test inference; null for n<=rank or constant target.

## Selected-model assumption evidence

### linearity: Binned residual means vs fitted (descriptive)

Bin-mean residual span is 0.373 residual SD; inspect the plotted trend.

A span >0.5 SD is a declared visual screening heuristic, not a significance test. A small span does not establish linearity.

### independence: Durbin-Watson (source row order; descriptive only)

DW=2.0610; lag-1 ACF=-0.0332.

These observations have no justified temporal sampling order. DW cannot establish independence; group dependence remains possible.

### homoscedasticity: Koenker-Breusch-Pagan on full-rank equivalent design

LM=1.4912, p=0.222; constant variance not rejected.

Rank-reduced auxiliary design includes an intercept. Nonrejection is not proof; temporal dependence and selection can invalidate nominal p-values.

### normality: Shapiro-Wilk

p=5.32e-09, skew=-0.725, kurtosis=3.637; normality rejected.

Shapiro p-values are not used beyond 5000 observations. Large-sample tests detect small departures; fitted/selected and serially dependent residuals are not iid raw samples. Inspect histogram and Q-Q.

### multicollinearity: Design rank, condition number and VIF (intercept excluded)

Design rank 2/2; 0 perfectly collinear columns; 0 finite VIF values above 10.

VIF >10 is a screening convention, not a universal cutoff. Full one-hot sets create exact aliasing; null VIF plus status denotes infinity/undefined, not zero. Individual aliased coefficients are not unique.

## Interval interpretation

Classical mean CI and individual PI are pointwise working-model intervals. HC3/HAC mean intervals are asymptotic approximations. None correct selection uncertainty or ensure temporal/heteroscedastic PI coverage.
Mean CI concerns E[Y|X]; individual PI adds residual variability for one new Y. Nominal coverage is not asserted. Outside-span rows have no estimable interval. Bounds are not clipped; grades outside 0-20 or negative Wh bounds expose the linear working model limitation.

G1/G2 are historical first/second-period grades; this student early-30 scenario excludes them. Including them changes prediction availability and practical use.

Feature-count validation R2 is reconstructed from saved fold RMSE and exact fold target SST; no test data, model refitting or fabricated validation scores. LASSO displays only its best-alpha point at final-refit support size, not a fabricated k path.

All tests are exploratory after feature/method selection; they are not causal inference or simultaneous guarantees. FAST has only two outer folds; FULL was not rerun.

See PHASE3.md for methodology, statistical references and output definitions.
